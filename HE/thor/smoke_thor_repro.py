#!/usr/bin/env python3
"""
Faithful THOR reproduction smoke (Branch A).

Mirrors ``thirdparty/THOR-main/forward.ipynb`` for **one encoder layer**:
  official Liberate engine + keys0 + DualRail encoded weights + THOR Softmax/GeLU/LN
  → decrypt intermediates → compare to ``poly_model_inference`` dense poly
  intermediates (same Softmax/LN/GeLU scheme levels as this smoke).

DualRail linear patches (``linear_eval`` binds)::

  - PT mask complement for score/context (replace ``2**41`` integer trick)
  - ``--rest-div`` default **1** (THOR-native): score ``scale=1`` + KEY bake
    ``1/(64·δ1·δ2)`` per layer (exp entry prescaled).

Prereqs (under ``thirdparty/THOR-main/``):
  keys/keys0/, liberate/.../resources/, datasets/mrpc/,
  finetuned_models/mrpc/model.safetensors,
  encoded_models_new/mrpc/{att,ff,pooler,cls}.pkl  (from ``encode.py``)

Usage (from repo root or HE/thor)::

  python3 HE/thor/smoke_thor_repro.py --layer 0 --sample 0
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_THOR_ROOT = os.path.abspath(os.path.join(_HE, "..", "thirdparty", "THOR-main"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _HE not in sys.path:
    sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()

from engine import (  # noqa: E402
    THOR_DEFAULT_PARAMS,
    create_engine,
    load_thor_keys,
)
from thor.bert import ThorBertAttention, ThorBertFF  # noqa: E402
from thor.data import ThorDataEncryptor  # noqa: E402
from thor.linear import ThorLinearEvaluator  # noqa: E402
from thor.utils.matrix import ld  # noqa: E402


# Slot bands used by forward.ipynb plot_variables (head h, pack band j)
_H_INDICES = [np.where(np.arange(0, 2**11) % 16 == i)[0] for i in range(12)]


def encoded_layer_pkl(dataset: str, kind: str, layer_idx: int) -> str:
    """``encoded_models_new/{dataset}/{ff|att}_L{n}.pkl``."""
    if kind not in ("ff", "att"):
        raise ValueError(f"kind must be ff|att, got {kind!r}")
    return os.path.join(
        _THOR_ROOT,
        "encoded_models_new",
        dataset,
        f"{kind}_L{int(layer_idx)}.pkl",
    )


def _thor_paths(dataset: str) -> dict[str, str]:
    enc = os.path.join(_THOR_ROOT, "encoded_models_new", dataset)
    return {
        "dataset": os.path.join(_THOR_ROOT, "datasets", dataset),
        "finetuned": os.path.join(
            _THOR_ROOT, "finetuned_models", dataset, "model.safetensors"
        ),
        "encoded_dir": enc,
        "att": os.path.join(enc, "att.pkl"),
        "ff": os.path.join(enc, "ff.pkl"),
        "pooler": os.path.join(enc, "pooler.pkl"),
        "cls": os.path.join(enc, "cls.pkl"),
    }


def _require_assets(
    paths: dict[str, str],
    *,
    layers: list[int] | None = None,
    need_pooler_cls: bool = False,
) -> None:
    missing = []
    for k in ("dataset", "finetuned"):
        if not os.path.exists(paths[k]):
            missing.append(f"{k}={paths[k]}")
    dataset = os.path.basename(paths.get("encoded_dir") or "") or "mrpc"
    if "encoded_models_new" in (paths.get("encoded_dir") or ""):
        dataset = os.path.basename(paths["encoded_dir"])
    if layers:
        for li in layers:
            for kind in ("ff", "att"):
                p = encoded_layer_pkl(dataset, kind, li)
                if not os.path.isfile(p):
                    missing.append(f"{kind}_L{li}={p}")
    if need_pooler_cls:
        for k in ("pooler", "cls"):
            if not os.path.exists(paths[k]):
                missing.append(f"{k}={paths[k]}")
    if missing:
        raise FileNotFoundError(
            "Missing THOR assets "
            + ", ".join(missing)
            + ". Encode with: python3 HE/thor/encode_thor_weights.py"
        )


def _slim_encoded_layer_weights(weights: dict, layer_idx: int) -> dict:
    """Drop non-layer DualRail PTs in-place (avoids 2× peak during filter)."""
    needle = f".layer.{layer_idx}."
    drop = [k for k in weights if needle not in k]
    for k in drop:
        del weights[k]
    if not weights:
        raise KeyError(f"no encoded keys for layer {layer_idx}")
    return weights


def _drop_he_weight_keys(module, keys: list[str] | tuple[str, ...]) -> None:
    """Drop finished DualRail PT tiles from CPU/CUDA to free RAM/VRAM."""
    import gc

    w = getattr(module, "weights", None)
    if not isinstance(w, dict):
        return
    dropped = 0
    for key in keys:
        arr = w.get(key)
        if arr is None:
            continue
        try:
            for index in np.ndindex(arr.shape):
                cell = arr[index]
                if isinstance(cell, list):
                    for j, item in enumerate(cell):
                        cell[j] = None
                        del item
                    arr[index] = None
                elif torch.is_tensor(cell):
                    arr[index] = None
                else:
                    arr[index] = None
        except Exception:
            pass
        w[key] = None
        dropped += 1
    if dropped:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(
                f"  [mem] dropped {dropped} weight key(s) "
                f"cuda_alloc={used:.2f} GiB",
                flush=True,
            )
        except Exception:
            pass


def _release_he_module_weights(module) -> None:
    """Release any remaining DualRail weights on a Thor module."""
    keys = list(getattr(module, "keys", []) or [])
    if not keys and isinstance(getattr(module, "weights", None), dict):
        keys = list(module.weights.keys())
    _drop_he_weight_keys(module, keys)
    try:
        module.devices = []
    except Exception:
        pass


def _transpose_for_scores(x: torch.Tensor, attention_m) -> torch.Tensor:
    """HF BERT Q/K/V layout (newer transformers dropped ``transpose_for_scores``)."""
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attention_m.attention_head_size)
    return x.view(*hidden_shape).transpose(1, 2)


def _plain_layer_refs(model_plain, batch, attention_mask, device):
    """Per-layer plaintext intermediates (same as forward.ipynb cell 21)."""
    hidden_states = []
    refs = []
    with torch.no_grad():
        hs = model_plain.bert.embeddings(
            input_ids=batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
        )
        for layer_idx in range(12):
            layer = model_plain.bert.encoder.layer[layer_idx]
            attention_m = layer.attention.self
            bert_output_m = layer.attention.output

            q = _transpose_for_scores(attention_m.query(hs), attention_m)
            k = _transpose_for_scores(attention_m.key(hs), attention_m)
            v = _transpose_for_scores(attention_m.value(hs), attention_m)
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                attention_m.attention_head_size
            )
            if hasattr(model_plain, "get_extended_attention_mask"):
                extended = model_plain.get_extended_attention_mask(
                    attention_mask, batch["input_ids"].shape
                ).to(device)
            else:
                # newer HF: create_extended_attention_mask_for_decoder-style helpers
                extended = attention_mask[:, None, None, :].to(dtype=hs.dtype)
                extended = (1.0 - extended) * torch.finfo(hs.dtype).min
            sftmx_in = attention_scores + extended
            sftmx_out = torch.nn.functional.softmax(sftmx_in, dim=-1)
            att_context = torch.matmul(sftmx_out, v)
            context_layer = att_context.permute(0, 2, 1, 3).contiguous()
            all_head = getattr(
                attention_m, "all_head_size", attention_m.num_attention_heads * attention_m.attention_head_size
            )
            context_layer = context_layer.view(
                context_layer.size()[:-2] + (all_head,)
            )
            dense_output = bert_output_m.dense(context_layer)
            ln1_in = dense_output + hs
            ln1_out = bert_output_m.LayerNorm(ln1_in)
            gelu_in = layer.intermediate.dense(ln1_out)
            gelu_out = layer.intermediate.intermediate_act_fn(gelu_in)
            dense2_out = layer.output.dense(gelu_out)
            ln2_in = dense2_out + ln1_out
            ln2_out = layer.output.LayerNorm(ln2_in)

            refs.append(
                {
                    "x": hs.cpu().numpy().squeeze(),
                    "q": q.cpu().numpy().squeeze(),
                    # Pre-mask scores: HE ``sftmx_in`` / after_bts_score matches this
                    # (Softmax applies mask via ×mask on exp, not additive -inf).
                    "att_score": attention_scores.cpu().numpy().squeeze(),
                    "sftmx_in": sftmx_in.cpu().numpy().squeeze(),
                    "sftmx_out": sftmx_out.cpu().numpy().squeeze(),
                    "att_context": att_context.cpu().numpy().squeeze(),
                    "ln1_in": ln1_in.cpu().numpy().squeeze(),
                    "ln1_out": ln1_out.cpu().numpy().squeeze(),
                    "gelu_in": gelu_in.cpu().numpy().squeeze(),
                    "gelu_out": gelu_out.cpu().numpy().squeeze(),
                    "dense2_out": dense2_out.cpu().numpy().squeeze(),
                    "ln2_in": ln2_in.cpu().numpy().squeeze(),
                    "ln2_out": ln2_out.cpu().numpy().squeeze(),
                }
            )
            hidden_states.append(hs.cpu().numpy().squeeze())
            hs = ln2_out
    return refs


def _poly_layer_refs(model_poly, batch, attention_mask, device):
    """
    Dense poly intermediates (same site keys as ``_plain_layer_refs``).

    Requires ``install_eager_attention_poly_patch`` + ``apply_polynomial_scheme``
    so GeLU/LN modules and ``_poly_softmax_fn`` are set. Softmax uses the poly
    evaluator (``_plain_layer_refs`` hardcodes ``F.softmax``).
    """
    refs = []
    with torch.no_grad():
        hs = model_poly.bert.embeddings(
            input_ids=batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
        )
        for layer_idx in range(12):
            layer = model_poly.bert.encoder.layer[layer_idx]
            attention_m = layer.attention.self
            bert_output_m = layer.attention.output

            q = _transpose_for_scores(attention_m.query(hs), attention_m)
            k = _transpose_for_scores(attention_m.key(hs), attention_m)
            v = _transpose_for_scores(attention_m.value(hs), attention_m)
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                attention_m.attention_head_size
            )

            poly_fn = getattr(attention_m, "_poly_softmax_fn", None)
            if poly_fn is None:
                raise RuntimeError(
                    f"L{layer_idx}: missing _poly_softmax_fn — "
                    "apply_polynomial_scheme before _poly_layer_refs"
                )
            key_valid = (attention_mask > 0).to(dtype=attention_scores.dtype)
            key_valid = key_valid[:, None, None, :].expand_as(attention_scores)
            sftmx_out = poly_fn(
                attention_scores, dim=-1, key_valid_mask=key_valid
            )

            if hasattr(model_poly, "get_extended_attention_mask"):
                extended = model_poly.get_extended_attention_mask(
                    attention_mask, batch["input_ids"].shape
                ).to(device)
            else:
                extended = attention_mask[:, None, None, :].to(dtype=hs.dtype)
                extended = (1.0 - extended) * torch.finfo(hs.dtype).min
            sftmx_in = attention_scores + extended

            att_context = torch.matmul(sftmx_out, v)
            context_layer = att_context.permute(0, 2, 1, 3).contiguous()
            all_head = getattr(
                attention_m,
                "all_head_size",
                attention_m.num_attention_heads * attention_m.attention_head_size,
            )
            context_layer = context_layer.view(
                context_layer.size()[:-2] + (all_head,)
            )
            dense_output = bert_output_m.dense(context_layer)
            ln1_in = dense_output + hs
            ln1_out = bert_output_m.LayerNorm(ln1_in)
            gelu_in = layer.intermediate.dense(ln1_out)
            gelu_out = layer.intermediate.intermediate_act_fn(gelu_in)
            dense2_out = layer.output.dense(gelu_out)
            ln2_in = dense2_out + ln1_out
            ln2_out = layer.output.LayerNorm(ln2_in)

            refs.append(
                {
                    "x": hs.cpu().numpy().squeeze(),
                    "q": q.cpu().numpy().squeeze(),
                    "att_score": attention_scores.cpu().numpy().squeeze(),
                    "sftmx_in": sftmx_in.cpu().numpy().squeeze(),
                    "sftmx_out": sftmx_out.cpu().numpy().squeeze(),
                    "att_context": att_context.cpu().numpy().squeeze(),
                    "ln1_in": ln1_in.cpu().numpy().squeeze(),
                    "ln1_out": ln1_out.cpu().numpy().squeeze(),
                    "gelu_in": gelu_in.cpu().numpy().squeeze(),
                    "gelu_out": gelu_out.cpu().numpy().squeeze(),
                    "dense2_out": dense2_out.cpu().numpy().squeeze(),
                    "ln2_in": ln2_in.cpu().numpy().squeeze(),
                    "ln2_out": ln2_out.cpu().numpy().squeeze(),
                }
            )
            hs = ln2_out
    return refs


def _forward_layer(
    *,
    engine,
    evaluator,
    thor_attention,
    thor_ff,
    layer_idx: int,
    x,
    thor_attention_mask,
    devices,
    bts_hook=None,
    level_probe=None,
    rem_guard=None,
):
    """Copy of forward.ipynb ``forward_layer`` (HE path only).

    Bootstrap sites go through ``dualrail_bts`` + ``BootstrapHook`` (plan only).

    ``rem_guard``: fail-fast when HE rem ≠ DP ``rem_trace`` (plan / plan_mock).
    """
    from dualrail_bts import (
        refresh_cts,
        refresh_dualrail8,
        refresh_dualrail_ff,
        plan_refresh_cts,
        plan_refresh_dualrail8,
        plan_refresh_dualrail_ff,
    )
    from softmax_he_ct import align_bypass_to_main
    from level_probe import ProbeStop

    def _as_object_array(seq):
        """Build object ndarray without np.asarray (Liberate CT ⇔ CUDA tensor)."""
        seq = list(seq)
        out = np.empty(len(seq), dtype=object)
        for i, v in enumerate(seq):
            out[i] = v
        return out

    def _he_rem(cts) -> int:
        return int(engine.num_levels) - int(list(cts)[0].level_calc)

    def _probe(site: str, cts, *, note: str = "") -> None:
        if level_probe is not None:
            level_probe.record_first(site, cts, note=note)
            if level_probe.should_stop(site):
                raise ProbeStop(site, level_probe, cts=cts)
        elif rem_guard is not None:
            rem = _he_rem(cts)
            extra = f" ({note})" if note else ""
            print(
                f"  [probe] {site}: remaining={rem}{extra}",
                flush=True,
            )
        if rem_guard is not None:
            rem_guard.check_probe(site, _he_rem(cts))

    thor_attention.to(devices)
    # FF DualRail PTs stay on CPU until after LN1 — halves peak VRAM vs loading both.
    prefix = f"L{int(layer_idx)}"
    print(
        f"HE forward layer_idx={layer_idx} "
        f"bts={getattr(bts_hook, 'mode', 'none')}"
        f"{'/mock' if getattr(engine, '_mock_bootstrap_info', None) else ''}",
        flush=True,
    )

    # Layer entry is always 8 real (L0: encrypt_embedding_real8; L1+: ln2_out).
    if x.shape != (8,):
        raise ValueError(
            f"layer entry must be 8 real packs, got x.shape={x.shape}"
        )

    # 固定 pack 8 real → 4 cplx（非微事件）；残差旁路仍为 ``x``
    x_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        x_cplx[i] = engine.cc_add(x[i], engine.imult(x[i + 4]))

    # Layer-entry rem（对齐 DP ``__initial__`` / 跨层出口）——须在 QKV before-boot 之前。
    _probe("entry_cplx", x_cplx)

    # Plan site ``linear_qkv`` before：刷 4 cplx；若刷则 unpack 回写残差 ``x``。
    # 必须在 emb-duplicate 之前：否则 unpack 会把 duplicate 写进残差。
    _qkv_ev = f"{prefix}.linear_qkv"
    if bts_hook is not None and bts_hook.should_refresh(_qkv_ev):
        print(
            f"  [smoke_thor_repro] linear_qkv refresh "
            f"pack_mode={bts_hook.pack_mode(_qkv_ev)}",
            flush=True,
        )
        from dualrail_bts import unpack_cplx4_to_real8

        x_cplx = _as_object_array(
            plan_refresh_cts(
                engine,
                x_cplx,
                bts_hook,
                _qkv_ev,
            )
        )
        x = unpack_cplx4_to_real8(
            engine,
            x_cplx,
            hook=bts_hook,
            event_name=_qkv_ev,
        )

    # L0 embedding 8-real already has %16∈[6,12) filled; L1+ LN layout needs
    # emb-duplicate via cc_add(z, rotate_left(z, -6)) — only on QKV ``x_cplx``.
    if layer_idx != 0:
        for i in range(4):
            x_cplx[i] = engine.cc_add(
                x_cplx[i], engine.rotate_left(x_cplx[i], -6)
            )

    x_cplx_rots = evaluator.make_rotated_copies(x_cplx)
    q_wo_rescale = thor_attention.query(x_cplx_rots)
    k = thor_attention.key(x_cplx_rots)
    v = thor_attention.value(x_cplx_rots)
    _probe("after_qkv", q_wo_rescale, note="Q before rescale")
    _probe("after_qkv_v", v, note="V parked")

    # Plan site ``linear_att_score`` before：刷 Q+K（V 旁路不动）。
    _qk_ev = f"{prefix}.linear_att_score"
    if bts_hook is not None and bts_hook.should_refresh(_qk_ev):
        print(
            f"  [smoke_thor_repro] Q∥K before-score refresh "
            f"event={_qk_ev} pack_mode={bts_hook.pack_mode(_qk_ev)}",
            flush=True,
        )
        _qk = list(q_wo_rescale) + list(k)
        _qk = list(
            plan_refresh_cts(
                engine,
                _qk,
                bts_hook,
                _qk_ev,
            )
        )
        q_wo_rescale = _as_object_array(_qk[:4])
        k = _as_object_array(_qk[4:])

    l_k = evaluator.transpose_upper_to_lower(k)
    _probe("after_transpose_k", l_k)
    l_k_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        l_k_cplx[i] = engine.cc_add(
            engine.level_up(l_k[i], l_k[i].level_calc + 1),
            engine.imult(evaluator.rotate_internal(l_k[i], 64, mode="att")),
        )
        l_k_cplx[i] = engine.rescale(l_k_cplx[i])
    _probe("after_k_complexify", l_k_cplx)

    q = np.full_like(q_wo_rescale, None, dtype=object)
    for i in range(4):
        q[i] = engine.rescale(q_wo_rescale[i])
    _probe("after_q_rescale", q)
    q_copies = evaluator.make_copies(q)
    _probe("after_make_copies", q_copies)
    sftmx_in = thor_attention.calculate_attention_score(
        l_k_cplx, q_copies, bootstrap=False, scale=1, rescale=False
    )
    _probe("after_att_score", sftmx_in)
    # Q/K linear done — free DualRail PT tiles before Softmax peak CTs.
    _drop_he_weight_keys(
        thor_attention,
        ("query.weight", "query.bias", "key.weight", "key.bias"),
    )

    # Softmax 前 DualRail：仅 plan 站点（score / stockmeyer）。
    if bts_hook is not None:
        for _sev in (
            f"{prefix}.softmax_exp_stockmeyer",
        ):
            if not bts_hook.should_refresh(_sev):
                continue
            print(
                f"  [smoke_thor_repro] pre-Softmax DualRail "
                f"event={_sev} pack_mode={bts_hook.pack_mode(_sev)}",
                flush=True,
            )
            from dualrail_bts import refresh_dualrail8

            sftmx_in = _as_object_array(
                refresh_dualrail8(
                    engine,
                    list(sftmx_in),
                    bts_hook,
                    _sev,
                    scale_after=0.5,
                )
            )
            break
    _probe("after_bts_score", sftmx_in)

    sftmx_out = thor_attention.softmax(
        x=sftmx_in, attention_mask=thor_attention_mask, rescale=False, debug=False, sk=None
    )
    _probe("after_softmax", sftmx_out)

    # Plan site ``linear_att_context`` before：刷 128 α（主路 inputs；V 旁路见下）。
    _ctx_ev = f"{prefix}.linear_att_context"
    if bts_hook is not None and bts_hook.should_refresh(_ctx_ev):
        print(
            f"  [smoke_thor_repro] linear_att_context refresh "
            f"pack_mode={bts_hook.pack_mode(_ctx_ev)}",
            flush=True,
        )
        sftmx_out = _as_object_array(
            plan_refresh_cts(
                engine,
                list(sftmx_out),
                bts_hook,
                _ctx_ev,
            )
        )

    v_cplx = np.full((2,), None, dtype=object)
    for i in range(2):
        v_cplx[i] = engine.cc_add(v[i], engine.imult(v[i + 2]))
    # score 为主路；V 为旁路：只动 V（必要时 bts），绝不 level_up score。
    from softmax_he_ct import align_bypass_to_main

    for j in range(2):
        v_cplx[j] = align_bypass_to_main(engine, sftmx_out[0], v_cplx[j])
    for i in range(2):
        v_cplx[i] = engine.rescale(v_cplx[i])
    sftmx_out_rescale = np.full((128,), None, dtype=object)
    for j in range(128):
        sftmx_out_rescale[j] = engine.rescale(sftmx_out[j])
    att_context = thor_attention.calculate_attention_context(
        v_cplx, sftmx_out_rescale, rescale=False
    )
    _probe("after_att_context", att_context)
    _drop_he_weight_keys(
        thor_attention,
        ("value.weight", "value.bias"),
    )
    _ctx_rem = int(engine.num_levels) - int(att_context[0].level_calc)
    print(
        f"  [smoke_thor_repro] after_att_context rem={_ctx_rem} "
        f"lc={att_context[0].level_calc}",
        flush=True,
    )

    # Plan site ``bridge_rot_attn_dense`` before：刷 2 context（主路 inputs）。
    _rot_ev = f"{prefix}.bridge_rot_attn_dense"
    if bts_hook is not None and bts_hook.should_refresh(_rot_ev):
        print(
            f"  [smoke_thor_repro] bridge_rot_attn_dense refresh "
            f"pack_mode={bts_hook.pack_mode(_rot_ev)}",
            flush=True,
        )
        att_context = _as_object_array(
            plan_refresh_cts(
                engine,
                att_context,
                bts_hook,
                _rot_ev,
            )
        )
    _probe("after_bts_context", att_context)

    att_context_rots = thor_attention.evaluator.make_rotated_copies(att_context)

    # Plan site ``linear_attn_dense`` before：刷 32 rot copies（主路 inputs）。
    _dense_ev = f"{prefix}.linear_attn_dense"
    if bts_hook is not None and bts_hook.should_refresh(_dense_ev):
        print(
            f"  [smoke_thor_repro] linear_attn_dense refresh "
            f"pack_mode={bts_hook.pack_mode(_dense_ev)}",
            flush=True,
        )
        att_context_rots = _as_object_array(
            plan_refresh_cts(
                engine,
                att_context_rots,
                bts_hook,
                _dense_ev,
            )
        )
    dense_output = thor_attention.dense(att_context_rots)
    _probe("after_attn_dense", dense_output)
    _drop_he_weight_keys(
        thor_attention,
        ("dense.weight", "dense.bias"),
    )
    x_out_sum = np.full((8,), None, dtype=object)
    for i in range(4):
        # 主路 dense_output；旁路 x（fork 残差）向主路对齐后再 add
        xi = align_bypass_to_main(engine, dense_output[i], x[i])
        xip4 = align_bypass_to_main(engine, dense_output[i + 4], x[i + 4])
        x_out_sum[i] = engine.add(xi, dense_output[i])
        x_out_sum[i + 4] = engine.add(xip4, dense_output[i + 4])
    ln1_in = x_out_sum
    ln1_out = thor_attention.layernorm(x=ln1_in, sk=None)
    _probe("after_ln1", ln1_out)
    # Attn LayerNorm linear done; free γ/β before FF weights hit GPU.
    _drop_he_weight_keys(
        thor_attention,
        ("LayerNorm.weight", "LayerNorm.bias"),
    )
    thor_ff.to(devices)

    # Plan site ``bridge_rot_ff1`` before：刷 8 LN1（主路 inputs）。
    # FF 残差旁路由 HE ``align_bypass_to_main`` 对齐，不进 DP。
    _ff_rot_ev = f"{prefix}.bridge_rot_ff1"
    if bts_hook is not None and bts_hook.should_refresh(_ff_rot_ev):
        print(
            f"  [smoke_thor_repro] bridge_rot_ff1 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_rot_ev)}",
            flush=True,
        )
        ln1_out = _as_object_array(
            plan_refresh_dualrail8(
                engine,
                ln1_out,
                bts_hook,
                _ff_rot_ev,
                scale_after=0.5,
            )
        )
    _probe("after_bts_bridge_rot_ff1", ln1_out)

    l = np.full((64,), None, dtype=object)
    mask = np.full((engine.num_slots,), 1, dtype=int)
    mask[np.arange(engine.num_slots) % 16 >= 6] = 0
    for i in range(4):
        temp = engine.cc_add(ln1_out[i], engine.imult(ln1_out[i + 4]))
        temp = engine.mc_mult(mask, temp)
        l[16 * i] = engine.cc_add(temp, engine.rotate_left(temp, -8))
        for j in range(1, 16):
            index = 16 * i + j
            l[index] = engine.rotate_left(l[index - 1], 2**11)
    _probe("after_ff_bridge_mask", l, note="mc_mult on LN1→rot64")

    # Plan site ``linear_ff_dense1`` before：刷 64 rot copies（已是复数侧，直接刷）。
    _ff_d1_ev = f"{prefix}.linear_ff_dense1"
    if bts_hook is not None and bts_hook.should_refresh(_ff_d1_ev):
        print(
            f"  [smoke_thor_repro] linear_ff_dense1 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_d1_ev)}",
            flush=True,
        )
        l = _as_object_array(
            plan_refresh_cts(
                engine,
                list(l),
                bts_hook,
                _ff_d1_ev,
            )
        )

    gelu_in_wo_bs = thor_ff.dense1(l)
    _probe("after_ff_dense1", gelu_in_wo_bs[0])
    _drop_he_weight_keys(
        thor_ff,
        ("dense1.weight", "dense1.bias"),
    )
    # GeLU f1/f2/reconstruct hooks live inside ``he_gelu_cheb`` / ``gelu_poly_pack``.
    _probe("after_ff_dense1_pre_gelu", gelu_in_wo_bs[0])

    gelu_out = thor_ff.gelu(x=gelu_in_wo_bs)
    _probe("after_gelu", gelu_out[0])

    # Plan site ``linear_ff_dense2`` before：DualRail 刷 GeLU 出口 (2,8)。
    _ff_d2_ev = f"{prefix}.linear_ff_dense2"
    if bts_hook is not None and bts_hook.should_refresh(_ff_d2_ev):
        print(
            f"  [smoke_thor_repro] linear_ff_dense2 refresh "
            f"pack_mode={bts_hook.pack_mode(_ff_d2_ev)}",
            flush=True,
        )
        gelu_out = plan_refresh_dualrail_ff(
            engine,
            gelu_out,
            bts_hook,
            _ff_d2_ev,
        )
        _probe("after_bts_post_gelu", gelu_out[0])

    dense2_out = thor_ff.dense2(gelu_out)
    _probe("after_ff_dense2", dense2_out)
    _drop_he_weight_keys(
        thor_ff,
        ("dense2.weight", "dense2.bias"),
    )
    ln2_in = np.full((8,), None, dtype=object)
    for i in range(8):
        # 主路 dense2_out；旁路 ln1_out（FF 残差）向主路对齐后再 add
        ri = align_bypass_to_main(engine, dense2_out[i], ln1_out[i])
        ln2_in[i] = engine.add(ri, dense2_out[i])
    # LN2 DualRail / scale hooks 在 ``he_layernorm_poly``（``ln2_scale`` 起）

    ln2_out = thor_ff.layernorm(x=ln2_in, sk=None)
    _probe("after_ln2", ln2_out)
    # No level_up(→21): next layer / plan decides rem; do not burn CT for old W@21.

    _drop_he_weight_keys(
        thor_ff,
        ("LayerNorm.weight", "LayerNorm.bias"),
    )
    _release_he_module_weights(thor_attention)
    _release_he_module_weights(thor_ff)
    return ln2_out, {
        "x": x,
        "q": q_wo_rescale,
        "sftmx_in": sftmx_in,
        "sftmx_out": sftmx_out,
        "att_context": att_context,
        "ln1_in": ln1_in,
        "ln1_out": ln1_out,
        "gelu_in_wo_bs": gelu_in_wo_bs,
        "gelu_out": gelu_out,
        "dense2_out": dense2_out,
        "ln2_in": ln2_in,
        "ln2_out": ln2_out,
    }


def _he_vs_plain_score_packs(
    engine,
    sk,
    he_ct,
    att_score_hf: np.ndarray,
    *,
    task: str,
    layer_idx: int,
    post_dualrail_bs: bool = False,
    rest_div: float = 8.0,
) -> tuple[float, float, float]:
    """
    Compare HE score CTs to plain HF ``att_score`` via dense decode.

    KEY bake ``1/(64·δ1·δ2)`` → decode multiply ``64·δ1·δ2`` for HF match.
    """
    import sys
    from pathlib import Path

    he_root = str(Path(__file__).resolve().parents[1])
    repo_root = str(Path(__file__).resolve().parents[2])
    for p in (he_root, repo_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    from thor_encoder_linear_core import bert_base
    from softmax_poly import score_hf_decode_bake  # noqa: WPS433

    from linear_eval import decode_score_packs  # noqa: WPS433

    cfg = bert_base()
    if isinstance(he_ct, np.ndarray) and he_ct.ndim > 1:
        he_ct = he_ct[0]
    packs = []
    for i in range(8):
        dec = np.asarray(
            engine.decrode(he_ct[i], sk, is_real=True), dtype=np.float64
        ).ravel()
        packs.append(dec)

    bake = score_hf_decode_bake(task, layer_idx)
    if abs(float(rest_div) - 1.0) > 1e-12:
        print(
            f"    [sftmx_in dense] WARN rest_div={rest_div:g} with production "
            f"scale=1 distorts logits; use --rest-div 1 for abs HF match",
            flush=True,
        )
    he_dense = decode_score_packs(packs, cfg) * bake

    scores_hf = np.asarray(att_score_hf, dtype=np.float64)
    if scores_hf.ndim == 4:
        scores_hf = scores_hf.squeeze(0)
    # HF (h, query, key) → DualRail / slot_softmax (h, key, query)
    ref = np.transpose(scores_hf, (0, 2, 1))

    cur = he_dense.ravel()
    ref_z = ref.ravel()
    denom = float(np.dot(cur, cur))
    if denom > 1e-30:
        g_ls = float(np.dot(cur, ref_z) / denom)
    else:
        g_ls = float("nan")
    err_ls = (
        float(np.max(np.abs(cur * g_ls - ref_z))) if denom > 1e-30 else float("inf")
    )
    err_g1 = float(np.max(np.abs(cur - ref_z)))
    corr = (
        float(np.corrcoef(cur, ref_z)[0, 1])
        if np.std(cur) > 1e-15 and np.std(ref_z) > 1e-15
        else float("nan")
    )
    print(
        f"    [sftmx_in dense] bake=×{bake:g} rest_div={rest_div:g} "
        f"post_bs_div2={post_dualrail_bs}  "
        f"max|HE×bake|={float(np.max(np.abs(cur))):.4g}  "
        f"max|ref|={float(np.max(np.abs(ref_z))):.4g}  "
        f"err@gain1={err_g1:.3e}  err@LS={err_ls:.3e} ls_gain={g_ls:.4g}",
        flush=True,
    )
    # Post-bootstrap: abs noise O(1e-1) on |score|~10; gate on relative err.
    ref_peak = float(np.max(np.abs(ref_z)))
    if post_dualrail_bs and ref_peak > 1e-8:
        err_gate = err_g1 / ref_peak
        print(
            f"    [sftmx_in dense] post-bs gate=rel {err_gate:.3e} "
            f"(abs={err_g1:.3e})",
            flush=True,
        )
    else:
        err_gate = err_g1
    return err_gate, corr, g_ls


def _he_vs_plain_diag(
    engine,
    sk,
    he_ct,
    plain: np.ndarray,
    *,
    name: str,
    task: str,
    layer_idx: int,
    pack_i: int = 0,
    band_j: int = 0,
    head: int = 0,
    rest_div: float = 8.0,
    post_dualrail_bs: bool = False,
) -> tuple[float, float, float]:
    """
    Compare one DualRail diagonal slice (notebook plot convention).

    Returns (max|err| after scale + LS gain, corr, ls_gain).

    ``sftmx_in`` should be compared to plain **pre-mask** ``att_score`` (see
    ``name_map``): HE applies the attention mask inside Softmax, not as
    additive -inf on the logits CT.

    ``post_dualrail_bs``: unused for amplitude (bts unpack is identity).
    """
    if isinstance(he_ct, np.ndarray) and he_ct.ndim > 1:
        he_ct = he_ct[0]
    dec = engine.decrode(he_ct[pack_i], sk, is_real=True)
    current = dec[2**11 * band_j : 2**11 * (band_j + 1)][_H_INDICES[head]]

    g = plain
    if g.ndim == 3:
        g = g[head].T
    elif name in ("gelu_in", "gelu_out", "gelu_in_wo_bs"):
        g = np.vsplit(g.T, 24)[0]
    else:
        g = np.vsplit(g.T, 6)[head]
    ref = ld(g, pack_i * 16 + band_j)

    cur = np.asarray(current, dtype=np.float64).ravel()
    if post_dualrail_bs and name == "sftmx_in":
        pass  # DualRail bts unpack is identity; do not undo a phantom ×2
    if name == "sftmx_in":
        import sys
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from softmax_poly import score_hf_decode_bake  # noqa: WPS433

        bake = score_hf_decode_bake(task, layer_idx)
        cur = cur[:40] * bake
        ref = ref[:40]
    elif name == "gelu_in_wo_bs":
        from gelu_poly import gelu_scale_for_layer  # noqa: WPS433

        cur = cur * float(gelu_scale_for_layer(layer_idx))
    elif name == "ln2_in":
        cur = cur / 2.0

    n = min(len(cur), len(ref))
    cur, ref = cur[:n], np.asarray(ref[:n], dtype=np.float64)
    denom = float(np.dot(cur, cur))
    if denom > 1e-30:
        g_ls = float(np.dot(cur, ref) / denom)
        cur_cal = cur * g_ls
    else:
        g_ls = float("nan")
        cur_cal = cur
    err = float(np.max(np.abs(cur_cal - ref)))
    err_g1 = float(np.max(np.abs(cur - ref)))
    if np.std(cur) < 1e-15 or np.std(ref) < 1e-15:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(cur, ref)[0, 1])
    if name == "sftmx_in":
        print(
            f"    [sftmx_in diag] max|HE×scale|={float(np.max(np.abs(cur))):.4g}  "
            f"max|ref|={float(np.max(np.abs(ref))):.4g}  "
            f"err@gain1={err_g1:.3e}  post_bs_div2={post_dualrail_bs}",
            flush=True,
        )
    return err, corr, g_ls

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument(
        "--check",
        nargs="+",
        default=["sftmx_in", "att_context", "ln1_out", "ln2_out"],
        help="intermediate names to score (notebook convention)",
    )
    ap.add_argument(
        "--gelu",
        choices=("thor", "cheb"),
        default="cheb",
        help="cheb=repo Chebyshev (default); thor=upstream Stockmeyer tanh",
    )
    ap.add_argument(
        "--gelu-level",
        type=int,
        default=2,
        help="repo GeLU scheme level 0/1/2 when --gelu cheb",
    )
    ap.add_argument(
        "--ln",
        choices=("thor", "poly"),
        default="poly",
        help="poly=repo layernorm_poly (default); thor=upstream he_layernorm*",
    )
    ap.add_argument(
        "--ln-level",
        type=int,
        default=2,
        help="repo LN scheme level 0/1/2 when --ln poly",
    )
    ap.add_argument(
        "--softmax",
        choices=("thor", "poly"),
        default="poly",
        help="poly=repo Stockmeyer+aSOR (default); thor=upstream he_softmax1/2",
    )
    ap.add_argument(
        "--softmax-level",
        type=int,
        default=2,
        help="repo Softmax scheme level 0/1/2 when --softmax poly",
    )
    ap.add_argument(
        "--bootstrap",
        choices=("plan", "plan_mock"),
        default="plan_mock",
        help="plan=bts_ops placements + real bs; "
        "plan_mock=bts_ops placements + decrypt→reencrypt (depth reconcile).",
    )
    ap.add_argument(
        "--mock-land-level",
        type=int,
        default=None,
        help="enc level_calc after mock refresh (default: num_levels-14)",
    )
    ap.add_argument(
        "--plan-discard",
        action="store_true",
        help="after plan bts, short-chain level_up discard (off until DEPTH_* calibrated)",
    )
    ap.add_argument(
        "--encrypt-level",
        type=int,
        default=None,
        help="embedding encrypt level_calc (default: align to plan initial rem)",
    )
    ap.add_argument(
        "--probe-levels",
        action="store_true",
        help="print level_calc/remaining at linear & fixed-bts sites",
    )
    ap.add_argument(
        "--probe-stop",
        type=str,
        default=None,
        help="with --probe-levels, stop after this site (e.g. after_bts_score)",
    )
    ap.add_argument(
        "--rem-tol",
        type=int,
        default=0,
        help="allowed |HE rem − DP rem| at probe/boot sites (plan/plan_mock; default 0)",
    )
    ap.add_argument(
        "--no-rem-guard",
        action="store_true",
        help="disable fail-fast rem_trace checks (plan/plan_mock)",
    )
    ap.add_argument(
        "--input",
        choices=("auto", "embedding", "plain-hidden"),
        default="auto",
        help="layer input CT source: auto=embedding for L0 else plain hidden; "
        "plain-hidden encrypts plaintext BERT h_{layer} (needed for L2 Softmax)",
    )
    ap.add_argument(
        "--rest-div",
        type=float,
        default=1.0,
        help="Score n≥1 mask /rest_div. Default 1 = THOR-native (scale=1 + "
        "KEY/8). Use 8 only with island scale=1/√d; with scale=1 it "
        "distorts sftmx_in vs HF.",
    )
    ap.add_argument(
        "--depth-profile-out",
        type=str,
        default=None,
        help="Route B: write HE depth profile JSON (use with --probe-levels)",
    )
    ap.add_argument(
        "--depth-profile-in",
        type=str,
        default=None,
        help="Route B: load HE depth profile JSON → depth_overrides for optimize_bootstrap",
    )
    ap.add_argument(
        "--depth-profile-unsafe",
        action="store_true",
        help="with --depth-profile-in, also apply fused LN/Softmax probe lumps "
        "(default: safe linear/bridge segments only)",
    )
    ap.add_argument(
        "--depth-profile-validate",
        action="store_true",
        help="with --depth-profile-in, anchor-check probes vs baseline DP rem_trace "
        "before applying overrides (Route B phase 2)",
    )
    ap.add_argument(
        "--depth-profile-require-dp-match",
        action="store_true",
        help="with --depth-profile-validate, only keep overrides where "
        "measured depth == DP depth_charged",
    )
    ap.add_argument(
        "--n0-amp",
        type=float,
        default=None,
        help=argparse.SUPPRESS,  # deprecated alias → rest_div
    )
    args = ap.parse_args()
    if args.n0_amp is not None:
        # Legacy: n0_amp=8 meant dense via n0 boost; now rest_div=8.
        args.rest_div = float(args.n0_amp) if float(args.n0_amp) != 1.0 else 1.0

    os.chdir(_THOR_ROOT)
    paths = _thor_paths(args.dataset)
    layer_idx = int(args.layer)
    _require_assets(paths, layers=[layer_idx])

    devices = [0]
    bts_mock = args.bootstrap in ("mock", "plan_mock")
    print(
        "create engine + load keys0 "
        f"({'rot only; mock bts' if bts_mock else 'bootstrap'}) ...",
        flush=True,
    )
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=not bts_mock,
        with_rot_deltas=True,
    )
    sk, pk = keys["sk"], keys["pk"]
    print(f"  ready ({time.time() - t0:.1f}s) slots={engine.num_slots}", flush=True)

    if bts_mock:
        from dualrail_bts import install_mock_bootstrap
        from bts_ops import BOOTSTRAP_DEPTH_BUDGET as _BUDGET

        land = args.mock_land_level
        if land is None:
            # Always land rem=budget; DualRail unpack ×½ supplies real_pack −1.
            mock_info = install_mock_bootstrap(
                engine, sk, pk, rem_after=int(_BUDGET)
            )
        else:
            mock_info = install_mock_bootstrap(
                engine, sk, pk, land_level=land
            )
        print(
            f"  bootstrap MOCK: decrypt→encodecrypt @ level_calc="
            f"{mock_info['land_level']} "
            f"(rem≈{mock_info.get('rem_after', int(engine.num_levels)-int(mock_info['land_level']))})",
            flush=True,
        )

    print("load plaintext BERT + DualRail encoded weights ...", flush=True)
    # Offline: use local finetuned dir (no HuggingFace hub).
    _repo = os.path.abspath(os.path.join(_HE, ".."))
    local_model = os.path.join(_repo, "finetuned_weight", args.dataset)
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    # ThorDataEncryptor hardcodes tokenizer checkpoint 'bert-base-uncased'.
    _tok_orig = AutoTokenizer.from_pretrained

    def _tok_local(pretrained_model_name_or_path, *a, **k):
        if pretrained_model_name_or_path == "bert-base-uncased":
            pretrained_model_name_or_path = local_model
            k.setdefault("local_files_only", True)
        return _tok_orig(pretrained_model_name_or_path, *a, **k)

    AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]
    import thor.data as _thor_data

    _thor_data.AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]
    model_plain = AutoModelForSequenceClassification.from_pretrained(
        local_model,
        local_files_only=True,
        output_hidden_states=True,
        attn_implementation="eager",
    )
    model_plain.eval()
    print(f"Model loaded from {local_model}", flush=True)
    device = torch.device("cpu")
    model_plain.to(device)

    att_path = encoded_layer_pkl(args.dataset, "att", layer_idx)
    ff_path = encoded_layer_pkl(args.dataset, "ff", layer_idx)
    with open(att_path, "rb") as f:
        weights_pt = pickle.load(f)
    import gc

    gc.collect()
    with open(ff_path, "rb") as f:
        ff_weights = pickle.load(f)
    gc.collect()
    print(
        f"  encoded weights L{layer_idx} "
        f"(att={len(weights_pt)} from {att_path}; "
        f"ff={len(ff_weights)} from {ff_path})",
        flush=True,
    )

    evaluator = ThorLinearEvaluator(engine)
    # Build only the requested layer (partial encode is OK for smoke).
    thor_attention = ThorBertAttention(evaluator, weights_pt, layer_idx)
    weights_pt.clear()
    del weights_pt
    thor_ff = ThorBertFF(evaluator, ff_weights, layer_idx)
    ff_weights.clear()
    del ff_weights
    gc.collect()
    print(
        f"  encoded weights slimmed to layer {layer_idx} "
        f"(held only inside Thor modules; pickle dicts freed)",
        flush=True,
    )

    # DualRail linear: QKV / score / context / FF patches.
    from functools import partial

    from linear_eval import (  # noqa: WPS433
        bind_att_context_mask_complement,
        bind_att_score_mask_complement,
        ensure_ct_ct_matmul_masks,
    )
    from thor.nonlinear.softmax import he_softmax

    rest_div = float(args.rest_div)
    if rest_div <= 0:
        rest_div = 1.0
    ensure_ct_ct_matmul_masks(
        evaluator,
        engine,
        n_max=64,
        level=15,
        bank="ct_ct_matmul_score",
        value_scale=1.0 / rest_div,
    )
    bind_att_score_mask_complement(thor_attention, rest_div=rest_div)
    bind_att_context_mask_complement(thor_attention)
    # Optional: shrink Softmax domain when rest_div>1 shrinks logits (island).
    if abs(rest_div - 1.0) > 1e-12:
        inv = 1.0 / rest_div
        if layer_idx == 2:
            thor_attention.softmax = partial(
                he_softmax,
                engine=engine,
                min_x=-70.0 * inv,
                max_x=70.0 * inv,
                n=2,
                l=4,
                inv_epsilon=2 ** (-18),
                output_alpha=0.01,
                debug=False,
                sk=None,
            )
        else:
            thor_attention.softmax = partial(
                he_softmax,
                engine=engine,
                min_x=-27.2493 * inv,
                max_x=21.72692 * inv,
                n=2,
                l=2,
                inv_epsilon=2 ** (-11),
                output_alpha=0.01,
                debug=False,
                sk=None,
            )
    print(
        f"  DualRail linear patches: score rest_div={rest_div} "
        f"(mask /{rest_div:g}) + PT masks; context PT masks; "
        f"Softmax ranges ×{1.0/rest_div:g}",
        flush=True,
    )

    # --- Bootstrap plan (Phase 3) ---
    from bootstrap_hook import BootstrapHook
    from bts_ops import (
        BOOTSTRAP_DEPTH_BUDGET,
        DEFAULT_L0_ENTRY_REMAINING,
        enc_level_from_plan_initial_rem,
        optimize_bootstrap,
        remaining_to_level_calc,
    )
    from gelu_poly import gelu_level_allowed
    from cost import NUM_LAYERS
    bts_hook = BootstrapHook(engine, mode="always")
    # Build 48-d scheme from CLI poly levels (thor slots still get high=2).
    sm_lv = int(args.softmax_level) if args.softmax == "poly" else 2
    ln_lv = int(args.ln_level) if args.ln == "poly" else 2
    ge_lv = int(args.gelu_level) if args.gelu == "cheb" else 2
    scheme = []
    for li in range(NUM_LAYERS):
        g_lv = ge_lv if gelu_level_allowed(li, ge_lv) else 2
        scheme.extend([sm_lv, ln_lv, g_lv, ln_lv])
    assert len(scheme) == 48
    # L0 入口：HE probe rem = num_levels − level_calc 须等于 DP initial_rem。
    # 8real→4cplx pack（cc_add/imult）不烧 rem，故 encrypt 直接用
    # remaining_to_level_calc(initial_rem)（enc@15 → rem=14），不再套 pack 税 −1。
    plan_initial_rem = DEFAULT_L0_ENTRY_REMAINING
    thor_enc_level = remaining_to_level_calc(
        plan_initial_rem, int(engine.num_levels)
    )
    depth_overrides = None
    if args.depth_profile_in:
        from he_depth_profile import DepthProfile, build_depth_overrides, validate_profile_vs_dp

        _prof = DepthProfile.load(args.depth_profile_in)
        _base_plan = None
        if args.depth_profile_validate:
            _base_plan = optimize_bootstrap(
                args.dataset,
                scheme,
                budget=BOOTSTRAP_DEPTH_BUDGET,
                initial_level=plan_initial_rem,
                phase="C",
            )
            _vrows = validate_profile_vs_dp(
                _prof,
                _base_plan,
                layer_idx=layer_idx,
                require_dp_match=args.depth_profile_require_dp_match,
            )
            for _vr in _vrows:
                _flag = "OK" if _vr.accept else "SKIP"
                print(
                    f"  profile validate [{_flag}] {_vr.event_name}: "
                    f"meas={_vr.measured} dp={_vr.dp_depth} "
                    f"anchor={'Y' if _vr.anchor_ok else 'N'} {_vr.detail}",
                    flush=True,
                )
        depth_overrides = build_depth_overrides(
            _prof,
            safe_only=not args.depth_profile_unsafe,
            plan_result=_base_plan,
            layer_idx=layer_idx,
            validate=args.depth_profile_validate,
            require_dp_match=args.depth_profile_require_dp_match,
        )
        print(
            f"  Route B depth_overrides: {len(depth_overrides)} entries "
            f"({'unsafe' if args.depth_profile_unsafe else 'safe'}"
            f"{', validated' if args.depth_profile_validate else ''}) "
            f"from {args.depth_profile_in}",
            flush=True,
        )
    plan_result = optimize_bootstrap(
        args.dataset,
        scheme,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        initial_level=plan_initial_rem,
        phase="C",
        depth_overrides=depth_overrides,
    )
    if args.bootstrap in ("plan", "plan_mock"):
        bts_hook.load_plan(
            plan_result, apply_discards=bool(args.plan_discard)
        )
        print(
            f"  bootstrap {'plan_mock' if args.bootstrap == 'plan_mock' else 'plan'}: "
            f"{plan_result.summary()} "
            f"(initial_rem={plan_initial_rem}; enc level_calc={thor_enc_level}; "
            f"discard={'on' if args.plan_discard else 'off'})",
            flush=True,
        )
        from rem_guard import RemGuard, write_rem_trace

        rem_guard = None
        if not args.no_rem_guard:
            rem_guard = RemGuard.from_bootstrap_result(
                plan_result,
                layer_idx=layer_idx,
                tol=int(args.rem_tol),
            )
            bts_hook.rem_guard = rem_guard
            trace_path = f"/tmp/l{layer_idx}_rem_trace.tsv"
            write_rem_trace(plan_result, trace_path, layer_idx=layer_idx)
            print(rem_guard.dump_layer_trace(), flush=True)
            print(f"  rem_trace written → {trace_path}", flush=True)
        else:
            rem_guard = None
    else:
        rem_guard = None

    if args.gelu == "cheb":
        from gelu_he_ct import bind_ff_gelu_cheb

        gp = bind_ff_gelu_cheb(
            thor_ff,
            layer_idx=layer_idx,
            level=int(args.gelu_level),
            suppress_post_bts=True,
            bts_hook=bts_hook,
        )
        print(
            f"  GeLU swapped → Chebyshev ({gp.scheme_name}, depth_he={gp.depth_he})",
            flush=True,
        )
    if args.ln == "poly":
        from layernorm_he_ct import bind_layer_layernorm

        p_ln1, p_ln2 = bind_layer_layernorm(
            thor_attention,
            thor_ff,
            task=args.dataset,
            layer_idx=layer_idx,
            level=int(args.ln_level),
            bts_hook=bts_hook,
        )
        print(
            f"  LN swapped → poly "
            f"ln1(iters={p_ln1.invsqrt_max_iters}, "
            f"var=[{p_ln1.min_var:.4g},{p_ln1.max_var:.4g}]) "
            f"ln2(iters={p_ln2.invsqrt_max_iters}, "
            f"var=[{p_ln2.min_var:.4g},{p_ln2.max_var:.4g}])",
            flush=True,
        )
    if args.softmax == "poly":
        from softmax_he_ct import bind_attention_softmax

        sp = bind_attention_softmax(
            thor_attention,
            task=args.dataset,
            layer_idx=layer_idx,
            level=int(args.softmax_level),
            bts_hook=bts_hook,
        )
        print(
            f"  Softmax swapped → poly "
            f"(δ2={sp.delta2:g}, shift={sp.shift:g}, "
            f"iters_σ={sp.asor_max_iters_sigma}, "
            f"iters_Σy²={sp.asor_max_iters_sum_sq}, "
            f"e0_σ={sp.e0_sigma:.4g})",
            flush=True,
        )

    print("encrypt validation sample ...", flush=True)
    emb_model = model_plain.bert.embeddings
    data_encryptor = ThorDataEncryptor(
        args.dataset,
        paths["dataset"],
        embedding_model=emb_model,
        ckks_engine=engine,
        test=False,
    )
    batch = None
    for idx, b in enumerate(data_encryptor.eval_dataloader):
        if idx == args.sample:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"sample {args.sample} not in dataloader")

    data = {k: v for k, v in batch.items() if k in ("input_ids", "token_type_ids")}
    embedding = data_encryptor.embed_data(data)
    attention_mask = batch["attention_mask"]
    thor_attention_mask = data_encryptor.encode_attention_mask(
        attention_mask.cpu().numpy().squeeze(), level=15
    )

    batch_cpu = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }
    # HE 对拍参考：与 plan/HE 同一 48 维 scheme（含 GeLU 层组约束）。
    _repo_root = os.path.abspath(os.path.join(_HE, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from poly_model_inference import (  # noqa: WPS433
        apply_polynomial_scheme,
        install_eager_attention_poly_patch,
    )

    ref_scheme = list(scheme)
    install_eager_attention_poly_patch()
    apply_polynomial_scheme(model_plain, ref_scheme, args.dataset)
    plain_refs = _poly_layer_refs(
        model_plain, batch_cpu, batch_cpu["attention_mask"], device
    )
    print(
        f"  compare refs = poly_model_inference "
        f"(same 48-d scheme as plan; "
        f"CLI sm/ln/gelu="
        f"{args.softmax_level}/{args.ln_level}/{args.gelu_level})",
        flush=True,
    )
    # Poly refs are numpy; drop HF BERT params (keep encryptor until DualRail enc).
    del model_plain, emb_model
    import gc as _gc_free

    _gc_free.collect()

    # Notebook default level=20; plan mode uses initial remaining_keep → level_calc.
    if args.encrypt_level is not None:
        enc_level = int(args.encrypt_level)
    elif args.bootstrap in ("plan", "plan_mock"):
        keep0 = bts_hook.remaining_keep_after(None)
        if keep0 is None:
            keep0 = plan_initial_rem
        plan_lc = remaining_to_level_calc(int(keep0), int(engine.num_levels))
        enc_level = thor_enc_level
        print(
            f"  plan initial rem={plan_initial_rem} "
            f"(discard keep={keep0} → lc≈{plan_lc}; discard off) "
            f"encrypt level_calc={enc_level} → HE rem={plan_initial_rem}",
            flush=True,
        )
    else:
        enc_level = enc_level_from_plan_initial_rem(
            DEFAULT_L0_ENTRY_REMAINING, int(engine.num_levels)
        )

    input_mode = args.input
    if input_mode == "auto":
        input_mode = "embedding" if layer_idx == 0 else "plain-hidden"
    if input_mode == "embedding":
        enc_arr = np.asarray(embedding, dtype=np.float64)
        if enc_arr.ndim == 3:
            enc_arr = enc_arr.squeeze(0)
        print(
            f"  encrypt 8-real from embedding {tuple(enc_arr.shape)} "
            f"level_calc={enc_level}",
            flush=True,
        )
    else:
        enc_arr = np.asarray(plain_refs[layer_idx]["x"], dtype=np.float64)
        if enc_arr.ndim == 3:
            enc_arr = enc_arr.squeeze(0)
        if enc_arr.shape != (128, 768):
            raise ValueError(
                f"poly hidden shape {enc_arr.shape} ≠ (128, 768); "
                "cannot encrypt as layer input"
            )
        print(
            f"  encrypt 8-real from poly h_L{layer_idx} {tuple(enc_arr.shape)} "
            f"level_calc={enc_level} (isolates layer; not HE-chained)",
            flush=True,
        )
    from linear_eval import encrypt_embedding_real8

    x = encrypt_embedding_real8(engine, enc_arr, pk, level=enc_level)
    del data_encryptor, embedding, enc_arr
    _gc_free.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from level_probe import LevelProbe, ProbeStop
    from rem_guard import RemMismatchError

    depth_recorder = None
    if args.depth_profile_out:
        from he_depth_profile import DepthProfileRecorder

        depth_recorder = DepthProfileRecorder(
            task=args.dataset,
            scheme=scheme,
            num_levels=int(engine.num_levels),
            enc_level=int(enc_level),
            layer_idx=layer_idx,
        )
        depth_recorder.attach(bts_hook)

    level_probe = None
    if args.probe_levels or args.depth_profile_out:
        level_probe = LevelProbe(
            num_levels=int(engine.num_levels),
            stop_after=args.probe_stop,
        )

    t1 = time.time()
    try:
        _ln2, he_vars = _forward_layer(
            engine=engine,
            evaluator=evaluator,
            thor_attention=thor_attention,
            thor_ff=thor_ff,
            layer_idx=layer_idx,
            x=x,
            thor_attention_mask=thor_attention_mask,
            devices=devices,
            bts_hook=bts_hook,
            level_probe=level_probe,
            rem_guard=rem_guard,
        )
    except RemMismatchError as exc:
        print(f"  REM GUARD STOP ({time.time() - t1:.1f}s): {exc}", flush=True)
        if rem_guard is not None:
            print(rem_guard.dump_layer_trace(), flush=True)
        return 3
    except ProbeStop as exc:
        print(f"  probe stop @ {exc.site} ({time.time() - t1:.1f}s)", flush=True)
        print(exc.probe.summary(), flush=True)
        if depth_recorder is not None and level_probe is not None:
            depth_recorder.ingest_probe(level_probe)
        if args.depth_profile_out and depth_recorder is not None:
            prof = depth_recorder.finalize()
            prof.save(args.depth_profile_out)
            depth_recorder.detach()
            print(
                f"  depth profile → {args.depth_profile_out} "
                f"({len(prof.segment_depths)} segments, {len(prof.boots)} boots)",
                flush=True,
            )
        # Early compare when stopping at score / context (DualRail linear check).
        _early_map = {
            "after_att_score": "sftmx_in",
            "after_bts_score": "sftmx_in",
            "after_att_context": "att_context",
            "after_bts_context": "att_context",
        }
        early_name = _early_map.get(exc.site)
        if (
            early_name is not None
            and early_name in args.check
            and exc.cts is not None
        ):
            pref = plain_refs[layer_idx]
            # HE sftmx_in is pre-mask; poly key is att_score.
            plain_key = "att_score" if early_name == "sftmx_in" else early_name
            post_bs = exc.site.endswith("_bts_score") or exc.site.endswith(
                "_bts_context"
            )
            if early_name == "sftmx_in":
                err, corr, gain_ls = _he_vs_plain_score_packs(
                    engine,
                    sk,
                    exc.cts,
                    pref[plain_key],
                    task=args.dataset,
                    layer_idx=layer_idx,
                    post_dualrail_bs=post_bs,
                    rest_div=float(args.rest_div),
                )
            else:
                err, corr, gain_ls = _he_vs_plain_diag(
                    engine,
                    sk,
                    exc.cts,
                    pref[plain_key],
                    name=early_name,
                    task=args.dataset,
                    layer_idx=layer_idx,
                    rest_div=float(args.rest_div),
                    post_dualrail_bs=post_bs,
                )
            status = "OK" if err < args.tol else "BAD"
            print(
                f"  [{status}] {early_name} @{exc.site}: max|err|={err:.3e}  "
                f"corr={corr:.4f}  ls_gain={gain_ls:.4g}  tol={args.tol}",
                flush=True,
            )
            print("PASS" if err < args.tol else "FAIL")
            return 0 if err < args.tol else 2
        return 0
    print(f"  HE layer done ({time.time() - t1:.1f}s)", flush=True)
    if depth_recorder is not None and level_probe is not None:
        depth_recorder.ingest_probe(level_probe)
    if args.depth_profile_out and depth_recorder is not None:
        prof = depth_recorder.finalize()
        prof.save(args.depth_profile_out)
        depth_recorder.detach()
        print(
            f"  depth profile → {args.depth_profile_out} "
            f"({len(prof.segment_depths)} segments, {len(prof.boots)} boots)",
            flush=True,
        )
    mock_info = getattr(engine, "_mock_bootstrap_info", None)
    if mock_info is not None:
        print(
            f"  mock bootstrap calls={mock_info['calls']['n']} "
            f"land_level={mock_info['land_level']} "
            f"(rem≈{mock_info.get('rem_after', '?')})",
            flush=True,
        )
    if level_probe is not None:
        print(level_probe.summary(), flush=True)
    summ = bts_hook.summary()
    print(
        f"  BootstrapHook: mode={summ['mode']} calls={summ['calls']} "
        f"ct_boots={summ['total_ct_bootstraps']}",
        flush=True,
    )
    for ev, n in sorted(summ["by_event"].items()):
        if n:
            print(f"    {ev}: {n}", flush=True)

    pref = plain_refs[layer_idx]
    # Map HE names → poly ref keys（与 poly_model_inference 中间量同名）
    name_map = {
        # HE score CT is pre-mask; poly sftmx_in includes additive mask.
        "sftmx_in": "att_score",
        "sftmx_out": "sftmx_out",
        "att_context": "att_context",
        "ln1_in": "ln1_in",
        "ln1_out": "ln1_out",
        "gelu_in_wo_bs": "gelu_in",
        "gelu_out": "gelu_out",
        "dense2_out": "dense2_out",
        "ln2_in": "ln2_in",
        "ln2_out": "ln2_out",
        "x": "x",
        "q": "q",
    }

    ok = True
    cheb_params = getattr(thor_ff, "_gelu_cheb_params", None)
    for name in args.check:
        he = he_vars[name]
        plain_key = name_map.get(name, name)
        plain_arr = pref[plain_key]
        # Fair Cheb check: twin of **HE** gelu_in (same DualRail slice), not
        # plain BERT gelu_in — upstream att/LN error would otherwise dominate.
        if (
            name == "gelu_out"
            and args.gelu == "cheb"
            and cheb_params is not None
            and "gelu_in_wo_bs" in he_vars
        ):
            from slot_gelu_he import gelu_poly_slots

            pack_i, band_j, head = 0, 0, 0
            gin_ct = he_vars["gelu_in_wo_bs"]
            if isinstance(gin_ct, np.ndarray) and gin_ct.ndim > 1:
                gin_ct = gin_ct[0]
            gin_dec = engine.decrode(gin_ct[pack_i], sk, is_real=True)
            gin_slice = np.asarray(
                gin_dec[2**11 * band_j : 2**11 * (band_j + 1)][_H_INDICES[head]],
                dtype=np.float64,
            ).ravel()
            ref_slice = gelu_poly_slots(gin_slice, cheb_params)

            gout_ct = he
            if isinstance(gout_ct, np.ndarray) and gout_ct.ndim > 1:
                gout_ct = gout_ct[0]
            gout_dec = engine.decrode(gout_ct[pack_i], sk, is_real=True)
            cur = np.asarray(
                gout_dec[2**11 * band_j : 2**11 * (band_j + 1)][_H_INDICES[head]],
                dtype=np.float64,
            ).ravel()
            n = min(len(cur), len(ref_slice))
            cur, ref_slice = cur[:n], np.asarray(ref_slice[:n], dtype=np.float64)
            denom = float(np.dot(cur, cur))
            gain_ls = (
                float(np.dot(cur, ref_slice) / denom) if denom > 1e-30 else float("nan")
            )
            cur_cal = cur * gain_ls if denom > 1e-30 else cur
            err = float(np.max(np.abs(cur_cal - ref_slice)))
            corr = (
                float(np.corrcoef(cur, ref_slice)[0, 1])
                if np.std(cur) > 1e-15 and np.std(ref_slice) > 1e-15
                else float("nan")
            )
            print(
                f"  (gelu_out ref = Cheb twin of HE gelu_in, "
                f"{cheb_params.scheme_name})",
                flush=True,
            )
        else:
            if name == "sftmx_in":
                err, corr, gain_ls = _he_vs_plain_score_packs(
                    engine,
                    sk,
                    he,
                    plain_arr,
                    task=args.dataset,
                    layer_idx=layer_idx,
                    post_dualrail_bs=False,
                    rest_div=float(args.rest_div),
                )
            else:
                err, corr, gain_ls = _he_vs_plain_diag(
                    engine,
                    sk,
                    he,
                    plain_arr,
                    name=name,
                    task=args.dataset,
                    layer_idx=layer_idx,
                    rest_div=float(args.rest_div),
                    post_dualrail_bs=False,
                )
        status = "OK" if err < args.tol else "BAD"
        if err >= args.tol:
            ok = False
        print(
            f"  [{status}] {name}: max|err|={err:.3e}  corr={corr:.4f}  "
            f"ls_gain={gain_ls:.4g}  tol={args.tol}",
            flush=True,
        )

    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
