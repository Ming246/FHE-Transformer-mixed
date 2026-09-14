#!/usr/bin/env python3
"""
Encoder **HE-isomorphic plain simulation** (no Liberate / no CKKS).

DualRail score / context / W_O / FF packing — same operator graph as
``smoke_thor_repro.py``, with poly Softmax / LN / GeLU (``slot_*_he``).
No bootstrap, no encrypt/decrypt.

Compares L0..N checkpoints against **both**:
  - exact plaintext BERT (``_plain_layer_refs``)
  - dense poly BERT (``poly_model_inference`` Softmax/GeLU/LN at ``--level``)

Usage::

  python3 HE/thor/smoke_encoder_plain_sim.py --until-layer 0 --sample 0
  python3 HE/thor/smoke_encoder_plain_sim.py --until-layer 0 --level 2 --rest-div 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE_ROOT, ".."))
for p in (_HERE, _HE_ROOT, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE_ROOT)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)


def _ensure_he_on_path() -> None:
    """``linear_eval`` → ``dualrail_encode`` calls ``ensure_thor_path`` and drops ``HE/``."""
    if _HE_ROOT not in sys.path:
        sys.path.insert(0, _HE_ROOT)

import thor_encoder_linear_core as core  # noqa: E402
from linear_eval import (  # noqa: E402
    complexify_v_plain,
    compute_attention_context_dualrail_island,
    compute_attention_score_dualrail_island,
    decode_qkv_upper_to_dense,
    decode_score_packs,
    encode_ff_dense1_bias_plain,
    encode_ff_dense2_bias_plain,
    encode_qkv_upper_from_dense,
    ff_dense1_plain,
    ff_dense2_plain,
    ff_input_rots_from_lower8_plain,
    mainline_qkv_ref,
    compute_wo_dualrail_plain,
)
_ensure_he_on_path()
from slot_decode import (  # noqa: E402
    decode_input_from_lower_diagonals,
    merge_heads,
)
from slot_gelu_he import GeluHeParams, slot_gelu_he  # noqa: E402
from slot_layernorm_he import (  # noqa: E402
    LayerNormHeParams,
    encode_affine_to_input_lower,
    slot_layernorm_he,
)
from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    encode_key_valid_mask_packs,
    slot_softmax_he,
)
_ensure_he_on_path()
from gelu_poly import gelu_ff1_encode_scale  # noqa: E402
from poly_model_inference import (  # noqa: E402
    apply_polynomial_scheme,
    install_eager_attention_poly_patch,
)
from softmax_poly import k_key_encode_scale, score_hf_decode_bake  # noqa: E402
from smoke_thor_repro import (  # noqa: E402
    _plain_layer_refs,
    _require_assets,
    _thor_paths,
    _transpose_for_scores,
)


def _uniform_poly_scheme(level: int) -> list[int]:
    """48-dim scheme: Softmax/LN at ``level``; GeLU respects layer-group forbid."""
    from cost import NUM_LAYERS
    from gelu_poly import gelu_level_allowed

    lv = int(level)
    out: list[int] = []
    for layer_idx in range(NUM_LAYERS):
        g_lv = lv if gelu_level_allowed(layer_idx, lv) else 2
        out.extend([lv, lv, g_lv, lv])
    return out


def _poly_layer_refs(model_poly, batch, attention_mask, device):
    """
    Dense poly intermediates (same sites as ``_plain_layer_refs``).

    Requires ``install_eager_attention_poly_patch`` + ``apply_polynomial_scheme``
    already applied so GeLU/LN modules and ``_poly_softmax_fn`` are set.
    Softmax uses the poly evaluator (``_plain_layer_refs`` hardcodes ``F.softmax``).
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
                    "att_score": attention_scores.cpu().numpy().squeeze(),
                    "att_context": att_context.cpu().numpy().squeeze(),
                    "ln1_out": ln1_out.cpu().numpy().squeeze(),
                    "ln2_out": ln2_out.cpu().numpy().squeeze(),
                    "gelu_in": gelu_in.cpu().numpy().squeeze(),
                    "gelu_out": gelu_out.cpu().numpy().squeeze(),
                }
            )
            hs = ln2_out
    return refs


def _k_scale_bake(task: str, layer_idx: int) -> tuple[float, float]:
    return k_key_encode_scale(task, layer_idx), score_hf_decode_bake(task, layer_idx)


def _layer_weights(model, layer_idx: int) -> dict[str, np.ndarray]:
    bl = model.bert.encoder.layer[layer_idx]
    att = bl.attention.self
    return {
        "w_q": att.query.weight.detach().cpu().numpy().astype(np.float64),
        "w_k": att.key.weight.detach().cpu().numpy().astype(np.float64),
        "w_v": att.value.weight.detach().cpu().numpy().astype(np.float64),
        "w_o": bl.attention.output.dense.weight.detach().cpu().numpy().astype(
            np.float64
        ),
        "w1": bl.intermediate.dense.weight.detach().cpu().numpy().astype(np.float64),
        "w2": bl.output.dense.weight.detach().cpu().numpy().astype(np.float64),
        "bq": att.query.bias.detach().cpu().numpy().astype(np.float64),
        "bk": att.key.bias.detach().cpu().numpy().astype(np.float64),
        "bv": att.value.bias.detach().cpu().numpy().astype(np.float64),
        "bo": bl.attention.output.dense.bias.detach().cpu().numpy().astype(np.float64),
        "b1": bl.intermediate.dense.bias.detach().cpu().numpy().astype(np.float64),
        "b2": bl.output.dense.bias.detach().cpu().numpy().astype(np.float64),
    }


def _layer_poly_meta(model, layer_idx: int, cfg: core.ThorConfig, task: str, level: int):
    bl = model.bert.encoder.layer[layer_idx]
    ln1, ln2 = bl.attention.output.LayerNorm, bl.output.LayerNorm
    g1 = ln1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = ln1.bias.detach().cpu().numpy().astype(np.float64)
    g2 = ln2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = ln2.bias.detach().cpu().numpy().astype(np.float64)
    eps = float(ln1.eps)
    return {
        "softmax_he_params": SoftmaxHeParams.from_softmax_poly(
            task, layer_idx, level=level
        ),
        "gelu_he_params": GeluHeParams.from_gelu_poly(layer_idx, level=level),
        "ln1_he_params": LayerNormHeParams.from_layernorm_poly(
            task, layer_idx, "ln1", level=level
        ),
        "ln2_he_params": LayerNormHeParams.from_layernorm_poly(
            task, layer_idx, "ln2", level=level
        ),
        "ln1_gamma_beta": encode_affine_to_input_lower(g1, b1, cfg),
        "ln2_gamma_beta": encode_affine_to_input_lower(g2, b2, cfg),
        "eps": eps,
    }


def _qkv_upper_with_bias(
    x_dense: np.ndarray,
    w: dict[str, np.ndarray],
    cfg: core.ThorConfig,
    *,
    task: str,
    layer_idx: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """PC-MM Q/K/V upper packs + HF bias (KEY bake on K)."""
    k_scale, _ = _k_scale_bake(task, layer_idx)
    q_pc = mainline_qkv_ref(x_dense, w["w_q"], scale=1.0, out_packs=4)
    k_pc = mainline_qkv_ref(x_dense, w["w_k"], scale=k_scale, out_packs=4)
    v_pc = mainline_qkv_ref(x_dense, w["w_v"], scale=1.0, out_packs=4)
    q_d = decode_qkv_upper_to_dense(q_pc, cfg) + w["bq"]
    k_d = decode_qkv_upper_to_dense(k_pc, cfg) + k_scale * w["bk"]
    v_d = decode_qkv_upper_to_dense(v_pc, cfg) + w["bv"]
    return (
        encode_qkv_upper_from_dense(q_d, cfg),
        encode_qkv_upper_from_dense(k_d, cfg),
        encode_qkv_upper_from_dense(v_d, cfg),
    )


def forward_encoder_layer_dualrail_plain(
    x8: list[np.ndarray],
    w: dict[str, np.ndarray],
    cfg: core.ThorConfig,
    masks: core.MaskBank,
    *,
    layer_idx: int,
    rest_div: float,
    mask_packs: list[np.ndarray],
    softmax_he_params: SoftmaxHeParams,
    gelu_he_params: GeluHeParams,
    ln1_he_params: LayerNormHeParams,
    ln2_he_params: LayerNormHeParams,
    ln1_gamma_beta,
    ln2_gamma_beta,
    ln_eps: float,
    task: str,
) -> dict[str, object]:
    """
    One layer, HE-isomorphic DualRail plain twin (no CKKS).

    Returns dense + pack intermediates for checkpoint compare.
    """
    qp, kp, vp = _qkv_upper_with_bias(
        decode_input_from_lower_diagonals(x8, cfg),
        w,
        cfg,
        task=task,
        layer_idx=layer_idx,
    )

    sftmx_in = compute_attention_score_dualrail_island(
        qp,
        kp,
        cfg,
        scale=1.0,
        gain=1.0,
        rest_div=float(rest_div),
        ct_ct=masks.ct_ct_score,
    )
    alpha8 = slot_softmax_he(sftmx_in, mask_packs, cfg, softmax_he_params)
    # Context: DualRail ttemp BSGS + ``make_copies(α)`` — same as
    # ``encoder_layer_he`` plain island (HE CT Softmax exit differs; see TODO).
    ctx_packs = compute_attention_context_dualrail_island(
        alpha8,
        vp,
        cfg,
        scale=1.0,
        gain=1.0,
        ct_ct=masks.ct_ct_context,
        alpha_copy_mode="make_copies",
    )

    wo_packs = compute_wo_dualrail_plain(
        complexify_v_plain(ctx_packs), w["w_o"], cfg, scale=1.0, out_packs=8
    )
    wo_dense = decode_input_from_lower_diagonals(wo_packs, cfg) + w["bo"]
    ln1_in = core.slot_vec_add(
        x8, list(core.encode_input_lower_diagonals(wo_dense, cfg))
    )
    g1, b1 = ln1_gamma_beta
    ln1_out = slot_layernorm_he(ln1_in, g1, b1, cfg, ln1_he_params, eps=ln_eps)

    x_rots = ff_input_rots_from_lower8_plain(ln1_out)
    gelu_scale = gelu_ff1_encode_scale(layer_idx)
    b1_packs = encode_ff_dense1_bias_plain(w["b1"], cfg, scale=gelu_scale)
    gelu_in_wo_bs = ff_dense1_plain(
        x_rots,
        w["w1"],
        cfg,
        scale=gelu_scale,
        out_packs=8,
        bias_28=b1_packs,
    )
    gelu_out = slot_gelu_he(gelu_in_wo_bs, gelu_he_params)
    assert isinstance(gelu_out, list) and isinstance(gelu_out[0], list)
    b2_packs = encode_ff_dense2_bias_plain(w["b2"], cfg)
    dense2_out = ff_dense2_plain(
        gelu_out,
        w["w2"],
        cfg,
        scale=1.0,
        out_packs=8,
        bias8=b2_packs,
    )
    ln2_in = core.slot_vec_add(ln1_out, dense2_out)
    g2, b2 = ln2_gamma_beta
    ln2_out = slot_layernorm_he(ln2_in, g2, b2, cfg, ln2_he_params, eps=ln_eps)

    return {
        "x8": x8,
        "sftmx_in": sftmx_in,
        "alpha8": alpha8,
        "att_context": ctx_packs,
        "ln1_in": ln1_in,
        "ln1_out": ln1_out,
        "gelu_in_wo_bs": gelu_in_wo_bs,
        "gelu_out": gelu_out,
        "dense2_out": dense2_out,
        "ln2_in": ln2_in,
        "ln2_out": ln2_out,
    }


def _stats(name: str, got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    err = float(np.max(np.abs(g - r)))
    if np.std(g) > 1e-15 and np.std(r) > 1e-15:
        corr = float(np.corrcoef(g, r)[0, 1])
    else:
        corr = float("nan")
    print(
        f"  [{name}] max|err|={err:.3e} corr={corr:.6f}",
        flush=True,
    )
    return err, corr


def _compare_site(
    name: str,
    got: dict,
    pref: dict,
    cfg: core.ThorConfig,
    *,
    task: str,
    layer_idx: int,
    tol: float,
    ref_tag: str,
) -> bool:
    _, bake = _k_scale_bake(task, layer_idx)

    if name == "sftmx_in":
        he_dense = decode_score_packs(got["sftmx_in"], cfg) * bake
        ref = np.transpose(np.asarray(pref["att_score"], dtype=np.float64), (0, 2, 1))
        err, _corr = _stats(f"{ref_tag}.{name}", he_dense, ref)
    elif name == "att_context":
        got_d = decode_qkv_upper_to_dense(got["att_context"], cfg)
        ref_h = np.asarray(pref["att_context"], dtype=np.float64)
        ref_d = merge_heads(ref_h, cfg)
        err, _corr = _stats(f"{ref_tag}.{name}", got_d, ref_d)
    elif name == "ln1_out":
        got_d = decode_input_from_lower_diagonals(got["ln1_out"], cfg)
        ref_d = np.asarray(pref["ln1_out"], dtype=np.float64)
        err, _corr = _stats(f"{ref_tag}.{name}", got_d, ref_d)
    elif name == "ln2_out":
        got_d = decode_input_from_lower_diagonals(got["ln2_out"], cfg)
        ref_d = np.asarray(pref["ln2_out"], dtype=np.float64)
        err, _corr = _stats(f"{ref_tag}.{name}", got_d, ref_d)
    else:
        raise ValueError(f"unknown check site: {name}")

    gate = err < tol
    tag = "PASS" if gate else "FAIL"
    print(
        f"  L{layer_idx}.{name} vs {ref_tag}: {tag} (tol={tol})",
        flush=True,
    )
    return gate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--until-layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--rest-div", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument(
        "--tol-poly",
        type=float,
        default=None,
        help="tol vs dense poly (default: same as --tol)",
    )
    ap.add_argument(
        "--check",
        nargs="+",
        default=["sftmx_in", "att_context", "ln1_out", "ln2_out"],
    )
    args = ap.parse_args()
    tol_poly = float(args.tol if args.tol_poly is None else args.tol_poly)

    until = int(args.until_layer)
    if until < 0 or until > 11:
        raise ValueError(f"until-layer 须在 0..11，收到 {until}")

    paths = _thor_paths(args.dataset)
    _require_assets(paths)

    cfg = core.bert_base()
    cfg.validate()
    masks = core.MaskBank(cfg)

    local_model = os.path.join(_REPO, "finetuned_weight", args.dataset)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _tok_orig = AutoTokenizer.from_pretrained

    def _tok_local(name, *a, **k):
        if name == "bert-base-uncased":
            name = local_model
            k.setdefault("local_files_only", True)
        return _tok_orig(name, *a, **k)

    AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]

    print(
        f"HE-iso plain-sim L0..L{until} sample={args.sample} "
        f"poly_level={args.level} rest_div={args.rest_div} "
        f"(DualRail; vs BERT + vs poly)",
        flush=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        local_model, local_files_only=True, output_hidden_states=True
    )
    model.eval()

    from thor.data import ThorDataEncryptor

    import thor.data as _td

    _td.AutoTokenizer.from_pretrained = _tok_local  # type: ignore[method-assign]

    class _PlainEngine:
        num_slots = cfg.num_slots

    enc = ThorDataEncryptor(
        args.dataset,
        paths["dataset"],
        embedding_model=model.bert.embeddings,
        ckks_engine=_PlainEngine(),
        test=False,
    )
    batch = None
    for idx, b in enumerate(enc.eval_dataloader):
        if idx == args.sample:
            batch = b
            break
    if batch is None:
        raise RuntimeError(f"sample {args.sample} not in dataloader")

    device = torch.device("cpu")
    batch_cpu = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }
    plain_refs = _plain_layer_refs(
        model, batch_cpu, batch_cpu["attention_mask"], device
    )

    install_eager_attention_poly_patch()
    apply_polynomial_scheme(
        model, _uniform_poly_scheme(int(args.level)), args.dataset
    )
    poly_refs = _poly_layer_refs(
        model, batch_cpu, batch_cpu["attention_mask"], device
    )

    data = {k: v for k, v in batch.items() if k in ("input_ids", "token_type_ids")}
    emb = enc.embed_data(data)
    if emb.ndim == 3:
        emb = emb.squeeze(0)
    key_valid = (batch["attention_mask"].cpu().numpy().squeeze() > 0).astype(
        np.float64
    )
    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)

    x8 = list(core.encode_input_lower_diagonals(emb.astype(np.float64), cfg))
    ok = True
    level = int(args.level)
    task = args.dataset

    for layer_idx in range(until + 1):
        w = _layer_weights(model, layer_idx)
        meta = _layer_poly_meta(model, layer_idx, cfg, task, level)
        print(f"\n--- L{layer_idx} DualRail plain-sim forward ---", flush=True)
        got = forward_encoder_layer_dualrail_plain(
            x8,
            w,
            cfg,
            masks,
            layer_idx=layer_idx,
            rest_div=float(args.rest_div),
            mask_packs=mask_packs,
            softmax_he_params=meta["softmax_he_params"],
            gelu_he_params=meta["gelu_he_params"],
            ln1_he_params=meta["ln1_he_params"],
            ln2_he_params=meta["ln2_he_params"],
            ln1_gamma_beta=meta["ln1_gamma_beta"],
            ln2_gamma_beta=meta["ln2_gamma_beta"],
            ln_eps=meta["eps"],
            task=task,
        )
        pref_bert = plain_refs[layer_idx]
        pref_poly = poly_refs[layer_idx]
        for site in args.check:
            ok = (
                _compare_site(
                    site,
                    got,
                    pref_bert,
                    cfg,
                    task=task,
                    layer_idx=layer_idx,
                    tol=float(args.tol),
                    ref_tag="BERT",
                )
                and ok
            )
            ok = (
                _compare_site(
                    site,
                    got,
                    pref_poly,
                    cfg,
                    task=task,
                    layer_idx=layer_idx,
                    tol=tol_poly,
                    ref_tag="poly",
                )
                and ok
            )
        x8 = got["ln2_out"]  # type: ignore[assignment]

    print("\nOVERALL PASS" if ok else "\nOVERALL FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
