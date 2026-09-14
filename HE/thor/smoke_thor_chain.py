#!/usr/bin/env python3
"""
HE chain smoke: run consecutive DualRail encoder layers (default L0→L1).

Encrypts embedding once, feeds each layer's ``ln2_out`` into the next
(notebook ``forward_layer`` chaining). Compares the **last** layer's
intermediates to plaintext BERT.

Weight memory: each layer reads ``ff_L{{n}}.pkl`` / ``att_L{{n}}.pkl``
(one DualRail encode file per layer), then releases that layer.

Usage::

  python3 HE/thor/smoke_thor_chain.py --layers 0 1 2 \\
    --gelu cheb --ln poly --softmax poly --bootstrap plan_mock

  python3 HE/thor/smoke_thor_chain.py --layers 0 1 2 3 4 5 \\
    --bootstrap plan_mock --check ln2_out \\
    --save-out results/he_chain/mrpc_L5_ln2_out.pkl

  python3 HE/thor/smoke_thor_chain.py --layers 6 7 8 \\
    --bootstrap plan_mock --check ln2_out \\
    --load-in results/he_chain/mrpc_L5_ln2_out.pkl \\
    --save-out results/he_chain/mrpc_L8_ln2_out.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
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
from smoke_thor_repro import (  # noqa: E402
    _forward_layer,
    _he_vs_plain_diag,
    _plain_layer_refs,
    _release_he_module_weights,
    _require_assets,
    _thor_paths,
    encoded_layer_pkl,
)
from thor.bert import ThorBertAttention, ThorBertFF  # noqa: E402
from thor.data import ThorDataEncryptor  # noqa: E402
from thor.linear import ThorLinearEvaluator  # noqa: E402


def _gc_cuda() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _slim_encoded_pkl_to_temp(
    src_path: str, layers: list[int], *, label: str
) -> dict[int, str]:
    """Load full DualRail pkl in a **subprocess**; write one temp pkl per layer.

    Child exits after dump so the 9–19 GiB deserialize RSS returns to the OS
    before the next pkl / engine load (in-process GC is not enough on ~46 GiB).
    """
    import json
    import subprocess
    import tempfile

    print(
        f"  slim {label}: load {src_path} → per-layer temps {layers} "
        f"(subprocess) ...",
        flush=True,
    )
    t0 = time.time()
    out_dir = tempfile.mkdtemp(prefix=f"thor_{label}_slim_")
    manifest = os.path.join(out_dir, "manifest.json")
    worker = r"""
import json, os, pickle, sys, tempfile
src, out_dir, layers_json, label = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
layers = json.loads(layers_json)
with open(src, "rb") as f:
    weights = pickle.load(f)
paths = {}
for li in layers:
    needle = f".layer.{li}."
    layer_w = {k: weights.pop(k) for k in list(weights) if needle in k}
    if not layer_w:
        raise SystemExit(f"no keys for layer {li}")
    fd, dst = tempfile.mkstemp(prefix=f"thor_{label}_L{li}_", suffix=".pkl", dir=out_dir)
    os.close(fd)
    with open(dst, "wb") as f:
        pickle.dump(layer_w, f, protocol=pickle.HIGHEST_PROTOCOL)
    paths[str(li)] = {"path": dst, "n_keys": len(layer_w), "bytes": os.path.getsize(dst)}
    del layer_w
n_left = len(weights)
weights.clear()
del weights
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"paths": paths, "n_left": n_left}, f)
"""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            src_path,
            out_dir,
            json.dumps(list(layers)),
            label,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"slim {label} subprocess failed (rc={proc.returncode}): {err[:2000]}"
        )
    with open(manifest, encoding="utf-8") as f:
        meta = json.load(f)
    out: dict[int, str] = {}
    for li_s, info in meta["paths"].items():
        li = int(li_s)
        dst = info["path"]
        # Move out of mkdtemp dir so atexit can unlink individually.
        final = os.path.join(
            tempfile.gettempdir(),
            f"thor_{label}_L{li}_{os.getpid()}_{li}.pkl",
        )
        os.replace(dst, final)
        out[li] = final
        print(
            f"  slim {label} L{li}: {info['n_keys']} keys → {final} "
            f"({info['bytes'] / (1024**3):.2f} GiB)",
            flush=True,
        )
    try:
        os.rmdir(out_dir)
    except OSError:
        pass
    _gc_cuda()
    print(
        f"  slim {label}: discarded {meta.get('n_left', '?')} leftover keys "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )
    return out


def _extract_and_take_layer_weights(weights: dict, layer_idx: int) -> dict:
    """Move ``.layer.{idx}.`` keys out of ``weights`` (caller owns the rest)."""
    needle = f".layer.{layer_idx}."
    out = {}
    for k in list(weights.keys()):
        if needle in k:
            out[k] = weights.pop(k)
    if not out:
        raise KeyError(f"no encoded keys for layer {layer_idx}")
    return out


def _bind_poly(
    thor_attention,
    thor_ff,
    *,
    dataset: str,
    layer_idx: int,
    gelu: str,
    gelu_level: int,
    ln: str,
    ln_level: int,
    softmax: str,
    softmax_level: int,
    bts_hook,
) -> None:
    if gelu == "cheb":
        from gelu_he_ct import bind_ff_gelu_cheb
        from gelu_poly import gelu_level_allowed

        ge_lv = int(gelu_level)
        if not gelu_level_allowed(layer_idx, ge_lv):
            print(
                f"  L{layer_idx} GeLU mid forbidden → high "
                f"(same fallback as optimize_bootstrap scheme)",
                flush=True,
            )
            ge_lv = 2
        gp = bind_ff_gelu_cheb(
            thor_ff,
            layer_idx=layer_idx,
            level=ge_lv,
            suppress_post_bts=True,
            bts_hook=bts_hook,
        )
        print(
            f"  L{layer_idx} GeLU → Cheb ({gp.scheme_name}, depth_he={gp.depth_he})",
            flush=True,
        )
    if ln == "poly":
        from layernorm_he_ct import bind_layer_layernorm

        p_ln1, p_ln2 = bind_layer_layernorm(
            thor_attention,
            thor_ff,
            task=dataset,
            layer_idx=layer_idx,
            level=int(ln_level),
            bts_hook=bts_hook,
        )
        print(
            f"  L{layer_idx} LN → poly "
            f"ln1(iters={p_ln1.invsqrt_max_iters}) "
            f"ln2(iters={p_ln2.invsqrt_max_iters})",
            flush=True,
        )
    if softmax == "poly":
        from softmax_he_ct import bind_attention_softmax

        sp = bind_attention_softmax(
            thor_attention,
            task=dataset,
            layer_idx=layer_idx,
            level=int(softmax_level),
            bts_hook=bts_hook,
        )
        print(
            f"  L{layer_idx} Softmax → poly "
            f"(δ2={sp.delta2:g}, shift={sp.shift:g}, "
            f"iters_σ={sp.asor_max_iters_sigma}, "
            f"iters_Σy²={sp.asor_max_iters_sum_sq})",
            flush=True,
        )


def _assert_layer_weights(weights: dict, layer_idx: int, *, kind: str) -> None:
    if kind == "att":
        keys = [
            f"bert.encoder.layer.{layer_idx}.attention.self.query.weight",
            f"bert.encoder.layer.{layer_idx}.attention.output.dense.weight",
            f"bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.weight",
        ]
    else:
        keys = [
            f"bert.encoder.layer.{layer_idx}.intermediate.dense.weight",
            f"bert.encoder.layer.{layer_idx}.output.dense.weight",
            f"bert.encoder.layer.{layer_idx}.output.LayerNorm.weight",
        ]
    missing = [k for k in keys if weights.get(k) is None]
    if missing:
        raise RuntimeError(
            f"{kind} layer {layer_idx} weights missing/None: {missing}. "
            f"Encode with: python3 HE/thor/encode_thor_weights.py --layers {layer_idx}"
        )


def _is_datastruct(obj) -> bool:
    return type(obj).__name__ == "DataStruct" and hasattr(obj, "level_calc")


def _object_array(seq):
    """``np.asarray`` unpacks NamedTuples; fill an object array by index."""
    n = len(seq)
    out = np.empty((n,), dtype=object)
    for i, v in enumerate(seq):
        out[i] = v
    return out


def _tensor_tree_to_cpu(obj):
    """Move Liberate ``DataStruct`` / tensor trees to CPU for pickle."""
    import torch

    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.detach().cpu().contiguous()
    if _is_datastruct(obj):
        return obj._replace(data=_tensor_tree_to_cpu(obj.data))
    if isinstance(obj, tuple):
        return tuple(_tensor_tree_to_cpu(v) for v in obj)
    if isinstance(obj, list):
        return [_tensor_tree_to_cpu(v) for v in obj]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        out = np.empty(obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            out[idx] = _tensor_tree_to_cpu(obj[idx])
        return out
    return obj


def _tensor_tree_to_device(obj, device):
    """Inverse of ``_tensor_tree_to_cpu`` (same tree, tensors on ``device``)."""
    import torch

    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=False)
    if _is_datastruct(obj):
        return obj._replace(data=_tensor_tree_to_device(obj.data, device))
    if isinstance(obj, tuple):
        return tuple(_tensor_tree_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [_tensor_tree_to_device(v, device) for v in obj]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        out = np.empty(obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            out[idx] = _tensor_tree_to_device(obj[idx], device)
        return out
    return obj


def _load_chain_entry_cts(
    path: str,
    *,
    engine,
    dataset: str,
    sample: int,
    first_layer: int,
):
    """Load a ``--save-out`` payload and move CTs back to the engine device."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("dataset") != dataset:
        raise SystemExit(
            f"--load-in dataset={payload.get('dataset')!r} "
            f"!= --dataset {dataset!r}"
        )
    if int(payload.get("sample", -1)) != int(sample):
        raise SystemExit(
            f"--load-in sample={payload.get('sample')} != --sample {sample}"
        )
    next_layer = payload.get("next_layer")
    if next_layer is None:
        next_layer = int(payload["last_layer"]) + 1
    if int(next_layer) != int(first_layer):
        raise SystemExit(
            f"--load-in is L{payload.get('last_layer')} exit "
            f"(next=L{next_layer}); --layers starts at {first_layer}"
        )
    saved_nl = payload.get("num_levels")
    if saved_nl is not None and int(saved_nl) != int(engine.num_levels):
        raise RuntimeError(
            f"--load-in num_levels={saved_nl} != engine {engine.num_levels}"
        )
    device = engine.ntt.devices[0]
    x = _tensor_tree_to_device(payload["x"], device)
    thor_attention_mask = _tensor_tree_to_device(
        payload["thor_attention_mask"], device
    )
    if getattr(x, "shape", None) != (8,):
        raise RuntimeError(
            f"--load-in x must be 8 real packs, got shape={getattr(x, 'shape', None)}"
        )
    ct0 = x[0]
    lc = int(getattr(ct0, "level_calc"))
    rem = int(engine.num_levels) - lc
    saved_rem = payload.get("rem")
    if saved_rem is not None and int(saved_rem) != rem:
        raise RuntimeError(
            f"loaded rem={rem} (level_calc={lc}) != saved rem={saved_rem}"
        )
    return x, thor_attention_mask, payload, rem, lc


def _save_chain_exit_cts(
    path: str,
    *,
    x,
    thor_attention_mask,
    meta: dict,
) -> None:
    """Pickle last-layer ``ln2_out`` (next-layer 8-real entry) + mask CTs."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = {
        **meta,
        "x": _tensor_tree_to_cpu(_object_array(x)),
        "thor_attention_mask": _tensor_tree_to_cpu(
            _object_array(thor_attention_mask)
        ),
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1],
        help="consecutive layer indices to chain (default: 0 1)",
    )
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument(
        "--check",
        nargs="+",
        default=["att_context", "ln1_out", "ln2_out"],
        help="intermediates to score on the **last** layer only",
    )
    ap.add_argument("--gelu", choices=("thor", "cheb"), default="cheb")
    ap.add_argument("--gelu-level", type=int, default=2)
    ap.add_argument("--ln", choices=("thor", "poly"), default="poly")
    ap.add_argument("--ln-level", type=int, default=2)
    ap.add_argument("--softmax", choices=("thor", "poly"), default="poly")
    ap.add_argument("--softmax-level", type=int, default=2)
    ap.add_argument(
        "--bootstrap",
        choices=("plan", "plan_mock"),
        default="plan_mock",
        help="plan / plan_mock only",
    )
    ap.add_argument("--plan-discard", action="store_true")
    ap.add_argument("--encrypt-level", type=int, default=None)
    ap.add_argument(
        "--mock-land-level",
        type=int,
        default=None,
        help="mock bootstrap land level_calc (default: num_levels - budget)",
    )
    ap.add_argument(
        "--probe-levels",
        action="store_true",
        help="print level_calc/remaining at linear & fixed-bts sites",
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
        "--rest-div",
        type=float,
        default=1.0,
        help="Score n≥1 mask /rest_div (default 1 = THOR-native)",
    )
    ap.add_argument(
        "--depth-profile-in",
        type=str,
        default=None,
        help="Route B: load merged HE depth profile JSON → depth_overrides",
    )
    ap.add_argument(
        "--depth-profile-unsafe",
        action="store_true",
        help="with --depth-profile-in, also apply fused LN/Softmax probe lumps",
    )
    ap.add_argument(
        "--save-out",
        type=str,
        default=None,
        help="on PASS, pickle last-layer ln2_out (next-layer entry CTs) to this path",
    )
    ap.add_argument(
        "--load-in",
        type=str,
        default=None,
        help="resume from a --save-out pkl (skip embedding encrypt; "
        "--layers must start at that file's next_layer)",
    )
    ap.add_argument(
        "--overlay-dir",
        action="append",
        default=[],
        help="dir named …/L{{n}} with ff.pkl+att.pkl; slim that layer from here "
        "instead of the merged DualRail pkl (avoids 20G+ merge OOM)",
    )
    args = ap.parse_args()

    layers = list(args.layers)
    if not layers:
        raise SystemExit("--layers must be non-empty")
    for a, b in zip(layers, layers[1:]):
        if b != a + 1:
            raise SystemExit(
                f"--layers must be consecutive (got {layers}); "
                f"gap between {a} and {b}"
            )

    os.chdir(_THOR_ROOT)
    paths = _thor_paths(args.dataset)
    overlay_by_layer: dict[int, str] = {}
    for raw in args.overlay_dir:
        d = os.path.abspath(raw)
        m = re.search(r"L(\d+)$", os.path.basename(d.rstrip(os.sep)))
        if not m:
            raise SystemExit(
                f"--overlay-dir must end with L{{n}} (got {raw!r})"
            )
        overlay_by_layer[int(m.group(1))] = d

    layer_ff_paths: dict[int, str] = {}
    layer_att_paths: dict[int, str] = {}
    for li in layers:
        if li in overlay_by_layer:
            d = overlay_by_layer[li]
            layer_ff_paths[li] = os.path.join(d, "ff.pkl")
            layer_att_paths[li] = os.path.join(d, "att.pkl")
        else:
            layer_ff_paths[li] = encoded_layer_pkl(args.dataset, "ff", li)
            layer_att_paths[li] = encoded_layer_pkl(args.dataset, "att", li)
    _require_assets(paths, layers=[li for li in layers if li not in overlay_by_layer])
    for li in layers:
        for kind, p in (
            ("ff", layer_ff_paths[li]),
            ("att", layer_att_paths[li]),
        ):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"missing {kind} L{li}: {p}. "
                    "Encode with: python3 HE/thor/encode_thor_weights.py "
                    f"--layers {li}"
                )
    print(
        f"per-layer DualRail weights {layers}: "
        + ", ".join(
            f"L{li}=ff:{os.path.basename(layer_ff_paths[li])}"
            f"/att:{os.path.basename(layer_att_paths[li])}"
            for li in layers
        ),
        flush=True,
    )

    devices = [0]
    bts_mock = args.bootstrap == "plan_mock"
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

    print("load plaintext BERT (refs + embedding; unload before HE weights) ...", flush=True)
    _repo = os.path.abspath(os.path.join(_HE, ".."))
    local_model = os.path.join(_repo, "finetuned_weight", args.dataset)
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

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
        local_model, local_files_only=True, output_hidden_states=True
    )
    model_plain.eval()
    device = torch.device("cpu")
    model_plain.to(device)

    evaluator = ThorLinearEvaluator(engine)

    from functools import partial

    from linear_eval import (  # noqa: WPS433
        bind_att_context_mask_complement,
        bind_att_score_mask_complement,
        ensure_ct_ct_matmul_masks,
        encrypt_embedding_real8,
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
    print(
        f"  DualRail linear patches: score rest_div={rest_div:g}",
        flush=True,
    )

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
    sm_lv = int(args.softmax_level) if args.softmax == "poly" else 2
    ln_lv = int(args.ln_level) if args.ln == "poly" else 2
    ge_lv = int(args.gelu_level) if args.gelu == "cheb" else 2
    scheme = []
    for li in range(NUM_LAYERS):
        g_lv = ge_lv if gelu_level_allowed(li, ge_lv) else 2
        scheme.extend([sm_lv, ln_lv, g_lv, ln_lv])
    # L0 入口：HE rem = num_levels − level_calc = initial_rem（8real 加密后再 pack）。
    plan_initial_rem = DEFAULT_L0_ENTRY_REMAINING
    thor_enc_level = remaining_to_level_calc(
        plan_initial_rem, int(engine.num_levels)
    )
    depth_overrides = None
    if args.depth_profile_in:
        from he_depth_profile import DepthProfile, build_depth_overrides

        _prof = DepthProfile.load(args.depth_profile_in)
        depth_overrides = build_depth_overrides(
            _prof,
            safe_only=not args.depth_profile_unsafe,
        )
        print(
            f"  Route B depth_overrides: {len(depth_overrides)} entries "
            f"({'unsafe' if args.depth_profile_unsafe else 'safe'}) "
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
            f"(chain={layers}; initial_rem={plan_initial_rem}; "
            f"enc level_calc={thor_enc_level}; "
            f"discard={'on' if args.plan_discard else 'off'})",
            flush=True,
        )

    rem_guard_enabled = (
        args.bootstrap in ("plan", "plan_mock") and not args.no_rem_guard
    )
    if rem_guard_enabled:
        from rem_guard import RemGuard, RemMismatchError, write_rem_trace
    else:
        RemGuard = None  # type: ignore[misc, assignment]
        RemMismatchError = RuntimeError  # type: ignore[misc, assignment]
        write_rem_trace = None  # type: ignore[misc, assignment]

    print(
        "load-in + last-layer plain refs ..."
        if args.load_in
        else "encrypt validation sample + plain refs ...",
        flush=True,
    )
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

    batch_cpu = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }
    plain_refs_all = _plain_layer_refs(
        model_plain, batch_cpu, batch_cpu["attention_mask"], device
    )
    last_layer = layers[-1]
    pref = plain_refs_all[last_layer]
    embed_ids = {
        k: v for k, v in batch.items() if k in ("input_ids", "token_type_ids")
    }
    attention_mask = batch["attention_mask"]
    del plain_refs_all, model_plain, emb_model, batch, batch_cpu
    _gc_cuda()
    print("  HF BERT unloaded after plain refs", flush=True)

    if args.load_in:
        load_path = os.path.abspath(args.load_in)
        x, saved_mask, load_meta, loaded_rem, loaded_lc = _load_chain_entry_cts(
            load_path,
            engine=engine,
            dataset=args.dataset,
            sample=int(args.sample),
            first_layer=layers[0],
        )
        print(
            f"  resume {load_path}: L{load_meta.get('last_layer')} ln2_out "
            f"→ L{layers[0]} entry rem={loaded_rem} level_calc={loaded_lc} "
            f"({os.path.getsize(load_path)} bytes)",
            flush=True,
        )
        # Older saves used np.asarray on NamedTuples and dropped PT wrappers.
        # Mask is a deterministic encode of the same sample; re-encode.
        if _is_datastruct(saved_mask[0]):
            thor_attention_mask = saved_mask
            print("  resume: using saved thor_attention_mask PTs", flush=True)
        else:
            thor_attention_mask = data_encryptor.encode_attention_mask(
                attention_mask.cpu().numpy().squeeze(), level=15
            )
            print(
                "  resume: re-encoded thor_attention_mask "
                "(saved mask was not DataStruct)",
                flush=True,
            )
        if rem_guard_enabled:
            g_entry = RemGuard.from_bootstrap_result(
                plan_result,
                layer_idx=layers[0],
                tol=int(args.rem_tol),
            )
            dp_entry = g_entry.after_event.get("__initial__")
            if dp_entry is None:
                raise RuntimeError(
                    f"DP rem_trace has no L{layers[0]} entry rem; stop"
                )
            if abs(int(dp_entry) - int(loaded_rem)) > int(args.rem_tol):
                raise RuntimeError(
                    f"loaded L{layers[0]} entry rem={loaded_rem} "
                    f"!= DP __initial__={dp_entry}. Stop; do not continue."
                )
            print(
                f"  resume rem matches DP L{layers[0]} __initial__={dp_entry}",
                flush=True,
            )
        del data_encryptor, load_meta, saved_mask, embed_ids, attention_mask
        _gc_cuda()
    else:
        embedding = data_encryptor.embed_data(embed_ids)
        thor_attention_mask = data_encryptor.encode_attention_mask(
            attention_mask.cpu().numpy().squeeze(), level=15
        )
        if args.encrypt_level is not None:
            enc_level = int(args.encrypt_level)
        elif args.bootstrap in ("plan", "plan_mock"):
            keep0 = bts_hook.remaining_keep_after(None)
            if keep0 is None:
                keep0 = plan_initial_rem
            enc_level = thor_enc_level
            print(
                f"  plan initial rem={plan_initial_rem} "
                f"encrypt level_calc={enc_level} → HE rem={plan_initial_rem}",
                flush=True,
            )
        else:
            enc_level = enc_level_from_plan_initial_rem(
                DEFAULT_L0_ENTRY_REMAINING, int(engine.num_levels)
            )

        enc_arr = np.asarray(embedding, dtype=np.float64)
        if enc_arr.ndim == 3:
            enc_arr = enc_arr.squeeze(0)
        print(
            f"  encrypt 8-real from embedding {tuple(enc_arr.shape)} "
            f"level_calc={enc_level}",
            flush=True,
        )
        x = encrypt_embedding_real8(engine, enc_arr, pk, level=enc_level)
        del data_encryptor, embedding, enc_arr, embed_ids, attention_mask
        _gc_cuda()
        print("  plaintext embedding / encryptor dropped", flush=True)

    from level_probe import LevelProbe

    level_probe = None
    if args.probe_levels:
        level_probe = LevelProbe(num_levels=int(engine.num_levels))

    he_vars_last = None
    t_chain = time.time()
    rem_guard_fail = False
    for li in layers:
        # Per-layer: load slim pkl → build → forward → release.
        print(f"  L{li} load slim weight pickles ...", flush=True)
        with open(layer_att_paths[li], "rb") as f:
            att_w = pickle.load(f)
        with open(layer_ff_paths[li], "rb") as f:
            ff_w = pickle.load(f)
        _assert_layer_weights(att_w, li, kind="att")
        _assert_layer_weights(ff_w, li, kind="ff")
        print(
            f"  L{li} weights loaded (att={len(att_w)} ff={len(ff_w)} keys)",
            flush=True,
        )
        att = ThorBertAttention(evaluator, att_w, li)
        ff = ThorBertFF(evaluator, ff_w, li)
        att_w.clear()
        ff_w.clear()
        del att_w, ff_w
        _gc_cuda()

        bind_att_score_mask_complement(att, rest_div=rest_div)
        bind_att_context_mask_complement(att)
        if abs(rest_div - 1.0) > 1e-12:
            inv = 1.0 / rest_div
            if li == 2:
                att.softmax = partial(
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
                att.softmax = partial(
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
        _bind_poly(
            att,
            ff,
            dataset=args.dataset,
            layer_idx=li,
            gelu=args.gelu,
            gelu_level=args.gelu_level,
            ln=args.ln,
            ln_level=args.ln_level,
            softmax=args.softmax,
            softmax_level=args.softmax_level,
            bts_hook=bts_hook,
        )

        rem_guard = None
        if rem_guard_enabled:
            rem_guard = RemGuard.from_bootstrap_result(
                plan_result,
                layer_idx=li,
                tol=int(args.rem_tol),
            )
            bts_hook.rem_guard = rem_guard
            trace_path = f"/tmp/l{li}_rem_trace.tsv"
            write_rem_trace(plan_result, trace_path, layer_idx=li)
            print(rem_guard.dump_layer_trace(), flush=True)
            print(f"  rem_trace written → {trace_path}", flush=True)
        t1 = time.time()
        try:
            # _forward_layer: attention.to(GPU) first; FF.to after LN1.
            x, he_vars = _forward_layer(
                engine=engine,
                evaluator=evaluator,
                thor_attention=att,
                thor_ff=ff,
                layer_idx=li,
                x=x,
                thor_attention_mask=thor_attention_mask,
                devices=devices,
                bts_hook=bts_hook,
                level_probe=level_probe,
                rem_guard=rem_guard,
            )
        except RemMismatchError as exc:
            print(
                f"  REM GUARD STOP L{li} ({time.time() - t1:.1f}s): {exc}",
                flush=True,
            )
            if rem_guard is not None:
                print(rem_guard.dump_layer_trace(), flush=True)
            rem_guard_fail = True
            _release_he_module_weights(att)
            _release_he_module_weights(ff)
            del att, ff
            _gc_cuda()
            break
        print(f"  HE L{li} done ({time.time() - t1:.1f}s)", flush=True)
        if rem_guard is not None:
            print(
                f"  rem_guard L{li}: {len(rem_guard.checked)} checks passed",
                flush=True,
            )
        if li == last_layer:
            he_vars_last = he_vars
        else:
            del he_vars
        _release_he_module_weights(att)
        _release_he_module_weights(ff)
        del att, ff
        _gc_cuda()
        print(f"  L{li} modules released", flush=True)

    _gc_cuda()
    print(f"  HE chain {layers} total ({time.time() - t_chain:.1f}s)", flush=True)

    if rem_guard_fail:
        print("FAIL (rem_guard)", flush=True)
        return 3

    mock_info = getattr(engine, "_mock_bootstrap_info", None)
    if mock_info is not None:
        print(
            f"  mock bootstrap calls={mock_info['calls']['n']} "
            f"land_level={mock_info['land_level']}",
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

    name_map = {
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
    if he_vars_last is None:
        print("FAIL (no HE output)", flush=True)
        return 2
    ok = True
    print(f"score last layer L{last_layer} vs plain:", flush=True)
    for name in args.check:
        he = he_vars_last[name]
        plain_arr = pref[name_map.get(name, name)]
        err, corr, gain_ls = _he_vs_plain_diag(
            engine,
            sk,
            he,
            plain_arr,
            name=name,
            task=args.dataset,
            layer_idx=last_layer,
            rest_div=float(args.rest_div),
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
    if ok and args.save_out:
        ct0 = x[0]
        lc = int(getattr(ct0, "level_calc"))
        rem = int(engine.num_levels) - lc
        out_path = os.path.abspath(args.save_out)
        _save_chain_exit_cts(
            out_path,
            x=x,
            thor_attention_mask=thor_attention_mask,
            meta={
                "dataset": args.dataset,
                "sample": int(args.sample),
                "layers": list(layers),
                "last_layer": int(last_layer),
                "next_layer": int(last_layer) + 1,
                "bootstrap": args.bootstrap,
                "softmax_level": int(args.softmax_level),
                "ln_level": int(args.ln_level),
                "gelu_level": int(args.gelu_level),
                "num_levels": int(engine.num_levels),
                "level_calc": lc,
                "rem": rem,
                "note": (
                    f"L{last_layer} ln2_out = L{int(last_layer)+1} entry "
                    "(8 real DualRail packs)"
                ),
            },
        )
        print(
            f"saved L{last_layer} ln2_out (L{int(last_layer)+1} entry) "
            f"rem={rem} level_calc={lc} → {out_path} "
            f"({os.path.getsize(out_path)} bytes)",
            flush=True,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
