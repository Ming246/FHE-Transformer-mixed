#!/usr/bin/env python3
"""
Nonlinear gates (plain HE-isomorphic ≈ reference; approx error, not machine eps):

  Softmax — slot_softmax_he vs dense row-softmax (dense-scale logits)
  LayerNorm — slot_layernorm_he vs NumPy LN
  GeLU — slot_gelu_he vs tanh-gelu

Depth from poly configs. HE CT: liberate smoke_*_reactive.py

Usage::

  python3 audit_nonlinear_gate.py
  python3 audit_nonlinear_gate.py --ops gelu --tol 5e-2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))


def _ensure_paths() -> None:
    # HE first (slot_*_he); thor second; repo for gelu_poly / softmax_poly.
    for p in (_HE, _HERE, _REPO):
        if p in sys.path:
            sys.path.remove(p)
    sys.path[:0] = [_HE, _HERE, _REPO]


_ensure_paths()

from thor_encoder_linear_core import bert_base, encode_input_lower_diagonals  # noqa: E402


def _softmax_ref(scores: np.ndarray) -> np.ndarray:
    x = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def gate_softmax(cfg, rng, tol: float) -> bool:
    _ensure_paths()
    from linear_eval import decode_score_packs, dense_qk_scores, encode_score_packs_from_dense

    _ensure_paths()  # linear_eval → ensure_thor_path drops HE/
    from slot_softmax_he import (
        SoftmaxHeParams,
        encode_key_valid_mask_packs,
        slot_softmax_he,
    )

    print("=== Softmax (slot_softmax_he ≈ dense softmax) ===", flush=True)
    q = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.clip(dense_qk_scores(q, k, cfg, scale=scale), -8.0, 8.0)
    ref = _softmax_ref(scores)
    packs = encode_score_packs_from_dense(scores, cfg)
    key_valid = np.ones(cfg.seq_len, dtype=np.float64)
    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)
    params = SoftmaxHeParams.from_softmax_poly("mrpc", 0, level=2)
    t0 = time.time()
    out = slot_softmax_he(
        packs, mask_packs, cfg, params, exp_input_prescaled=False
    )
    he = decode_score_packs(out, cfg)
    err = float(np.max(np.abs(he - ref)))
    ok = err < tol
    try:
        from cost import softmax_poly_depth

        depth = softmax_poly_depth("mrpc", 0, 2)
    except Exception:
        depth = "?"
    print(
        f"[1] plain HE-iso ≡ dense softmax: max|err|={err:.3e}  "
        f"{'PASS' if ok else 'FAIL'}  tol={tol}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    print(f"[2] theory depth (cost.softmax_poly_depth)={depth}", flush=True)
    print("[3] HE CT: liberate/smoke_softmax_reactive.py", flush=True)
    return ok


def gate_layernorm(cfg, rng, tol: float) -> bool:
    _ensure_paths()
    from slot_layernorm_he import (
        LayerNormHeParams,
        encode_affine_to_input_lower,
        slot_layernorm_he,
    )

    print("=== LayerNorm (slot_layernorm_he ≈ NumPy LN) ===", flush=True)
    x = rng.normal(0.0, 0.5, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    gamma = np.ones(cfg.hidden_dim, dtype=np.float64)
    beta = np.zeros(cfg.hidden_dim, dtype=np.float64)
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    ref = (x - mu) / np.sqrt(var + 1e-12) * gamma + beta
    x_packs = encode_input_lower_diagonals(x, cfg)
    g_packs, b_packs = encode_affine_to_input_lower(gamma, beta, cfg)
    ref_packs = encode_input_lower_diagonals(ref, cfg)
    params = LayerNormHeParams.from_layernorm_poly("mrpc", 0, "ln1", level=2)
    t0 = time.time()
    out = slot_layernorm_he(x_packs, g_packs, b_packs, cfg, params)
    err = max(float(np.max(np.abs(out[i] - ref_packs[i]))) for i in range(len(out)))
    ok = err < tol
    print(
        f"[1] plain HE-iso ≡ LN: max|err|={err:.3e}  "
        f"{'PASS' if ok else 'FAIL'}  tol={tol}  "
        f"iters={params.invsqrt_max_iters}  ({time.time()-t0:.1f}s)",
        flush=True,
    )
    print(
        f"[2] theory depth ≈ invsqrt_iters+3 = {params.invsqrt_max_iters + 3}",
        flush=True,
    )
    print("[3] HE CT: liberate/smoke_layernorm_reactive.py", flush=True)
    return ok


def gate_gelu(cfg, rng, tol: float) -> bool:
    _ensure_paths()
    from slot_gelu_he import GeluHeParams, slot_gelu_he

    print("=== GeLU (slot_gelu_he ≈ tanh-gelu) ===", flush=True)
    x = rng.normal(0.0, 0.3, (cfg.num_slots,)).astype(np.float64)
    ref = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    params = GeluHeParams.from_gelu_poly(0, level=2)
    x_slot = x / float(params.C)  # FC1 output scale (``x_phys/C``)
    t0 = time.time()
    out = slot_gelu_he([x_slot], params)
    err = float(np.max(np.abs(np.asarray(out[0]) - ref)))
    ok = err < tol
    print(
        f"[1] plain HE-iso ≡ gelu: max|err|={err:.3e}  "
        f"{'PASS' if ok else 'FAIL'}  tol={tol}  depth_he={params.depth_he}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )
    print(f"[2] theory depth_he={params.depth_he} scheme={params.scheme_name}", flush=True)
    print("[3] HE CT: liberate/smoke_gelu_reactive.py", flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument(
        "--ops",
        type=str,
        default="softmax,layernorm,gelu",
        help="comma list: softmax,layernorm,gelu",
    )
    args = ap.parse_args()
    cfg = bert_base()
    cfg.validate()
    rng = np.random.default_rng(args.seed)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    results: dict[str, bool] = {}
    dispatch = {
        "softmax": gate_softmax,
        "layernorm": gate_layernorm,
        "gelu": gate_gelu,
    }
    for name in ops:
        if name not in dispatch:
            print(f"unknown op {name}", flush=True)
            results[name] = False
            continue
        try:
            results[name] = dispatch[name](cfg, rng, args.tol)
        except Exception as e:
            print(f"{name.upper()} GATE ERROR: {type(e).__name__}: {e}", flush=True)
            results[name] = False

    ok = all(results.values()) if results else False
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    print("NONLINEAR GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
