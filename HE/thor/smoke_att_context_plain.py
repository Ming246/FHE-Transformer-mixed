#!/usr/bin/env python3
"""
途径 A 明文 smoke：DualRail context 岛 ≡ dense α@V / geometric（~machine eps）。

不启 Liberate。流水线：

  alpha(8) + V(4) → make_copies(α)(128) + V complexify(2)
  → complex ttemp BSGS → /gain(=1) → 4 纯实 V-layout packs
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
if _HE not in sys.path:
    sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()
if _HE not in sys.path:
    sys.path.append(_HE)

from linear_eval import (  # noqa: E402
    compute_attention_context_dualrail_island,
    decode_qkv_upper_to_dense,
    dense_alpha_v_context,
    encode_qkv_upper_from_dense,
    encode_score_packs_from_dense,
)

if _HE not in sys.path:
    sys.path.append(_HE)

from thor_encoder_linear_core import (  # noqa: E402
    _build_ct_ct_masks,
    _slot_cc_context_scatter,
    bert_base,
    make_copies,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument(
        "--copy-mode",
        choices=("make_copies", "final", "identity"),
        default="make_copies",
        help="make_copies: ≡ geometric/dense; final/identity: Softmax DualRail I/O twin",
    )
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    print(
        f"DualRail island plain context gain={args.gain} "
        f"copy={args.copy_mode} tol={args.tol}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    v_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    logits = rng.normal(0.0, 0.5, (cfg.num_heads, cfg.seq_len, cfg.seq_len))
    m = logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits - m)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    v_packs = encode_qkv_upper_from_dense(v_dense, cfg)
    a_packs = encode_score_packs_from_dense(alpha, cfg)
    ctx_ref = dense_alpha_v_context(alpha, v_dense, cfg)

    print("geometric context ref ...", flush=True)
    t0 = time.time()
    # avoid full MaskBank: only context scatter
    from thor_encoder_linear_core import _build_cc_context_scatter_pairs

    scatter = _build_cc_context_scatter_pairs(cfg)
    a_copies_real = make_copies(a_packs, cfg)
    geo = _slot_cc_context_scatter(v_packs, a_copies_real, cfg, scatter, 1.0)
    geo_dec = decode_qkv_upper_to_dense(geo, cfg)
    geo_err = float(np.max(np.abs(geo_dec - ctx_ref)))
    print(f"  geometric vs dense max|err|={geo_err:.3e} ({time.time()-t0:.1f}s)", flush=True)
    if geo_err > 1e-8:
        print("FAIL: geometric ≠ dense", flush=True)
        return 2

    print("DualRail context island ...", flush=True)
    t1 = time.time()
    ct_ct = _build_ct_ct_masks(cfg, cfg.n_in_slot)
    island = compute_attention_context_dualrail_island(
        a_packs,
        v_packs,
        cfg,
        gain=args.gain,
        ct_ct=ct_ct,
        alpha_copy_mode=args.copy_mode,
    )
    print(f"  done {time.time()-t1:.1f}s", flush=True)

    isl_dec = decode_qkv_upper_to_dense(island, cfg)
    err_dense = float(np.max(np.abs(isl_dec - ctx_ref)))
    err_geo = max(float(np.max(np.abs(a - b))) for a, b in zip(island, geo))
    d = np.abs(isl_dec - ctx_ref)
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    corr = float(np.corrcoef(isl_dec.ravel(), ctx_ref.ravel())[0, 1])
    print(
        f"island vs dense max|err|={err_dense:.3e} corr={corr:.4f} "
        f"ref@argmax={ctx_ref[idx]:.6e} he={isl_dec[idx]:.6e} idx={idx}",
        flush=True,
    )
    print(f"island vs geometric packs max|err|={err_geo:.3e}", flush=True)

    ok = err_dense <= args.tol
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
