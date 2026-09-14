#!/usr/bin/env python3
"""
途径 A 明文 smoke：DualRail score 岛 ≡ 纯实 geometric / dense K@Q.T。

不启 Liberate。流水线：

  Q/K upper packs → transpose+complexify(K) + DualRail make_copies(Q)
  → complex ttemp BSGS → unpack/gain → 8 纯实 score packs
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
    compute_attention_score_dualrail_island,
    decode_score_packs,
    dense_qk_scores,
    encode_qkv_upper_from_dense,
)

# dualrail_encode / ensure_thor_path may drop HE; re-append before core imports
if _HE not in sys.path:
    sys.path.append(_HE)

from thor_encoder_linear_core import (  # noqa: E402
    _build_cc_score_scatter_pairs,
    _build_ct_ct_masks,
    _slot_cc_score_scatter,
    bert_base,
    make_copies_real,
    transpose_upper_to_lower,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="vs dense/geometric; plaintext DualRail should be ~machine eps",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=8.0)
    ap.add_argument("--scale", type=float, default=None)
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    scale = (
        float(args.scale)
        if args.scale is not None
        else 1.0 / np.sqrt(float(cfg.head_dim))
    )
    print(
        f"DualRail island plain score gain={args.gain} scale={scale:.6f} "
        f"tol={args.tol}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    q_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    k_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    q_packs = encode_qkv_upper_from_dense(q_dense, cfg)
    k_packs = encode_qkv_upper_from_dense(k_dense, cfg)
    scores_ref = dense_qk_scores(q_dense, k_dense, cfg, scale=scale)

    print("geometric ref (score scatter only) ...", flush=True)
    t0 = time.time()
    scatter = _build_cc_score_scatter_pairs(cfg)
    ct_ct = _build_ct_ct_masks(cfg, cfg.head_dim)
    k_lower = transpose_upper_to_lower(k_packs, cfg)
    q_copies_real = make_copies_real(q_packs, cfg)
    geo = _slot_cc_score_scatter(k_lower, q_copies_real, cfg, scatter, scale)
    geo_dec = decode_score_packs(geo, cfg)
    geo_err = float(np.max(np.abs(geo_dec - scores_ref)))
    print(f"  geometric vs dense max|err|={geo_err:.3e} ({time.time()-t0:.1f}s)", flush=True)
    if geo_err > 1e-8:
        print("FAIL: geometric ≠ dense", flush=True)
        return 2

    print("DualRail island ...", flush=True)
    t1 = time.time()
    island = compute_attention_score_dualrail_island(
        q_packs, k_packs, cfg, scale=scale, gain=args.gain, ct_ct=ct_ct
    )
    print(f"  done {time.time()-t1:.1f}s", flush=True)

    isl_dec = decode_score_packs(island, cfg)
    err_dense = float(np.max(np.abs(isl_dec - scores_ref)))
    err_geo = max(
        float(np.max(np.abs(a - b))) for a, b in zip(island, geo)
    )
    # argmax detail vs dense
    d = np.abs(isl_dec - scores_ref)
    idx = np.unravel_index(int(np.argmax(d)), d.shape)
    print(
        f"island vs dense max|err|={err_dense:.3e} "
        f"ref@argmax={scores_ref[idx]:.6e} he={isl_dec[idx]:.6e} idx={idx}",
        flush=True,
    )
    print(f"island vs geometric packs max|err|={err_geo:.3e}", flush=True)

    ok = err_dense <= args.tol and err_geo <= args.tol
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
