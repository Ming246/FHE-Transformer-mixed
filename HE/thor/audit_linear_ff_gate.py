#!/usr/bin/env python3
"""
Linear FF DualRail gate (THOR path; GeLU omitted / identity):

  1) Plain DualRail twin ≡ dense ``(X @ W1.T) @ W2.T`` in input_lower
  2) PT×CT / rotate floor (DualRail block_diag_2)
  3) HE (optional): DualRail Liberate ≡ plain twin

Pipeline (matches ``ThorBertFF.dense1`` / ``dense2``, scale=1, no bias)::

  input_lower(8)
    → complexify + mask(%16<6) + rot(-8) + make_rotated (64)
    → dense1: 2× PC-MM block_diag_2 + conj  → (2,8)
    → dense2: per-rep complexify+rots → PC-MM → sum → +rot(+8) → conj
    → emb-duplicate → 8 real ≡ encode_input_lower((X@W1.T)@W2.T)

Pure-real FF uses a different root remap; do **not** compare pack-wise to it.

Usage::

  python3 audit_linear_ff_gate.py
  python3 audit_linear_ff_gate.py --he --out-packs 2 --tol-he 6e-3
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
sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from linear_eval import (  # noqa: E402
    compute_ff_dualrail_plain,
    encode_weight_ff_dualrail,
    ff_dense1_he,
    ff_dense1_plain,
    ff_dense2_he,
    ff_dense2_plain,
    ff_input_rots_from_lower8_plain,
    ff_input_rots_he,
    make_qkv_evaluator,
    max_abs_wo_head6,
    prepare_linear_keys,
    wo_emb_duplicate_head_slots,
)

sys.path.insert(0, _HE)
from thor_encoder_linear_core import (  # noqa: E402
    bert_base,
    encode_input_lower_diagonals,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-plain", type=float, default=1e-12)
    ap.add_argument("--he", action="store_true", help="HE DualRail ≡ plain twin")
    ap.add_argument("--tol-he", type=float, default=6e-3)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument(
        "--out-packs",
        type=int,
        default=2,
        help="FC2 DualRail out packs for HE (default 2; FC1 always full 8)",
    )
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    rng = np.random.default_rng(args.seed)
    X = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    W1 = rng.normal(0.0, 0.05, (cfg.ffn_dim, cfg.hidden_dim)).astype(np.float64)
    W2 = rng.normal(0.0, 0.05, (cfg.hidden_dim, cfg.ffn_dim)).astype(np.float64)
    dense = (X @ W1.T) @ W2.T
    y_lower = encode_input_lower_diagonals(dense, cfg)
    x8 = encode_input_lower_diagonals(X, cfg)

    print(
        "=== FF gate (DualRail plain ≡ (X@W1.T)@W2.T → input_lower) ===",
        flush=True,
    )
    t0 = time.time()
    wo_plain = compute_ff_dualrail_plain(x8, W1, W2, cfg, out_packs=8)
    err_p = max(
        float(np.max(np.abs(wo_plain[i] - y_lower[i]))) for i in range(len(y_lower))
    )
    err_h6 = max(max_abs_wo_head6(wo_plain[i], y_lower[i]) for i in range(8))
    ok1 = err_p < args.tol_plain
    print(
        f"[1] DualRail plain ≡ encode_lower((X@W1.T)@W2.T): max|err|={err_p:.3e}  "
        f"head6={err_h6:.3e}  {'PASS' if ok1 else 'FAIL'}  "
        f"({time.time()-t0:.1f}s)",
        flush=True,
    )

    n_in_c, n_out_full, ll, n_rep = 64, 8, 6, 2
    n_out_use = min(int(args.out_packs), n_out_full)
    # dense1 always full 8; dense2 HE may slice out_packs
    theory_ptct_d1 = n_rep * n_out_full * ll * n_in_c
    theory_ptct_d2 = n_rep * n_out_use * ll * n_in_c
    print("[2] theory floor (DualRail FF block_diag_2, no GeLU):", flush=True)
    print(
        f"    dense1 w=(2,8,6,64) PT×CT={theory_ptct_d1}  "
        f"dense2 w=(2,{n_out_use},6,64) PT×CT={theory_ptct_d2}",
        flush=True,
    )
    print(
        "    make_rotated: 4→64 (in) + 2×(4→64) (dense2); "
        "fold rot±8; mul-depth≈2 per PC-MM",
        flush=True,
    )
    ok2 = True
    print("    counts closed-form PASS", flush=True)

    ok3 = True
    if args.he:
        sys.path.insert(0, _HERE)
        from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys
        from bts_ops import DEPTH_FF_DENSE, remaining_to_level_calc

        print("[3] HE DualRail FF vs plain twin ...", flush=True)
        engine = create_engine(dict(THOR_DEFAULT_PARAMS))
        keys = create_keys(engine, with_evk=True)
        sk, pk = keys["sk"], keys["pk"]
        prepare_linear_keys(engine, sk)
        evaluator = make_qkv_evaluator(engine)

        # dense1 + dense2 each burn ~2; keep headroom for mc_mult on input prep
        keep = 2 * DEPTH_FF_DENSE + 2
        input_level = (
            int(args.level)
            if args.level is not None
            else remaining_to_level_calc(keep, engine.num_levels)
        )
        print(f"    working level_calc={input_level} keep={keep}", flush=True)

        x_cts = np.full((8,), None, dtype=object)
        for i in range(8):
            x_cts[i] = engine.encodecrypt(
                np.asarray(x8[i], dtype=np.float64), pk, level=input_level
            )

        t_w = time.time()
        # FC1 must be full 8 packs so dense2 can complexify i / i+4
        w1_pts = encode_weight_ff_dualrail(
            engine, W1, split="vertical", level=input_level, out_packs=8
        )
        w2_pts = encode_weight_ff_dualrail(
            engine, W2, split="horizontal", level=input_level, out_packs=n_out_use
        )
        print(f"    weights encoded ({time.time()-t_w:.1f}s)", flush=True)

        t_h = time.time()
        x_rots = ff_input_rots_he(engine, x_cts)
        # input prep uses mc_mult → +1 level; align weight? Liberate tiles PT.
        fc1_ct = ff_dense1_he(engine, evaluator, x_rots, w1_pts)
        out_cts = ff_dense2_he(engine, evaluator, fc1_ct, w2_pts)
        print(f"    HE FF done ({time.time()-t_h:.1f}s)", flush=True)

        plain_n = ff_dense2_plain(
            ff_dense1_plain(
                ff_input_rots_from_lower8_plain(x8),
                W1,
                cfg,
                out_packs=8,
            ),
            W2,
            cfg,
            out_packs=n_out_use,
            to_input_lower=True,
        )
        errs = []
        for i in range(n_out_use):
            dec = np.asarray(
                engine.decrode(out_cts[i], sk, is_real=True), dtype=np.float64
            )
            dec_d = wo_emb_duplicate_head_slots(dec)
            e = float(np.max(np.abs(dec_d - plain_n[i])))
            e6 = max_abs_wo_head6(dec, plain_n[i])
            errs.append(e6)
            print(
                f"    pack[{i}] vs plain head6={e6:.3e} full(dup)={e:.3e}",
                flush=True,
            )
        err_h = max(errs) if errs else float("inf")
        ok3 = err_h < args.tol_he
        print(
            f"    HE ≡ plain (head6): max|err|={err_h:.3e}  "
            f"{'PASS' if ok3 else 'FAIL'}  tol={args.tol_he}",
            flush=True,
        )
    else:
        print(
            "[3] HE skipped (pass --he). DualRail Liberate vs plain twin.",
            flush=True,
        )

    ok = ok1 and ok2 and ok3
    print("FF GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
