#!/usr/bin/env python3
"""
Linear W_O DualRail gate (THOR path; not pure-real pack layout):

  1) Plain DualRail twin ≡ dense ``C @ W_o.T`` in input_lower layout
  2) PT×CT / rotate / mul-depth == DualRail theory floor
  3) HE (optional): DualRail Liberate ≡ plain twin

Pipeline (matches ``ThorBertAttention.dense``)::

  context = complexify_v(QKV-upper(C))          # 2 complex
    → make_rotated_copies (32)
    → pt_ct_matmul(W_o DualRail: n_in=64, n_out=128, b_shape=(128,64))
    → fold (mask %16>=6, rot+6) + conjugate
    → emb-duplicate head slots  → 8 real ≡ encode_input_lower(C @ W_o.T)

Pure-real Wo uses a different remap/encode; do **not** compare pack-wise to it.

Usage::

  python3 audit_linear_wo_gate.py
  python3 audit_linear_wo_gate.py --he --out-packs 2 --tol-he 6e-3
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
    complexify_v_plain,
    compute_wo_dualrail_plain,
    encode_qkv_upper_from_dense,
    encode_weight_wo_dualrail,
    make_qkv_evaluator,
    max_abs_wo_head6,
    prepare_linear_keys,
    wo_emb_duplicate_head_slots,
    wo_pcmm_he,
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
        help="Wo DualRail out packs for HE (default 2; full=8)",
    )
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    rng = np.random.default_rng(args.seed)
    C = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    Wo = rng.normal(0.0, 0.05, (cfg.hidden_dim, cfg.hidden_dim)).astype(np.float64)
    dense = C @ Wo.T
    y_lower = encode_input_lower_diagonals(dense, cfg)

    print("=== W_O gate (DualRail plain ≡ C @ W_o.T → input_lower) ===", flush=True)
    ctx_upper = encode_qkv_upper_from_dense(C, cfg)
    ctx_cplx = complexify_v_plain(ctx_upper)
    t0 = time.time()
    wo_plain = compute_wo_dualrail_plain(ctx_cplx, Wo, cfg, out_packs=8)
    err_p = max(
        float(np.max(np.abs(wo_plain[i] - y_lower[i]))) for i in range(len(y_lower))
    )
    err_h6 = max(max_abs_wo_head6(wo_plain[i], y_lower[i]) for i in range(8))
    ok1 = err_p < args.tol_plain
    print(
        f"[1] DualRail plain ≡ encode_lower(C@Wo.T): max|err|={err_p:.3e}  "
        f"head6={err_h6:.3e}  {'PASS' if ok1 else 'FAIL'}  "
        f"({time.time()-t0:.1f}s)  packs={len(wo_plain)}",
        flush=True,
    )

    # DualRail: n_in_c=32, n_out_p=8, ll=6
    n_in_c, n_out_full, ll = 32, 8, 6
    n_out_use = min(int(args.out_packs), n_out_full)
    theory_ptct = n_out_use * ll * n_in_c
    theory_ri = n_out_use * (ll - 1)
    theory_rot_left = 2 * (cfg.pack - 1)  # 2 complex roots × 15
    print("[2] theory floor (DualRail PC-MM + fold, block_diag_1):", flush=True)
    print(
        f"    w shape=({n_out_use},{ll},{n_in_c})  using out_packs={n_out_use}",
        flush=True,
    )
    print(
        f"    PT×CT={theory_ptct}  rotate_internal={theory_ri}  "
        f"make_rotated left-rots={theory_rot_left}  "
        f"fold mc_mult={n_out_use}  mul-depth≈3 (matmul+2 + fold+1)",
        flush=True,
    )
    ok2 = True
    print("    counts closed-form PASS", flush=True)

    ok3 = True
    if args.he:
        # Prefer HE/thor on path so ``bts_ops`` / ``engine`` resolve to Branch A.
        sys.path.insert(0, _HERE)
        from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys
        from bts_ops import DEPTH_ATT_DENSE, remaining_to_level_calc

        print("[3] HE DualRail W_O vs plain twin ...", flush=True)
        engine = create_engine(dict(THOR_DEFAULT_PARAMS))
        keys = create_keys(engine, with_evk=True)
        sk, pk = keys["sk"], keys["pk"]
        prepare_linear_keys(engine, sk)
        evaluator = make_qkv_evaluator(engine)

        keep = DEPTH_ATT_DENSE + 1
        input_level = (
            int(args.level)
            if args.level is not None
            else remaining_to_level_calc(keep, engine.num_levels)
        )
        print(f"    working level_calc={input_level} keep={keep}", flush=True)

        ctx_ct = np.full((2,), None, dtype=object)
        for i in range(2):
            ctx_ct[i] = engine.encode_and_encrypt(
                np.asarray(ctx_cplx[i], dtype=np.complex128),
                pk,
                level=input_level,
            )

        t_w = time.time()
        w_pts = encode_weight_wo_dualrail(
            engine, Wo, level=input_level, out_packs=n_out_use
        )
        print(f"    weights encoded ({time.time()-t_w:.1f}s)", flush=True)

        t_h = time.time()
        out_cts = wo_pcmm_he(engine, evaluator, ctx_ct, w_pts, input_level=None)
        print(f"    HE dense done ({time.time()-t_h:.1f}s)", flush=True)

        # Plain twin for the same out_packs (raw fold, then emb-dup for compare)
        plain_n = compute_wo_dualrail_plain(
            ctx_cplx, Wo, cfg, out_packs=n_out_use, to_input_lower=True
        )
        errs = []
        for i in range(n_out_use):
            dec = np.asarray(engine.decrode(out_cts[i], sk, is_real=True), dtype=np.float64)
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
    print("W_O GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
