#!/usr/bin/env python3
"""
Single-projection DualRail PC-MM smoke (QKV-style) on Liberate.

Pipeline (no mult_scalar):
  DualRail encrypt X **at working level_calc** → make_rotated_copies → pt_ct_matmul
  → cc_add(·, conjugate(·))  # real extract; weight /2 already baked
  (Do not enc@L0 then long level_up — Liberate destroys precision.)

Compare decrypted real part to mainline pure-real PC-MM
(``thor_encoder_linear_core._qkv_pc_from_enc``).

Default ``--out-packs 1`` encodes 384 weight PTs (~several minutes).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import ensure_thor_path

ensure_thor_path()

from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from bts_ops import DEPTH_QKV, remaining_to_level_calc  # noqa: E402
from linear_eval import (  # noqa: E402
    encode_weight_qkv_dualrail,
    encrypt_embedding_dualrail,
    mainline_qkv_ref,
    make_qkv_evaluator,
    prepare_linear_keys,
    qkv_pcmm_he,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-packs", type=int, default=1, choices=(1, 2, 3, 4))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=6e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="override Liberate level_calc; default from bts_ops keep=DEPTH_QKV",
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-0.3, 0.3, (128, 768)).astype(np.float64)
    w = rng.uniform(-0.05, 0.05, (768, 768)).astype(np.float64)

    print("mainline pure-real PC-MM reference ...", flush=True)
    t_ref = time.time()
    ref = mainline_qkv_ref(x, w, scale=args.scale, out_packs=args.out_packs)
    print(f"  ref ready ({time.time()-t_ref:.1f}s) packs={len(ref)}", flush=True)

    t0 = time.time()
    print(f"creating engine {THOR_DEFAULT_PARAMS} ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    print(f"engine ready {time.time()-t0:.1f}s slots={engine.num_slots}", flush=True)

    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_linear_keys(engine, sk)

    # Working entry: keep depth for this chunk + 1 headroom (pt_ct_matmul
    # does level_up to lev+2; Liberate cannot land on num_levels).
    keep = DEPTH_QKV + 1
    input_level = (
        args.level
        if args.level is not None
        else remaining_to_level_calc(keep, engine.num_levels)
    )
    print(
        f"working level_calc={input_level} (keep_remaining={keep}, "
        f"num_levels={engine.num_levels})",
        flush=True,
    )

    print("evaluator (block_diag_1 masks only) ...", flush=True)
    evaluator = make_qkv_evaluator(engine)

    print("encrypt DualRail embedding at working level ...", flush=True)
    t1 = time.time()
    x_ct = encrypt_embedding_dualrail(engine, x, pk, level=input_level)
    print(f"  encrypted ({time.time()-t1:.1f}s) level={x_ct[0].level_calc}", flush=True)

    print(
        f"encode DualRail Wqkv out_packs={args.out_packs} scale={args.scale} "
        f"at level={input_level} ...",
        flush=True,
    )
    t2 = time.time()
    w_pts = encode_weight_qkv_dualrail(
        engine,
        w,
        level=input_level,
        scale=args.scale,
        out_packs=args.out_packs,
    )
    print(
        f"  weights shape={w_pts.shape} ({time.time()-t2:.1f}s)",
        flush=True,
    )

    print("HE PC-MM (rotate + pt_ct_matmul + conj real extract) ...", flush=True)
    t3 = time.time()
    q_ct = qkv_pcmm_he(engine, evaluator, x_ct, w_pts)
    print(
        f"  done ({time.time()-t3:.1f}s) out_level={q_ct[0].level_calc}",
        flush=True,
    )

    errs = []
    for i in range(args.out_packs):
        dec = np.asarray(engine.decrode(q_ct[i], sk, is_real=True), dtype=np.float64)
        err = float(np.max(np.abs(dec - ref[i])))
        mean = float(np.mean(np.abs(dec - ref[i])))
        errs.append(err)
        print(f"  pack[{i}] max|err|={err:.3e} mean={mean:.3e}", flush=True)

    ok = all(e < args.tol for e in errs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
