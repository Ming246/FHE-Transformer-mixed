#!/usr/bin/env python3
"""Chunk smoke: DualRail ``make_copies`` (Q) vs NumPy twin."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import ensure_thor_path

ensure_thor_path()

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    ensure_make_copies_masks,
    make_qkv_evaluator,
    numpy_make_copies_dualrail,
    prepare_make_copies_keys,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-2)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="override level_calc; default from bts_ops DEPTH_MAKE_COPIES keep",
    )
    ap.add_argument("--check", type=int, default=8, help="how many of 64 outs to check")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    msgs = [rng.uniform(-0.3, 0.3, 2**15).astype(np.float64) for _ in range(4)]
    print("numpy DualRail make_copies ref ...", flush=True)
    t0 = time.time()
    ref = numpy_make_copies_dualrail(msgs)
    print(f"  ref ready ({time.time()-t0:.1f}s) n={len(ref)}", flush=True)

    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_make_copies_keys(engine, sk)
    evaluator = make_qkv_evaluator(engine)
    ensure_make_copies_masks(evaluator, engine)
    hook = BootstrapHook(engine, mode="record")

    from bts_ops import DEPTH_MAKE_COPIES, remaining_to_level_calc

    keep = max(DEPTH_MAKE_COPIES, 2)
    work_level = (
        args.level
        if args.level is not None
        else remaining_to_level_calc(keep, engine.num_levels)
    )
    print(f"working level_calc={work_level} (keep={keep})", flush=True)

    cts = np.full((4,), None, dtype=object)
    for i in range(4):
        cts[i] = engine.encodecrypt(msgs[i], pk, level=work_level)

    print("HE make_copies ...", flush=True)
    t1 = time.time()
    copies = evaluator.make_copies(cts)  # default scale=1/2; masks have 1/4
    hook.refresh("L0.linear_make_copies", copies, indices=[])
    print(f"  done ({time.time()-t1:.1f}s) shape={copies.shape}", flush=True)

    errs = []
    for i in range(args.check):
        dec = np.asarray(engine.decrode(copies[i], sk, is_real=True), dtype=np.float64)
        err = float(np.max(np.abs(dec - ref[i])))
        errs.append(err)
        print(f"  out[{i}] max|err|={err:.3e}", flush=True)

    print("bootstrap_hook:", hook.summary())
    ok = all(e < args.tol for e in errs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
