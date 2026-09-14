#!/usr/bin/env python3
"""
Chunk smoke: ``transpose_upper_to_lower`` (K) on Liberate vs mainline NumPy.

Inputs: 4 random real slot vectors (stand-in for DualRail-extracted K).
No mult_scalar. Uses ``bts_ops`` event name for hook record (plan dry-run).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _HE not in sys.path:
    sys.path.insert(0, _HE)
from path_setup import ensure_thor_path

ensure_thor_path()
# thor_encoder_linear_core lives under HE/; keep HE on path after ensure_thor_path
if _HE not in sys.path:
    sys.path.append(_HE)

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    ensure_transpose_masks,
    make_qkv_evaluator,
    prepare_transpose_keys,
)
from thor_encoder_linear_core import (  # noqa: E402
    bert_base,
    transpose_upper_to_lower,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tol",
        type=float,
        default=2e-2,
        help="transpose does many mask×ct; noise floor ~1e-2",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--level",
        type=int,
        default=None,
        help="override level_calc; default from bts_ops DEPTH_TRANSPOSE_K keep",
    )
    args = ap.parse_args()

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    upper = [
        rng.uniform(-0.3, 0.3, cfg.num_slots).astype(np.float64) for _ in range(4)
    ]
    print("mainline transpose ref ...", flush=True)
    t0 = time.time()
    ref = transpose_upper_to_lower(upper, cfg)
    print(f"  ref ready ({time.time()-t0:.1f}s)", flush=True)

    print("creating engine ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_transpose_keys(engine, sk)

    from bts_ops import DEPTH_TRANSPOSE_K, remaining_to_level_calc

    keep = max(DEPTH_TRANSPOSE_K, 2)  # transpose uses several rescales
    work_level = (
        args.level
        if args.level is not None
        else remaining_to_level_calc(keep, engine.num_levels)
    )
    print(f"working level_calc={work_level} (keep={keep})", flush=True)

    evaluator = make_qkv_evaluator(engine)
    ensure_transpose_masks(evaluator, engine)

    hook = BootstrapHook(engine, mode="record")
    cts = np.full((4,), None, dtype=object)
    for i in range(4):
        cts[i] = engine.encodecrypt(upper[i], pk, level=work_level)

    print("HE transpose_upper_to_lower ...", flush=True)
    t1 = time.time()
    lower = evaluator.transpose_upper_to_lower(cts)
    # named site for future plan mode (L0.linear_transpose_k)
    hook.refresh("L0.linear_transpose_k", lower, indices=[])
    print(f"  done ({time.time()-t1:.1f}s) level={lower[0].level_calc}", flush=True)

    errs = []
    for i in range(4):
        dec = np.asarray(engine.decrode(lower[i], sk, is_real=True), dtype=np.float64)
        err = float(np.max(np.abs(dec - ref[i])))
        errs.append(err)
        print(f"  pack[{i}] max|err|={err:.3e}", flush=True)

    print("bootstrap_hook:", hook.summary())
    ok = all(e < args.tol for e in errs)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
