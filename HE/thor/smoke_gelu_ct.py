#!/usr/bin/env python3
"""
Smoke: Liberate Chebyshev GeLU vs plaintext ``gelu_poly`` / slot twin.

Encrypts a single CT (not full (2,8) BERT pack) at working level, runs
``gelu_poly_ct``, compares decrypt to ``gelu_poly_slots``.

Usage::

  python3 HE/thor/smoke_gelu_ct.py --layer 0 --level 2
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
sys.path.append(_HE)

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from gelu_he_ct import gelu_poly_ct  # noqa: E402
from slot_gelu_he import GeluHeParams, gelu_poly_slots  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=0, help="poly scheme level 0/1/2")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-level", type=int, default=None)
    args = ap.parse_args()

    params = GeluHeParams.from_gelu_poly(args.layer, level=args.level)
    print(
        f"GeLU params layer={args.layer} level={args.level} "
        f"kind={params.kind} C={params.C} depth_he={params.depth_he} "
        f"scheme={params.scheme_name}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    x_true = rng.uniform(-float(params.C), float(params.C), size=256).astype(np.float64)
    x_slot = x_true / float(params.C)
    print("create engine + load keys (no bootstrap) ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    n = engine.num_slots
    msg = np.zeros(n, dtype=np.float64)
    msg[: x_slot.shape[0]] = x_slot

    from bts_ops import remaining_to_level_calc  # noqa: E402

    # Unit test: encodecrypt at GeLU entry rem — no bootstrap.
    want_rem = 14
    work_level = (
        int(args.work_level)
        if args.work_level is not None
        else remaining_to_level_calc(want_rem, engine.num_levels)
    )
    print(
        f"encodecrypt @ level_calc={work_level} rem≈{want_rem} "
        f"num_levels={engine.num_levels} (no bts)",
        flush=True,
    )
    ct = engine.encodecrypt(msg, pk, level=work_level)
    print(
        f"  entry level={ct.level_calc} "
        f"remaining={engine.num_levels - ct.level_calc}",
        flush=True,
    )

    lv0 = int(ct.level_calc)
    t0 = time.time()
    ct_out = gelu_poly_ct(engine, ct, params)
    lv1 = int(ct_out.level_calc)
    used = lv1 - lv0
    print(
        f"HE Cheb GeLU done ({time.time()-t0:.1f}s) "
        f"level {lv0}→{lv1} used={used} depth_he={params.depth_he} "
        f"overhead={used - int(params.depth_he)}",
        flush=True,
    )

    he = np.real(np.asarray(engine.decrode(ct_out, sk), dtype=np.complex128))
    ref = gelu_poly_slots(msg, params)
    n_cmp = int(x_true.shape[0])
    err = float(np.max(np.abs(he[:n_cmp] - ref[:n_cmp])))
    corr = float(np.corrcoef(he[:n_cmp], ref[:n_cmp])[0, 1])
    print(
        f"vs slot twin: max|err|={err:.3e} corr={corr:.4f} "
        f"max|he|={float(np.max(np.abs(he[:n_cmp]))):.3e} "
        f"max|ref|={float(np.max(np.abs(ref[:n_cmp]))):.3e}",
        flush=True,
    )
    ok = err < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
