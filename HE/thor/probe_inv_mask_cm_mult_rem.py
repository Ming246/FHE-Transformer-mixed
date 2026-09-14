#!/usr/bin/env python3
"""
Measure rem around Softmax inv_mask ``cm_mult`` and DualRail exit.

Mirrors ``update_inv_D_fixed`` tail + ``dualrail_softmax_exit``::

  inv @ low rem → bootstrap (mock | real) → rem_before
  → cm_mult(short mask) → rem_after_cm
  → dualrail_softmax_exit → rem(sftmx_out)

Usage::

  python3 HE/thor/probe_inv_mask_cm_mult_rem.py --mode mock
  python3 HE/thor/probe_inv_mask_cm_mult_rem.py --mode real
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
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HERE)

from dualrail_bts import install_mock_bootstrap  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from softmax_he_ct import dualrail_softmax_exit  # noqa: E402
from bts_ops import BOOTSTRAP_DEPTH_BUDGET  # noqa: E402


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _short_mask() -> np.ndarray:
    # Same as update_inv_D_fixed(final_inv=True)
    return np.array(([1.0] * 12 + [0.0] * 4) * (2**7), dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("mock", "real"), required=True)
    ap.add_argument(
        "--start-rem",
        type=int,
        default=2,
        help="encrypt inv/exp at this rem before bootstrap",
    )
    ap.add_argument(
        "--d-delta",
        type=float,
        default=0.00388556,
        help="plain d_delta for exit scale_k (typical Σy² value)",
    )
    args = ap.parse_args()

    nl_hint = 29
    print(
        f"=== inv_mask cm_mult rem probe  mode={args.mode} "
        f"start_rem={args.start_rem} ===",
        flush=True,
    )
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=(args.mode == "real"),
        with_rot_deltas=True,
    )
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    print(
        f"  engine ready ({time.time()-t0:.1f}s) nl={nl} slots={engine.num_slots}",
        flush=True,
    )

    if args.mode == "mock":
        info = install_mock_bootstrap(
            engine, sk, pk, rem_after=int(BOOTSTRAP_DEPTH_BUDGET)
        )
        print(
            f"  MOCK land_level={info['land_level']} "
            f"rem≈{info.get('rem_after', nl - info['land_level'])}",
            flush=True,
        )
    else:
        print("  REAL engine.bootstrap (no mock)", flush=True)

    start_lc = nl - int(args.start_rem)
    n = int(engine.num_slots)
    rng = np.random.default_rng(0)

    # Synthetic post-Σy² state: inv (real) + 8 exp packs (real).
    inv_msg = rng.normal(0.0, 1e-3, size=n).astype(np.float64)
    inv_d = engine.encodecrypt(inv_msg, pk, level=start_lc)
    exp_u = []
    for _ in range(8):
        msg = rng.normal(0.0, 1e-4, size=n).astype(np.float64)
        exp_u.append(engine.encodecrypt(msg, pk, level=start_lc))

    print(
        f"\n  [0] encrypt inv/exp @ lc={start_lc} rem={_rem(engine, inv_d)}",
        flush=True,
    )

    t1 = time.time()
    inv_d = engine.bootstrap(inv_d)
    print(
        f"  [1] after bootstrap({args.mode})  "
        f"lc={inv_d.level_calc} rem={_rem(engine, inv_d)}  "
        f"({time.time()-t1:.1f}s)",
        flush=True,
    )

    rem_before = _rem(engine, inv_d)
    lc_before = int(inv_d.level_calc)
    # Same length as update_inv_D_fixed(final_inv=True); Liberate cm_mult accepts it.
    masking = _short_mask()
    print(
        f"  [1b] masking size={masking.size} (THOR short final_inv mask)",
        flush=True,
    )

    inv_d2 = engine.cm_mult(inv_d, masking)
    rem_after_cm = _rem(engine, inv_d2)
    lc_after = int(inv_d2.level_calc)
    print(
        f"  [2] cm_mult(inv, short_mask)  "
        f"lc {lc_before}→{lc_after}  rem {rem_before}→{rem_after_cm}  "
        f"Δrem={rem_before - rem_after_cm}",
        flush=True,
    )

    # Align exp to inv rem for exit (level_up / re-encrypt at inv rem).
    # Use re-encrypt at same rem as post-cm inv to avoid level_up noise questions.
    land_lc = int(inv_d2.level_calc)
    exp_aligned = []
    for i in range(8):
        msg = np.asarray(engine.decrode(exp_u[i], sk, is_real=True), dtype=np.float64)
        exp_aligned.append(engine.encodecrypt(msg, pk, level=land_lc))
    print(
        f"  [3] re-enc exp @ same lc as post-cm inv "
        f"(lc={land_lc} rem={nl - land_lc})",
        flush=True,
    )

    t2 = time.time()
    sftmx_out = dualrail_softmax_exit(
        engine,
        exp_aligned,
        inv_d2,
        d_delta=float(args.d_delta),
        rescale=False,
        bts_hook=None,
        bts_event=None,
    )
    rem_out = _rem(engine, sftmx_out[0])
    print(
        f"  [4] after dualrail_softmax_exit  "
        f"sftmx_out[0] lc={sftmx_out[0].level_calc} rem={rem_out}  "
        f"({time.time()-t2:.1f}s)",
        flush=True,
    )
    print(
        f"\n  SUMMARY mode={args.mode}: "
        f"boot→rem={rem_before}, "
        f"cm_mult→rem={rem_after_cm} (Δ={rem_before - rem_after_cm}), "
        f"sftmx_out→rem={rem_out} "
        f"(exit Δ from post-cm = {rem_after_cm - rem_out})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
