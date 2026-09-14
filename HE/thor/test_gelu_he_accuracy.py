#!/usr/bin/env python3
"""
HE Chebyshev GeLU accuracy — single formal-size CT.

Physical ``x`` uniform on a chosen interval, encode ``t = x/C``, one CKKS CT
with full ``num_slots`` (THOR_DEFAULT). Compare decrypt to slot twin.

**No bootstrap**: encrypt directly at ``want_rem`` level (unit-test style).
Keys load without bs/rot bundles.

Reports: max/mean absolute error, max/mean relative error (|GELU ref|≥rel_floor).

Usage::

  # 默认：high，x∈[5-C,C-5]，满槽
  python3 HE/thor/test_gelu_he_accuracy.py --layer 0

  # 改采样点数 n（前 n 个 slot 均匀取点，其余填 0；CT 仍是正式大小）
  python3 HE/thor/test_gelu_he_accuracy.py --layer 0 --n 1024

  # margin：x∈[margin-C, C-margin]
  python3 HE/thor/test_gelu_he_accuracy.py --layer 0 --margin 10

  # 或直接指定区间
  python3 HE/thor/test_gelu_he_accuracy.py --layer 0 --x-lo -20 --x-hi 20
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
sys.path.insert(0, _HERE)
sys.path.insert(0, _HE)
sys.path.insert(0, _REPO)
from path_setup import ensure_thor_path

ensure_thor_path()

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from gelu_he_ct import gelu_poly_ct  # noqa: E402
from slot_gelu_he import GeluHeParams, gelu_poly_slots  # noqa: E402


def _report(name: str, he: np.ndarray, ref: np.ndarray, *, rel_floor: float) -> dict:
    err = np.abs(he - ref)
    mask = np.abs(ref) >= rel_floor
    if mask.any():
        rel = err[mask] / np.abs(ref[mask])
        max_rel = float(np.max(rel))
        mean_rel = float(np.mean(rel))
        n_rel = int(mask.sum())
    else:
        max_rel = float("nan")
        mean_rel = float("nan")
        n_rel = 0
    out = {
        "max_abs": float(np.max(err)),
        "mean_abs": float(np.mean(err)),
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "n_rel": n_rel,
    }
    imax = int(np.argmax(err))
    print(f"--- {name} ---", flush=True)
    print(
        f"  max_abs  = {out['max_abs']:.6e}\n"
        f"  mean_abs = {out['mean_abs']:.6e}\n"
        f"  max_rel  = {out['max_rel']:.6e}  (|ref|>={rel_floor:g}, n={n_rel})\n"
        f"  mean_rel = {out['mean_rel']:.6e}\n"
        f"  worst abs @ he={he[imax]:.6g}  ref={ref[imax]:.6g}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--level",
        type=int,
        default=2,
        help="0/1/2 = low/mid/high (default high)",
    )
    ap.add_argument("--want-rem", type=int, default=14)
    ap.add_argument(
        "--rel-floor",
        type=float,
        default=1e-4,
        help="relative error only where |GELU ref| >= this",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=None,
        help="sample count (linspace); default=num_slots. "
        "Padded to full CT with zeros; errors only on first n slots.",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=5.0,
        help="if --x-lo/--x-hi unset: x in [margin-C, C-margin]",
    )
    ap.add_argument("--x-lo", type=float, default=None, help="explicit physical x min")
    ap.add_argument("--x-hi", type=float, default=None, help="explicit physical x max")
    args = ap.parse_args()

    params = GeluHeParams.from_gelu_poly(args.layer, level=args.level)
    C = float(params.C)
    if args.x_lo is not None or args.x_hi is not None:
        if args.x_lo is None or args.x_hi is None:
            raise SystemExit("set both --x-lo and --x-hi, or neither")
        lo, hi = float(args.x_lo), float(args.x_hi)
    else:
        margin = float(args.margin)
        lo, hi = margin - C, C - margin
    if lo >= hi:
        raise SystemExit(f"empty range [{lo}, {hi}]")

    print(
        f"GeLU HE accuracy  layer={args.layer} level={args.level}  "
        f"scheme={params.scheme_name}  C={C}  depth_he={params.depth_he}",
        flush=True,
    )

    print("create engine + load keys (no bootstrap keys) ...", flush=True)
    t_eng = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    n_slots = int(engine.num_slots)
    n_use = int(args.n) if args.n is not None else n_slots
    if n_use < 1 or n_use > n_slots:
        raise SystemExit(f"--n must be in 1..{n_slots}, got {n_use}")
    print(
        f"  ready ({time.time() - t_eng:.1f}s)  num_slots={n_slots}  "
        f"num_levels={engine.num_levels}",
        flush=True,
    )
    print(
        f"x ∈ [{lo:g}, {hi:g}]  t=x/C  samples n={n_use}  "
        f"(CT padded to {n_slots})",
        flush=True,
    )

    msg = np.zeros(n_slots, dtype=np.float64)
    x_phys = np.linspace(lo, hi, n_use, dtype=np.float64)
    msg[:n_use] = x_phys / C

    from bts_ops import remaining_to_level_calc  # noqa: E402

    # Unit test: encrypt straight to GeLU entry rem — no bootstrap.
    target_lc = remaining_to_level_calc(int(args.want_rem), engine.num_levels)
    print(f"encodecrypt @ rem≈{args.want_rem} (no bts) ...", flush=True)
    t_enc = time.time()
    ct = engine.encodecrypt(msg, pk, level=target_lc)
    lv0 = int(ct.level_calc)
    print(
        f"  entry level={lv0} rem={engine.num_levels - lv0}  ({time.time() - t_enc:.1f}s)",
        flush=True,
    )

    print("gelu_poly_ct ...", flush=True)
    t0 = time.time()
    ct_out = gelu_poly_ct(engine, ct, params)
    lv1 = int(ct_out.level_calc)
    used = lv1 - lv0
    print(
        f"  done ({time.time() - t0:.1f}s)  level {lv0}→{lv1}  "
        f"used={used}  depth_he={params.depth_he}  "
        f"overhead={used - int(params.depth_he)}",
        flush=True,
    )

    he = np.real(np.asarray(engine.decrode(ct_out, sk), dtype=np.complex128))
    he = np.asarray(he, dtype=np.float64).ravel()[:n_use]
    ref = np.asarray(gelu_poly_slots(msg[:n_use], params), dtype=np.float64).ravel()[
        :n_use
    ]

    print(f"compared slots: {n_use}", flush=True)
    _report("HE vs slot twin (Cheb PS-tree)", he, ref, rel_floor=float(args.rel_floor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
