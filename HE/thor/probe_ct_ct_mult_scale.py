#!/usr/bin/env python3
"""
Liberate CT×CT / PT×CT multiply + decrypt smoke (working region).

Encrypt two messages at after-bootstrap land (rem=14 → level_calc=15),
then compare decrypt vs a*b for several multiply styles.

Expected (num_levels=29, scale_bits=41, engine.scale=2^41):

  encrypt                     OK
  ct_ct_mult(rescale=False)   FAIL after post-rescale decrypt
  ct_ct_mult(rescale=True)    OK  (Liberate default: pre-rescale inputs)
  auto_ct_ct_mult             OK  (same as rescale=True)
  pt_ct_mult + rescale        OK  (QKV-style)
  pt_ct_mult no rescale       FAIL

Usage::

  python3 HE/thor/probe_ct_ct_mult_scale.py
  python3 HE/thor/probe_ct_ct_mult_scale.py --rem 10
  python3 HE/thor/probe_ct_ct_mult_scale.py --rem 20   # 2^59 chain segment
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
if _HE in sys.path:
    sys.path.remove(_HE)
sys.path.insert(0, _HE)

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402


def _report(tag: str, he, ref, *, engine, nl: int, ct=None) -> None:
    he = np.asarray(he, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    corr = (
        float(np.corrcoef(he, ref)[0, 1])
        if np.std(he) > 1e-15 and np.std(ref) > 1e-15
        else float("nan")
    )
    mask = np.abs(ref) > 1e-4
    med = float(np.median(he[mask] / ref[mask])) if np.any(mask) else float("nan")
    err = float(np.max(np.abs(he - ref)))
    meta = ""
    if ct is not None:
        lc = int(ct.level_calc)
        rem = nl - lc
        q = float(
            engine.ctx.q[
                engine.ntt.p.destination_arrays[lc][engine.ntt.p.rescaler_loc[lc]][0]
            ]
        )
        meta = f"  lc={lc} rem={rem} rescale_q≈2^{np.log2(q):.0f}"
    ok = np.isfinite(corr) and corr > 0.999 and abs(med - 1.0) < 1e-3
    print(
        f"{'OK' if ok else 'BAD'}  {tag}{meta}\n"
        f"     max|he|={np.max(np.abs(he)):.4g}  max|ref|={np.max(np.abs(ref)):.4g}  "
        f"corr={corr:.6f}  med(he/ref)={med:.6g}  err@1={err:.3e}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rem",
        type=int,
        default=5,
        help="encrypt remaining depth (14=bts land; 10=deeper work; 20=2^59 segment)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    rem = int(args.rem)
    lc = nl - rem
    if rem <= 0 or lc < 0 or lc >= nl:
        raise SystemExit(f"bad rem={rem} for num_levels={nl}")

    rng = np.random.default_rng(args.seed)
    a = rng.normal(0.0, 10,engine.num_slots).astype(np.float64)
    b = rng.normal(0.0, 10,engine.num_slots).astype(np.float64)
    ref = a * b

    print(
        f"=== ct×ct / pt×ct smoke  rem={rem} → level_calc={lc}  "
        f"engine.scale=2^{int(np.log2(float(engine.scale)))} "
        f"({float(engine.scale):.6g}) ===",
        flush=True,
    )

    cta = engine.encodecrypt(a, pk, level=lc)
    ctb = engine.encodecrypt(b, pk, level=lc)
    _report("encrypt a", engine.decrode(cta, sk, is_real=True), a, engine=engine, nl=nl, ct=cta)

    # A: rescale=False then post-rescale
    print("\n-- A  ct_ct_mult(rescale=False) --", flush=True)
    prod0 = engine.ct_ct_mult(cta, ctb, rescale=False, relin=True)
    _report(
        "decrypt product (no post-rs)",
        engine.decrode(prod0, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=prod0,
    )
    prod0_rs = engine.rescale(prod0)
    _report(
        "decrypt after post-rescale",
        engine.decrode(prod0_rs, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=prod0_rs,
    )

    # B: Liberate default pre-rescale
    print("\n-- B  ct_ct_mult(rescale=True)  [default] --", flush=True)
    prod1 = engine.ct_ct_mult(cta, ctb, rescale=True, relin=True)
    _report(
        "decrypt product",
        engine.decrode(prod1, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=prod1,
    )

    # C: auto wrapper
    print("\n-- C  auto_ct_ct_mult --", flush=True)
    prod2 = engine.auto_ct_ct_mult(cta, ctb)
    _report(
        "decrypt product",
        engine.decrode(prod2, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=prod2,
    )

    # D: PT×CT (QKV-style)
    print("\n-- D  pt_ct_mult --", flush=True)
    pt_a = engine.encode(a, level=lc)
    pct = engine.pt_ct_mult(pt_a, ctb)
    _report(
        "decrypt (no rescale)",
        engine.decrode(pct, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=pct,
    )
    pct_rs = engine.rescale(pct)
    _report(
        "decrypt after rescale",
        engine.decrode(pct_rs, sk, is_real=True),
        ref,
        engine=engine,
        nl=nl,
        ct=pct_rs,
    )

    print(
        "\nNote: rem=14 is bootstrap land (2^41 rescale primes). "
        "rem=20 sits in the earlier 2^59 segment — behavior differs.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
