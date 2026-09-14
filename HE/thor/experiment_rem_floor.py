#!/usr/bin/env python3
"""
Measure Liberate rem after plain vs DualRail bootstrap, and rem=1 multiply floor.

  python3 HE/thor/experiment_rem_floor.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
sys.path[:0] = [_HERE, _HE, _REPO]

from path_setup import ensure_thor_path  # noqa: E402

ensure_thor_path()
sys.path.append(_HE)

from bootstrap_hook import BootstrapHook  # noqa: E402
from dualrail_bts import (  # noqa: E402
    install_mock_bootstrap,
    refresh_dualrail8,
    uninstall_mock_bootstrap,
)
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402


def rem(ct, engine) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def try_op(name: str, fn) -> tuple[bool, str, int | None]:
    try:
        out = fn()
        r = rem(out, fn.__self__ if hasattr(fn, "__self__") else None)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None
    # rem from closure if engine passed differently
    return True, "ok", None


def main() -> int:
    print("=== create engine + mock bootstrap ===", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=True, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    nslots = int(engine.num_slots)
    nl = int(engine.num_levels)
    print(f"num_levels={nl}  budget={nl - 15} (typical land lc=15 → rem=14)", flush=True)

    mock = install_mock_bootstrap(engine, sk, pk)
    land_lc = int(mock["land_level"])
    print(f"mock bootstrap land level_calc={land_lc} → rem={nl - land_lc}", flush=True)

    hook = BootstrapHook(engine, mode="always")

    # --- Part A: plain vs DualRail landing rem ---
    print("\n=== A. bootstrap landing rem (after full refresh path) ===", flush=True)

    msg = np.random.default_rng(0).normal(0, 0.1, nslots).astype(np.float64)
    ct0 = engine.encodecrypt(msg, pk, level=land_lc)

    ct_plain = engine.bootstrap(ct0)
    print(
        f"  plain bootstrap (1 CT): rem {rem(ct0, engine)}→{rem(ct_plain, engine)} "
        f"(lc {ct0.level_calc}→{ct_plain.level_calc})",
        flush=True,
    )

    packs = np.empty(8, dtype=object)
    for i in range(8):
        packs[i] = engine.encodecrypt(msg * (0.9 + 0.01 * i), pk, level=land_lc)
    r_before = rem(packs[0], engine)

    out_dr = refresh_dualrail8(
        engine,
        list(packs),
        hook,
        "test.dualrail8",
        scale_after=0.5,
    )
    print(
        f"  DualRail8 pack→bts→unpack×½: rem {r_before}→{rem(out_dr[0], engine)} "
        f"(lc {packs[0].level_calc}→{out_dr[0].level_calc})",
        flush=True,
    )
    # stepwise inside dualrail (manual)
    temp = engine.cc_add(packs[0], engine.imult(packs[4]))
    print(f"    after pack (1 cplx): rem={rem(temp, engine)} lc={temp.level_calc}", flush=True)
    temp_b = engine.bootstrap(temp)
    print(
        f"    after bootstrap (cplx, before unpack): rem={rem(temp_b, engine)} "
        f"lc={temp_b.level_calc}",
        flush=True,
    )
    conj = engine.conjugate(temp_b)
    re = engine.cc_add(temp_b, conj)
    im = engine.imult(engine.cc_sub(conj, temp_b))
    print(f"    after unpack re/im (no ×½): rem re={rem(re, engine)} im={rem(im, engine)}", flush=True)
    re_half = engine.mult_scalar(re, 0.5)
    im_half = engine.mult_scalar(im, 0.5)
    print(
        f"    after unpack ×½ mult_scalar: rem re={rem(re_half, engine)} im={rem(im_half, engine)}",
        flush=True,
    )

    # plan_refresh direct (single CT / cplx spine)
    packs2 = np.empty(8, dtype=object)
    for i in range(8):
        packs2[i] = engine.encodecrypt(msg, pk, level=land_lc)
    out_plain8 = list(hook.refresh("test.plain8", list(packs2)))
    print(
        f"  plain bootstrap (8 CTs, no DualRail): rem→{rem(out_plain8[0], engine)} "
        f"lc={out_plain8[0].level_calc}",
        flush=True,
    )

    lane = np.array(([1.0] * 6 + [0.0] * 10) * (nslots // 16), dtype=np.float64)

    # --- Part B: multiply floor at each starting rem ---
    print("\n=== B. multiply at starting rem (rescale=True where applicable) ===", flush=True)
    print("  rem = num_levels - level_calc", flush=True)

    ops = [
        ("cm_mult+rescale", lambda e, c, m: e.cm_mult(c, m)),
        ("square+rescale", lambda e, c, m: e.square(c)),
        (
            "auto_ct_ct_mult+rescale",
            lambda e, c, m: e.auto_ct_ct_mult(c, m, relin=True, rescale=True),
        ),
        ("mult_scalar(0.5)", lambda e, c, m: e.mult_scalar(c, 0.5)),
        ("add (no rescale)", lambda e, c, m: e.add(c, c)),
    ]

    for start_rem in range(8, 0, -1):
        lc = nl - start_rem
        ct_a = engine.encodecrypt(msg, pk, level=lc)
        ct_b = engine.encodecrypt(msg * 0.7, pk, level=lc)
        assert rem(ct_a, engine) == start_rem
        print(f"\n  --- start rem={start_rem} (lc={lc}) ---", flush=True)
        for op_name, op_fn in ops:
            try:
                if "ct_ct" in op_name:
                    out = op_fn(engine, ct_a, ct_b)
                elif op_name.startswith("add"):
                    out = op_fn(engine, ct_a, ct_b)
                else:
                    out = op_fn(engine, ct_a, lane)
                r_after = rem(out, engine)
                print(f"    {op_name:28s} PASS  rem {start_rem}→{r_after}", flush=True)
            except Exception as e:
                err = str(e).splitlines()[0]
                print(f"    {op_name:28s} FAIL  {type(e).__name__}: {err}", flush=True)

    # --- Part C: chain multiplies from rem=14 until fail ---
    print("\n=== C. chain cm_mult+rescale from rem=14 until fail ===", flush=True)
    ct = engine.encodecrypt(msg, pk, level=land_lc)
    ct = engine.bootstrap(ct)
    r0 = rem(ct, engine)
    print(f"  after bootstrap rem={r0}", flush=True)
    step = 0
    while True:
        try:
            ct = engine.cm_mult(ct, lane)
            step += 1
            print(f"    step {step}: rem={rem(ct, engine)} lc={ct.level_calc}", flush=True)
        except Exception as e:
            print(
                f"    step {step + 1} FAIL at entry rem={rem(ct, engine)}: "
                f"{type(e).__name__}",
                flush=True,
            )
            break

    uninstall_mock_bootstrap(engine)
    print("\n=== done ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
