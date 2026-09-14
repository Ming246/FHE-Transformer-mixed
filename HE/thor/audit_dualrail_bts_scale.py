#!/usr/bin/env python3
"""
Isolated DualRail pack → bootstrap → unpack scale / noise audit.

Does NOT touch DP placement. Confirms:
  - THOR score DualRail unpack = +conj (2·Re/2·Im), no ×½ (bert.py:155-161)
  - scale_after=0.5 restores pre-bs amplitude
  - ls_gain≈1 ⇒ scale OK; abs err with ls≈1 ⇒ Liberate bs noise

Encrypt at level_calc=20 (no square-burn — that destroys amplitude).

Usage:
  python3 HE/thor/audit_dualrail_bts_scale.py --mock
  python3 HE/thor/audit_dualrail_bts_scale.py          # real bs
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

from dualrail_bts import install_mock_bootstrap, refresh_dualrail8  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402

ENC_LEVEL = 20


def _stats(got: np.ndarray, ref: np.ndarray) -> dict:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    err = np.abs(g - r)
    peak = float(np.max(np.abs(r))) + 1e-30
    denom = float(np.dot(g, g)) + 1e-30
    return {
        "max_abs": float(np.max(err)),
        "max_rel": float(np.max(err) / peak),
        "ls_gain": float(np.dot(g, r) / denom),
        "corr": float(np.corrcoef(g, r)[0, 1]),
        "max_got": float(np.max(np.abs(g))),
        "max_ref": float(np.max(np.abs(r))),
    }


def _dec(engine, ct, sk) -> np.ndarray:
    return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64)


def audit_plain_bs(engine, sk, pk, amp: float, *, burn: int = 6) -> dict:
    """Single-CT bootstrap noise (square-burn so bs is exercised)."""
    rng = np.random.default_rng(0)
    msg = rng.uniform(-amp, amp, engine.num_slots).astype(np.float64)
    ct = engine.encodecrypt(msg, pk, level=0)
    for _ in range(burn):
        ct = engine.rescale(engine.square(ct))
    before = _dec(engine, ct, sk)
    after = _dec(engine, engine.bootstrap(ct), sk)
    return _stats(after, before)


def audit_dualrail8(
    engine,
    sk,
    pk,
    amp: float,
    *,
    scale_after: float | None,
    seed: int = 1,
) -> dict:
    rng = np.random.default_rng(seed)
    packs_pt = [
        rng.uniform(-amp, amp, engine.num_slots).astype(np.float64) for _ in range(8)
    ]
    packs = [engine.encodecrypt(p, pk, level=ENC_LEVEL) for p in packs_pt]
    pre = [_dec(engine, packs[i], sk) for i in range(8)]

    # THOR score unpack expectation: 2·Re / 2·Im
    ref: list[np.ndarray] = [None] * 8  # type: ignore[list-item]
    for i in range(4):
        ref[i] = 2.0 * pre[i]
        ref[i + 4] = 2.0 * pre[i + 4]
    if scale_after is not None:
        s = float(scale_after)
        ref = [x * s for x in ref]

    out = refresh_dualrail8(
        engine,
        packs,
        hook=None,
        event_name="audit.dualrail8",
        scale_after=scale_after,
    )
    got = [_dec(engine, out[i], sk) for i in range(8)]
    st = _stats(np.concatenate(got), np.concatenate(ref))
    st["scale_after"] = scale_after
    st["level"] = int(packs[0].level_calc)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--amps", type=float, nargs="+", default=[0.3, 3.0, 12.0])
    ap.add_argument("--skip-plain", action="store_true", help="skip square-burn plain bs")
    args = ap.parse_args()

    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=not args.mock,
        with_rot_deltas=True,
        rot_deltas=[1, 2048],
    )
    sk, pk = keys["sk"], keys["pk"]
    if args.mock:
        info = install_mock_bootstrap(engine, sk, pk)
        print(f"  MOCK land_level={info['land_level']}", flush=True)
    print(f"  ready ({time.time() - t0:.1f}s) num_levels={engine.num_levels}", flush=True)

    mode = "mock" if args.mock else "real"
    print(f"\n=== DualRail / scale audit ({mode} bs, enc level={ENC_LEVEL}) ===")
    print(
        "THOR score DualRail unpack = +conj (2·Re/2·Im), NO ×½.\n"
        "ls≈1 ⇒ scale OK; abs err with ls≈1 ⇒ bs noise (not a scale bug).\n",
        flush=True,
    )

    for amp in args.amps:
        print(f"-- amp=±{amp:g} --", flush=True)
        if not args.skip_plain:
            plain = audit_plain_bs(engine, sk, pk, amp)
            print(
                f"  plain CT bs (burned): max|err|={plain['max_abs']:.3e}  "
                f"rel={plain['max_rel']:.3e}  ls={plain['ls_gain']:.4g}",
                flush=True,
            )
        d2 = audit_dualrail8(engine, sk, pk, amp, scale_after=1.0)
        print(
            f"  DualRail8 (no /2):    max|err|={d2['max_abs']:.3e}  "
            f"rel={d2['max_rel']:.3e}  ls={d2['ls_gain']:.4g}  "
            f"corr={d2['corr']:.4f}  max|got|={d2['max_got']:.4g} "
            f"(expect ~{2 * amp:g})",
            flush=True,
        )
        d05 = audit_dualrail8(engine, sk, pk, amp, scale_after=0.5, seed=2)
        print(
            f"  DualRail8 ×0.5→pre:   max|err|={d05['max_abs']:.3e}  "
            f"rel={d05['max_rel']:.3e}  ls={d05['ls_gain']:.4g}  "
            f"corr={d05['corr']:.4f}",
            flush=True,
        )
        print(flush=True)

    print("PASS (audit finished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
