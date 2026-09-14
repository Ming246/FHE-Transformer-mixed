#!/usr/bin/env python3
"""
Plain (non-DualRail) bootstrap noise study.

1) Different message amplitudes → abs/rel err after real bs
2) Different pre-bs level_calc (encrypt directly at that level)
3) Control: level_up only (no bs) to separate 「改 level」误差

Compare post-bs decrypt vs **pre-bs decrypt** (not raw message) so encoding
noise is cancelled from the bootstrap delta.

Usage:
  python3 HE/thor/audit_bootstrap_noise.py
  python3 HE/thor/audit_bootstrap_noise.py --amps 0.3 3 12 --levels 10 20 27
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

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import MAX_LEVEL_UP_SPAN, safe_level_up  # noqa: E402


def _dec(engine, ct, sk) -> np.ndarray:
    return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64)


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _err(got: np.ndarray, ref: np.ndarray) -> dict:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    e = np.abs(g - r)
    peak = float(np.max(np.abs(r))) + 1e-30
    denom = float(np.dot(g, g)) + 1e-30
    return {
        "max_abs": float(np.max(e)),
        "mean_abs": float(np.mean(e)),
        "max_rel": float(np.max(e) / peak),
        "ls": float(np.dot(g, r) / denom),
    }


def make_msg(engine, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, engine.num_slots).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 12.0])
    ap.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25, 27],
        help="pre-bs level_calc via encodecrypt(level=...)",
    )
    ap.add_argument(
        "--level-up-spans",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="control: level_up span from enc@0 (no bootstrap)",
    )
    ap.add_argument("--control-amp", type=float, default=3.0)
    args = ap.parse_args()

    print("create engine + keys (real bootstrap)...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=True,
        with_rot_deltas=True,
        rot_deltas=[1, 2048],
    )
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    print(f"  ready ({time.time() - t0:.1f}s) num_levels={nl}", flush=True)

    # ---- Control: encode only + level_up only ----
    print("\n=== CONTROL: encode / level_up only (no bootstrap) ===", flush=True)
    amp_c = float(args.control_amp)
    msg_c = make_msg(engine, amp_c, seed=0)
    ct0 = engine.encodecrypt(msg_c, pk, level=0)
    d0 = _dec(engine, ct0, sk)
    e_enc = _err(d0, msg_c)
    print(
        f"  enc@0 vs msg amp={amp_c:g}: max|err|={e_enc['max_abs']:.3e}  "
        f"rel={e_enc['max_rel']:.3e}  ls={e_enc['ls']:.6g}",
        flush=True,
    )
    for span in args.level_up_spans:
        if MAX_LEVEL_UP_SPAN is not None and span > MAX_LEVEL_UP_SPAN:
            print(f"  level_up span={span}: SKIP (>{MAX_LEVEL_UP_SPAN})", flush=True)
            continue
        ct = engine.encodecrypt(msg_c, pk, level=0)
        pre = _dec(engine, ct, sk)
        dst = int(ct.level_calc) + int(span)
        ct_up = safe_level_up(engine, ct, dst)
        post = _dec(engine, ct_up, sk)
        e = _err(post, pre)
        e_msg = _err(post, msg_c)
        print(
            f"  level_up 0→{dst} (span={span}): vs_pre max|err|={e['max_abs']:.3e}  "
            f"rel={e['max_rel']:.3e} | vs_msg max|err|={e_msg['max_abs']:.3e}  "
            f"lc={ct_up.level_calc} rem={_rem(engine, ct_up)}",
            flush=True,
        )

    # Also: encrypt directly at high level vs enc@0 (different encode path)
    print("\n=== CONTROL: encodecrypt at different levels (no bs) ===", flush=True)
    for lc in args.levels:
        if lc < 0 or lc >= nl:
            print(f"  enc@{lc}: SKIP (out of range)", flush=True)
            continue
        msg = make_msg(engine, amp_c, seed=1)
        ct = engine.encodecrypt(msg, pk, level=int(lc))
        d = _dec(engine, ct, sk)
        e = _err(d, msg)
        print(
            f"  enc@{lc} rem={_rem(engine, ct)}: vs_msg max|err|={e['max_abs']:.3e}  "
            f"rel={e['max_rel']:.3e}  ls={e['ls']:.6g}",
            flush=True,
        )

    # ---- Main: amp × pre-bs level → bootstrap ----
    print(
        "\n=== BOOTSTRAP: err = decrypt(post) − decrypt(pre) "
        "(encoding cancelled) ===",
        flush=True,
    )
    print(
        f"{'amp':>6} {'pre_lc':>6} {'pre_rem':>7} {'post_lc':>7} {'post_rem':>8} "
        f"{'max|err|':>10} {'rel':>10} {'ls':>8} {'vs_msg':>10}",
        flush=True,
    )
    rows = []
    for amp in args.amps:
        for lc in args.levels:
            if lc < 0 or lc >= nl:
                continue
            msg = make_msg(engine, float(amp), seed=100 + int(amp * 10) + lc)
            ct = engine.encodecrypt(msg, pk, level=int(lc))
            pre = _dec(engine, ct, sk)
            pre_lc, pre_rem = int(ct.level_calc), _rem(engine, ct)
            t1 = time.time()
            ct_bs = engine.bootstrap(ct)
            dt = time.time() - t1
            post = _dec(engine, ct_bs, sk)
            e = _err(post, pre)
            e_msg = _err(post, msg)
            post_lc, post_rem = int(ct_bs.level_calc), _rem(engine, ct_bs)
            print(
                f"{amp:6g} {pre_lc:6d} {pre_rem:7d} {post_lc:7d} {post_rem:8d} "
                f"{e['max_abs']:10.3e} {e['max_rel']:10.3e} {e['ls']:8.4g} "
                f"{e_msg['max_abs']:10.3e}  ({dt:.0f}s)",
                flush=True,
            )
            rows.append(
                {
                    "amp": amp,
                    "pre_lc": pre_lc,
                    "max_abs": e["max_abs"],
                    "max_rel": e["max_rel"],
                    "ls": e["ls"],
                }
            )

    # Summary: amp sweep at fixed mid level; level sweep at fixed amp
    print("\n=== SUMMARY ===", flush=True)
    mid_lc = 20 if 20 in args.levels else args.levels[len(args.levels) // 2]
    print(f"  By amp @ pre_lc={mid_lc} (bs noise vs pre):", flush=True)
    for r in rows:
        if r["pre_lc"] == mid_lc:
            print(
                f"    amp={r['amp']:g}: max|err|={r['max_abs']:.3e}  "
                f"rel={r['max_rel']:.3e}  ls={r['ls']:.4g}",
                flush=True,
            )
    mid_amp = 3.0 if 3.0 in args.amps else args.amps[len(args.amps) // 2]
    print(f"  By pre_lc @ amp={mid_amp:g}:", flush=True)
    for r in rows:
        if abs(r["amp"] - mid_amp) < 1e-12:
            print(
                f"    pre_lc={r['pre_lc']}: max|err|={r['max_abs']:.3e}  "
                f"rel={r['max_rel']:.3e}  ls={r['ls']:.4g}",
                flush=True,
            )

    print("PASS (bootstrap noise study finished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
