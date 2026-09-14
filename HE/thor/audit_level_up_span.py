#!/usr/bin/env python3
"""
打破 ``MAX_LEVEL_UP_SPAN=8``：直接 ``engine.level_up``，测大跨度误差。

默认从 land=15（bts 落点）一次性跳到 16 … num_levels-1（含 span>8）。
对照：从 lc=0 大跨度（README 称前段会坏）。

Usage:
  python3 HE/thor/audit_level_up_span.py
  python3 HE/thor/audit_level_up_span.py --lands 0 15 --amps 1 3
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
from linear_eval import MAX_LEVEL_UP_SPAN  # noqa: E402


def _err(got, ref) -> dict:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    e = np.abs(g - r)
    peak = float(np.max(np.abs(r))) + 1e-30
    return {
        "max_abs": float(np.max(e)),
        "mean_abs": float(np.mean(e)),
        "max_rel": float(np.max(e) / peak),
    }


def make_msg(engine, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-amp, amp, engine.num_slots).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", type=float, nargs="+", default=[0.1, 1.0, 3.0])
    ap.add_argument(
        "--lands",
        type=int,
        nargs="+",
        default=[15, 0],
        help="start level_calc (15=bts land; 0=chain head)",
    )
    ap.add_argument(
        "--dst-max",
        type=int,
        default=None,
        help="max dst level_calc (default: num_levels-1)",
    )
    ap.add_argument(
        "--also-step",
        action="store_true",
        help="also successive +1 up to same dst (slower)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    dst_max = int(args.dst_max) if args.dst_max is not None else nl - 1
    print(
        f"  ready ({time.time() - t0:.1f}s) num_levels={nl} "
        f"MAX_LEVEL_UP_SPAN(policy)={MAX_LEVEL_UP_SPAN} "
        f"dst_max={dst_max}  (this script ignores the policy)",
        flush=True,
    )

    for land in args.lands:
        land = int(land)
        if land < 0 or land >= dst_max:
            print(f"\nSKIP land={land}", flush=True)
            continue
        print(
            f"\n=== CT one-jump: enc@{land} → level_up(dst) "
            f"(span 1..{dst_max - land}; policy limit={MAX_LEVEL_UP_SPAN}) ===",
            flush=True,
        )
        print(
            f"{'amp':>8} {'dst':>5} {'span':>5} {'policy':>8} "
            f"{'vs_pre max|e|':>14} {'vs_pre rel':>12} "
            f"{'vs_msg max|e|':>14} {'enc_only':>12}",
            flush=True,
        )
        for amp in args.amps:
            amp = float(amp)
            msg = make_msg(engine, amp, seed=args.seed)
            ct0 = engine.encodecrypt(msg, pk, level=land)
            pre = np.asarray(engine.decrode(ct0, sk, is_real=True), dtype=np.float64)
            e_enc = _err(pre, msg)
            for dst in range(land + 1, dst_max + 1):
                span = dst - land
                flag = (
                    "uncapped"
                    if MAX_LEVEL_UP_SPAN is None
                    else ("OK" if span <= MAX_LEVEL_UP_SPAN else "over_cap")
                )
                try:
                    ct1 = engine.level_up(ct0, int(dst))
                    post = np.asarray(
                        engine.decrode(ct1, sk, is_real=True), dtype=np.float64
                    )
                    e_pre = _err(post, pre)
                    e_msg = _err(post, msg)
                    print(
                        f"{amp:8.3g} {dst:5d} {span:5d} {flag:>8} "
                        f"{e_pre['max_abs']:14.3e} {e_pre['max_rel']:12.3e} "
                        f"{e_msg['max_abs']:14.3e} {e_enc['max_abs']:12.3e}",
                        flush=True,
                    )
                except Exception as ex:
                    print(
                        f"{amp:8.3g} {dst:5d} {span:5d} {flag:>8} "
                        f"FAIL {type(ex).__name__}: {ex}",
                        flush=True,
                    )

        if args.also_step:
            print(
                f"\n=== CT step_+1 vs one_jump from land={land} (amp=1) ===",
                flush=True,
            )
            print(
                f"{'dst':>5} {'span':>5} {'one_jump':>14} {'step_+1':>14}",
                flush=True,
            )
            msg = make_msg(engine, 1.0, seed=args.seed)
            for dst in range(land + 1, dst_max + 1):
                span = dst - land
                ct0 = engine.encodecrypt(msg, pk, level=land)
                pre = np.asarray(
                    engine.decrode(ct0, sk, is_real=True), dtype=np.float64
                )
                try:
                    e_j = _err(
                        np.asarray(
                            engine.decrode(engine.level_up(ct0, dst), sk, is_real=True),
                            dtype=np.float64,
                        ),
                        pre,
                    )
                    cur = ct0
                    for s in range(1, span + 1):
                        cur = engine.level_up(cur, land + s)
                    e_s = _err(
                        np.asarray(
                            engine.decrode(cur, sk, is_real=True), dtype=np.float64
                        ),
                        pre,
                    )
                    print(
                        f"{dst:5d} {span:5d} {e_j['max_abs']:14.3e} "
                        f"{e_s['max_abs']:14.3e}",
                        flush=True,
                    )
                except Exception as ex:
                    print(f"{dst:5d} {span:5d} FAIL {ex}", flush=True)

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
