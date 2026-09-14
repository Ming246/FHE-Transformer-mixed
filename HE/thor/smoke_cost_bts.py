#!/usr/bin/env python3
"""
Smoke: thor-local ``cost_bts`` — depth vs DualRail bts for a few schemes.

No GPU. Run::

  python3 HE/thor/smoke_cost_bts.py
  python3 HE/thor/smoke_cost_bts.py --task mrpc
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from bts_ops import scheme_all  # noqa: E402
from cost import compute_scheme_cost  # noqa: E402
from cost_bts import (  # noqa: E402
    BOOTSTRAP_DEPTH_BUDGET,
    compute_f_cost,
    compute_scheme_bts_detail,
    f_cost_label,
)


def _print_scheme(label: str, task: str, scheme: list[int]) -> None:
    t0 = time.time()
    depth = int(compute_scheme_cost(task, scheme))
    f_depth = compute_f_cost(task, scheme, cost_mode="depth")
    detail = compute_scheme_bts_detail(task, scheme)
    f_bts = detail.bts_count
    # sanity: API matches detail
    assert compute_f_cost(task, scheme, cost_mode="bts") == f_bts
    dt = time.time() - t0
    print(
        f"[{label}] depth_sum={depth}  f_depth={f_depth}  "
        f"f_bts={f_bts}  {detail.summary()}  ({dt:.2f}s)",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc")
    args = ap.parse_args()

    print(
        f"task={args.task}  budget={BOOTSTRAP_DEPTH_BUDGET}  "
        f"depth_label={f_cost_label('depth')}  "
        f"bts_label={f_cost_label('bts')}",
        flush=True,
    )

    schemes = [
        ("all_low", scheme_all(0)),
        ("all_mid", scheme_all(1)),
        ("all_high", scheme_all(2)),
    ]
    # One mixed: high softmax early, low elsewhere (still valid GeLU groups).
    mixed = scheme_all(0)
    for li in (0, 1, 2):
        mixed[li * 4 + 0] = 2  # softmax high
        mixed[li * 4 + 1] = 1
        mixed[li * 4 + 3] = 1
    schemes.append(("mix_early_sm_high", mixed))

    for name, sch in schemes:
        _print_scheme(name, args.task, sch)

    # Monotonicity: low ≤ mid ≤ high bts (same geom; higher poly → ≥ depth → ≥ bts)
    b_low = compute_f_cost(args.task, scheme_all(0), cost_mode="bts")
    b_mid = compute_f_cost(args.task, scheme_all(1), cost_mode="bts")
    b_high = compute_f_cost(args.task, scheme_all(2), cost_mode="bts")
    mono = b_low <= b_mid <= b_high
    print(
        f"monotonicity low≤mid≤high bts: {b_low}≤{b_mid}≤{b_high} → "
        f"{'OK' if mono else 'BAD'}",
        flush=True,
    )
    # bts should be in a sensible ballpark vs HE hook (~19 CT/layer × 12 ≈ 228;
    # plan counts CT-weighted events, often higher with mid-event bts)
    if b_low < 1 or b_high > 5000:
        print(f"FAIL: bts out of sanity range ({b_low}, {b_high})", flush=True)
        return 2
    if not mono:
        print("FAIL: bts not monotonic in scheme level", flush=True)
        return 2
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
