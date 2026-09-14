#!/usr/bin/env python3
"""
level_up / PT-at-15 误差表（无 bootstrap）。

Liberate ``level_up`` 只接受 CT。本脚本测两块：

  CT) encodecrypt@15 → level_up 到 dst → decrypt
      误差：vs pre-up decrypt、vs 明文 msg

  PT) encode@15（权重想定）在更高 CT level 上使用：
      - encode@L / decode@L（同 level 编解码底噪）
      - encode@15 / decode@dst（故意错 correction，对照）
      - encode@15 × CT(ones@dst) 经 pt_ct_mult 再 decrypt（生产 tile 路径）

Usage:
  python3 HE/thor/audit_level_up_pt_ct.py
  python3 HE/thor/audit_level_up_pt_ct.py --amps 0.1 1 3 10 --dsts 16 18 21 23
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
    ap.add_argument(
        "--amps",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 3.0, 10.0],
        help="message half-range (±amp)",
    )
    ap.add_argument(
        "--dsts",
        type=int,
        nargs="+",
        default=[16, 17, 18, 19, 20, 21, 22, 23],
        help="level_up / use destinations (from land 15)",
    )
    ap.add_argument("--land", type=int, default=15, help="encode/encrypt start level")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    land = int(args.land)
    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    print(
        f"  ready ({time.time() - t0:.1f}s) num_levels={nl} land={land}",
        flush=True,
    )
    print(
        "  note: Liberate level_up is CT-only; PT 'up' = tile via pt_ct_mult "
        "to CT.level_calc",
        flush=True,
    )

    dsts = [d for d in args.dsts if land < d < nl]
    amps = [float(a) for a in args.amps]

    # ------------------------------------------------------------------ CT
    print("\n=== CT) encodecrypt@land → level_up → decrypt ===", flush=True)
    print(
        f"{'amp':>8} {'dst':>5} {'span':>5} "
        f"{'vs_pre max|e|':>14} {'vs_pre rel':>12} "
        f"{'vs_msg max|e|':>14} {'vs_msg rel':>12} "
        f"{'enc_only max|e|':>16}",
        flush=True,
    )
    for amp in amps:
        msg = make_msg(engine, amp, seed=args.seed)
        ct0 = engine.encodecrypt(msg, pk, level=land)
        pre = np.asarray(engine.decrode(ct0, sk, is_real=True), dtype=np.float64)
        e_enc = _err(pre, msg)
        for dst in dsts:
            ct1 = engine.level_up(ct0, int(dst))
            post = np.asarray(engine.decrode(ct1, sk, is_real=True), dtype=np.float64)
            e_pre = _err(post, pre)
            e_msg = _err(post, msg)
            print(
                f"{amp:8.3g} {dst:5d} {dst - land:5d} "
                f"{e_pre['max_abs']:14.3e} {e_pre['max_rel']:12.3e} "
                f"{e_msg['max_abs']:14.3e} {e_msg['max_rel']:12.3e} "
                f"{e_enc['max_abs']:16.3e}",
                flush=True,
            )

    # ------------------------------------------------------------------ PT encode/decode
    print(
        "\n=== PT-A) encode@L / decode@L vs msg (same-level; no level_up API) ===",
        flush=True,
    )
    print(
        f"{'amp':>8} {'L':>5} {'max|e|':>14} {'rel':>12}",
        flush=True,
    )
    levels_pt = sorted(set([land] + dsts))
    for amp in amps:
        msg = make_msg(engine, amp, seed=args.seed)
        for L in levels_pt:
            pt = engine.encode(msg, level=int(L))
            got = np.asarray(engine.decode(pt, level=int(L), is_real=True), dtype=np.float64)
            e = _err(got, msg)
            print(
                f"{amp:8.3g} {L:5d} {e['max_abs']:14.3e} {e['max_rel']:12.3e}",
                flush=True,
            )

    print(
        "\n=== PT-B) encode@land / decode@dst (mismatched correction; should be bad) ===",
        flush=True,
    )
    print(
        f"{'amp':>8} {'dst':>5} {'max|e|':>14} {'rel':>12}",
        flush=True,
    )
    for amp in amps:
        msg = make_msg(engine, amp, seed=args.seed)
        pt = engine.encode(msg, level=land)
        for dst in dsts:
            got = np.asarray(
                engine.decode(pt, level=int(dst), is_real=True), dtype=np.float64
            )
            e = _err(got, msg)
            print(
                f"{amp:8.3g} {dst:5d} {e['max_abs']:14.3e} {e['max_rel']:12.3e}",
                flush=True,
            )

    # ------------------------------------------------------------------ PT tiled via pt×ct
    print(
        "\n=== PT-C) encode@land × CT(ones@dst) via pt_ct_mult → decrypt "
        "(production tile path) ===",
        flush=True,
    )
    print(
        f"{'amp':>8} {'dst':>5} {'max|e|':>14} {'rel':>12} "
        f"{'ones_enc max|e|':>16}",
        flush=True,
    )
    ones = np.ones(engine.num_slots, dtype=np.float64)
    for amp in amps:
        msg = make_msg(engine, amp, seed=args.seed)
        pt = engine.encode(msg, level=land)
        for dst in dsts:
            ct_one = engine.encodecrypt(ones, pk, level=int(dst))
            e_one = _err(
                np.asarray(engine.decrode(ct_one, sk, is_real=True), dtype=np.float64),
                ones,
            )
            ct_out = engine.pt_ct_mult(pt, ct_one)
            # pt×ct usually needs rescale in HE; Liberate pt_ct_mult returns
            # un-rescaled product — check level and whether rescale is required.
            # Match THOR bias/weight use: often followed by rescale in linear.
            try:
                ct_rs = engine.rescale(ct_out)
                got = np.asarray(
                    engine.decrode(ct_rs, sk, is_real=True), dtype=np.float64
                )
            except Exception:
                got = np.asarray(
                    engine.decrode(ct_out, sk, is_real=True), dtype=np.float64
                )
            e = _err(got, msg)
            print(
                f"{amp:8.3g} {dst:5d} {e['max_abs']:14.3e} {e['max_rel']:12.3e} "
                f"{e_one['max_abs']:16.3e}",
                flush=True,
            )

    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
