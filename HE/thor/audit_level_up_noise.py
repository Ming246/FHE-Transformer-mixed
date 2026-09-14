#!/usr/bin/env python3
"""
level_up 误差专项（无 bootstrap 噪声混入的主表；可选从真 bts 落点出发）。

计划（本脚本实现）:
  A) 从不同起点 level_calc 各 up **一次**（span=1）→ 看起点是否影响
  B) 从固定起点连续 / 一次跳多步（span=1,2,4,8）→ 看 up 次数/跨度
  C) 从真 bootstrap 落点 (lc=15) 再 up 不同跨度 → 贴近生产「bts 后再 discard」
  D) 对照：同消息仅加密解密（无 level_up）

误差一律 decrypt(after) − decrypt(before)，消掉编码差。

Usage:
  python3 HE/thor/audit_level_up_noise.py              # 含真 bts 落点段
  python3 HE/thor/audit_level_up_noise.py --skip-bts   # 只做纯 level_up（快）
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
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument(
        "--start-levels",
        type=int,
        nargs="+",
        default=[0, 5, 10, 15, 20, 24],
        help="A: start level_calc for single +1 up",
    )
    ap.add_argument(
        "--spans",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="B/C: level_up spans (≤ MAX_LEVEL_UP_SPAN)",
    )
    ap.add_argument("--base-for-spans", type=int, default=0, help="B: fixed start lc")
    ap.add_argument(
        "--skip-bts",
        action="store_true",
        help="skip section C (no rotk load if also no other need — still load rot if not skip)",
    )
    args = ap.parse_args()

    need_bts = not args.skip_bts
    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=need_bts,
        with_rot_deltas=True,
        rot_deltas=[1, 2048],
    )
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    amp = float(args.amp)
    print(
        f"  ready ({time.time() - t0:.1f}s) num_levels={nl} "
        f"MAX_LEVEL_UP_SPAN={MAX_LEVEL_UP_SPAN} amp={amp:g}",
        flush=True,
    )

    msg = make_msg(engine, amp, seed=0)

    # ---- D encode control ----
    print("\n=== D) encode only ===", flush=True)
    ct = engine.encodecrypt(msg, pk, level=0)
    e = _err(_dec(engine, ct, sk), msg)
    print(
        f"  enc@0 vs msg: max|err|={e['max_abs']:.3e}  rel={e['max_rel']:.3e}",
        flush=True,
    )

    # ---- A: different starts, up once ----
    print("\n=== A) different start lc → level_up +1 once ===", flush=True)
    print(
        f"{'start':>6} {'dst':>5} {'rem0':>5} {'rem1':>5} "
        f"{'max|err|':>10} {'rel':>10}",
        flush=True,
    )
    for lc0 in args.start_levels:
        if lc0 < 0 or lc0 + 1 >= nl:
            print(f"{lc0:6d}  SKIP (out of range)", flush=True)
            continue
        ct = engine.encodecrypt(msg, pk, level=int(lc0))
        pre = _dec(engine, ct, sk)
        ct2 = safe_level_up(engine, ct, int(lc0) + 1)
        post = _dec(engine, ct2, sk)
        e = _err(post, pre)
        print(
            f"{lc0:6d} {lc0+1:5d} {_rem(engine, ct):5d} {_rem(engine, ct2):5d} "
            f"{e['max_abs']:10.3e} {e['max_rel']:10.3e}",
            flush=True,
        )

    # ---- B: fixed start, various spans (one jump) ----
    base = int(args.base_for_spans)
    print(
        f"\n=== B) from lc={base}, one jump of span S "
        f"(vs successive +1 × S) ===",
        flush=True,
    )
    print(
        f"{'span':>5} {'mode':>10} {'dst':>5} {'max|err|':>10} {'rel':>10}",
        flush=True,
    )
    for span in args.spans:
        if span < 1 or (
            MAX_LEVEL_UP_SPAN is not None and span > MAX_LEVEL_UP_SPAN
        ):
            print(f"{span:5d} SKIP (span not in 1..{MAX_LEVEL_UP_SPAN})", flush=True)
            continue
        dst = base + span
        if dst >= nl:
            print(f"{span:5d} SKIP (dst>={nl})", flush=True)
            continue
        # one jump
        ct = engine.encodecrypt(msg, pk, level=base)
        pre = _dec(engine, ct, sk)
        ct_j = safe_level_up(engine, ct, dst)
        e_j = _err(_dec(engine, ct_j, sk), pre)
        print(
            f"{span:5d} {'one_jump':>10} {dst:5d} "
            f"{e_j['max_abs']:10.3e} {e_j['max_rel']:10.3e}",
            flush=True,
        )
        # successive +1
        ct = engine.encodecrypt(msg, pk, level=base)
        pre = _dec(engine, ct, sk)
        cur = ct
        for s in range(span):
            cur = safe_level_up(engine, cur, base + s + 1)
        e_s = _err(_dec(engine, cur, sk), pre)
        print(
            f"{span:5d} {'step_+1':>10} {dst:5d} "
            f"{e_s['max_abs']:10.3e} {e_s['max_rel']:10.3e}",
            flush=True,
        )

    # ---- C: from real bootstrap land ----
    if need_bts:
        print(
            "\n=== C) real bootstrap → land lc=15, then level_up span S ===",
            flush=True,
        )
        print(
            f"{'span':>5} {'dst':>5} {'max|err| vs post-bs':>20} {'rel':>10} "
            f"{'| vs pre-bs cumul':>18}",
            flush=True,
        )
        ct = engine.encodecrypt(msg, pk, level=20)
        pre_bs = _dec(engine, ct, sk)
        t1 = time.time()
        ct_bs = engine.bootstrap(ct)
        print(f"  bootstrap done ({time.time()-t1:.0f}s) lc={ct_bs.level_calc}", flush=True)
        land = int(ct_bs.level_calc)
        post_bs = _dec(engine, ct_bs, sk)
        e_bs = _err(post_bs, pre_bs)
        print(
            f"  bts alone vs pre: max|err|={e_bs['max_abs']:.3e}  "
            f"rel={e_bs['max_rel']:.3e}",
            flush=True,
        )
        for span in args.spans:
            if span < 1 or (
                MAX_LEVEL_UP_SPAN is not None and span > MAX_LEVEL_UP_SPAN
            ):
                continue
            dst = land + span
            if dst >= nl:
                print(f"{span:5d} SKIP dst={dst}", flush=True)
                continue
            ct_up = safe_level_up(engine, ct_bs, dst)
            post = _dec(engine, ct_up, sk)
            e_vs_land = _err(post, post_bs)
            e_vs_pre = _err(post, pre_bs)
            print(
                f"{span:5d} {dst:5d} {e_vs_land['max_abs']:20.3e} "
                f"{e_vs_land['max_rel']:10.3e} {e_vs_pre['max_abs']:18.3e}",
                flush=True,
            )
    else:
        print("\n=== C) skipped (--skip-bts) ===", flush=True)

    print("PASS (level_up noise study finished)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
