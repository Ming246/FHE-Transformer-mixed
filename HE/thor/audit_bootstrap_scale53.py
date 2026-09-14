#!/usr/bin/env python3
"""
Self-keygen with Liberate ``presets.params['bootstrapping']`` (scale_bits=53)
and measure:
  - post-bootstrap land level_calc / remaining depth
  - bootstrap noise (decrypt post − pre) vs message amplitude

Does NOT use official keys0.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import ensure_thor_path

ensure_thor_path()

from liberate.fhe import presets  # noqa: E402
from liberate.fhe.bootstrapping import ckks_bootstrapping as bs  # noqa: E402

from engine import create_engine  # noqa: E402


def _dec(engine, ct, sk) -> np.ndarray:
    return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64)


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _err(got, ref) -> dict:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    e = np.abs(g - r)
    peak = float(np.max(np.abs(r))) + 1e-30
    denom = float(np.dot(g, g)) + 1e-30
    return {
        "max_abs": float(np.max(e)),
        "max_rel": float(np.max(e) / peak),
        "ls": float(np.dot(g, r) / denom),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amps", type=float, nargs="+", default=[0.3, 3.0, 12.0])
    ap.add_argument(
        "--pre-levels",
        type=int,
        nargs="+",
        default=None,
        help="pre-bs level_calc list (default: mid + near-exhaust)",
    )
    ap.add_argument(
        "--quantum",
        choices=("pre_quantum", "post_quantum"),
        default="post_quantum",
        help="CKKS context quantum flag (preset caches lean post_quantum)",
    )
    args = ap.parse_args()

    params = dict(presets.params["bootstrapping"])
    # Preset omits quantum; Liberate default / scale50 caches use post_quantum.
    params["quantum"] = str(args.quantum)
    params["devices"] = [0]
    print("params:", params, flush=True)

    t0 = time.time()
    print("create engine ...", flush=True)
    engine = create_engine(params)
    nl = int(engine.num_levels)
    print(
        f"  engine ready ({time.time()-t0:.1f}s) num_levels={nl} "
        f"slots={engine.num_slots} scale={engine.scale:.6g} "
        f"(~2^{np.log2(float(engine.scale)):.2f})",
        flush=True,
    )

    print("keygen sparse-sk / pk / evk / gk / conjk ...", flush=True)
    t1 = time.time()
    # Bootstrap path expects sparse Hamming-wt secret (Liberate bs helper).
    sk = bs.create_secret_key_sparse(engine, include_special=True, h=192)
    pk = engine.create_public_key(sk)
    evk = engine.create_evk(sk)
    engine.add_pk(pk)
    engine.add_evk(evk)
    gk = engine.create_galois_key(sk)
    engine.add_gk(gk)
    conjk = engine.create_conjugation_key(sk)
    engine.add_conj_key(conjk)
    print(f"  base keys ({time.time()-t1:.1f}s)", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"  cuda_reserved≈{torch.cuda.memory_reserved(0)/(1024**3):.1f}GiB",
            flush=True,
        )

    print("create_bs_key (heavy) ...", flush=True)
    t2 = time.time()
    rotk_dict = bs.create_bs_key(engine, sk)
    print(
        f"  create_bs_key done ({time.time()-t2:.1f}s) n_keys={len(rotk_dict)}",
        flush=True,
    )
    bs.create_cts_stc_const(engine)
    engine.add_bs_key(rotk_dict)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"  cuda_reserved≈{torch.cuda.memory_reserved(0)/(1024**3):.1f}GiB",
            flush=True,
        )

    # Ordinary rotates used lightly in smoke; generate a couple for safety.
    engine.add_rot_keys_from_sk([1, 2048], sk)

    # ---- land depth ----
    print("\n=== bootstrap land depth ===", flush=True)
    rng = np.random.default_rng(0)
    msg = rng.uniform(-0.3, 0.3, engine.num_slots).astype(np.float64)

    for tag, mk in [
        ("enc@0+burn8", "burn"),
        ("enc@mid", "mid"),
        ("enc@near_end", "end"),
    ]:
        if mk == "burn":
            ct = engine.encodecrypt(msg, pk, level=0)
            for _ in range(min(8, max(0, nl - 3))):
                ct = engine.rescale(engine.square(ct))
        elif mk == "mid":
            lc = max(0, nl // 2)
            ct = engine.encodecrypt(msg, pk, level=lc)
        else:
            lc = max(0, nl - 3)
            ct = engine.encodecrypt(msg, pk, level=lc)
        pre_lc, pre_r = int(ct.level_calc), _rem(engine, ct)
        t = time.time()
        ct2 = engine.bootstrap(ct)
        dt = time.time() - t
        print(
            f"  {tag}: pre lc={pre_lc} rem={pre_r} → "
            f"post lc={ct2.level_calc} rem={_rem(engine, ct2)}  ({dt:.0f}s)",
            flush=True,
        )

    land_lc = int(ct2.level_calc)
    land_rem = _rem(engine, ct2)

    # ---- noise vs amp ----
    pre_levels = args.pre_levels
    if pre_levels is None:
        # a few points above land if possible, plus near exhaust
        pre_levels = sorted(
            {
                max(0, land_lc - 5),
                land_lc,
                min(nl - 2, land_lc + 5),
                max(0, nl - 3),
            }
        )
    print(
        f"\n=== bootstrap noise (post−pre) land=lc{land_lc}/rem{land_rem} ===",
        flush=True,
    )
    print(
        f"{'amp':>6} {'pre_lc':>6} {'pre_rem':>7} {'max|err|':>10} {'rel':>10} {'ls':>8}",
        flush=True,
    )
    for amp in args.amps:
        for lc in pre_levels:
            if lc < 0 or lc >= nl:
                continue
            m = rng.uniform(-amp, amp, engine.num_slots).astype(np.float64)
            ct = engine.encodecrypt(m, pk, level=int(lc))
            pre = _dec(engine, ct, sk)
            ct_bs = engine.bootstrap(ct)
            post = _dec(engine, ct_bs, sk)
            e = _err(post, pre)
            print(
                f"{amp:6g} {int(ct.level_calc):6d} {_rem(engine, ct):7d} "
                f"{e['max_abs']:10.3e} {e['max_rel']:10.3e} {e['ls']:8.4g}",
                flush=True,
            )

    print("\nPASS (scale_bits=53 self-keygen bootstrap study)")
    print(
        f"SUMMARY land_rem={land_rem} land_lc={land_lc} num_levels={nl} "
        f"scale_bits≈{int(round(np.log2(float(engine.scale))))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
