#!/usr/bin/env python3
"""
Library smoke: real Liberate/THOR ``engine.bootstrap``.

Does NOT use ``mult_scalar``. Loads official keys like ``forward.ipynb``:
rotk_dict → bs_key; ordinary rotates from sk.
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

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys, status  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--burn-levels",
        type=int,
        default=8,
        help="square+rescale this many times before bootstrap (consume depth)",
    )
    args = ap.parse_args()

    info = status()
    print("status:", {k: info[k] for k in (
        "liberate", "thor.ckks", "cuda", "gpu",
        "resources_ready", "keys0_ready", "has_scale_primes",
    ) if k in info})
    if not info.get("resources_ready") or not info.get("keys0_ready"):
        print("FAIL: official resources/keys missing", file=sys.stderr)
        return 1

    t0 = time.time()
    print("creating engine...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS), verbose=args.verbose)
    print(f"engine ready {time.time()-t0:.1f}s slots={engine.num_slots}", flush=True)

    t1 = time.time()
    print("loading thor keys (bs_key=rotk_dict + rot from sk)...", flush=True)
    # Bootstrap only needs a few ordinary rots for this smoke; keep load lighter.
    keys = load_thor_keys(
        engine,
        with_bootstrap=True,
        with_rot_deltas=True,
        rot_deltas=[1, 2048],
    )
    sk, pk = keys["sk"], keys["pk"]
    print(f"keys ready {time.time()-t1:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    m = rng.uniform(-0.3, 0.3, engine.num_slots).astype(np.float64)
    ct = engine.encodecrypt(m, pk, level=0)
    level0 = ct.level_calc
    print(f"encrypt level={level0}", flush=True)

    # Burn multiplicative depth so bootstrap is actually needed / exercised.
    ct_work = ct
    for i in range(args.burn_levels):
        ct_work = engine.square(ct_work)
        ct_work = engine.rescale(ct_work)
        print(f"  after square#{i+1} level={ct_work.level_calc}", flush=True)

    before = np.asarray(engine.decrode(ct_work, sk, is_real=True))
    level_before = ct_work.level_calc
    print(f"pre-bootstrap level={level_before}", flush=True)

    hook = BootstrapHook(engine, mode="always")
    t2 = time.time()
    print("running engine.bootstrap ...", flush=True)
    ct_bs = hook.refresh("smoke_bootstrap", ct_work)
    dt = time.time() - t2
    level_after = ct_bs.level_calc
    after = np.asarray(engine.decrode(ct_bs, sk, is_real=True))
    err = float(np.max(np.abs(after - before)))
    mean = float(np.mean(np.abs(after - before)))
    print(
        f"bootstrap done in {dt:.1f}s  level {level_before}->{level_after}  "
        f"real max|err|={err:.3e} mean={mean:.3e}",
        flush=True,
    )
    print("bootstrap_hook:", hook.summary())

    # Library smoke: bootstrap runs and plaintext is preserved vs pre-bootstrap.
    # (Liberate level_calc after bs is a fixed land point; do not require decrease.)
    ok = err < 5e-2
    print("PASS" if ok else "FAIL")
    if not ok:
        print(
            f"hint: err={err:.3e} levels {level_before}->{level_after}",
            file=sys.stderr,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
