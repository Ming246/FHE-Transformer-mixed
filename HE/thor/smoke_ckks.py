#!/usr/bin/env python3
"""Real Liberate ciphertext smoke (THOR logN=16 params)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import diagnose, ensure_thor_path

ensure_thor_path()

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from liberate.fhe.data_struct import DataStruct  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    info = diagnose()
    print(json.dumps(info, indent=2, default=str))
    if info.get("liberate") != "ok" or info.get("thor.ckks") != "ok":
        print("FAIL: Liberate/THOR import broken", file=sys.stderr)
        return 1
    if not info.get("has_scale_primes"):
        print(
            "FAIL: scale_primes.pkl missing — run gen_scale_primes_minimal.py",
            file=sys.stderr,
        )
        return 1

    params = dict(THOR_DEFAULT_PARAMS)
    t0 = time.time()
    print(f"creating CkksEngine params={params} ...", flush=True)
    engine = create_engine(params, verbose=args.verbose)
    print(
        f"engine ready in {time.time()-t0:.1f}s  num_slots={engine.num_slots}",
        flush=True,
    )

    keys = create_keys(engine, with_evk=True)
    sk = keys["sk"]

    m = engine.example(-0.5, 0.5)
    ct = engine.encodecrypt(m, keys["pk"], level=0)
    assert isinstance(ct, DataStruct)
    out = np.asarray(engine.decrode(ct, sk))
    err0 = float(np.max(np.abs(out - m)))
    err0_re = float(np.max(np.abs(out.real - m.real)))
    print(f"enc/dec max|err| complex={err0:.3e}  real={err0_re:.3e}")

    ct_add = engine.add(ct, ct)
    out_add = np.asarray(engine.decrode(ct_add, sk))
    err_add = float(np.max(np.abs(out_add - 2 * m)))
    err_add_re = float(np.max(np.abs(out_add.real - 2 * m.real)))
    print(f"ct+ct     max|err| complex={err_add:.3e}  real={err_add_re:.3e}")

    engine.add_rot_keys_from_sk([1], sk)
    ct_rot = engine.rotate_left(ct, 1)
    out_rot = np.asarray(engine.decrode(ct_rot, sk))
    ref = np.roll(m, -1)
    err_rot = float(np.max(np.abs(out_rot - ref)))
    err_rot_re = float(np.max(np.abs(out_rot.real - ref.real)))
    print(f"rot_left1 max|err| complex={err_rot:.3e}  real={err_rot_re:.3e}")

    hook = BootstrapHook(engine, mode="record")
    _ = hook.refresh("smoke_after_linear", ct_rot)
    print("bootstrap_hook:", hook.summary())
    assert hook.summary()["total_ct_bootstraps"] == 1

    # Official resources: real-part ~1e-5; complex abs can hit ~2e-3 from imag leakage.
    ok = err0_re < 1e-4 and err_add_re < 1e-4 and err_rot_re < 1e-4
    print("PASS" if ok else "FAIL")
    if not ok:
        print(
            "Note: if only complex abs is large (~2e-3) but real is fine, "
            "that is imag leakage — check real-part metrics above.",
            file=sys.stderr,
        )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
