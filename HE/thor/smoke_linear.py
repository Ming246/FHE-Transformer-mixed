#!/usr/bin/env python3
"""
Linear primitive smoke on real Liberate ciphertexts.

1. DualRail pack/unpack (cc_add + imult + conjugate) — THOR hidden 8↔4
2. ``ThorLinearEvaluator.make_rotated_copies`` — 4 → 64 PC-MM inputs
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_setup import ensure_thor_path

ensure_thor_path()

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from geometry import bert_base_complex  # noqa: E402
from thor.linear import ThorLinearEvaluator  # noqa: E402


def main() -> int:
    geom = bert_base_complex()
    params = dict(THOR_DEFAULT_PARAMS)
    t0 = time.time()
    print(f"creating engine {params} ...", flush=True)
    engine = create_engine(params)
    print(f"ready in {time.time()-t0:.1f}s slots={engine.num_slots}", flush=True)
    assert engine.num_slots == geom.num_slots

    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]

    engine.add_rot_keys_from_sk([2048], sk)
    engine.add_conj_key(engine.create_conjugation_key(sk))

    evaluator = ThorLinearEvaluator(engine)
    hook = BootstrapHook(engine, mode="record")
    rng = np.random.default_rng(0)
    noise_tol = 5e-3

    # --- DualRail: pack 2 real CTs → 1 complex → unpack (no mult_scalar; ×2 form) ---
    # Note: mult_scalar/pt×ct is noisy with homemade scale_primes; THOR unpack uses
    # mult_scalar(·, 1/2) which we defer until official resources or a calibrated path.
    re_m = rng.uniform(-0.3, 0.3, engine.num_slots).astype(np.float64)
    im_m = rng.uniform(-0.3, 0.3, engine.num_slots).astype(np.float64)
    ct_re = engine.encodecrypt(re_m, pk, level=0)
    ct_im = engine.encodecrypt(im_m, pk, level=0)
    ct_cplx = engine.cc_add(ct_re, engine.imult(ct_im))
    conj = engine.conjugate(ct_cplx)
    out_re_2 = engine.cc_add(ct_cplx, conj)  # ≈ 2·re
    out_im_2 = engine.imult(engine.cc_sub(conj, ct_cplx))  # ≈ 2·im
    err_re = float(
        np.max(np.abs(np.asarray(engine.decrode(out_re_2, sk)).real - 2 * re_m))
    )
    err_im = float(
        np.max(np.abs(np.asarray(engine.decrode(out_im_2, sk)).real - 2 * im_m))
    )
    print(f"DualRail pack/unpack(×2)  re|err|={err_re:.3e}  im|err|={err_im:.3e}")

    # --- make_rotated_copies on 4 complex CTs ---
    msgs = []
    cts = np.full((geom.ct_hidden_cplx,), None, dtype=object)
    for i in range(geom.ct_hidden_cplx):
        m = (
            rng.uniform(-0.3, 0.3, engine.num_slots)
            + 1j * rng.uniform(-0.3, 0.3, engine.num_slots)
        ).astype(np.complex128)
        msgs.append(m)
        cts[i] = engine.encodecrypt(m, pk, level=0)

    rots = evaluator.make_rotated_copies(cts)
    assert rots.shape == (geom.ct_pc_rot_cplx,)
    err_r0 = float(np.max(np.abs(np.asarray(engine.decrode(rots[0], sk)) - msgs[0])))
    err_r1 = float(
        np.max(np.abs(np.asarray(engine.decrode(rots[1], sk)) - np.roll(msgs[0], -2048)))
    )
    print(f"rotated[0] max|err|={err_r0:.3e}")
    print(f"rotated[1]=rot2048 max|err|={err_r1:.3e}")

    hook.refresh("after_pc_rot", rots, indices=[0, 1])
    print("bootstrap_hook:", hook.summary())

    ok = (
        err_re < noise_tol
        and err_im < noise_tol
        and err_r0 < noise_tol
        and err_r1 < noise_tol
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
