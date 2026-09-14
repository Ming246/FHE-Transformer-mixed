#!/usr/bin/env python3
"""
Two bootstrap error-mitigation experiments (scale_bits=41, self-keygen).

1) Post-bs recovery scalar: α*decrypt(post) vs pre — best α by LS / grid
2) Pre-amplify by s, bootstrap, divide by s — if abs noise ~const, err~1/s

Usage:
  python3 HE/thor/audit_bootstrap_scale_tricks.py
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

from liberate.fhe.bootstrapping import ckks_bootstrapping as bs  # noqa: E402

from engine import create_engine  # noqa: E402


def _dec(engine, ct, sk):
    return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64)


def _err(got, ref):
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


def setup_engine():
    params = {
        "logN": 16,
        "scale_bits": 41,
        "num_special_primes": 4,
        "num_scales": None,
        "devices": [0],
        "quantum": "post_quantum",
        "bias_guard": False,
    }
    print("create engine + keys (scale_bits=41 sparse)...", flush=True)
    t0 = time.time()
    engine = create_engine(params)
    sk = bs.create_secret_key_sparse(engine, h=192)
    pk = engine.create_public_key(sk)
    evk = engine.create_evk(sk)
    engine.add_pk(pk)
    engine.add_evk(evk)
    engine.add_gk(engine.create_galois_key(sk))
    engine.add_conj_key(engine.create_conjugation_key(sk))
    rotk = bs.create_bs_key(engine, sk)
    bs.create_cts_stc_const(engine)
    engine.add_bs_key(rotk)
    print(
        f"  ready ({time.time()-t0:.1f}s) num_levels={engine.num_levels}",
        flush=True,
    )
    return engine, sk, pk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", type=float, default=0.3, help="message half-range")
    ap.add_argument("--level", type=int, default=18, help="pre-bs level_calc")
    ap.add_argument(
        "--pre-scales",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        help="scheme2: multiply before bs, divide after",
    )
    args = ap.parse_args()

    engine, sk, pk = setup_engine()
    nl = int(engine.num_levels)
    lc = min(int(args.level), nl - 3)
    amp = float(args.amp)
    rng = np.random.default_rng(0)
    msg = rng.uniform(-amp, amp, engine.num_slots).astype(np.float64)

    # ---- baseline ----
    ct0 = engine.encodecrypt(msg, pk, level=lc)
    pre = _dec(engine, ct0, sk)
    t1 = time.time()
    ct_bs = engine.bootstrap(ct0)
    print(f"baseline bootstrap ({time.time()-t1:.0f}s) land_lc={ct_bs.level_calc}", flush=True)
    post = _dec(engine, ct_bs, sk)
    base = _err(post, pre)
    print(
        f"\n=== BASELINE amp=±{amp:g} pre_lc={lc} ===\n"
        f"  max|err|={base['max_abs']:.3e}  mean={base['mean_abs']:.3e}  "
        f"rel={base['max_rel']:.3e}  ls={base['ls']:.6g}",
        flush=True,
    )

    # ---- (1) post-bs recovery scalar ----
    print("\n=== (1) post-bs × α (plaintext rescale of decrypt) ===", flush=True)
    # LS optimal: α = <post,pre>/<post,post>
    alpha_ls = float(np.dot(post, pre) / (np.dot(post, post) + 1e-30))
    e_ls = _err(alpha_ls * post, pre)
    print(
        f"  α_LS={alpha_ls:.8f}: max|err|={e_ls['max_abs']:.3e}  "
        f"rel={e_ls['max_rel']:.3e}  (vs base {base['max_abs']:.3e})",
        flush=True,
    )
    # small grid around 1
    best = (1.0, base["max_abs"])
    for a in np.linspace(0.98, 1.02, 41):
        ea = _err(a * post, pre)["max_abs"]
        if ea < best[1]:
            best = (float(a), ea)
    print(
        f"  α_grid_best={best[0]:.6f}: max|err|={best[1]:.3e}  "
        f"improvement={base['max_abs']/best[1]:.3f}x",
        flush=True,
    )
    # also try applying α via mult_scalar on CT then decrypt
    if int(ct_bs.level_calc) < nl - 1:
        ct_a = engine.mult_scalar(ct_bs, alpha_ls)
        post_a = _dec(engine, ct_a, sk)
        e_cta = _err(post_a, pre)
        print(
            f"  CT mult_scalar(α_LS): max|err|={e_cta['max_abs']:.3e}  "
            f"rel={e_cta['max_rel']:.3e}",
            flush=True,
        )
    else:
        print("  CT mult_scalar skipped (no rem after land)", flush=True)

    # ---- (2) pre-amplify s, bs, /s ----
    print(
        "\n=== (2) pre ×s → bootstrap → /s  (hope abs noise / s) ===",
        flush=True,
    )
    print(
        f"{'s':>8} {'max|err|':>10} {'mean':>10} {'rel':>10} {'ls':>10} {'vs_base':>8}",
        flush=True,
    )
    for s in args.pre_scales:
        s = float(s)
        # Encrypt amplified message (cleaner than mult_scalar before bs if rem tight)
        msg_s = msg * s
        # guard: keep |msg_s| not insane for encode
        if np.max(np.abs(msg_s)) > 1e3:
            print(f"{s:8g} SKIP |msg*s| too large", flush=True)
            continue
        ct = engine.encodecrypt(msg_s, pk, level=lc)
        pre_s = _dec(engine, ct, sk)
        ct2 = engine.bootstrap(ct)
        # divide after bs on CT if rem allows, else plaintext
        if int(ct2.level_calc) < nl - 1:
            ct2 = engine.mult_scalar(ct2, 1.0 / s)
            got = _dec(engine, ct2, sk)
            # compare to original pre (unscaled): use pre from baseline msg
            # Better: compare to pre_s/s ≈ msg
            ref = pre_s / s
        else:
            got = _dec(engine, ct2, sk) / s
            ref = pre_s / s
        e = _err(got, ref)
        ratio = base["max_abs"] / (e["max_abs"] + 1e-30)
        print(
            f"{s:8g} {e['max_abs']:10.3e} {e['mean_abs']:10.3e} "
            f"{e['max_rel']:10.3e} {e['ls']:10.6g} {ratio:8.2f}x",
            flush=True,
        )

    # Also: same s via CT mult_scalar before bs (consumes 1 level)
    print(
        "\n=== (2b) CT: mult_scalar(s) → bs → mult_scalar(1/s) ===",
        flush=True,
    )
    print(
        f"{'s':>8} {'max|err|':>10} {'rel':>10} {'vs_base':>8}",
        flush=True,
    )
    for s in args.pre_scales:
        s = float(s)
        if s == 1.0:
            continue
        ct = engine.encodecrypt(msg, pk, level=lc)
        if int(ct.level_calc) >= nl - 1:
            print(f"{s:8g} SKIP no rem for pre-scale", flush=True)
            continue
        pre_ref = _dec(engine, ct, sk)
        ct = engine.mult_scalar(ct, s)
        ct2 = engine.bootstrap(ct)
        if int(ct2.level_calc) >= nl - 1:
            got = _dec(engine, ct2, sk) / s
        else:
            got = _dec(engine, engine.mult_scalar(ct2, 1.0 / s), sk)
        e = _err(got, pre_ref)
        ratio = base["max_abs"] / (e["max_abs"] + 1e-30)
        print(
            f"{s:8g} {e['max_abs']:10.3e} {e['max_rel']:10.3e} {ratio:8.2f}x",
            flush=True,
        )

    print("\nPASS (bootstrap scale-trick study)")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
