#!/usr/bin/env python3
"""
Stepwise GeLU HE diagnosis: where does error blow up?

Compares three flows on x ∈ [x_lo, x_hi] (physical), t = x/C:
  1. HE       — Liberate CT ops, decrypt after each step
  2. sim      — plaintext mirror of HE (PS-tree + same affine/composite)
  3. poly     — gelu_poly Clenshaw reference (production plaintext)

**No bootstrap**: encrypt at ``want_rem`` entry level (unit-test style).

Usage::

  python3 HE/thor/test_gelu_he_stepwise.py
  python3 HE/thor/test_gelu_he_stepwise.py --x-lo -3.5 --x-hi 3.5 --n 1000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
sys.path[:0] = [_HERE, _HE, _REPO]
from path_setup import ensure_thor_path

ensure_thor_path()

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from gelu_he_ct import (  # noqa: E402
    _ct_add,
    _ct_add_scalar,
    _ct_mul,
    _ct_scale,
    affine_to_z_ct,
    cheb_eval_ct,
    cheb_eval_ps_tree_ct,
)
from gelu_poly import GeluPolyEvaluator, gelu_config_for_layer  # noqa: E402
from slot_gelu_he import (  # noqa: E402
    GeluHeParams,
    affine_to_z_slots,
    cheb_eval_ps_tree_slots,
    cheb_eval_slots,
)


def _dec(engine, sk, ct, n: int) -> np.ndarray:
    v = np.real(np.asarray(engine.decrode(ct, sk), dtype=np.complex128))
    return np.asarray(v, dtype=np.float64).ravel()[:n]


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.abs(a - b)
    return float(np.max(d)), float(np.mean(d))


@dataclass
class StepTrace:
    name: str
    he: np.ndarray
    sim: np.ndarray
    poly: np.ndarray


def _gelu_poly_stepwise(x_phys: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """Clenshaw composite (gelu_poly production path)."""
    from gelu_poly import _affine_to_z, _cheb_eval, _coeffs_to_f64

    device = torch.device("cpu")
    x64 = torch.tensor(x_phys, dtype=torch.float64)
    t = (x64 / cfg["C"]).numpy()
    f1_c = _coeffs_to_f64(cfg["f1_cheb_coeffs"], device)
    f2_c = _coeffs_to_f64(cfg["f2_cheb_coeffs"], device)
    a1, b1 = cfg["f1_domain"]
    a2, b2 = cfg["f2_domain"]
    f1_z = _affine_to_z(torch.tensor(t), a1, b1).numpy()
    f1 = _cheb_eval(f1_c, torch.tensor(t), cfg["f1_domain"]).numpy()
    f2_z = _affine_to_z(torch.tensor(f1), a2, b2).numpy()
    f2 = _cheb_eval(f2_c, torch.tensor(f1), cfg["f2_domain"]).numpy()
    y = f1 + f2
    half = y + 0.5
    x_scaled = t * cfg["C"]  # = x_phys
    out = x_scaled * half
    return {
        "t_in": t,
        "f1_z": f1_z,
        "f1": f1,
        "f2_affine_z": f2_z,
        "f2": f2,
        "y": y,
        "half_plus": half,
        "x_phys": x_scaled,
        "gelu_out": out,
    }


def _gelu_sim_stepwise(t_in: np.ndarray, params: GeluHeParams) -> dict[str, np.ndarray]:
    """Plaintext mirror of gelu_poly_ct (PS-tree, same domains)."""
    assert params.f1_cheb_coeffs and params.f1_domain
    assert params.f2_cheb_coeffs and params.f2_domain
    f1_z = affine_to_z_slots(t_in, params.f1_domain)
    f1 = cheb_eval_ps_tree_slots(params.f1_cheb_coeffs, f1_z)
    f2_z = affine_to_z_slots(f1, params.f2_domain)
    f2 = cheb_eval_ps_tree_slots(params.f2_cheb_coeffs, f2_z)
    y = f1 + f2
    half = y + 0.5
    x_phys = t_in * float(params.C)
    out = x_phys * half
    return {
        "t_in": t_in,
        "f1_z": f1_z,
        "f1": f1,
        "f2_affine_z": f2_z,
        "f2": f2,
        "y": y,
        "half_plus": half,
        "x_phys": x_phys,
        "gelu_out": out,
    }


def _gelu_he_stepwise(
    engine, sk, ct_in, params: GeluHeParams, n: int
) -> dict[str, np.ndarray]:
    """HE gelu_poly_ct with decrypt after each step."""
    assert params.f1_cheb_coeffs and params.f1_domain
    assert params.f2_cheb_coeffs and params.f2_domain
    t = ct_in
    f1_z_ct = affine_to_z_ct(engine, t, params.f1_domain)
    f1 = cheb_eval_ct(
        engine, params.f1_cheb_coeffs, t, params.f1_domain, rescale_combo=True
    )
    f2_z_ct = affine_to_z_ct(engine, f1, params.f2_domain)
    f2 = cheb_eval_ps_tree_ct(
        engine,
        params.f2_cheb_coeffs,
        f2_z_ct,
        rescale_combo=True,
    )
    y = _ct_add(engine, f1, f2)
    half = _ct_add_scalar(engine, y, 0.5)
    x_phys = _ct_scale(engine, t, float(params.C))
    out = _ct_mul(engine, x_phys, half)
    return {
        "t_in": _dec(engine, sk, t, n),
        "f1_z": _dec(engine, sk, f1_z_ct, n),
        "f1": _dec(engine, sk, f1, n),
        "f2_affine_z": _dec(engine, sk, f2_z_ct, n),
        "f2": _dec(engine, sk, f2, n),
        "y": _dec(engine, sk, y, n),
        "half_plus": _dec(engine, sk, half, n),
        "x_phys": _dec(engine, sk, x_phys, n),
        "gelu_out": _dec(engine, sk, out, n),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--x-lo", type=float, default=-3.5)
    ap.add_argument("--x-hi", type=float, default=3.5)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--want-rem", type=int, default=14)
    args = ap.parse_args()

    params = GeluHeParams.from_gelu_poly(args.layer, level=args.level)
    cfg = gelu_config_for_layer(args.layer, args.level)
    C = float(params.C)
    n_use = int(args.n)

    x_phys = np.linspace(float(args.x_lo), float(args.x_hi), n_use, dtype=np.float64)
    t_in = x_phys / C

    print(
        f"stepwise GeLU  L{args.layer} high  {params.scheme_name}  C={C}\n"
        f"x∈[{args.x_lo},{args.x_hi}]  n={n_use}",
        flush=True,
    )

    poly = _gelu_poly_stepwise(x_phys, cfg)
    sim = _gelu_sim_stepwise(t_in, params)

    print("create engine + keys (no bts); encodecrypt @ want_rem ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=False)
    sk, pk = keys["sk"], keys["pk"]
    n_slots = int(engine.num_slots)
    if n_use > n_slots:
        raise SystemExit(f"n={n_use} > num_slots={n_slots}")
    from bts_ops import remaining_to_level_calc  # noqa: E402

    msg = np.zeros(n_slots, dtype=np.float64)
    msg[:n_use] = t_in
    # Unit test: start already at GeLU entry rem — no bootstrap.
    target_lc = remaining_to_level_calc(int(args.want_rem), engine.num_levels)
    ct = engine.encodecrypt(msg, pk, level=target_lc)
    print(
        f"  ready ({time.time()-t0:.1f}s)  level={ct.level_calc}  "
        f"rem={engine.num_levels - ct.level_calc}  (no bts)",
        flush=True,
    )

    print("run HE stepwise ...", flush=True)
    t1 = time.time()
    he = _gelu_he_stepwise(engine, sk, ct, params, n_use)
    print(f"  done ({time.time()-t1:.1f}s)", flush=True)

    # sanity: sim vs poly at t_in should match slot twin
    ev = GeluPolyEvaluator(args.layer, args.level)
    poly_end = ev(torch.from_numpy(x_phys)).numpy()
    sm_end = x_phys * (0.5 + sim["y"])
    print(
        f"sanity end: max|sim-poly|={_err(sm_end, poly_end)[0]:.3e}  "
        f"max|sim-slot|={_err(sm_end, x_phys*(0.5+sim['y']))[0]:.3e}",
        flush=True,
    )

    step_order = [
        "t_in",
        "f1_z",
        "f1",
        "f2_affine_z",
        "f2",
        "y",
        "half_plus",
        "x_phys",
        "gelu_out",
    ]
    labels = {
        "t_in": "input t=x/C",
        "f1_z": "f1 affine→z (≈t)",
        "f1": "f1 cheb_eval_ct",
        "f2_affine_z": "f2 affine→z",
        "f2": "f2 cheb_eval (PS-tree)",
        "y": "y = f1+f2",
        "half_plus": "0.5+y",
        "x_phys": "x·C",
        "gelu_out": "x·(0.5+y)",
    }

    print(
        f"\n{'step':22} {'max|HE-sim|':>12} {'max|HE-poly|':>12} "
        f"{'max|sim-poly|':>12}  diagnosis",
        flush=True,
    )
    print("-" * 78, flush=True)

    first_he_sim_spike = None
    first_he_poly_spike = None
    prev_he_sim = 0.0
    prev_he_poly = 0.0

    for key in step_order:
        h = he[key][:n_use]
        s = sim[key][:n_use]
        p = poly[key][:n_use]
        es, ms = _err(h, s)
        ep, mp = _err(h, p)
        esp, msp = _err(s, p)

        diag = []
        if esp > 1e-10:
            diag.append("sim≠poly(Clenshaw)")
        if es > max(1e-6, 10 * prev_he_sim + 1e-6):
            diag.append("HE↗vs sim")
            if first_he_sim_spike is None:
                first_he_sim_spike = key
        if ep > max(1e-6, 10 * prev_he_poly + 1e-6):
            diag.append("HE↗vs poly")
            if first_he_poly_spike is None:
                first_he_poly_spike = key
        if es < 1e-4 and ep > 0.01:
            diag.append("approx not HE")
        if not diag:
            diag.append("ok")

        print(
            f"{labels[key]:22} {es:12.3e} {ep:12.3e} {esp:12.3e}  {','.join(diag)}",
            flush=True,
        )
        prev_he_sim = es
        prev_he_poly = ep

    print("\n--- cheb_eval_ct sub-breakdown (f1 / f2) ---", flush=True)
    print(
        f"f1: max|HE-sim|={_err(he['f1'], sim['f1'])[0]:.3e}  "
        f"max|sim-poly|={_err(sim['f1'], poly['f1'])[0]:.3e}  "
        f"(f1 domain={params.f1_domain})",
        flush=True,
    )
    print(
        f"f2 affine: max|HE-sim|={_err(he['f2_affine_z'], sim['f2_affine_z'])[0]:.3e}  "
        f"max|sim-poly|={_err(sim['f2_affine_z'], poly['f2_affine_z'])[0]:.3e}  "
        f"(f2 domain={params.f2_domain})",
        flush=True,
    )
    print(
        f"f2 eval: max|HE-sim|={_err(he['f2'], sim['f2'])[0]:.3e}  "
        f"max|sim-poly|={_err(sim['f2'], poly['f2'])[0]:.3e}",
        flush=True,
    )

    print("\n--- conclusion ---", flush=True)
    if first_he_sim_spike:
        print(
            f"First HE intrinsic jump (HE vs sim): after **{labels[first_he_sim_spike]}**",
            flush=True,
        )
    else:
        print("No sharp HE-vs-sim jump; error accumulates gradually.", flush=True)
    gelu_es = _err(he["gelu_out"], sim["gelu_out"])[0]
    gelu_ep = _err(he["gelu_out"], poly_end)[0]
    gelu_sp = _err(sim["gelu_out"], poly_end)[0]
    if gelu_es > 0.05:
        print(f"HE vs sim at output: max={gelu_es:.3e} → CKKS noise dominates.", flush=True)
    elif gelu_sp > 0.01:
        print(f"sim vs poly at output: max={gelu_sp:.3e} → PS-tree vs Clenshaw drift.", flush=True)
    else:
        print(
            f"Output max|HE-poly|={gelu_ep:.3e}; check intermediate steps above.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
