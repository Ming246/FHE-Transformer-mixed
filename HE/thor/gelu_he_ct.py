#!/usr/bin/env python3
"""
Liberate DualRail GeLU — repo Chebyshev coeffs, PS-tree eval on Chebyshev basis.

I/O matches THOR ``he_gelu``: ciphertext array shape ``(2, 8)``.

Eval (aligned with ``slot_gelu_he`` / ``nolinear/gelu_chebyshev``):
  Build T_0..T_n via T_{a+b}=2 T_a T_b - T_|a-b| (depth ≈ ⌈log₂ n⌉), then
  ∑ c_k T_k with one lazy rescale.   Composite pack path:
    ``gelu_f1_pack`` → ``gelu_f2_pack`` → ``gelu_y_pack`` → ``gelu_reconstruct_pack``
  (f1 / f2 / reconstruct = separate micro-events). FC1 bake ``/C``; restore ``×C``.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
if _HE not in sys.path:
    sys.path.append(_HE)
if _REPO not in sys.path:
    sys.path.append(_REPO)

from slot_gelu_he import GeluHeParams  # noqa: E402


def _ct_scale(engine, ct, s: float, *, rescale: bool = True):
    if s == 1.0:
        return ct
    if float(s) == int(s):
        return engine.mult_int_scalar(ct, int(s))
    return engine.mult_scalar(ct, float(s), rescale=rescale)


def _ct_add_scalar(engine, ct, s: float):
    if s == 0.0:
        return ct
    return engine.add_scalar(ct, float(s))


def _ct_add(engine, a, b):
    if hasattr(engine, "auto_cc_add"):
        return engine.auto_cc_add(a, b)
    return engine.add(a, b)


def _ct_sub(engine, a, b):
    if hasattr(engine, "auto_cc_sub"):
        return engine.auto_cc_sub(a, b)
    if hasattr(engine, "sub"):
        return engine.sub(a, b)
    return engine.cc_sub(a, b)


def _ct_mul(engine, a, b):
    """ct×ct with auto level + rescale (Liberate)."""
    if a is b:
        return engine.square(a)
    return engine.auto_ct_ct_mult(a, b)


def _ct_double(engine, ct):
    """×2 without a multiply: add(ct, ct)."""
    return _ct_add(engine, ct, ct)


def _domain_is_unit(domain: tuple[float, float], *, tol: float = 1e-12) -> bool:
    a, b = float(domain[0]), float(domain[1])
    return abs(a + 1.0) <= tol and abs(b - 1.0) <= tol


def affine_to_z_ct(engine, x, domain: tuple[float, float]):
    """z = (2x-a-b)/(b-a). Free when domain is already [-1, 1]."""
    if _domain_is_unit(domain):
        return x
    a, b = float(domain[0]), float(domain[1])
    denom = b - a
    if denom == 0.0:
        raise ValueError(f"invalid Chebyshev domain {domain}")
    # OpenFHE: y = -1 + 2(x-a)/(b-a) = α x - (1+β), α=2/(b-a), β=aα
    alpha = 2.0 / denom
    beta = a * alpha
    z = _ct_scale(engine, x, alpha)  # rescale=True — one level
    return _ct_add_scalar(engine, z, -1.0 - beta)


def _build_Tk_balanced_tree_ct(engine, z, max_k: int) -> list[Any]:
    """T0=1, T1=z, T_{a+b}=2 Ta Tb - T_|a-b|."""
    T: list[Any | None] = [None] * (max_k + 1)

    def build(k: int):
        cached = T[k]
        if cached is not None:
            return cached
        if k == 0:
            out = _ct_add_scalar(engine, _ct_scale(engine, z, 0), 1.0)
        elif k == 1:
            out = z
        else:
            a = k // 2
            b = k - a
            ta = build(a)
            tb = build(b)
            tab = build(abs(a - b))
            prod = _ct_mul(engine, ta, tb)
            two_prod = _ct_double(engine, prod)
            out = _ct_sub(engine, two_prod, tab)
        T[k] = out
        return out

    for k in range(max_k + 1):
        build(k)
    return [T[k] for k in range(max_k + 1)]


def _align_level(engine, ct, dst_level: int):
    if int(ct.level_calc) < int(dst_level):
        return engine.level_up(ct, int(dst_level))
    return ct


def _cheb_linear_combo_lazy(
    engine, tk: list[Any], coeffs: list[float], *, do_rescale: bool = True
):
    """
    ∑ c_k T_k with optional single rescale (OpenFHE-style absorption).

    1) level-align all used T_k to the deepest one
    2) mult_scalar(c_k, rescale=False) on each
    3) add
    4) optional engine.rescale (composite shares one rescale after f1+f2)
    """
    used: list[tuple[Any, float]] = [
        (tk[k], float(ck)) for k, ck in enumerate(coeffs) if ck != 0.0
    ]
    if not used:
        z = tk[1] if len(tk) > 1 else tk[0]
        return _ct_add_scalar(engine, _ct_scale(engine, z, 0), 0.0)

    max_lv = max(int(t.level_calc) for t, _ in used)
    out = None
    for t, ck in used:
        t = _align_level(engine, t, max_lv)
        term = _ct_scale(engine, t, ck, rescale=False)
        out = term if out is None else _ct_add(engine, out, term)
    assert out is not None
    if do_rescale:
        return engine.rescale(out)
    return out


def cheb_eval_ps_tree_ct(
    engine,
    coeffs: tuple[float, ...] | list[float],
    z,
    *,
    rescale_combo: bool = True,
):
    c = [float(v) for v in coeffs]
    if not c:
        return _ct_scale(engine, z, 0)
    if len(c) == 1:
        # Constant poly: no T-tree; optional rescale unused.
        return _ct_add_scalar(engine, _ct_scale(engine, z, 0), c[0])
    # Only need T_0..T_deg for nonzero support (still build intermediates).
    deg = max((k for k, ck in enumerate(c) if ck != 0.0), default=0)
    tk = _build_Tk_balanced_tree_ct(engine, z, deg)
    # Pad / trim coeff list to deg
    c = c[: deg + 1]
    return _cheb_linear_combo_lazy(
        engine, tk, c, do_rescale=rescale_combo
    )


def cheb_eval_ct(
    engine,
    coeffs,
    x,
    domain: tuple[float, float],
    *,
    rescale_combo: bool = True,
):
    """Chebyshev on ``domain``: affine → PS-tree ∑ c_k T_k(z)."""
    return cheb_eval_ps_tree_ct(
        engine,
        coeffs,
        affine_to_z_ct(engine, x, domain),
        rescale_combo=rescale_combo,
    )


def _require_composite(params: GeluHeParams, *, where: str) -> None:
    if params.kind != "composite":
        raise ValueError(
            f"{where} expects composite scheme, got kind={params.kind!r} "
            f"({params.scheme_name})"
        )
    assert params.f1_cheb_coeffs is not None and params.f1_domain is not None
    assert params.f2_cheb_coeffs is not None and params.f2_domain is not None


def _map_pack(fn, packs: np.ndarray) -> np.ndarray:
    """Apply ``fn(ct)`` to each DualRail pack of shape ``(2, 8)``."""
    if packs.shape != (2, 8):
        raise ValueError(f"expected DualRail shape (2, 8), got {packs.shape}")
    out = np.full((2, 8), None, dtype=object)
    for i in range(2):
        for j in range(8):
            out[i, j] = fn(packs[i, j])
    return out


def gelu_f1_pack(engine, t: np.ndarray, params: GeluHeParams) -> np.ndarray:
    """
    Micro-event candidate: f1 Chebyshev on FC1 scale ``t = x_phys/C``.

    ``t`` shape ``(2, 8)`` → ``f1`` shape ``(2, 8)``.
    """
    _require_composite(params, where="gelu_f1_pack")
    return _map_pack(
        lambda ct: cheb_eval_ct(
            engine,
            params.f1_cheb_coeffs,
            ct,
            params.f1_domain,
            rescale_combo=True,
        ),
        t,
    )


def gelu_f2_pack(engine, f1: np.ndarray, params: GeluHeParams) -> np.ndarray:
    """
    Micro-event candidate: f2 Chebyshev on ``f1`` (incl. domain affine).

    ``f1`` shape ``(2, 8)`` → ``f2`` shape ``(2, 8)``.
    """
    _require_composite(params, where="gelu_f2_pack")
    return _map_pack(
        lambda ct: cheb_eval_ct(
            engine,
            params.f2_cheb_coeffs,
            ct,
            params.f2_domain,
            rescale_combo=True,
        ),
        f1,
    )


def gelu_y_pack(engine, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """``y = f1 + f2`` on DualRail ``(2, 8)`` (no rem burn)."""
    if f1.shape != (2, 8) or f2.shape != (2, 8):
        raise ValueError(f"gelu_y_pack expects (2,8), got {f1.shape} {f2.shape}")
    out = np.full((2, 8), None, dtype=object)
    for i in range(2):
        for j in range(8):
            out[i, j] = _ct_add(engine, f1[i, j], f2[i, j])
    return out


def gelu_reconstruct_pack(
    engine,
    t: np.ndarray,
    y: np.ndarray,
    params: GeluHeParams,
    *,
    bts_hook=None,
    bts_prefix: str | None = None,
) -> np.ndarray:
    """
    Micro-event ``gelu_reconstruct``: spine ``y`` → ``half_plus`` → ``× x_phys``.

    - Input spine: ``y = f1+f2`` (shape ``(2, 8)``); output = GeLU
    - Bypass: ``x_phys = t·C``（``×C`` 不计主路 depth；t rem 足以完成 scale）
    - Plan hook ``{prefix}.gelu_reconstruct``：在 ``f1+f2`` 之后、``ct×ct`` 之前刷 ``y``
    - 显式 ``align_bypass_to_main(half_plus, x_phys)``，不依赖 ``_ct_mul`` 对齐
    - Depth = 1（仅 ``_ct_mul(x_phys, half_plus)``）
    """
    from dualrail_bts import plan_refresh_dualrail_ff
    from softmax_he_ct import align_bypass_to_main

    _require_composite(params, where="gelu_reconstruct_pack")
    if t.shape != (2, 8) or y.shape != (2, 8):
        raise ValueError(
            f"reconstruct expects (2,8) t/y, got t={t.shape} y={y.shape}"
        )
    C = float(params.C)

    if bts_prefix and bts_hook is not None:
        recon_ev = f"{bts_prefix}.gelu_reconstruct"
        if bts_hook.should_refresh(recon_ev):
            print(
                f"  [gelu_he_ct] {recon_ev} refresh "
                f"pack_mode={bts_hook.pack_mode(recon_ev)}",
                flush=True,
            )
            y = plan_refresh_dualrail_ff(engine, y, bts_hook, recon_ev)

    out = np.full((2, 8), None, dtype=object)
    for i in range(2):
        for j in range(8):
            half_plus = _ct_add_scalar(engine, y[i, j], 0.5)
            x_phys = _ct_scale(engine, t[i, j], C)
            x_phys = align_bypass_to_main(engine, half_plus, x_phys)
            out[i, j] = _ct_mul(engine, x_phys, half_plus)
    return out


def gelu_poly_pack(
    engine,
    t: np.ndarray,
    params: GeluHeParams,
    *,
    bts_hook=None,
    bts_prefix: str | None = None,
) -> np.ndarray:
    """
    Full DualRail GeLU on shape ``(2, 8)``: f1 → f2 → y → reconstruct.

    Plan hooks:
      - ``{prefix}.gelu_f1`` before：刷 ``t``
      - ``{prefix}.gelu_f2`` before：刷 ``f1``
      - ``{prefix}.gelu_reconstruct``：刷 ``y=f1+f2``；``x_phys`` 旁路对齐后 ct×ct
    """
    from dualrail_bts import plan_refresh_dualrail_ff

    if bts_prefix and bts_hook is not None:
        f1_ev = f"{bts_prefix}.gelu_f1"
        if bts_hook.should_refresh(f1_ev):
            print(
                f"  [gelu_he_ct] {f1_ev} refresh "
                f"pack_mode={bts_hook.pack_mode(f1_ev)}",
                flush=True,
            )
            t = plan_refresh_dualrail_ff(engine, t, bts_hook, f1_ev)

    f1 = gelu_f1_pack(engine, t, params)

    if bts_prefix and bts_hook is not None:
        f2_ev = f"{bts_prefix}.gelu_f2"
        if bts_hook.should_refresh(f2_ev):
            print(
                f"  [gelu_he_ct] {f2_ev} refresh "
                f"pack_mode={bts_hook.pack_mode(f2_ev)}",
                flush=True,
            )
            f1 = plan_refresh_dualrail_ff(engine, f1, bts_hook, f2_ev)

    f2 = gelu_f2_pack(engine, f1, params)
    y = gelu_y_pack(engine, f1, f2)
    return gelu_reconstruct_pack(
        engine,
        t,
        y,
        params,
        bts_hook=bts_hook,
        bts_prefix=bts_prefix,
    )


def gelu_poly_ct(engine, x, params: GeluHeParams, *, reactive_bts: bool = False):
    """
    Single-CT helper (unit tests). Prefer ``gelu_poly_pack`` / stage APIs
    for production DualRail ``(2, 8)``.
    """
    from softmax_he_ct import align_bypass_to_main

    del reactive_bts
    _require_composite(params, where="gelu_poly_ct")
    f1 = cheb_eval_ct(
        engine,
        params.f1_cheb_coeffs,
        x,
        params.f1_domain,
        rescale_combo=True,
    )
    f2 = cheb_eval_ct(
        engine,
        params.f2_cheb_coeffs,
        f1,
        params.f2_domain,
        rescale_combo=True,
    )
    y = _ct_add(engine, f1, f2)
    half_plus = _ct_add_scalar(engine, y, 0.5)
    x_phys = _ct_scale(engine, x, float(params.C))
    x_phys = align_bypass_to_main(engine, half_plus, x_phys)
    return _ct_mul(engine, x_phys, half_plus)


def _refresh_dualrail_pack(
    engine, x: np.ndarray, *, target_level: int | None = None
) -> np.ndarray:
    """
    Bootstrap DualRail (2, 8) like THOR FF:
    pack re+i·im → bootstrap → unpack → ×1/2（bs 后乘，避免耗尽跳过 half）。

    ``target_level`` optional discard after bts; default leave at land (≈15).
    Do not hardcode 21 (old QKV W encode level).
    """
    if x.shape != (2, 8):
        raise ValueError(f"expected (2, 8), got {x.shape}")
    out = np.full((2, 8), None, dtype=object)
    for i in range(8):
        temp = engine.cc_add(x[0, i], engine.imult(x[1, i]))
        temp = engine.bootstrap(temp)
        if target_level is not None and int(temp.level_calc) < int(target_level):
            temp = engine.level_up(temp, int(target_level))
        conj = engine.conjugate(temp)
        out[0, i] = engine.mult_scalar(engine.cc_add(temp, conj), 0.5)
        out[1, i] = engine.mult_scalar(
            engine.imult(engine.cc_sub(conj, temp)), 0.5
        )
    return out


def he_gelu_cheb(
    engine,
    x: np.ndarray,
    params: GeluHeParams | None = None,
    *,
    layer_idx: int = 0,
    level: int = 2,
    input_scale: float = 1.0,
    sk=None,
    initial_btp: bool = False,
    suppress_post_bts: bool = False,
    bts_hook=None,
    bts_prefix: str | None = None,
):
    """
    Drop-in for THOR ``he_gelu(engine, x, ...)``.

    ``x`` shape ``(2, 8)`` at FC1 output scale (``x_phys/C``; FC1 W+b baked ``/C``).
    Runs pack-level ``gelu_poly_pack`` (f1 → f2 → y → reconstruct) with plan hooks.

    ``suppress_post_bts``：plan 路径由 forward 的 ``linear_ff_dense2``
    段前 hook 负责（与 ``linear_ff_dense1`` 同形），避免 reactive bootstrap。
    """
    del sk, initial_btp, input_scale
    if params is None:
        params = GeluHeParams.from_gelu_poly(layer_idx, level=level)
    if x.shape != (2, 8):
        raise ValueError(f"he_gelu_cheb expects shape (2, 8), got {x.shape}")
    if bts_prefix is None:
        bts_prefix = f"L{int(layer_idx)}"

    out = gelu_poly_pack(
        engine,
        x,
        params,
        bts_hook=bts_hook,
        bts_prefix=bts_prefix,
    )
    # Non-plan callers：出口 rem 须够进 dense2；plan 路径用 suppress_post_bts。
    rem = int(engine.num_levels) - int(out[0, 0].level_calc)
    if rem < 3 and not suppress_post_bts:
        raise RuntimeError(
            f"post-GeLU rem={rem}<3; refresh via linear_ff_dense2 "
            "before dense2 (no reactive DualRail)"
        )
    return out


def bind_ff_gelu_cheb(
    thor_ff,
    *,
    layer_idx: int,
    level: int = 2,
    input_scale: float = 1.0,
    suppress_post_bts: bool = False,
    bts_hook=None,
):
    """Replace ``thor_ff.gelu`` with Chebyshev Liberate implementation."""
    from functools import partial

    del input_scale
    params = GeluHeParams.from_gelu_poly(layer_idx, level=level)
    thor_ff.gelu = partial(
        he_gelu_cheb,
        engine=thor_ff.engine,
        params=params,
        layer_idx=layer_idx,
        level=level,
        suppress_post_bts=suppress_post_bts,
        bts_hook=bts_hook,
        bts_prefix=f"L{int(layer_idx)}",
    )
    thor_ff._gelu_cheb_params = params
    return params
