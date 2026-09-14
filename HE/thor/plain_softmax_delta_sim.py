#!/usr/bin/env python3
"""
Plaintext twin of ``softmax_he_ct`` with THOR-style Δ bookkeeping.

Convention (matches ``_DeltaCt`` / HE):
  *stored* ``v`` is what the ciphertext holds;
  *logical* ``= v / delta`` is the quantity plain ``slot_softmax_he`` keeps in CT.

All slot ops mirror ``he_asor_ct`` / ``update_inv_D_fixed`` / ``he_softmax_poly``
without Liberate — no bootstrap / bts (logical identity).

Compare against:
  - ``slot_softmax_he`` (same layout, no separate Δ on inv)
  - ``softmax_poly.thor_softmax`` on decoded dense scores
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    rotsum,
    stockmeyer_poly_slots,
    sum_over_keys,
    thor_exp_slots,
)
from softmax_poly import (  # noqa: E402
    MAX_SUM_SQ_ROUNDS,
    THOR_INPUT_SCALE,
    _check_power_of_two,
    thor_eps2_from_en,
)
from thor_encoder_linear_core import (  # noqa: E402
    ThorConfig,
    add_vec,
    pt_mult,
    rotate_left,
)


def head_mask(num_slots: int, pad: float = 0.0) -> np.ndarray:
    return np.array(
        ([1.0] * 12 + [float(pad)] * 4) * (num_slots // 16), dtype=np.float64
    )


@dataclass
class PlainDeltaCt:
    """Plain twin of ``softmax_he_ct._DeltaCt``."""

    v: np.ndarray
    delta: float = 1.0

    @property
    def logical(self) -> np.ndarray:
        return self.v / float(self.delta)

    def copy(self) -> "PlainDeltaCt":
        return PlainDeltaCt(np.asarray(self.v, dtype=np.float64).copy(), self.delta)


def _neg(v: np.ndarray) -> np.ndarray:
    return pt_mult(np.full_like(v, -1.0), v)


def _add_scalar(v: np.ndarray, c: float) -> np.ndarray:
    return add_vec(v, np.full_like(v, float(c), dtype=np.float64))


def _ct_ct_mult(a: PlainDeltaCt, b: PlainDeltaCt) -> PlainDeltaCt:
    return PlainDeltaCt(pt_mult(a.v, b.v), a.delta * b.delta)


def _mult_int_scalar(dc: PlainDeltaCt, k: int) -> PlainDeltaCt:
    k = int(k)
    return PlainDeltaCt(pt_mult(dc.v, float(k)), dc.delta * float(k))


def _square(dc: PlainDeltaCt) -> PlainDeltaCt:
    return PlainDeltaCt(pt_mult(dc.v, dc.v), dc.delta * dc.delta)


def plain_he_asor_ct(
    numerator: np.ndarray,
    denominator: np.ndarray,
    e0: float,
    max_iters: int,
) -> tuple[np.ndarray, float, float]:
    """Mirror ``he_asor_ct``; returns ``(inv_v, d_delta, en)``."""
    an = PlainDeltaCt(np.asarray(numerator, dtype=np.float64).copy(), 1.0)
    bn = PlainDeltaCt(np.asarray(denominator, dtype=np.float64).copy(), 1.0)
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 must be > 0, got {e0}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters must be ≥ 1, got {max_iters}")

    for _ in range(max_iters):
        kn = 2.0 / (en + 1.0)
        b_temp = PlainDeltaCt(
            _neg(_add_scalar(bn.v, -2.0 / kn * bn.delta)),
            bn.delta,
        )
        an = _ct_ct_mult(an, b_temp)
        an = PlainDeltaCt(an.v, an.delta / (kn**2))
        bn = _ct_ct_mult(bn, b_temp)
        bn = PlainDeltaCt(bn.v, bn.delta / (kn**2))
        en = kn * en * (2.0 - kn * en)

        scale_adjust = int(1.0 / bn.delta / 2**8)
        if scale_adjust > 1:
            # Real CKKS: z+conj(z)=2Re; slot vectors here are real → ×2
            an = PlainDeltaCt(pt_mult(an.v, 2.0), an.delta * 2.0)
            bn = PlainDeltaCt(pt_mult(bn.v, 2.0), bn.delta * 2.0)
            scale_adjust = int(1.0 / bn.delta / 2**8)
        else:
            scale_adjust = 1
        bn = _mult_int_scalar(bn, scale_adjust)
        an = _mult_int_scalar(an, scale_adjust)

    return an.v, float(an.delta), en


def plain_update_inv_D_fixed(
    exp_u: list[np.ndarray],
    mask_packs: list[np.ndarray],
    inv_d: np.ndarray,
    d_delta: float,
    en: float,
    *,
    max_iters: int,
    final_inv: bool = False,
    cfg: ThorConfig,
) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    """Mirror ``update_inv_D_fixed`` (no bts / align_bypass)."""
    inv_dc = PlainDeltaCt(np.asarray(inv_d, dtype=np.float64).copy(), float(d_delta))
    inv_list = [
        pt_mult(np.asarray(m, dtype=np.float64), inv_dc.v)
        for m in mask_packs
    ]

    k = max(int(1.0 / float(d_delta) / 2.0), 1)
    # Index like HE ``update_inv_D_fixed`` (not append 0,4,1,5,…).
    exp_2u: list[np.ndarray | None] = [None] * 8
    for i in range(4):
        twice_exp0 = pt_mult(np.asarray(exp_u[i], dtype=np.float64), 2.0)
        twice_exp1 = pt_mult(np.asarray(exp_u[i + 4], dtype=np.float64), 2.0)
        t0 = pt_mult(pt_mult(twice_exp0, inv_list[i]), float(k))
        t1 = pt_mult(pt_mult(twice_exp1, inv_list[i + 4]), float(k))
        exp_2u[i] = pt_mult(t0, t0)
        exp_2u[i + 4] = pt_mult(t1, t1)

    summation = np.asarray(exp_2u[0], dtype=np.float64).copy()
    for i in range(1, 8):
        summation = add_vec(summation, np.asarray(exp_2u[i], dtype=np.float64))
    summation = rotsum(summation, cfg.slot_stride)

    nslots = int(cfg.num_slots)
    e0_2 = thor_eps2_from_en(en, seq_len=int(cfg.seq_len))
    enc_one = head_mask(nslots, pad=0.0)
    inv_v, d_delta, en = plain_he_asor_ct(
        enc_one, summation, e0_2, max_iters
    )

    masking = np.array(([1.0] * 12 + [0.0] * 4) * (2**11), dtype=np.float64)
    if final_inv:
        short = np.array(([1.0] * 12 + [0.0] * 4) * (2**7), dtype=np.float64)
        masking = np.ones(nslots, dtype=np.float64)
        masking[: short.size] = short
    inv_v = pt_mult(inv_v, masking)

    return exp_2u, inv_v, d_delta, en


def plain_dualrail_softmax_exit(
    exp_u: list[np.ndarray],
    inv_d: np.ndarray,
    d_delta: float,
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """
    Plain twin of ``dualrail_softmax_exit``: 8 real → 128 real α copies.

    Uses stored ``inv_d`` (not ``inv/d_delta``) and ``scale_k`` on exp — same
    as HE before decrypt-side ``2·Re`` / ``2·Im`` unpack.
    """
    if len(exp_u) != 8:
        raise ValueError(f"expected 8 exp packs, got {len(exp_u)}")
    inv_d = np.asarray(inv_d, dtype=np.float64)
    stride = int(cfg.slot_stride)
    rotated_inv = [inv_d.copy()]
    for _ in range(15):
        rotated_inv.append(rotate_left(rotated_inv[-1], -stride))

    scale_k = int(1.0 / (2.0 * float(d_delta))) + 1
    sk = float(scale_k)
    cplx_softmax1: list[np.ndarray] = []
    cplx_softmax2: list[np.ndarray] = []
    for i in range(4):
        exp_cplx = np.asarray(exp_u[i], dtype=np.float64).astype(np.complex128)
        exp_cplx = exp_cplx + 1j * np.asarray(exp_u[i + 4], dtype=np.float64)
        exp_cplx = exp_cplx * sk
        for j in range(16):
            masked = exp_cplx * rotated_inv[j]
            copied = rotsum(np.real(masked), stride) + 1j * rotsum(
                np.imag(masked), stride
            )
            copied_real = np.real(copied + np.conj(copied))
            copied_imag = np.real(1j * (np.conj(copied) - copied))
            cplx_softmax1.append(copied_real)
            cplx_softmax2.append(copied_imag)
    return cplx_softmax1 + cplx_softmax2


def decode_alpha_128_to_packs8(
    alpha_128: list[np.ndarray],
    *,
    extract_gain: float = 2.0,
) -> list[np.ndarray]:
    """
    Decrypt-side unpack: undo ``cc_add(z,conj(z))`` / ``imult(conj(z)-z)`` → 8 packs.
    """
    if len(alpha_128) != 128:
        raise ValueError(f"expected 128 alpha copies, got {len(alpha_128)}")
    inv_g = 1.0 / float(extract_gain)
    packs: list[np.ndarray] = []
    for i in range(4):
        packs.append(np.asarray(alpha_128[i * 16], dtype=np.float64) * inv_g)
    for i in range(4):
        packs.append(np.asarray(alpha_128[64 + i * 16], dtype=np.float64) * inv_g)
    return packs


def _softmax_delta_core(
    score_packs: list[np.ndarray],
    mask_packs: list[np.ndarray],
    cfg: ThorConfig,
    params: SoftmaxHeParams,
    *,
    exp_input_prescaled: bool,
) -> tuple[list[np.ndarray], np.ndarray, float, list[np.ndarray]]:
    """Shared spine through Σy²; returns ``(exp_u, inv_d, d_delta, mask_packs)``."""
    d2 = float(params.delta2)
    n2 = _check_power_of_two("delta2", d2)
    if n2 > MAX_SUM_SQ_ROUNDS:
        raise ValueError(f"delta2={d2} → n2={n2} > MAX_SUM_SQ_ROUNDS")

    exp_u = thor_exp_slots(
        score_packs, mask_packs, params, exp_input_prescaled=exp_input_prescaled
    )

    sigma_exp = np.asarray(exp_u[0], dtype=np.float64).copy()
    for i in range(1, 8):
        sigma_exp = add_vec(sigma_exp, exp_u[i])
    sigma_exp = rotsum(sigma_exp, cfg.slot_stride)

    enc_one = head_mask(int(cfg.num_slots), pad=0.0)
    inv_d, d_delta, en = plain_he_asor_ct(
        enc_one,
        sigma_exp,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
    )

    for r in range(n2):
        iters_r = int(params.asor_max_iters_sum_sq[r])
        if iters_r < 1:
            raise ValueError(f"Σy² round {r} iters={iters_r} invalid")
        exp_u, inv_d, d_delta, en = plain_update_inv_D_fixed(
            exp_u,
            mask_packs,
            inv_d,
            d_delta,
            en,
            max_iters=iters_r,
            final_inv=(r == n2 - 1),
            cfg=cfg,
        )
    return exp_u, inv_d, d_delta, mask_packs


def plain_softmax_normalize_8packs(
    exp_u: list[np.ndarray],
    inv_d: np.ndarray,
    d_delta: float,
    mask_packs: list[np.ndarray],
) -> list[np.ndarray]:
    """
    8-pack 概率：``exp_u * inv`` 的 logical 值（``inv_v / d_delta``）× mask。

    对应 ``update_inv_D`` 末轮 aSOR 后的归一化，不含 DualRail exit 的 ``scale_k``。
    """
    inv_log = np.asarray(inv_d, dtype=np.float64) / float(d_delta)
    out: list[np.ndarray] = []
    for i, m in enumerate(mask_packs):
        y = pt_mult(np.asarray(exp_u[i], dtype=np.float64), inv_log)
        out.append(pt_mult(np.asarray(m, dtype=np.float64), y))
    return out


def he_softmax_delta_sim(
    score_packs: list[np.ndarray],
    mask_packs: list[np.ndarray],
    cfg: ThorConfig,
    params: SoftmaxHeParams,
    *,
    exp_input_prescaled: bool = True,
    full_he_exit: bool = False,
) -> list[np.ndarray]:
    """
    Plaintext HE Softmax twin.

    ``full_he_exit=False``: 8 packs via ``exp × inv/Δ`` (≈ ``thor_softmax``).
    ``full_he_exit=True``: 128 DualRail copies (``dualrail_softmax_exit`` twin).
    """
    if len(score_packs) != 8:
        raise ValueError(f"expected 8 score packs, got {len(score_packs)}")

    exp_u, inv_d, d_delta, mask_packs = _softmax_delta_core(
        score_packs,
        mask_packs,
        cfg,
        params,
        exp_input_prescaled=exp_input_prescaled,
    )
    if full_he_exit:
        return plain_dualrail_softmax_exit(exp_u, inv_d, d_delta, cfg)
    return plain_softmax_normalize_8packs(exp_u, inv_d, d_delta, mask_packs)


def he_softmax_delta_sim_8packs(
    score_packs: list[np.ndarray],
    mask_packs: list[np.ndarray],
    cfg: ThorConfig,
    params: SoftmaxHeParams,
    *,
    exp_input_prescaled: bool = True,
) -> list[np.ndarray]:
    """Alias: 8-pack shortcut (no DualRail exit)."""
    return he_softmax_delta_sim(
        score_packs,
        mask_packs,
        cfg,
        params,
        exp_input_prescaled=exp_input_prescaled,
        full_he_exit=False,
    )


def pack_max_abs_diff(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    err = 0.0
    for x, y in zip(a, b):
        d = np.max(np.abs(np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)))
        err = max(err, float(d))
    return err


def pack_ls_gain(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    cur = np.concatenate([np.ravel(x) for x in a])
    ref = np.concatenate([np.ravel(x) for x in b])
    denom = float(np.dot(cur, cur))
    if denom < 1e-30:
        return float("nan")
    return float(np.dot(cur, ref) / denom)
