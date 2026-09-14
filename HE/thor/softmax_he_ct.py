#!/usr/bin/env python3
"""
Liberate DualRail Softmax — repo ``softmax_poly`` fixed-iter aSOR.

I/O matches THOR ``he_softmax1/2``:
  in:  ``x`` shape ``(8,)`` real score CTs (post score DualRail bootstrap)
  out: shape ``(128,)`` DualRail α copies for ``calculate_attention_context``

Algorithm (same shape as THOR ``he_softmax`` / ``he_inv`` / ``update_inv_D``):
  - Stockmeyer exp from ``SoftmaxHeParams`` (shift / δ1 / δ2 / THOR_INPUT_SCALE)
  - aSOR inverse: THOR ``he_inv`` CT graph + ``delta`` bookkeeping, but
    **fixed** ``max_iters`` and production ``e0`` (no α early-stop)
  - Σy² rounds via ``update_inv_D`` twin; ``e0_2 = en / seq_len`` (repo)
  - Native DualRail 8→128 exit (``D_delta`` kept for scale_k)
"""
from __future__ import annotations

import math
import os
import sys
from functools import partial

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

from slot_softmax_he import SoftmaxHeParams  # noqa: E402
from softmax_poly import (  # noqa: E402
    MAX_SUM_SQ_ROUNDS,
    _check_power_of_two,
    thor_eps2_from_en,
)
from thor.nonlinear.polynomial import evaluate_polynomial_stockmeyer  # noqa: E402


def align_bypass_to_main(engine, main_ct, bypass_ct):
    """
    旁路向主路对齐：不改 main。bypass 更耗尽则先 bootstrap，再 level_up 到 main.lc。
    """
    main_lc = int(main_ct.level_calc)
    b_lc = int(bypass_ct.level_calc)
    if b_lc > main_lc:
        bypass_ct = engine.bootstrap(bypass_ct)
        b_lc = int(bypass_ct.level_calc)
    if b_lc < main_lc:
        bypass_ct = engine.level_up(bypass_ct, main_lc)
    return bypass_ct


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _probe_rem(rem_probe, name: str, engine, ct) -> None:
    """Optional micro-event rem probe: ``rem_probe(name, rem)``."""
    if rem_probe is None or ct is None:
        return
    rem_probe(str(name), _rem(engine, ct))


def _head_mask(num_slots: int, pad: float = 0.0) -> np.ndarray:
    return np.array(([1.0] * 12 + [float(pad)] * 4) * (num_slots // 16), dtype=np.float64)


class _DeltaCt:
    """THOR ``Ciphertext``: logical inv ≈ ``ct / delta``; exit uses ``D_delta``."""

    __slots__ = ("ciphertext", "delta")

    def __init__(self, ciphertext, delta: float = 1.0):
        self.ciphertext = ciphertext
        self.delta = float(delta)


def _asor_pack_bootstrap(
    engine,
    an: _DeltaCt,
    bn: _DeltaCt,
    *,
    bts_hook=None,
    bts_event: str | None = None,
) -> tuple[_DeltaCt, _DeltaCt]:
    """DualRail pack (an, bn) → bootstrap → unpack ×½（恒等；禁止 ×16 / /1.2）。"""
    an.ciphertext, bn.ciphertext = engine.auto_level(
        an.ciphertext, bn.ciphertext
    )
    temp = engine.add(an.ciphertext, engine.imult(bn.ciphertext))
    if bts_hook is None or not bts_event:
        raise RuntimeError(
            "aSOR pack bootstrap requires bts_hook + bts_event (plan-only)"
        )
    temp1 = bts_hook.refresh(bts_event, temp)
    conj = engine.conjugate(temp1)
    half = 0.5
    an = _DeltaCt(
        engine.mult_scalar(engine.add(temp1, conj), half), an.delta
    )
    bn = _DeltaCt(
        engine.mult_scalar(engine.imult(engine.sub(conj, temp1)), half),
        bn.delta,
    )
    return an, bn


def he_asor_ct(
    engine,
    numerator,
    denominator,
    e0: float,
    max_iters: int,
    *,
    bts_hook=None,
    bts_iter_event=None,
    rem_probe=None,
):
    """
    Fixed-iter aSOR (production). Same CT ops as THOR ``he_inv``, but
    ``for _ in range(max_iters)`` instead of ``while en < 1-alpha``.

    ``bts_iter_event(i)`` → plan event for before-boot at iteration ``i``
    (unique site; e.g. ``L0.softmax_asor_sigma_i4``).

    Returns ``(inv_ct, D_delta, en)`` — do **not** fold ``1/delta`` into the CT;
    DualRail exit / ``update_inv_D`` consume ``D_delta`` like native THOR.
    """
    an = _DeltaCt(numerator, 1.0)
    bn = _DeltaCt(denominator, 1.0)
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 must be > 0, got {e0}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters must be ≥ 1, got {max_iters}")

    for i in range(max_iters):
        event = bts_iter_event(i) if bts_iter_event is not None else None
        if (
            bts_hook is not None
            and event
            and bts_hook.should_refresh(event)
        ):
            an, bn = _asor_pack_bootstrap(
                engine,
                an,
                bn,
                bts_hook=bts_hook,
                bts_event=event,
            )

        kn = 2.0 / (en + 1.0)
        b_temp = _DeltaCt(
            engine.negate(
                engine.add_scalar(bn.ciphertext, -2.0 / kn * bn.delta)
            ),
            bn.delta,
        )
        an = _DeltaCt(
            engine.auto_ct_ct_mult(an.ciphertext, b_temp.ciphertext),
            an.delta * b_temp.delta,
        )
        an = _DeltaCt(an.ciphertext, an.delta / (kn**2))
        bn = _DeltaCt(
            engine.auto_ct_ct_mult(bn.ciphertext, b_temp.ciphertext),
            bn.delta * b_temp.delta,
        )
        bn = _DeltaCt(bn.ciphertext, bn.delta / (kn**2))
        en = kn * en * (2.0 - kn * en)

        scale_adjust = int(1.0 / bn.delta / 2**8)
        if scale_adjust > 1:
            conj_an = engine.conjugate(an.ciphertext)
            an = _DeltaCt(engine.add(an.ciphertext, conj_an), an.delta * 2.0)
            conj_bn = engine.conjugate(bn.ciphertext)
            bn = _DeltaCt(engine.add(bn.ciphertext, conj_bn), bn.delta * 2.0)
            scale_adjust = int(1.0 / bn.delta / 2**8)
        else:
            scale_adjust = 1
        bn = _DeltaCt(
            engine.mult_int_scalar(bn.ciphertext, scale_adjust),
            bn.delta * scale_adjust,
        )
        an = _DeltaCt(
            engine.mult_int_scalar(an.ciphertext, scale_adjust),
            an.delta * scale_adjust,
        )
        if event:
            # strip optional ``L*.`` prefix for probe keys
            suf = str(event).split(".", 1)[-1]
            _probe_rem(rem_probe, suf, engine, an.ciphertext)

    return an.ciphertext, an.delta, en


def _refresh_dualrail8(engine, packs, *, max_level_calc: int | None = None):
    """
    Bootstrap 8 real Softmax packs；``×1/2`` before bs cancels ``2·Re/Im`` unpack
    so the post-score DualRail 2× convention is preserved.

    Gate on ``level_calc`` only (see ``he_asor_ct`` — ``ct.level`` can lie).
    """
    packs = list(packs)
    if max_level_calc is None:
        max_level_calc = int(engine.num_levels) - 5
    if max(int(ct.level_calc) for ct in packs) <= max_level_calc:
        return packs
    print(
        f"  [softmax_he_ct] DualRail refresh packs "
        f"level_calc={max(int(ct.level_calc) for ct in packs)} "
        f"thresh={max_level_calc}",
        flush=True,
    )
    out = list(packs)
    for i in range(4):
        temp = engine.cc_add(out[i], engine.imult(out[i + 4]))
        temp = engine.bootstrap(temp)
        # Leave at bts land; no hardcoded level_up(→16).
        conj = engine.conjugate(temp)
        out[i] = engine.cc_add(temp, conj)
        out[i + 4] = engine.imult(engine.cc_sub(conj, temp))
        # bs 后乘 0.5（耗尽时不能在 bs 前 mult_scalar）
        out[i] = engine.mult_scalar(out[i], 0.5)
        out[i + 4] = engine.mult_scalar(out[i + 4], 0.5)
    return out


def update_inv_D_fixed(
    engine,
    exp_u,
    attention_mask,
    inv_d,
    d_delta: float,
    en: float,
    *,
    max_iters: int,
    final_inv: bool = False,
    bts_hook=None,
    bts_apply_event: str | None = None,
    bts_norm_prefix: str | None = None,
    rem_probe=None,
):
    """
    THOR ``update_inv_D`` twin with fixed-iter aSOR.

    ``e0`` for the new inverse is ``thor_eps2_from_en(en)`` (repo), not
    THOR's ``en/128/2``.

    ``bts_apply_event``：``softmax_sumsq_r*_apply`` 段首刷 **inv**；段内
    inv→summation（含 head ``cc_add`` + ``rotsum``，仍 1 CT）。

    ``bts_apply_event`` 同名族 ``…_enc_one``：rotsum 后 summation(1)→an/bn(2)，
    段首刷 summation，再 ``encode_and_encrypt(enc_one)``。

    ``bts_norm_prefix``：``…_asor{i}`` / ``…_inv`` / ``…_inv_mask``。

    ``he_asor_ct`` 写回的 ``inv_d`` 是全新倒数 CT，与 apply 前 inv **同名不同义**。
    """
    apply_suf = (
        str(bts_apply_event).split(".", 1)[-1] if bts_apply_event else None
    )
    # before softmax_sumsq_r*_apply
    if (
        bts_hook is not None
        and bts_apply_event
        and bts_hook.should_refresh(bts_apply_event)
    ):
        inv_d = bts_hook.refresh(bts_apply_event, inv_d)

    exp_2u = np.full((8,), None, dtype=object)
    inv_list = [
        engine.rescale(engine.pt_ct_mult(attention_mask[i], inv_d))
        for i in range(8)
    ]
    for i in range(4):
        # inv_list 为主路；exp 旁路向 inv 对齐（不烧 inv rem）
        exp_u[i] = align_bypass_to_main(engine, inv_list[i], exp_u[i])
        exp_u[i + 4] = align_bypass_to_main(
            engine, inv_list[i + 4], exp_u[i + 4]
        )
        k = max(int(1.0 / float(d_delta) / 2.0), 1)
        exp_2u[i] = engine.square(
            engine.mult_int_scalar(
                engine.ct_ct_mult(
                    engine.cc_add(exp_u[i], exp_u[i]), inv_list[i]
                ),
                k,
            )
        )
        exp_2u[i + 4] = engine.square(
            engine.mult_int_scalar(
                engine.ct_ct_mult(
                    engine.cc_add(exp_u[i + 4], exp_u[i + 4]), inv_list[i + 4]
                ),
                k,
            )
        )

    summation = exp_2u[0]
    for i in range(1, 8):
        summation = engine.cc_add(summation, exp_2u[i])
    summation = engine.rotsum(summation, interval=2**11)
    if apply_suf:
        _probe_rem(rem_probe, apply_suf, engine, summation)

    # softmax_sumsq_r*_enc_one：summation(1) → enc_one+summation(2)
    enc_ev = (
        bts_apply_event.replace("_apply", "_enc_one")
        if bts_apply_event
        else None
    )
    if (
        bts_hook is not None
        and enc_ev
        and bts_hook.should_refresh(enc_ev)
    ):
        summation = bts_hook.refresh(enc_ev, summation)

    nslots = int(engine.num_slots)
    e0_2 = thor_eps2_from_en(en)
    enc_one = engine.encode_and_encrypt(
        _head_mask(nslots, pad=0.0),
        level=int(engine.num_levels) - int(summation.level),
    )
    if enc_ev:
        _probe_rem(
            rem_probe, str(enc_ev).split(".", 1)[-1], engine, summation
        )
    inv_d, d_delta, en = he_asor_ct(
        engine,
        enc_one,
        summation,
        e0_2,
        max_iters,
        bts_hook=bts_hook,
        bts_iter_event=(
            (lambda gi, p=bts_norm_prefix: f"{p}_asor{gi}")
            if bts_norm_prefix
            else None
        ),
        rem_probe=rem_probe,
    )
    # 图上无 drop-bn ``*_inv``：aSOR 返回 an 后直接接 ``*_inv_mask``。
    mask_ev = f"{bts_norm_prefix}_inv_mask" if bts_norm_prefix else None
    if (
        bts_hook is not None
        and mask_ev
        and bts_hook.should_refresh(mask_ev)
    ):
        inv_d = bts_hook.refresh(mask_ev, inv_d)

    # Match THOR update_inv_D masking exactly (including short final mask).
    masking = np.array(([1.0] * 12 + [0.0] * 4) * (2**11), dtype=np.float64)
    if final_inv:
        masking = np.array(([1.0] * 12 + [0.0] * 4) * (2**7), dtype=np.float64)
    inv_d = engine.cm_mult(inv_d, masking)
    if mask_ev:
        _probe_rem(
            rem_probe, str(mask_ev).split(".", 1)[-1], engine, inv_d
        )
    return exp_2u, inv_d, d_delta, en


def thor_exp_ct(
    engine,
    packs,
    params: SoftmaxHeParams,
    *,
    exp_input_prescaled: bool = True,
    bts_hook=None,
    bts_square_event: str | None = None,
    rem_probe=None,
):
    """Stockmeyer deg-15 exp + δ1 squares on a pack list (typically 8 score CTs).

    Score CTs must already be ``x/(64·δ1·δ2)``. Between Stockmeyer (#4) and
    square (#5), may ``before``-boot ``bts_square_event`` (``softmax_exp_square_0``).
    Returns a list of exp CTs (same length as ``packs``).
    """
    if not exp_input_prescaled:
        raise ValueError(
            "thor_exp_ct requires exp_input_prescaled=True "
            "(non-prescaled mult_scalar would burn +1 rem vs Stockmeyer-only path)"
        )
    packs = list(packs)
    if not packs:
        raise ValueError("thor_exp_ct expects a non-empty pack list")

    d1 = float(params.delta1)
    d2 = float(params.delta2)
    n1 = _check_power_of_two("delta1", d1)
    scale = math.exp(params.shift / d2 / d1)
    inv_scale = 1.0 / scale
    coeffs_asc = np.array(tuple(reversed(params.exp_coeffs)), dtype=np.float64)
    baked = coeffs_asc * inv_scale

    # #4 softmax_exp_stockmeyer（段内不可 mid）
    out = [
        evaluate_polynomial_stockmeyer(engine, baked, p) for p in packs
    ]
    _probe_rem(rem_probe, "softmax_exp_stockmeyer", engine, out[0])

    # #4 → #5：plan 说刷就刷
    if (
        bts_hook is not None
        and bts_square_event
        and bts_hook.should_refresh(bts_square_event)
    ):
        out = list(bts_hook.refresh(bts_square_event, out))

    # #5 softmax_exp_square_0（及 δ1>2 时的后续平方）
    for i in range(len(out)):
        for _ in range(n1):
            out[i] = engine.square(out[i])
    _probe_rem(rem_probe, "softmax_exp_square_0", engine, out[0])
    return out


def dualrail_softmax_exit(
    engine,
    exp_u,
    inv_d,
    *,
    d_delta: float = 1.0,
    rescale: bool = False,
    bts_hook=None,
    bts_event: str | None = None,
    rem_probe=None,
):
    """Native THOR DualRail exit: 8 real → 128 real α copies.

    Plan 仅刷主路 **inv**（``direct``，1 CT；入口 rem≥``DEPTH_SOFTMAX_EXIT``）。
    旁路 exp 不进 DP：

      1. 8 real → 4 ``exp_cplx``
      2. 一次 ``align_bypass_to_main(exp→rotated_inv[0])``
         （``rotate_left`` 不烧 rem，16 个 rotated_inv 同 level）
      3. ``mult_int_scalar(scale_k)``（不改 ``level_calc``，对齐关系保持）
      4. ``auto_ct_ct_mult(exp_cplx, rotated_inv[j])``

    不再对 exp 做 rem≤1 本地 bootstrap：ct×ct 预算由主路 inv 的 plan/bts 保证。
    """
    if (
        bts_hook is not None
        and bts_event
        and bts_hook.should_refresh(bts_event)
    ):
        inv_d = bts_hook.refresh(bts_event, inv_d)

    rotated_inv = [inv_d]
    for _ in range(15):
        rotated_inv.append(engine.rotate_left(rotated_inv[-1], -(2**11)))

    scale_k = int(1.0 / (2.0 * float(d_delta))) + 1
    # Pack + align once to inv (all rotated_inv share inv's level).
    exp_cplx = [None] * 4
    for i in range(4):
        packed = engine.cc_add(exp_u[i], engine.imult(exp_u[i + 4]))
        packed = align_bypass_to_main(engine, rotated_inv[0], packed)
        # rem-neutral; stays level-matched with rotated_inv
        exp_cplx[i] = engine.mult_int_scalar(packed, scale_k)

    cplx_softmax1 = []
    cplx_softmax2 = []
    for i in range(4):
        for j in range(16):
            masked = engine.auto_ct_ct_mult(exp_cplx[i], rotated_inv[j])
            if rescale:
                masked = engine.rescale(masked)
            copied = engine.rotsum(masked, interval=2**11)
            copied_real = engine.cc_add(copied, engine.conjugate(copied))
            cplx_softmax1.append(copied_real)
            copied_imag = engine.cc_sub(engine.conjugate(copied), copied)
            copied_imag = engine.imult(copied_imag)
            cplx_softmax2.append(copied_imag)

    softmax_128 = cplx_softmax1 + cplx_softmax2
    result = np.full((128,), None, dtype=object)
    for i in range(128):
        result[i] = softmax_128[i]
    _probe_rem(rem_probe, "softmax_dualrail_exit", engine, result[0])
    return result


def he_softmax_poly(
    engine,
    x,
    attention_mask,
    params: SoftmaxHeParams,
    *,
    layer_idx: int = 0,
    rescale: bool = False,
    debug: bool = False,
    sk=None,
    bts_hook=None,
    bts_prefix: str | None = None,
    rem_probe=None,
):
    """
    Drop-in for THOR ``he_softmax1/2`` using repo fixed-iter aSOR.

    Mirrors ``he_softmax`` control flow with ``l = delta2``:
      σ-aSOR (iters_σ, e0_σ) → ``log2(l)`` Σy² updates (iters_Σy²_*) → DualRail exit.

    ``rem_probe(name, rem)``：可选，在各微事件结束后回调（无中途 bts 时核对 depth）。
    """
    del debug, sk
    if bts_prefix is None:
        bts_prefix = f"L{int(layer_idx)}"
    if getattr(x, "shape", None) != (8,) and len(x) != 8:
        raise ValueError(
            f"he_softmax_poly expects 8 score CTs, got {getattr(x, 'shape', len(x))}"
        )

    d2 = float(params.delta2)
    n2 = _check_power_of_two("delta2", d2)
    if n2 > MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"delta2={d2} → {n2} Σy² rounds > MAX_SUM_SQ_ROUNDS={MAX_SUM_SQ_ROUNDS}"
        )
    if n2 < 1:
        raise ValueError(f"poly Softmax expects delta2≥2 (n2≥1), got {d2}")

    # before #4 stockmeyer；#5 square 在 thor_exp_ct 内
    # ``softmax_exp_stockmeyer`` 段首 DualRail 刷在 ``smoke_thor_repro``
    # （8 real→4 cplx，与 plan cost_ct=4 / real_pack 一致）。这里不再 hook。
    packs = [x[i] for i in range(8)]
    _probe_rem(rem_probe, "entry", engine, packs[0])

    # #4 → [#5 前可选 bts] → #5 → [#6 前可选 bts] → #6 mask
    exp_cts = thor_exp_ct(
        engine,
        packs,
        params,
        bts_hook=bts_hook,
        bts_square_event=f"{bts_prefix}.softmax_exp_square_0",
        rem_probe=rem_probe,
    )
    mask_ev = f"{bts_prefix}.softmax_attn_mask"
    if bts_hook is not None and bts_hook.should_refresh(mask_ev):
        exp_cts = list(bts_hook.refresh(mask_ev, exp_cts))
    exp_u = np.full((8,), None, dtype=object)
    for i in range(8):
        exp_u[i] = engine.rescale(engine.pt_ct_mult(attention_mask[i], exp_cts[i]))
    _probe_rem(rem_probe, "softmax_attn_mask", engine, exp_u[0])

    # softmax_sigma_aggregate：8 exp_u → sigma_exp(1)；exp_u→AUX
    agg_ev = f"{bts_prefix}.softmax_sigma_aggregate"
    if bts_hook is not None and bts_hook.should_refresh(agg_ev):
        exp_u = list(bts_hook.refresh(agg_ev, exp_u))
    sigma_exp = exp_u[0]
    for i in range(1, 8):
        sigma_exp = engine.cc_add(sigma_exp, exp_u[i])
    sigma_exp = engine.rotsum(sigma_exp, interval=2**11)
    _probe_rem(rem_probe, "softmax_sigma_aggregate", engine, sigma_exp)

    # softmax_sigma_enc_one：sigma_exp(1) → enc_one+sigma_exp(2, he_asor an/bn)
    enc_ev = f"{bts_prefix}.softmax_sigma_enc_one"
    if bts_hook is not None and bts_hook.should_refresh(enc_ev):
        sigma_exp = bts_hook.refresh(enc_ev, sigma_exp)

    nslots = int(engine.num_slots)
    enc_one = engine.encode_and_encrypt(
        _head_mask(nslots, pad=0.0),
        level=int(engine.num_levels) - int(sigma_exp.level),
    )
    _probe_rem(rem_probe, "softmax_sigma_enc_one", engine, sigma_exp)
    rec_n = len(getattr(bts_hook, "records", []) or [])
    inv_d, d_delta, en = he_asor_ct(
        engine,
        enc_one,
        sigma_exp,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
        bts_hook=bts_hook,
        bts_iter_event=lambda i, p=bts_prefix: f"{p}.softmax_asor_sigma_i{i}",
        rem_probe=rem_probe,
    )
    # σ-aSOR spine 刷了 an/bn 时，AUX 上的 exp packs 一并恢复
    new_recs = getattr(bts_hook, "records", [])[rec_n:]
    if any(
        "softmax_asor_sigma_i" in str(getattr(r, "event_name", ""))
        and int(getattr(r, "ct_count", 0) or 0) > 0
        for r in new_recs
    ):
        for i in range(8):
            exp_u[i] = engine.bootstrap(exp_u[i])

    # n2 Σy² rounds (δ2=2 → 1; δ2=4 → 2), last marked final_inv.
    for r in range(n2):
        iters_r = int(params.asor_max_iters_sum_sq[r])
        if iters_r < 1:
            raise ValueError(
                f"Σy² round {r} asor_max_iters={iters_r} invalid (delta2={d2})"
            )
        exp_u, inv_d, d_delta, en = update_inv_D_fixed(
            engine,
            exp_u,
            attention_mask,
            inv_d,
            d_delta,
            en,
            max_iters=iters_r,
            final_inv=(r == n2 - 1),
            bts_hook=bts_hook,
            bts_apply_event=f"{bts_prefix}.softmax_sumsq_r{r}_apply",
            bts_norm_prefix=f"{bts_prefix}.softmax_norm_r{r}",
            rem_probe=rem_probe,
        )

    return dualrail_softmax_exit(
        engine,
        exp_u,
        inv_d,
        d_delta=d_delta,
        rescale=rescale,
        bts_hook=bts_hook,
        bts_event=f"{bts_prefix}.softmax_dualrail_exit",
        rem_probe=rem_probe,
    )


def bind_attention_softmax(
    thor_attention,
    *,
    task: str,
    layer_idx: int,
    level: int = 2,
    bts_hook=None,
) -> SoftmaxHeParams:
    """Replace ``thor_attention.softmax`` with repo fixed-iter aSOR Softmax."""
    params = SoftmaxHeParams.from_softmax_poly(task, layer_idx, level=level)
    thor_attention.softmax = partial(
        he_softmax_poly,
        engine=thor_attention.engine,
        params=params,
        layer_idx=int(layer_idx),
        bts_hook=bts_hook,
        bts_prefix=f"L{int(layer_idx)}",
    )
    thor_attention._softmax_poly_params = params
    return params
