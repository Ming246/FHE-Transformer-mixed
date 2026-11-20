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


def _looks_ct(x) -> bool:
    try:
        int(x.level_calc)
        return True
    except Exception:
        return False


def _unwrap_ct(x):
    """Peel ``[ct]`` wrappers from mock ``encodecrypt`` / list packs."""
    while isinstance(x, (list, tuple)) and len(x) == 1 and not _looks_ct(x):
        x = x[0]
    if _looks_ct(x):
        return x
    if isinstance(x, (list, tuple)):
        for y in x:
            y = _unwrap_ct(y)
            if _looks_ct(y):
                return y
    try:
        import numpy as np

        if isinstance(x, np.ndarray) and x.dtype == object:
            for y in x.ravel():
                y = _unwrap_ct(y)
                if _looks_ct(y):
                    return y
    except Exception:
        pass
    return x


def _as_ct_list(cts):
    # Liberate DataStruct subclasses list — never iterate a live CT.
    if _looks_ct(cts):
        return [cts], None, True
    try:
        import numpy as np

        if isinstance(cts, np.ndarray):
            flat = [_unwrap_ct(cts.ravel()[i]) for i in range(cts.size)]
            return flat, cts.shape, False
    except Exception:
        pass
    if isinstance(cts, (list, tuple)):
        return [_unwrap_ct(x) for x in cts], None, False
    return [_unwrap_ct(cts)], None, True


def _restore_ct_list(out, *, shape, wrap_single: bool):
    if wrap_single:
        return out[0]
    if shape is not None:
        import numpy as np

        arr = np.empty(shape, dtype=object)
        for i, v in enumerate(out):
            arr.ravel()[i] = v
        return arr
    return out


def align_join(
    engine,
    main_cts,
    bypass_cts,
    *,
    bts_hook=None,
    bypass_event: str | None = None,
):
    """
    会合：可选按 plan 刷旁路，然后两侧 ``level_up`` 到同一 ``level_calc``
    （= min rem）。plan 模式下不反应式刷旁路；无 plan 时旁路更耗尽则刷旁路。
    主路刷新由调用方 ``hook.refresh(主路事件)`` 完成。
    返回 ``(main_cts, bypass_cts)``（形状与输入相同）。
    """
    mains, m_shape, m_one = _as_ct_list(main_cts)
    byps, b_shape, b_one = _as_ct_list(bypass_cts)
    plan = bts_hook is not None and getattr(bts_hook, "mode", None) == "plan"
    if plan:
        if (
            bypass_event
            and bts_hook.should_refresh(bypass_event)
        ):
            byps_ref = bts_hook.refresh(bypass_event, byps)
            byps, _, _ = _as_ct_list(byps_ref)
    else:
        main_lc = int(mains[0].level_calc)
        if int(byps[0].level_calc) > main_lc:
            try:
                from bts_prescale import bootstrap_prescaled
            except ImportError:
                from .bts_prescale import bootstrap_prescaled  # type: ignore
            try:
                from bts_decrypt_probe import bts_probe_label
            except ImportError:
                from .bts_decrypt_probe import bts_probe_label  # type: ignore
            with bts_probe_label(engine, bypass_event or "align_bypass"):
                byps = [
                    bootstrap_prescaled(engine, ct, postscale=True) for ct in byps
                ]
    target = max(int(mains[0].level_calc), int(byps[0].level_calc))
    for i, ct in enumerate(mains):
        if int(ct.level_calc) < target:
            mains[i] = engine.level_up(ct, target)
    for i, ct in enumerate(byps):
        if int(ct.level_calc) < target:
            byps[i] = engine.level_up(ct, target)
    return (
        _restore_ct_list(mains, shape=m_shape, wrap_single=m_one),
        _restore_ct_list(byps, shape=b_shape, wrap_single=b_one),
    )


def align_bypass_to_main(
    engine, main_ct, bypass_ct, *, bts_hook=None, bypass_event=None
):
    """兼容旧接口：返回对齐后的旁路。主路可能已被 level_up，调用方应改用 ``align_join``。"""
    _main, bypass = align_join(
        engine,
        main_ct,
        bypass_ct,
        bts_hook=bts_hook,
        bypass_event=bypass_event,
    )
    return bypass


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


def _asor_prepare_boot(engine, an: _DeltaCt, bn: _DeltaCt) -> tuple[_DeltaCt, _DeltaCt]:
    """Official pre-boot lift. Logical ``ct/delta`` stays put.

    Runs whenever a plan boot fires, not only when the level is exhausted.
    ``scale_adjust > 1`` (``delta < 1/16``) clears the imaginary part, then
    multiplies ``bn`` and ``bn.delta`` by the same integer.
    """
    scale_adjust = int(1.0 / (bn.delta * 4.0) / 2.0)
    if scale_adjust > 1:
        an = _DeltaCt(
            engine.add(an.ciphertext, engine.conjugate(an.ciphertext)),
            an.delta * 2.0,
        )
        bn = _DeltaCt(
            engine.add(bn.ciphertext, engine.conjugate(bn.ciphertext)),
            bn.delta * 2.0,
        )
        scale_adjust = int(1.0 / (bn.delta * 4.0) / 2.0)
    else:
        scale_adjust = 1
    an_ct, bn_ct = engine.auto_level(an.ciphertext, bn.ciphertext)
    bn = _DeltaCt(
        engine.mult_int_scalar(bn_ct, scale_adjust),
        bn.delta * scale_adjust,
    )
    an = _DeltaCt(an_ct, an.delta)
    return an, bn


def _asor_pack_pre_gain(an: _DeltaCt, bn: _DeltaCt) -> int:
    """Integer ×gain before packing ``an+i·bn`` into CKKS bootstrap.

    Late σ-aSOR has ``1/σ`` already O(10²–10³) and raw ``an≈ logical·delta``
    already O(1). A fixed ×64 then sends |m|~100 into bootstrap (L2 i5:
    an_raw≈1.73 → ×64≈110), which is not bootstrap noise — it is overflow.
    Cap so an estimated |m| stays ≲ 0.4. Tiny-delta early iters still get ×64.
    """
    delta = max(float(an.delta), float(bn.delta), 1e-16)
    raw_est = 1000.0 * delta
    room = 0.4 / max(raw_est, 1e-12)
    gain = 1
    g = 64
    while g > 1:
        if room >= g:
            gain = g
            break
        g //= 2
    return int(gain)


def _asor_pack_bootstrap(
    engine,
    an: _DeltaCt,
    bn: _DeltaCt,
    *,
    bts_hook=None,
    bts_event: str | None = None,
) -> tuple[_DeltaCt, _DeltaCt]:
    """Plan boot that leaves logical ``ct/delta`` unchanged.

    Pack ``z=an+i·bn``, optional integer prescale, bootstrap, unpack.
    Unpack ``/2`` undoes ``z+conj=2·Re``. ``delta`` is unchanged.
    """
    if bts_hook is None or not bts_event:
        raise RuntimeError(
            "aSOR pack bootstrap requires bts_hook + bts_event (plan-only)"
        )
    an, bn = _asor_prepare_boot(engine, an, bn)
    pre = _asor_pack_pre_gain(an, bn)
    _an_pre, _bn_pre = an.ciphertext, bn.ciphertext
    temp = engine.add(an.ciphertext, engine.imult(bn.ciphertext))
    _site_probe = None
    if bts_hook.should_refresh(bts_event):
        coll = getattr(engine, "_bts_site_probe", None)
        if coll is not None:
            _site_probe = coll
    try:
        from bts_site_pack_probe import pack_mode_for_probe, suppress_hook_site_probe
    except ImportError:
        from .bts_site_pack_probe import (  # type: ignore
            pack_mode_for_probe,
            suppress_hook_site_probe,
        )
    # This call only: refresh must not apply the repo-wide ×100.
    bare = getattr(bts_hook, "bare_scale_events", None)
    added_bare = False
    if bare is not None and bts_event not in bare:
        bare.add(bts_event)
        added_bare = True
    try:
        if pre != 1:
            temp = engine.mult_int_scalar(temp, int(pre))
        with suppress_hook_site_probe(engine):
            temp1 = bts_hook.refresh(bts_event, temp)
    finally:
        if added_bare:
            bare.discard(bts_event)
    # Undo prescale. /2 undoes z+conj. One rescale.
    conj = engine.conjugate(temp1)
    restore = 1.0 / float(pre) / 2.0
    print(
        f"  [asor pack bts] {bts_event} pre=×{pre} "
        f"an.delta={an.delta:.3e} bn.delta={bn.delta:.3e}",
        flush=True,
    )
    an = _DeltaCt(engine.mult_scalar(engine.add(temp1, conj), restore), an.delta)
    bn = _DeltaCt(
        engine.mult_scalar(engine.imult(engine.sub(conj, temp1)), restore),
        bn.delta,
    )
    if _site_probe is not None:
        _site_probe.record_real_pair(
            bts_event,
            pack_mode_for_probe(bts_hook, bts_event),
            _an_pre,
            _bn_pre,
            an.ciphertext,
            bn.ciphertext,
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
    (unique site; e.g. ``L0.softmax_norm_r0_asor4``). σ-aSOR does **not**
    pass this hook (no mid-iter an/bn boot).

    Returns ``(inv_ct, D_delta, en)`` — do **not** fold ``1/delta`` into the CT;
    DualRail exit / ``update_inv_D`` consume ``D_delta`` like native THOR.
    """
    an = _DeltaCt(numerator, 1.0)
    bn = _DeltaCt(denominator, 1.0)
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 must be > 0, got {e0}")
    if getattr(engine, "_mag_sk", None) is not None:
        from mag_probe import report as _mag_report

        _tag0 = (
            bts_iter_event(0)
            if bts_iter_event is not None
            else "asor"
        )
        _stem = str(_tag0).rsplit("_i", 1)[0].rsplit("_asor", 1)[0]
        _mag_report(engine, f"{_stem}.enter_num", an.ciphertext, delta=an.delta)
        _mag_report(
            engine,
            f"{_stem}.enter_den",
            bn.ciphertext,
            delta=bn.delta,
            note=f"e0={en:.4e}",
        )
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
        if getattr(engine, "_mag_sk", None) is not None:
            from mag_probe import report as _mag_report

            tag = event or f"asor_i{i}"
            _mag_report(engine, f"{tag}.an", an.ciphertext, delta=an.delta)
            _mag_report(engine, f"{tag}.bn", bn.ciphertext, delta=bn.delta)

    if getattr(engine, "_mag_inv_audit", False) and bts_iter_event is not None:
        from mag_probe import audit_asor_output

        audit_asor_output(
            engine,
            str(bts_iter_event(0)),
            denominator,
            an.ciphertext,
            float(an.delta),
            float(e0),
            int(max_iters),
        )

    return an.ciphertext, an.delta, en


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
    if (
        getattr(engine, "_mag_sk", None) is not None
        and not getattr(engine, "_mag_inv_audit", False)
        and not getattr(engine, "_mag_ln_audit", False)
        and bts_apply_event
        and "sumsq_r0_apply" in str(bts_apply_event)
        and getattr(engine, "_mag_att_score", None) is not None
    ):
        from mag_probe import ExpInvStop, compare_exp_inv_inputs

        compare_exp_inv_inputs(
            engine, exp_u, inv_d, d_delta=float(d_delta)
        )
        raise ExpInvStop("update_inv_D_fixed inputs exp_u/inv_d saved")
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
    bypass_ev = None
    if bts_apply_event and ".softmax_sumsq_" in str(bts_apply_event):
        head, tail = str(bts_apply_event).split(".", 1)
        bypass_ev = f"{head}.bypass.{tail.replace('_apply', '_exp')}"
    inv_list, exp_u = align_join(
        engine,
        inv_list,
        exp_u,
        bts_hook=bts_hook,
        bypass_event=bypass_ev,
    )
    for i in range(4):
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
    pre_rotsum = summation
    # THOR ``update_inv_D``：求和后、rotsum 前必刷这一条 summation。
    sum_ev = (
        bts_apply_event.replace("_apply", "_sum")
        if bts_apply_event
        else None
    )
    if (
        bts_hook is not None
        and sum_ev
        and bts_hook.should_refresh(sum_ev)
    ):
        summation = bts_hook.refresh(sum_ev, summation)
    summation = engine.rotsum(summation, interval=2**11)
    if (
        getattr(engine, "_mag_sk", None) is not None
        and bts_apply_event
        and "sumsq_r0_apply" in str(bts_apply_event)
    ):
        from mag_probe import compare_sumsq

        compare_sumsq(
            engine,
            exp_u,
            inv_list,
            pre_rotsum,
            k=max(int(1.0 / float(d_delta) / 2.0), 1),
            d_delta=float(d_delta),
            after_rotsum=summation,
        )
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
    bts_exp_event: str | None = None,
    rem_probe=None,
):
    """Native THOR DualRail exit: 8 real → 128 real α copies.

    Plan 刷主路 **inv**（``direct``，1 CT）。旁路 exp 由
    ``bypass.softmax_exit_exp`` + ``align_join``（可 level_up inv）。
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
    exp_cplx = [None] * 4
    for i in range(4):
        exp_cplx[i] = engine.cc_add(exp_u[i], engine.imult(exp_u[i + 4]))
    rotated_inv, exp_cplx = align_join(
        engine,
        rotated_inv,
        exp_cplx,
        bts_hook=bts_hook,
        bypass_event=bts_exp_event,
    )
    for i in range(4):
        exp_cplx[i] = engine.mult_int_scalar(exp_cplx[i], scale_k)

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
    if getattr(engine, "_mag_sk", None) is not None:
        from mag_probe import report as _mag_report

        _mag_report(
            engine,
            f"{bts_prefix}.softmax_exp",
            exp_cts,
            note="plain exp pieces are ≤ σ",
        )
    if getattr(engine, "_mag_exp_audit", False):
        from mag_probe import audit_score_and_exp

        audit_score_and_exp(engine, packs, exp_cts)
    try:
        from chain_bts_mock_probe import active_chain_probe
    except ImportError:
        from .chain_bts_mock_probe import active_chain_probe  # type: ignore
    _chain = active_chain_probe(engine)
    if _chain is not None:
        _exp_probe = [
            exp_cts[i]
            for i in range(len(exp_cts))
            if hasattr(exp_cts[i], "level_calc")
        ]
        if len(_exp_probe) != 8:
            _ty = [type(exp_cts[i]).__name__ for i in range(min(8, len(exp_cts)))]
            raise RuntimeError(
                f"softmax_exp_cts probe: expected 8 CTs, got {len(_exp_probe)} "
                f"(types={_ty})"
            )
        _chain.touch("softmax_exp_cts", _exp_probe)
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
    if getattr(engine, "_mag_sk", None) is not None:
        from mag_probe import report as _mag_report

        _mag_report(
            engine,
            f"{bts_prefix}.softmax_sigma",
            sigma_exp,
            note="plain σ on val0 L8 is about 6e-5..0.1",
        )
    if _chain is not None:
        _chain.touch("softmax_sigma_exp", sigma_exp)

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
    inv_d, d_delta, en = he_asor_ct(
        engine,
        enc_one,
        sigma_exp,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
        bts_hook=bts_hook,
        bts_iter_event=None,
        rem_probe=rem_probe,
    )
    inv_ev = f"{bts_prefix}.softmax_asor_sigma_inv"
    if bts_hook is not None and bts_hook.should_refresh(inv_ev):
        inv_d = bts_hook.refresh(inv_ev, inv_d)
    if _chain is not None:
        _chain.touch("softmax_inv_d_sigma", inv_d)

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
        if _chain is not None:
            _chain.touch(f"softmax_sumsq_r{r}_exp_u", exp_u)
            _chain.touch(f"softmax_sumsq_r{r}_inv_d", inv_d)

    return dualrail_softmax_exit(
        engine,
        exp_u,
        inv_d,
        d_delta=d_delta,
        rescale=rescale,
        bts_hook=bts_hook,
        bts_event=f"{bts_prefix}.softmax_dualrail_exit",
        bts_exp_event=f"{bts_prefix}.bypass.softmax_exit_exp",
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
