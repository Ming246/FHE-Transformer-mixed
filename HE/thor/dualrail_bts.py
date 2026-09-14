"""DualRail pack → bootstrap → unpack helpers (named BootstrapHook sites)."""
from __future__ import annotations

from typing import Any

import numpy as np


def mock_bootstrap_land_level(engine, *, rem_after: int = 14) -> int:
    """``level_calc`` matching typical THOR DualRail bootstrap land (rem≈14)."""
    return max(0, int(engine.num_levels) - int(rem_after))


def install_mock_bootstrap(
    engine,
    sk,
    pk,
    *,
    land_level: int | None = None,
    rem_after: int = 14,
) -> dict:
    """
    Replace ``engine.bootstrap`` with decrypt → ``encodecrypt`` at rem ``rem_after``.

    Only swaps the bootstrap kernel. DualRail ``real_pack`` still does
    pack → bootstrap → unpack×½ outside; the ×½ burns 1 rem so effective
    rem after a full refresh is ``rem_after − 1`` (budget−1 when rem_after=14).

    Diagnostic only: strips bootstrap noise so mul/rot/poly errors are visible.
    """
    if land_level is None:
        land_level = mock_bootstrap_land_level(engine, rem_after=int(rem_after))
    land_level = int(land_level)
    real_bootstrap = engine.bootstrap
    calls = {"n": 0}

    def _mock(ct):
        calls["n"] += 1
        msg = np.asarray(engine.decrode(ct, sk), dtype=np.complex128)
        return engine.encodecrypt(msg, pk, level=land_level)

    engine.bootstrap = _mock  # type: ignore[method-assign]
    info = {
        "land_level": land_level,
        "rem_after": int(engine.num_levels) - land_level,
        "calls": calls,
        "real_bootstrap": real_bootstrap,
    }
    engine._mock_bootstrap_info = info  # type: ignore[attr-defined]
    return info


def uninstall_mock_bootstrap(engine) -> None:
    info = getattr(engine, "_mock_bootstrap_info", None)
    if info is None:
        return
    engine.bootstrap = info["real_bootstrap"]  # type: ignore[method-assign]
    delattr(engine, "_mock_bootstrap_info")


def refresh_dualrail8(
    engine,
    packs,
    hook: Any | None,
    event_name: str,
    *,
    scale_before: float | None = None,
    scale_after: float | None = 0.5,
) -> list:
    """
    Bootstrap 8 real DualRail packs as 4 complex CTs.

    解包恒等：``z+conj=2Re``，默认 ``scale_after=0.5``。禁止靠后面算子吃 ×2。
    ``hook is None``：强制 ``engine.bootstrap``（旁路伴侣，不计 plan）。
    否则走 ``hook.refresh``（plan / always / noop）。
    """
    packs = list(packs)
    if len(packs) != 8:
        raise ValueError(f"expected 8 packs, got {len(packs)}")
    temps = []
    for i in range(4):
        temp = engine.cc_add(packs[i], engine.imult(packs[i + 4]))
        if scale_before is not None:
            if int(temp.level_calc) < int(engine.num_levels) - 1:
                temp = engine.mult_scalar(temp, float(scale_before))
        temps.append(temp)

    if hook is None:
        temps = [engine.bootstrap(t) for t in temps]
        out = list(packs)
        for i in range(4):
            conj = engine.conjugate(temps[i])
            out[i] = engine.cc_add(temps[i], conj)
            out[i + 4] = engine.imult(engine.cc_sub(conj, temps[i]))
        s = 0.5 if scale_after is None else float(scale_after)
        for i in range(8):
            out[i] = engine.mult_scalar(out[i], s)
        return out

    prev = bool(getattr(hook, "apply_discards", False))
    hook.apply_discards = False
    try:
        temps = list(hook.refresh(event_name, temps))
    finally:
        hook.apply_discards = prev

    out = list(packs)
    for i in range(4):
        conj = engine.conjugate(temps[i])
        out[i] = engine.cc_add(temps[i], conj)
        out[i + 4] = engine.imult(engine.cc_sub(conj, temps[i]))
    s = 0.5 if scale_after is None else float(scale_after)
    for i in range(8):
        out[i] = engine.mult_scalar(out[i], s)
    keep = None
    if prev and hasattr(hook, "remaining_keep_after"):
        keep = hook.remaining_keep_after(event_name)
    if keep is not None and hasattr(hook, "apply_discard"):
        for i in range(8):
            out[i] = hook.apply_discard(out[i], remaining_keep=int(keep))
    return out


def refresh_dualrail_ff(
    engine,
    x: np.ndarray,
    hook: Any | None,
    event_name: str,
) -> np.ndarray:
    """Bootstrap FF live ``(2, 8)`` as 8 complex packs."""
    if x.shape != (2, 8):
        raise ValueError(f"expected (2, 8), got {x.shape}")
    temps = []
    for i in range(8):
        temp = engine.cc_add(x[0, i], engine.imult(x[1, i]))
        temps.append(temp)

    if hook is None:
        temps = [engine.bootstrap(t) for t in temps]
        out = np.full((2, 8), None, dtype=object)
        for i in range(8):
            conj = engine.conjugate(temps[i])
            out[0, i] = engine.mult_scalar(engine.cc_add(temps[i], conj), 0.5)
            out[1, i] = engine.mult_scalar(
                engine.imult(engine.cc_sub(conj, temps[i])), 0.5
            )
        return out

    prev = bool(getattr(hook, "apply_discards", False))
    hook.apply_discards = False
    try:
        temps = list(hook.refresh(event_name, temps))
    finally:
        hook.apply_discards = prev

    out = np.full((2, 8), None, dtype=object)
    for i in range(8):
        conj = engine.conjugate(temps[i])
        out[0, i] = engine.mult_scalar(engine.cc_add(temps[i], conj), 0.5)
        out[1, i] = engine.mult_scalar(
            engine.imult(engine.cc_sub(conj, temps[i])), 0.5
        )
    keep = None
    if prev and hasattr(hook, "remaining_keep_after"):
        keep = hook.remaining_keep_after(event_name)
    if keep is not None and hasattr(hook, "apply_discard"):
        if int(keep) < 12:
            print(
                f"  [dualrail_bts] skip discard @ {event_name}: "
                f"keep={keep}<12 (GeLU-entry needs full depth)",
                flush=True,
            )
        else:
            for i in range(8):
                out[0, i] = hook.apply_discard(out[0, i], remaining_keep=int(keep))
                out[1, i] = hook.apply_discard(out[1, i], remaining_keep=int(keep))
    return out


def refresh_cts(
    engine,
    cts,
    hook: Any | None,
    event_name: str,
):
    """Bootstrap flat cplx CT sequence (``direct`` pack_mode)."""
    seq = list(cts)
    if hook is None:
        return [engine.bootstrap(c) for c in seq]
    return list(hook.refresh(event_name, seq))


def plan_refresh_dualrail8(
    engine,
    packs,
    hook: Any | None,
    event_name: str,
    *,
    scale_before: float | None = None,
    scale_after: float | None = 0.5,
) -> list:
    """
    Plan 驱动的 8-pack 刷新（``real_pack``：8 real→4 cplx→bts→unpack×½）。
    """
    packs = list(packs)
    if hook is None:
        raise RuntimeError(f"plan_refresh_dualrail8 requires hook ({event_name})")
    if not hook.should_refresh(event_name):
        return packs
    return refresh_dualrail8(
        engine,
        packs,
        hook,
        event_name,
        scale_before=scale_before,
        scale_after=scale_after,
    )


def plan_refresh_dualrail_ff(
    engine,
    x: np.ndarray,
    hook: Any | None,
    event_name: str,
) -> np.ndarray:
    """Plan 驱动的 FF (2,8) 刷新（``real_pack`` 路径）。"""
    if hook is None:
        raise RuntimeError(f"plan_refresh_dualrail_ff requires hook ({event_name})")
    if not hook.should_refresh(event_name):
        return x
    return refresh_dualrail_ff(engine, x, hook, event_name)


def unpack_cplx4_to_real8(
    engine,
    cplx4,
    *,
    scale_after: float | None = 0.5,
    hook: Any | None = None,
    event_name: str | None = None,
):
    """
    4 cplx → 8 real（``z+conj`` / ``imult(conj−z)``，默认 ``×½``）。

    用于 ``linear_qkv`` 段首刷完 4 cplx 后回写残差旁路 ``x``。
    """
    cplx4 = list(cplx4)
    if len(cplx4) != 4:
        raise ValueError(f"expected 4 cplx, got {len(cplx4)}")
    out = [None] * 8
    for i in range(4):
        conj = engine.conjugate(cplx4[i])
        out[i] = engine.cc_add(cplx4[i], conj)
        out[i + 4] = engine.imult(engine.cc_sub(conj, cplx4[i]))
    s = 0.5 if scale_after is None else float(scale_after)
    if s != 1.0:
        for i in range(8):
            out[i] = engine.mult_scalar(out[i], s)
    if (
        hook is not None
        and event_name
        and bool(getattr(hook, "apply_discards", False))
        and hasattr(hook, "remaining_keep_after")
        and hasattr(hook, "apply_discard")
    ):
        keep = hook.remaining_keep_after(event_name)
        if keep is not None:
            for i in range(8):
                out[i] = hook.apply_discard(out[i], remaining_keep=int(keep))
    # Do not np.asarray — Liberate CT may expose CUDA __array__ and crash.
    arr = np.empty(8, dtype=object)
    for i in range(8):
        arr[i] = out[i]
    return arr


def plan_refresh_cts(
    engine,
    cts,
    hook: Any | None,
    event_name: str,
):
    """Plan 驱动的复数扁平淡刷新（``direct``）；未放置则原样返回。"""
    if hook is None:
        raise RuntimeError(f"plan_refresh_cts requires hook ({event_name})")
    if not hook.should_refresh(event_name):
        return list(cts)
    return refresh_cts(engine, cts, hook, event_name)


def plan_refresh_an_bn(
    engine,
    an,
    bn,
    hook: Any | None,
    event_name: str,
):
    """
    Plan 驱动的 inv√ / aSOR 双密文刷新。

    - ``direct``：单条或已是复数 spine → 逐条 bootstrap（通常仅 1 条）。
    - ``real_pack``：2 real → 1 cplx bts → unpack×½（落地 rem=budget−1）。
    """
    if hook is None:
        raise RuntimeError(f"plan_refresh_an_bn requires hook ({event_name})")
    if not hook.should_refresh(event_name):
        return an, bn
    if hook.pack_mode(event_name) == "direct":
        out = list(hook.refresh(event_name, [an, bn]))
        if len(out) != 2:
            raise RuntimeError(
                f"plan_refresh_an_bn direct refresh expected 2 CTs, got {len(out)} "
                f"at {event_name}"
            )
        return out[0], out[1]
    an, bn = engine.auto_level(an, bn)
    temp = engine.add(an, engine.imult(bn))
    temp1 = hook.refresh(event_name, temp)
    conj = engine.conjugate(temp1)
    half = 0.5
    an = engine.mult_scalar(engine.add(temp1, conj), half)
    bn = engine.mult_scalar(engine.imult(engine.sub(conj, temp1)), half)
    return an, bn
