"""DualRail pack → bootstrap → unpack helpers (named BootstrapHook sites)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

try:
    from bts_prescale import (
        bootstrap_prescaled,
        dualrail_unpack_scale,
    )
except ImportError:
    from .bts_prescale import (  # type: ignore
        bootstrap_prescaled,
        dualrail_unpack_scale,
    )


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

    Only swaps the bootstrap kernel. Callers still do ``bts_prescale`` /
    DualRail unpack outside; effective rem after a full refresh is
    ``rem_after − 1`` (postscale or fused unpack×(½·INV)).

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


def install_event_mock_bootstrap(
    engine,
    sk,
    pk,
    event_suffixes: list[str],
    *,
    land_level: int | None = None,
    rem_after: int = 14,
) -> dict:
    """Real ``engine.bootstrap``, except named plan events use the mock kernel.

    Match is on ``engine._bts_probe_label`` (set by ``BootstrapHook.refresh``):
    label equals the suffix or ends with ``.{suffix}``. A suffix ending in
    ``_`` also matches when that text occurs anywhere in the label
    (``softmax_`` covers every softmax micro-event, including
    ``L2.softmax_exp_stockmeyer``). Prescale / postscale stay on the caller
    path, so landed rem matches a real boot.
    """
    suffixes = tuple(str(s) for s in event_suffixes if str(s))
    if not suffixes:
        raise ValueError("event_suffixes is empty")
    mock_info = install_mock_bootstrap(
        engine, sk, pk, land_level=land_level, rem_after=int(rem_after)
    )
    real_bs = mock_info["real_bootstrap"]
    mock_bs = engine.bootstrap
    calls = {"mock": 0, "real": 0}

    def _hit(label: str) -> bool:
        for s in suffixes:
            if label == s or label.endswith("." + s):
                return True
            if s.endswith("_") and s in label:
                return True
        return False

    def _select(ct):
        label = str(getattr(engine, "_bts_probe_label", None) or "")
        if _hit(label):
            calls["mock"] += 1
            if calls["mock"] == 1 or label not in _select.seen:  # type: ignore[attr-defined]
                _select.seen.add(label)  # type: ignore[attr-defined]
                print(f"  [bts kernel] {label} MOCK", flush=True)
            return mock_bs(ct)
        calls["real"] += 1
        return real_bs(ct)

    _select.seen = set()  # type: ignore[attr-defined]
    engine.bootstrap = _select  # type: ignore[method-assign]
    info = {
        **mock_info,
        "real_bootstrap": real_bs,
        "mock_bootstrap": mock_bs,
        "calls": calls,
        "event_suffixes": suffixes,
        "event_mock": True,
    }
    engine._mock_bootstrap_info = info  # type: ignore[attr-defined]
    return info


def uninstall_mock_bootstrap(engine) -> None:
    info = getattr(engine, "_mock_bootstrap_info", None)
    if info is None:
        return
    engine.bootstrap = info["real_bootstrap"]  # type: ignore[method-assign]
    delattr(engine, "_mock_bootstrap_info")


def install_hybrid_plan_bootstrap(
    engine,
    sk,
    pk,
    *,
    land_level: int | None = None,
    rem_after: int = 14,
) -> dict:
    """
    ``BootstrapHook.refresh`` (plan sites) → real bootstrap; all other
    ``engine.bootstrap`` (旁路 ``align_bypass_to_main``, force refresh, …) → mock.
    """
    mock_info = install_mock_bootstrap(
        engine, sk, pk, land_level=land_level, rem_after=int(rem_after)
    )
    real_bs = mock_info["real_bootstrap"]
    mock_bs = engine.bootstrap
    calls = {"plan": 0, "mock": 0}

    def _hybrid(ct):
        if getattr(engine, "_bts_use_plan_kernel", False):
            calls["plan"] += 1
            return real_bs(ct)
        calls["mock"] += 1
        return mock_bs(ct)

    engine.bootstrap = _hybrid  # type: ignore[method-assign]
    info = {
        **mock_info,
        "real_bootstrap": real_bs,
        "mock_bootstrap": mock_bs,
        "calls": calls,
        "hybrid": True,
    }
    engine._hybrid_bootstrap_info = info  # type: ignore[attr-defined]
    return info


@contextmanager
def plan_bootstrap_kernel(engine: Any) -> Iterator[None]:
    """While active, ``engine.bootstrap`` uses the real kernel (hybrid mode)."""
    prev = bool(getattr(engine, "_bts_use_plan_kernel", False))
    engine._bts_use_plan_kernel = True  # type: ignore[attr-defined]
    try:
        yield
    finally:
        engine._bts_use_plan_kernel = prev  # type: ignore[attr-defined]


def _resolve_dualrail_scale_after(scale_after: float | None) -> float:
    """None → fused ×(½·INV); explicit value used as-is (full post scale)."""
    if scale_after is None:
        return dualrail_unpack_scale()
    return float(scale_after)


def refresh_dualrail8(
    engine,
    packs,
    hook: Any | None,
    event_name: str,
    *,
    scale_before: float | None = None,
    scale_after: float | None = None,
) -> list:
    """
    Bootstrap 8 real DualRail packs as 4 complex CTs.

    解包恒等：``z+conj=2Re``，默认 ``scale_after=½·INV``（融合消息恢复）。
    刷前 ``×100``（无深度）在 ``hook.refresh`` / companion 路径内完成。
    ``hook is None``：强制 bootstrap（旁路伴侣，不计 plan）。
    否则走 ``hook.refresh``（plan / always / noop）。
    """
    packs = list(packs)
    if len(packs) != 8:
        raise ValueError(f"expected 8 packs, got {len(packs)}")
    try:
        from bts_site_pack_probe import active_collector, probe_collector
    except ImportError:
        from .bts_site_pack_probe import active_collector, probe_collector  # type: ignore
    _coll = active_collector(engine)
    _pre_real8 = list(packs) if _coll is not None else None
    _site_probe = probe_collector(engine, hook, event_name)
    temps = []
    for i in range(4):
        temp = engine.cc_add(packs[i], engine.imult(packs[i + 4]))
        if scale_before is not None:
            if int(temp.level_calc) < int(engine.num_levels) - 1:
                temp = engine.mult_scalar(temp, float(scale_before))
        temps.append(temp)

    s = _resolve_dualrail_scale_after(scale_after)

    if hook is None:
        try:
            from bts_decrypt_probe import bts_probe_label
        except ImportError:
            from .bts_decrypt_probe import bts_probe_label  # type: ignore
        with bts_probe_label(engine, f"companion:{event_name}"):
            temps = [
                bootstrap_prescaled(engine, t, postscale=False) for t in temps
            ]
        out = list(packs)
        for i in range(4):
            conj = engine.conjugate(temps[i])
            out[i] = engine.cc_add(temps[i], conj)
            out[i + 4] = engine.imult(engine.cc_sub(conj, temps[i]))
        for i in range(8):
            out[i] = engine.mult_scalar(out[i], s)
        if _coll is not None and _pre_real8 is not None:
            _coll.record_ct_sequence(
                f"bypass.companion:{event_name}",
                "real_pack",
                f"real_pack_{len(_pre_real8)}",
                _pre_real8,
                out,
            )
        return out

    prev = bool(getattr(hook, "apply_discards", False))
    hook.apply_discards = False
    try:
        from bts_site_pack_probe import pack_mode_for_probe, suppress_hook_site_probe
    except ImportError:
        from .bts_site_pack_probe import (  # type: ignore
            pack_mode_for_probe,
            suppress_hook_site_probe,
        )
    with suppress_hook_site_probe(engine):
        try:
            temps = list(hook.refresh(event_name, temps))
        finally:
            hook.apply_discards = prev

    out = list(packs)
    for i in range(4):
        conj = engine.conjugate(temps[i])
        out[i] = engine.cc_add(temps[i], conj)
        out[i + 4] = engine.imult(engine.cc_sub(conj, temps[i]))
    for i in range(8):
        out[i] = engine.mult_scalar(out[i], s)
    keep = None
    if prev and hasattr(hook, "remaining_keep_after"):
        keep = hook.remaining_keep_after(event_name)
    if keep is not None and hasattr(hook, "apply_discard"):
        for i in range(8):
            out[i] = hook.apply_discard(out[i], remaining_keep=int(keep))
    if _site_probe is not None and _pre_real8 is not None:
        _site_probe.record_ct_sequence(
            event_name,
            pack_mode_for_probe(hook, event_name),
            f"real_pack_{len(_pre_real8)}",
            _pre_real8,
            out,
        )
    return out


def refresh_dualrail_ff(
    engine,
    x: np.ndarray,
    hook: Any | None,
    event_name: str,
) -> np.ndarray:
    """Bootstrap FF live ``(2, 8)`` as 8 complex packs (fused unpack×(½·INV))."""
    if x.shape != (2, 8):
        raise ValueError(f"expected (2, 8), got {x.shape}")
    try:
        from bts_site_pack_probe import active_collector, probe_collector
    except ImportError:
        from .bts_site_pack_probe import active_collector, probe_collector  # type: ignore
    _coll = active_collector(engine)
    _pre_ff = np.array(x, copy=True) if _coll is not None else None
    _site_probe = probe_collector(engine, hook, event_name)
    temps = []
    for i in range(8):
        temp = engine.cc_add(x[0, i], engine.imult(x[1, i]))
        temps.append(temp)

    s = dualrail_unpack_scale()

    if hook is None:
        try:
            from bts_decrypt_probe import bts_probe_label
        except ImportError:
            from .bts_decrypt_probe import bts_probe_label  # type: ignore
        with bts_probe_label(engine, f"companion:{event_name}"):
            temps = [
                bootstrap_prescaled(engine, t, postscale=False) for t in temps
            ]
        out = np.full((2, 8), None, dtype=object)
        for i in range(8):
            conj = engine.conjugate(temps[i])
            out[0, i] = engine.mult_scalar(engine.cc_add(temps[i], conj), s)
            out[1, i] = engine.mult_scalar(
                engine.imult(engine.cc_sub(conj, temps[i])), s
            )
        if _coll is not None and _pre_ff is not None:
            _coll.record_dualrail_ff(
                f"bypass.companion:{event_name}",
                "real_pack",
                _pre_ff,
                out,
            )
        return out

    prev = bool(getattr(hook, "apply_discards", False))
    hook.apply_discards = False
    try:
        from bts_site_pack_probe import pack_mode_for_probe, suppress_hook_site_probe
    except ImportError:
        from .bts_site_pack_probe import (  # type: ignore
            pack_mode_for_probe,
            suppress_hook_site_probe,
        )
    with suppress_hook_site_probe(engine):
        try:
            temps = list(hook.refresh(event_name, temps))
        finally:
            hook.apply_discards = prev

    out = np.full((2, 8), None, dtype=object)
    for i in range(8):
        conj = engine.conjugate(temps[i])
        out[0, i] = engine.mult_scalar(engine.cc_add(temps[i], conj), s)
        out[1, i] = engine.mult_scalar(
            engine.imult(engine.cc_sub(conj, temps[i])), s
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
    if _site_probe is not None and _pre_ff is not None:
        _site_probe.record_dualrail_ff(
            event_name, pack_mode_for_probe(hook, event_name), _pre_ff, out
        )
    return out


def refresh_cts(
    engine,
    cts,
    hook: Any | None,
    event_name: str,
):
    """Bootstrap flat cplx CT sequence (``direct`` pack_mode; ×100→bts→×0.01)."""
    seq = list(cts)
    if hook is None:
        try:
            from bts_decrypt_probe import bts_probe_label
        except ImportError:
            from .bts_decrypt_probe import bts_probe_label  # type: ignore
        try:
            from bts_site_pack_probe import active_collector
        except ImportError:
            from .bts_site_pack_probe import active_collector  # type: ignore
        coll = active_collector(engine)
        pre = list(seq)
        with bts_probe_label(engine, f"companion:{event_name}"):
            out = [
                bootstrap_prescaled(engine, c, postscale=True) for c in seq
            ]
        if coll is not None:
            coll.record_ct_sequence(
                f"bypass.companion:{event_name}",
                "direct",
                f"direct_{len(pre)}",
                pre,
                out,
            )
        return out
    return list(hook.refresh(event_name, seq))


def plan_refresh_dualrail8(
    engine,
    packs,
    hook: Any | None,
    event_name: str,
    *,
    scale_before: float | None = None,
    scale_after: float | None = None,
) -> list:
    """
    Plan 驱动的 8-pack 刷新（``real_pack``：8 real→4 cplx→bts→unpack×(½·INV)）。
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

    用于 ``linear_qkv`` 段首 **direct** 刷完 4 cplx 后回写残差 ``x``。
    消息 ×INV 已在 ``hook.refresh``（direct postscale）完成，此处只做 DualRail ×½。
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

    - ``direct``：逐条 bootstrap（×100→bts→×0.01）。
    - ``real_pack``：2 real → 1 cplx bts → unpack×(½·INV)。
    """
    if hook is None:
        raise RuntimeError(f"plan_refresh_an_bn requires hook ({event_name})")
    if not hook.should_refresh(event_name):
        return an, bn
    try:
        from bts_site_pack_probe import (
            pack_mode_for_probe,
            probe_collector,
            suppress_hook_site_probe,
        )
    except ImportError:
        from .bts_site_pack_probe import (  # type: ignore
            pack_mode_for_probe,
            probe_collector,
            suppress_hook_site_probe,
        )
    _site_probe = probe_collector(engine, hook, event_name)
    pm = pack_mode_for_probe(hook, event_name)
    if hook.pack_mode(event_name) == "direct":
        out = list(hook.refresh(event_name, [an, bn]))
        if len(out) != 2:
            raise RuntimeError(
                f"plan_refresh_an_bn direct refresh expected 2 CTs, got {len(out)} "
                f"at {event_name}"
            )
        return out[0], out[1]
    an, bn = engine.auto_level(an, bn)
    _an_pre, _bn_pre = an, bn
    temp = engine.add(an, engine.imult(bn))
    with suppress_hook_site_probe(engine):
        temp1 = hook.refresh(event_name, temp)
    conj = engine.conjugate(temp1)
    half = dualrail_unpack_scale()
    an = engine.mult_scalar(engine.add(temp1, conj), half)
    bn = engine.mult_scalar(engine.imult(engine.sub(conj, temp1)), half)
    if _site_probe is not None:
        _site_probe.record_real_pair(event_name, pm, _an_pre, _bn_pre, an, bn)
    return an, bn
