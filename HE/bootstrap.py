"""Pre/post message scaling around Liberate ``engine.bootstrap``.

Goal: amplify plaintext before bootstrap (better absolute SNR), then restore.

- **Pre** ``×BTS_MSG_SCALE`` via ``mult_int_scalar`` — no rescale, **0 rem**.
- **Post** ``×BTS_MSG_INV`` via ``mult_scalar`` (rescale) — **1 rem**.
  - ``direct``: applied immediately after bootstrap.
  - ``real_pack``: fused into DualRail unpack ``×½`` → single ``×(½·INV)``.

Both pack modes thus land at ``budget − 1`` after a full refresh.
"""
from __future__ import annotations

from typing import Any

# Integer amplify before bts (mult_int_scalar, no depth).
BTS_MSG_SCALE: int = 128
# Restore after bts (mult_scalar + rescale → 1 rem), or fused into DualRail ×½.
BTS_MSG_INV: float = 1.0/128.0
# Rem tax paid once per bootstrap refresh (direct postscale or real_pack unpack).
BTS_POST_DEPTH: int = 1
DUALRAIL_HALF: float = 0.5


def dualrail_unpack_scale(half: float = DUALRAIL_HALF) -> float:
    """Fused DualRail unpack ×½ and message ×INV (one rescale / 1 rem)."""
    return float(half) * float(BTS_MSG_INV)


def bts_prescale(engine: Any, ct: Any) -> Any:
    """``×BTS_MSG_SCALE`` with no depth (integer scalar)."""
    return engine.mult_int_scalar(ct, int(BTS_MSG_SCALE))


def bts_postscale(engine: Any, ct: Any) -> Any:
    """``×BTS_MSG_INV`` with rescale (1 rem)."""
    return engine.mult_scalar(ct, float(BTS_MSG_INV))


def bootstrap_prescaled(engine: Any, ct: Any, *, postscale: bool = True) -> Any:
    """
    Prescale → bootstrap → optional postscale.

    ``postscale=False`` when DualRail unpack will apply ``dualrail_unpack_scale``.
    """
    ct = bts_prescale(engine, ct)
    ct = engine.bootstrap(ct)
    if postscale:
        ct = bts_postscale(engine, ct)
    return ct

"""DualRail pack → bootstrap → unpack helpers (named BootstrapHook sites)."""
from typing import Any

import numpy as np

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
    temps = []
    for i in range(4):
        temp = engine.cc_add(packs[i], engine.imult(packs[i + 4]))
        if scale_before is not None:
            if int(temp.level_calc) < int(engine.num_levels) - 1:
                temp = engine.mult_scalar(temp, float(scale_before))
        temps.append(temp)

    s = _resolve_dualrail_scale_after(scale_after)

    if hook is None:
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
    """Bootstrap FF live ``(2, 8)`` as 8 complex packs (fused unpack×(½·INV))."""
    if x.shape != (2, 8):
        raise ValueError(f"expected (2, 8), got {x.shape}")
    temps = []
    for i in range(8):
        temp = engine.cc_add(x[0, i], engine.imult(x[1, i]))
        temps.append(temp)

    s = dualrail_unpack_scale()

    if hook is None:
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
        return [bootstrap_prescaled(engine, c, postscale=True) for c in seq]
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
    half = dualrail_unpack_scale()
    an = engine.mult_scalar(engine.add(temp1, conj), half)
    bn = engine.mult_scalar(engine.imult(engine.sub(conj, temp1)), half)
    return an, bn

"""Apply ``HE/bts_ops`` placements via ``BootstrapHook`` (Branch A)."""
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Sequence

BootstrapMode = Literal["noop", "always", "record", "plan"]
PackMode = Literal["direct", "real_pack"]


@dataclass
class BootstrapRecord:
    event_name: str
    ct_count: int
    mode: str


def _normalize_pack_mode(value: str | None) -> PackMode:
    """Map legacy ``plain``/``dualrail`` or new ``direct``/``real_pack``."""
    if value in ("real_pack", "dualrail"):
        return "real_pack"
    return "direct"


@dataclass
class BootstrapHook:
    """
    Liberate bootstrap call sites.

    - ``noop`` / ``record``：不刷新（测线性）
    - ``always``：立刻 ``engine.bootstrap``
    - ``plan``：仅当 ``event_name`` 落在 ``plan_events`` 时刷新
      （由 ``bts_ops.optimize_bootstrap`` 的 placements 填充）
    """

    engine: Any
    mode: BootstrapMode = "always"
    records: list[BootstrapRecord] = field(default_factory=list)
    handlers: dict[str, Callable[[Sequence[Any]], Sequence[Any]]] = field(
        default_factory=dict
    )
    plan_events: set[str] = field(default_factory=set)
    plan_pack_modes: dict[str, PackMode] = field(default_factory=dict)
    plan_events_by_name: dict[str, Any] = field(default_factory=dict, repr=False)
    level_discards: list[Any] = field(default_factory=list)
    apply_discards: bool = False
    _keep_after: dict[str, int] = field(default_factory=dict, repr=False)
    rem_guard: Any = None
    # Events that call ``engine.bootstrap`` with no ×BTS_MSG_SCALE / ×INV.
    bare_scale_events: set[str] = field(default_factory=set)

    def load_plan(self, result: Any, *, apply_discards: bool | None = None) -> None:
        """Ingest ``BootstrapResult.placements`` (+ optional ``level_discards``)."""
        self.plan_events = {p.event_name for p in result.placements}
        self.plan_pack_modes = {}
        for p in result.placements:
            raw = getattr(p, "pack_mode", None) or getattr(p, "style", None)
            pm = _normalize_pack_mode(raw)
            prev = self.plan_pack_modes.get(p.event_name)
            if prev == "real_pack" or pm == "real_pack":
                self.plan_pack_modes[p.event_name] = "real_pack"
            else:
                self.plan_pack_modes[p.event_name] = pm
        self.plan_events_by_name = {
            e.name: e for e in getattr(result, "events", []) or []
        }
        self.mode = "plan"
        self.level_discards = list(getattr(result, "level_discards", []) or [])
        if apply_discards is not None:
            self.apply_discards = bool(apply_discards)
        self._keep_after = {}
        for d in self.level_discards:
            if d.reason == "initial":
                self._keep_after["__initial__"] = d.remaining_keep
            elif d.site:
                self._keep_after[d.site] = d.remaining_keep

    def pack_mode(self, event_name: str) -> PackMode:
        """``direct`` (rem=budget) | ``real_pack`` (rem=budget−unpack tax)."""
        if event_name in self.plan_pack_modes:
            return self.plan_pack_modes[event_name]
        ev = self.plan_events_by_name.get(event_name)
        if ev is not None:
            try:
                from bts_ops import pack_mode_for_event
            except ImportError:
                from .bts_ops import pack_mode_for_event  # type: ignore
            return pack_mode_for_event(ev)  # type: ignore[arg-type]
        return "direct"

    def boot_style(self, event_name: str) -> str:
        """Deprecated alias: ``direct``→``plain``, ``real_pack``→``dualrail``."""
        return "dualrail" if self.pack_mode(event_name) == "real_pack" else "plain"

    def remaining_keep_after(self, event_name: str | None = None) -> int | None:
        key = "__initial__" if event_name is None else event_name
        return self._keep_after.get(key)

    def apply_discard(
        self,
        cts: Sequence[Any] | Any,
        *,
        remaining_keep: int,
        indices: Iterable[int] | None = None,
    ) -> Any:
        try:
            from bts_ops import remaining_to_level_calc
        except ImportError:
            from .bts_ops import remaining_to_level_calc  # type: ignore
        from linear import safe_level_up

        target = remaining_to_level_calc(remaining_keep, self.engine.num_levels)
        try:
            from liberate.fhe.data_struct import DataStruct

            is_datastruct = isinstance(cts, DataStruct)
        except Exception:
            is_datastruct = False
        try:
            import numpy as np

            is_ndarray = isinstance(cts, np.ndarray)
        except Exception:
            is_ndarray = False
        wrap_single = is_datastruct or (
            (not is_ndarray) and not isinstance(cts, (list, tuple))
        )
        seq = [cts] if wrap_single else list(cts)
        idxs = list(range(len(seq)) if indices is None else indices)
        out = list(seq)
        for i in idxs:
            if out[i].level_calc < target:
                out[i] = safe_level_up(self.engine, out[i], target)
        if wrap_single:
            return out[0]
        if is_ndarray:
            arr = np.empty(len(out), dtype=object)
            for j, v in enumerate(out):
                arr[j] = v
            return arr
        return out

    def should_refresh(self, event_name: str) -> bool:
        if self.mode in ("noop", "record"):
            return False
        if self.mode == "always":
            return True
        if self.mode == "plan":
            return event_name in self.plan_events
        raise ValueError(f"unknown bootstrap mode: {self.mode!r}")

    def refresh(
        self,
        event_name: str,
        cts: Sequence[Any] | Any,
        *,
        indices: Iterable[int] | None = None,
    ) -> Any:
        try:
            from liberate.fhe.data_struct import DataStruct

            is_datastruct = isinstance(cts, DataStruct)
        except Exception:
            is_datastruct = False

        try:
            import numpy as np

            is_ndarray = isinstance(cts, np.ndarray)
        except Exception:
            is_ndarray = False

        wrap_single = is_datastruct or (
            (not is_ndarray) and not isinstance(cts, (list, tuple))
        )
        if wrap_single:
            seq = [cts]
        else:
            seq = list(cts)

        idxs = list(range(len(seq)) if indices is None else indices)

        do_boot = self.should_refresh(event_name)
        rem_before = None
        if do_boot and seq and self.engine is not None:
            try:
                rem_before = int(self.engine.num_levels) - int(seq[idxs[0]].level_calc)
            except Exception:
                rem_before = None
        self.records.append(
            BootstrapRecord(
                event_name=event_name,
                ct_count=len(idxs) if do_boot else 0,
                mode=self.mode,
            )
        )

        def _to_object_array(items: list) -> Any:
            arr = np.empty(len(items), dtype=object)
            for j, v in enumerate(items):
                arr[j] = v
            return arr

        if event_name in self.handlers:
            out = list(self.handlers[event_name](seq))
            if wrap_single:
                return out[0]
            return _to_object_array(out) if is_ndarray else out

        if not do_boot:
            return cts

        out = list(seq)
        # real_pack: only ×100 here; DualRail unpack fuses ×(½·0.01).
        # direct: ×100 → bts → ×0.01 (pays the 1 rem post tax).
        # bare_scale_events: bootstrap only, no extra multiply.
        bare = event_name in self.bare_scale_events
        do_post = (not bare) and self.pack_mode(event_name) == "direct"
        for i in idxs:
            if bare:
                out[i] = self.engine.bootstrap(out[i])
            else:
                out[i] = bootstrap_prescaled(
                    self.engine, out[i], postscale=do_post
                )
        rem_after = None
        try:
            rem_after = int(self.engine.num_levels) - int(out[idxs[0]].level_calc)
        except Exception:
            rem_after = None
        if rem_before is not None and rem_after is not None:
            print(
                f"  [bts rem] {event_name}: rem {rem_before}→{rem_after} "
                f"pack_mode={self.pack_mode(event_name)} "
                f"prescale={'1' if bare else BTS_MSG_SCALE} "
                f"postscale={'no' if bare else ('yes' if do_post else 'unpack')}",
                flush=True,
            )
            if self.rem_guard is not None:
                self.rem_guard.check_boot(event_name, rem_before, rem_after)
        keep = self._keep_after.get(event_name)
        if (
            self.apply_discards
            and keep is not None
            and self.engine is not None
        ):
            self._maybe_discard(out, idxs, event_name, int(keep))
        if wrap_single:
            return out[0]
        return _to_object_array(out) if is_ndarray else out

    def _maybe_discard(
        self,
        out: list,
        idxs: list[int],
        event_name: str,
        keep: int,
    ) -> None:
        from linear import MAX_LEVEL_UP_SPAN
        try:
            from bts_ops import remaining_to_level_calc
        except ImportError:
            from .bts_ops import remaining_to_level_calc  # type: ignore

        target = remaining_to_level_calc(keep, int(self.engine.num_levels))
        for i in idxs:
            src = int(out[i].level_calc)
            span = target - src
            if span <= 0:
                continue
            if MAX_LEVEL_UP_SPAN is not None and span > int(MAX_LEVEL_UP_SPAN):
                print(
                    f"  [BootstrapHook] skip discard @ {event_name}: "
                    f"span={span}>{MAX_LEVEL_UP_SPAN} "
                    f"(keep={keep}, level_calc {src}→{target})",
                    flush=True,
                )
                continue
            out[i] = self.apply_discard(out[i], remaining_keep=keep)

    def summary(self) -> dict:
        by: dict[str, int] = {}
        for r in self.records:
            by[r.event_name] = by.get(r.event_name, 0) + r.ct_count
        return {
            "mode": self.mode,
            "calls": len(self.records),
            "total_ct_bootstraps": sum(r.ct_count for r in self.records),
            "plan_events": sorted(self.plan_events),
            "by_event": by,
        }
