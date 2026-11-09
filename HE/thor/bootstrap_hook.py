"""Apply ``HE/thor/bts_ops`` placements via ``BootstrapHook`` (Branch A)."""
from __future__ import annotations

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
        try:
            from linear_eval import safe_level_up
        except ImportError:
            from .linear_eval import safe_level_up  # type: ignore

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

        pre_for_probe = [seq[i] for i in idxs]
        out = list(seq)
        try:
            from bts_decrypt_probe import bts_probe_label
        except ImportError:
            from .bts_decrypt_probe import bts_probe_label  # type: ignore
        try:
            from bts_prescale import BTS_MSG_SCALE, bootstrap_prescaled
        except ImportError:
            from .bts_prescale import BTS_MSG_SCALE, bootstrap_prescaled  # type: ignore
        # real_pack: only ×100 here; DualRail unpack fuses ×(½·0.01).
        # direct: ×100 → bts → ×0.01 (pays the 1 rem post tax).
        # bare_scale_events: bootstrap only, no extra multiply.
        bare = event_name in self.bare_scale_events
        do_post = (not bare) and self.pack_mode(event_name) == "direct"
        try:
            from dualrail_bts import plan_bootstrap_kernel
        except ImportError:
            from .dualrail_bts import plan_bootstrap_kernel  # type: ignore
        with bts_probe_label(self.engine, event_name):
            with plan_bootstrap_kernel(self.engine):
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
        coll = getattr(self.engine, "_bts_site_probe", None)
        suppressed = bool(
            getattr(self.engine, "_bts_site_probe_suppress_hook", False)
        )
        pm = self.pack_mode(event_name)
        if coll is not None and not suppressed and pm == "direct":
            try:
                from bts_site_pack_probe import pack_mode_for_probe
            except ImportError:
                from .bts_site_pack_probe import pack_mode_for_probe  # type: ignore
            post_for_probe = [out[i] for i in idxs]
            coll.record_direct_hook_refresh(
                event_name,
                pack_mode_for_probe(self, event_name),
                pre_for_probe,
                post_for_probe,
            )
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
        try:
            from linear_eval import MAX_LEVEL_UP_SPAN
        except ImportError:
            from .linear_eval import MAX_LEVEL_UP_SPAN  # type: ignore
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
