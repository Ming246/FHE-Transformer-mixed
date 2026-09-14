"""
Fail-fast HE remaining-depth vs DP ``rem_trace`` from ``optimize_bootstrap``.

Plan builds a spine rem trajectory; during mock/plan HE, each probe / bts site
compares HE ``num_levels - level_calc`` to the expected rem. First mismatch
raises ``RemMismatchError`` so the graph can be fixed before continuing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


# HE probe site → DP event suffix (after_event rem, unless prefer_boot).
# Only **spine** (main-path) CTs. K/V side-path / 岛内中段探针不映射：
#   after_qkv_v, after_transpose_k, after_k_complexify, after_q_rescale,
#   after_make_copies（现并入 atomic ``linear_att_score``）
#
# ``entry_cplx`` → ``__initial__``：L0 = 全局 spine 起点；L>0 = 本层入口 rem
# （= 上一层出口 / 本层首事件 rem_before），见 ``from_bootstrap_result``。
PROBE_TO_EVENT: dict[str, str] = {
    "entry_cplx": "__initial__",
    "after_qkv": "linear_qkv",
    "after_att_score": "linear_att_score",
    "after_bts_score": "softmax_exp_stockmeyer",
    "after_softmax": "softmax_dualrail_exit",
    "after_att_context": "linear_att_context",
    "after_bts_context": "bridge_rot_attn_dense",
    "after_attn_dense": "linear_attn_dense",
    "after_ln1": "ln1_post_invsqrt",
    # after_bts_bridge_rot_ff1：仅可选 DualRail，DEPTH_FF_BRIDGE 尚未烧
    # （mc_mult 在 after_ff_bridge_mask）；勿对 after_event。
    "after_ff_bridge_mask": "bridge_rot_ff1",
    "after_ff_dense1": "linear_ff_dense1",
    "after_ff_dense1_pre_gelu": "linear_ff_dense1",
    "after_gelu": "gelu_reconstruct",
    "after_bts_post_gelu": "linear_ff_dense2",
    "after_ff_dense2": "linear_ff_dense2",
    "after_ln2": "ln2_post_invsqrt",
}

# After a DualRail / fixed bts probe, prefer rem *after* the boot row.
PREFER_BOOT: frozenset[str] = frozenset(
    {
        "after_bts_score",
        "after_bts_context",
        "after_bts_post_gelu",
    }
)

# HE boots where rem_before is not the DP spine CT:
# （post_invsqrt 现只刷主路 denom，不再 DualRail 刷旁路 num）
BYPASS_BOOT_SUFFIXES: frozenset[str] = frozenset()

# Aliases when plan boots an alternate site name (true micro-events only).
EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "bridge_rot_attn_dense": ("bridge_rot_attn_dense",),
    "softmax_exp_stockmeyer": ("softmax_exp_stockmeyer",),
    "gelu_stockmeyer": (
        "gelu_f1",
        "gelu_f2",
        "gelu_reconstruct",
        "linear_ff_dense2",
    ),
    "gelu_f1": ("gelu_f1", "gelu_f2", "gelu_reconstruct"),
    "gelu_f2": ("gelu_f2", "gelu_reconstruct", "linear_ff_dense2"),
    "gelu_reconstruct": ("gelu_reconstruct", "linear_ff_dense2"),
    "linear_ff_dense2": (
        "linear_ff_dense2",
        "gelu_reconstruct",
        "gelu_f2",
        "gelu_f1",
    ),
    "ln2_prep": ("ln2_scale",),  # 旧名别名
    "ln1_prep": ("ln1_scale",),  # 旧名别名
    "ln1_scale": ("ln1_scale",),
    "ln2_scale": ("ln2_scale",),
}


class RemMismatchError(RuntimeError):
    """HE rem diverged from DP rem_trace — stop and fix the graph."""

    def __init__(
        self,
        *,
        site: str,
        he_rem: int,
        expect_rem: int | None,
        detail: str = "",
    ):
        self.site = site
        self.he_rem = he_rem
        self.expect_rem = expect_rem
        msg = (
            f"REM MISMATCH at {site}: HE rem={he_rem} "
            f"expect={expect_rem}"
        )
        if detail:
            msg = f"{msg} | {detail}"
        super().__init__(msg)


def _layer_entry_rem(rem_trace: Iterable[Any], *, layer_idx: int) -> int | None:
    """DP rem at the start of ``layer_idx`` on the continuous spine."""
    if layer_idx <= 0:
        for row in rem_trace:
            name = getattr(row, "event_name", "") or ""
            kind = getattr(row, "kind", "")
            if name == "__initial__" and kind in ("start", "after_event"):
                return int(getattr(row, "rem_after", 0))
        return None
    prefix = f"L{int(layer_idx)}."
    for row in rem_trace:
        name = getattr(row, "event_name", "") or ""
        if not name.startswith(prefix):
            continue
        return int(getattr(row, "rem_before", 0))
    return None


@dataclass
class RemGuard:
    """Compare HE remaining depth to DP ``BootstrapResult.rem_trace``."""

    layer_idx: int
    budget: int
    tol: int = 0
    rem_trace: list[Any] = field(default_factory=list)
    plan_boot_events: set[str] = field(default_factory=set)
    # suffix → rem_after (last wins)
    after_event: dict[str, int] = field(default_factory=dict)
    after_boot: dict[str, int] = field(default_factory=dict)
    before_boot: dict[str, int] = field(default_factory=dict)
    checked: list[tuple[str, int, int]] = field(default_factory=list)
    enabled: bool = True

    @classmethod
    def from_bootstrap_result(
        cls,
        result: Any,
        *,
        layer_idx: int,
        tol: int = 0,
    ) -> RemGuard:
        prefix = f"L{int(layer_idx)}."
        g = cls(
            layer_idx=int(layer_idx),
            budget=int(getattr(result, "budget", 14)),
            tol=int(tol),
            rem_trace=list(getattr(result, "rem_trace", []) or []),
            plan_boot_events={
                p.event_name
                for p in getattr(result, "placements", [])
                if str(p.event_name).startswith(prefix)
            },
        )
        for row in g.rem_trace:
            name = getattr(row, "event_name", "") or ""
            kind = getattr(row, "kind", "")
            ra = int(getattr(row, "rem_after", 0))
            rb = int(getattr(row, "rem_before", 0))
            if not name.startswith(prefix):
                continue
            suf = name[len(prefix) :]
            if kind == "after_event":
                g.after_event[suf] = ra
                g.after_event[name] = ra
            elif kind == "after_boot":
                g.after_boot[suf] = ra
                g.after_boot[name] = ra
                g.before_boot[suf] = rb
                g.before_boot[name] = rb
            elif kind == "before_boot":
                g.before_boot[suf] = rb
                g.before_boot[name] = rb

        # entry_cplx → ``__initial__``：本层入口 rem（跨层连续，非每层重置为 14）。
        # L0 = 全局 start；L>0 = 本层首行 rem_before（= 上一层出口）。
        entry_rem = _layer_entry_rem(g.rem_trace, layer_idx=int(layer_idx))
        if entry_rem is not None:
            g.after_event["__initial__"] = entry_rem
        return g

    def _lookup(
        self,
        suf: str,
        *,
        prefer_boot: bool = False,
    ) -> tuple[int | None, str]:
        aliases = EVENT_ALIASES.get(suf, (suf,))
        if prefer_boot:
            for a in aliases:
                if a in self.after_boot:
                    return self.after_boot[a], f"after_boot:{a}"
            for a in aliases:
                if a in self.after_event:
                    return self.after_event[a], f"after_event:{a}"
        else:
            for a in aliases:
                if a in self.after_event:
                    return self.after_event[a], f"after_event:{a}"
            for a in aliases:
                if a in self.after_boot:
                    return self.after_boot[a], f"after_boot:{a}"
        return None, "missing"

    def dump_layer_trace(self) -> str:
        prefix = f"L{self.layer_idx}."
        lines = [
            f"DP rem_trace L{self.layer_idx} "
            f"(budget={self.budget}, boots={len(self.plan_boot_events)}):"
        ]
        entry = self.after_event.get("__initial__")
        if entry is not None:
            lines.append(
                f"  {'layer_entry':12} {'__initial__':42} rem {entry}→{entry}"
            )
        for row in self.rem_trace:
            name = getattr(row, "event_name", "") or ""
            if not name.startswith(prefix):
                continue
            kind = getattr(row, "kind", "")
            rb = getattr(row, "rem_before", "")
            ra = getattr(row, "rem_after", "")
            br = getattr(row, "boot_reason", None) or ""
            extra = f" boot={br}" if br else ""
            lines.append(
                f"  {kind:12} {name:42} rem {rb}→{ra}{extra}"
            )
        return "\n".join(lines)

    def check_rem(
        self,
        site: str,
        he_rem: int,
        *,
        expect: int | None = None,
        detail: str = "",
    ) -> None:
        if not self.enabled:
            return
        if expect is None:
            return
        if abs(int(he_rem) - int(expect)) <= self.tol:
            self.checked.append((site, int(he_rem), int(expect)))
            return
        raise RemMismatchError(
            site=site,
            he_rem=int(he_rem),
            expect_rem=int(expect),
            detail=detail,
        )

    def check_probe(self, site: str, he_rem: int) -> None:
        if not self.enabled:
            return
        suf = PROBE_TO_EVENT.get(site)
        if suf is None:
            return
        prefer = site in PREFER_BOOT
        # Pre-Softmax DualRail boot is plan-dependent (L0 rem=2 needs boot; L1 rem=6
        # may skip). When no placement at score/stockmeyer, after_bts_score is still
        # probed before stockmeyer runs — compare to linear_att_score, not post-stockmeyer rem.
        if site == "after_bts_score":
            prefix = f"L{self.layer_idx}."
            pre_softmax = (f"{prefix}softmax_exp_stockmeyer",)
            if not any(ev in self.plan_boot_events for ev in pre_softmax):
                suf = "linear_att_score"
                prefer = False
        expect, src = self._lookup(suf, prefer_boot=prefer)
        self.check_rem(
            site,
            he_rem,
            expect=expect,
            detail=f"DP {src} (event={suf})",
        )
        print(
            f"  [rem_ok] {site}: HE rem={he_rem} == DP {expect} ({src})",
            flush=True,
        )

    def check_boot(
        self,
        event_name: str,
        rem_before: int,
        rem_after: int,
    ) -> None:
        if not self.enabled:
            return
        prefix = f"L{self.layer_idx}."
        if not event_name.startswith(prefix):
            return
        suf = event_name[len(prefix) :]
        # Unexpected reactive / pull-forward boot: still require rem_after ≈ budget
        # but flag if event was never a planned spine boot.
        planned = event_name in self.plan_boot_events
        bypass = suf in BYPASS_BOOT_SUFFIXES
        exp_before = self.before_boot.get(suf)
        if exp_before is None:
            for a in EVENT_ALIASES.get(suf, ()):
                if a in self.before_boot:
                    exp_before = self.before_boot[a]
                    break
        exp_after = self.after_boot.get(suf)
        if exp_after is None:
            for a in EVENT_ALIASES.get(suf, ()):
                if a in self.after_boot:
                    exp_after = self.after_boot[a]
                    break
        if not planned:
            raise RemMismatchError(
                site=f"boot:{event_name}",
                he_rem=int(rem_before),
                expect_rem=exp_before,
                detail=(
                    f"HE boot not in plan placements "
                    f"(rem {rem_before}→{rem_after}); "
                    f"likely pull-forward / safety DualRail"
                ),
            )
        if exp_before is not None and not bypass:
            self.check_rem(
                f"boot_before:{event_name}",
                rem_before,
                expect=exp_before,
                detail="DP before_boot rem",
            )
        elif bypass:
            print(
                f"  [rem_ok] boot {event_name}: rem_before={rem_before} "
                f"(bypass/AUX; skip spine rem_before check)",
                flush=True,
            )
        if exp_after is not None:
            # HE [bts rem] is pre-unpack land (=budget); DP after_boot includes
            # DualRail unpack tax. Accept either.
            if abs(int(rem_after) - int(exp_after)) <= self.tol or abs(
                int(rem_after) - int(self.budget)
            ) <= self.tol:
                self.checked.append(
                    (f"boot_after:{event_name}", int(rem_after), int(exp_after))
                )
            else:
                raise RemMismatchError(
                    site=f"boot_after:{event_name}",
                    he_rem=int(rem_after),
                    expect_rem=int(exp_after),
                    detail=(
                        f"DP after_boot={exp_after} (or land budget={self.budget})"
                    ),
                )
        if not bypass:
            print(
                f"  [rem_ok] boot {event_name}: "
                f"{rem_before}→{rem_after} "
                f"(DP {exp_before}→{exp_after})",
                flush=True,
            )
        else:
            print(
                f"  [rem_ok] boot {event_name}: "
                f"{rem_before}→{rem_after} (bypass; DP spine {exp_before}→{exp_after})",
                flush=True,
            )


def write_rem_trace(result: Any, path: str, *, layer_idx: int | None = None) -> None:
    """Write full or per-layer rem_trace for offline inspection."""
    rows: Iterable[Any] = getattr(result, "rem_trace", []) or []
    lines = [
        "# kind\tevent\trem_before\trem_after\tdepth_charged\tboot_reason\tboot_pack_mode"
    ]
    prefix = f"L{layer_idx}." if layer_idx is not None else None
    for row in rows:
        name = getattr(row, "event_name", "") or ""
        if prefix is not None and name != "__initial__" and not name.startswith(prefix):
            continue
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    getattr(row, "kind", ""),
                    name,
                    getattr(row, "rem_before", ""),
                    getattr(row, "rem_after", ""),
                    getattr(row, "depth_charged", ""),
                    getattr(row, "boot_reason", "") or "",
                    getattr(row, "boot_pack_mode", None)
                    or getattr(row, "boot_style", "")
                    or "",
                )
            )
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
