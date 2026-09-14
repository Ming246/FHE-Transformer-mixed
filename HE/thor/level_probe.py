"""Record Liberate ``level_calc`` / remaining at named HE forward sites."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LevelProbe:
    """Collect remaining-depth samples; optional early stop after a site."""

    num_levels: int
    stop_after: str | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)

    def remaining(self, ct) -> int:
        return int(self.num_levels) - int(ct.level_calc)

    def record(self, site: str, ct, *, note: str = "") -> None:
        rem = self.remaining(ct)
        row = {
            "site": site,
            "level_calc": int(ct.level_calc),
            "remaining": rem,
            "note": note,
        }
        self.samples.append(row)
        extra = f" ({note})" if note else ""
        print(
            f"  [probe] {site}: level_calc={row['level_calc']} "
            f"remaining={rem}{extra}",
            flush=True,
        )

    def record_first(self, site: str, cts, *, note: str = "") -> None:
        seq = list(cts)
        if not seq:
            raise ValueError(f"empty cts at {site}")
        self.record(site, seq[0], note=note)

    def should_stop(self, site: str) -> bool:
        return self.stop_after is not None and site == self.stop_after

    def deltas(self) -> list[tuple[str, str, int]]:
        """Consecutive remaining drops (positive = depth consumed)."""
        out: list[tuple[str, str, int]] = []
        for a, b in zip(self.samples, self.samples[1:]):
            out.append((a["site"], b["site"], a["remaining"] - b["remaining"]))
        return out

    def summary(self) -> str:
        lines = ["level_probe summary (site, level_calc, remaining):"]
        for s in self.samples:
            lines.append(
                f"  {s['site']:40s} lc={s['level_calc']:3d} rem={s['remaining']:3d}"
                + (f"  # {s['note']}" if s["note"] else "")
            )
        lines.append("deltas (rem_before - rem_after):")
        for a, b, d in self.deltas():
            lines.append(f"  {a:28s} → {b:28s}  Δrem={d}")
        return "\n".join(lines)


class ProbeStop(Exception):
    """Raised when ``LevelProbe.stop_after`` is hit (clean early exit)."""

    def __init__(self, site: str, probe: LevelProbe, cts=None):
        super().__init__(site)
        self.site = site
        self.probe = probe
        self.cts = cts
