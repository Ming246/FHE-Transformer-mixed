#!/usr/bin/env python3
"""
Route B — HE-driven bootstrap planning.

Record **measured** remaining-depth consumption from real Liberate HE forward
(``smoke_thor_repro`` / ``smoke_thor_chain`` with ``--probe-levels``), then
feed ``depth_overrides`` into ``bts_ops.optimize_bootstrap``.

Usage (profile one layer, plan_mock = plan sites + mock refresh)::

  python3 HE/thor/smoke_thor_repro.py --layer 0 --bootstrap plan_mock \\
    --probe-levels --no-rem-guard --depth-profile-out /tmp/L0_depth.json \\
    --softmax-level 2 --ln-level 2 --gelu-level 2 --gelu cheb --ln poly --softmax poly

Merge + build overrides::

  python3 HE/thor/he_depth_profile.py merge /tmp/L0_depth.json /tmp/L1_depth.json \\
    -o /tmp/depth_merged.json

  python3 HE/thor/he_depth_profile.py to-overrides /tmp/depth_merged.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rem_guard import PROBE_TO_EVENT, PREFER_BOOT  # noqa: E402

# Spine segment depth = rem(before_site) − rem(after_site).
# Skip side-path probes (K transpose / complexify / q_rescale) between DP events.
SPINE_SEGMENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("entry_cplx", "after_qkv", "linear_qkv"),
    ("after_qkv", "after_att_score", "linear_att_score"),
    ("after_softmax", "after_att_context", "linear_att_context"),
    ("after_bts_context", "after_attn_dense", "linear_attn_dense"),
    ("after_ln1", "after_ff_bridge_mask", "bridge_rot_ff1"),
    ("after_ff_bridge_mask", "after_ff_dense1", "linear_ff_dense1"),
    ("after_gelu", "after_ff_dense2", "linear_ff_dense2"),
)

# Probe Δrem 与单段 spine 事件 1:1 对齐；勿含 boot 后探针或融合段。
SAFE_PROBE_SUFFIXES: frozenset[str] = frozenset(
    {
        "linear_qkv",
        "linear_att_score",
        "linear_att_context",
        "linear_attn_dense",
        "linear_ff_dense1",
        "linear_ff_dense2",
        "bridge_rot_ff1",
    }
)


@dataclass
class DepthBootRow:
  event_name: str
  rem_before: int
  rem_after: int
  style: str
  ct_count: int
  layer_idx: int


@dataclass
class DepthProfile:
  task: str
  scheme: list[int]
  num_levels: int
  enc_level: int
  layer_indices: list[int]
  boots: list[DepthBootRow] = field(default_factory=list)
  probes: list[dict[str, Any]] = field(default_factory=list)
  # event_name (L{n}.suffix) → measured spine depth (Δrem, no boot land)
  segment_depths: dict[str, int] = field(default_factory=dict)
  # event_name → rem after boot (land rem, for dualrail unpack calibration)
  boot_land_rem: dict[str, int] = field(default_factory=dict)

  def to_json(self) -> str:
    return json.dumps(asdict(self), indent=2, sort_keys=True)

  @classmethod
  def from_json(cls, text: str) -> DepthProfile:
    d = json.loads(text)
    boots = [DepthBootRow(**b) for b in d.pop("boots", [])]
    return cls(boots=boots, **d)

  def save(self, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
      f.write(self.to_json() + "\n")

  @classmethod
  def load(cls, path: str) -> DepthProfile:
    with open(path, encoding="utf-8") as f:
      return cls.from_json(f.read())


class DepthProfileRecorder:
  """Attach to ``BootstrapHook`` + ``LevelProbe`` during HE forward."""

  def __init__(
    self,
    *,
    task: str,
    scheme: list[int],
    num_levels: int,
    enc_level: int,
    layer_idx: int,
  ) -> None:
    self.task = task
    self.scheme = list(scheme)
    self.num_levels = int(num_levels)
    self.enc_level = int(enc_level)
    self.layer_indices = [int(layer_idx)]
    self.boots: list[DepthBootRow] = []
    self.probes: list[dict[str, Any]] = []
    self._hook: Any = None
    self._orig_refresh: Any = None

  def attach(self, hook: Any) -> None:
    self._hook = hook
    self._orig_refresh = hook.refresh

    def _wrapped_refresh(event_name: str, cts, *, indices=None):
      seq = [cts] if not isinstance(cts, (list, tuple)) else list(cts)
      rem_before = None
      if seq and hook.engine is not None:
        try:
          rem_before = int(hook.engine.num_levels) - int(seq[0].level_calc)
        except Exception:
          pass
      out = self._orig_refresh(event_name, cts, indices=indices)
      do_boot = hook.should_refresh(event_name)
      if do_boot and rem_before is not None:
        try:
          oseq = [out] if not isinstance(out, (list, tuple)) else list(out)
          rem_after = int(hook.engine.num_levels) - int(oseq[0].level_calc)
        except Exception:
          rem_after = None
        if rem_after is not None:
          li = _layer_from_event(event_name)
          self.boots.append(
            DepthBootRow(
              event_name=event_name,
              rem_before=int(rem_before),
              rem_after=int(rem_after),
              pack_mode=str(hook.pack_mode(event_name)),
              ct_count=1,
              layer_idx=li,
            )
          )
      return out

    hook.refresh = _wrapped_refresh  # type: ignore[method-assign]

  def detach(self) -> None:
    if self._hook is not None and self._orig_refresh is not None:
      self._hook.refresh = self._orig_refresh  # type: ignore[method-assign]
    self._hook = None
    self._orig_refresh = None

  def ingest_probe(self, probe: Any) -> None:
    self.probes.extend(list(getattr(probe, "samples", []) or []))

  def finalize(self) -> DepthProfile:
    seg = _segment_depths_from_spine_probes(
      self.probes,
      layer_idx=self.layer_indices[0],
    )
    land = {b.event_name: b.rem_after for b in self.boots if b.ct_count > 0}
    return DepthProfile(
      task=self.task,
      scheme=self.scheme,
      num_levels=self.num_levels,
      enc_level=self.enc_level,
      layer_indices=list(self.layer_indices),
      boots=list(self.boots),
      probes=list(self.probes),
      segment_depths=seg,
      boot_land_rem=land,
    )


def _layer_from_event(event_name: str) -> int:
  head = event_name.split(".", 1)[0]
  if head.startswith("L") and head[1:].isdigit():
    return int(head[1:])
  return 0


def _probe_rem_by_site(probes: list[dict[str, Any]]) -> dict[str, int]:
  out: dict[str, int] = {}
  for p in probes:
    site = str(p.get("site", ""))
    if site:
      out[site] = int(p["remaining"])
  return out


def _segment_depths_from_spine_probes(
    probes: list[dict[str, Any]],
    *,
    layer_idx: int,
) -> dict[str, int]:
  """Event-aligned spine depths: Δrem across explicit (before, after) probe pairs."""
  prefix = f"L{int(layer_idx)}."
  rem = _probe_rem_by_site(probes)
  out: dict[str, int] = {}
  for before_site, after_site, suffix in SPINE_SEGMENT_PAIRS:
    if before_site not in rem or after_site not in rem:
      continue
    delta = int(rem[before_site]) - int(rem[after_site])
    if delta <= 0:
      continue
    ev = f"{prefix}{suffix}"
    out[ev] = max(int(out.get(ev, 0)), int(delta))
  return out


def _segment_depths_from_probes(
    probes: list[dict[str, Any]],
    *,
    layer_idx: int,
) -> dict[str, int]:
  """Legacy consecutive-probe mapping (kept for diff / debug)."""
  prefix = f"L{int(layer_idx)}."
  out: dict[str, int] = {}
  for a, b in zip(probes, probes[1:]):
    delta = int(a["remaining"]) - int(b["remaining"])
    if delta <= 0:
      continue
    suf = PROBE_TO_EVENT.get(b["site"])
    if suf is None:
      continue
    if b["site"] in PREFER_BOOT:
      continue
    ev = f"{prefix}{suf}"
    out[ev] = max(int(out.get(ev, 0)), int(delta))
  return out


def _dp_after_rem_and_depth(
    rem_trace: list[Any],
    *,
    layer_idx: int,
) -> tuple[dict[str, int], dict[str, int]]:
  """``event_name → rem_after`` and ``→ depth_charged`` from DP ``after_event`` rows."""
  prefix = f"L{int(layer_idx)}."
  rem_after: dict[str, int] = {}
  depth: dict[str, int] = {}
  for row in rem_trace:
    if getattr(row, "kind", None) != "after_event":
      continue
    name = str(row.event_name)
    if not name.startswith(prefix):
      continue
    rem_after[name] = int(row.rem_after)
    depth[name] = int(row.depth_charged)
  return rem_after, depth


@dataclass
class ProfileValidationRow:
  event_name: str
  measured: int
  dp_depth: int | None
  anchor_ok: bool
  accept: bool
  detail: str = ""


def validate_profile_vs_dp(
    profile: DepthProfile,
    plan_result: Any,
    *,
    layer_idx: int | None = None,
    require_dp_match: bool = False,
    require_anchors: bool = True,
) -> list[ProfileValidationRow]:
  """
  Cross-check spine profile against baseline DP ``rem_trace``.

  ``anchor_ok``: after-probe rem matches DP ``after_event`` rem for that suffix.
  ``accept``: anchors pass and (measured==dp or not ``require_dp_match``).
  """
  li = int(layer_idx if layer_idx is not None else profile.layer_indices[0])
  segments = _effective_segment_depths(profile, layer_idx=li)
  rem_after, dp_depth = _dp_after_rem_and_depth(plan_result.rem_trace, layer_idx=li)
  rem_by_site = _probe_rem_by_site(profile.probes)
  rows: list[ProfileValidationRow] = []

  for before_site, after_site, suffix in SPINE_SEGMENT_PAIRS:
    ev = f"L{li}.{suffix}"
    measured = int(segments.get(ev, 0))
    if measured <= 0:
      continue
    dp_d = dp_depth.get(ev)
    anchor_ok = True
    detail_parts: list[str] = []
    if require_anchors:
      suf_map = PROBE_TO_EVENT.get(after_site)
      if suf_map == suffix or suffix in ("ln2_scale", "ln2_prep"):
        he_after = rem_by_site.get(after_site)
        dp_r = rem_after.get(ev)
        if he_after is not None and dp_r is not None and he_after != dp_r:
          anchor_ok = False
          detail_parts.append(f"probe {after_site} rem={he_after} != DP {dp_r}")
      he_before = rem_by_site.get(before_site)
      if he_before is not None and dp_d is not None:
        pass
    if dp_d is None:
      detail_parts.append("no DP after_event row")
    match = dp_d is not None and measured == int(dp_d)
    accept = anchor_ok and (match or not require_dp_match)
    rows.append(
      ProfileValidationRow(
        event_name=ev,
        measured=measured,
        dp_depth=dp_d,
        anchor_ok=anchor_ok,
        accept=accept,
        detail="; ".join(detail_parts),
      )
    )
  return rows


def _effective_segment_depths(
    profile: DepthProfile,
    *,
    layer_idx: int | None = None,
) -> dict[str, int]:
  if len(profile.layer_indices) > 1 and layer_idx is None:
    if profile.segment_depths:
      return dict(profile.segment_depths)
    out: dict[str, int] = {}
    for li in profile.layer_indices:
      out.update(
        _segment_depths_from_spine_probes(profile.probes, layer_idx=int(li))
      )
    return out
  li = int(layer_idx if layer_idx is not None else profile.layer_indices[0])
  if profile.probes:
    return _segment_depths_from_spine_probes(profile.probes, layer_idx=li)
  return dict(profile.segment_depths)


def merge_profiles(profiles: list[DepthProfile]) -> DepthProfile:
  if not profiles:
    raise ValueError("merge_profiles: empty")
  base = profiles[0]
  merged = DepthProfile(
    task=base.task,
    scheme=list(base.scheme),
    num_levels=base.num_levels,
    enc_level=base.enc_level,
    layer_indices=[],
    boots=[],
    probes=[],
    segment_depths={},
    boot_land_rem={},
  )
  for p in profiles:
    if p.task != merged.task:
      raise ValueError(f"task mismatch {p.task!r} vs {merged.task!r}")
    merged.layer_indices.extend(p.layer_indices)
    merged.boots.extend(p.boots)
    merged.probes.extend(p.probes)
    for li in p.layer_indices:
      merged.segment_depths.update(
        _segment_depths_from_spine_probes(p.probes, layer_idx=int(li))
      )
    merged.boot_land_rem.update(p.boot_land_rem)
  merged.layer_indices = sorted(set(merged.layer_indices))
  return merged


def build_depth_overrides(
    profile: DepthProfile,
    *,
    suffix_fallback: bool = True,
    safe_only: bool = False,
    plan_result: Any | None = None,
    layer_idx: int | None = None,
    validate: bool = False,
    require_dp_match: bool = False,
) -> dict[str, int]:
  """
  ``event_name → depth`` for ``build_bootstrap_events(..., depth_overrides=)``.

  Uses measured ``segment_depths``; optionally copies missing suffixes from
  any profiled layer (same scheme / geometry).

  ``safe_only``: only linear/bridge segments where probe→event mapping is 1:1
  (excludes fused LN/Softmax probe lumps).

  ``validate`` + ``plan_result``: run ``validate_profile_vs_dp`` and only keep
  accepted segments (anchor-aligned; optional ``require_dp_match``).
  """
  raw = _effective_segment_depths(profile, layer_idx=layer_idx)
  if safe_only:
    raw = {
      k: int(v)
      for k, v in raw.items()
      if k.split(".", 1)[-1] in SAFE_PROBE_SUFFIXES
    }
  if validate and plan_result is not None:
    vrows = validate_profile_vs_dp(
      profile,
      plan_result,
      layer_idx=layer_idx,
      require_dp_match=require_dp_match,
    )
    accepted = {r.event_name for r in vrows if r.accept}
    raw = {k: int(v) for k, v in raw.items() if k in accepted}
  overrides = dict(raw)
  if not suffix_fallback:
    return overrides
  by_suffix: dict[str, int] = {}
  for ev, d in overrides.items():
    suf = ev.split(".", 1)[-1]
    by_suffix.setdefault(suf, int(d))
  return {**by_suffix, **overrides}


def optimize_from_profile(
    task: str,
    scheme: list[int],
    profile: DepthProfile | str,
    *,
    budget: int | None = None,
    initial_level: int | None = None,
    phase: str = "C",
) -> Any:
  """``optimize_bootstrap`` with HE-measured ``depth_overrides``."""
  if isinstance(profile, str):
    profile = DepthProfile.load(profile)
  from bts_ops import BOOTSTRAP_DEPTH_BUDGET, optimize_bootstrap

  overrides = build_depth_overrides(profile)
  return optimize_bootstrap(
    task,
    scheme,
    budget=budget or BOOTSTRAP_DEPTH_BUDGET,
    initial_level=initial_level,
    phase=phase,
    depth_overrides=overrides,
  )


def _main() -> int:
  ap = argparse.ArgumentParser(description="HE depth profile utilities (Route B)")
  sub = ap.add_subparsers(dest="cmd", required=True)

  m = sub.add_parser("merge", help="merge profile JSON files")
  m.add_argument("inputs", nargs="+")
  m.add_argument("-o", "--output", required=True)

  t = sub.add_parser("to-overrides", help="print depth_overrides JSON")
  t.add_argument("profile")
  t.add_argument("--safe-only", action="store_true")
  t.add_argument(
    "--validate",
    action="store_true",
    help="require probe anchors vs baseline DP rem_trace",
  )
  t.add_argument(
    "--require-dp-match",
    action="store_true",
    help="with --validate, only keep measured==DP depth_charged",
  )
  t.add_argument("--task", default="mrpc")
  t.add_argument("--layer", type=int, default=0)

  v = sub.add_parser("validate", help="compare profile segments to baseline DP")
  v.add_argument("profile")
  v.add_argument("--task", default="mrpc")
  v.add_argument("--layer", type=int, default=0)
  v.add_argument("--require-dp-match", action="store_true")

  args = ap.parse_args()
  if args.cmd == "merge":
    profiles = [DepthProfile.load(p) for p in args.inputs]
    out = merge_profiles(profiles)
    out.save(args.output)
    print(f"merged {len(profiles)} profiles → {args.output}", flush=True)
    print(
      f"  layers={out.layer_indices} segments={len(out.segment_depths)} "
      f"boots={len(out.boots)}",
      flush=True,
    )
    return 0
  if args.cmd == "to-overrides":
    prof = DepthProfile.load(args.profile)
    plan_result = None
    if args.validate:
      from bts_ops import BOOTSTRAP_DEPTH_BUDGET, optimize_bootstrap

      plan_result = optimize_bootstrap(
        args.task,
        prof.scheme,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        phase="C",
      )
    ov = build_depth_overrides(
      prof,
      safe_only=args.safe_only,
      plan_result=plan_result,
      layer_idx=args.layer,
      validate=args.validate,
      require_dp_match=args.require_dp_match,
    )
    print(json.dumps(ov, indent=2, sort_keys=True))
    return 0
  if args.cmd == "validate":
    from bts_ops import BOOTSTRAP_DEPTH_BUDGET, optimize_bootstrap

    prof = DepthProfile.load(args.profile)
    plan_result = optimize_bootstrap(
      args.task,
      prof.scheme,
      budget=BOOTSTRAP_DEPTH_BUDGET,
      phase="C",
    )
    rows = validate_profile_vs_dp(
      prof,
      plan_result,
      layer_idx=args.layer,
      require_dp_match=args.require_dp_match,
    )
    for r in rows:
      flag = "OK" if r.accept else "SKIP"
      print(
        f"[{flag}] {r.event_name}: measured={r.measured} dp={r.dp_depth} "
        f"anchor={'Y' if r.anchor_ok else 'N'} {r.detail}",
        flush=True,
      )
    return 0
  return 1


if __name__ == "__main__":
  raise SystemExit(_main())
