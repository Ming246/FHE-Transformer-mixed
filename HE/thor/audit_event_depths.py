#!/usr/bin/env python3
"""
Verify micro-event ``Event.depth`` vs HE spine Δrem.

Offline (DP table only)::

  python3 HE/thor/audit_event_depths.py --task mrpc --layer 0

With HE probe log (from smoke_thor_repro --probe-levels)::

  python3 HE/thor/audit_event_depths.py --task mrpc --layer 0 \\
    --he-log /tmp/l0_depth.log

``1:1`` rows: single micro-event with probe pair.
``sum`` rows: fused nonlinear block (probes only at block ends).
Mid-segment ``[bts rem]`` is folded into effective Δrem.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from bts_ops import build_bootstrap_events  # noqa: E402
from cost import NUM_LAYERS  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402


# (probe_before_candidates, probe_after, mode, suffixes)
# before: first present probe wins (prefer post-boot site when plan refreshed).
MEASURABLE: tuple[tuple[tuple[str, ...], str, str, tuple[str, ...]], ...] = (
    (("entry_cplx",), "after_qkv", "1:1", ("linear_qkv",)),
    (("after_qkv",), "after_att_score", "1:1", ("linear_att_score",)),
    (("after_bts_score",), "after_softmax", "sum", ("softmax_",)),
    (("after_softmax",), "after_att_context", "1:1", ("linear_att_context",)),
    (
        ("after_bts_context", "after_att_context"),
        "after_attn_dense",
        "1:1",
        ("linear_attn_dense",),
    ),
    (("after_attn_dense",), "after_ln1", "sum", ("ln1_",)),
    (
        ("after_bts_bridge_rot_ff1", "after_ln1"),
        "after_ff_bridge_mask",
        "1:1",
        ("bridge_rot_ff1",),
    ),
    (("after_ff_bridge_mask",), "after_ff_dense1", "1:1", ("linear_ff_dense1",)),
    (
        ("after_ff_dense1", "after_ff_dense1_pre_gelu"),
        "after_gelu",
        "sum",
        ("gelu_",),
    ),
    (
        ("after_bts_post_gelu", "after_gelu"),
        "after_ff_dense2",
        "1:1",
        ("linear_ff_dense2",),
    ),
    (("after_ff_dense2",), "after_ln2", "sum", ("ln2_",)),
)


def _scheme(*, softmax=2, ln=2, gelu=2) -> list[int]:
    out: list[int] = []
    for li in range(NUM_LAYERS):
        g = gelu if gelu_level_allowed(li, gelu) else 2
        out.extend([softmax, ln, g, ln])
    return out


def _parse_probes(text: str) -> dict[str, int]:
    rem: dict[str, int] = {}
    for m in re.finditer(
        r"\[probe\]\s+(\S+):\s+level_calc=\d+\s+remaining=(\d+)", text
    ):
        rem[m.group(1)] = int(m.group(2))
    return rem


def _parse_bts_rem(text: str) -> list[tuple[str, int, int, str]]:
    rows: list[tuple[str, int, int, str]] = []
    for m in re.finditer(
        r"\[bts rem\]\s+(\S+):\s+rem\s+(\d+)→(\d+)\s+(?:pack_mode|style)=(\S+)", text
    ):
        rows.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    return rows


def _land_rem_after_boot(rem_after_log: int, pack_mode: str, *, budget: int = 14) -> int:
    """
    ``[bts rem]`` is logged inside ``BootstrapHook.refresh`` *before* real_pack
    unpack×½. DP / continuing HE spine land at ``budget - 1`` for real_pack.
    """
    ra = int(rem_after_log)
    pm = pack_mode
    if pm in ("real_pack", "dualrail") and ra >= int(budget):
        return int(budget) - 1
    return ra


def _effective_delta(
    rem_before: int,
    rem_after: int,
    boots: list[tuple[str, int, int, str]],
    *,
    budget: int = 14,
) -> int:
    """
    Spine depth burned between two probes, counting rem drops across mid-boots.

    DualRail boots use post-unpack land rem (budget−1), not the raw log land.
    """
    if not boots:
        return int(rem_before) - int(rem_after)
    total = 0
    cur = int(rem_before)
    for _name, rb, ra, st in boots:
        total += max(0, cur - int(rb))
        cur = _land_rem_after_boot(ra, st, budget=budget)
    total += max(0, cur - int(rem_after))
    return total


def _boots_for_keys(
    bts_rows: list[tuple[str, int, int, str]],
    *,
    layer: int,
    keys: tuple[str, ...],
) -> list[tuple[str, int, int, str]]:
    pref = f"L{int(layer)}."
    out: list[tuple[str, int, int, str]] = []
    for name, rb, ra, st in bts_rows:
        if not name.startswith(pref):
            continue
        suf = name[len(pref) :]
        if any(suf == k or suf.startswith(k) for k in keys):
            out.append((name, rb, ra, st))
    return out


def _event_depth_map(events, layer: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        if e.layer_idx != layer:
            continue
        suf = e.name.split(".", 1)[1]
        out[suf] = int(e.depth)
    return out


def _sum_depths(
    depth_by_suf: dict[str, int], keys: tuple[str, ...]
) -> tuple[int, list[str]]:
    hit: list[str] = []
    total = 0
    for suf, d in depth_by_suf.items():
        if any(suf == k or suf.startswith(k) for k in keys):
            hit.append(f"{suf}={d}")
            total += d
    return total, hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--softmax-level", type=int, default=2)
    ap.add_argument("--ln-level", type=int, default=2)
    ap.add_argument("--gelu-level", type=int, default=2)
    ap.add_argument("--he-log", type=str, default=None)
    ap.add_argument("--tol", type=int, default=1)
    args = ap.parse_args()

    scheme = _scheme(
        softmax=args.softmax_level, ln=args.ln_level, gelu=args.gelu_level
    )
    events = build_bootstrap_events(args.task, scheme, phase="C")
    depth_by = _event_depth_map(events, args.layer)

    print(f"=== L{args.layer} DP micro-event depths (phase C) ===", flush=True)
    for e in events:
        if e.layer_idx != args.layer:
            continue
        suf = e.name.split(".", 1)[1]
        print(f"  {suf:36s} depth={e.depth:3d} kind={e.kind}", flush=True)

    he_rem: dict[str, int] = {}
    bts_rows: list[tuple[str, int, int, str]] = []
    if args.he_log:
        with open(args.he_log, encoding="utf-8", errors="replace") as f:
            text = f.read()
        he_rem = _parse_probes(text)
        bts_rows = _parse_bts_rem(text)
        print(
            f"\n=== HE probes from {args.he_log} "
            f"({len(he_rem)} sites, {len(bts_rows)} bts) ===",
            flush=True,
        )
        for k in sorted(he_rem):
            print(f"  {k:28s} rem={he_rem[k]}", flush=True)
        for name, rb, ra, st in bts_rows:
            print(f"  [bts] {name}: {rb}→{ra} ({st})", flush=True)

    print("\n=== Measured vs declared ===", flush=True)
    n_ok = n_miss = n_bad = 0
    for befores, after, mode, keys in MEASURABLE:
        dp_d, parts = _sum_depths(depth_by, keys)
        label = "+".join(keys) if mode == "sum" else keys[0]
        if not args.he_log:
            print(
                f"  [DP-only] {befores[0]}→{after}  {label}: declared={dp_d}  "
                f"({' '.join(parts) if mode == 'sum' else ''})",
                flush=True,
            )
            continue
        before = next((b for b in befores if b in he_rem), None)
        if before is None or after not in he_rem:
            print(
                f"  [MISS] {befores}→{after}  {label}: declared={dp_d}",
                flush=True,
            )
            n_miss += 1
            continue
        boots = _boots_for_keys(bts_rows, layer=args.layer, keys=keys)
        measured = _effective_delta(he_rem[before], he_rem[after], boots)
        diff = measured - dp_d
        ok = abs(diff) <= int(args.tol)
        flag = "OK" if ok else "BAD"
        if ok:
            n_ok += 1
        else:
            n_bad += 1
        boot_note = f" boots={len(boots)}" if boots else ""
        print(
            f"  [{flag}] {before}→{after}  {label}: "
            f"HE_eff={measured} declared={dp_d} diff={diff:+d}  "
            f"(rem {he_rem[before]}→{he_rem[after]}{boot_note})",
            flush=True,
        )
        if mode == "sum" and parts:
            print(f"         parts: {', '.join(parts)}", flush=True)

    if args.he_log:
        print(f"\nsummary: ok={n_ok} bad={n_bad} miss={n_miss}", flush=True)
        return 0 if n_bad == 0 else 2
    print(
        "\n(no --he-log: DP table only. Run smoke with --probe-levels then re-run.)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
