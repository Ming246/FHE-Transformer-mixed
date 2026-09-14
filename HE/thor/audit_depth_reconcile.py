#!/usr/bin/env python3
"""
Reconcile ``bts_ops`` DP remaining-depth trace vs HE probe / ``[bts rem]`` logs.

Offline (no GPU)::

  python3 HE/thor/audit_depth_reconcile.py --task mrpc --layers 0

With HE log from ``smoke_thor_repro --bootstrap plan_mock --probe-levels``::

  python3 HE/thor/audit_depth_reconcile.py --he-log /tmp/l0_plan_mock.log --layers 0
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

from bts_ops import (  # noqa: E402
    BOOTSTRAP_DEPTH_BUDGET,
    build_bootstrap_events,
    optimize_bootstrap,
    simulate_remaining_trace,
)
from cost import NUM_LAYERS  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402

# Probe site → nearest DP event (suffix) for rem comparison after that step.
PROBE_TO_EVENT = {
    "entry_cplx": "linear_qkv",
    "after_qkv": "linear_qkv",
    "after_att_score": "linear_att_score",
    # HE Softmax-entry DualRail = before softmax_exp_stockmeyer; rem after boot ≈ budget.
    "after_bts_score": "softmax_exp_stockmeyer",
    "after_softmax": "softmax_dualrail_exit",
    "after_att_context": "linear_att_context",
    # plan may boot bridge_rot_attn_dense
    "after_bts_context": "bridge_rot_attn_dense",
    "after_attn_dense": "linear_attn_dense",
    "after_ln1": "ln1_invsqrt_",  # last ln1 invsqrt*
    "after_bts_bridge_rot_ff1": "bridge_rot_ff1",
    "after_ff_bridge_mask": "bridge_rot_ff1",
    "after_ff_dense1": "linear_ff_dense1",
    # GeLU boots are inside he_gelu_cheb (gelu_f1 / gelu_f2 / gelu_reconstruct)
    "after_gelu": "gelu_reconstruct",
    "after_ff_dense2": "linear_ff_dense2",
    "after_ln2": "ln2_invsqrt_",
}

# Sites where HE remaps DP mid-boot to *before* the op — rem after HE ≠ DP after_event.
REMAPPED_BOOT_SITES = frozenset(
    {
        "after_bts_score",
        "after_gelu",
    }
)

# Segment Δrem (probe_a → probe_b) vs sum of DP depths between events.
SEGMENT_CHECKS = (
    ("entry_cplx", "after_att_score", "pre_softmax_linear"),
    ("after_bts_score", "after_softmax", "softmax_body"),
    ("after_softmax", "after_att_context", "att_context"),
    ("after_bts_context", "after_attn_dense", "attn_dense"),
    ("after_attn_dense", "after_ln1", "ln1"),
    ("after_ln1", "after_ff_dense1", "ff_bridge_dense1"),
    ("after_ff_dense1", "after_gelu", "gelu"),
    ("after_gelu", "after_ff_dense2", "dense2"),
    ("after_ff_dense2", "after_ln2", "ln2"),
)


def _scheme(
    *,
    softmax_level: int = 2,
    ln_level: int = 2,
    gelu_level: int = 2,
) -> list[int]:
    out: list[int] = []
    for li in range(NUM_LAYERS):
        g = gelu_level if gelu_level_allowed(li, gelu_level) else 2
        out.extend([softmax_level, ln_level, g, ln_level])
    return out


def _parse_probe(log: str) -> dict[str, int]:
    """site → remaining (last sample wins)."""
    rem: dict[str, int] = {}
    for m in re.finditer(
        r"\[probe\]\s+(\S+):\s+level_calc=\d+\s+remaining=(\d+)", log
    ):
        rem[m.group(1)] = int(m.group(2))
    return rem


def _parse_bts_rem(log: str) -> list[tuple[str, int, int, str]]:
    """event, rem_before, rem_after, style."""
    rows = []
    for m in re.finditer(
        r"\[bts rem\]\s+(\S+):\s+rem\s+(\d+)→(\d+)\s+(?:pack_mode|style)=(\S+)", log
    ):
        rows.append((m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)))
    return rows


def _after_event_rem(trace, events, layer: int) -> dict[str, int]:
    """event suffix → rem_after last after_event for that name on layer."""
    out: dict[str, int] = {}
    for row in trace:
        if row.kind != "after_event" or row.event_index is None:
            continue
        ev = events[row.event_index]
        if ev.layer_idx != layer:
            continue
        suf = ev.name.split(".", 1)[1]
        out[suf] = row.rem_after
        out[ev.name] = row.rem_after
    return out


def _after_boot_rem(trace, events, layer: int) -> dict[str, int]:
    """event suffix → rem_after last after_boot for that name on layer."""
    out: dict[str, int] = {}
    for row in trace:
        if row.kind != "after_boot" or row.event_index is None:
            continue
        ev = events[row.event_index]
        if ev.layer_idx != layer:
            continue
        suf = ev.name.split(".", 1)[1]
        out[suf] = row.rem_after
        out[ev.name] = row.rem_after
    return out


def _lookup_rem(
    after: dict[str, int],
    boot: dict[str, int],
    suf: str,
    layer: int,
    *,
    prefer_boot: bool = False,
) -> int | None:
    if suf.endswith("_"):
        return _match_ln_last(after, suf, layer)
    if prefer_boot and suf in boot:
        return boot[suf]
    if suf in after:
        return after[suf]
    if suf in boot:
        return boot[suf]
    # aliases among true micro-events
    if suf == "bridge_rot_attn_dense":
        for alt in ("bridge_rot_attn_dense",):
            if prefer_boot and alt in boot:
                return boot[alt]
            if alt in after:
                return after[alt]
    if suf in (
        "gelu_f1",
        "gelu_f2",
        "gelu_reconstruct",
        "gelu_stockmeyer",
    ):
        for alt in (
            "gelu_f1",
            "gelu_f2",
            "gelu_reconstruct",
            "linear_ff_dense2",
        ):
            if prefer_boot and alt in boot:
                return boot[alt]
            if alt in after:
                return after[alt]
    if suf == "softmax_exp_stockmeyer":
        for alt in ("softmax_exp_stockmeyer",):
            if prefer_boot and alt in boot:
                return boot[alt]
            if alt in after:
                return after[alt]
    return None


def _match_ln_last(after: dict[str, int], prefix: str, layer: int) -> int | None:
    """Best rem for ln*_invsqrt_* last step."""
    key_pref = f"L{layer}.{prefix}"
    best = None
    best_i = -1
    for k, v in after.items():
        if not k.startswith(key_pref):
            continue
        m = re.search(r"_(\d+)$", k)
        if not m:
            continue
        i = int(m.group(1))
        if i >= best_i:
            best_i = i
            best = v
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--softmax-level", type=int, default=2)
    ap.add_argument("--ln-level", type=int, default=2)
    ap.add_argument("--gelu-level", type=int, default=2)
    ap.add_argument("--num-levels", type=int, default=29)
    ap.add_argument("--enc-level", type=int, default=14)
    ap.add_argument("--he-log", default=None)
    ap.add_argument("--tol", type=int, default=1, help="allowed |Δrem| mismatch")
    args = ap.parse_args()

    scheme = _scheme(
        softmax_level=args.softmax_level,
        ln_level=args.ln_level,
        gelu_level=args.gelu_level,
    )
    # HE probe: rem = num_levels − level_calc；plan init = num_levels − enc_level − 1。
    init = max(1, int(args.num_levels) - int(args.enc_level) - 1)
    events = build_bootstrap_events(args.task, scheme, phase="C")
    result = optimize_bootstrap(
        args.task,
        scheme,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        initial_level=init,
        phase="C",
    )
    trace = simulate_remaining_trace(
        events,
        result.placements,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        initial_level=init,
    )
    print(
        f"DP: {result.summary()}  init_rem={init} "
        f"(enc={args.enc_level}, num_levels={args.num_levels})",
        flush=True,
    )

    he_probe: dict[str, int] = {}
    he_bts: list[tuple[str, int, int, str]] = []
    if args.he_log:
        text = open(args.he_log, encoding="utf-8", errors="replace").read()
        he_probe = _parse_probe(text)
        he_bts = _parse_bts_rem(text)
        print(
            f"HE log: probe_sites={len(he_probe)} bts_rem_lines={len(he_bts)}",
            flush=True,
        )

    ok = True
    for li in args.layers:
        print(f"\n=== Layer {li} ===", flush=True)
        after = _after_event_rem(trace, events, li)
        boot = _after_boot_rem(trace, events, li)
        boots = [
            p
            for p in result.placements
            if p.event_name.startswith(f"L{li}.")
        ]
        print(f"  plan boots={len(boots)}:", flush=True)
        for p in boots:
            print(
                f"    {p.reason:12s} {p.pack_mode:10s} ct={p.ct_count:3d} "
                f"{p.event_name}",
                flush=True,
            )

        print("  DP rem @ probe-aligned events:", flush=True)
        for site, suf in PROBE_TO_EVENT.items():
            prefer = site in REMAPPED_BOOT_SITES or site.startswith("after_bts_")
            rem = _lookup_rem(after, boot, suf, li, prefer_boot=prefer)
            label = f"L{li}.{suf}{'*' if suf.endswith('_') else ''}"
            print(f"    {site:28s} → {label:40s} rem={rem}", flush=True)

        if not he_probe:
            continue

        print("  HE vs DP rem @ probe sites:", flush=True)
        for site, suf in PROBE_TO_EVENT.items():
            if site not in he_probe:
                continue
            he_r = he_probe[site]
            prefer = site in REMAPPED_BOOT_SITES or site.startswith("after_bts_")
            dp_r = _lookup_rem(after, boot, suf, li, prefer_boot=prefer)
            if dp_r is None:
                print(f"    {site}: HE rem={he_r}  DP=n/a", flush=True)
                continue
            diff = he_r - dp_r
            if site in REMAPPED_BOOT_SITES:
                status = "REMAP" if abs(diff) > args.tol else "OK"
            elif site in ("after_ln1", "after_ln2") and diff > args.tol:
                # HE LN 内部 reactive bts → 出口 rem 高于 DP
                status = "HE_REACTIVE"
            elif site in ("after_ff_dense2",) and abs(diff) <= 2:
                # GeLU mid→entry DualRail 后 rem 级联偏差（HE post@13−2 vs DP）
                status = "REMAP"
            elif abs(diff) <= args.tol:
                status = "OK"
            else:
                status = "MISMATCH"
                ok = False
            print(
                f"    [{status}] {site:28s} HE={he_r:3d} DP={dp_r:3d} "
                f"Δ(HE-DP)={diff:+d}",
                flush=True,
            )

        print("  Segment Δrem (HE probe vs DP event rem drop):", flush=True)
        for a, b, name in SEGMENT_CHECKS:
            if a not in he_probe or b not in he_probe:
                continue
            he_d = he_probe[a] - he_probe[b]
            sa, sb = PROBE_TO_EVENT[a], PROBE_TO_EVENT[b]
            ra = _lookup_rem(
                after, boot, sa, li, prefer_boot=a.startswith("after_bts_")
            )
            rb = _lookup_rem(
                after, boot, sb, li, prefer_boot=b.startswith("after_bts_")
            )
            if ra is None or rb is None:
                print(f"    {name}: HE Δ={he_d}  DP=n/a", flush=True)
                continue
            dp_d = ra - rb
            if a in REMAPPED_BOOT_SITES or b in REMAPPED_BOOT_SITES:
                status = "REMAP"
            elif name in ("ln1", "ln2") and he_d + 2 < dp_d:
                # layernorm_he_ct reactive mid-bts → HE 净耗 < DP 全深度
                status = "HE_REACTIVE"
            elif abs(he_d - dp_d) <= args.tol:
                status = "OK"
            elif he_d >= 0 and dp_d >= 0:
                status = "MISMATCH"
                ok = False
            else:
                status = "BOOT"
            print(
                f"    [{status}] {name:20s} {a}→{b}: "
                f"HE Δ={he_d:+3d}  DP Δ={dp_d:+3d}",
                flush=True,
            )

        if he_bts:
            print("  HE [bts rem] vs plan:", flush=True)
            plan_names = {p.event_name for p in boots}
            for ev, rb, ra, st in he_bts:
                if not ev.startswith(f"L{li}."):
                    continue
                in_plan = ev in plan_names
                status = "OK" if in_plan else "UNPLANNED"
                if status != "OK":
                    ok = False
                print(
                    f"    [{status}] {ev}: rem {rb}→{ra} pack_mode={st} "
                    f"in_plan={in_plan}",
                    flush=True,
                )
            fired = {ev for ev, _, _, _ in he_bts if ev.startswith(f"L{li}.")}
            missed = sorted(plan_names - fired)
            # Mid-event mapped to entry may not fire under that name — warn only.
            for name in missed:
                print(
                    f"    [WARN] plan site never logged [bts rem]: {name}",
                    flush=True,
                )

    print("\nPASS" if ok else "\nFAIL (see MISMATCH / UNPLANNED)", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
