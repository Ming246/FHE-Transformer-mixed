#!/usr/bin/env python3
"""
Reconcile ``bts_ops`` plan placements vs HE ``BootstrapHook`` wired sites.

No GPU required for plan side. Optionally parse a prior HE smoke log.

  python3 HE/thor/smoke_plan_he_reconcile.py --task mrpc --layers 0
  python3 HE/thor/smoke_plan_he_reconcile.py --scope attn_softmax --layers 0
  python3 HE/thor/smoke_plan_he_reconcile.py --he-log /tmp/chain_l0_l1_l2.log
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from bts_ops import (  # noqa: E402
    BOOTSTRAP_DEPTH_BUDGET,
    optimize_bootstrap,
)
from cost import NUM_LAYERS  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402


# Plan sites with HE BootstrapHook wiring (suffix after ``L{n}.``).
HE_WIRED_SUFFIXES = (
    "linear_qkv",
    "linear_att_score",
    "softmax_exp_stockmeyer",
    "softmax_exp_square_0",
    "softmax_attn_mask",
    "softmax_sigma_aggregate",
    "softmax_sigma_enc_one",
    "softmax_dualrail_exit",
    "linear_att_context",
    "bridge_rot_attn_dense",
    "linear_attn_dense",
    "ln1_scale",
    "ln1_var",
    "ln1_enc_one",
    "ln2_scale",
    "ln2_var",
    "ln2_enc_one",
    "gelu_f1",
    "gelu_f2",
    "gelu_reconstruct",
    "linear_ff_dense2",
    "softmax_asor_sigma_i",  # prefix match + int
    "softmax_sumsq_r",  # + N_apply / N_enc_one
    "softmax_norm_r",  # + N_asorM / N_inv_mask
)

# Micro-events from layer entry through ``linear_attn_dense`` (phase C).
ATTN_SOFTMAX_SUFFIXES = frozenset(
    {
        "linear_qkv",
        "linear_att_score",
        "softmax_exp_stockmeyer",
        "softmax_exp_square_0",
        "softmax_attn_mask",
        "softmax_sigma_aggregate",
        "softmax_sigma_enc_one",
        "softmax_dualrail_exit",
        "linear_att_context",
        "bridge_rot_attn_dense",
        "linear_attn_dense",
    }
)


def _scheme_like_smoke(
    *,
    softmax_level: int = 2,
    ln_level: int = 2,
    gelu_level: int = 2,
) -> list[int]:
    scheme: list[int] = []
    for li in range(NUM_LAYERS):
        g = gelu_level if gelu_level_allowed(li, gelu_level) else 2
        scheme.extend([softmax_level, ln_level, g, ln_level])
    return scheme


def _he_wired(name: str) -> bool:
    suf = name.split(".", 1)[1] if "." in name else name
    if suf in HE_WIRED_SUFFIXES:
        return True
    if re.match(r"softmax_asor_sigma_i\d+$", suf):
        return True
    if re.match(r"softmax_sumsq_r\d+_apply$", suf):
        return True
    if re.match(r"softmax_sumsq_r\d+_enc_one$", suf):
        return True
    if re.match(r"softmax_norm_r\d+_asor\d+$", suf):
        return True
    if re.match(r"softmax_norm_r\d+_inv_mask$", suf):
        return True
    return False


def _in_scope(suf: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope != "attn_softmax":
        raise ValueError(f"unknown scope {scope!r}")
    if suf in ATTN_SOFTMAX_SUFFIXES:
        return True
    if re.match(r"softmax_asor_sigma_i\d+$", suf):
        return True
    if re.match(r"softmax_sumsq_r\d+_apply$", suf):
        return True
    if re.match(r"softmax_sumsq_r\d+_enc_one$", suf):
        return True
    if re.match(r"softmax_norm_r\d+_asor\d+$", suf):
        return True
    if re.match(r"softmax_norm_r\d+_inv_mask$", suf):
        return True
    return False


def _parse_he_log(path: str) -> dict[str, int]:
    text = open(path, encoding="utf-8", errors="replace").read()
    by_event: dict[str, int] = {}
    in_block = False
    for line in text.splitlines():
        if "BootstrapHook:" in line:
            in_block = True
            by_event.clear()
            continue
        if in_block:
            m = re.match(r"\s+(L\d+\.[\w.]+):\s+(\d+)\s*$", line)
            if m:
                by_event[m.group(1)] = int(m.group(2))
                continue
            if line.strip() and not line.startswith(" ") and by_event:
                break
            if line.startswith("score ") or line.startswith("PASS") or line.startswith(
                "FAIL"
            ):
                break
    return by_event


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--softmax-level", type=int, default=2)
    ap.add_argument("--ln-level", type=int, default=2)
    ap.add_argument("--gelu-level", type=int, default=2)
    ap.add_argument("--he-log", default=None)
    ap.add_argument(
        "--scope",
        choices=("all", "attn_softmax"),
        default="attn_softmax",
        help="attn_softmax: linear_qkv … linear_attn_dense",
    )
    ap.add_argument(
        "--require-he-log",
        action="store_true",
        help="FAIL if --he-log missing (default: plan subseteq wired check only)",
    )
    args = ap.parse_args()

    scheme = _scheme_like_smoke(
        softmax_level=args.softmax_level,
        ln_level=args.ln_level,
        gelu_level=args.gelu_level,
    )
    result = optimize_bootstrap(
        args.task, scheme, budget=BOOTSTRAP_DEPTH_BUDGET, phase="C"
    )
    print(
        f"plan: {result.summary()}  scheme levels "
        f"sm/ln/gelu={args.softmax_level}/{args.ln_level}/{args.gelu_level}  "
        f"scope={args.scope}",
        flush=True,
    )

    he_log_counts: dict[str, int] = {}
    if args.he_log:
        he_log_counts = _parse_he_log(args.he_log)
        print(
            f"HE log {args.he_log}: {len(he_log_counts)} hook events, "
            f"ct_sum={sum(he_log_counts.values())}",
            flush=True,
        )
    elif args.require_he_log:
        print("FAIL: --require-he-log set but no --he-log", flush=True)
        return 2

    ok = True
    for li in args.layers:
        prefix = f"L{li}."
        layer_place = [
            p
            for p in result.placements
            if p.event_name.startswith(prefix)
            and _in_scope(p.event_name.split(".", 1)[1], args.scope)
        ]
        plan_ct = sum(p.ct_count for p in layer_place)
        print(f"\n=== Layer {li} (scope={args.scope}) ===", flush=True)
        print(
            f"  plan placements={len(layer_place)}  plan_ct_sum={plan_ct}",
            flush=True,
        )

        unwired = [p for p in layer_place if not _he_wired(p.event_name)]
        if unwired:
            ok = False
            for p in unwired:
                print(
                    f"  [BAD] plan site not wired in HE: {p.reason} "
                    f"ct={p.ct_count} {p.event_name}",
                    flush=True,
                )
        elif layer_place:
            print(
                f"  [OK] all {len(layer_place)} plan sites have HE hook wires",
                flush=True,
            )
        else:
            print("  [OK] no plan placements in scope", flush=True)

        by_reason = Counter(p.reason for p in layer_place)
        for reason, n in sorted(by_reason.items()):
            print(f"    reason={reason}: {n}", flush=True)
        for p in layer_place:
            print(
                f"    - {p.reason:12s} ct={p.ct_count:3d} {p.event_name}",
                flush=True,
            )

        if he_log_counts:
            he_layer = {
                k: v
                for k, v in he_log_counts.items()
                if k.startswith(prefix)
                and _in_scope(k.split(".", 1)[1], args.scope)
            }
            he_ct = sum(he_layer.values())
            print(f"  HE hook ct_sum={he_ct}  events={len(he_layer)}", flush=True)
            plan_names = {p.event_name for p in result.placements}
            for name, n in sorted(he_layer.items()):
                if name not in plan_names:
                    print(
                        f"  [BAD] HE fired {name} x{n} but not in plan_events",
                        flush=True,
                    )
                    ok = False
            for p in layer_place:
                he_n = he_layer.get(p.event_name)
                if he_n is None:
                    print(
                        f"  [WARN] plan placed {p.event_name} ct={p.ct_count} "
                        f"but HE log has no fire (stale log or path skipped)",
                        flush=True,
                    )
                    continue
                if he_n != p.ct_count:
                    print(
                        f"  [BAD] {p.event_name}: plan_ct={p.ct_count} "
                        f"he_log_ct={he_n}",
                        flush=True,
                    )
                    ok = False
                else:
                    print(
                        f"  [OK] {p.event_name}: plan_ct=he_log_ct={he_n}",
                        flush=True,
                    )

    print("\nPASS" if ok else "\nFAIL", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
