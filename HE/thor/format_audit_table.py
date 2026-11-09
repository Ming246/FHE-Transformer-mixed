#!/usr/bin/env python3
"""Pretty-print mock/bts error-audit CSV (rel as %)."""
from __future__ import annotations

import argparse
import csv
import sys


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.4f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv_path, newline="", encoding="utf-8")))
    if not rows:
        print("empty csv", file=sys.stderr)
        return 1
    hdr = (
        f"{'L':>2} {'checkpoint':<14} {'max|plain|':>10} "
        f"{'max_abs':>10} {'max_rel%':>10} {'mean_abs':>10} {'mean_rel%':>10} "
        f"{'max_abs_ls':>10} {'max_rel_ls%':>11} {'corr':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{int(r['layer']):>2} {r['checkpoint']:<14} "
            f"{float(r['max_abs_plain']):>10.4g} "
            f"{float(r['max_abs_g1']):>10.3e} {_pct(r['max_rel_g1']):>10} "
            f"{float(r['mean_abs_g1']):>10.3e} {_pct(r['mean_rel_g1']):>10} "
            f"{float(r['max_abs_ls']):>10.3e} {_pct(r['max_rel_ls']):>11} "
            f"{float(r['corr']):>8.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
