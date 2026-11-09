#!/usr/bin/env python3
"""Side-by-side mock vs bts dense error audit (L0 checkpoints vs poly plain)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["checkpoint"]] = row
    return out


def _f(row: dict, k: str) -> float:
    return float(row[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True)
    ap.add_argument("--bts", required=True)
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    mock = _load(args.mock)
    bts = _load(args.bts)
    keys = sorted(set(mock) & set(bts))
    fields = [
        "checkpoint",
        "n_elem",
        "max_abs_plain",
        "mean_abs_plain",
        "mock_max_abs_g1",
        "bts_max_abs_g1",
        "delta_max_abs_g1",
        "mock_max_rel_g1",
        "bts_max_rel_g1",
        "delta_max_rel_g1",
        "mock_mean_abs_g1",
        "bts_mean_abs_g1",
        "delta_mean_abs_g1",
        "mock_mean_rel_g1",
        "bts_mean_rel_g1",
        "delta_mean_rel_g1",
        "mock_max_abs_ls",
        "bts_max_abs_ls",
        "delta_max_abs_ls",
        "mock_max_rel_ls",
        "bts_max_rel_ls",
        "delta_max_rel_ls",
        "mock_mean_abs_ls",
        "bts_mean_abs_ls",
        "delta_mean_abs_ls",
        "mock_mean_rel_ls",
        "bts_mean_rel_ls",
        "delta_mean_rel_ls",
        "mock_corr",
        "bts_corr",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ck in keys:
            m, b = mock[ck], bts[ck]
            row = {
                "checkpoint": ck,
                "n_elem": m["n_elem"],
                "max_abs_plain": _f(m, "max_abs_plain"),
                "mean_abs_plain": _f(m, "mean_abs_plain"),
                "mock_max_abs_g1": _f(m, "max_abs_g1"),
                "bts_max_abs_g1": _f(b, "max_abs_g1"),
                "delta_max_abs_g1": _f(b, "max_abs_g1") - _f(m, "max_abs_g1"),
                "mock_max_rel_g1": _f(m, "max_rel_g1"),
                "bts_max_rel_g1": _f(b, "max_rel_g1"),
                "delta_max_rel_g1": _f(b, "max_rel_g1") - _f(m, "max_rel_g1"),
                "mock_mean_abs_g1": _f(m, "mean_abs_g1"),
                "bts_mean_abs_g1": _f(b, "mean_abs_g1"),
                "delta_mean_abs_g1": _f(b, "mean_abs_g1") - _f(m, "mean_abs_g1"),
                "mock_mean_rel_g1": _f(m, "mean_rel_g1"),
                "bts_mean_rel_g1": _f(b, "mean_rel_g1"),
                "delta_mean_rel_g1": _f(b, "mean_rel_g1") - _f(m, "mean_rel_g1"),
                "mock_max_abs_ls": _f(m, "max_abs_ls"),
                "bts_max_abs_ls": _f(b, "max_abs_ls"),
                "delta_max_abs_ls": _f(b, "max_abs_ls") - _f(m, "max_abs_ls"),
                "mock_max_rel_ls": _f(m, "max_rel_ls"),
                "bts_max_rel_ls": _f(b, "max_rel_ls"),
                "delta_max_rel_ls": _f(b, "max_rel_ls") - _f(m, "max_rel_ls"),
                "mock_mean_abs_ls": _f(m, "mean_abs_ls"),
                "bts_mean_abs_ls": _f(b, "mean_abs_ls"),
                "delta_mean_abs_ls": _f(b, "mean_abs_ls") - _f(m, "mean_abs_ls"),
                "mock_mean_rel_ls": _f(m, "mean_rel_ls"),
                "bts_mean_rel_ls": _f(b, "mean_rel_ls"),
                "delta_mean_rel_ls": _f(b, "mean_rel_ls") - _f(m, "mean_rel_ls"),
                "mock_corr": _f(m, "corr"),
                "bts_corr": _f(b, "corr"),
            }
            w.writerow(row)
    print(f"wrote {args.out} ({len(keys)} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
