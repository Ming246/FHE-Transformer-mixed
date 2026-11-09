#!/usr/bin/env python3
"""Merge mock-chain and bts-chain --metrics-csv outputs into one compare table."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load(path: str) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["layer"]), row["checkpoint"])
            out[key] = row
    return out


def _f(row: dict, k: str) -> float:
    return float(row[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", required=True, help="plan_mock chain metrics CSV")
    ap.add_argument("--bts", required=True, help="plan (real bts) chain metrics CSV")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    mock = _load(args.mock)
    bts = _load(args.bts)
    keys = sorted(set(mock) | set(bts))

    fields = [
        "layer",
        "checkpoint",
        "max_abs_plain",
        "mock_max_abs_he",
        "bts_max_abs_he",
        "mock_abs_err_g1",
        "bts_abs_err_g1",
        "delta_abs_err_g1_bts_minus_mock",
        "mock_rel_err_g1",
        "bts_rel_err_g1",
        "delta_rel_err_g1",
        "mock_abs_err_ls",
        "bts_abs_err_ls",
        "delta_abs_err_ls",
        "mock_rel_err_ls",
        "bts_rel_err_ls",
        "delta_rel_err_ls",
        "mock_rmse_g1",
        "bts_rmse_g1",
        "delta_he_amp_abs",
        "mock_corr",
        "bts_corr",
        "mock_ls_gain",
        "bts_ls_gain",
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for layer, ck in keys:
            m = mock.get((layer, ck))
            b = bts.get((layer, ck))
            if m is None or b is None:
                continue
            plain = _f(m, "max_abs_plain")
            row = {
                "layer": layer,
                "checkpoint": ck,
                "max_abs_plain": plain,
                "mock_max_abs_he": _f(m, "max_abs_he"),
                "bts_max_abs_he": _f(b, "max_abs_he"),
                "mock_abs_err_g1": _f(m, "abs_err_g1"),
                "bts_abs_err_g1": _f(b, "abs_err_g1"),
                "delta_abs_err_g1_bts_minus_mock": _f(b, "abs_err_g1")
                - _f(m, "abs_err_g1"),
                "mock_rel_err_g1": _f(m, "rel_err_g1"),
                "bts_rel_err_g1": _f(b, "rel_err_g1"),
                "delta_rel_err_g1": _f(b, "rel_err_g1") - _f(m, "rel_err_g1"),
                "mock_abs_err_ls": _f(m, "abs_err_ls"),
                "bts_abs_err_ls": _f(b, "abs_err_ls"),
                "delta_abs_err_ls": _f(b, "abs_err_ls") - _f(m, "abs_err_ls"),
                "mock_rel_err_ls": _f(m, "rel_err_ls"),
                "bts_rel_err_ls": _f(b, "rel_err_ls"),
                "delta_rel_err_ls": _f(b, "rel_err_ls") - _f(m, "rel_err_ls"),
                "mock_rmse_g1": _f(m, "rmse_g1"),
                "bts_rmse_g1": _f(b, "rmse_g1"),
                "delta_he_amp_abs": abs(_f(b, "max_abs_he") - _f(m, "max_abs_he")),
                "mock_corr": _f(m, "corr"),
                "bts_corr": _f(b, "corr"),
                "mock_ls_gain": _f(m, "ls_gain"),
                "bts_ls_gain": _f(b, "ls_gain"),
            }
            w.writerow(row)

    print(f"wrote {args.out} ({len(keys)} keys, {sum(1 for k in keys if k in mock and k in bts)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
