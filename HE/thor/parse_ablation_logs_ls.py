#!/usr/bin/env python3
"""Best-effort table from ablation logs (only abs_err_ls = old max|err| after LS)."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

_LINE = re.compile(
    r"\[(?:OK|BAD)\] L(\d+)\.(\w+): max\|err\|=([\d.e+-]+)\s+"
    r"corr=([\d.+-nan]+)\s+ls_gain=([\d.e+-]+)"
)


def _parse_log(path: Path) -> dict[tuple[int, str], dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[tuple[int, str], dict] = {}
    for m in _LINE.finditer(text):
        layer, ck, err, corr, gain = m.groups()
        out[(int(layer), ck)] = {
            "abs_err_ls": float(err),
            "corr": corr,
            "ls_gain": float(gain),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock-log", default="/tmp/thor_logs/bts_ablation/A_L1_mock.log")
    ap.add_argument("--bts-log", default="/tmp/thor_logs/bts_ablation/A_L1_real.log")
    ap.add_argument(
        "-o",
        "--out",
        default="results/he_chain/metrics/compare_from_ablation_L01_ls_only.csv",
    )
    args = ap.parse_args()
    mock = _parse_log(Path(args.mock_log))
    bts = _parse_log(Path(args.bts_log))
    keys = sorted(set(mock) & set(bts))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "checkpoint",
                "mock_abs_err_ls",
                "bts_abs_err_ls",
                "delta_abs_err_ls",
                "mock_corr",
                "bts_corr",
                "note",
            ],
        )
        w.writeheader()
        for k in keys:
            m, b = mock[k], bts[k]
            w.writerow(
                {
                    "layer": k[0],
                    "checkpoint": k[1],
                    "mock_abs_err_ls": m["abs_err_ls"],
                    "bts_abs_err_ls": b["abs_err_ls"],
                    "delta_abs_err_ls": b["abs_err_ls"] - m["abs_err_ls"],
                    "mock_corr": m["corr"],
                    "bts_corr": b["corr"],
                    "note": "no max|plain| in old logs; rel needs new --metrics-csv run",
                }
            )
    print(f"wrote {args.out} ({len(keys)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
