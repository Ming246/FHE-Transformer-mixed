#!/usr/bin/env bash
# Mock vs real bts at named chain nodes (incl. Softmax internals); L0–L3.
set -uo pipefail
cd /workspace
SNAP=/workspace/results/he_chain/snapshots/mock_nodes
OUT=/workspace/results/he_chain/metrics/chain_mock_vs_bts_nodes_L0123.csv
LOG=/tmp/thor_logs/chain_nodes_L0123
mkdir -p "$SNAP" "$(dirname "$OUT")" "$LOG"
# Previous snapshots stored matrix-restored vectors; raw slot compare cannot reuse them.
rm -f "$SNAP"/mock_L*.npz "$OUT" \
  "${OUT%.csv}_aligned.txt" "${OUT%.csv}.tsv" "${OUT%.csv}_semicolon.csv"

COMMON=(
  --dataset mrpc --sample 0
  --softmax poly --softmax-level 1
  --ln poly --ln-level 1
  --gelu cheb --gelu-level 1
  --rest-div 1
  --layers 0 1 2 3
  --chain-snapshot-dir "$SNAP"
)

echo "==== mock capture L0-L3 ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan_mock \
  2>&1 | tee "$LOG/mock.log"

echo "==== real bts compare L0-L3 ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan \
  --chain-compare-csv "$OUT" \
  2>&1 | tee "$LOG/bts.log"

echo "==== done → $OUT"
python3 - <<'PY'
import csv
from pathlib import Path
src = Path("/workspace/results/he_chain/metrics/chain_mock_vs_bts_nodes_L0123.csv")
dst = src.with_name(src.stem + "_aligned.txt")
rows = list(csv.DictReader(src.open()))
# Fixed widths. Site names longer than the old 12-col slot were shifting every later field.
specs = [
    ("L", 1, "left", lambda r: r["layer"]),
    ("site", 24, "left", lambda r: r["site"]),
    ("max_abs", 11, "right", lambda r: f"{float(r['max_abs_err']):.4e}"),
    ("mean_abs", 11, "right", lambda r: f"{float(r['mean_abs_err']):.4e}"),
    ("rel_max%", 10, "right", lambda r: f"{float(r['rel_max_pct']):.4f}"),
    ("rel_mean%", 10, "right", lambda r: f"{float(r['rel_mean_pct']):.4f}"),
    ("n_rel", 7, "right", lambda r: str(int(float(r["n_rel_slots"])))),
    ("max|ref|", 11, "right", lambda r: f"{float(r['max_abs_ref']):.4e}"),
    ("max|cur|", 11, "right", lambda r: f"{float(r['max_abs_cur']):.4e}"),
    ("n_elem", 8, "right", lambda r: str(int(float(r["n_elem"])))),
]

def cell(text, width, align):
    return f"{text:<{width}}" if align == "left" else f"{text:>{width}}"

header = "  ".join(cell(name, width, align) for name, width, align, _ in specs)
rule = "  ".join("-" * width for _, width, _, _ in specs)
body = []
for r in rows:
    body.append("  ".join(cell(fn(r), width, align) for _, width, align, fn in specs))
dst.write_text(
    "\n".join([
        "# mock (ref) vs plan real bts (cur) — raw slot vectors, min |ref|=0.01",
        header,
        rule,
        *body,
        "",
    ])
)
print(dst)
PY
