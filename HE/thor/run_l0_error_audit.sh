#!/usr/bin/env bash
# L0 only: dense mock vs real bts vs poly plain (max/mean abs & rel). ~2× single-layer HE time.
set -uo pipefail
cd /workspace
OUT=results/he_chain/metrics/l0_audit
LOG=/tmp/thor_logs/l0_audit
mkdir -p "$OUT" "$LOG"
rm -f "$OUT"/mock_L0_audit.csv "$OUT"/bts_L0_audit.csv

COMMON=(
  --dataset mrpc --sample 0
  --softmax poly --softmax-level 1
  --ln poly --ln-level 1
  --gelu cheb --gelu-level 1
  --check sftmx_in att_context ln1_out ln2_out
  --tol 1.0 --rest-div 1
  --layers 0
  --no-rem-guard
)

echo "==== L0 mock chain ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan_mock \
  --error-audit-csv "$OUT/mock_L0_audit.csv" \
  2>&1 | tee "$LOG/mock_L0.log"

echo "==== L0 real bts chain ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan \
  --error-audit-csv "$OUT/bts_L0_audit.csv" \
  2>&1 | tee "$LOG/bts_L0.log"

python3 HE/thor/merge_l0_error_audit.py \
  --mock "$OUT/mock_L0_audit.csv" \
  --bts "$OUT/bts_L0_audit.csv" \
  -o "$OUT/L0_mock_vs_bts_dense.csv"

echo "==== 汇总表: $OUT/L0_mock_vs_bts_dense.csv"
