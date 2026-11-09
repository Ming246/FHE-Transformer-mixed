#!/usr/bin/env bash
# Continuous L0→L2 mock vs real bts; export per-checkpoint metrics (tol only labels OK/BAD).
set -uo pipefail
cd /workspace
MET=results/he_chain/metrics
LOG=/tmp/thor_logs/metrics_L012
mkdir -p "$MET" "$LOG"

COMMON=(
  --dataset mrpc --sample 0
  --softmax poly --softmax-level 1
  --ln poly --ln-level 1
  --gelu cheb --gelu-level 1
  --check sftmx_in att_context ln1_out ln2_out
  --tol 0.15 --rest-div 1
  --layers 0 1 2
)

echo "==== mock L0-L2 ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan_mock \
  --metrics-csv "$MET/mock_L012.csv" \
  2>&1 | tee "$LOG/mock_L012.log"

echo "==== bts L0-L2 ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan \
  --metrics-csv "$MET/bts_L012.csv" \
  2>&1 | tee "$LOG/bts_L012.log"

python3 HE/thor/merge_chain_metrics.py \
  --mock "$MET/mock_L012.csv" \
  --bts "$MET/bts_L012.csv" \
  -o "$MET/compare_mock_vs_bts_L012.csv"

echo "==== done → $MET/compare_mock_vs_bts_L012.csv"
