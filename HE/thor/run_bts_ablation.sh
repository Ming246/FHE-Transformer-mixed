#!/usr/bin/env bash
# Bootstrap error ablation (mrpc sample 0, all-high, prescale ×100 in bts_prescale.py).
set -uo pipefail
cd /workspace
OUT=/workspace/results/he_chain
LOG=/tmp/thor_logs/bts_ablation
mkdir -p "$OUT" "$LOG"

COMMON=(
  --dataset mrpc --sample 0
  --softmax poly --softmax-level 1
  --ln poly --ln-level 1
  --gelu cheb --gelu-level 1
  --check sftmx_in att_context ln1_out ln2_out
  --tol 0.15 --rest-div 1
)

echo "==== $(date -Is) Phase A: save L1 exits (L0→L1) ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan_mock --layers 0 1 \
  --save-out "$OUT/L1_exit_mock.pkl" \
  2>&1 | tee "$LOG/A_L1_mock.log"

python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan --layers 0 1 \
  --save-out "$OUT/L1_exit_real.pkl" \
  2>&1 | tee "$LOG/A_L1_real.log"

echo "==== $(date -Is) Phase B: L2 plan vs plan_mock (real L1 entry) ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan --layers 2 \
  --load-in "$OUT/L1_exit_real.pkl" \
  2>&1 | tee "$LOG/B_L2_real_entry_plan.log"

python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan_mock --layers 2 \
  --load-in "$OUT/L1_exit_real.pkl" \
  2>&1 | tee "$LOG/B_L2_real_entry_mock.log"

echo "==== $(date -Is) Phase C: L2 entry swap (plan, mock vs real L1 exit) ===="
python3 -u HE/thor/smoke_thor_chain.py "${COMMON[@]}" \
  --bootstrap plan --layers 2 \
  --load-in "$OUT/L1_exit_mock.pkl" \
  2>&1 | tee "$LOG/C_L2_mock_entry_plan.log"

echo "==== $(date -Is) Phase D: isolated L2 (plain-hidden, no chain) ===="
python3 -u HE/thor/smoke_thor_repro.py --dataset mrpc --sample 0 --layer 2 \
  --input plain-hidden --bootstrap plan \
  --softmax poly --softmax-level 1 --ln poly --ln-level 1 --gelu cheb --gelu-level 1 \
  --check sftmx_in att_context ln1_out ln2_out --tol 0.15 --rest-div 1 \
  2>&1 | tee "$LOG/D_L2_isolated_plan.log"

python3 -u HE/thor/smoke_thor_repro.py --dataset mrpc --sample 0 --layer 2 \
  --input plain-hidden --bootstrap plan_mock \
  --softmax poly --softmax-level 1 --ln poly --ln-level 1 --gelu cheb --gelu-level 1 \
  --check sftmx_in att_context ln1_out ln2_out --tol 0.15 --rest-div 1 \
  2>&1 | tee "$LOG/D_L2_isolated_mock.log"

echo "==== $(date -Is) DONE; grep summaries: ===="
rg -n "\[entry L|\[OK\] L|\[BAD\] L|all_ok=|^PASS|^FAIL" "$LOG"/*.log || true
