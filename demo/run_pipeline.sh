#!/usr/bin/env bash
# demo 管线：标定 α → 排序门控 →（可选）短进化 → 对比图
# 在仓库根执行：bash demo/run_pipeline.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASK="${TASK:-mrpc}"
SAMPLES="${SAMPLES:-64}"
N_RANDOM="${N_RANDOM:-40}"
POP="${POP:-60}"
GENS="${GENS:-80}"
SKIP_EVO="${SKIP_EVO:-0}"

echo "== [1/4] calibrate α (single-site KL) =="
python3 demo/calibrate_alpha_s1.py --task "$TASK" --samples "$SAMPLES"

echo "== [2/4] proxy rank gate (ΣS / ΣαS / ΣK) =="
python3 demo/eval_proxy_rank.py \
  --task "$TASK" \
  --samples "$SAMPLES" \
  --n-random "$N_RANDOM" \
  --calib-dir "demo/results/alpha_calib/${TASK}"

if [[ "$SKIP_EVO" != "1" ]]; then
  echo "== [3/4] short evolution with ΣαS =="
  python3 demo/evolution_score_alpha.py \
    --tasks "$TASK" \
    --pop "$POP" \
    --gens "$GENS" \
    --sensitive-dir results/sensitive_scores_1 \
    --alpha-csv "demo/results/alpha_calib/${TASK}" \
    --eval-samples "$SAMPLES"

  echo "== [4/4] compare plot =="
  python3 demo/evolution_compare_alpha.py --tasks "$TASK"
else
  echo "== skip evolution / plot (SKIP_EVO=1) =="
fi

echo "done. outputs under demo/results/"
