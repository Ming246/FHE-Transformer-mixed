"""
对比三种 Taylor 敏感度 CSV 的一致性。

| 方法 ID | 脚本 | 目录 | 公式 |
|---------|------|------|------|
| first_order | sensitive_score_1.py | results/sensitive_scores_1/ | \|Σ_j g_j·Δy_j\| |
| diagonal_2nd | sensitive_score.py | results/sensitive_scores/ | \|Σ_j (g_j·Δy_j + ½·g_j²·Δy_j²)\| |
| full_hvp | sensitive_score_2.py | results/sensitive_scores_2/ | \|g^T Δy + ½ Δy^T H Δy\| |

指标（144 个 slot = 48 行 × S_low|S_mid|S_high，按 task 汇总）：
  - Spearman ρ / Kendall τ：排序一致性（进化搜索主要关心 rank）
  - log-Pearson r：数量级线性相关
  - median / p90 / max 相对误差：|a−b| / max(|a|,|b|, ε)
  - top10 / top20 overlap：高 S slot 集合 Jaccard
  - 综合 verdict：基于 ρ 与 median rel err 的分档
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]

SCORE_SOURCES = {
    "first_order": "./results/sensitive_scores_1/",
    "diagonal_2nd": "./results/sensitive_scores/",
    "full_hvp": "./results/sensitive_scores_2/",
}

SCORE_SOURCE_LABELS = {
    "first_order": "一阶 |Σ g·Δy|",
    "diagonal_2nd": "对角二阶 |Σ(g·Δy+½g²Δy²)|",
    "full_hvp": "完整 HVP |g^TΔy+½Δy^THΔy|",
}

OUTPUT_DIR = "./results/score_compare/"
LEVEL_COLS = ("S_low", "S_mid", "S_high")
LEVEL_KEYS = ("low", "mid", "high")
KINDS = ("softmax", "ln1", "gelu", "ln2")
EPS = 1e-30
TOP_K_LIST = (10, 20)

# ρ 与 median rel err 联合分档
VERDICT_THRESHOLDS = (
    (0.99, 0.05, "近乎一致"),
    (0.95, 0.15, "高度一致"),
    (0.85, 0.35, "相关但有明显差异"),
)
# ==================================================


@dataclass
class PairwiseMetrics:
    task: str
    method_a: str
    method_b: str
    n_slots: int
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    log_pearson_r: float
    log_pearson_p: float
    median_rel_err: float
    p90_rel_err: float
    max_rel_err: float
    top10_jaccard: float
    top20_jaccard: float
    verdict: str


def slot_key(layer_idx: int, kind: str, level: str) -> str:
    return f"L{layer_idx}_{kind}_{level}"


def load_task_scores(csv_dir: str, task_name: str) -> dict[str, float]:
    path = os.path.join(csv_dir, f"{task_name}_sensitivity.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    scores: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layer_idx = int(row["layer_idx"])
            kind = row["kind"]
            for col, level in zip(LEVEL_COLS, LEVEL_KEYS):
                scores[slot_key(layer_idx, kind, level)] = float(row[col])
    return scores


def relative_errors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), EPS)
    return np.abs(a - b) / denom


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if len(a) == 0:
        return float("nan")
    k = min(k, len(a))
    idx_a = set(np.argsort(a)[-k:].tolist())
    idx_b = set(np.argsort(b)[-k:].tolist())
    union = idx_a | idx_b
    if not union:
        return 1.0
    return len(idx_a & idx_b) / len(union)


def classify_verdict(rho: float, median_rel: float) -> str:
    for rho_min, rel_max, label in VERDICT_THRESHOLDS:
        if rho >= rho_min and median_rel <= rel_max:
            return label
    return "差异显著"


def compare_pair(
    task: str,
    method_a: str,
    scores_a: dict[str, float],
    method_b: str,
    scores_b: dict[str, float],
) -> PairwiseMetrics:
    keys = sorted(set(scores_a) & set(scores_b))
    if not keys:
        raise ValueError(f"{task}: {method_a} 与 {method_b} 无共同 slot")

    a = np.array([scores_a[k] for k in keys], dtype=np.float64)
    b = np.array([scores_b[k] for k in keys], dtype=np.float64)

    rho, rho_p = spearmanr(a, b)
    tau, tau_p = kendalltau(a, b)

    log_a = np.log10(np.maximum(a, EPS))
    log_b = np.log10(np.maximum(b, EPS))
    if np.std(log_a) < 1e-15 or np.std(log_b) < 1e-15:
        log_r, log_p = float("nan"), float("nan")
    else:
        log_r, log_p = pearsonr(log_a, log_b)

    rel = relative_errors(a, b)
    verdict = classify_verdict(float(rho), float(np.median(rel)))

    return PairwiseMetrics(
        task=task,
        method_a=method_a,
        method_b=method_b,
        n_slots=len(keys),
        spearman_rho=float(rho),
        spearman_p=float(rho_p),
        kendall_tau=float(tau),
        kendall_p=float(tau_p),
        log_pearson_r=float(log_r),
        log_pearson_p=float(log_p),
        median_rel_err=float(np.median(rel)),
        p90_rel_err=float(np.percentile(rel, 90)),
        max_rel_err=float(np.max(rel)),
        top10_jaccard=topk_jaccard(a, b, TOP_K_LIST[0]),
        top20_jaccard=topk_jaccard(a, b, TOP_K_LIST[1]),
        verdict=verdict,
    )


def compare_by_kind(
    task: str,
    method_a: str,
    scores_a: dict[str, float],
    method_b: str,
    scores_b: dict[str, float],
) -> list[tuple[str, float, float, str]]:
    """按 kind 返回 (kind, spearman, median_rel, verdict)。"""
    rows = []
    for kind in KINDS:
        keys = sorted(k for k in scores_a if f"_{kind}_" in k and k in scores_b)
        if len(keys) < 3:
            continue
        a = np.array([scores_a[k] for k in keys], dtype=np.float64)
        b = np.array([scores_b[k] for k in keys], dtype=np.float64)
        rho, _ = spearmanr(a, b)
        med_rel = float(np.median(relative_errors(a, b)))
        rows.append((kind, float(rho), med_rel, classify_verdict(float(rho), med_rel)))
    return rows


def print_pair_report(m: PairwiseMetrics, kind_rows: list[tuple[str, float, float, str]]) -> None:
    la = SCORE_SOURCE_LABELS.get(m.method_a, m.method_a)
    lb = SCORE_SOURCE_LABELS.get(m.method_b, m.method_b)
    print(f"\n--- {m.task.upper()}: {m.method_a} vs {m.method_b} ---")
    print(f"  {la}")
    print(f"  vs {lb}")
    print(f"  slots={m.n_slots}  verdict={m.verdict}")
    print(
        f"  Spearman ρ={m.spearman_rho:.4f} (p={m.spearman_p:.2e})  "
        f"Kendall τ={m.kendall_tau:.4f} (p={m.kendall_p:.2e})"
    )
    print(
        f"  log-Pearson r={m.log_pearson_r:.4f} (p={m.log_pearson_p:.2e})  "
        f"rel err median={m.median_rel_err * 100:.2f}% "
        f"p90={m.p90_rel_err * 100:.2f}% max={m.max_rel_err * 100:.2f}%"
    )
    print(
        f"  top10 Jaccard={m.top10_jaccard:.3f}  "
        f"top20 Jaccard={m.top20_jaccard:.3f}"
    )
    if kind_rows:
        parts = [
            f"{k}: ρ={rho:.3f} med_rel={med * 100:.2f}% ({v})"
            for k, rho, med, v in kind_rows
        ]
        print(f"  按 kind: {' | '.join(parts)}")


def save_summary_csv(rows: list[PairwiseMetrics], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "task",
        "method_a",
        "method_b",
        "n_slots",
        "spearman_rho",
        "spearman_p",
        "kendall_tau",
        "kendall_p",
        "log_pearson_r",
        "log_pearson_p",
        "median_rel_err",
        "p90_rel_err",
        "max_rel_err",
        "top10_jaccard",
        "top20_jaccard",
        "verdict",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in rows:
            writer.writerow({k: getattr(m, k) for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="对比三种敏感度 CSV 的一致性")
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "pairwise_summary.csv"),
        help="汇总 CSV 路径",
    )
    args = parser.parse_args()

    print("三种敏感度方法对比")
    print("verdict 规则: ρ≥0.99 & med_rel≤5% → 近乎一致; "
          "ρ≥0.95 & med_rel≤15% → 高度一致; "
          "ρ≥0.85 & med_rel≤35% → 相关但有明显差异; 否则 → 差异显著")

    all_metrics: list[PairwiseMetrics] = []
    method_ids = list(SCORE_SOURCES.keys())

    for task in args.tasks:
        loaded: dict[str, dict[str, float]] = {}
        missing = []
        for mid, csv_dir in SCORE_SOURCES.items():
            try:
                loaded[mid] = load_task_scores(csv_dir, task)
            except FileNotFoundError:
                missing.append(f"{mid} ({csv_dir})")

        if missing:
            print(f"\n[{task}] 警告：缺少 {', '.join(missing)}")

        if len(loaded) < 2:
            print(f"[{task}] 跳过：可用方法不足 2 个")
            continue

        for ma, mb in combinations(sorted(loaded.keys()), 2):
            m = compare_pair(task, ma, loaded[ma], mb, loaded[mb])
            kind_rows = compare_by_kind(task, ma, loaded[ma], mb, loaded[mb])
            print_pair_report(m, kind_rows)
            all_metrics.append(m)

    if not all_metrics:
        print("\n无有效对比结果（请先运行三种 sensitive_score*.py 生成 CSV）")
        return

    save_summary_csv(all_metrics, args.output)
    print(f"\n已保存汇总：{args.output}")

    # 三方法 Spearman 矩阵（按 task 平均）
    print(f"\n{'=' * 60}")
    print("跨 task 平均 Spearman ρ（方法对）")
    print(f"{'=' * 60}")
    for ma, mb in combinations(method_ids, 2):
        rhos = [m.spearman_rho for m in all_metrics if m.method_a == ma and m.method_b == mb]
        if rhos:
            print(f"  {ma} vs {mb}: mean ρ = {np.mean(rhos):.4f}  (n={len(rhos)})")


if __name__ == "__main__":
    main()
