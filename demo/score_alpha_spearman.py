"""
demo：ΣαS vs Output KL 散点（仅按 ΣαS / score 分层采样）。

对齐仓库 score_spearman.py 的 score 分层逻辑，但 proxy 为：
  score = sum_i alpha_i * S_i

默认 α：demo/results/alpha_rank/for_evolution/<task>/<task>_alpha.csv
默认 S：results/sensitive_scores_1/

输出（demo/results/score_alpha_spearman/）：
  {task}_random_score.csv
  random_score.pdf
  summary_score.csv

用法（仓库根）：
  python3 demo/score_alpha_spearman.py
  python3 demo/score_alpha_spearman.py --n 200 --eval-samples 64
  python3 demo/score_alpha_spearman.py --alpha-dir demo/results/alpha_rank/for_evolution
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from _repo import DEMO_RESULTS_DIR, REPO_ROOT, SENSITIVE_S1_DIR, chdir_repo

chdir_repo()

from cost import compute_scheme_cost  # noqa: E402
from evolution_infer import SchemeEvaluator, device  # noqa: E402
from evolution_score import (  # noqa: E402
    LOSS_REL_TOL,
    build_allowed_levels_table,
    load_sensitivity_matrix,
)
from score_spearman import (  # noqa: E402
    PLOT_LOG_FLOOR,
    TASK_PLOT_STYLES,
    assign_tolerant_ranks,
    kendall_tau_proxy_tolerant,
    sample_score_stratified_schemes,
)

from evolution_score_alpha import compute_f_loss, load_alpha_weights  # noqa: E402

TASK_NAMES = ["mrpc", "rte", "sst2"]
DEFAULT_ALPHA_DIR = os.path.join(
    DEMO_RESULTS_DIR, "alpha_rank", "for_evolution"
)
OUTPUT_DIR = os.path.join(DEMO_RESULTS_DIR, "score_alpha_spearman")
RANDOM_SCHEME_COUNT = 200
RANDOM_SCHEME_SEED = 42
EVAL_SEED = 42


@dataclass
class SchemeRow:
    scheme: list[int]
    score_alpha: float
    score_s: float
    output_kl: float
    total_depth: int


def resolve_alpha_csv(alpha_dir_or_file: str, task: str) -> str:
    if os.path.isfile(alpha_dir_or_file):
        return alpha_dir_or_file
    nested = os.path.join(alpha_dir_or_file, task, f"{task}_alpha.csv")
    flat = os.path.join(alpha_dir_or_file, f"{task}_alpha.csv")
    if os.path.isfile(nested):
        return nested
    if os.path.isfile(flat):
        return flat
    raise FileNotFoundError(
        f"找不到 {task} 的 alpha CSV：试过\n  {nested}\n  {flat}"
    )


def weighted_s_mat(
    s_mat: dict[tuple[int, str], dict[int, float]],
    alpha: dict[tuple[int, str], float],
) -> dict[tuple[int, str], dict[int, float]]:
    """把 α 折进 S，供 score 分层采样（档位贡献 = α·S）。"""
    out: dict[tuple[int, str], dict[int, float]] = {}
    for key, levels in s_mat.items():
        w = float(alpha[key])
        out[key] = {lv: w * float(v) for lv, v in levels.items()}
    return out


def run_task(
    task: str,
    *,
    sensitive_dir: str,
    alpha_csv: str,
    n_schemes: int,
    seed: int,
    eval_samples: int | None,
    n_score_bins: int | None,
) -> list[SchemeRow]:
    s_mat = load_sensitivity_matrix(task, sensitive_dir)
    alpha = load_alpha_weights(alpha_csv)
    s_mat_w = weighted_s_mat(s_mat, alpha)
    allowed = build_allowed_levels_table()
    rng = random.Random(seed)

    print(f"\n{'=' * 60}")
    print(
        f"ΣαS 分层随机 — {task.upper()}  n={n_schemes}  seed={seed}"
    )
    print(f"  S: {sensitive_dir}")
    print(f"  α: {alpha_csv}")
    print(f"  设备: {device}")
    print(f"{'=' * 60}")

    schemes, bins, counts = sample_score_stratified_schemes(
        n_schemes,
        rng,
        allowed,
        s_mat_w,
        n_bins=n_score_bins,
    )
    print(f"  生成 scheme {len(schemes)} 个（档数={len(bins)}）")
    print(f"  各档计数：{counts}")

    evaluator = SchemeEvaluator(
        task, eval_samples=eval_samples, seed=EVAL_SEED
    )
    print(f"  eval: {evaluator.eval_desc}")

    rows: list[SchemeRow] = []
    n = len(schemes)
    for i, scheme in enumerate(schemes, start=1):
        score_a = compute_f_loss(scheme, s_mat, alpha=alpha)
        score_s = compute_f_loss(scheme, s_mat, alpha=None)
        m = evaluator.evaluate_for_scheme(scheme)
        rows.append(
            SchemeRow(
                scheme=scheme,
                score_alpha=float(score_a),
                score_s=float(score_s),
                output_kl=float(m.output_kl),
                total_depth=int(compute_scheme_cost(task, scheme)),
            )
        )
        if i % max(1, n // 5) == 0 or i == n:
            print(f"  已推理 {i}/{n} 个 scheme…")
    return rows


def report_corr(proxy: np.ndarray, truth: np.ndarray, label: str) -> dict[str, float]:
    if len(proxy) < 3:
        nan = float("nan")
        print(f"  {label}: n={len(proxy)} 不足")
        return {
            "spearman_rho": nan,
            "spearman_p": nan,
            "kendall_tau": nan,
            "kendall_p": nan,
        }
    ranks = assign_tolerant_ranks(proxy)
    n_groups = len(np.unique(ranks))
    rho, rho_p = spearmanr(ranks, truth)
    tau, tau_p = kendall_tau_proxy_tolerant(proxy, truth)
    print(
        f"  {label}(容差 τ={LOSS_REL_TOL:.0%}, 组数={n_groups}/{len(proxy)}) "
        f"vs Output KL(严格): "
        f"Spearman ρ={rho:.4f} (p={rho_p:.4e})  "
        f"Kendall τ={tau:.4f} (p={tau_p:.4e})"
    )
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
    }


def save_csv(task: str, rows: list[SchemeRow], alpha_csv: str, sensitive_dir: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{task}_random_score.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "task",
                "score_alpha",
                "score_s",
                "total_depth",
                "output_kl",
                "scheme",
                "alpha_csv",
                "sensitive_dir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    task,
                    f"{r.score_alpha:.10e}",
                    f"{r.score_s:.10e}",
                    r.total_depth,
                    f"{r.output_kl:.10e}",
                    str(r.scheme),
                    alpha_csv,
                    sensitive_dir,
                ]
            )
    return path


def save_scatter_pdf(task_rows: dict[str, list[SchemeRow]]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "random_score.pdf")
    tasks = [t for t in TASK_NAMES if t in task_rows and task_rows[t]]
    for t in task_rows:
        if t not in tasks and task_rows[t]:
            tasks.append(t)
    if not tasks:
        return path

    n = len(tasks)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(
        1, n, figsize=(4.0 * n, 3.8), squeeze=False, constrained_layout=True
    )
    for col, task in enumerate(tasks):
        rows = task_rows[task]
        xs = np.asarray([r.score_alpha for r in rows], dtype=np.float64)
        ys = np.asarray([r.output_kl for r in rows], dtype=np.float64)
        style = TASK_PLOT_STYLES.get(
            task.lower(), {"color": "#64748b", "marker": "D"}
        )
        rho, _ = spearmanr(xs, ys)
        ax = axes[0, col]
        ax.scatter(
            np.maximum(xs, PLOT_LOG_FLOOR),
            np.maximum(ys, PLOT_LOG_FLOOR),
            s=26,
            alpha=0.7,
            c=style["color"],
            marker=style["marker"],
            edgecolors="none",
            zorder=2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"score ($\sum\alpha S$)")
        if col == 0:
            ax.set_ylabel("output_kl")
        ax.set_title(f"{task.upper()}  (n={len(rows)}, ρ={rho:.3f})")
        ax.grid(True, which="both", alpha=0.3, zorder=0)
    fig.suptitle(r"random_score: $\sum\alpha S$ vs Output KL", fontsize=12)
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="demo：ΣαS 按 score 分层采样，散点 vs Output KL"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument("--n", type=int, default=RANDOM_SCHEME_COUNT)
    parser.add_argument("--seed", type=int, default=RANDOM_SCHEME_SEED)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="验证子集；省略=全验证集",
    )
    parser.add_argument("--sensitive-dir", default=SENSITIVE_S1_DIR)
    parser.add_argument(
        "--alpha-dir",
        default=DEFAULT_ALPHA_DIR,
        help="α 目录（含 <task>/<task>_alpha.csv）或单文件",
    )
    parser.add_argument(
        "--score-bins",
        type=int,
        default=None,
        help="ΣαS log 分档数；默认自适应",
    )
    args = parser.parse_args()
    eval_samples = args.eval_samples

    print(f"REPO_ROOT={REPO_ROOT}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")

    plot_rows: dict[str, list[SchemeRow]] = {}
    summary: list[dict] = []

    for task in args.tasks:
        alpha_csv = resolve_alpha_csv(args.alpha_dir, task)
        rows = run_task(
            task,
            sensitive_dir=args.sensitive_dir,
            alpha_csv=alpha_csv,
            n_schemes=args.n,
            seed=args.seed,
            eval_samples=eval_samples,
            n_score_bins=args.score_bins,
        )
        path = save_csv(task, rows, alpha_csv, args.sensitive_dir)
        print(f"  明细：{path}")

        score_a = np.array([r.score_alpha for r in rows], dtype=np.float64)
        score_s = np.array([r.score_s for r in rows], dtype=np.float64)
        kl = np.array([r.output_kl for r in rows], dtype=np.float64)
        depths = np.array([r.total_depth for r in rows], dtype=np.float64)
        print(f"\n--- {task.upper()} 汇总 (n={len(rows)}) ---")
        if len(depths):
            print(
                f"  depth: min={depths.min():.0f}  max={depths.max():.0f}  "
                f"mean={depths.mean():.1f}"
            )
        m_a = report_corr(score_a, kl, "ΣαS")
        m_s = report_corr(score_s, kl, "ΣS")

        plot_rows[task] = rows
        summary.append(
            {
                "task": task,
                "n_schemes": len(rows),
                "alpha_csv": alpha_csv,
                "sensitive_dir": args.sensitive_dir,
                "spearman_rho_alpha": f"{m_a['spearman_rho']:.6f}",
                "kendall_tau_alpha": f"{m_a['kendall_tau']:.6f}",
                "spearman_rho_s": f"{m_s['spearman_rho']:.6f}",
                "kendall_tau_s": f"{m_s['kendall_tau']:.6f}",
            }
        )

    pdf = save_scatter_pdf(plot_rows)
    print(f"\n散点图：{pdf}")

    summary_path = os.path.join(OUTPUT_DIR, "summary_score.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        fields = list(summary[0].keys()) if summary else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"汇总：{summary_path}")


if __name__ == "__main__":
    main()
