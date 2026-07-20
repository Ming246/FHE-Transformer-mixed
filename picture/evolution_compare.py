"""
对比三种进化搜索 Pareto 前沿（仅 Output KL）：
  - evolution_results/sensitive_scores_1/  （一阶 ΣS vs cost）
  - evolution_results/sensitive_scores_2/  （二阶 ΣS vs cost）
  - evolution_kl_results/                 （Output KL vs cost）

横轴：乘法深度和 total_depth；纵轴：Output KL（上行 log，下行线性原值）。
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置区 =====================
TASK_NAMES = ("mrpc", "rte", "sst2")

# 相对仓库根目录（本文件在 picture/ 下），避免从 picture/ 运行时找不到 ./results
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCORE1_PARETO_DIR = os.path.join(
    _REPO_ROOT, "results", "evolution_results", "sensitive_scores_1"
)
SCORE2_PARETO_DIR = os.path.join(
    _REPO_ROOT, "results", "evolution_results", "sensitive_scores_2"
)
KL_PARETO_DIR = os.path.join(_REPO_ROOT, "results", "evolution_kl_results")

SCORE_FILE_PATTERN = "{task}_pareto.csv"
KL_FILE_PATTERN = "{task}_pareto_kl.csv"

OUTPUT_PDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "evolution_compare.pdf",
)

LOG_FLOOR = 1e-6  # log 轴下界，避免 0

METHODS = (
    {
        "key": "score1",
        "label": r"$\Sigma S_1$ search",
        "dir": SCORE1_PARETO_DIR,
        "pattern": SCORE_FILE_PATTERN,
        "color": "#2563eb",
        "marker": "o",
    },
    {
        "key": "score2",
        "label": r"$\Sigma S_2$ search",
        "dir": SCORE2_PARETO_DIR,
        "pattern": SCORE_FILE_PATTERN,
        "color": "#16a34a",
        "marker": "s",
    },
    {
        "key": "kl",
        "label": "KL search",
        "dir": KL_PARETO_DIR,
        "pattern": KL_FILE_PATTERN,
        "color": "#ea580c",
        "marker": "^",
    },
)
# ==================================================


@dataclass
class ParetoPoint:
    depth: float
    kl: float


def load_pareto_csv(path: str) -> list[ParetoPoint]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    points: list[ParetoPoint] = []
    n_skip = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                depth = float(row["total_depth"])
                kl = float(row["output_kl"])
            except (TypeError, ValueError):
                n_skip += 1
                continue
            if not np.isfinite(depth) or not np.isfinite(kl):
                n_skip += 1
                continue
            points.append(ParetoPoint(depth=depth, kl=max(kl, 0.0)))
    if n_skip:
        print(f"  警告：{path} 跳过 {n_skip} 条无效 depth/KL 行")
    return points


def load_task_points(
    task: str,
    *,
    method_dirs: dict[str, str] | None = None,
) -> dict[str, list[ParetoPoint]]:
    out: dict[str, list[ParetoPoint]] = {}
    for method in METHODS:
        directory = (method_dirs or {}).get(method["key"], method["dir"])
        path = os.path.join(directory, method["pattern"].format(task=task))
        out[method["key"]] = load_pareto_csv(path)
    return out


def pareto_envelope(
    depths: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """按 depth 升序，取累积最小值（depth–metric 平面下的下包络）。"""
    mask = np.isfinite(depths) & np.isfinite(values)
    depths = depths[mask]
    values = values[mask]
    if len(depths) == 0:
        return depths, values
    order = np.argsort(depths)
    d = depths[order]
    v = values[order]
    env_v = np.minimum.accumulate(v)
    return d, env_v


def plot_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    log_scale: bool,
    show_title: bool,
) -> None:
    for method in METHODS:
        pts = task_data[method["key"]]
        depths = np.array([p.depth for p in pts], dtype=np.float64)
        ys = np.array([p.kl for p in pts], dtype=np.float64)
        finite = np.isfinite(depths) & np.isfinite(ys)
        depths = depths[finite]
        ys = ys[finite]
        if log_scale:
            ys = np.maximum(ys, LOG_FLOOR)

        ax.scatter(
            depths,
            ys,
            s=28,
            c=method["color"],
            marker=method["marker"],
            alpha=0.55,
            edgecolors="white",
            linewidths=0.4,
            label=method["label"],
            zorder=2,
        )

        env_d, env_y = pareto_envelope(depths, ys)
        if len(env_d):
            ax.plot(
                env_d,
                env_y,
                color=method["color"],
                linewidth=1.6,
                alpha=0.9,
                zorder=3,
            )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Output KL (log)")
    else:
        ax.set_ylabel("Output KL")
    ax.set_xlabel("Depth sum")
    if show_title:
        ax.set_title(task.upper(), fontsize=11, fontweight="medium")
    ax.grid(
        True,
        which="both" if log_scale else "major",
        alpha=0.25,
        linewidth=0.6,
    )
    ax.set_axisbelow(True)


def build_figure(
    task_data: dict[str, dict[str, list[ParetoPoint]]],
    tasks: list[str],
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    n = len(tasks)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(3.8 * n, 7.2),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
    )

    for col, task in enumerate(tasks):
        plot_panel(
            axes[0, col],
            task,
            task_data[task],
            log_scale=True,
            show_title=True,
        )
        plot_panel(
            axes[1, col],
            task,
            task_data[task],
            log_scale=False,
            show_title=False,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
    )

    fig.suptitle(
        r"Pareto archive: $\Sigma S_1$ / $\Sigma S_2$ search vs KL-direct search",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比三种进化搜索 Pareto 前沿（仅 Output KL）"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES))
    parser.add_argument("--output", default=OUTPUT_PDF)
    parser.add_argument("--score1-dir", default=SCORE1_PARETO_DIR)
    parser.add_argument("--score2-dir", default=SCORE2_PARETO_DIR)
    parser.add_argument("--kl-dir", default=KL_PARETO_DIR)
    args = parser.parse_args()

    method_dirs = {
        "score1": args.score1_dir,
        "score2": args.score2_dir,
        "kl": args.kl_dir,
    }

    task_data: dict[str, dict[str, list[ParetoPoint]]] = {}
    for task in args.tasks:
        task_data[task] = load_task_points(task, method_dirs=method_dirs)
        counts = {
            m["key"]: len(task_data[task][m["key"]]) for m in METHODS
        }
        print(
            f"{task.upper()}: "
            f"ΣS1={counts['score1']}  ΣS2={counts['score2']}  "
            f"KL={counts['kl']}"
        )

    fig = build_figure(task_data, args.tasks)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
