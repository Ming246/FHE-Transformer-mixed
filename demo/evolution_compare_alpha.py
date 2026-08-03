"""
demo 对比图：ΣS1 / ΣαS1 / KL 三条 Pareto（Output KL vs depth）。

默认：
  ΣS1  ← 仓库 results/evolution_results/sensitive_scores_1/
  ΣαS1 ← demo/results/evolution_alpha/
  KL   ← 仓库 results/evolution_kl_results/

不改动 picture/evolution_compare.py。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from _repo import (  # noqa: E402
    DEMO_RESULTS_DIR,
    EVOLUTION_KL_DIR,
    EVOLUTION_S1_DIR,
    REPO_ROOT,
)

TASK_NAMES = ("mrpc", "rte", "sst2", "cola", "qnli", "mnli")
SCORE1_PARETO_DIR = EVOLUTION_S1_DIR
ALPHA_PARETO_DIR = os.path.join(DEMO_RESULTS_DIR, "evolution_alpha")
KL_PARETO_DIR = EVOLUTION_KL_DIR

SCORE_FILE_PATTERN = "{task}_pareto.csv"
KL_FILE_PATTERN = "{task}_pareto_kl.csv"

OUTPUT_PDF = os.path.join(DEMO_RESULTS_DIR, "evolution_compare_alpha.pdf")
LOG_FLOOR = 1e-6

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
        "key": "alpha",
        "label": r"$\Sigma\alpha S_1$ search",
        "dir": ALPHA_PARETO_DIR,
        "pattern": SCORE_FILE_PATTERN,
        "color": "#7c3aed",
        "marker": "D",
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


@dataclass
class ParetoPoint:
    depth: float
    kl: float


def load_pareto_csv(path: str) -> list[ParetoPoint]:
    if not os.path.isfile(path):
        print(f"  警告：缺少 {path}")
        return []
    points: list[ParetoPoint] = []
    n_skip = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                depth = float(row["total_depth"])
                kl = float(row["output_kl"])
            except (TypeError, ValueError, KeyError):
                n_skip += 1
                continue
            if not np.isfinite(depth) or not np.isfinite(kl):
                n_skip += 1
                continue
            points.append(ParetoPoint(depth=depth, kl=max(kl, 0.0)))
    if n_skip:
        print(f"  警告：{path} 跳过 {n_skip} 条")
    return points


def load_task_points(
    task: str,
    *,
    method_dirs: dict[str, str],
) -> dict[str, list[ParetoPoint]]:
    out: dict[str, list[ParetoPoint]] = {}
    for method in METHODS:
        directory = method_dirs.get(method["key"], method["dir"])
        path = os.path.join(directory, method["pattern"].format(task=task))
        out[method["key"]] = load_pareto_csv(path)
    return out


def pareto_envelope(
    depths: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(depths) & np.isfinite(values)
    depths = depths[mask]
    values = values[mask]
    if len(depths) == 0:
        return depths, values
    order = np.argsort(depths)
    d = depths[order]
    v = values[order]
    return d, np.minimum.accumulate(v)


def plot_panel(
    ax: plt.Axes,
    task_data: dict[str, list[ParetoPoint]],
    *,
    log_scale: bool,
    show_title: bool,
    task: str,
) -> None:
    for method in METHODS:
        pts = task_data[method["key"]]
        depths = np.array([p.depth for p in pts], dtype=np.float64)
        ys = np.array([p.kl for p in pts], dtype=np.float64)
        finite = np.isfinite(depths) & np.isfinite(ys)
        depths, ys = depths[finite], ys[finite]
        if log_scale:
            ys = np.maximum(ys, LOG_FLOOR)
        if len(depths) == 0:
            continue
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
    ax.grid(True, which="both" if log_scale else "major", alpha=0.25, linewidth=0.6)
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
            task_data[task],
            log_scale=True,
            show_title=True,
            task=task,
        )
        plot_panel(
            axes[1, col],
            task_data[task],
            log_scale=False,
            show_title=False,
            task=task,
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
        r"Pareto: $\Sigma S_1$ / $\Sigma\alpha S_1$ vs KL-direct",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="demo：ΣS1 / ΣαS1 / KL Pareto 对比")
    parser.add_argument("--tasks", nargs="+", default=["mrpc"])
    parser.add_argument("--output", default=OUTPUT_PDF)
    parser.add_argument("--score1-dir", default=SCORE1_PARETO_DIR)
    parser.add_argument("--alpha-dir", default=ALPHA_PARETO_DIR)
    parser.add_argument("--kl-dir", default=KL_PARETO_DIR)
    args = parser.parse_args()

    method_dirs = {
        "score1": args.score1_dir,
        "alpha": args.alpha_dir,
        "kl": args.kl_dir,
    }
    print(f"REPO_ROOT={REPO_ROOT}")
    task_data: dict[str, dict[str, list[ParetoPoint]]] = {}
    for task in args.tasks:
        task_data[task] = load_task_points(task, method_dirs=method_dirs)
        counts = {m["key"]: len(task_data[task][m["key"]]) for m in METHODS}
        print(
            f"{task.upper()}: ΣS1={counts['score1']}  "
            f"ΣαS1={counts['alpha']}  KL={counts['kl']}"
        )

    fig = build_figure(task_data, args.tasks)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
