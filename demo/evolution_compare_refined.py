"""
demo 对比三条 Pareto：ΣS1 / ΣαS(nnls) / KL。
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

OUTPUT_PDF = os.path.join(DEMO_RESULTS_DIR, "evolution_compare_alpha.pdf")
LOG_FLOOR = 1e-6

METHODS = (
    {
        "key": "score1",
        "label": r"$\Sigma S_1$",
        "dir": EVOLUTION_S1_DIR,
        "pattern": "{task}_pareto.csv",
        "color": "#2563eb",
        "marker": "o",
    },
    {
        "key": "alpha_nnls",
        "label": r"$\Sigma\alpha S_1$ (rank)",
        "dir": os.path.join(DEMO_RESULTS_DIR, "evolution_rank"),
        "pattern": "{task}_pareto.csv",
        "color": "#16a34a",
        "marker": "s",
    },
    {
        "key": "kl",
        "label": "KL search",
        "dir": EVOLUTION_KL_DIR,
        "pattern": "{task}_pareto_kl.csv",
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
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                depth = float(row["total_depth"])
                kl = float(row["output_kl"])
            except (TypeError, ValueError, KeyError):
                continue
            if np.isfinite(depth) and np.isfinite(kl):
                points.append(ParetoPoint(depth=depth, kl=max(kl, 0.0)))
    return points


def pareto_envelope(depths, values):
    mask = np.isfinite(depths) & np.isfinite(values)
    depths, values = depths[mask], values[mask]
    if len(depths) == 0:
        return depths, values
    order = np.argsort(depths)
    d, v = depths[order], values[order]
    return d, np.minimum.accumulate(v)


def plot_panel(ax, task_data, *, log_scale, show_title, task, method_dirs):
    del method_dirs  # 数据已在 task_data 中预加载
    for method in METHODS:
        pts = task_data.get(method["key"], [])
        depths = np.array([p.depth for p in pts], dtype=np.float64)
        ys = np.array([p.kl for p in pts], dtype=np.float64)
        finite = np.isfinite(depths) & np.isfinite(ys)
        depths, ys = depths[finite], ys[finite]
        if len(depths) == 0:
            continue
        if log_scale:
            ys = np.maximum(ys, LOG_FLOOR)
        ax.scatter(
            depths,
            ys,
            s=22,
            c=method["color"],
            marker=method["marker"],
            alpha=0.5,
            edgecolors="white",
            linewidths=0.3,
            label=method["label"],
            zorder=2,
        )
        env_d, env_y = pareto_envelope(depths, ys)
        if len(env_d):
            ax.plot(
                env_d,
                env_y,
                color=method["color"],
                linewidth=1.5,
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


def main():
    parser = argparse.ArgumentParser(
        description="对比 ΣS1 / ΣαS(nnls) / KL 三条 Pareto"
    )
    parser.add_argument("--tasks", nargs="+", default=["mrpc", "rte", "sst2"])
    parser.add_argument("--output", default=OUTPUT_PDF)
    parser.add_argument("--score1-dir", default=EVOLUTION_S1_DIR)
    parser.add_argument(
        "--alpha-nnls-dir",
        default=os.path.join(DEMO_RESULTS_DIR, "evolution_rank"),
    )
    parser.add_argument("--kl-dir", default=EVOLUTION_KL_DIR)
    args = parser.parse_args()

    method_dirs = {
        "score1": args.score1_dir,
        "alpha_nnls": args.alpha_nnls_dir,
        "kl": args.kl_dir,
    }
    print(f"REPO_ROOT={REPO_ROOT}")
    task_data = {}
    for task in args.tasks:
        task_data[task] = {}
        counts = {}
        for m in METHODS:
            path = os.path.join(method_dirs[m["key"]], m["pattern"].format(task=task))
            pts = load_pareto_csv(path)
            task_data[task][m["key"]] = pts
            counts[m["key"]] = len(pts)
        print(f"{task.upper()}: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    n = len(args.tasks)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(3.8 * n, 7.2),
        sharex="col",
        squeeze=False,
        constrained_layout=True,
    )
    for col, task in enumerate(args.tasks):
        plot_panel(
            axes[0, col],
            task_data[task],
            log_scale=True,
            show_title=True,
            task=task,
            method_dirs=method_dirs,
        )
        plot_panel(
            axes[1, col],
            task_data[task],
            log_scale=False,
            show_title=False,
            task=task,
            method_dirs=method_dirs,
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
        r"Pareto: $\Sigma S_1$ / $\Sigma\alpha S_1$ (rank) vs KL",
        fontsize=12,
        fontweight="medium",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
