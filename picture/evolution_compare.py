"""
对比进化搜索 Pareto 前沿（仅 Output KL）：
  - evolution_results/sensitive_scores_1/{task}_pareto_full.csv  （ΣS₁ 完整 archive）
  - evolution_results/sensitive_scores_1/{task}_pareto.csv       （ΣS₁ 深度筛选后）
  - evolution_kl_results/{task}_pareto_kl.csv                    （Output KL 搜索）

横轴：乘法深度和 total_depth；纵轴：Output KL（上行 log，下行线性原值）。

输出：
  - evolution_compare.pdf           ：三系（full / filtered / KL），纵轴 Output KL
  - evolution_compare_clean.pdf     ：仅蓝+橙；剔除 |Δacc|≥2% 或 flips≥2% 的点
  - evolution_compare_quality.pdf   ：蓝+橙；上行 |Δacc|（仅散点）、下行 flips_pct
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置区 =====================
TASK_NAMES = ("mrpc", "rte", "sst2")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SCORE1_PARETO_DIR = os.path.join(
    _REPO_ROOT, "results", "evolution_results", "sensitive_scores_1"
)
KL_PARETO_DIR = os.path.join(_REPO_ROOT, "results", "evolution_kl_results")

SCORE_FILE_PATTERN = "{task}_pareto.csv"
SCORE_FULL_FILE_PATTERN = "{task}_pareto_full.csv"
KL_FILE_PATTERN = "{task}_pareto_kl.csv"

_PIC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF = os.path.join(_PIC_DIR, "evolution_compare.pdf")
OUTPUT_PDF_CLEAN = os.path.join(_PIC_DIR, "evolution_compare_clean.pdf")
OUTPUT_PDF_QUALITY = os.path.join(_PIC_DIR, "evolution_compare_quality.pdf")

LOG_FLOOR = 1e-6
# 质量门：|accuracy_delta| 或 flips_pct 达到该百分点则剔除（clean 图）
QUALITY_PCT_MAX = 2.0

METHODS: tuple[dict[str, Any], ...] = (
    {
        "key": "score1_full",
        "label": r"$\Sigma S_1$ full",
        "dir": SCORE1_PARETO_DIR,
        "pattern": SCORE_FULL_FILE_PATTERN,
        "color": "#94a3b8",
        "marker": "D",
    },
    {
        "key": "score1",
        "label": r"$\Sigma S_1$ filtered",
        "dir": SCORE1_PARETO_DIR,
        "pattern": SCORE_FILE_PATTERN,
        "color": "#2563eb",
        "marker": "o",
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

# clean 图：仅蓝（filtered）+ 橙（KL）
METHODS_CLEAN: tuple[dict[str, Any], ...] = tuple(
    m for m in METHODS if m["key"] in ("score1", "kl")
)
# ==================================================


@dataclass
class ParetoPoint:
    depth: float
    kl: float
    accuracy_delta_pct: float | None = None  # 百分点，如 +0.49 → 0.49
    flips_pct: float | None = None


_PCT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$")


def parse_pct(value: str | None) -> float | None:
    """解析 CSV 中的百分点字符串（'+0.49%' / '3.92%'）→ float 百分点。"""
    if value is None or str(value).strip() == "":
        return None
    m = _PCT_RE.match(str(value))
    if not m:
        return None
    return float(m.group(1))


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
            except (TypeError, ValueError, KeyError):
                n_skip += 1
                continue
            if not np.isfinite(depth) or not np.isfinite(kl):
                n_skip += 1
                continue
            points.append(
                ParetoPoint(
                    depth=depth,
                    kl=max(kl, 0.0),
                    accuracy_delta_pct=parse_pct(row.get("accuracy_delta")),
                    flips_pct=parse_pct(row.get("flips_pct")),
                )
            )
    if n_skip:
        print(f"  警告：{path} 跳过 {n_skip} 条无效 depth/KL 行")
    return points


def passes_quality_gate(
    p: ParetoPoint, *, max_pct: float = QUALITY_PCT_MAX
) -> bool:
    """保留 |Δacc|<max_pct 且 flips<max_pct 的点；缺字段则剔除。"""
    if p.accuracy_delta_pct is None or p.flips_pct is None:
        return False
    if abs(p.accuracy_delta_pct) >= max_pct:
        return False
    if p.flips_pct >= max_pct:
        return False
    return True


def filter_quality(
    points: list[ParetoPoint], *, max_pct: float = QUALITY_PCT_MAX
) -> list[ParetoPoint]:
    return [p for p in points if passes_quality_gate(p, max_pct=max_pct)]


def load_task_points(
    task: str,
    *,
    method_dirs: dict[str, str] | None = None,
    methods: tuple[dict[str, Any], ...] = METHODS,
) -> dict[str, list[ParetoPoint]]:
    out: dict[str, list[ParetoPoint]] = {}
    for method in methods:
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


def y_output_kl(p: ParetoPoint) -> float | None:
    return p.kl


def y_abs_acc_delta_pct(p: ParetoPoint) -> float | None:
    if p.accuracy_delta_pct is None:
        return None
    return abs(p.accuracy_delta_pct)


def y_flips_pct(p: ParetoPoint) -> float | None:
    return p.flips_pct


def plot_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    methods: tuple[dict[str, Any], ...],
    y_fn,
    ylabel: str,
    log_scale: bool,
    show_title: bool,
    draw_envelope: bool = True,
) -> None:
    for method in methods:
        pts = task_data[method["key"]]
        depths_list: list[float] = []
        ys_list: list[float] = []
        for p in pts:
            y = y_fn(p)
            if y is None or not np.isfinite(p.depth) or not np.isfinite(y):
                continue
            depths_list.append(p.depth)
            ys_list.append(float(y))
        depths = np.asarray(depths_list, dtype=np.float64)
        ys = np.asarray(ys_list, dtype=np.float64)
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

        if draw_envelope:
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
    ax.set_ylabel(ylabel)
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
    *,
    methods: tuple[dict[str, Any], ...] = METHODS,
    title: str = r"Pareto archive: $\Sigma S_1$ full / filtered vs KL-direct search",
    top_y_fn=y_output_kl,
    top_ylabel: str = "Output KL (log)",
    top_log: bool = True,
    top_draw_envelope: bool = True,
    bottom_y_fn=y_output_kl,
    bottom_ylabel: str = "Output KL",
    bottom_log: bool = False,
    bottom_draw_envelope: bool = True,
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
            methods=methods,
            y_fn=top_y_fn,
            ylabel=top_ylabel,
            log_scale=top_log,
            show_title=True,
            draw_envelope=top_draw_envelope,
        )
        plot_panel(
            axes[1, col],
            task,
            task_data[task],
            methods=methods,
            y_fn=bottom_y_fn,
            ylabel=bottom_ylabel,
            log_scale=bottom_log,
            show_title=False,
            draw_envelope=bottom_draw_envelope,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
    )

    fig.suptitle(title, fontsize=12, fontweight="medium")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 ΣS₁ full/filtered 与 KL 搜索的 Pareto 前沿（Output KL）"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES))
    parser.add_argument("--output", default=OUTPUT_PDF)
    parser.add_argument("--output-clean", default=OUTPUT_PDF_CLEAN)
    parser.add_argument("--output-quality", default=OUTPUT_PDF_QUALITY)
    parser.add_argument("--score1-dir", default=SCORE1_PARETO_DIR)
    parser.add_argument("--kl-dir", default=KL_PARETO_DIR)
    parser.add_argument(
        "--quality-pct",
        type=float,
        default=QUALITY_PCT_MAX,
        help="clean 图剔除阈值：|Δacc| 或 flips 达到该百分点则丢弃（默认 2）",
    )
    args = parser.parse_args()

    method_dirs = {
        "score1_full": args.score1_dir,
        "score1": args.score1_dir,
        "kl": args.kl_dir,
    }

    task_data: dict[str, dict[str, list[ParetoPoint]]] = {}
    for task in args.tasks:
        task_data[task] = load_task_points(task, method_dirs=method_dirs)
        counts = {m["key"]: len(task_data[task][m["key"]]) for m in METHODS}
        print(
            f"{task.upper()}: "
            f"full={counts['score1_full']}  "
            f"filtered={counts['score1']}  "
            f"KL={counts['kl']}"
        )

    # --- 原图：三系 ---
    fig = build_figure(task_data, args.tasks, methods=METHODS)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")

    # --- clean 图：仅蓝+橙，质量门过滤 ---
    task_data_clean: dict[str, dict[str, list[ParetoPoint]]] = {}
    for task in args.tasks:
        task_data_clean[task] = {}
        parts = []
        for method in METHODS_CLEAN:
            key = method["key"]
            before = task_data[task][key]
            after = filter_quality(before, max_pct=args.quality_pct)
            task_data_clean[task][key] = after
            parts.append(f"{key} {len(before)}→{len(after)}")
        print(f"{task.upper()} clean (|Δacc|or flips≥{args.quality_pct:g}% dropped): {', '.join(parts)}")

    fig_clean = build_figure(
        task_data_clean,
        args.tasks,
        methods=METHODS_CLEAN,
        title=(
            r"Pareto archive: $\Sigma S_1$ filtered vs KL "
            rf"($|\Delta$acc$|$ & flips $<${args.quality_pct:g}\%)"
        ),
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_clean)) or ".", exist_ok=True
    )
    fig_clean.savefig(args.output_clean, format="pdf", bbox_inches="tight")
    plt.close(fig_clean)
    print(f"Saved: {args.output_clean}")

    # --- quality 图：蓝+橙；上行 |Δacc| 仅散点、下行 flips ---
    fig_q = build_figure(
        task_data,
        args.tasks,
        methods=METHODS_CLEAN,
        title=(
            r"Pareto archive: $|\Delta$acc$|$ / flips vs depth "
            r"($\Sigma S_1$ filtered / KL)"
        ),
        top_y_fn=y_abs_acc_delta_pct,
        top_ylabel=r"$|\Delta$acc$|$ (%)",
        top_log=False,
        top_draw_envelope=False,
        bottom_y_fn=y_flips_pct,
        bottom_ylabel="flips (%)",
        bottom_log=False,
        bottom_draw_envelope=True,
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_quality)) or ".", exist_ok=True
    )
    fig_q.savefig(args.output_quality, format="pdf", bbox_inches="tight")
    plt.close(fig_q)
    print(f"Saved: {args.output_quality}")


if __name__ == "__main__":
    main()
