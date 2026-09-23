"""
对比进化搜索 Pareto（默认同时画 ΣS₁ 与 KL search）：

  校验集 / 验证集 各两张图（共 4 张）：
    1) Output KL Pareto（上行 log，下行线性；只散点、不连线）
    2) 准确率 + 翻转率（上行 accuracy = baseline + Δacc；下行 flips_pct）

横轴默认 ``f_cost``（DualRail DP bts，与 evolution_score / evolution_kl 搜索目标一致）。
旧 CSV 无 ``f_cost`` 时回退 ``total_depth``；``--x-col total_depth`` 可强制画深度。

数据：
  - evolution_results/sensitive_scores_1/{task}_pareto_{calib|validation}.csv
  - evolution_kl_results/{task}_pareto_kl_{calib|validation}.csv

输出（默认写入 picture/）：
  - evolution_compare_pareto_calib.pdf
  - evolution_compare_acc_calib.pdf
  - evolution_compare_pareto_validation.pdf
  - evolution_compare_acc_validation.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置区 =====================
TASK_NAMES = ("mrpc", "rte", "sst2", "cola","qnli","mnli")
EVAL_SPLITS = ("calib", "validation")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PIC_DIR = os.path.dirname(os.path.abspath(__file__))

SCORE1_PARETO_DIR = os.path.join(
    _REPO_ROOT, "results", "evolution_results", "sensitive_scores_1"
)
KL_PARETO_DIR = os.path.join(_REPO_ROOT, "results", "evolution_kl_results")

SCORE_FILE_PATTERN = "{task}_pareto_{split}.csv"
KL_FILE_PATTERN = "{task}_pareto_kl_{split}.csv"

BASELINE_CACHE_PATH = os.path.join(_PIC_DIR, "baseline_acc_cache.json")

LOG_FLOOR = 1e-6
X_COL_DEFAULT = "f_cost"
X_LABELS = {
    "f_cost": "bts",
    "total_depth": "Depth sum",
}

METHODS: tuple[dict[str, Any], ...] = (
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
# ==================================================


@dataclass
class ParetoPoint:
    cost: float  # 横轴：默认 f_cost（bts）
    kl: float
    accuracy_delta_pct: float | None = None  # 百分点，如 +0.49 → 0.49
    flips_pct: float | None = None
    depth: float | None = None  # CSV total_depth，仅备查


_PCT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$")


def parse_pct(value: str | None) -> float | None:
    """解析 CSV 中的百分点字符串（'+0.49%' / '3.92%'）→ float 百分点。"""
    if value is None or str(value).strip() == "":
        return None
    m = _PCT_RE.match(str(value))
    if not m:
        return None
    return float(m.group(1))


def load_pareto_csv(path: str, *, x_col: str = X_COL_DEFAULT) -> list[ParetoPoint]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if x_col not in X_LABELS:
        raise ValueError(f"x_col 须为 {tuple(X_LABELS)}，当前 {x_col!r}")

    points: list[ParetoPoint] = []
    n_skip = 0
    used_fallback = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                raw = row.get(x_col)
                if raw is None or str(raw).strip() == "":
                    if x_col == "f_cost" and row.get("total_depth"):
                        raw = row["total_depth"]
                        used_fallback += 1
                    else:
                        n_skip += 1
                        continue
                cost = float(raw)
                kl = float(row["output_kl"])
            except (TypeError, ValueError, KeyError):
                n_skip += 1
                continue
            if not np.isfinite(cost) or not np.isfinite(kl):
                n_skip += 1
                continue
            depth_raw = row.get("total_depth")
            try:
                depth = float(depth_raw) if depth_raw not in (None, "") else None
            except (TypeError, ValueError):
                depth = None
            points.append(
                ParetoPoint(
                    cost=cost,
                    kl=max(kl, 0.0),
                    accuracy_delta_pct=parse_pct(row.get("accuracy_delta")),
                    flips_pct=parse_pct(row.get("flips_pct")),
                    depth=depth,
                )
            )
    if n_skip:
        print(f"  警告：{path} 跳过 {n_skip} 条无效 {x_col}/KL 行")
    if used_fallback:
        print(
            f"  警告：{path} 有 {used_fallback} 行无 f_cost，已用 total_depth 作横轴"
        )
    return points


def load_task_points(
    task: str,
    split: str,
    *,
    method_dirs: dict[str, str] | None = None,
    methods: tuple[dict[str, Any], ...] = METHODS,
    x_col: str = X_COL_DEFAULT,
) -> dict[str, list[ParetoPoint]]:
    out: dict[str, list[ParetoPoint]] = {}
    for method in methods:
        directory = (method_dirs or {}).get(method["key"], method["dir"])
        path = os.path.join(
            directory, method["pattern"].format(task=task, split=split)
        )
        if not os.path.isfile(path):
            print(f"  跳过缺失文件：{path}")
            out[method["key"]] = []
            continue
        out[method["key"]] = load_pareto_csv(path, x_col=x_col)
    return out


def _load_baseline_cache(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_baseline_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
        f.write("\n")


def resolve_baseline_acc_pct(
    task: str,
    split: str,
    *,
    cache_path: str = BASELINE_CACHE_PATH,
    force_refresh: bool = False,
) -> float:
    """
    原始准确率（百分点，如 87.25）。
    优先读缓存；否则用 SchemeEvaluator 在对应 split 上跑 baseline。
    """
    cache = _load_baseline_cache(cache_path)
    key = f"{task}/{split}"
    if not force_refresh and key in cache:
        return float(cache[key]) * 100.0

    from evolution_infer import SchemeEvaluator
    from poly_model_inference import CALIB_INDICES_DIR, CALIB_SEED

    print(f"  计算 baseline 准确率：{task}/{split} …")
    evaluator = SchemeEvaluator(
        task,
        eval_samples=None,
        eval_split=split,
        calib_seed=CALIB_SEED,
        calib_indices_dir=CALIB_INDICES_DIR,
    )
    acc = float(evaluator.baseline_acc)
    cache[key] = acc
    _save_baseline_cache(cache_path, cache)
    print(f"  baseline_acc={acc:.4f} → 已缓存 {cache_path}")
    return acc * 100.0


def plot_pareto_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    methods: tuple[dict[str, Any], ...],
    log_scale: bool,
    show_title: bool,
    xlabel: str,
) -> None:
    for method in methods:
        pts = task_data.get(method["key"], [])
        xs_list: list[float] = []
        ys_list: list[float] = []
        for p in pts:
            if not np.isfinite(p.cost) or not np.isfinite(p.kl):
                continue
            xs_list.append(p.cost)
            ys_list.append(float(p.kl))
        xs = np.asarray(xs_list, dtype=np.float64)
        ys = np.asarray(ys_list, dtype=np.float64)
        if log_scale:
            ys = np.maximum(ys, LOG_FLOOR)

        ax.scatter(
            xs,
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

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Output KL (log)")
    else:
        ax.set_ylabel("Output KL")
    ax.set_xlabel(xlabel)
    if show_title:
        ax.set_title(task.upper(), fontsize=11, fontweight="medium")
    ax.grid(
        True,
        which="both" if log_scale else "major",
        alpha=0.25,
        linewidth=0.6,
    )
    ax.set_axisbelow(True)


def plot_acc_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    methods: tuple[dict[str, Any], ...],
    baseline_pct: float,
    show_title: bool,
    xlabel: str,
) -> None:
    # baseline 虚线（每任务一条；legend 只标一次）
    ax.axhline(
        baseline_pct,
        color="#64748b",
        linestyle="--",
        linewidth=1.4,
        alpha=0.9,
        label="baseline",
        zorder=1,
    )

    for method in methods:
        pts = task_data.get(method["key"], [])
        pairs: list[tuple[float, float]] = []
        for p in pts:
            if p.accuracy_delta_pct is None:
                continue
            if not np.isfinite(p.cost) or not np.isfinite(p.accuracy_delta_pct):
                continue
            pairs.append((p.cost, baseline_pct + p.accuracy_delta_pct))
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([a for _, a in pairs], dtype=np.float64)

        ax.scatter(
            depths,
            ys,
            s=28,
            c=method["color"],
            marker=method["marker"],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            label=method["label"],
            zorder=3,
        )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel(xlabel)
    if show_title:
        ax.set_title(task.upper(), fontsize=11, fontweight="medium")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_flips_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    methods: tuple[dict[str, Any], ...],
    show_title: bool,
    xlabel: str,
) -> None:
    for method in methods:
        pts = task_data.get(method["key"], [])
        pairs: list[tuple[float, float]] = []
        for p in pts:
            if p.flips_pct is None:
                continue
            if not np.isfinite(p.cost) or not np.isfinite(p.flips_pct):
                continue
            pairs.append((p.cost, p.flips_pct))
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([f for _, f in pairs], dtype=np.float64)

        ax.scatter(
            depths,
            ys,
            s=28,
            c=method["color"],
            marker=method["marker"],
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
            label=method["label"],
            zorder=3,
        )

    ax.set_ylabel("Flip rate (%)")
    ax.set_xlabel(xlabel)
    if show_title:
        ax.set_title(task.upper(), fontsize=11, fontweight="medium")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _apply_rc() -> None:
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


def build_pareto_figure(
    task_data: dict[str, dict[str, list[ParetoPoint]]],
    tasks: list[str],
    *,
    split: str,
    methods: tuple[dict[str, Any], ...] = METHODS,
    xlabel: str = X_LABELS[X_COL_DEFAULT],
) -> plt.Figure:
    _apply_rc()
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
        plot_pareto_panel(
            axes[0, col],
            task,
            task_data[task],
            methods=methods,
            log_scale=True,
            show_title=True,
            xlabel=xlabel,
        )
        plot_pareto_panel(
            axes[1, col],
            task,
            task_data[task],
            methods=methods,
            log_scale=False,
            show_title=False,
            xlabel=xlabel,
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
    fig.suptitle(
        rf"{' vs '.join(m['label'] for m in methods)} ({split})",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def build_acc_figure(
    task_data: dict[str, dict[str, list[ParetoPoint]]],
    tasks: list[str],
    baselines_pct: dict[str, float],
    *,
    split: str,
    methods: tuple[dict[str, Any], ...] = METHODS,
    xlabel: str = X_LABELS[X_COL_DEFAULT],
) -> plt.Figure:
    _apply_rc()
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
        plot_acc_panel(
            axes[0, col],
            task,
            task_data[task],
            methods=methods,
            baseline_pct=baselines_pct[task],
            show_title=True,
            xlabel=xlabel,
        )
        plot_flips_panel(
            axes[1, col],
            task,
            task_data[task],
            methods=methods,
            show_title=False,
            xlabel=xlabel,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # 去重 legend（多列可能重复）
    seen: set[str] = set()
    uniq_h, uniq_l = [], []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq_h.append(h)
        uniq_l.append(lab)
    fig.legend(
        uniq_h,
        uniq_l,
        loc="upper center",
        ncol=len(uniq_l),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
    )
    fig.suptitle(
        rf"Accuracy & flip rate vs {xlabel} ({split})",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def default_output_path(kind: str, split: str) -> str:
    return os.path.join(_PIC_DIR, f"evolution_compare_{kind}_{split}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto 与准确率/翻转率图（默认同时画 ΣS₁ 与 KL search）"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(EVAL_SPLITS),
        default=list(EVAL_SPLITS),
        help="评估集：默认同时画 calib 与 validation",
    )
    parser.add_argument("--score1-dir", default=SCORE1_PARETO_DIR)
    parser.add_argument("--kl-dir", default=KL_PARETO_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("score1", "kl"),
        default=["score1", "kl"],
        help="要画的曲线；默认 score1 + kl。只要其一：--methods score1",
    )
    parser.add_argument(
        "--x-col",
        choices=tuple(X_LABELS),
        default=X_COL_DEFAULT,
        help="横轴列：f_cost=bts（默认）；total_depth=乘法深度和",
    )
    parser.add_argument(
        "--baseline-cache",
        default=BASELINE_CACHE_PATH,
        help="baseline 准确率缓存 JSON",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="强制重算并刷新 baseline 缓存",
    )
    parser.add_argument(
        "--output-dir",
        default=_PIC_DIR,
        help="PDF 输出目录（默认 picture/）",
    )
    args = parser.parse_args()

    method_dirs = {
        "score1": args.score1_dir,
        "kl": args.kl_dir,
    }
    methods = tuple(m for m in METHODS if m["key"] in args.methods)
    if not methods:
        raise SystemExit("未选择任何 --methods")
    xlabel = X_LABELS[args.x_col]
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n=== split={split}  x={args.x_col} ({xlabel}) ===")
        task_data: dict[str, dict[str, list[ParetoPoint]]] = {}
        for task in args.tasks:
            task_data[task] = load_task_points(
                task,
                split,
                method_dirs=method_dirs,
                methods=methods,
                x_col=args.x_col,
            )
            counts = "  ".join(
                f"{m['key']}={len(task_data[task][m['key']])}" for m in methods
            )
            print(f"{task.upper()}: {counts}")

        fig_p = build_pareto_figure(
            task_data,
            args.tasks,
            split=split,
            methods=methods,
            xlabel=xlabel,
        )
        out_p = os.path.join(
            args.output_dir, f"evolution_compare_pareto_{split}.pdf"
        )
        fig_p.savefig(out_p, format="pdf", bbox_inches="tight")
        plt.close(fig_p)
        print(f"Saved: {out_p}")

        # 2) Accuracy fluctuation
        baselines_pct: dict[str, float] = {}
        for task in args.tasks:
            baselines_pct[task] = resolve_baseline_acc_pct(
                task,
                split,
                cache_path=args.baseline_cache,
                force_refresh=args.refresh_baseline,
            )
            print(f"  {task} baseline={baselines_pct[task]:.2f}%")

        fig_a = build_acc_figure(
            task_data,
            args.tasks,
            baselines_pct,
            split=split,
            methods=methods,
            xlabel=xlabel,
        )
        out_a = os.path.join(
            args.output_dir, f"evolution_compare_acc_{split}.pdf"
        )
        fig_a.savefig(out_a, format="pdf", bbox_inches="tight")
        plt.close(fig_a)
        print(f"Saved: {out_a}")


if __name__ == "__main__":
    main()
