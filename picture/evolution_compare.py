"""
对比进化搜索 Pareto（ΣS₁ filtered vs KL search）：

  校验集 / 验证集 各两张图（共 4 张）：
    1) Output KL Pareto（上行 log，下行线性 + 下包络）
    2) 准确率 + 翻转率（上行 accuracy = baseline + Δacc；下行 flips_pct）

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


def load_task_points(
    task: str,
    split: str,
    *,
    method_dirs: dict[str, str] | None = None,
    methods: tuple[dict[str, Any], ...] = METHODS,
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
) -> None:
    for method in methods:
        pts = task_data.get(method["key"], [])
        depths_list: list[float] = []
        ys_list: list[float] = []
        for p in pts:
            if not np.isfinite(p.depth) or not np.isfinite(p.kl):
                continue
            depths_list.append(p.depth)
            ys_list.append(float(p.kl))
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


def plot_acc_panel(
    ax: plt.Axes,
    task: str,
    task_data: dict[str, list[ParetoPoint]],
    *,
    methods: tuple[dict[str, Any], ...],
    baseline_pct: float,
    show_title: bool,
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
            if not np.isfinite(p.depth) or not np.isfinite(p.accuracy_delta_pct):
                continue
            pairs.append((p.depth, baseline_pct + p.accuracy_delta_pct))
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([a for _, a in pairs], dtype=np.float64)

        ax.plot(
            depths,
            ys,
            color=method["color"],
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )
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
    ax.set_xlabel("Depth sum")
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
) -> None:
    for method in methods:
        pts = task_data.get(method["key"], [])
        pairs: list[tuple[float, float]] = []
        for p in pts:
            if p.flips_pct is None:
                continue
            if not np.isfinite(p.depth) or not np.isfinite(p.flips_pct):
                continue
            pairs.append((p.depth, p.flips_pct))
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[0])
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([f for _, f in pairs], dtype=np.float64)

        ax.plot(
            depths,
            ys,
            color=method["color"],
            linewidth=1.5,
            alpha=0.85,
            zorder=2,
        )
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
    ax.set_xlabel("Depth sum")
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
        )
        plot_pareto_panel(
            axes[1, col],
            task,
            task_data[task],
            methods=methods,
            log_scale=False,
            show_title=False,
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
        rf"Pareto: $\Sigma S_1$ filtered vs KL ({split})",
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
        )
        plot_flips_panel(
            axes[1, col],
            task,
            task_data[task],
            methods=methods,
            show_title=False,
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
        rf"Accuracy & flip rate vs depth ({split})",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def default_output_path(kind: str, split: str) -> str:
    return os.path.join(_PIC_DIR, f"evolution_compare_{kind}_{split}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ΣS₁ / KL Pareto 与准确率/翻转率对比（calib + validation 共 4 图）"
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
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        print(f"\n=== split={split} ===")
        task_data: dict[str, dict[str, list[ParetoPoint]]] = {}
        for task in args.tasks:
            task_data[task] = load_task_points(
                task, split, method_dirs=method_dirs
            )
            counts = {
                m["key"]: len(task_data[task][m["key"]]) for m in METHODS
            }
            print(
                f"{task.upper()}: score1={counts['score1']}  KL={counts['kl']}"
            )

        # 1) Pareto KL
        fig_p = build_pareto_figure(task_data, args.tasks, split=split)
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
            task_data, args.tasks, baselines_pct, split=split
        )
        out_a = os.path.join(
            args.output_dir, f"evolution_compare_acc_{split}.pdf"
        )
        fig_a.savefig(out_a, format="pdf", bbox_inches="tight")
        plt.close(fig_a)
        print(f"Saved: {out_a}")


if __name__ == "__main__":
    main()
