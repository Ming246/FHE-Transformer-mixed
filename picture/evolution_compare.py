"""
对比进化搜索 Pareto（默认同时画 ΣS₁ 与 KL search）：

  校验集 / 验证集 各两张图（共 4 张）：
    1) Output KL Pareto（log 纵轴）
    2) 翻转率 Flip rate

横轴默认 ``$C_{\mathrm{BTS}}$``（``f_cost`` / DualRail DP bts）。
旧 CSV 无 ``f_cost`` 时回退 ``total_depth``；``--x-col total_depth`` 可强制画深度。

散点：保留 CSV 中全部点（含被支配解）。
连线：KL / Flip 对 (cost, metric) 再做一次非支配筛选后连接
（与 ``evolution_compare-pareto.py`` 的 ``filter_pareto_kl_cost`` 同形）。

数据：
  - evolution_results/sensitive_scores_1/{task}_pareto_{calib|validation}.csv
  - evolution_kl_results/{task}_pareto_kl_{calib|validation}.csv
  - picture/evol_low_high.json（全 low / 全 high 对照点，特殊标记）

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
#TASK_NAMES = ("mrpc", "rte", "sst2", "cola","qnli","mnli")
TASK_NAMES = ("mrpc")
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
LOW_HIGH_JSON_PATH = os.path.join(_PIC_DIR, "evol_low_high.json")

LOG_FLOOR = 1e-6
X_COL_DEFAULT = "f_cost"
X_LABELS = {
    "f_cost": r"$C_{\mathrm{BTS}}$",
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

# 全 low / 全 high 对照点样式（不连线，仅特殊散点）
REF_STYLES: tuple[dict[str, Any], ...] = (
    {
        "key": "all_low",
        "label": "all low",
        "color": "#C00000",
        "marker": "s",
        "size": 88,
        "edgecolors": "#f8fafc",
        "linewidths": 1.0,
    },
    {
        "key": "all_high",
        "label": "all high",
        "color": "#059669",
        "marker": "D",
        "size": 88,
        "edgecolors": "#f8fafc",
        "linewidths": 1.0,
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


def load_low_high_json(
    path: str,
    *,
    x_col: str = X_COL_DEFAULT,
    split: str | None = None,
) -> dict[str, dict[str, ParetoPoint]]:
    """
    读 ``evol_low_high.json`` → task → {all_low|all_high → ParetoPoint}。

    ``split`` 若给定且与 JSON ``meta.eval_split`` / 任务 ``eval_split`` 不一致则跳过该任务。
    横轴：``f_cost`` 或 ``total_depth``（与 ``--x-col`` 一致）。
    """
    if x_col not in X_LABELS:
        raise ValueError(f"x_col 须为 {tuple(X_LABELS)}，当前 {x_col!r}")
    if not os.path.isfile(path):
        print(f"  跳过缺失 low/high JSON：{path}")
        return {}

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    meta_split = (payload.get("meta") or {}).get("eval_split")
    tasks_raw = payload.get("tasks") or {}
    out: dict[str, dict[str, ParetoPoint]] = {}
    for task, block in tasks_raw.items():
        if not isinstance(block, dict):
            continue
        task_split = block.get("eval_split", meta_split)
        if split is not None and task_split is not None and task_split != split:
            continue
        refs: dict[str, ParetoPoint] = {}
        for style in REF_STYLES:
            key = style["key"]
            row = block.get(key)
            if not isinstance(row, dict):
                continue
            if x_col == "f_cost":
                cost = row.get("f_cost")
            else:
                cost = row.get("total_depth")
            kl = row.get("output_kl")
            if cost is None or kl is None:
                continue
            try:
                cost_f = float(cost)
                kl_f = float(kl)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(cost_f) and np.isfinite(kl_f)):
                continue
            acc_pct = row.get("accuracy_delta_pct")
            flips = row.get("flips_pct")
            depth = row.get("total_depth")
            try:
                acc_pct_f = (
                    float(acc_pct)
                    if acc_pct is not None and np.isfinite(float(acc_pct))
                    else None
                )
            except (TypeError, ValueError):
                acc_pct_f = None
            try:
                flips_f = (
                    float(flips)
                    if flips is not None and np.isfinite(float(flips))
                    else None
                )
            except (TypeError, ValueError):
                flips_f = None
            try:
                depth_f = (
                    float(depth)
                    if depth is not None and np.isfinite(float(depth))
                    else None
                )
            except (TypeError, ValueError):
                depth_f = None
            refs[key] = ParetoPoint(
                cost=cost_f,
                kl=kl_f,
                accuracy_delta_pct=acc_pct_f,
                flips_pct=flips_f,
                depth=depth_f,
            )
        if refs:
            out[task] = refs
    return out


def scatter_ref_points(
    ax: plt.Axes,
    refs: dict[str, ParetoPoint] | None,
    *,
    y_of,
    log_floor: bool = False,
    with_labels: bool = True,
) -> None:
    """在轴上画 all_low / all_high 特殊点；``y_of(p) -> y``。"""
    if not refs:
        return
    for style in REF_STYLES:
        p = refs.get(style["key"])
        if p is None:
            continue
        y = y_of(p)
        if y is None or not np.isfinite(p.cost) or not np.isfinite(y):
            continue
        yy = float(y)
        if log_floor:
            yy = max(yy, LOG_FLOOR)
        ax.scatter(
            [p.cost],
            [yy],
            s=style["size"],
            c=style["color"],
            marker=style["marker"],
            alpha=0.95,
            edgecolors=style["edgecolors"],
            linewidths=style["linewidths"],
            label=style["label"] if with_labels else None,
            zorder=5,
        )


def filter_pareto_kl_cost(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """
    双目标最小化：output_kl 与横轴 cost（默认 f_cost / bts）。

    p 支配 q ⟺ kl(p)≤kl(q) 且 cost(p)≤cost(q)，且至少一项严格更小。
    只保留非支配点，按 cost、再按 kl 升序。
    """
    keep: list[ParetoPoint] = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                q.kl <= p.kl
                and q.cost <= p.cost
                and (q.kl < p.kl or q.cost < p.cost)
            ):
                dominated = True
                break
        if not dominated:
            keep.append(p)
    keep.sort(key=lambda p: (p.cost, p.kl))
    return keep


def filter_pareto_metric_cost(
    pairs: list[tuple[float, float]],
    *,
    maximize_metric: bool,
) -> list[tuple[float, float]]:
    """
    对 (cost, metric) 做非支配筛选；cost 恒最小化。
    maximize_metric=True → accuracy；False → flips / KL。
    """
    keep: list[tuple[float, float]] = []
    for i, (c_i, m_i) in enumerate(pairs):
        dominated = False
        for j, (c_j, m_j) in enumerate(pairs):
            if i == j:
                continue
            if maximize_metric:
                better_or_eq = c_j <= c_i and m_j >= m_i
                strictly = c_j < c_i or m_j > m_i
            else:
                better_or_eq = c_j <= c_i and m_j <= m_i
                strictly = c_j < c_i or m_j < m_i
            if better_or_eq and strictly:
                dominated = True
                break
        if not dominated:
            keep.append((c_i, m_i))
    if maximize_metric:
        keep.sort(key=lambda t: (t[0], -t[1]))
    else:
        keep.sort(key=lambda t: (t[0], t[1]))
    return keep


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
    refs: dict[str, ParetoPoint] | None = None,
    ref_labels: bool = False,
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

        # 散点：全部点（含被支配）
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
        # 连线：仅 (cost, kl) 非支配前沿
        front = filter_pareto_kl_cost(
            [p for p in pts if np.isfinite(p.cost) and np.isfinite(p.kl)]
        )
        if front:
            fx = np.asarray([p.cost for p in front], dtype=np.float64)
            fy = np.asarray([float(p.kl) for p in front], dtype=np.float64)
            if log_scale:
                fy = np.maximum(fy, LOG_FLOOR)
            ax.plot(
                fx,
                fy,
                color=method["color"],
                linewidth=1.6,
                alpha=0.9,
                zorder=3,
            )

    scatter_ref_points(
        ax,
        refs,
        y_of=lambda p: p.kl,
        log_floor=log_scale,
        with_labels=ref_labels,
    )

    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("KL Divergence")
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
    refs: dict[str, ParetoPoint] | None = None,
    ref_labels: bool = False,
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
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([a for _, a in pairs], dtype=np.float64)

        # 散点：全部点（Accuracy 不连线）
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

    scatter_ref_points(
        ax,
        refs,
        y_of=lambda p: (
            None
            if p.accuracy_delta_pct is None
            else baseline_pct + p.accuracy_delta_pct
        ),
        with_labels=ref_labels,
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
    refs: dict[str, ParetoPoint] | None = None,
    ref_labels: bool = False,
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
        depths = np.asarray([d for d, _ in pairs], dtype=np.float64)
        ys = np.asarray([f for _, f in pairs], dtype=np.float64)

        # 散点：全部点
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
        # 连线：仅 (cost, flips) 非支配前沿
        front = filter_pareto_metric_cost(pairs, maximize_metric=False)
        if front:
            fx = np.asarray([d for d, _ in front], dtype=np.float64)
            fy = np.asarray([f for _, f in front], dtype=np.float64)
            ax.plot(
                fx,
                fy,
                color=method["color"],
                linewidth=1.5,
                alpha=0.85,
                zorder=2,
            )

    scatter_ref_points(
        ax,
        refs,
        y_of=lambda p: p.flips_pct,
        with_labels=ref_labels,
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


def _task_grid_shape(n: int) -> tuple[int, int]:
    """默认 2×3；任务数不足时按 3 列排布。"""
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols)) if n else 1
    return nrows, ncols


def _dedupe_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
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
        ncol=max(1, len(uniq_l)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        bbox_transform=fig.transFigure,
    )


def build_pareto_figure(
    task_data: dict[str, dict[str, list[ParetoPoint]]],
    tasks: list[str],
    *,
    split: str,
    methods: tuple[dict[str, Any], ...] = METHODS,
    xlabel: str = X_LABELS[X_COL_DEFAULT],
    refs_by_task: dict[str, dict[str, ParetoPoint]] | None = None,
) -> plt.Figure:
    _apply_rc()
    n = len(tasks)
    nrows, ncols = _task_grid_shape(n)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    refs_by_task = refs_by_task or {}
    for idx, task in enumerate(tasks):
        r, c = divmod(idx, ncols)
        refs = refs_by_task.get(task)
        plot_pareto_panel(
            axes[r, c],
            task,
            task_data[task],
            methods=methods,
            log_scale=True,
            show_title=True,
            xlabel=xlabel,
            refs=refs,
            ref_labels=(idx == 0),
        )
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    _dedupe_legend(fig, axes[0, 0])
    fig.suptitle(
        rf"{' vs '.join(m['label'] for m in methods)} ({split})",
        fontsize=12,
        fontweight="medium",
    )
    return fig


def build_acc_figure(
    task_data: dict[str, dict[str, list[ParetoPoint]]],
    tasks: list[str],
    *,
    split: str,
    methods: tuple[dict[str, Any], ...] = METHODS,
    xlabel: str = X_LABELS[X_COL_DEFAULT],
    refs_by_task: dict[str, dict[str, ParetoPoint]] | None = None,
) -> plt.Figure:
    """仅 Flip rate（2×3 网格；文件名仍用 evolution_compare_acc_*）。"""
    _apply_rc()
    n = len(tasks)
    nrows, ncols = _task_grid_shape(n)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.8 * ncols, 3.4 * nrows),
        sharex=False,
        squeeze=False,
        constrained_layout=True,
    )
    refs_by_task = refs_by_task or {}
    for idx, task in enumerate(tasks):
        r, c = divmod(idx, ncols)
        refs = refs_by_task.get(task)
        plot_flips_panel(
            axes[r, c],
            task,
            task_data[task],
            methods=methods,
            show_title=True,
            xlabel=xlabel,
            refs=refs,
            ref_labels=(idx == 0),
        )
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    _dedupe_legend(fig, axes[0, 0])
    fig.suptitle(
        rf"Flip rate vs {xlabel} ({split})",
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
        "--low-high-json",
        default=LOW_HIGH_JSON_PATH,
        help="全 low/high 对照 JSON（evol_low_high.py 输出）",
    )
    parser.add_argument(
        "--no-low-high",
        action="store_true",
        help="不画 all_low / all_high 对照点",
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
        refs_by_task: dict[str, dict[str, ParetoPoint]] = {}
        if not args.no_low_high:
            refs_by_task = load_low_high_json(
                args.low_high_json, x_col=args.x_col, split=split
            )
            if refs_by_task:
                print(
                    f"  low/high refs: {sorted(refs_by_task)} "
                    f"(from {args.low_high_json})"
                )
            else:
                print(
                    f"  low/high refs: 无匹配 split={split} "
                    f"（{args.low_high_json}）"
                )

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
            ref_n = len(refs_by_task.get(task, {}))
            print(f"{task.upper()}: {counts}  refs={ref_n}")

        fig_p = build_pareto_figure(
            task_data,
            args.tasks,
            split=split,
            methods=methods,
            xlabel=xlabel,
            refs_by_task=refs_by_task,
        )
        out_p = os.path.join(
            args.output_dir, f"evolution_compare_pareto_{split}.pdf"
        )
        fig_p.savefig(out_p, format="pdf", bbox_inches="tight")
        plt.close(fig_p)
        print(f"Saved: {out_p}")

        fig_a = build_acc_figure(
            task_data,
            args.tasks,
            split=split,
            methods=methods,
            xlabel=xlabel,
            refs_by_task=refs_by_task,
        )
        out_a = os.path.join(
            args.output_dir, f"evolution_compare_acc_{split}.pdf"
        )
        fig_a.savefig(out_a, format="pdf", bbox_inches="tight")
        plt.close(fig_a)
        print(f"Saved: {out_a}")


if __name__ == "__main__":
    main()
