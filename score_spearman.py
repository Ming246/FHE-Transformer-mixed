"""
方案级 ΣS proxy 与真实校验集指标的相关性评估。

  · 分层随机：按乘法深度或 ΣS（log）区间分层采样合法 scheme，
    在校验集上推理后与 Output KL 做 Spearman / Kendall。

排序容差仅作用于 proxy ΣS（与 evolution_score 的 f_loss 一致，单一相对容差 τ=LOSS_REL_TOL）：
  严格更好：L(x1) < (1−τ)·L(x2)
  不比 x2 差：L(x1) ≤ (1+τ)·L(x2)
  Output KL 仍用严格数值序。

  非传递性：原始「比值容差」不成等价关系。实现上对 ΣS 按升序贪心分组
 （组代表=组内最小值，v≤(1+τ)·rep 则并入），强制划成互斥组后再赋 average rank，
  从而得到可传递的全预序；并非对 pairwise 关系做传递闭包。

敏感度 CSV：SCORE_CSV_DIR（默认 sensitive_scores_1，校验集上计算）。
推理评估：默认全量校验集（calib）；可用 --eval-samples 抽样加速。
输出：
  {task}_random_{depth|score}.csv
  random_{depth|score}.pdf（各任务并排 log–log 散点）
  summary_{depth|score}.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from cost import (
    SCHEME_LEN,
    compute_scheme_cost,
    gelu_poly_depth,
    layernorm_poly_depth,
    scheme_index,
    softmax_poly_depth,
)
from evolution_infer import SchemeEvaluator, device
from evolution_score import (
    LOSS_REL_TOL,
    build_allowed_levels_table,
    compute_f_loss,
    iter_positions,
    load_sensitivity_matrix,
    loss_not_worse,
    loss_strictly_better,
    random_legal_scheme,
)
from poly_model_inference import CALIB_INDICES_DIR, CALIB_SEED

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]

# 换 score 版本时改此目录（需含 {task}_sensitivity.csv）
SCORE_CSV_DIR = "./results/sensitive_scores_1/"

# 深度 / ΣS 分层均匀采样
RANDOM_SCHEME_COUNT = 200
RANDOM_SCHEME_SEED = 42
# 随机方案分层方式：depth（默认）| score（ΣS，log 等宽档）
RANDOM_STRATIFY = "score"

# 将 [d_min, d_max] 等宽分为若干档；None = 按深度跨度自适应（约每 20 深度一档，上限 20）
DEPTH_BIN_COUNT: int | None = None
DEPTH_BIN_WIDTH_HINT = 20
DEPTH_BIN_COUNT_MAX = 20
DEPTH_BIN_COUNT_MIN = 5

# ΣS 分层：在 log10(ΣS) 上等宽分档
SCORE_BIN_COUNT: int | None = None
SCORE_BIN_WIDTH_HINT = 0.25  # 每个档位约 0.25 个 log10
SCORE_BIN_COUNT_MAX = 20
SCORE_BIN_COUNT_MIN = 5

# None = 全校验集；整数 = 固定子集加速
EVAL_SAMPLES: int | None = None
EVAL_SEED = 42
EVAL_SPLIT = "calib"

OUTPUT_DIR = "./results/score_spearman/"

# 多任务子图样式（random_{depth|score}.pdf）
TASK_PLOT_STYLES = {
    "mrpc": {"color": "#2563eb", "marker": "o"},
    "rte": {"color": "#ea580c", "marker": "^"},
    "sst2": {"color": "#16a34a", "marker": "s"},
    "cola": {"color": "#7c3aed", "marker": "D"},
    "qnli": {"color": "#db2777", "marker": "v"},
    "mnli": {"color": "#0d9488", "marker": "P"},
}
PLOT_LOG_FLOOR = 1e-8
# ==================================================


def format_pvalue(p: float) -> str:
    """三位有效数字的 p 值（科学计数法，如 2.39e-03）。"""
    if not np.isfinite(p):
        return "nan"
    return f"{p:.2e}"


@dataclass
class SchemeMetrics:
    scheme: list[int]
    score_sum: float
    output_kl: float
    total_depth: int = 0
    loss_delta: float = 0.0  # 推理副产物，不参与相关性


def scheme_total_depth(task_name: str, scheme: list[int]) -> int:
    return int(compute_scheme_cost(task_name, scheme))


def build_slot_depth_table(
    task_name: str, allowed_table: list[tuple[int, ...]]
) -> list[dict[int, int]]:
    """每 slot 合法档位 → 深度贡献（与 compute_scheme_cost 可加分解一致）。"""
    table: list[dict[int, int]] = [{} for _ in range(SCHEME_LEN)]
    for layer_idx, kind in iter_positions():
        idx = scheme_index(layer_idx, kind)
        for level in allowed_table[idx]:
            if kind == "softmax":
                depth = softmax_poly_depth(task_name, layer_idx, level)
            elif kind == "gelu":
                depth = gelu_poly_depth(layer_idx, level)
            else:
                depth = layernorm_poly_depth(task_name, layer_idx, kind, level)
            table[idx][level] = int(depth)
    return table


def fast_scheme_depth(scheme: list[int], slot_depths: list[dict[int, int]]) -> int:
    return sum(slot_depths[i][level] for i, level in enumerate(scheme))


def depth_bounds(
    allowed_table: list[tuple[int, ...]],
    slot_depths: list[dict[int, int]],
) -> tuple[int, int, list[int], list[int]]:
    """合法档位下的最小/最大深度（低档→深，高档→浅）。"""
    deepest = [min(levels) for levels in allowed_table]
    shallowest = [max(levels) for levels in allowed_table]
    d_max = fast_scheme_depth(deepest, slot_depths)
    d_min = fast_scheme_depth(shallowest, slot_depths)
    if d_min > d_max:
        d_min, d_max = d_max, d_min
        deepest, shallowest = shallowest, deepest
    return d_min, d_max, deepest, shallowest


def choose_depth_bin_count(d_min: int, d_max: int) -> int:
    if DEPTH_BIN_COUNT is not None:
        return max(DEPTH_BIN_COUNT_MIN, int(DEPTH_BIN_COUNT))
    span = max(d_max - d_min, 1)
    n = int(np.ceil(span / DEPTH_BIN_WIDTH_HINT))
    return int(np.clip(n, DEPTH_BIN_COUNT_MIN, DEPTH_BIN_COUNT_MAX))


def make_depth_bins(d_min: int, d_max: int, n_bins: int) -> list[tuple[int, int]]:
    """等宽闭开区间 [lo, hi)，最后一档为 [lo, d_max+1)。"""
    if n_bins <= 1 or d_min >= d_max:
        return [(d_min, d_max + 1)]
    edges = np.linspace(d_min, d_max, n_bins + 1)
    bins: list[tuple[int, int]] = []
    for i in range(n_bins):
        lo = int(np.floor(edges[i]))
        hi = int(np.floor(edges[i + 1])) if i < n_bins - 1 else d_max + 1
        if i == 0:
            lo = d_min
        if hi <= lo:
            hi = lo + 1
        bins.append((lo, hi))
    fixed: list[tuple[int, int]] = []
    for lo, hi in bins:
        if fixed and lo < fixed[-1][1]:
            lo = fixed[-1][1]
        if hi <= lo:
            continue
        fixed.append((lo, hi))
    if not fixed:
        return [(d_min, d_max + 1)]
    last_lo, _ = fixed[-1]
    fixed[-1] = (last_lo, d_max + 1)
    return fixed


def _biased_start_scheme(
    rng: random.Random,
    allowed_table: list[tuple[int, ...]],
    deepest: list[int],
    shallowest: list[int],
    d_min: int,
    d_max: int,
    target_mid: float,
) -> list[int]:
    r = rng.random()
    if r < 0.2:
        return list(deepest)
    if r < 0.4:
        return list(shallowest)
    if r < 0.75:
        frac = 0.0 if d_max == d_min else (target_mid - d_min) / (d_max - d_min)
        frac = float(np.clip(frac, 0.0, 1.0))
        # frac 大 → 更深 → 倾向低档；在 mid 档混入扰动
        start: list[int] = []
        for levels in allowed_table:
            u = rng.random()
            if u < frac * 0.7:
                start.append(min(levels))
            elif u > 1.0 - (1.0 - frac) * 0.7:
                start.append(max(levels))
            else:
                start.append(rng.choice(levels))
        return start
    return random_legal_scheme(rng, allowed_table)


def _nudge_scheme_toward_depth(
    scheme: list[int],
    allowed_table: list[tuple[int, ...]],
    slot_depths: list[dict[int, int]],
    target_lo: int,
    target_hi: int,
    rng: random.Random,
    max_steps: int = 80,
) -> list[int] | None:
    """用预计算 slot 深度抬/降档，逼近目标区间。"""
    cur = list(scheme)
    depth = fast_scheme_depth(cur, slot_depths)
    for _ in range(max_steps):
        if target_lo <= depth < target_hi:
            return cur
        want_deeper = depth < target_lo
        idxs = list(range(SCHEME_LEN))
        rng.shuffle(idxs)
        moved = False
        for idx in idxs:
            levels = allowed_table[idx]
            level = cur[idx]
            cur_d = slot_depths[idx][level]
            if want_deeper:
                candidates = [
                    lv for lv in levels if slot_depths[idx][lv] > cur_d
                ]
            else:
                candidates = [
                    lv for lv in levels if slot_depths[idx][lv] < cur_d
                ]
            if not candidates:
                continue
            new_lv = rng.choice(candidates)
            depth += slot_depths[idx][new_lv] - cur_d
            cur[idx] = new_lv
            moved = True
            break
        if not moved:
            break
    if target_lo <= depth < target_hi:
        return cur
    return None


def sample_depth_stratified_schemes(
    task_name: str,
    n: int,
    rng: random.Random,
    allowed_table: list[tuple[int, ...]],
    *,
    n_bins: int | None = None,
) -> tuple[list[list[int]], list[tuple[int, int]], list[int]]:
    """
    在 [d_min, d_max] 上等宽分档，各档尽量均匀配额采样不重复合法 scheme。

    返回 (schemes, bins, per_bin_counts)。
    """
    slot_depths = build_slot_depth_table(task_name, allowed_table)
    d_min, d_max, deepest, shallowest = depth_bounds(allowed_table, slot_depths)
    n_bins = choose_depth_bin_count(d_min, d_max) if n_bins is None else n_bins
    bins = make_depth_bins(d_min, d_max, n_bins)
    n_bins = len(bins)

    base = n // n_bins
    rem = n % n_bins
    quotas = [base + (1 if i < rem else 0) for i in range(n_bins)]

    print(
        f"  深度范围：[{d_min}, {d_max}]  分档={n_bins}  "
        f"总目标={n}  每档配额≈{base}"
    )
    for i, ((lo, hi), q) in enumerate(zip(bins, quotas)):
        print(f"    bin[{i}] depth∈[{lo}, {hi})  quota={q}")

    seen: set[tuple[int, ...]] = set()
    bucket: list[list[list[int]]] = [[] for _ in range(n_bins)]
    max_attempts_per_bin = max(quotas) * 80 + 500

    for bin_i, ((lo, hi), quota) in enumerate(zip(bins, quotas)):
        if quota <= 0:
            continue
        attempts = 0
        target_mid = 0.5 * (lo + hi - 1)
        while len(bucket[bin_i]) < quota and attempts < max_attempts_per_bin:
            attempts += 1
            start = _biased_start_scheme(
                rng,
                allowed_table,
                deepest,
                shallowest,
                d_min,
                d_max,
                target_mid,
            )
            scheme = _nudge_scheme_toward_depth(
                start, allowed_table, slot_depths, lo, hi, rng
            )
            if scheme is None:
                continue
            key = tuple(scheme)
            if key in seen:
                continue
            seen.add(key)
            bucket[bin_i].append(scheme)

        if len(bucket[bin_i]) < quota:
            print(
                f"  警告：bin[{bin_i}] [{lo},{hi}) 仅采到 "
                f"{len(bucket[bin_i])}/{quota}"
            )

    total = sum(len(b) for b in bucket)
    if total < n:
        print(f"  分层采样不足 ({total}/{n})，用随机合法 scheme 补齐…")
        fill_attempts = (n - total) * 50 + 500
        for _ in range(fill_attempts):
            if total >= n:
                break
            scheme = random_legal_scheme(rng, allowed_table)
            key = tuple(scheme)
            if key in seen:
                continue
            depth = fast_scheme_depth(scheme, slot_depths)
            shortfalls = [
                (quotas[i] - len(bucket[i]), i) for i in range(n_bins)
            ]
            shortfalls.sort(reverse=True)
            placed = False
            for need, i in shortfalls:
                if need <= 0:
                    continue
                lo, hi = bins[i]
                if lo <= depth < hi:
                    bucket[i].append(scheme)
                    seen.add(key)
                    total += 1
                    placed = True
                    break
            if not placed:
                best_i = min(
                    range(n_bins),
                    key=lambda i: abs(
                        depth - (bins[i][0] + bins[i][1] - 1) / 2
                    ),
                )
                if len(bucket[best_i]) < quotas[best_i] or total < n:
                    bucket[best_i].append(scheme)
                    seen.add(key)
                    total += 1

    schemes: list[list[int]] = []
    counts: list[int] = []
    for bin_i in range(n_bins):
        schemes.extend(bucket[bin_i])
        counts.append(len(bucket[bin_i]))

    if len(schemes) > n:
        rng.shuffle(schemes)
        schemes = schemes[:n]
        counts = [0] * n_bins
        for s in schemes:
            d = fast_scheme_depth(s, slot_depths)
            for i, (lo, hi) in enumerate(bins):
                if lo <= d < hi:
                    counts[i] += 1
                    break

    print(f"  实际每档样本数：{counts}  合计={len(schemes)}")
    return schemes, bins, counts


def build_slot_score_table(
    s_mat: dict,
    allowed_table: list[tuple[int, ...]],
) -> list[dict[int, float]]:
    """每 slot 合法档位 → S 贡献（与 compute_f_loss 可加分解一致）。"""
    table: list[dict[int, float]] = []
    for idx, (layer_idx, kind) in enumerate(iter_positions()):
        table.append(
            {lv: float(s_mat[(layer_idx, kind)][lv]) for lv in allowed_table[idx]}
        )
    return table


def fast_scheme_score(scheme: list[int], slot_scores: list[dict[int, float]]) -> float:
    return float(sum(slot_scores[i][level] for i, level in enumerate(scheme)))


def score_bounds(
    allowed_table: list[tuple[int, ...]],
    slot_scores: list[dict[int, float]],
) -> tuple[float, float, list[int], list[int]]:
    """合法档位下 ΣS 最小/最大（低档位往往 S 更大）。"""
    high_s = [min(levels) for levels in allowed_table]
    low_s = [max(levels) for levels in allowed_table]
    s_max = fast_scheme_score(high_s, slot_scores)
    s_min = fast_scheme_score(low_s, slot_scores)
    if s_min > s_max:
        s_min, s_max = s_max, s_min
        high_s, low_s = low_s, high_s
    s_min = max(s_min, PLOT_LOG_FLOOR)
    s_max = max(s_max, s_min * (1.0 + 1e-12))
    return s_min, s_max, high_s, low_s


def choose_score_bin_count(s_min: float, s_max: float) -> int:
    if SCORE_BIN_COUNT is not None:
        return max(SCORE_BIN_COUNT_MIN, int(SCORE_BIN_COUNT))
    span = max(np.log10(s_max) - np.log10(s_min), 1e-6)
    n = int(np.ceil(span / SCORE_BIN_WIDTH_HINT))
    return int(np.clip(n, SCORE_BIN_COUNT_MIN, SCORE_BIN_COUNT_MAX))


def make_log_score_bins(
    s_min: float, s_max: float, n_bins: int
) -> list[tuple[float, float]]:
    """log10 等宽闭开区间 [lo, hi)，最后一档右端略扩大以包含 s_max。"""
    if n_bins <= 1 or s_min >= s_max:
        return [(s_min, s_max * 1.0000001)]
    edges = np.logspace(np.log10(s_min), np.log10(s_max), n_bins + 1)
    bins: list[tuple[float, float]] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1]) if i < n_bins - 1 else float(s_max) * 1.0000001
        if hi <= lo:
            hi = lo * 1.0000001
        bins.append((lo, hi))
    return bins


def _biased_start_scheme_score(
    rng: random.Random,
    allowed_table: list[tuple[int, ...]],
    high_s: list[int],
    low_s: list[int],
    s_min: float,
    s_max: float,
    target_mid: float,
) -> list[int]:
    r = rng.random()
    if r < 0.2:
        return list(high_s)
    if r < 0.4:
        return list(low_s)
    if r < 0.75:
        frac = 0.0 if s_max <= s_min else (
            (np.log10(max(target_mid, PLOT_LOG_FLOOR)) - np.log10(s_min))
            / (np.log10(s_max) - np.log10(s_min))
        )
        frac = float(np.clip(frac, 0.0, 1.0))
        # frac 大 → 更高 ΣS → 倾向低档位
        start: list[int] = []
        for levels in allowed_table:
            u = rng.random()
            if u < frac * 0.7:
                start.append(min(levels))
            elif u > 1.0 - (1.0 - frac) * 0.7:
                start.append(max(levels))
            else:
                start.append(rng.choice(levels))
        return start
    return random_legal_scheme(rng, allowed_table)


def _nudge_scheme_toward_score(
    scheme: list[int],
    allowed_table: list[tuple[int, ...]],
    slot_scores: list[dict[int, float]],
    target_lo: float,
    target_hi: float,
    rng: random.Random,
    max_steps: int = 80,
) -> list[int] | None:
    cur = list(scheme)
    score = fast_scheme_score(cur, slot_scores)
    for _ in range(max_steps):
        if target_lo <= score < target_hi:
            return cur
        want_higher = score < target_lo
        idxs = list(range(SCHEME_LEN))
        rng.shuffle(idxs)
        moved = False
        for idx in idxs:
            levels = allowed_table[idx]
            level = cur[idx]
            cur_s = slot_scores[idx][level]
            if want_higher:
                candidates = [
                    lv for lv in levels if slot_scores[idx][lv] > cur_s
                ]
            else:
                candidates = [
                    lv for lv in levels if slot_scores[idx][lv] < cur_s
                ]
            if not candidates:
                continue
            new_lv = rng.choice(candidates)
            score += slot_scores[idx][new_lv] - cur_s
            cur[idx] = new_lv
            moved = True
            break
        if not moved:
            break
    if target_lo <= score < target_hi:
        return cur
    return None


def sample_score_stratified_schemes(
    n: int,
    rng: random.Random,
    allowed_table: list[tuple[int, ...]],
    s_mat: dict,
    *,
    n_bins: int | None = None,
) -> tuple[list[list[int]], list[tuple[float, float]], list[int]]:
    """
    在 [s_min, s_max] 上按 log10(ΣS) 等宽分档，各档尽量均匀配额采样。

    返回 (schemes, bins, per_bin_counts)。
    """
    slot_scores = build_slot_score_table(s_mat, allowed_table)
    s_min, s_max, high_s, low_s = score_bounds(allowed_table, slot_scores)
    n_bins = choose_score_bin_count(s_min, s_max) if n_bins is None else n_bins
    bins = make_log_score_bins(s_min, s_max, n_bins)
    n_bins = len(bins)

    base = n // n_bins
    rem = n % n_bins
    quotas = [base + (1 if i < rem else 0) for i in range(n_bins)]

    print(
        f"  ΣS 范围：[{s_min:.4e}, {s_max:.4e}]  log分档={n_bins}  "
        f"总目标={n}  每档配额≈{base}"
    )
    for i, ((lo, hi), q) in enumerate(zip(bins, quotas)):
        print(f"    bin[{i}] ΣS∈[{lo:.4e}, {hi:.4e})  quota={q}")

    seen: set[tuple[int, ...]] = set()
    bucket: list[list[list[int]]] = [[] for _ in range(n_bins)]
    max_attempts_per_bin = max(quotas) * 80 + 500

    for bin_i, ((lo, hi), quota) in enumerate(zip(bins, quotas)):
        if quota <= 0:
            continue
        attempts = 0
        target_mid = float(np.sqrt(lo * hi))  # log 中点
        while len(bucket[bin_i]) < quota and attempts < max_attempts_per_bin:
            attempts += 1
            start = _biased_start_scheme_score(
                rng, allowed_table, high_s, low_s, s_min, s_max, target_mid
            )
            scheme = _nudge_scheme_toward_score(
                start, allowed_table, slot_scores, lo, hi, rng
            )
            if scheme is None:
                continue
            key = tuple(scheme)
            if key in seen:
                continue
            seen.add(key)
            bucket[bin_i].append(scheme)

        if len(bucket[bin_i]) < quota:
            print(
                f"  警告：bin[{bin_i}] [{lo:.3e},{hi:.3e}) 仅采到 "
                f"{len(bucket[bin_i])}/{quota}"
            )

    total = sum(len(b) for b in bucket)
    if total < n:
        print(f"  分层采样不足 ({total}/{n})，用随机合法 scheme 补齐…")
        fill_attempts = (n - total) * 50 + 500
        for _ in range(fill_attempts):
            if total >= n:
                break
            scheme = random_legal_scheme(rng, allowed_table)
            key = tuple(scheme)
            if key in seen:
                continue
            score = fast_scheme_score(scheme, slot_scores)
            shortfalls = [
                (quotas[i] - len(bucket[i]), i) for i in range(n_bins)
            ]
            shortfalls.sort(reverse=True)
            placed = False
            for need, i in shortfalls:
                if need <= 0:
                    continue
                lo, hi = bins[i]
                if lo <= score < hi:
                    bucket[i].append(scheme)
                    seen.add(key)
                    total += 1
                    placed = True
                    break
            if not placed:
                best_i = min(
                    range(n_bins),
                    key=lambda i: abs(
                        np.log10(max(score, PLOT_LOG_FLOOR))
                        - 0.5
                        * (
                            np.log10(max(bins[i][0], PLOT_LOG_FLOOR))
                            + np.log10(max(bins[i][1], PLOT_LOG_FLOOR))
                        )
                    ),
                )
                if len(bucket[best_i]) < quotas[best_i] or total < n:
                    bucket[best_i].append(scheme)
                    seen.add(key)
                    total += 1

    schemes: list[list[int]] = []
    counts: list[int] = []
    for bin_i in range(n_bins):
        schemes.extend(bucket[bin_i])
        counts.append(len(bucket[bin_i]))

    if len(schemes) > n:
        rng.shuffle(schemes)
        schemes = schemes[:n]
        counts = [0] * n_bins
        for s in schemes:
            score = fast_scheme_score(s, slot_scores)
            for i, (lo, hi) in enumerate(bins):
                if lo <= score < hi:
                    counts[i] += 1
                    break

    print(f"  实际每档样本数：{counts}  合计={len(schemes)}")
    return schemes, bins, counts


def random_source_tag(stratify: str) -> str:
    """随机结果文件标签：random_depth / random_score。"""
    key = stratify.strip().lower()
    if key not in ("depth", "score"):
        raise ValueError(f"未知分层方式：{stratify!r}（需 depth 或 score）")
    return f"random_{key}"


def evaluate_schemes(
    task_name: str,
    schemes: list[list[int]],
    s_mat: dict,
    *,
    eval_samples: int | None,
    eval_split: str = EVAL_SPLIT,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> list[SchemeMetrics]:
    evaluator = SchemeEvaluator(
        task_name,
        eval_samples=eval_samples,
        seed=EVAL_SEED,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    print(f"  推理评估集：{evaluator.eval_desc}")
    out: list[SchemeMetrics] = []
    n = len(schemes)
    for i, scheme in enumerate(schemes, start=1):
        score_sum = compute_f_loss(scheme, s_mat)
        total_depth = scheme_total_depth(task_name, scheme)
        m = evaluator.evaluate_for_scheme(scheme)
        out.append(
            SchemeMetrics(
                scheme=scheme,
                score_sum=score_sum,
                loss_delta=float(m.loss_delta),
                output_kl=float(m.output_kl),
                total_depth=total_depth,
            )
        )
        if i % max(1, n // 5) == 0 or i == n:
            print(f"  已推理 {i}/{n} 个 scheme…")
    return out


def assign_tolerant_ranks(values: np.ndarray) -> np.ndarray:
    """
    仅用于 ΣS：将越小越好的度量折叠为有结名次（average rank）。

    按升序扫描；组代表固定为组内最小值 rep，若 loss_not_worse(v, rep)
    （即 v ≤ (1+τ)·rep）则并入同组，否则开新组。

    这样得到的是一组互斥划分上的全预序，回避了原始比值关系的非传递性；
    链式「彼此接近但首尾差很多」不会无限并成一组（相对固定的 rep）。
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    ranks = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ranks

    order = np.argsort(values, kind="mergesort")
    rank_start = 1
    i = 0
    while i < n:
        rep_idx = order[i]
        rep = float(values[rep_idx])
        group = [rep_idx]
        j = i + 1
        while j < n:
            idx = int(order[j])
            v = float(values[idx])
            if loss_not_worse(v, rep):
                group.append(idx)
                j += 1
                continue
            break
        avg_rank = 0.5 * (rank_start + rank_start + len(group) - 1)
        for idx in group:
            ranks[idx] = avg_rank
        rank_start += len(group)
        i = j
    return ranks


def tolerant_pair_sign(a: float, b: float) -> int:
    """ΣS 成对比较：-1 表示 a 严格更好，+1 表示 b 严格更好，0 表示容差内难分。"""
    if loss_strictly_better(a, b):
        return -1
    if loss_strictly_better(b, a):
        return 1
    return 0


def strict_pair_sign(a: float, b: float) -> int:
    """真值指标成对比较：严格数值序（越小越好）。"""
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def kendall_tau_proxy_tolerant(
    proxy: np.ndarray, truth: np.ndarray
) -> tuple[float, float]:
    """
    Kendall τ-b：仅 proxy（ΣS）侧用 STRICT 容差成对符号；
    truth（Output KL）侧严格比较。
    """
    a = np.asarray(proxy, dtype=np.float64)
    b = np.asarray(truth, dtype=np.float64)
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")

    concordant = 0
    discordant = 0
    tie_a = 0
    tie_b = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sa = tolerant_pair_sign(float(a[i]), float(a[j]))
            sb = strict_pair_sign(float(b[i]), float(b[j]))
            if sa == 0:
                tie_a += 1
            if sb == 0:
                tie_b += 1
            if sa != 0 and sb != 0:
                if sa == sb:
                    concordant += 1
                else:
                    discordant += 1

    n0 = n * (n - 1) // 2
    denom = np.sqrt(float(n0 - tie_a) * float(n0 - tie_b))
    if denom <= 0.0:
        return float("nan"), float("nan")
    tau = (concordant - discordant) / denom
    # p：proxy 容差名次 vs truth 原始值上的标准 Kendall
    _, p = kendalltau(assign_tolerant_ranks(a), b)
    return float(tau), float(p)


def rank_correlation_report(
    proxy: np.ndarray,
    truth: np.ndarray,
    label_proxy: str,
    label_truth: str,
) -> dict[str, float]:
    """proxy=ΣS（容差名次）；truth=Output KL（严格序）。"""
    if len(proxy) < 3:
        print(f"  {label_proxy} vs {label_truth}: 样本不足 (n={len(proxy)})")
        nan = float("nan")
        return {
            "spearman_rho": nan,
            "spearman_p": nan,
            "kendall_tau": nan,
            "kendall_p": nan,
        }

    ranks_proxy = assign_tolerant_ranks(proxy)
    n_groups = len(np.unique(ranks_proxy))
    # spearmanr 会对两侧再取秩；等值容差名次保持为结，truth 走严格 midrank
    rho, rho_p = spearmanr(ranks_proxy, truth)
    tau, tau_p = kendall_tau_proxy_tolerant(proxy, truth)
    print(
        f"  {label_proxy}(容差 τ={LOSS_REL_TOL:.0%}, "
        f"组数={n_groups}/{len(proxy)}) vs {label_truth}(严格) "
        f"(n={len(proxy)}): "
        f"Spearman ρ={rho:.4f} (p={rho_p:.4e})  "
        f"Kendall τ={tau:.4f} (p={tau_p:.4e})"
    )
    return {
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
    }


def summarize_correlations(
    task_name: str,
    source: str,
    rows: list[SchemeMetrics],
) -> dict[str, float | str | int]:
    print(f"\n--- {source} / {task_name.upper()} 汇总 (n={len(rows)}) ---")
    score_sums = np.array([r.score_sum for r in rows], dtype=np.float64)
    kl = np.array([r.output_kl for r in rows], dtype=np.float64)
    depths = np.array([r.total_depth for r in rows], dtype=np.float64)
    if len(depths):
        print(
            f"  depth: min={depths.min():.0f}  max={depths.max():.0f}  "
            f"mean={depths.mean():.1f}  std={depths.std():.1f}"
        )

    m_kl = rank_correlation_report(score_sums, kl, "ΣS", "Output KL")

    return {
        "task": task_name,
        "source": source,
        "n_schemes": len(rows),
        **{f"kl_{k}": v for k, v in m_kl.items()},
    }


def run_random(
    task_name: str,
    *,
    csv_dir: str,
    n_schemes: int,
    seed: int,
    eval_samples: int | None,
    stratify: str = "depth",
    n_depth_bins: int | None = None,
    n_score_bins: int | None = None,
    eval_split: str = EVAL_SPLIT,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> list[SchemeMetrics]:
    stratify = stratify.strip().lower()
    label = "深度分层" if stratify == "depth" else "ΣS分层"
    print(f"\n{'=' * 60}")
    print(
        f"{label}随机 — {task_name.upper()}  "
        f"n={n_schemes}  seed={seed}  stratify={stratify}"
    )
    print(f"score={csv_dir}  设备：{device}  eval_split={eval_split}")
    print(f"{'=' * 60}")

    s_mat = load_sensitivity_matrix(task_name, csv_dir)
    allowed_table = build_allowed_levels_table()
    rng = random.Random(seed)
    if stratify == "depth":
        schemes, bins, counts = sample_depth_stratified_schemes(
            task_name,
            n_schemes,
            rng,
            allowed_table,
            n_bins=n_depth_bins,
        )
    elif stratify == "score":
        schemes, bins, counts = sample_score_stratified_schemes(
            n_schemes,
            rng,
            allowed_table,
            s_mat,
            n_bins=n_score_bins,
        )
    else:
        raise ValueError(f"未知 stratify={stratify!r}（需 depth 或 score）")

    print(f"  生成 scheme {len(schemes)} 个（档数={len(bins)}）")
    print(f"  各档计数：{counts}")

    return evaluate_schemes(
        task_name,
        schemes,
        s_mat,
        eval_samples=eval_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )


def save_metrics_csv(
    task_name: str,
    source: str,
    rows: list[SchemeMetrics],
    csv_dir: str,
) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{task_name}_{source}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "source",
                "score_sum",
                "total_depth",
                "output_kl",
                "loss_delta",
                "scheme",
                "score_csv_dir",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    task_name,
                    source,
                    f"{r.score_sum:.10e}",
                    r.total_depth,
                    f"{r.output_kl:.10f}",
                    f"{r.loss_delta:.10f}",
                    str(r.scheme),
                    csv_dir,
                ]
            )
    return path


def save_score_kl_scatter_pdf(
    source: str,
    task_rows: dict[str, list[SchemeMetrics]],
) -> str:
    """
    同一 PDF：各任务并排 log–log 散点（ΣS vs Output KL）。
    输出 {OUTPUT_DIR}/{source}.pdf（source 如 random_depth / random_score）。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{source}.pdf")
    tasks = [t for t in TASK_NAMES if t in task_rows and task_rows[t]]
    for t in task_rows:
        if t not in tasks and task_rows[t]:
            tasks.append(t)
    if not tasks:
        return path

    n = len(tasks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, task_name in enumerate(tasks):
        row, col = divmod(i, ncols)
        rows = task_rows[task_name]
        xs = np.asarray([r.score_sum for r in rows], dtype=np.float64)
        ys = np.asarray([r.output_kl for r in rows], dtype=np.float64)
        style = TASK_PLOT_STYLES.get(
            task_name.lower(),
            {"color": "#64748b", "marker": "D"},
        )
        rho, rho_p = spearmanr(xs, ys)

        ax = axes[row, col]
        xs_p = np.maximum(xs, PLOT_LOG_FLOOR)
        ys_p = np.maximum(ys, PLOT_LOG_FLOOR)
        ax.scatter(
            xs_p,
            ys_p,
            s=26,
            alpha=0.7,
            c=style["color"],
            marker=style["marker"],
            edgecolors="none",
            zorder=2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"score_sum ($\sum S$)")
        if col == 0:
            ax.set_ylabel("output_kl")
        ax.set_title(
            f"{task_name.upper()}  (n={len(rows)}, "
            f"ρ={rho:.3f}, p={format_pvalue(float(rho_p))})"
        )
        ax.grid(True, which="both", alpha=0.3, zorder=0)

    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"{source}", fontsize=12)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_summary_csv(summary_rows: list[dict], *, tag: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"summary_{tag}.csv")
    fieldnames = [
        "task",
        "source",
        "n_schemes",
        "score_csv_dir",
        "spearman_rho_kl",
        "spearman_p_kl",
        "kendall_tau_kl",
        "kendall_p_kl",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ΣS proxy 与 Output KL 的排序相关性（校验集推理）"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=EVAL_SAMPLES,
        help="校验子集大小；省略则用配置区 EVAL_SAMPLES（默认全量校验集）",
    )
    parser.add_argument(
        "--eval-split",
        choices=("calib", "validation"),
        default=EVAL_SPLIT,
        help="推理评估集：默认 calib（校验集）",
    )
    parser.add_argument(
        "--calib-seed",
        type=int,
        default=CALIB_SEED,
        help=(
            f"校验集索引 seed（默认 {CALIB_SEED}）；"
            "读 {task}_calib_indices_seed{seed}.json"
        ),
    )
    parser.add_argument(
        "--calib-indices-dir",
        default=CALIB_INDICES_DIR,
        help="calib 索引 JSON 目录（coverage_metric 输出，只读）",
    )
    parser.add_argument(
        "--random-schemes",
        type=int,
        default=RANDOM_SCHEME_COUNT,
        help="分层随机 scheme 数量",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SCHEME_SEED,
        help="随机采样种子",
    )
    parser.add_argument(
        "--stratify",
        choices=("depth", "score"),
        default=RANDOM_STRATIFY,
        help="随机方案分层：depth（默认）或 score（ΣS/log）",
    )
    parser.add_argument(
        "--depth-bins",
        type=int,
        default=None,
        help="深度分档数；默认按 [d_min,d_max] 跨度自适应",
    )
    parser.add_argument(
        "--score-bins",
        type=int,
        default=None,
        help="ΣS(log) 分档数；默认按 log10 跨度自适应",
    )
    args = parser.parse_args()

    stratify = args.stratify.strip().lower()
    random_tag = random_source_tag(stratify)

    print(f"score CSV 目录：{SCORE_CSV_DIR}")
    print(f"eval-split：{args.eval_split}")
    if args.eval_split == "calib":
        print(
            f"  calib indices: seed={args.calib_seed} "
            f"dir={args.calib_indices_dir}（只读，不重建）"
        )
    print(f"分层随机：stratify={stratify}")
    if stratify == "depth":
        bins_msg = (
            str(args.depth_bins)
            if args.depth_bins is not None
            else f"自适应(宽≈{DEPTH_BIN_WIDTH_HINT}, "
            f"[{DEPTH_BIN_COUNT_MIN},{DEPTH_BIN_COUNT_MAX}])"
        )
        print(
            f"随机 depth：n={args.random_schemes}  seed={args.random_seed}  "
            f"depth_bins={bins_msg}"
        )
    else:
        bins_msg = (
            str(args.score_bins)
            if args.score_bins is not None
            else f"自适应(log10宽≈{SCORE_BIN_WIDTH_HINT}, "
            f"[{SCORE_BIN_COUNT_MIN},{SCORE_BIN_COUNT_MAX}])"
        )
        print(
            f"随机 score：n={args.random_schemes}  seed={args.random_seed}  "
            f"score_bins={bins_msg}"
        )

    summary_rows: list[dict] = []
    plot_random: dict[str, list[SchemeMetrics]] = {}

    for task_name in args.tasks:
        rows_random = run_random(
            task_name,
            csv_dir=SCORE_CSV_DIR,
            n_schemes=args.random_schemes,
            seed=args.random_seed,
            eval_samples=args.eval_samples,
            stratify=stratify,
            n_depth_bins=args.depth_bins,
            n_score_bins=args.score_bins,
            eval_split=args.eval_split,
            calib_seed=args.calib_seed,
            calib_indices_dir=args.calib_indices_dir,
        )
        summary_random = summarize_correlations(
            task_name, random_tag, rows_random
        )
        path_random = save_metrics_csv(
            task_name, random_tag, rows_random, SCORE_CSV_DIR
        )
        plot_random[task_name] = rows_random
        print(f"  随机明细：{path_random}")
        summary_rows.append(
            {
                "task": task_name,
                "source": random_tag,
                "n_schemes": summary_random["n_schemes"],
                "score_csv_dir": SCORE_CSV_DIR,
                "spearman_rho_kl": (
                    f"{summary_random['kl_spearman_rho']:.6f}"
                ),
                "spearman_p_kl": f"{summary_random['kl_spearman_p']:.6e}",
                "kendall_tau_kl": (
                    f"{summary_random['kl_kendall_tau']:.6f}"
                ),
                "kendall_p_kl": f"{summary_random['kl_kendall_p']:.6e}",
            }
        )

    if plot_random:
        pdf = save_score_kl_scatter_pdf(random_tag, plot_random)
        print(f"\n随机图：{pdf}")

    if summary_rows:
        path = save_summary_csv(summary_rows, tag=stratify)
        print(f"汇总：{path}")


if __name__ == "__main__":
    main()
