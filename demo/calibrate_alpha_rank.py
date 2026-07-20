"""
对照采样 + 桶内 pairwise 排序损失标定 α。

配方：
  1) 正例：KL Pareto 方案（尽量全收）
  2) 硬负例：ΣS1 Pareto 方案（尽量全收）
  3) 稀疏扰动：每 depth 四分位桶内采若干「仅 1–2 位置降档」方案
  4) 几乎不采多位置乱降的随机方案

拟合：在同 depth 桶内，对 KL 序对最大化 score 同序
  loss = Σ softplus( score_worse - score_better 的反号 ) 即 softplus(s_good - s_bad)
  其中 score = Σ α_i S_i；硬对照（KL前沿 vs S1前沿）加权更大。
  最后 soft 压缩动态范围到 target_range。

输出：
  demo/results/alpha_rank/<task>/{task}_alpha.csv
  demo/results/alpha_rank/for_evolution/<task>/{task}_alpha.csv

用法：
  python3 demo/calibrate_alpha_rank.py --task mrpc rte sst2 --samples 64
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from _repo import (  # noqa: E402
    DEMO_RESULTS_DIR,
    EVOLUTION_KL_DIR,
    EVOLUTION_S1_DIR,
    REPO_ROOT,
    SENSITIVE_S1_DIR,
    chdir_repo,
)

chdir_repo()

from cost import (  # noqa: E402
    SCHEME_LEN,
    SCHEME_ORIGINAL,
    compute_scheme_cost,
    scheme_index,
)
from evolution_infer import SchemeEvaluator  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402

from calibrate_alpha_s1 import (  # noqa: E402
    KINDS,
    LEVELS,
    iter_positions,
    level_allowed,
    load_s1_matrix,
)
from refine_alpha import (  # noqa: E402
    normalize_median,
    write_alpha_csv,
    load_kl_single,
)

POLY_LEVELS = (0, 1, 2)


@dataclass
class CalibItem:
    scheme: list[int]
    source: str  # kl_pareto | s1_pareto | sparse
    depth: float
    output_kl: float = float("nan")


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "gelu":
        return tuple(lv for lv in POLY_LEVELS if gelu_level_allowed(layer_idx, lv))
    return POLY_LEVELS


def load_pareto_items(path: str, source: str) -> list[CalibItem]:
    if not os.path.isfile(path):
        return []
    items: list[CalibItem] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("scheme"):
                continue
            scheme = list(ast.literal_eval(row["scheme"]))
            if len(scheme) != SCHEME_LEN:
                continue
            try:
                depth = float(row["total_depth"])
            except (KeyError, TypeError, ValueError):
                depth = float(compute_scheme_cost("mrpc", scheme))  # fallback unused
            kl = float("nan")
            if row.get("output_kl"):
                try:
                    kl = float(row["output_kl"])
                except ValueError:
                    kl = float("nan")
            items.append(CalibItem(scheme=scheme, source=source, depth=depth, output_kl=kl))
    return items


def all_high_scheme() -> list[int]:
    return [2] * SCHEME_LEN


def sparse_degrade(
    rng: random.Random,
    *,
    n_flip: int,
) -> list[int]:
    """全 high 基线，随机改 1–2 个合法位置到 low/mid。"""
    scheme = all_high_scheme()
    slots: list[tuple[int, str]] = []
    for layer_idx, kind in iter_positions():
        if allowed_levels(layer_idx, kind):
            slots.append((layer_idx, kind))
    rng.shuffle(slots)
    n_flip = max(1, min(n_flip, len(slots)))
    for layer_idx, kind in slots[:n_flip]:
        levels = allowed_levels(layer_idx, kind)
        # 倾向 low（0），偶尔 mid
        if len(levels) == 1:
            lv = levels[0]
        else:
            lv = 0 if 0 in levels and rng.random() < 0.7 else (
                1 if 1 in levels else levels[0]
            )
        scheme[scheme_index(layer_idx, kind)] = lv
    return scheme


def depth_bin_edges(depths: list[float], n_bins: int = 4) -> np.ndarray:
    d = np.asarray(depths, dtype=np.float64)
    if len(d) < n_bins:
        return np.array([d.min(), d.max() + 1e-6])
    edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([d.min(), d.max() + 1e-6])
    return edges


def assign_bin(depth: float, edges: np.ndarray) -> int:
    """edges 长度为 n_bins+1；返回 0..n_bins-1。"""
    if len(edges) < 2:
        return 0
    # 右开区间，最后一桶右闭
    idx = int(np.searchsorted(edges, depth, side="right") - 1)
    return int(np.clip(idx, 0, len(edges) - 2))


def build_calib_set(
    task: str,
    *,
    seed: int,
    n_sparse_per_bin: int,
    n_bins: int,
) -> list[CalibItem]:
    rng = random.Random(seed)
    s1 = load_pareto_items(
        os.path.join(EVOLUTION_S1_DIR, f"{task}_pareto.csv"), "s1_pareto"
    )
    klp = load_pareto_items(
        os.path.join(EVOLUTION_KL_DIR, f"{task}_pareto_kl.csv"), "kl_pareto"
    )
    # 修正 depth（用本 task cost）
    for it in s1 + klp:
        it.depth = float(compute_scheme_cost(task, it.scheme))

    frontier = s1 + klp
    edges = depth_bin_edges([it.depth for it in frontier], n_bins=n_bins)

    sparse: list[CalibItem] = []
    for b in range(len(edges) - 1):
        for _ in range(n_sparse_per_bin):
            n_flip = 1 if rng.random() < 0.55 else 2
            scheme = sparse_degrade(rng, n_flip=n_flip)
            # 拒绝过深/过浅：尽量落在桶附近——多试几次
            best = scheme
            best_pen = 1e18
            for _try in range(12):
                s = sparse_degrade(rng, n_flip=n_flip)
                d = float(compute_scheme_cost(task, s))
                # 目标落在桶中心
                mid = 0.5 * (edges[b] + edges[b + 1])
                pen = abs(d - mid)
                if pen < best_pen:
                    best_pen, best = pen, s
            d = float(compute_scheme_cost(task, best))
            sparse.append(CalibItem(scheme=best, source="sparse", depth=d))

    # 去重，保留优先：kl_pareto > s1_pareto > sparse
    priority = {"kl_pareto": 0, "s1_pareto": 1, "sparse": 2}
    uniq: dict[tuple[int, ...], CalibItem] = {}
    for it in sorted(frontier + sparse, key=lambda x: priority[x.source]):
        key = tuple(it.scheme)
        if key not in uniq:
            uniq[key] = it
    items = list(uniq.values())
    print(
        f"  calib set: total={len(items)}  "
        f"kl_pareto={sum(i.source=='kl_pareto' for i in items)}  "
        f"s1_pareto={sum(i.source=='s1_pareto' for i in items)}  "
        f"sparse={sum(i.source=='sparse' for i in items)}  "
        f"depth_edges={np.round(edges,1)}"
    )
    return items


def feature_row(scheme: list[int], s_mat) -> np.ndarray:
    x = np.zeros(SCHEME_LEN, dtype=np.float64)
    for i, (layer_idx, kind) in enumerate(iter_positions()):
        level = scheme[scheme_index(layer_idx, kind)]
        if level == SCHEME_ORIGINAL:
            continue
        x[i] = s_mat[(layer_idx, kind)].s_at(level)
    return x


def softplus(z: np.ndarray | float) -> np.ndarray | float:
    # stable softplus
    z = np.asarray(z, dtype=np.float64)
    return np.where(z > 20, z, np.log1p(np.exp(np.clip(z, -40, 40))))


def dyn_range(alpha: np.ndarray) -> float:
    a = alpha[np.isfinite(alpha) & (alpha > 0)]
    return float(a.max() / a.min()) if len(a) else float("nan")


def soft_power(alpha: np.ndarray, p: float) -> np.ndarray:
    return normalize_median(np.power(np.maximum(alpha, 1e-12), p))


def target_power(alpha: np.ndarray, target: float) -> float:
    r = dyn_range(alpha)
    if not math.isfinite(r) or r <= target:
        return 1.0
    return float(math.log(target) / math.log(r))


def fit_alpha_pairwise(
    X: np.ndarray,
    y: np.ndarray,
    sources: list[str],
    depths: np.ndarray,
    *,
    n_bins: int = 4,
    hard_weight: float = 3.0,
    ridge: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    最小化桶内排序铰链：希望 KL 更好 ⇒ score 更小。
    score = X α, α>0。
    """
    edges = depth_bin_edges(list(depths), n_bins=n_bins)
    bins: list[list[int]] = [[] for _ in range(len(edges) - 1)]
    for i, d in enumerate(depths):
        bins[assign_bin(float(d), edges)].append(i)

    # 预生成序对 (good_idx, bad_idx, weight)
    pairs: list[tuple[int, int, float]] = []
    for idxs in bins:
        if len(idxs) < 2:
            continue
        for a in idxs:
            for b in idxs:
                if a >= b:
                    continue
                ya, yb = y[a], y[b]
                if not (math.isfinite(ya) and math.isfinite(yb)):
                    continue
                if abs(ya - yb) < 1e-12:
                    continue
                if ya < yb:
                    good, bad = a, b
                else:
                    good, bad = b, a
                w = 1.0
                sa, sb = sources[good], sources[bad]
                # 硬对照：正例来自 KL 前沿、负例来自 S1 前沿
                if sa == "kl_pareto" and sb == "s1_pareto":
                    w = hard_weight
                elif sa == "kl_pareto" or sb == "s1_pareto":
                    w = 1.5
                # KL 差距越大越重要（log 尺度）
                gap = abs(math.log(max(y[bad], 1e-12)) - math.log(max(y[good], 1e-12)))
                w *= 1.0 + min(gap, 5.0)
                pairs.append((good, bad, w))

    if not pairs:
        return np.ones(SCHEME_LEN, dtype=np.float64)

    print(f"  pairwise pairs={len(pairs)}  bins={len(bins)}")

    rng = np.random.default_rng(seed)
    # 参数：log α，初始化 0 → α=1
    x0 = rng.normal(0.0, 0.01, size=SCHEME_LEN)

    def pack_alpha(log_a: np.ndarray) -> np.ndarray:
        a = np.exp(np.clip(log_a, -8.0, 8.0))
        return normalize_median(a)

    def objective(log_a: np.ndarray) -> float:
        alpha = pack_alpha(log_a)
        scores = X @ alpha
        loss = 0.0
        for good, bad, w in pairs:
            # 希望 score[good] < score[bad]；若 good 更大则惩罚
            loss += float(w * softplus(scores[good] - scores[bad]))
        loss /= max(len(pairs), 1)
        loss += ridge * float(np.mean(log_a ** 2))
        return loss

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 400, "ftol": 1e-10},
    )
    alpha = pack_alpha(res.x)
    print(f"  optimize success={res.success}  loss={res.fun:.6f}  dyn={dyn_range(alpha):.2f}")
    return alpha


def depth_stratified_spearman(
    scores: np.ndarray, y: np.ndarray, depths: np.ndarray, n_bins: int = 4
) -> float:
    """各 depth 桶内 Spearman 再平均（更贴搜索）。"""
    edges = depth_bin_edges(list(depths), n_bins=n_bins)
    rhos = []
    for b in range(len(edges) - 1):
        idxs = [i for i, d in enumerate(depths) if assign_bin(float(d), edges) == b]
        if len(idxs) < 4:
            continue
        s = scores[idxs]
        t = y[idxs]
        if np.std(s) < 1e-15 or np.std(t) < 1e-15:
            continue
        rho, _ = spearmanr(s, t)
        if math.isfinite(rho):
            rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def run_task(
    task: str,
    *,
    samples: int | None,
    seed: int,
    n_sparse_per_bin: int,
    n_bins: int,
    target_range: float,
    hard_weight: float,
    out_root: str,
    sensitive_dir: str,
    calib_root: str,
) -> None:
    out_dir = os.path.join(out_root, task)
    os.makedirs(out_dir, exist_ok=True)
    s_mat = load_s1_matrix(task, sensitive_dir)
    kl_path = os.path.join(calib_root, task, f"{task}_kl_single.csv")
    kl_table = load_kl_single(kl_path) if os.path.isfile(kl_path) else {}

    print(f"\n=== {task.upper()} rank-α calib ===")
    items = build_calib_set(
        task, seed=seed, n_sparse_per_bin=n_sparse_per_bin, n_bins=n_bins
    )

    # 评估 KL：Pareto 可选用 CSV 缓存加速，但为与 sparse 一致仍统一重算
    ev = SchemeEvaluator(task, eval_samples=samples, seed=seed)
    print(f"  eval: {ev.eval_desc}")
    for i, it in enumerate(items, 1):
        it.output_kl = float(ev.evaluate_for_scheme(it.scheme).output_kl)
        it.depth = float(compute_scheme_cost(task, it.scheme))
        if i % 25 == 0 or i == len(items):
            print(f"  evaluated {i}/{len(items)}")

    X = np.stack([feature_row(it.scheme, s_mat) for it in items])
    y = np.array([it.output_kl for it in items], dtype=np.float64)
    depths = np.array([it.depth for it in items], dtype=np.float64)
    sources = [it.source for it in items]

    alpha_raw = fit_alpha_pairwise(
        X,
        y,
        sources,
        depths,
        n_bins=n_bins,
        hard_weight=hard_weight,
        ridge=1e-3,
        seed=seed,
    )
    p = target_power(alpha_raw, target_range)
    alpha = soft_power(alpha_raw, p) if p < 1.0 else alpha_raw
    print(f"  soft p={p:.3f}  dyn={dyn_range(alpha):.2f}")

    # 门控
    ones = np.ones(SCHEME_LEN)
    score_s = X @ ones
    score_a = X @ alpha
    rho_s, _ = spearmanr(score_s, y)
    rho_a, _ = spearmanr(score_a, y)
    tau_a, _ = kendalltau(score_a, y)
    rho_s_bin = depth_stratified_spearman(score_s, y, depths, n_bins)
    rho_a_bin = depth_stratified_spearman(score_a, y, depths, n_bins)
    print(
        f"  gate global:  ΣS ρ={rho_s:.4f}  ΣαS ρ={rho_a:.4f}  τ={tau_a:.4f}\n"
        f"  gate depth-bin mean Spearman:  ΣS ρ={rho_s_bin:.4f}  ΣαS ρ={rho_a_bin:.4f}"
    )

    # 桶内硬对照：同桶 KL-pareto vs S1-pareto 的 score 逆序率
    edges = depth_bin_edges(list(depths), n_bins=n_bins)
    inv_s = inv_a = n_pair = 0
    for b in range(len(edges) - 1):
        kl_idx = [
            i
            for i, it in enumerate(items)
            if it.source == "kl_pareto" and assign_bin(it.depth, edges) == b
        ]
        s1_idx = [
            i
            for i, it in enumerate(items)
            if it.source == "s1_pareto" and assign_bin(it.depth, edges) == b
        ]
        for i in kl_idx:
            for j in s1_idx:
                if y[i] >= y[j]:
                    continue  # 只要 KL 正例确实更好的对
                n_pair += 1
                if score_s[i] > score_s[j]:
                    inv_s += 1
                if score_a[i] > score_a[j]:
                    inv_a += 1
    if n_pair:
        print(
            f"  hard-contrast inversion (want low):  "
            f"ΣS {inv_s}/{n_pair}={inv_s/n_pair:.2%}  "
            f"ΣαS {inv_a}/{n_pair}={inv_a/n_pair:.2%}"
        )

    write_alpha_csv(
        os.path.join(out_dir, f"{task}_alpha.csv"),
        alpha,
        method=f"pairwise_rank_soft{p:.2f}",
        s_mat=s_mat,
        kl=kl_table,
    )
    # 保存标定集
    with open(os.path.join(out_dir, f"{task}_calib_schemes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "depth", "output_kl", "score_s", "score_alpha", "scheme"])
        for i, it in enumerate(items):
            w.writerow(
                [
                    it.source,
                    f"{it.depth:.0f}",
                    f"{it.output_kl:.10e}",
                    f"{score_s[i]:.10e}",
                    f"{score_a[i]:.10e}",
                    repr(it.scheme),
                ]
            )
    with open(os.path.join(out_dir, f"{task}_gate.txt"), "w") as f:
        f.write(f"spearman_S={rho_s:.6f}\n")
        f.write(f"spearman_alpha={rho_a:.6f}\n")
        f.write(f"spearman_S_depthbin={rho_s_bin:.6f}\n")
        f.write(f"spearman_alpha_depthbin={rho_a_bin:.6f}\n")
        f.write(f"kendall_alpha={tau_a:.6f}\n")
        f.write(f"dyn_range={dyn_range(alpha):.6f}\n")
        f.write(f"soft_p={p:.6f}\n")
        f.write(f"n_calib={len(items)}\n")

    bundle = os.path.join(out_root, "for_evolution", task)
    os.makedirs(bundle, exist_ok=True)
    with open(os.path.join(out_dir, f"{task}_alpha.csv")) as fin, open(
        os.path.join(bundle, f"{task}_alpha.csv"), "w"
    ) as fout:
        fout.write(fin.read())
    print(f"  saved {out_dir}/{task}_alpha.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="对照采样 + pairwise 排序标定 α")
    parser.add_argument("--task", nargs="+", default=["mrpc", "rte", "sst2"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sparse-per-bin", type=int, default=10)
    parser.add_argument("--n-bins", type=int, default=4)
    parser.add_argument("--target-range", type=float, default=25.0)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    parser.add_argument("--sensitive-dir", default=SENSITIVE_S1_DIR)
    parser.add_argument(
        "--calib-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_calib"),
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_rank"),
    )
    args = parser.parse_args()
    samples = None if args.samples <= 0 else args.samples
    print(f"REPO_ROOT={REPO_ROOT}")

    for task in args.task:
        run_task(
            task,
            samples=samples,
            seed=args.seed,
            n_sparse_per_bin=args.n_sparse_per_bin,
            n_bins=args.n_bins,
            target_range=args.target_range,
            hard_weight=args.hard_weight,
            out_root=args.out_root,
            sensitive_dir=args.sensitive_dir,
            calib_root=args.calib_root,
        )
    print(f"\nbundle: {args.out_root}/for_evolution")


if __name__ == "__main__":
    main()
