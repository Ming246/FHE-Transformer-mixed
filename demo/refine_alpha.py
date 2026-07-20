"""
优化 α / 代理：诊断 + 多种标定，门控选优。

问题（ratio-median 原版）：
  α 动态范围可达 ~80×，全 low 方案下 L8+L2 softmax 独占 ΣαS 的 ~77%，
  NSGA 几乎只优化少数位置 → 全量 Pareto 反而差于未加权 ΣS。

本脚本从已有 {task}_kl_single.csv + S1 生成多种权重/代理，并在方案集上比 Spearman：

  ratio_raw      — 原 median(K/S)（对照）
  ratio_klw      — 按 K 加权的 log(K/S) 几何平均，再 median 归一
  ratio_clip4    — ratio_klw 后 log-clip 到 [1/4, 4]，再归一
  ratio_clip3    — 同上，M=3
  nnls_clip4     — 多方案非负最小二乘拟合 α（有界），初值=ratio_klw
  sum_k          — 直接用单点 K 表作「敏感度」（推荐强基线）

输出：
  demo/results/alpha_refined/<task>/
    {task}_alpha_<method>.csv
    {task}_sensitivity_k.csv   （S_low/mid/high := K_*，供 evolution 当 sensitive-dir）
    {task}_gate_summary.csv

用法：
  python3 demo/refine_alpha.py --task mrpc rte sst2 --samples 64 --n-random 60
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
from scipy.optimize import nnls
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
    NUM_LAYERS,
    SCHEME_LEN,
    SCHEME_ORIGINAL,
    compute_scheme_cost,
    scheme_index,
)
from evolution_infer import SchemeEvaluator  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402

from calibrate_alpha_s1 import (  # noqa: E402
    EPS,
    KINDS,
    KL_RATIO_FLOOR,
    LEVEL_NAMES,
    LEVELS,
    SlotRow,
    iter_positions,
    level_allowed,
    load_s1_matrix,
)

POLY_LEVELS = (0, 1, 2)


def load_kl_single(path: str) -> dict[tuple[int, str, int], float]:
    out: dict[tuple[int, str, int], float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("skipped", "")).strip() in ("1", "true", "True"):
                continue
            try:
                layer_idx = int(row["layer_idx"])
                kind = row["kind"]
                level = int(row["level"])
                kl = float(row["output_kl"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(kl):
                out[(layer_idx, kind, level)] = max(0.0, kl)
    return out


def normalize_median(raw: np.ndarray) -> np.ndarray:
    finite = raw[np.isfinite(raw) & (raw > 0)]
    scale = float(np.median(finite)) if len(finite) else 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    out = np.ones_like(raw, dtype=np.float64)
    for i, v in enumerate(raw):
        out[i] = (float(v) / scale) if math.isfinite(v) and v > 0 else 1.0
    return out


def log_clip(alpha: np.ndarray, max_ratio: float) -> np.ndarray:
    """将 α 限制在 [1/M, M]（相对 median=1 的空间），再重新 median 归一。"""
    lo, hi = 1.0 / max_ratio, max_ratio
    clipped = np.clip(alpha, lo, hi)
    return normalize_median(clipped)


def alpha_ratio_median(
    kl: dict[tuple[int, str, int], float],
    s_mat: dict[tuple[int, str], SlotRow],
) -> np.ndarray:
    raw = []
    for layer_idx, kind in iter_positions():
        ratios = []
        slot = s_mat[(layer_idx, kind)]
        for level in LEVELS:
            if not level_allowed(layer_idx, kind, level):
                continue
            k = kl.get((layer_idx, kind, level))
            if k is None or k < KL_RATIO_FLOOR:
                continue
            ratios.append(k / (slot.s_at(level) + EPS))
        raw.append(float(np.median(ratios)) if ratios else float("nan"))
    return normalize_median(np.asarray(raw, dtype=np.float64))


def alpha_ratio_kl_weighted(
    kl: dict[tuple[int, str, int], float],
    s_mat: dict[tuple[int, str], SlotRow],
) -> np.ndarray:
    """
    α_i = exp( Σ_k w_k log(K/S) / Σ w )，w_k = K_{i,k}。
    大 KL 档主导比值，避免 mid/high 近零噪声。
    """
    raw = []
    for layer_idx, kind in iter_positions():
        slot = s_mat[(layer_idx, kind)]
        num = 0.0
        den = 0.0
        for level in LEVELS:
            if not level_allowed(layer_idx, kind, level):
                continue
            k = kl.get((layer_idx, kind, level), 0.0)
            if k < KL_RATIO_FLOOR:
                continue
            s = slot.s_at(level) + EPS
            num += k * math.log(k / s)
            den += k
        raw.append(math.exp(num / den) if den > 0 else float("nan"))
    return normalize_median(np.asarray(raw, dtype=np.float64))


def feature_matrix(
    schemes: list[list[int]],
    s_mat: dict[tuple[int, str], SlotRow],
) -> np.ndarray:
    """N×48，original→0。"""
    x = np.zeros((len(schemes), SCHEME_LEN), dtype=np.float64)
    for n, scheme in enumerate(schemes):
        for layer_idx, kind in iter_positions():
            idx = scheme_index(layer_idx, kind)
            level = scheme[idx]
            if level != SCHEME_ORIGINAL:
                x[n, idx] = s_mat[(layer_idx, kind)].s_at(level)
    return x


def fit_alpha_nnls(
    schemes: list[list[int]],
    y_kl: np.ndarray,
    s_mat: dict[tuple[int, str], SlotRow],
    *,
    init: np.ndarray,
    max_ratio: float = 4.0,
    ridge: float = 1e-3,
) -> np.ndarray:
    """
    min ||X α - y||² + ridge||α - 1||²，α≥0，再可选 log-clip。
    max_ratio<=0 或 >=1e6 时不做硬裁剪。
    """
    x = feature_matrix(schemes, s_mat)
    y = np.asarray(y_kl, dtype=np.float64)
    mask = np.isfinite(y) & (y >= 0)
    x, y = x[mask], y[mask]
    if len(y) < 10:
        alpha = normalize_median(init)
    else:
        eye = np.sqrt(ridge) * np.eye(SCHEME_LEN)
        x_aug = np.vstack([x, eye])
        y_aug = np.concatenate([y, np.sqrt(ridge) * np.ones(SCHEME_LEN)])
        alpha, _ = nnls(x_aug, y_aug)
        alpha = np.maximum(alpha, 1e-12)
        alpha = normalize_median(alpha)
    if max_ratio is None or max_ratio <= 0 or max_ratio >= 1e6:
        return alpha
    return log_clip(alpha, max_ratio)


def write_alpha_csv(
    path: str,
    alpha: np.ndarray,
    *,
    method: str,
    s_mat: dict[tuple[int, str], SlotRow],
    kl: dict[tuple[int, str, int], float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pos_idx",
                "layer_idx",
                "kind",
                "alpha",
                "method",
                "S_low",
                "S_mid",
                "S_high",
                "K_low",
                "K_mid",
                "K_high",
            ]
        )
        for i, (layer_idx, kind) in enumerate(iter_positions()):
            slot = s_mat[(layer_idx, kind)]
            w.writerow(
                [
                    i,
                    layer_idx,
                    kind,
                    f"{float(alpha[i]):.10e}",
                    method,
                    f"{slot.s_low:.10e}",
                    f"{slot.s_mid:.10e}",
                    f"{slot.s_high:.10e}",
                    f"{kl.get((layer_idx, kind, 0), float('nan')):.10e}",
                    f"{kl.get((layer_idx, kind, 1), float('nan')):.10e}",
                    f"{kl.get((layer_idx, kind, 2), float('nan')):.10e}",
                ]
            )


def write_sensitivity_from_k(
    path: str,
    kl: dict[tuple[int, str, int], float],
) -> None:
    """把单点 K 写成 sensitivity.csv，供 evolution_score 直接 ΣK。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer_idx", "kind", "S_low", "S_mid", "S_high"])
        for layer_idx, kind in iter_positions():
            vals = []
            for level in LEVELS:
                if not level_allowed(layer_idx, kind, level):
                    # 禁止档：给极大惩罚，避免被选（evolution 本就不会选）
                    vals.append(0.0)
                else:
                    vals.append(float(kl.get((layer_idx, kind, level), 0.0)))
            w.writerow(
                [layer_idx, kind, f"{vals[0]:.10e}", f"{vals[1]:.10e}", f"{vals[2]:.10e}"]
            )


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "gelu":
        return tuple(lv for lv in POLY_LEVELS if gelu_level_allowed(layer_idx, lv))
    return POLY_LEVELS


def random_legal_scheme(rng: random.Random) -> list[int]:
    scheme: list[int] = []
    for layer_idx in range(NUM_LAYERS):
        for kind in KINDS:
            choices = list(allowed_levels(layer_idx, kind)) + [SCHEME_ORIGINAL]
            scheme.append(rng.choice(choices))
    return scheme


def sample_schemes(task: str, n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pool = [random_legal_scheme(rng) for _ in range(max(n * 8, n))]
    scored = [(int(compute_scheme_cost(task, s)), s) for s in pool]
    scored.sort(key=lambda t: t[0])
    out: list[list[int]] = []
    for i in range(n):
        j = int(round(i * (len(scored) - 1) / max(n - 1, 1)))
        out.append(scored[j][1])
    uniq: dict[tuple[int, ...], list[int]] = {tuple(s): s for s in out}
    while len(uniq) < n:
        s = random_legal_scheme(rng)
        uniq[tuple(s)] = s
    return list(uniq.values())[:n]


def load_pareto_schemes(path: str, limit: int) -> list[list[int]]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("scheme"):
                continue
            scheme = list(ast.literal_eval(row["scheme"]))
            if len(scheme) == SCHEME_LEN:
                out.append(scheme)
            if len(out) >= limit:
                break
    return out


def proxy_sum_s(scheme: list[int], s_mat, alpha: np.ndarray | None = None) -> float:
    total = 0.0
    for i, (layer_idx, kind) in enumerate(iter_positions()):
        level = scheme[scheme_index(layer_idx, kind)]
        if level == SCHEME_ORIGINAL:
            continue
        s = s_mat[(layer_idx, kind)].s_at(level)
        w = 1.0 if alpha is None else float(alpha[i])
        total += w * s
    return total


def proxy_sum_k(scheme: list[int], kl: dict) -> float:
    total = 0.0
    for layer_idx, kind in iter_positions():
        level = scheme[scheme_index(layer_idx, kind)]
        if level == SCHEME_ORIGINAL:
            continue
        total += float(kl.get((layer_idx, kind, level), 0.0))
    return total


def corr(proxy: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(proxy) & np.isfinite(truth)
    if m.sum() < 3:
        return float("nan"), float("nan")
    rho, _ = spearmanr(proxy[m], truth[m])
    tau, _ = kendalltau(proxy[m], truth[m])
    return float(rho), float(tau)


def alpha_dynamic_range(alpha: np.ndarray) -> float:
    a = alpha[np.isfinite(alpha) & (alpha > 0)]
    if len(a) == 0:
        return float("nan")
    return float(a.max() / a.min())


@dataclass
class MethodSpec:
    key: str
    alpha: np.ndarray | None  # None → ΣK
    use_k: bool = False


def run_task(
    task: str,
    *,
    calib_dir: str,
    sensitive_dir: str,
    out_dir: str,
    eval_samples: int | None,
    seed: int,
    n_random: int,
    n_pareto: int,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    s_mat = load_s1_matrix(task, sensitive_dir)
    kl_path = os.path.join(calib_dir, f"{task}_kl_single.csv")
    if not os.path.isfile(kl_path):
        raise FileNotFoundError(f"缺少单点 KL：{kl_path}（先跑 calibrate_alpha_s1.py）")
    kl = load_kl_single(kl_path)

    a_raw = alpha_ratio_median(kl, s_mat)
    a_klw = alpha_ratio_kl_weighted(kl, s_mat)
    a_c4 = log_clip(a_klw, 4.0)
    a_c3 = log_clip(a_klw, 3.0)

    # 方案集（先采样；NNLS 用其 KL）
    schemes: list[list[int]] = sample_schemes(task, n_random, seed)
    for directory, pattern, lim in (
        (EVOLUTION_S1_DIR, f"{task}_pareto.csv", n_pareto),
        (EVOLUTION_KL_DIR, f"{task}_pareto_kl.csv", n_pareto),
    ):
        schemes.extend(load_pareto_schemes(os.path.join(directory, pattern), lim))
    uniq: dict[tuple[int, ...], list[int]] = {}
    for s in schemes:
        uniq[tuple(s)] = s
    schemes = list(uniq.values())

    print(f"\n=== {task.upper()} refine α === schemes={len(schemes)} samples={eval_samples}")
    evaluator = SchemeEvaluator(task, eval_samples=eval_samples, seed=seed)
    print(f"  eval: {evaluator.eval_desc}")
    y = np.array(
        [float(evaluator.evaluate_for_scheme(s).output_kl) for s in schemes],
        dtype=np.float64,
    )
    for i in range(0, len(schemes), 20):
        print(f"  evaluated {min(i + 20, len(schemes))}/{len(schemes)}")

    a_nnls = fit_alpha_nnls(schemes, y, s_mat, init=a_klw, max_ratio=4.0)

    methods: list[MethodSpec] = [
        MethodSpec("sum_s", None),  # special: alpha ones
        MethodSpec("ratio_raw", a_raw),
        MethodSpec("ratio_klw", a_klw),
        MethodSpec("ratio_clip4", a_c4),
        MethodSpec("ratio_clip3", a_c3),
        MethodSpec("nnls_clip4", a_nnls),
        MethodSpec("sum_k", None, use_k=True),
    ]
    # sum_s uses alpha=1
    ones = np.ones(SCHEME_LEN, dtype=np.float64)

    rows_out = []
    print("\n  --- gate (Spearman vs Output KL) ---")
    best_key, best_rho = "sum_s", -1.0
    for m in methods:
        if m.use_k:
            proxy = np.array([proxy_sum_k(s, kl) for s in schemes])
            dr = float("nan")
        elif m.key == "sum_s":
            proxy = np.array([proxy_sum_s(s, s_mat, ones) for s in schemes])
            dr = 1.0
        else:
            assert m.alpha is not None
            proxy = np.array([proxy_sum_s(s, s_mat, m.alpha) for s in schemes])
            dr = alpha_dynamic_range(m.alpha)
            write_alpha_csv(
                os.path.join(out_dir, f"{task}_alpha_{m.key}.csv"),
                m.alpha,
                method=m.key,
                s_mat=s_mat,
                kl=kl,
            )
        rho, tau = corr(proxy, y)
        print(f"  {m.key:12s}  ρ={rho:.4f}  τ={tau:.4f}  dyn_range={dr:.2f}")
        rows_out.append(
            {
                "method": m.key,
                "spearman_rho": rho,
                "kendall_tau": tau,
                "dyn_range": dr,
                "n": len(schemes),
            }
        )
        if math.isfinite(rho) and rho > best_rho and m.key != "sum_s":
            # 推荐用于进化的：优先 clip/nnls/sum_k（排除 raw）
            if m.key in ("ratio_clip4", "ratio_clip3", "nnls_clip4", "sum_k", "ratio_klw"):
                best_rho, best_key = rho, m.key

    # 也允许 sum_k 赢
    for r in rows_out:
        if r["method"] == "sum_k" and r["spearman_rho"] >= best_rho:
            best_key, best_rho = "sum_k", r["spearman_rho"]

    write_sensitivity_from_k(os.path.join(out_dir, f"{task}_sensitivity_k.csv"), kl)

    # 推荐指针
    recommend_path = os.path.join(out_dir, f"{task}_recommend.txt")
    with open(recommend_path, "w", encoding="utf-8") as f:
        f.write(f"best_method={best_key}\n")
        f.write(f"best_spearman={best_rho:.6f}\n")
        if best_key == "sum_k":
            f.write(
                f"evolution: --sensitive-dir {out_dir} "
                f"(use {task}_sensitivity_k.csv via a dir that contains it as "
                f"{{task}}_sensitivity.csv)\n"
            )
        else:
            f.write(
                f"evolution: --alpha-csv {out_dir}/{task}_alpha_{best_key}.csv "
                f"--sensitive-dir results/sensitive_scores_1\n"
            )

    # sensitivity_k 需要放在名为 sensitive 的目录且文件名 {task}_sensitivity.csv
    k_dir = os.path.join(out_dir, "sensitive_scores_k")
    os.makedirs(k_dir, exist_ok=True)
    src = os.path.join(out_dir, f"{task}_sensitivity_k.csv")
    dst = os.path.join(k_dir, f"{task}_sensitivity.csv")
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        fout.write(fin.read())

    summary = os.path.join(out_dir, f"{task}_gate_summary.csv")
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "spearman_rho", "kendall_tau", "dyn_range", "n"]
        )
        w.writeheader()
        for r in rows_out:
            w.writerow(
                {
                    "method": r["method"],
                    "spearman_rho": f"{r['spearman_rho']:.6f}",
                    "kendall_tau": f"{r['kendall_tau']:.6f}",
                    "dyn_range": (
                        f"{r['dyn_range']:.6f}"
                        if math.isfinite(r["dyn_range"])
                        else ""
                    ),
                    "n": r["n"],
                }
            )
    print(f"  recommend: {best_key} (ρ={best_rho:.4f})")
    print(f"  saved: {summary}")
    return best_key


def main() -> None:
    parser = argparse.ArgumentParser(description="优化 α / ΣK 代理并门控")
    parser.add_argument("--task", nargs="+", default=["mrpc", "rte", "sst2"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=60)
    parser.add_argument("--n-pareto", type=int, default=20)
    parser.add_argument(
        "--calib-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_calib"),
    )
    parser.add_argument("--sensitive-dir", default=SENSITIVE_S1_DIR)
    parser.add_argument(
        "--out-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_refined"),
    )
    args = parser.parse_args()
    eval_samples = None if args.samples <= 0 else args.samples
    print(f"REPO_ROOT={REPO_ROOT}")

    recommends = {}
    for task in args.task:
        recommends[task] = run_task(
            task,
            calib_dir=os.path.join(args.calib_root, task),
            sensitive_dir=args.sensitive_dir,
            out_dir=os.path.join(args.out_root, task),
            eval_samples=eval_samples,
            seed=args.seed,
            n_random=args.n_random,
            n_pareto=args.n_pareto,
        )
    print("\n=== summary recommends ===")
    for t, m in recommends.items():
        print(f"  {t}: {m}")


if __name__ == "__main__":
    main()
