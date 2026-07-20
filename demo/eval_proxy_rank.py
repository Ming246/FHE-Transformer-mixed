"""
门控评估：在多位置方案集上比较三种 proxy 与真 Output KL 的排序相关。

  ΣS   = sum_i S_{i,k_i}
  ΣαS  = sum_i α_i S_{i,k_i}
  ΣK   = sum_i K_{i,k_i}   （单点 KL 表；非法/缺测档当 0）

方案来源：
  - 分层随机（depth，复用 score_spearman 思路的简化版）
  - 可选：外部 Pareto CSV（若存在）

用法：
  python3 demo/eval_proxy_rank.py --task mrpc --samples 64 --n-random 40
  python3 demo/eval_proxy_rank.py --task mrpc --samples 64 --calib-dir demo/results/alpha_calib/mrpc
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
from evolution_infer import SchemeEvaluator, format_elapsed  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402

from calibrate_alpha_s1 import (  # noqa: E402
    KINDS,
    LEVELS,
    LEVEL_NAMES,
    EPS,
    iter_positions,
    level_allowed,
    load_s1_matrix,
)

POLY_LEVELS = (0, 1, 2)


@dataclass
class SchemeRow:
    scheme: list[int]
    source: str
    output_kl: float
    sum_s: float
    sum_alpha_s: float
    sum_k: float
    total_depth: float


def allowed_levels(layer_idx: int, kind: str) -> tuple[int, ...]:
    if kind == "gelu":
        return tuple(lv for lv in POLY_LEVELS if gelu_level_allowed(layer_idx, level=lv))
    return POLY_LEVELS


def load_alpha_vector(alpha_csv: str) -> np.ndarray:
    """按 pos_idx 顺序的长度 48 α；缺文件则全 1。"""
    alpha = np.ones(SCHEME_LEN, dtype=np.float64)
    if not os.path.isfile(alpha_csv):
        print(f"  警告：无 alpha CSV，ΣαS 退化为 ΣS：{alpha_csv}")
        return alpha
    with open(alpha_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["pos_idx"])
            alpha[idx] = float(row["alpha"])
    return alpha


def load_k_table(kl_single_csv: str) -> dict[tuple[int, str], dict[int, float]]:
    """(layer, kind) → {level: K}；缺测/跳过不写入。"""
    table: dict[tuple[int, str], dict[int, float]] = {
        (layer_idx, kind): {} for layer_idx, kind in iter_positions()
    }
    if not os.path.isfile(kl_single_csv):
        print(f"  警告：无 kl_single CSV，ΣK 不可用：{kl_single_csv}")
        return table
    with open(kl_single_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("skipped", "").strip() in ("1", "true", "True"):
                continue
            try:
                layer_idx = int(row["layer_idx"])
                kind = row["kind"]
                level = int(row["level"])
                kl = float(row["output_kl"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(kl):
                table[(layer_idx, kind)][level] = kl
    return table


def scheme_feature_vector(
    scheme: list[int],
    s_mat,
) -> np.ndarray:
    """长度 48：所选档的 S（original→0）。"""
    x = np.zeros(SCHEME_LEN, dtype=np.float64)
    for layer_idx, kind in iter_positions():
        idx = scheme_index(layer_idx, kind)
        level = scheme[idx]
        if level == SCHEME_ORIGINAL:
            x[idx] = 0.0
        else:
            x[idx] = s_mat[(layer_idx, kind)].s_at(level)
    return x


def scheme_sum_k(scheme: list[int], k_table: dict) -> float:
    total = 0.0
    for layer_idx, kind in iter_positions():
        level = scheme[scheme_index(layer_idx, kind)]
        if level == SCHEME_ORIGINAL:
            continue
        total += float(k_table[(layer_idx, kind)].get(level, 0.0))
    return total


def random_legal_scheme(rng: random.Random) -> list[int]:
    scheme: list[int] = []
    for layer_idx in range(NUM_LAYERS):
        for kind in KINDS:
            choices = list(allowed_levels(layer_idx, kind)) + [SCHEME_ORIGINAL]
            scheme.append(rng.choice(choices))
    return scheme


def sample_random_schemes(
    task_name: str,
    n: int,
    seed: int,
) -> list[list[int]]:
    """粗分层：按 depth 分位数尽量散开。"""
    rng = random.Random(seed)
    pool = [random_legal_scheme(rng) for _ in range(max(n * 8, n))]
    scored = [(int(compute_scheme_cost(task_name, s)), s) for s in pool]
    scored.sort(key=lambda t: t[0])
    if len(scored) <= n:
        return [s for _, s in scored]
    # 均匀取 n 个 depth 分位
    out: list[list[int]] = []
    for i in range(n):
        j = int(round(i * (len(scored) - 1) / max(n - 1, 1)))
        out.append(scored[j][1])
    # 去重
    uniq: dict[tuple[int, ...], list[int]] = {}
    for s in out:
        uniq[tuple(s)] = s
    # 不足则补随机
    while len(uniq) < n:
        s = random_legal_scheme(rng)
        uniq[tuple(s)] = s
    return list(uniq.values())[:n]


def load_pareto_schemes(path: str, limit: int | None = None) -> list[list[int]]:
    if not os.path.isfile(path):
        return []
    schemes: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "scheme" not in row or not row["scheme"]:
                continue
            scheme = ast.literal_eval(row["scheme"])
            if len(scheme) != SCHEME_LEN:
                continue
            schemes.append(list(scheme))
            if limit is not None and len(schemes) >= limit:
                break
    return schemes


def corr_report(proxy: np.ndarray, truth: np.ndarray, name: str) -> dict[str, float]:
    mask = np.isfinite(proxy) & np.isfinite(truth)
    p = proxy[mask]
    t = truth[mask]
    if len(p) < 3:
        print(f"  {name}: n={len(p)} 不足")
        return {"spearman_rho": float("nan"), "kendall_tau": float("nan"), "n": float(len(p))}
    rho, _ = spearmanr(p, t)
    tau, _ = kendalltau(p, t)
    print(f"  {name}: n={len(p)}  Spearman ρ={rho:.4f}  Kendall τ={tau:.4f}")
    return {
        "spearman_rho": float(rho),
        "kendall_tau": float(tau),
        "n": float(len(p)),
    }


def run_task(
    task_name: str,
    *,
    calib_dir: str,
    sensitive_dir: str,
    eval_samples: int | None,
    seed: int,
    n_random: int,
    n_pareto: int,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    alpha_csv = os.path.join(calib_dir, f"{task_name}_alpha.csv")
    kl_single = os.path.join(calib_dir, f"{task_name}_kl_single.csv")

    s_mat = load_s1_matrix(task_name, sensitive_dir)
    alpha = load_alpha_vector(alpha_csv)
    k_table = load_k_table(kl_single)

    schemes: list[tuple[str, list[int]]] = []
    for s in sample_random_schemes(task_name, n_random, seed):
        schemes.append(("random", s))
    for label, directory, pattern in (
        ("pareto_s1", EVOLUTION_S1_DIR, f"{task_name}_pareto.csv"),
        ("pareto_kl", EVOLUTION_KL_DIR, f"{task_name}_pareto_kl.csv"),
    ):
        path = os.path.join(directory, pattern)
        for s in load_pareto_schemes(path, limit=n_pareto):
            schemes.append((label, s))

    # 去重（保留先出现的 source）
    uniq: dict[tuple[int, ...], tuple[str, list[int]]] = {}
    for src, s in schemes:
        key = tuple(s)
        if key not in uniq:
            uniq[key] = (src, s)
    schemes = list(uniq.values())
    print(
        f"\n=== {task_name.upper()} proxy rank ===\n"
        f"  schemes={len(schemes)}  calib={calib_dir}\n"
        f"  samples={eval_samples if eval_samples else 'full'}"
    )

    evaluator = SchemeEvaluator(task_name, eval_samples=eval_samples, seed=seed)
    print(f"  eval: {evaluator.eval_desc}")

    rows: list[SchemeRow] = []
    for i, (src, scheme) in enumerate(schemes, start=1):
        m = evaluator.evaluate_for_scheme(scheme)
        x = scheme_feature_vector(scheme, s_mat)
        sum_s = float(x.sum())
        sum_as = float((alpha * x).sum())
        sum_k = scheme_sum_k(scheme, k_table)
        depth = float(compute_scheme_cost(task_name, scheme))
        rows.append(
            SchemeRow(
                scheme=scheme,
                source=src,
                output_kl=float(m.output_kl),
                sum_s=sum_s,
                sum_alpha_s=sum_as,
                sum_k=sum_k,
                total_depth=depth,
            )
        )
        if i % 10 == 0 or i == len(schemes):
            print(f"  evaluated {i}/{len(schemes)}")

    kl = np.array([r.output_kl for r in rows], dtype=np.float64)
    print("\n--- 排序相关（proxy vs Output KL）---")
    metrics = {
        "sum_s": corr_report(np.array([r.sum_s for r in rows]), kl, "ΣS"),
        "sum_alpha_s": corr_report(
            np.array([r.sum_alpha_s for r in rows]), kl, "ΣαS"
        ),
        "sum_k": corr_report(np.array([r.sum_k for r in rows]), kl, "ΣK"),
    }

    # 门控提示
    rho_s = metrics["sum_s"]["spearman_rho"]
    rho_a = metrics["sum_alpha_s"]["spearman_rho"]
    rho_k = metrics["sum_k"]["spearman_rho"]
    print("\n--- 门控 ---")
    if math.isfinite(rho_k) and math.isfinite(rho_s) and rho_k > rho_s + 0.05:
        print("  ΣK ≫ ΣS：加性 KL 表更强；可考虑直接用 ΣK 作 proxy。")
    if math.isfinite(rho_a) and math.isfinite(rho_s):
        if rho_a > rho_s + 0.02:
            print("  ΣαS 优于 ΣS：权重有效，可进入短进化验收。")
        elif rho_a < rho_s - 0.02:
            print("  ΣαS 差于 ΣS：检查标定子集/α 稳定性。")
        else:
            print("  ΣαS ≈ ΣS：单点权重收益有限。")
    if math.isfinite(rho_k) and math.isfinite(rho_a) and rho_k > rho_a + 0.05:
        print("  ΣK ≫ ΣαS：S 形状与 K 不一致，α 线性校正不够。")

    out_csv = os.path.join(out_dir, f"{task_name}_proxy_rank.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "output_kl",
                "sum_s",
                "sum_alpha_s",
                "sum_k",
                "total_depth",
                "scheme",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.source,
                    f"{r.output_kl:.10e}",
                    f"{r.sum_s:.10e}",
                    f"{r.sum_alpha_s:.10e}",
                    f"{r.sum_k:.10e}",
                    f"{r.total_depth:.0f}",
                    repr(r.scheme),
                ]
            )
    summary_path = os.path.join(out_dir, f"{task_name}_proxy_rank_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["proxy", "spearman_rho", "kendall_tau", "n"])
        for name, m in metrics.items():
            w.writerow(
                [name, f"{m['spearman_rho']:.6f}", f"{m['kendall_tau']:.6f}", int(m["n"])]
            )
    print(f"  saved: {out_csv}")
    print(f"  saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ΣS / ΣαS / ΣK vs Output KL 排序门控")
    parser.add_argument("--task", nargs="+", default=["mrpc"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=40)
    parser.add_argument("--n-pareto", type=int, default=20)
    parser.add_argument(
        "--calib-dir",
        default=None,
        help="含 {task}_alpha.csv / {task}_kl_single.csv 的目录；默认 demo/results/alpha_calib/<task>",
    )
    parser.add_argument("--sensitive-dir", default=SENSITIVE_S1_DIR)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(DEMO_RESULTS_DIR, "proxy_rank"),
    )
    args = parser.parse_args()
    eval_samples = None if args.samples <= 0 else args.samples

    print(f"REPO_ROOT={REPO_ROOT}")
    for task in args.task:
        calib = args.calib_dir or os.path.join(
            DEMO_RESULTS_DIR, "alpha_calib", task
        )
        run_task(
            task,
            calib_dir=calib,
            sensitive_dir=args.sensitive_dir,
            eval_samples=eval_samples,
            seed=args.seed,
            n_random=args.n_random,
            n_pareto=args.n_pareto,
            out_dir=os.path.join(args.out_dir, task),
        )


if __name__ == "__main__":
    main()
