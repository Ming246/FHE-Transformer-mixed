"""
软压缩 α（替代硬 clip），门控后按 task 选最优并写出进化用 CSV。

硬 clip 到 [1/M,M] 会把 SST2 上 ratio_klw 的 ρ 从 ~0.85 砍到 ~0.67。
软压缩：α ← normalize(α^p)，p∈(0,1]，动态范围从 R 变为 R^p，相对序大体保留。

用法：
  python3 demo/soft_alpha_select.py --samples 64
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from _repo import DEMO_RESULTS_DIR, REPO_ROOT, SENSITIVE_S1_DIR, chdir_repo

chdir_repo()

from refine_alpha import (  # noqa: E402
    SCHEME_LEN,
    corr,
    feature_matrix,
    fit_alpha_nnls,
    iter_positions,
    load_kl_single,
    load_pareto_schemes,
    load_s1_matrix,
    normalize_median,
    proxy_sum_k,
    proxy_sum_s,
    sample_schemes,
    write_alpha_csv,
)
from evolution_infer import SchemeEvaluator  # noqa: E402
from _repo import EVOLUTION_KL_DIR, EVOLUTION_S1_DIR  # noqa: E402


def load_alpha_csv(path: str) -> np.ndarray:
    a = np.ones(SCHEME_LEN, dtype=np.float64)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            a[int(row["pos_idx"])] = float(row["alpha"])
    return a


def soft_power(alpha: np.ndarray, p: float) -> np.ndarray:
    a = np.maximum(alpha, 1e-12)
    return normalize_median(np.power(a, p))


def dyn_range(alpha: np.ndarray) -> float:
    a = alpha[np.isfinite(alpha) & (alpha > 0)]
    return float(a.max() / a.min()) if len(a) else float("nan")


def target_power(alpha: np.ndarray, target_range: float = 25.0) -> float:
    """选 p 使 R^p ≈ target_range。"""
    r = dyn_range(alpha)
    if not math.isfinite(r) or r <= target_range:
        return 1.0
    return float(math.log(target_range) / math.log(r))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", nargs="+", default=["mrpc", "rte", "sst2"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-random", type=int, default=60)
    parser.add_argument("--n-pareto", type=int, default=20)
    parser.add_argument("--target-range", type=float, default=25.0)
    parser.add_argument(
        "--refined-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_refined"),
    )
    parser.add_argument(
        "--calib-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_calib"),
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_soft"),
    )
    args = parser.parse_args()
    eval_samples = None if args.samples <= 0 else args.samples
    print(f"REPO_ROOT={REPO_ROOT}")

    for task in args.task:
        refined = os.path.join(args.refined_root, task)
        calib = os.path.join(args.calib_root, task)
        out_dir = os.path.join(args.out_root, task)
        os.makedirs(out_dir, exist_ok=True)

        s_mat = load_s1_matrix(task, SENSITIVE_S1_DIR)
        kl = load_kl_single(os.path.join(calib, f"{task}_kl_single.csv"))
        a_klw = load_alpha_csv(os.path.join(refined, f"{task}_alpha_ratio_klw.csv"))
        a_raw = load_alpha_csv(os.path.join(calib, f"{task}_alpha.csv"))

        schemes = sample_schemes(task, args.n_random, args.seed)
        for d, pat in (
            (EVOLUTION_S1_DIR, f"{task}_pareto.csv"),
            (EVOLUTION_KL_DIR, f"{task}_pareto_kl.csv"),
        ):
            schemes.extend(
                load_pareto_schemes(os.path.join(d, pat), args.n_pareto)
            )
        uniq = {tuple(s): s for s in schemes}
        schemes = list(uniq.values())

        print(f"\n=== {task.upper()} soft-α gate  n={len(schemes)} ===")
        ev = SchemeEvaluator(task, eval_samples=eval_samples, seed=args.seed)
        y = np.array(
            [float(ev.evaluate_for_scheme(s).output_kl) for s in schemes],
            dtype=np.float64,
        )
        print(f"  eval done ({ev.eval_desc})")

        # nnls 不硬裁剪
        a_nnls = fit_alpha_nnls(
            schemes, y, s_mat, init=a_klw, max_ratio=1e9, ridge=1e-3
        )
        # fit_alpha_nnls 末尾仍 log_clip；用超大 max_ratio≈不裁，再 soft
        # 上面 max_ratio=1e9 几乎不裁

        p_klw = target_power(a_klw, args.target_range)
        p_raw = target_power(a_raw, args.target_range)
        p_nnls = target_power(a_nnls, args.target_range)

        candidates: dict[str, np.ndarray] = {
            "sum_s": np.ones(SCHEME_LEN),
            "raw": a_raw,
            "klw": a_klw,
            "nnls_noclip": a_nnls,
            f"klw_soft{p_klw:.2f}": soft_power(a_klw, p_klw),
            f"raw_soft{p_raw:.2f}": soft_power(a_raw, p_raw),
            f"nnls_soft{p_nnls:.2f}": soft_power(a_nnls, p_nnls),
            "klw_p0.35": soft_power(a_klw, 0.35),
            "nnls_p0.35": soft_power(a_nnls, 0.35),
        }

        ones = np.ones(SCHEME_LEN)
        rows = []
        best_key, best_rho = "sum_s", -1.0
        # 搜索必须控制动态范围；硬 clip 伤 ρ，软压缩把 dyn 压到 target_range
        dyn_budget = args.target_range * 1.05
        for key, alpha in candidates.items():
            if key == "sum_s":
                proxy = np.array([proxy_sum_s(s, s_mat, ones) for s in schemes])
            else:
                proxy = np.array([proxy_sum_s(s, s_mat, alpha) for s in schemes])
            rho, tau = corr(proxy, y)
            dr = 1.0 if key == "sum_s" else dyn_range(alpha)
            print(f"  {key:18s} ρ={rho:.4f}  τ={tau:.4f}  dyn={dr:.2f}")
            rows.append((key, rho, tau, dr))
            if key == "sum_s" or not math.isfinite(rho):
                continue
            if dr > dyn_budget:
                continue
            if rho > best_rho:
                best_rho, best_key = rho, key

        if best_key == "sum_s":
            # 回退：在全部候选里选 dyn 最小且 ρ 最高的 soft
            for key, rho, tau, dr in rows:
                if "soft" in key and math.isfinite(rho) and rho > best_rho:
                    best_rho, best_key = rho, key


        # ΣK 对照
        proxy_k = np.array([proxy_sum_k(s, kl) for s in schemes])
        rho_k, tau_k = corr(proxy_k, y)
        print(f"  {'sum_k':18s} ρ={rho_k:.4f}  τ={tau_k:.4f}")

        best_alpha = candidates[best_key]
        write_alpha_csv(
            os.path.join(out_dir, f"{task}_alpha.csv"),
            best_alpha,
            method=best_key,
            s_mat=s_mat,
            kl=kl,
        )
        with open(os.path.join(out_dir, f"{task}_select.txt"), "w") as f:
            f.write(f"best_method={best_key}\n")
            f.write(f"best_spearman={best_rho:.6f}\n")
            f.write(f"dyn_range={dyn_range(best_alpha):.6f}\n")
            f.write(f"sum_k_spearman={rho_k:.6f}\n")
        with open(os.path.join(out_dir, f"{task}_gate.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "spearman_rho", "kendall_tau", "dyn_range"])
            for key, rho, tau, dr in rows:
                w.writerow([key, f"{rho:.6f}", f"{tau:.6f}", f"{dr:.6f}"])
            w.writerow(["sum_k", f"{rho_k:.6f}", f"{tau_k:.6f}", ""])
        print(f"  → selected {best_key}  ρ={best_rho:.4f}  dyn={dyn_range(best_alpha):.2f}")
        print(f"  saved {out_dir}/{task}_alpha.csv")

    # 汇总目录供 evolution --alpha-csv
    bundle = os.path.join(args.out_root, "for_evolution")
    for task in args.task:
        td = os.path.join(bundle, task)
        os.makedirs(td, exist_ok=True)
        src = os.path.join(args.out_root, task, f"{task}_alpha.csv")
        dst = os.path.join(td, f"{task}_alpha.csv")
        with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
            fout.write(fin.read())
    print(f"\nbundle for evolution: {bundle}")


if __name__ == "__main__":
    main()
