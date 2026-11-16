"""
Pooler tanh 多项式：THOR 结构 p2(p1(z/C))，自拟合系数，幂基（普通基）。

  t = z / C ∈ [-1, 1]
  ŷ = p2(p1(t)) ≈ tanh(z) = tanh(C t)

拟合：与 GeLU 相同的 Chebyshev Remez，再 convert 成升幂 monomial，
以便沿用 THOR Stockmeyer（evaluate_polynomial_stockmeyer 要的就是普通基）。
p2 直接作用在 p1 的输出上（无 affine），与官方 he_tanh 一致。

用法（在 nolinear/ 下）：
  python3 eval_tanh.py                      # 默认拟合 C=5,3，并在真实 CLS 上测
  python3 eval_tanh.py --C 5 --task mrpc
  python3 eval_tanh.py --d1 15 --d2 15

输出：
  tanh_out/tanh_fit_C{C}_d{d1}_{d2}.json    系数（升幂 + THOR 风格降幂）
  tanh_out/{task}_tanh_poly_err.json        真实数据误差
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from numpy.polynomial.chebyshev import Chebyshev, chebval
from numpy.polynomial.polynomial import Polynomial, polyval

import gelu_chebyshev as gc
import pooler_tanh as pt

# ===================== 配置 =====================
TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
C_VALUES = [5.0, 3.0]  # 可调；真实 |z|≲2.2，C=5 把 t 放进 [-1,1]
D1 = 15
D2 = 15
GRID_N = 20001
REL_ERR_DENOM_MIN = 1e-4
JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tanh_out")
FIT_GRID_SIZE = 20000
REMEZ_MAX_ITER = 40
ZERO_EVEN = True  # tanh 为奇函数，清掉偶次幂（数值噪声）

# 官方 THOR 系数（降幂列表，与 tanh.py 一致；reverse 后升幂）
_THOR_P1_DESC = [
    -7.14529052e03,
    -7.76519925e01,
    2.74279201e04,
    2.45150249e02,
    -4.25793697e04,
    -3.01953016e02,
    3.42189880e04,
    1.82989351e02,
    -1.51158283e04,
    -5.64098990e01,
    3.58757327e03,
    8.17596753e00,
    -4.13341496e02,
    -4.29024545e-01,
    1.95056729e01,
    2.06201784e-03,
]
_THOR_P2_DESC = [
    -9.02573450e-03,
    -1.12320034e-04,
    1.08762008e-01,
    7.96793166e-04,
    -5.41327356e-01,
    -1.42873183e-03,
    1.46476749e00,
    -2.22416152e-03,
    -2.43259032e00,
    1.17381072e-02,
    2.74974898e00,
    -1.77631073e-02,
    -2.38934873e00,
    1.30194294e-02,
    2.02874846e00,
    -4.08442578e-03,
]
THOR_P1 = np.asarray(list(reversed(_THOR_P1_DESC)), dtype=np.float64)
THOR_P2 = np.asarray(list(reversed(_THOR_P2_DESC)), dtype=np.float64)
THOR_C = 40.0
# ======================================================================


def _as_increasing(c: np.ndarray, degree: int) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64).ravel()
    if c.size < degree + 1:
        c = np.pad(c, (0, degree + 1 - c.size))
    return c[: degree + 1]


def _zero_even_powers(c: np.ndarray) -> tuple[np.ndarray, float]:
    out = np.asarray(c, dtype=np.float64).copy()
    even = out[0::2].copy()
    max_even = float(np.max(np.abs(even))) if even.size else 0.0
    out[0::2] = 0.0
    return out, max_even


def cheb_to_monomial(cheb_c: np.ndarray, domain: tuple[float, float]) -> np.ndarray:
    """Chebyshev 系数 → 自变量在 domain 上的升幂普通基。"""
    ch = Chebyshev(np.asarray(cheb_c, dtype=np.float64), domain=domain)
    return np.asarray(ch.convert(kind=Polynomial).coef, dtype=np.float64)


def eval_monomial_pair(
    z: np.ndarray, scale: float, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    t = np.asarray(z, dtype=np.float64) / float(scale)
    u = polyval(t, p1)
    return polyval(u, p2)


def error_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    diff = yhat - y
    abs_e = np.abs(diff)
    denom = np.maximum(np.abs(y), REL_ERR_DENOM_MIN)
    rel = abs_e / denom
    corr = float(np.corrcoef(y, yhat)[0, 1]) if y.size > 1 else float("nan")
    return {
        "n": int(y.size),
        "mae": float(abs_e.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(abs_e.max()),
        "mean_rel": float(rel.mean()),
        "max_rel": float(rel.max()),
        "corr": corr,
        "yhat_min": float(yhat.min()),
        "yhat_max": float(yhat.max()),
    }


def _print_metrics(label: str, m: dict[str, float]) -> None:
    print(
        f"  {label:<28} mae={m['mae']:.4e}  rmse={m['rmse']:.4e}  "
        f"max_abs={m['max_abs']:.4e}  mean_rel={m['mean_rel']:.4e}  "
        f"corr={m['corr']:.6f}",
        flush=True,
    )


def fit_composite_power(
    C: float,
    d1: int = D1,
    d2: int = D2,
    *,
    zero_even: bool = ZERO_EVEN,
) -> dict[str, Any]:
    """
    1) Remez p1 ≈ tanh(C t)，t∈[-1,1]
    2) Remez p2 ≈ tanh(C t) 作为 p1(t) 的函数（复合 p2∘p1）
    3) 两段都转普通基；p2 的自变量是 p1 输出本身（与 THOR 相同）
    """
    a, b = -1.0, 1.0

    def target(t: np.ndarray) -> np.ndarray:
        return np.tanh(float(C) * np.asarray(t, dtype=np.float64))

    c1_cheb, err1 = gc.remez_minimax_cheb(
        target, d1, a, b, max_iter=REMEZ_MAX_ITER, grid_size=FIT_GRID_SIZE
    )
    t_s = np.linspace(a, b, FIT_GRID_SIZE, dtype=np.float64)
    u_s = chebval(t_s, c1_cheb)
    y_s = target(t_s)
    order = np.argsort(u_s)
    u_sorted = u_s[order]
    y_sorted = y_s[order]
    u_unique, idx = np.unique(u_sorted, return_index=True)
    if u_unique.size < d2 + 2:
        raise RuntimeError(f"C={C}: p1 值域点数不足以拟合 p2")
    y_unique = y_sorted[idx]
    u_lo, u_hi = float(u_unique[0]), float(u_unique[-1])
    if u_hi - u_lo < 1e-12:
        raise RuntimeError(f"C={C}: p1 值域退化")

    def g_of_u(u: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(u, dtype=np.float64), u_unique, y_unique)

    c2_cheb, err2 = gc.remez_minimax_cheb(
        g_of_u, d2, u_lo, u_hi, max_iter=REMEZ_MAX_ITER, grid_size=FIT_GRID_SIZE
    )

    p1 = _as_increasing(cheb_to_monomial(c1_cheb, (a, b)), d1)
    p2 = _as_increasing(cheb_to_monomial(c2_cheb, (u_lo, u_hi)), d2)
    max_even_p1 = max_even_p2 = 0.0
    if zero_even:
        p1, max_even_p1 = _zero_even_powers(p1)
        p2, max_even_p2 = _zero_even_powers(p2)

    yhat = eval_monomial_pair(t_s * C, C, p1, p2)
    grid_vs_target = error_metrics(y_s, yhat)
    p1_raw = _as_increasing(cheb_to_monomial(c1_cheb, (a, b)), d1)
    p2_raw = _as_increasing(cheb_to_monomial(c2_cheb, (u_lo, u_hi)), d2)
    y_cheb = gc.cheb_eval_on_domain(c2_cheb, u_s, (u_lo, u_hi), method="numpy")
    y_mono_raw = eval_monomial_pair(t_s * C, C, p1_raw, p2_raw)
    convert_mismatch = float(np.max(np.abs(y_mono_raw - y_cheb)))

    payload = {
        "C": float(C),
        "d1": int(d1),
        "d2": int(d2),
        "kind": "composite_p2_of_p1",
        "basis": "monomial_increasing",
        "p1_increasing": p1.tolist(),
        "p2_increasing": p2.tolist(),
        "p1_descending": p1[::-1].tolist(),
        "p2_descending": p2[::-1].tolist(),
        "p1_cheb": np.asarray(c1_cheb, dtype=np.float64).tolist(),
        "p2_cheb": np.asarray(c2_cheb, dtype=np.float64).tolist(),
        "p1_domain": [a, b],
        "p2_domain": [u_lo, u_hi],
        "remez_p1_max_err": float(err1),
        "remez_p2_max_err": float(err2),
        "zero_even": bool(zero_even),
        "max_even_p1_before_zero": max_even_p1,
        "max_even_p2_before_zero": max_even_p2,
        "convert_mismatch_max_abs": convert_mismatch,
        "grid_vs_tanh_Ct": grid_vs_target,
        "note": (
            "ŷ=p2(p1(z/C))，p1/p2 为升幂普通基；"
            "p1_descending 与 THOR tanh.py 列表方向相同（最高次在前）"
        ),
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    return payload


def save_fit(payload: dict[str, Any], json_dir: str) -> str:
    os.makedirs(json_dir, exist_ok=True)
    c = payload["C"]
    path = os.path.join(
        json_dir, f"tanh_fit_C{c:g}_d{payload['d1']}_{payload['d2']}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def collect_z(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None,
    calib_seed: int,
    calib_indices_dir: str,
) -> tuple[np.ndarray, dict[str, str]]:
    splits = ["calib", "validation"] if eval_split == "all" else [eval_split]
    parts: list[torch.Tensor] = []
    descs: dict[str, str] = {}
    model = None
    tokenizer = None
    for sp in splits:
        _stats, stacked, desc, model, tokenizer = pt.collect_tanh_input_for_split(
            task_name,
            device,
            eval_split=sp,
            num_samples=num_samples if sp == "validation" else None,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
            model=model,
            tokenizer=tokenizer,
        )
        parts.append(stacked)
        descs[sp] = desc
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    z = torch.cat(parts, dim=0).numpy().astype(np.float64, copy=False)
    return z, descs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="自拟合 THOR 式 tanh（普通基）并测真实误差")
    p.add_argument("--task", nargs="*", default=None)
    p.add_argument(
        "--eval-split",
        choices=("all", "validation", "calib"),
        default=pt.DEFAULT_EVAL_SPLIT,
    )
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=pt.BATCH_SIZE)
    p.add_argument("--calib-seed", type=int, default=pt.CALIB_SEED)
    p.add_argument("--calib-indices-dir", default=pt.CALIB_INDICES_DIR)
    p.add_argument("--json-dir", default=JSON_DIR)
    p.add_argument(
        "--C",
        default=",".join(str(c) for c in C_VALUES),
        help="逗号分隔的 C，默认 5,3",
    )
    p.add_argument("--d1", type=int, default=D1)
    p.add_argument("--d2", type=int, default=D2)
    p.add_argument("--keep-even", action="store_true", help="保留偶次幂")
    p.add_argument("--skip-thor", action="store_true", help="不测官方 C=40 系数")
    p.add_argument("--skip-real", action="store_true", help="只拟合+网格，不跑模型")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pt.BATCH_SIZE = int(args.batch_size)
    tasks = list(args.task) if args.task else list(TASK_NAMES)
    c_list = [float(s) for s in args.C.split(",") if s.strip()]
    zero_even = not args.keep_even
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(
        f"tanh fit p2(p1(z/C))  monomial  C={c_list}  d1={args.d1} d2={args.d2}  "
        f"zero_even={zero_even}  device={device}",
        flush=True,
    )

    fits: dict[str, dict[str, Any]] = {}
    t_grid = np.linspace(-1.0, 1.0, GRID_N, dtype=np.float64)
    print("\n拟合 + 网格 t∈[-1,1]，ŷ vs tanh(C t)", flush=True)
    for C in c_list:
        fit = fit_composite_power(
            C, args.d1, args.d2, zero_even=zero_even
        )
        path = save_fit(fit, args.json_dir)
        fits[str(C)] = fit
        print(
            f"  C={C:g}  p2_domain=[{fit['p2_domain'][0]:.4g},{fit['p2_domain'][1]:.4g}]  "
            f"convert_mis={fit['convert_mismatch_max_abs']:.2e}  "
            f"even_p1={fit['max_even_p1_before_zero']:.2e}  → {path}",
            flush=True,
        )
        _print_metrics("grid tanh(C t)", fit["grid_vs_tanh_Ct"])
        yhat = eval_monomial_pair(
            t_grid * C,
            C,
            np.asarray(fit["p1_increasing"]),
            np.asarray(fit["p2_increasing"]),
        )
        _print_metrics(
            "grid re-eval",
            error_metrics(np.tanh(C * t_grid), yhat),
        )

    if not args.skip_thor:
        y_thor = eval_monomial_pair(t_grid * THOR_C, THOR_C, THOR_P1, THOR_P2)
        print(f"\nTHOR 官方系数 C={THOR_C:g} 网格 vs tanh({THOR_C:g} t)", flush=True)
        _print_metrics("thor grid", error_metrics(np.tanh(THOR_C * t_grid), y_thor))

    if args.skip_real:
        return

    rows: list[tuple[str, dict[str, Any]]] = []
    os.makedirs(args.json_dir, exist_ok=True)
    for task in tasks:
        print(f"\n{'=' * 60}\n任务 {task}\n{'=' * 60}", flush=True)
        z, descs = collect_z(
            task,
            device,
            eval_split=args.eval_split,
            num_samples=args.samples,
            calib_seed=args.calib_seed,
            calib_indices_dir=args.calib_indices_dir,
        )
        y = np.tanh(z)
        print(
            f"  z∈[{z.min():.4g},{z.max():.4g}]  n={z.shape[0]}×{z.shape[1]}",
            flush=True,
        )
        by_method: dict[str, Any] = {}
        for C in c_list:
            fit = fits[str(C)]
            yhat = eval_monomial_pair(
                z,
                C,
                np.asarray(fit["p1_increasing"]),
                np.asarray(fit["p2_increasing"]),
            )
            m = error_metrics(y, yhat)
            m["t_abs_max"] = float(np.abs(z / C).max())
            by_method[f"fit_C{C:g}"] = m
            _print_metrics(f"fit C={C:g}", m)
        if not args.skip_thor:
            yhat = eval_monomial_pair(z, THOR_C, THOR_P1, THOR_P2)
            m = error_metrics(y, yhat)
            m["t_abs_max"] = float(np.abs(z / THOR_C).max())
            by_method["thor_C40"] = m
            _print_metrics("thor C=40", m)

        payload = {
            "task": task,
            "eval_split": "union" if args.eval_split == "all" else args.eval_split,
            "sources": list(descs.keys()),
            "split_desc": descs,
            "method": "p2(p1(z/C)) monomial; fit=Remez→power; thor=official coeffs",
            "rel_err_denom_min": REL_ERR_DENOM_MIN,
            "z_min": float(z.min()),
            "z_max": float(z.max()),
            "n_samples": int(z.shape[0]),
            "hidden_size": int(z.shape[1]),
            "by_method": by_method,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        path = os.path.join(args.json_dir, f"{task}_tanh_poly_err.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"💾 {path}", flush=True)
        rows.append((task, by_method))

    methods = [f"fit_C{C:g}" for C in c_list]
    if not args.skip_thor:
        methods.append("thor_C40")
    print("\n汇总 vs 精确 tanh(z)  (mae / max_abs)", flush=True)
    hdr = f"{'task':<8}" + "".join(f" {m:>22}" for m in methods)
    print(hdr, flush=True)
    for task, bm in rows:
        line = f"{task:<8}"
        for m in methods:
            e = bm[m]
            line += f" {e['mae']:.3e}/{e['max_abs']:.2e}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
