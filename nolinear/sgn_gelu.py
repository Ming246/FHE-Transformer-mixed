"""
分段多项式 GELU（sgn + nexus_gelu）

sgn(u)：在 u/C ∈ [-1, 1] 上用复合多项式近似符号函数（f、g 各嵌套两次）。
nexus_gelu(x)：以 ±sgn_bound 为分界，在区间内用多项式 p(x)，区间外用 x 拼接。

  a0 = sgn(x + sgn_bound)
  a1 = sgn(x - sgn_bound)
  y  = (a0 - a1) * p(x) + (a1 + 1) * x
  GELU(x) ≈ 0.5 * y
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import polynomial as nppoly

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

DEFAULT_C = 160.0
DEFAULT_SGN_BOUND = 3.5
DEFAULT_EXTRA_EVAL_X = (-3.5, 3.5)
REL_ERR_DENOM_MIN = 1e-4

# sgn 复合多项式系数（低次→高次，Horner / polyval 直接可用）
_F_COF = np.array([0, 315, 0, -420, 0, 378, 0, -180, 0, 35], dtype=np.float64) / 128.0
_G_COF = np.array(
    [0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623], dtype=np.float64
) / 1024.0

# 中间段多项式 p(x) 系数（0..12 次，低次→高次）
_P_COF = np.array(
    [
        0.000225775755,
        0.5,
        0.396880960,
        0.0,
        -0.0637042698,
        0.0,
        0.00838841647,
        0.0,
        -0.000717830961,
        0.0,
        0.0000349617829,
        0.0,
        -7.26059653e-7,
    ],
    dtype=np.float64,
)

GELU_OUTPUT_SUBDIR = "gelu"


def gelu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def evaluate_polynomial(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """标准幂基 Horner（coeffs：常数项→最高次）。"""
    return nppoly.polyval(np.asarray(x, dtype=np.float64), coeffs)


def sgn_approx(u: np.ndarray, C: float, *, strict: bool = False) -> np.ndarray:
    """
    复合多项式近似 sgn(u)，自变量 t = u/C 须在 [-1, 1]。
    strict=True 时若有点超出区间则抛 ValueError。
    """
    u = np.asarray(u, dtype=np.float64)
    t = u / C
    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_min < -1.0 - 1e-12 or t_max > 1.0 + 1e-12:
        msg = f"sgn 自变量 u/C 超出 [-1,1]：min={t_min:.6g}, max={t_max:.6g}（C={C}）"
        if strict:
            raise ValueError(msg)
        print(f"警告：{msg}")

    g1 = evaluate_polynomial(_G_COF, t)
    g2 = evaluate_polynomial(_G_COF, g1)
    f1 = evaluate_polynomial(_F_COF, g2)
    f2 = evaluate_polynomial(_F_COF, f1)
    return f2


def nexus_gelu(
    x: np.ndarray,
    C: float = DEFAULT_C,
    sgn_bound: float = DEFAULT_SGN_BOUND,
    *,
    strict_sgn: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    a0 = sgn_approx(x + sgn_bound, C, strict=strict_sgn)*0.5
    a1 = sgn_approx(x - sgn_bound, C, strict=strict_sgn)*0.5
    x_clamp = x*(a0 - a1)
    p = evaluate_polynomial(_P_COF, x_clamp)
    y = (a0 - a1) * p + (a1 + 0.5) * x
    return y


def eval_x_bounds(C: float) -> tuple[float, float]:
    """主评估区间 x ∈ [1-C, C-1]（与其它 GELU 脚本一致）。"""
    return 5 - C, C - 5


def eval_x_grid_interval(x_lo: float, x_hi: float, n: int = 20000) -> np.ndarray:
    return np.linspace(x_lo, x_hi, n)


def parse_x_interval(s: str) -> tuple[float, float]:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"区间格式应为 lo,hi，收到: {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise ValueError(f"区间左端须小于右端: [{lo}, {hi}]")
    return lo, hi


def required_C_for_x_range(x_lo: float, x_hi: float, sgn_bound: float) -> float:
    """保证 sgn(x±sgn_bound) 的 u/C 落在 [-1,1] 所需的最小 C。"""
    u_hi = max(abs(x_hi + sgn_bound), abs(x_hi - sgn_bound))
    u_lo = max(abs(x_lo + sgn_bound), abs(x_lo - sgn_bound))
    return max(u_hi, u_lo)


@dataclass
class GeluErrorMetrics:
    max_abs: float
    mean_abs: float
    rms: float
    max_rel_pct: float
    mean_rel_pct: float
    sgn_domain_violations: int


def compute_gelu_error_metrics(
    gelu_p: np.ndarray,
    gelu_t: np.ndarray,
    *,
    sgn_domain_violations: int = 0,
) -> GeluErrorMetrics:
    err = np.asarray(gelu_p, dtype=np.float64) - np.asarray(gelu_t, dtype=np.float64)
    abs_err = np.abs(err)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    rms = float(np.sqrt(np.mean(err**2)))

    denom_ok = np.abs(gelu_t) >= REL_ERR_DENOM_MIN
    if np.any(denom_ok):
        rel_pct = abs_err[denom_ok] / np.abs(gelu_t[denom_ok]) * 100.0
        max_rel_pct = float(np.max(rel_pct))
        mean_rel_pct = float(np.mean(rel_pct))
    else:
        max_rel_pct = 0.0
        mean_rel_pct = 0.0

    return GeluErrorMetrics(
        max_abs=max_abs,
        mean_abs=mean_abs,
        rms=rms,
        max_rel_pct=max_rel_pct,
        mean_rel_pct=mean_rel_pct,
        sgn_domain_violations=sgn_domain_violations,
    )


def count_sgn_domain_violations(
    x: np.ndarray, C: float, sgn_bound: float
) -> int:
    x = np.asarray(x, dtype=np.float64)
    viol = 0
    for u in (x + sgn_bound, x - sgn_bound):
        t = u / C
        viol += int(np.sum((t < -1.0) | (t > 1.0)))
    return viol


def eval_nexus_gelu_error(
    C: float,
    sgn_bound: float,
    x_interval: tuple[float, float],
    n: int = 20000,
) -> GeluErrorMetrics:
    x_lo, x_hi = x_interval
    x = eval_x_grid_interval(x_lo, x_hi, n)
    viol = count_sgn_domain_violations(x, C, sgn_bound)
    gelu_p = nexus_gelu(x, C, sgn_bound)
    gelu_t = gelu_exact(x)
    return compute_gelu_error_metrics(gelu_p, gelu_t, sgn_domain_violations=viol)


def _nolinear_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def resolve_report_path(C: float, user_path: str | None = None) -> str:
    out_dir = os.path.join(_nolinear_dir(), GELU_OUTPUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    name = user_path or f"sgn_gelu_report_C{int(C)}.txt"
    if not os.path.isabs(name):
        name = os.path.join(out_dir, os.path.basename(name))
    return name


def save_report(
    path: str,
    C: float,
    sgn_bound: float,
    primary: GeluErrorMetrics,
    extra: GeluErrorMetrics,
    x_primary: tuple[float, float],
    x_extra: tuple[float, float],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 分段 sgn + nexus_gelu 误差报告\n")
        f.write(f"# C = {C}, sgn_bound = {sgn_bound}\n")
        f.write(f"# 要求 |x±sgn_bound| <= C，否则 sgn 近似越界\n\n")
        for tag, m, interval in (
            ("主区间", primary, x_primary),
            ("额外区间", extra, x_extra),
        ):
            lo, hi = interval
            f.write(f"[{tag}] x ∈ [{lo:g}, {hi:g}]\n")
            f.write(f"  sgn 越界点数 = {m.sgn_domain_violations}\n")
            f.write(f"  最大绝对误差 = {m.max_abs:.6e}\n")
            f.write(f"  平均绝对误差 = {m.mean_abs:.6e}\n")
            f.write(f"  RMS 误差 = {m.rms:.6e}\n")
            f.write(
                f"  最大相对误差 = {m.max_rel_pct:.4f}% "
                f"(|GELU|>={REL_ERR_DENOM_MIN:.0e})\n"
            )
            f.write(
                f"  平均相对误差 = {m.mean_rel_pct:.4f}% "
                f"(|GELU|>={REL_ERR_DENOM_MIN:.0e})\n\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="分段 sgn GELU 误差测试")
    parser.add_argument("--C", type=float, default=DEFAULT_C, help="sgn 输入缩放 u/C")
    parser.add_argument(
        "--sgn-bound", type=float, default=DEFAULT_SGN_BOUND, help="分段边界（默认 3.5）"
    )
    parser.add_argument(
        "--eval-extra-x",
        type=str,
        default=f"{DEFAULT_EXTRA_EVAL_X[0]},{DEFAULT_EXTRA_EVAL_X[1]}",
        help="额外评估 x 区间 lo,hi（默认 -3.5,3.5）",
    )
    parser.add_argument("--grid", type=int, default=20000, help="评估网格点数")
    parser.add_argument("--report", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    C = args.C
    sgn_bound = args.sgn_bound
    extra_x = parse_x_interval(args.eval_extra_x)
    primary_x = eval_x_bounds(C)

    min_c = required_C_for_x_range(
        min(primary_x[0], extra_x[0]),
        max(primary_x[1], extra_x[1]),
        sgn_bound,
    )
    print("=" * 70)
    print("分段 sgn GELU — nexus_gelu 误差测试")
    print("=" * 70)
    print(f"C = {C}, sgn_bound = {sgn_bound}")
    print(f"主评估区间 x ∈ [{primary_x[0]:g}, {primary_x[1]:g}]")
    print(f"额外评估区间 x ∈ [{extra_x[0]:g}, {extra_x[1]:g}]")
    print(f"覆盖上述区间所需 C ≥ {min_c:.4g}（当前 C={'OK' if C >= min_c else '不足'}）")

    primary = eval_nexus_gelu_error(C, sgn_bound, primary_x, args.grid)
    extra = eval_nexus_gelu_error(C, sgn_bound, extra_x, args.grid)

    print(f"\n{'区间':<16} {'max_abs':<12} {'mean_abs':<12} {'sgn越界':<10}")
    print("-" * 52)
    for tag, m in (("主区间", primary), ("额外", extra)):
        print(
            f"{tag:<16} {m.max_abs:<12.4e} {m.mean_abs:<12.4e} "
            f"{m.sgn_domain_violations:<10d}"
        )

    report_path = resolve_report_path(C, args.report)
    save_report(report_path, C, sgn_bound, primary, extra, primary_x, extra_x)
    print(f"\n报告已保存：{report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
