"""
GELU minimax（Remez）多项式拟合与方案对比

拟合自变量 t = x/C ∈ [-1, 1]。

方案：
  1) 复合：内层 y(t) ≈ f1(t)+f2(f1(t))，GELU(x)=x*(0.5+y(t))
  2) 单段：内层 y(t) ≈ f(t)，GELU(x)=x*(0.5+y(t))，次数 15 / 31 / 63

GELU 误差评估（两段 x 区间，均可配置）：
  - 主区间：x ∈ [1-C, C-1]
  - 额外区间：默认 x ∈ [-3.5, 3.5]（--eval-extra-x）
"""
from __future__ import annotations

import json
import math
import os
import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import chebval, chebvander, Chebyshev
from scipy.signal import find_peaks

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

# 与 gelu.py 中搜索列表一致
COMPOSITE_CONFIGS = [
    (15, 15), (23,23), (27, 23), (23, 27),
    (27, 27), (31, 27), (27, 31), (31, 31),(15,7),(7,15),
]

SINGLE_DEGREES = [15, 31,39,47,55,63]
DEFAULT_C_VALUE = 80.0
T_DOMAIN = (-1.0, 1.0)

DEFAULT_TRUNCATION_THRESHOLD = 1e-6
REL_ERR_DENOM_MIN = 1e-4  # |GELU真值| 低于此值的点不参与相对误差统计

# 额外 GELU 评估区间（绝对 x，与 C 无关）；可通过 --eval-extra-x 修改
DEFAULT_EXTRA_EVAL_X_INTERVAL = (1-DEFAULT_C_VALUE, 3.5)


def eval_x_bounds(C: float) -> tuple[float, float]:
    """主评估区间：x ∈ [1-C, C-1]（排除拟合域端点各 1）。"""
    return 1.0 - C, C - 1.0


def eval_x_grid_interval(
    x_lo: float, x_hi: float, n: int = 20000
) -> np.ndarray:
    """指定 x 区间上的均匀网格。"""
    return np.linspace(x_lo, x_hi, n)


def eval_x_grid(C: float, n: int = 20000) -> np.ndarray:
    """x ∈ [1-C, C-1] 上的均匀网格。"""
    lo, hi = eval_x_bounds(C)
    return eval_x_grid_interval(lo, hi, n)


def eval_t_grid(C: float, n: int = 20000) -> np.ndarray:
    """与 eval_x_grid 对应的 t = x/C。"""
    return eval_x_grid(C, n) / C


def parse_x_interval(s: str) -> tuple[float, float]:
    """解析 'lo,hi' 为 (lo, hi)。"""
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"区间格式应为 lo,hi，收到: {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise ValueError(f"区间左端须小于右端: [{lo}, {hi}]")
    return lo, hi


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------

def y_exact_from_t(t: np.ndarray, C: float) -> np.ndarray:
    """y(t) = 0.5 * tanh(inner)，x = C*t。"""
    x = C * np.asarray(t, dtype=np.float64)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * np.tanh(inner)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_from_y(x: np.ndarray, y_t: np.ndarray) -> np.ndarray:
    """GELU(x) = x * (0.5 + y(x/C))。"""
    x = np.asarray(x, dtype=np.float64)
    return x * (0.5 + y_t)


def gelu_exact_from_t(t: np.ndarray, C: float) -> np.ndarray:
    """GELU(x) 在 t=x/C 处的精确值，x = C*t。"""
    return gelu_exact(C * np.asarray(t, dtype=np.float64))


# ---------------------------------------------------------------------------
# 多项式求值 / 深度
# ---------------------------------------------------------------------------

def poly_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """coeffs: 常数项 → 最高次。"""
    return Polynomial(coeffs)(x)


def paterson_stockmeyer_depth(degree: int) -> int:
    if degree <= 1:
        return 0
    return int(math.ceil(math.log2(degree)))


def truncate_coeffs(
    coeffs: np.ndarray, threshold: float = DEFAULT_TRUNCATION_THRESHOLD
) -> np.ndarray:
    """|系数| < threshold 的项置零；返回新数组（不修改入参）。"""
    out = np.asarray(coeffs, dtype=np.float64).copy()
    out[np.abs(out) < threshold] = 0.0
    return out


# ---------------------------------------------------------------------------
# Remez minimax（区间 [a,b] 上代数多项式）
# ---------------------------------------------------------------------------

def _remez_reference_points(
    signed_err: np.ndarray, grid: np.ndarray, degree: int
) -> np.ndarray:
    """
    在带符号误差曲线上选取 degree+2 个参考点（|e| 的局部极大，等波纹极值）。
    逻辑对齐经典 Remez / py-remezfit：对 |e| 做 find_peaks，必要时补端点。
    """
    n = degree + 2
    m = len(grid)
    if m < n:
        raise ValueError(f"grid_size={m} < degree+2={n}")

    min_peak_dist = max(1, m // (4 * n))
    idx_new: np.ndarray | None = None
    min_peak_width = max(2, m // n)
    while min_peak_width >= 1:
        peaks, _ = find_peaks(
            np.abs(signed_err),
            distance=min_peak_dist,
            width=min_peak_width,
        )
        peaks = peaks.astype(int)
        k = peaks.size
        if k >= n:
            idx_new = peaks
            break
        if (
            k == n - 2
            and peaks.size > 0
            and peaks[0] > min_peak_dist
            and peaks[-1] < m - 1 - min_peak_dist
        ):
            idx_new = np.hstack((0, peaks, m - 1))
            break
        if k == n - 1:
            if peaks.size > 0 and peaks[0] > min_peak_dist:
                idx_new = np.hstack((0, peaks))
            elif peaks.size > 0 and peaks[-1] < m - 1 - min_peak_dist:
                idx_new = np.hstack((peaks, m - 1))
            if idx_new is not None:
                break
        min_peak_width //= 2

    if idx_new is None or idx_new.size < n:
        idx_new = np.linspace(0, m - 1, n, dtype=int)
    elif idx_new.size > n:
        # 保留最右侧 n 个峰（与 py-remezfit 一致，区间右端极值更稳定）
        idx_new = idx_new[-n:]

    return grid[idx_new]


def _x_to_cheb_z(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """将 x∈[a,b] 映射到 Chebyshev 自变量 z∈[-1,1]。"""
    return (2.0 * x - a - b) / (b - a)


def _cheb_to_power_coeffs(cheb_c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Chebyshev 系数（z 域）→ 幂基系数（x 域，低次→高次）。"""
    ch = Chebyshev(cheb_c, domain=(a, b))
    return ch.convert(kind=Polynomial).coef


def remez_minimax(
    f: Callable[[np.ndarray], np.ndarray],
    degree: int,
    a: float,
    b: float,
    max_iter: int = 40,
    grid_size: int = 20000,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    在 [a,b] 上求 degree 次 minimax 近似多项式。
    返回 (coeffs 低次→高次, 估计最大误差)。

    实现要点：Chebyshev 基求解 + L2 初值 + |e| 峰选参考点；
    迭代中保留历史最优系数，仅在近似等波纹时交换参考点。
    grid 为 None 时在 [a,b] 上等距取 grid_size 个点；否则使用调用方提供的单调网格。
    正确性见 test_remez_minimax.py。
    """
    m = degree
    n_ref = m + 2

    use_custom_grid = grid is not None
    if grid is None:
        grid = np.linspace(a, b, grid_size)
    else:
        grid = np.asarray(grid, dtype=np.float64).ravel()
        if grid.size < n_ref:
            raise ValueError(f"grid 长度 {grid.size} < degree+2={n_ref}")
        if grid[0] < a - 1e-15 or grid[-1] > b + 1e-15:
            raise ValueError("grid 须落在 [a, b] 内")
    z_grid = _x_to_cheb_z(grid, a, b)
    f_grid = f(grid)

    # 初值：先 L2 Chebyshev 拟合，再取其误差极值作为参考点
    n_init = max(8 * m + 8, 512)
    if use_custom_grid and a > 0 and b > 0:
        # 非等距误差网格且区间为正时，L2 初值用几何间距
        x_init = np.geomspace(a, b, n_init)
    else:
        x_init = np.linspace(a, b, n_init)
    cheb_c = Chebyshev.fit(x_init, f(x_init), m, domain=(a, b)).coef.astype(np.float64)
    signed_err0 = f_grid - chebval(z_grid, cheb_c)
    x_ref = _remez_reference_points(signed_err0, grid, m)

    x_ref_prev: np.ndarray | None = None
    best_cheb_c = cheb_c.copy()
    best_err = float(np.max(np.abs(signed_err0)))

    for it in range(max_iter):
        f_ref = f(x_ref)
        z_ref = _x_to_cheb_z(x_ref, a, b)
        signs = (-1.0) ** np.arange(n_ref)
        A = np.hstack([chebvander(z_ref, m), -signs.reshape(-1, 1)])
        sol = np.linalg.solve(A, f_ref)
        cheb_c = sol[: m + 1]
        delta = float(sol[m + 1])

        signed_err = f_grid - chebval(z_grid, cheb_c)
        max_err = float(np.max(np.abs(signed_err)))
        if max_err < best_err:
            best_err = max_err
            best_cheb_c = cheb_c.copy()

        ripple_ratio = max_err / (abs(delta) + 1e-30)
        equioscillating = 0.25 <= ripple_ratio <= 4.0

        if x_ref_prev is not None and np.allclose(x_ref, x_ref_prev, rtol=0, atol=0):
            break
        if equioscillating and abs(max_err - abs(delta)) / (max_err + 1e-30) < 1e-4:
            break

        if it + 1 >= max_iter:
            break

        if not (equioscillating or it == 0):
            break
        x_ref_new = _remez_reference_points(signed_err, grid, m)
        if x_ref_prev is not None and np.allclose(x_ref_new, x_ref_prev, rtol=0, atol=0):
            break
        x_ref_prev = x_ref_new.copy()
        x_ref = x_ref_new

    coeffs = _cheb_to_power_coeffs(best_cheb_c, a, b)
    signed_err = f_grid - poly_eval(coeffs, grid)
    max_err = float(np.max(np.abs(signed_err)))
    return coeffs, max_err


# ---------------------------------------------------------------------------
# 拟合方案
# ---------------------------------------------------------------------------

@dataclass
class GeluErrorMetrics:
    max_abs: float
    mean_abs: float
    rms: float
    max_rel_pct: float
    mean_rel_pct: float


@dataclass
class FitResult:
    name: str
    kind: str  # "composite" | "single"
    C: float
    depth: int
    y_max_err: float  # composite / single: |y-y_hat| on t
    gelu_max_err: float
    gelu_mean_err: float
    gelu_rms_err: float
    gelu_max_rel_pct: float
    gelu_mean_rel_pct: float
    extra_eval_x_lo: float
    extra_eval_x_hi: float
    gelu_extra_max_err: float
    gelu_extra_mean_err: float
    gelu_extra_rms_err: float
    gelu_extra_max_rel_pct: float
    gelu_extra_mean_rel_pct: float
    params: dict


def fit_single_minimax(
    C: float,
    degree: int,
    truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
    extra_eval_x: tuple[float, float] = DEFAULT_EXTRA_EVAL_X_INTERVAL,
) -> FitResult:
    """单段 minimax：y(t) ≈ f(t)，GELU(x)=x*(0.5+y(t))；截断后再评估误差。"""
    a, b = T_DOMAIN

    def target(t):
        return y_exact_from_t(t, C)

    coeffs, _ = remez_minimax(target, degree, a, b)
    coeffs = truncate_coeffs(coeffs, truncation_threshold)

    def y_hat(t):
        return poly_eval(coeffs, t)

    t_test = eval_t_grid(C, 10000)
    y_max = float(np.max(np.abs(target(t_test) - y_hat(t_test))))
    gerr = _eval_gelu_error(C, y_hat)
    gerr_extra = _eval_gelu_error(C, y_hat, x_interval=extra_eval_x)
    x_extra_lo, x_extra_hi = extra_eval_x
    return FitResult(
        name=f"single_d{degree}",
        kind="single",
        C=C,
        depth=paterson_stockmeyer_depth(degree),
        y_max_err=y_max,
        gelu_max_err=gerr.max_abs,
        gelu_mean_err=gerr.mean_abs,
        gelu_rms_err=gerr.rms,
        gelu_max_rel_pct=gerr.max_rel_pct,
        gelu_mean_rel_pct=gerr.mean_rel_pct,
        extra_eval_x_lo=x_extra_lo,
        extra_eval_x_hi=x_extra_hi,
        gelu_extra_max_err=gerr_extra.max_abs,
        gelu_extra_mean_err=gerr_extra.mean_abs,
        gelu_extra_rms_err=gerr_extra.rms,
        gelu_extra_max_rel_pct=gerr_extra.max_rel_pct,
        gelu_extra_mean_rel_pct=gerr_extra.mean_rel_pct,
        params={
            "f_coeffs": coeffs.tolist(),
            "degree": degree,
            "approx": "y_single",
            "truncation_threshold": truncation_threshold,
        },
    )


def fit_composite_minimax(
    C: float,
    d1: int,
    d2: int,
    truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
    extra_eval_x: tuple[float, float] = DEFAULT_EXTRA_EVAL_X_INTERVAL,
) -> FitResult | None:
    a, b = T_DOMAIN

    def target(t):
        return y_exact_from_t(t, C)

    try:
        c1, _ = remez_minimax(target, d1, a, b)
    except np.linalg.LinAlgError:
        return None

    c1 = truncate_coeffs(c1, truncation_threshold)

    def f1(t):
        return poly_eval(c1, t)

    # f1 截断后再构造残差并搜索 f2
    t_s = np.linspace(a, b, 4000)
    u_s = f1(t_s)
    r_s = target(t_s) - u_s
    u_lo, u_hi = float(np.min(u_s)), float(np.max(u_s))
    if u_hi - u_lo < 1e-12:
        return None

    order = np.argsort(u_s)
    u_sorted = u_s[order]
    r_sorted = r_s[order]
    u_unique, idx_start = np.unique(u_sorted, return_index=True)
    if len(u_unique) < d2 + 2:
        return None
    r_unique = r_sorted[idx_start]

    def residual_u(u):
        return np.interp(u, u_unique, r_unique)

    try:
        c2, _ = remez_minimax(residual_u, d2, u_lo, u_hi)
    except np.linalg.LinAlgError:
        return None

    c2 = truncate_coeffs(c2, truncation_threshold)

    def y_hat(t):
        u = f1(t)
        return f1(t) + poly_eval(c2, u)

    y_test = eval_t_grid(C, 10000)
    y_max = float(np.max(np.abs(target(y_test) - y_hat(y_test))))
    gerr = _eval_gelu_error(C, y_hat)
    gerr_extra = _eval_gelu_error(C, y_hat, x_interval=extra_eval_x)
    x_extra_lo, x_extra_hi = extra_eval_x

    depth = paterson_stockmeyer_depth(d1) + paterson_stockmeyer_depth(d2)
    return FitResult(
        name=f"composite_{d1}_{d2}",
        kind="composite",
        C=C,
        depth=depth,
        y_max_err=y_max,
        gelu_max_err=gerr.max_abs,
        gelu_mean_err=gerr.mean_abs,
        gelu_rms_err=gerr.rms,
        gelu_max_rel_pct=gerr.max_rel_pct,
        gelu_mean_rel_pct=gerr.mean_rel_pct,
        extra_eval_x_lo=x_extra_lo,
        extra_eval_x_hi=x_extra_hi,
        gelu_extra_max_err=gerr_extra.max_abs,
        gelu_extra_mean_err=gerr_extra.mean_abs,
        gelu_extra_rms_err=gerr_extra.rms,
        gelu_extra_max_rel_pct=gerr_extra.max_rel_pct,
        gelu_extra_mean_rel_pct=gerr_extra.mean_rel_pct,
        params={
            "f1_coeffs": c1.tolist(),
            "f2_coeffs": c2.tolist(),
            "d1": d1,
            "d2": d2,
            "f2_domain": [u_lo, u_hi],
            "truncation_threshold": truncation_threshold,
        },
    )


def _compute_gelu_error_metrics(
    gelu_p: np.ndarray, gelu_t: np.ndarray
) -> GeluErrorMetrics:
    """绝对误差全点统计；相对误差仅 |gelu_t| >= REL_ERR_DENOM_MIN，结果为百分比。"""
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
    )


def _eval_gelu_direct_error(
    C: float,
    gelu_fn: Callable[[np.ndarray], np.ndarray],
    x_interval: tuple[float, float] | None = None,
) -> GeluErrorMetrics:
    """旧方案：GELU(x) ≈ gelu_fn(x/C)（直接近似 GELU，已弃用于单段拟合）。"""
    if x_interval is None:
        x_lo, x_hi = eval_x_bounds(C)
    else:
        x_lo, x_hi = x_interval
    x = eval_x_grid_interval(x_lo, x_hi)
    t = x / C
    return _compute_gelu_error_metrics(gelu_fn(t), gelu_exact(x))


def _eval_gelu_error(
    C: float,
    y_fn: Callable[[np.ndarray], np.ndarray],
    x_interval: tuple[float, float] | None = None,
) -> GeluErrorMetrics:
    """y(t) 经 GELU(x)=x*(0.5+y(t)) 还原后评估 GELU 误差（单段/复合共用）。"""
    if x_interval is None:
        x_lo, x_hi = eval_x_bounds(C)
    else:
        x_lo, x_hi = x_interval
    x = eval_x_grid_interval(x_lo, x_hi)
    t = x / C
    return _compute_gelu_error_metrics(gelu_from_y(x, y_fn(t)), gelu_exact(x))


def _format_gelu_metric_lines(
    m: GeluErrorMetrics, x_lo: float, x_hi: float
) -> list[str]:
    tag = f"x∈[{x_lo:g}, {x_hi:g}]"
    return [
        f"  GELU [{tag}] 最大绝对误差 = {m.max_abs:.6e}",
        f"  GELU [{tag}] 平均绝对误差 = {m.mean_abs:.6e}",
        f"  GELU [{tag}] RMS 误差 = {m.rms:.6e}",
        (
            f"  GELU [{tag}] 最大相对误差 = {m.max_rel_pct:.4f}% "
            f"(|GELU|>={REL_ERR_DENOM_MIN:.0e} 的点)"
        ),
        (
            f"  GELU [{tag}] 平均相对误差 = {m.mean_rel_pct:.4f}% "
            f"(|GELU|>={REL_ERR_DENOM_MIN:.0e} 的点)"
        ),
    ]


def _format_all_gelu_metric_lines(r: FitResult) -> list[str]:
    lo_c, hi_c = eval_x_bounds(r.C)
    lines = _format_gelu_metric_lines(
        GeluErrorMetrics(
            r.gelu_max_err,
            r.gelu_mean_err,
            r.gelu_rms_err,
            r.gelu_max_rel_pct,
            r.gelu_mean_rel_pct,
        ),
        lo_c,
        hi_c,
    )
    lines += _format_gelu_metric_lines(
        GeluErrorMetrics(
            r.gelu_extra_max_err,
            r.gelu_extra_mean_err,
            r.gelu_extra_rms_err,
            r.gelu_extra_max_rel_pct,
            r.gelu_extra_mean_rel_pct,
        ),
        r.extra_eval_x_lo,
        r.extra_eval_x_hi,
    )
    return lines


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

GELU_OUTPUT_SUBDIR = "gelu"


def _nolinear_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def format_C_tag(C: float) -> str:
    """C=64 -> C64；非整数用 C150p5 等形式。"""
    if float(C).is_integer():
        return f"C{int(C)}"
    s = f"{C:g}".replace(".", "p").replace("-", "m")
    return f"C{s}"


def gelu_output_dir() -> str:
    """nolinear/gelu/ 目录（不存在则创建）。"""
    out = os.path.join(_nolinear_dir(), GELU_OUTPUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


def resolve_gelu_output_path(
    C: float,
    filename: str,
    user_path: str | None = None,
) -> str:
    """
    生成输出文件绝对路径：位于 nolinear/gelu/，文件名含 C 标记。

    filename: 默认基名，如 gelu_minmax_report.txt
    user_path: 若指定，仅取其 basename 作为基名（仍放入 gelu/ 并加 C）
    """
    if user_path:
        if os.path.isabs(user_path):
            root, ext = os.path.splitext(user_path)
            if not ext:
                ext = ".txt"
            tag = format_C_tag(C)
            if not root.endswith(f"_{tag}"):
                return f"{root}_{tag}{ext}"
            return user_path
        filename = os.path.basename(user_path)

    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".txt"
    tag = format_C_tag(C)
    if not root.endswith(f"_{tag}"):
        root = f"{root}_{tag}"
    return os.path.join(gelu_output_dir(), root + ext)


def print_C_note(
    C: float, extra_eval_x: tuple[float, float] = DEFAULT_EXTRA_EVAL_X_INTERVAL
) -> None:
    x_lo, x_hi = eval_x_bounds(C)
    ex_lo, ex_hi = extra_eval_x
    print(
        f"\n参数 C = {C}：拟合 t = x/C ∈ [-1, 1]；"
        f"GELU 评估① x ∈ [{x_lo:.4g}, {x_hi:.4g}]（[1-C, C-1]）；"
        f"评估② x ∈ [{ex_lo:g}, {ex_hi:g}]。"
    )


# 保存到文件的系数均为标准幂基（非 Chebyshev）：低次→高次，Horner / numpy Polynomial 直接可用。
COEFF_BASIS_DOC = (
    "标准幂基多项式系数 (power basis)，从常数项到最高次项。"
    "自变量 t=x/C。单段/复合均先拟合 y(t)，GELU(x)=x*(0.5+y)；"
    "复合: y≈f1(t)+f2(f1(t))；单段: y≈f(t)。"
)


def save_coefficients(
    results: list[FitResult],
    path: str,
    C: float,
    truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
    extra_eval_x: tuple[float, float] = DEFAULT_EXTRA_EVAL_X_INTERVAL,
) -> None:
    """将全部方案的幂基系数写入 txt；同目录另存 JSON 便于程序读取。"""
    ranked = sorted(results, key=lambda r: r.gelu_max_err)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GELU minimax 多项式系数\n")
        f.write(f"# C = {C}\n")
        x_lo, x_hi = eval_x_bounds(C)
        f.write(f"# GELU 评估主区间 x = [{x_lo:g}, {x_hi:g}] ([1-C, C-1])\n")
        f.write(
            f"# GELU 评估额外区间 x = [{extra_eval_x[0]:g}, {extra_eval_x[1]:g}]\n"
        )
        f.write(f"# truncation_threshold = {truncation_threshold:.6e}\n")
        f.write(f"# {COEFF_BASIS_DOC}\n")
        f.write("# Remez 内部用 Chebyshev 基求解，落盘前已 convert(kind=Polynomial)。\n")
        f.write("# 落盘系数已截断；误差为截断后评估。\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"方案 {i}: {r.name} ({r.kind})\n")
            f.write(f"  C = {r.C}\n")
            f.write(
                f"  截断阈值 = {r.params.get('truncation_threshold', truncation_threshold):.6e}\n"
            )
            f.write(f"  系数基 = power (标准幂基)\n")
            f.write(f"  乘法深度(估计) = {r.depth}\n")
            f.write(f"  y(t) 最大误差 = {r.y_max_err:.12e}\n")
            for line in _format_all_gelu_metric_lines(r):
                f.write(line + "\n")
            if r.kind == "single":
                f.write("  近似形式: y(t) ≈ f(t), GELU(x) = x*(0.5+y(t)), t=x/C\n")
                f.write("  f(t) 标准幂基系数 (变量 t, 低次→高次):\n")
                for j, c in enumerate(r.params["f_coeffs"]):
                    f.write(f"    f[{j}] = {c:.15e}\n")
            else:
                f.write("  f1(t) 标准幂基系数 (变量 t, 低次→高次):\n")
                for j, c in enumerate(r.params["f1_coeffs"]):
                    f.write(f"    f1[{j}] = {c:.15e}\n")
                u_lo, u_hi = r.params["f2_domain"]
                f.write(f"  f2 定义域 u∈[{u_lo:.15e}, {u_hi:.15e}] (u=f1(t))\n")
                f.write("  f2(u) 标准幂基系数 (变量 u, 低次→高次):\n")
                for j, c in enumerate(r.params["f2_coeffs"]):
                    f.write(f"    f2[{j}] = {c:.15e}\n")
            f.write("-" * 80 + "\n")

    json_path = os.path.splitext(path)[0] + ".json"
    payload = {
        "C": C,
        "eval_x_primary": list(eval_x_bounds(C)),
        "eval_x_extra": list(extra_eval_x),
        "truncation_threshold": truncation_threshold,
        "coeff_basis": "power",
        "coeff_order": "low_to_high",
        "note": COEFF_BASIS_DOC,
        "schemes": [
            {
                "rank": i,
                "name": r.name,
                "kind": r.kind,
                "C": r.C,
                "depth": r.depth,
                "y_max_err": r.y_max_err,
                "gelu_max_err": r.gelu_max_err,
                "gelu_mean_err": r.gelu_mean_err,
                "gelu_rms_err": r.gelu_rms_err,
                "gelu_max_rel_pct": r.gelu_max_rel_pct,
                "gelu_mean_rel_pct": r.gelu_mean_rel_pct,
                "gelu_extra_max_err": r.gelu_extra_max_err,
                "gelu_extra_mean_err": r.gelu_extra_mean_err,
                "gelu_extra_rms_err": r.gelu_extra_rms_err,
                "gelu_extra_max_rel_pct": r.gelu_extra_max_rel_pct,
                "gelu_extra_mean_rel_pct": r.gelu_extra_mean_rel_pct,
                "rel_err_denom_min": REL_ERR_DENOM_MIN,
                "params": r.params,
            }
            for i, r in enumerate(ranked, 1)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2, ensure_ascii=False)


def save_report(results: list[FitResult], path: str) -> None:
    ranked = sorted(results, key=lambda r: r.gelu_max_err)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GELU minimax 方案对比（按 GELU 最大误差升序）\n")
        f.write("# 单段/复合均先拟合 y(t)，GELU=x*(0.5+y)；复合: y≈f1+f2(f1)；单段: y≈f(t)\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"排名 {i}: {r.name} ({r.kind})\n")
            f.write(f"  C = {r.C}\n")
            f.write(f"  乘法深度(估计) = {r.depth}\n")
            f.write(f"  y(t) 最大误差 = {r.y_max_err:.6e}\n")
            for line in _format_all_gelu_metric_lines(r):
                f.write(line + "\n")
            f.write("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GELU minimax (Remez) 方案搜索")
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C_VALUE,
        help="输入缩放：t=x/C∈[-1,1] 对应 x∈[-C,C]（默认 64）",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="报告基名（默认 gelu_minmax_report.txt，写入 nolinear/gelu/ 并加 _C<值>）",
    )
    parser.add_argument(
        "--coeffs",
        type=str,
        default=None,
        help="系数基名（默认 gelu_minmax_coeffs.txt，写入 nolinear/gelu/ 并加 _C<值>）",
    )
    parser.add_argument(
        "--truncation-threshold",
        type=float,
        default=DEFAULT_TRUNCATION_THRESHOLD,
        help="系数截断阈值：|c| < threshold 置零（默认 1e-6）",
    )
    parser.add_argument(
        "--eval-extra-x",
        type=str,
        default=f"{DEFAULT_EXTRA_EVAL_X_INTERVAL[0]},{DEFAULT_EXTRA_EVAL_X_INTERVAL[1]}",
        help="额外 GELU 评估 x 区间，格式 lo,hi（默认 -3.5,3.5）",
    )
    args = parser.parse_args()
    C = args.C
    trunc = args.truncation_threshold
    extra_eval_x = parse_x_interval(args.eval_extra_x)

    report_path = resolve_gelu_output_path(
        C, "gelu_minmax_report.txt", args.report
    )
    coeffs_path = resolve_gelu_output_path(
        C, "gelu_minmax_coeffs.txt", args.coeffs
    )

    print("=" * 70)
    print("GELU minimax 拟合 — Remez 极小极大")
    print("=" * 70)
    print_C_note(C, extra_eval_x)
    print(f"截断阈值 truncation_threshold = {trunc:.6e}")

    results: list[FitResult] = []

    print("\n[1/2] 单段 minimax：y(t) ≈ f(t), GELU(x)=x*(0.5+y)...")
    for deg in SINGLE_DEGREES:
        print(f"  拟合 degree={deg} ...", end=" ", flush=True)
        r = fit_single_minimax(C, deg, trunc, extra_eval_x=extra_eval_x)
        results.append(r)
        print(
            f"GELU max@[1-C,C-1]={r.gelu_max_err:.4e} "
            f"extra@[{extra_eval_x[0]:g},{extra_eval_x[1]:g}]={r.gelu_extra_max_err:.4e} "
            f"depth={r.depth}"
        )

    print("\n[2/2] 复合 minimax：y ≈ f1(t) + f2(f1(t))...")
    for d1, d2 in COMPOSITE_CONFIGS:
        print(f"  拟合 ({d1},{d2}) ...", end=" ", flush=True)
        r = fit_composite_minimax(C, d1, d2, trunc, extra_eval_x=extra_eval_x)
        if r is None:
            print("失败")
            continue
        results.append(r)
        print(
            f"GELU max@[1-C,C-1]={r.gelu_max_err:.4e} "
            f"extra={r.gelu_extra_max_err:.4e} depth={r.depth}"
        )

    if not results:
        print("无有效结果")
        return

    ranked = sorted(results, key=lambda r: r.gelu_max_err)

    print("\n" + "=" * 70)
    ex_tag = f"[{extra_eval_x[0]:g},{extra_eval_x[1]:g}]"
    print(
        f"{'排名':<4} {'方案':<22} {'深度':<5} "
        f"{'max@C':<11} {'max'+ex_tag:<11} {'mean@C':<11} {'mean'+ex_tag:<11}"
    )
    print("-" * 90)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<4} {r.name:<22} {r.depth:<5} "
            f"{r.gelu_max_err:<11.4e} {r.gelu_extra_max_err:<11.4e} "
            f"{r.gelu_mean_err:<11.4e} {r.gelu_extra_mean_err:<11.4e}"
        )

    save_report(results, report_path)
    save_coefficients(results, coeffs_path, C, trunc, extra_eval_x)
    print(f"\n报告已保存：{report_path}")
    print(f"系数已保存：{coeffs_path}")
    print(f"         JSON：{os.path.splitext(coeffs_path)[0] + '.json'}")
    print(f"  （{COEFF_BASIS_DOC}）")
    print("=" * 70)


if __name__ == "__main__":
    main()
