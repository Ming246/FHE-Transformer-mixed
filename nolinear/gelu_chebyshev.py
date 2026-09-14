"""
GELU minimax（Remez）— Chebyshev 基版本

与 gelu_minmax.py 相同拟合流程，但：
  - Remez 结果保留 Chebyshev 系数（不转幂基）
  - 求值：Clenshaw 递推 + 平衡二叉树构造 T_k(z) 后线性组合
  - 误差测试：Clenshaw / PS-tree / numpy.chebval 互验 + GELU 区间评估

拟合：y(t) ≈ poly(t)，GELU(x) = x * (0.5 + y(t))，t = x/C ∈ [-1, 1]
复合：y ≈ f1(t) + f2(f1(t))
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander, Chebyshev
from scipy.signal import find_peaks

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

# COMPOSITE_CONFIGS = [
#     (15, 15), (15, 27), (27, 15),
#     (31, 27), (27, 31), (31, 31),(15, 31),(31, 15)
# ]
COMPOSITE_CONFIGS = [
    (15, 15), (15, 27), (27, 15),(23,15),(15,23),
    (31, 27), (27, 31), (31, 31),(15, 31),(31, 15),(7,15),(15,7)
]
# COMPOSITE_CONFIGS = [
#     (15, 15),(11,15),(15,11),(15,23),(23,15),(23,23),
#     (15, 7), (7, 15),(15, 31),(31, 15),(13,15),(15,13),(13,13),(11,11)
# ]
SINGLE_DEGREES = [ 31, 39, 47, 55, 63]
DEFAULT_C_VALUE = 80.0
T_DOMAIN = (-1.0, 1.0)

DEFAULT_TRUNCATION_THRESHOLD = 1e-6
REL_ERR_DENOM_MIN = 1e-4
EVAL_MATCH_ATOL = 1e-10
EVAL_MATCH_RTOL = 1e-10

DEFAULT_EXTRA_EVAL_X_INTERVAL = (1-DEFAULT_C_VALUE, -3.5)

GELU_OUTPUT_SUBDIR = "gelu"
COEFF_BASIS_DOC = (
    "Chebyshev 基系数 (T_0..T_n)，自变量 z=(2x-a-b)/(b-a)∈[-1,1]。"
    "单段/复合均拟合 y(t)；GELU(x)=x*(0.5+y)。"
    "求值：Clenshaw 或平衡二叉树 T_k 组合。"
)


# ---------------------------------------------------------------------------
# 目标函数
# ---------------------------------------------------------------------------

def y_exact_from_t(t: np.ndarray, C: float) -> np.ndarray:
    x = C * np.asarray(t, dtype=np.float64)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * np.tanh(inner)


def gelu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def gelu_from_y(x: np.ndarray, y_t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (0.5 + y_t)


def eval_x_bounds(C: float) -> tuple[float, float]:
    return 1.0 - C, C - 1.0


def eval_x_grid_interval(x_lo: float, x_hi: float, n: int = 20000) -> np.ndarray:
    return np.linspace(x_lo, x_hi, n)


def eval_x_grid(C: float, n: int = 20000) -> np.ndarray:
    lo, hi = eval_x_bounds(C)
    return eval_x_grid_interval(lo, hi, n)


def eval_t_grid(C: float, n: int = 20000) -> np.ndarray:
    return eval_x_grid(C, n) / C


def parse_x_interval(s: str) -> tuple[float, float]:
    parts = s.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"区间格式应为 lo,hi，收到: {s!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if lo >= hi:
        raise ValueError(f"区间左端须小于右端: [{lo}, {hi}]")
    return lo, hi


def x_to_cheb_z(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return (2.0 * x - a - b) / (b - a)


def log2_ceil(n: int) -> int:
    if n <= 1:
        return 0
    return int(math.ceil(math.log2(n)))


# ---------------------------------------------------------------------------
# 同态加密乘法深度（CKKS/BFV：仅 ct×ct 计 1 层；pt 系数 / 加减 / 仿射映射不计）
# ---------------------------------------------------------------------------
#
# 推荐求值：平衡二叉树构造 T_k(z)，再 sum c_k·T_k(z)（c_k 明文）。
#   构造 T_degree 的 ct×ct 并行乘法深度 ≈ ceil(log2(degree))。
# 不推荐：顺序 Clenshaw，每步含 ct×ct 的 z·b，串行深度 = degree。
#
# 完整 GeLU：y 求值后还有 ct×ct 的 x·(0.5+y)，+1 层。
# 复合：f1(t) 与 f2(u) 串行（u 为 f1 密文输出），深度相加。

GELU_HE_RECONSTRUCT_DEPTH = 1
# Liberate: ∑ c_k T_k uses mult_scalar(rescale=False) then one rescale.
GELU_HE_CHEB_COMBO_RESCALE = 1


def cheb_he_depth_ps_tree(degree: int) -> int:
    """Chebyshev 级数（degree 次）在 HE 下用平衡 T_k 树的 ct×ct 乘法深度。"""
    return log2_ceil(degree)


def cheb_he_depth_clenshaw(degree: int) -> int:
    """顺序 Clenshaw 的 ct×ct 乘法深度（仅作对比，HE 部署勿用）。"""
    return max(0, degree)


def _domain_is_unit_depth(domain: tuple[float, float], *, tol: float = 1e-12) -> bool:
    a, b = float(domain[0]), float(domain[1])
    return abs(a + 1.0) <= tol and abs(b - 1.0) <= tol


def gelu_he_depth_breakdown(
    degree_or_d1: int,
    d2: int | None = None,
    *,
    eval_method: str = "ps_tree",
    f2_domain: tuple[float, float] | None = None,
):
    """
    返回 HE 乘法深度明细（total 为 ILP/cost 应用值）。

    PS-tree（生产）：每段 ⌈log₂ deg⌉（T_k 树）+ 1（系数组合 rescale）；
    f2 定义域非 [-1,1] 时再 +1（affine_to_z）；最后 reconstruct +1。

    single: total = depth(y) + combo + reconstruct
    composite: f1 + [f2_affine] + f2 + 2·combo + reconstruct
    """
    if eval_method not in ("ps_tree", "clenshaw"):
        raise ValueError(f"eval_method 应为 ps_tree|clenshaw，收到 {eval_method!r}")
    cheb_depth = (
        cheb_he_depth_ps_tree
        if eval_method == "ps_tree"
        else cheb_he_depth_clenshaw
    )
    combo = GELU_HE_CHEB_COMBO_RESCALE if eval_method == "ps_tree" else 0
    if d2 is None:
        y_dep = cheb_depth(degree_or_d1)
        return {
            "y_eval": y_dep,
            "cheb_combo_rescale": combo,
            "gelu_reconstruct": GELU_HE_RECONSTRUCT_DEPTH,
            "total": y_dep + combo + GELU_HE_RECONSTRUCT_DEPTH,
            "eval_method": eval_method,
        }
    f1_dep = cheb_depth(degree_or_d1)
    f2_dep = cheb_depth(d2)
    f2_aff = 0
    if f2_domain is not None and not _domain_is_unit_depth(f2_domain):
        f2_aff = 1
    return {
        "f1_eval": f1_dep,
        "f2_affine": f2_aff,
        "f2_eval": f2_dep,
        "cheb_combo_rescale": 2 * combo,
        "gelu_reconstruct": GELU_HE_RECONSTRUCT_DEPTH,
        "total": (
            f1_dep
            + f2_aff
            + f2_dep
            + 2 * combo
            + GELU_HE_RECONSTRUCT_DEPTH
        ),
        "eval_method": eval_method,
    }


def paterson_stockmeyer_depth(degree: int) -> int:
    """幂基 Paterson–Stockmeyer 深度（旧 gelu_minmax 占位，与 Cheb PS-tree 同阶）。"""
    return log2_ceil(degree)


def truncate_cheb_coeffs(
    cheb_c: np.ndarray, threshold: float = DEFAULT_TRUNCATION_THRESHOLD
) -> np.ndarray:
    out = np.asarray(cheb_c, dtype=np.float64).copy()
    out[np.abs(out) < threshold] = 0.0
    return out


# ---------------------------------------------------------------------------
# Chebyshev 求值：Clenshaw + 平衡二叉树 T_k
# ---------------------------------------------------------------------------

def clenshaw_cheb(cheb_c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Clenshaw 递推求 sum_k c_k T_k(z)，z∈[-1,1]。
    cheb_c: T_0..T_n 系数；z 可广播。
    """
    c = np.asarray(cheb_c, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if c.size == 0:
        return np.zeros_like(z, dtype=np.float64)
    if c.size == 1:
        return np.full(np.shape(z), c[0], dtype=np.float64)

    b_kp2 = np.zeros_like(z, dtype=np.float64)
    b_kp1 = np.zeros_like(z, dtype=np.float64)
    for k in range(c.size - 1, 0, -1):
        b_k = c[k] + 2.0 * z * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k
    return c[0] + z * b_kp1 - b_kp2


def _build_Tk_balanced_tree(z: np.ndarray, max_k: int) -> list[np.ndarray]:
    """
    平衡二叉树（分治 index）构造 T_0..T_max_k。
    递推：T_{a+b} = 2 T_a T_b - T_{|a-b|}，T_1=z，T_0=1。
    """
    z = np.asarray(z, dtype=np.float64)
    T: list[np.ndarray | None] = [None] * (max_k + 1)

    def build(k: int) -> np.ndarray:
        if T[k] is not None:
            return T[k]
        if k == 0:
            T[k] = np.ones_like(z, dtype=np.float64)
        elif k == 1:
            T[k] = z.copy()
        else:
            a = k // 2
            b = k - a
            Ta = build(a)
            Tb = build(b)
            Tab = build(abs(a - b))
            T[k] = 2.0 * Ta * Tb - Tab
        return T[k]

    for k in range(max_k + 1):
        build(k)
    return [T[k] for k in range(max_k + 1)]


def cheb_eval_ps_tree(cheb_c: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    平衡二叉树构造 T_k(z)，再线性组合系数。
    与 Clenshaw 数学等价，乘法深度约 O(log n) 层级（硬件模型用）。
    """
    c = np.asarray(cheb_c, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros_like(z, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    Tk = _build_Tk_balanced_tree(z, c.size - 1)
    out = np.zeros_like(z, dtype=np.float64)
    for k, ck in enumerate(c):
        if ck != 0.0:
            out = out + ck * Tk[k]
    return out


def cheb_eval_on_domain(
    cheb_c: np.ndarray,
    x: np.ndarray,
    domain: tuple[float, float],
    method: str = "clenshaw",
) -> np.ndarray:
    """在物理变量 x∈[a,b] 上求 Chebyshev 多项式。"""
    a, b = domain
    z = x_to_cheb_z(np.asarray(x, dtype=np.float64), a, b)
    if method == "clenshaw":
        return clenshaw_cheb(cheb_c, z)
    if method == "ps_tree":
        return cheb_eval_ps_tree(cheb_c, z)
    if method == "numpy":
        return chebval(z, cheb_c)
    raise ValueError(f"未知求值方法: {method}")


def max_eval_mismatch(
    cheb_c: np.ndarray,
    x: np.ndarray,
    domain: tuple[float, float],
) -> dict[str, float]:
    """Clenshaw / PS-tree / numpy.chebval 三者最大偏差。"""
    a, b = domain
    z = x_to_cheb_z(np.asarray(x, dtype=np.float64), a, b)
    v_cl = clenshaw_cheb(cheb_c, z)
    v_ps = cheb_eval_ps_tree(cheb_c, z)
    v_np = chebval(z, cheb_c)
    return {
        "clenshaw_vs_numpy": float(np.max(np.abs(v_cl - v_np))),
        "ps_tree_vs_numpy": float(np.max(np.abs(v_ps - v_np))),
        "clenshaw_vs_ps_tree": float(np.max(np.abs(v_cl - v_ps))),
    }


# ---------------------------------------------------------------------------
# Remez（返回 Chebyshev 系数）
# ---------------------------------------------------------------------------

def _remez_reference_points(
    signed_err: np.ndarray, grid: np.ndarray, degree: int
) -> np.ndarray:
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
        idx_new = idx_new[-n:]
    return grid[idx_new]


def remez_minimax_cheb(
    f: Callable[[np.ndarray], np.ndarray],
    degree: int,
    a: float,
    b: float,
    max_iter: int = 40,
    grid_size: int = 20000,
    grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """在 [a,b] 上 minimax；返回 (Chebyshev 系数 T_0..T_n, 最大误差)。"""
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

    z_grid = x_to_cheb_z(grid, a, b)
    f_grid = f(grid)

    n_init = max(8 * m + 8, 512)
    if use_custom_grid and a > 0 and b > 0:
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
        z_ref = x_to_cheb_z(x_ref, a, b)
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

    signed_err = f_grid - chebval(z_grid, best_cheb_c)
    max_err = float(np.max(np.abs(signed_err)))
    return best_cheb_c, max_err


# ---------------------------------------------------------------------------
# 拟合与误差
# ---------------------------------------------------------------------------

@dataclass
class GeluErrorMetrics:
    max_abs: float
    mean_abs: float
    rms: float
    max_rel_pct: float
    mean_rel_pct: float


@dataclass
class EvalSelfTest:
    clenshaw_vs_numpy: float
    ps_tree_vs_numpy: float
    clenshaw_vs_ps_tree: float


@dataclass
class FitResult:
    name: str
    kind: str
    C: float
    depth: int  # HE 乘法深度（PS-tree + GELU 还原 +1）
    depth_legacy_ps: int  # 旧占位：仅 ceil(log2 d) 之和，不含 GELU +1
    depth_he_clenshaw: int  # 若误用顺序 Clenshaw 的深度（对比用）
    depth_he_breakdown: dict[str, int]
    y_max_err: float
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
    eval_self_test: EvalSelfTest
    max_cheb_coeff_abs: float
    params: dict


def _make_y_hat_single(
    cheb_c: np.ndarray, domain: tuple[float, float], method: str = "clenshaw"
):
    def y_hat(t):
        return cheb_eval_on_domain(cheb_c, t, domain, method=method)

    return y_hat


def _make_y_hat_composite(
    c1: np.ndarray,
    c2: np.ndarray,
    domain_t: tuple[float, float],
    domain_u: tuple[float, float],
    method: str = "clenshaw",
):
    def y_hat(t):
        u = cheb_eval_on_domain(c1, t, domain_t, method=method)
        return u + cheb_eval_on_domain(c2, u, domain_u, method=method)

    return y_hat


def _compute_gelu_error_metrics(
    gelu_p: np.ndarray, gelu_t: np.ndarray
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
    )


def _eval_gelu_error(
    C: float,
    y_fn: Callable[[np.ndarray], np.ndarray],
    x_interval: tuple[float, float] | None = None,
) -> GeluErrorMetrics:
    if x_interval is None:
        x_lo, x_hi = eval_x_bounds(C)
    else:
        x_lo, x_hi = x_interval
    x = eval_x_grid_interval(x_lo, x_hi)
    t = x / C
    return _compute_gelu_error_metrics(gelu_from_y(x, y_fn(t)), gelu_exact(x))


def fit_single_minimax(
    C: float,
    degree: int,
    truncation_threshold: float = DEFAULT_TRUNCATION_THRESHOLD,
    extra_eval_x: tuple[float, float] = DEFAULT_EXTRA_EVAL_X_INTERVAL,
) -> FitResult:
    a, b = T_DOMAIN
    domain = (a, b)

    def target(t):
        return y_exact_from_t(t, C)

    cheb_c, _ = remez_minimax_cheb(target, degree, a, b)
    cheb_c = truncate_cheb_coeffs(cheb_c, truncation_threshold)

    y_hat_cl = _make_y_hat_single(cheb_c, domain, "clenshaw")
    y_hat_ps = _make_y_hat_single(cheb_c, domain, "ps_tree")

    t_test = eval_t_grid(C, 10000)
    y_max = float(np.max(np.abs(target(t_test) - y_hat_cl(t_test))))
    gerr = _eval_gelu_error(C, y_hat_cl)
    gerr_extra = _eval_gelu_error(C, y_hat_cl, x_interval=extra_eval_x)

    mismatch = max_eval_mismatch(cheb_c, t_test, domain)
    gerr_ps = _eval_gelu_error(C, y_hat_ps)
    ps_gelu_diff = abs(gerr_ps.max_abs - gerr.max_abs)

    x_extra_lo, x_extra_hi = extra_eval_x
    he = gelu_he_depth_breakdown(degree, eval_method="ps_tree")
    he_cl = gelu_he_depth_breakdown(degree, eval_method="clenshaw")
    legacy = paterson_stockmeyer_depth(degree)
    return FitResult(
        name=f"single_d{degree}",
        kind="single",
        C=C,
        depth=he["total"],
        depth_legacy_ps=legacy,
        depth_he_clenshaw=he_cl["total"],
        depth_he_breakdown=he,
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
        eval_self_test=EvalSelfTest(
            clenshaw_vs_numpy=mismatch["clenshaw_vs_numpy"],
            ps_tree_vs_numpy=mismatch["ps_tree_vs_numpy"],
            clenshaw_vs_ps_tree=mismatch["clenshaw_vs_ps_tree"],
        ),
        max_cheb_coeff_abs=float(np.max(np.abs(cheb_c))),
        params={
            "f_cheb_coeffs": cheb_c.tolist(),
            "f_domain": list(domain),
            "degree": degree,
            "approx": "y_single",
            "truncation_threshold": truncation_threshold,
            "gelu_clenshaw_vs_ps_tree_max": ps_gelu_diff,
            "depth_he_breakdown": he,
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
    domain_t = (a, b)

    def target(t):
        return y_exact_from_t(t, C)

    try:
        c1, _ = remez_minimax_cheb(target, d1, a, b)
    except np.linalg.LinAlgError:
        return None

    c1 = truncate_cheb_coeffs(c1, truncation_threshold)
    y_hat_f1 = _make_y_hat_single(c1, domain_t, "clenshaw")

    t_s = np.linspace(a, b, 4000)
    u_s = y_hat_f1(t_s)
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

    domain_u = (u_lo, u_hi)
    try:
        c2, _ = remez_minimax_cheb(residual_u, d2, u_lo, u_hi)
    except np.linalg.LinAlgError:
        return None

    c2 = truncate_cheb_coeffs(c2, truncation_threshold)

    y_hat_cl = _make_y_hat_composite(c1, c2, domain_t, domain_u, "clenshaw")
    y_hat_ps = _make_y_hat_composite(c1, c2, domain_t, domain_u, "ps_tree")

    t_test = eval_t_grid(C, 10000)
    y_max = float(np.max(np.abs(target(t_test) - y_hat_cl(t_test))))
    gerr = _eval_gelu_error(C, y_hat_cl)
    gerr_extra = _eval_gelu_error(C, y_hat_cl, x_interval=extra_eval_x)

    mismatch_f1 = max_eval_mismatch(c1, t_test, domain_t)
    u_test = y_hat_f1(t_test)
    mismatch_f2 = max_eval_mismatch(c2, u_test, domain_u)
    gerr_ps = _eval_gelu_error(C, y_hat_ps)

    x_extra_lo, x_extra_hi = extra_eval_x
    he = gelu_he_depth_breakdown(
        d1, d2, eval_method="ps_tree", f2_domain=domain_u
    )
    he_cl = gelu_he_depth_breakdown(d1, d2, eval_method="clenshaw")
    legacy = paterson_stockmeyer_depth(d1) + paterson_stockmeyer_depth(d2)
    max_cheb = max(float(np.max(np.abs(c1))), float(np.max(np.abs(c2))))

    return FitResult(
        name=f"composite_{d1}_{d2}",
        kind="composite",
        C=C,
        depth=he["total"],
        depth_legacy_ps=legacy,
        depth_he_clenshaw=he_cl["total"],
        depth_he_breakdown=he,
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
        eval_self_test=EvalSelfTest(
            clenshaw_vs_numpy=max(
                mismatch_f1["clenshaw_vs_numpy"], mismatch_f2["clenshaw_vs_numpy"]
            ),
            ps_tree_vs_numpy=max(
                mismatch_f1["ps_tree_vs_numpy"], mismatch_f2["ps_tree_vs_numpy"]
            ),
            clenshaw_vs_ps_tree=max(
                mismatch_f1["clenshaw_vs_ps_tree"],
                mismatch_f2["clenshaw_vs_ps_tree"],
                float(np.max(np.abs(y_hat_cl(t_test) - y_hat_ps(t_test)))),
            ),
        ),
        max_cheb_coeff_abs=max_cheb,
        params={
            "f1_cheb_coeffs": c1.tolist(),
            "f1_domain": list(domain_t),
            "f2_cheb_coeffs": c2.tolist(),
            "f2_domain": list(domain_u),
            "d1": d1,
            "d2": d2,
            "truncation_threshold": truncation_threshold,
            "eval_mismatch_f1": mismatch_f1,
            "eval_mismatch_f2": mismatch_f2,
            "gelu_clenshaw_vs_ps_tree_max": abs(gerr_ps.max_abs - gerr.max_abs),
            "depth_he_breakdown": he,
        },
    )


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _nolinear_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def format_C_tag(C: float) -> str:
    if float(C).is_integer():
        return f"C{int(C)}"
    s = f"{C:g}".replace(".", "p").replace("-", "m")
    return f"C{s}"


def gelu_output_dir() -> str:
    out = os.path.join(_nolinear_dir(), GELU_OUTPUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


def resolve_output_path(
    C: float, filename: str, user_path: str | None = None
) -> str:
    if user_path:
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
        f"GELU 评估① x ∈ [{x_lo:.4g}, {x_hi:.4g}]；"
        f"评估② x ∈ [{ex_lo:g}, {ex_hi:g}]。"
    )


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


def save_report(results: list[FitResult], path: str) -> None:
    ranked = sorted(results, key=lambda r: r.gelu_max_err)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GELU Chebyshev minimax 方案对比\n")
        f.write("# 系数：Chebyshev 基；求值：Clenshaw + 平衡二叉树 T_k\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"排名 {i}: {r.name} ({r.kind})\n")
            f.write(f"  C = {r.C}\n")
            f.write(f"  HE 乘法深度(PS-tree) = {r.depth}\n")
            f.write(f"  HE 乘法深度(Clenshaw,勿用) = {r.depth_he_clenshaw}\n")
            bd = r.depth_he_breakdown
            f.write(
                f"  深度明细: {', '.join(f'{k}={v}' for k, v in bd.items() if k != 'eval_method')}\n"
            )
            f.write(f"  max|Cheb coeff| = {r.max_cheb_coeff_abs:.6e}\n")
            f.write(f"  y(t) 最大误差 = {r.y_max_err:.6e}\n")
            st = r.eval_self_test
            f.write(f"  求值自检 Clenshaw↔numpy max = {st.clenshaw_vs_numpy:.6e}\n")
            f.write(f"  求值自检 PS-tree↔numpy max = {st.ps_tree_vs_numpy:.6e}\n")
            f.write(f"  求值自检 Clenshaw↔PS-tree max = {st.clenshaw_vs_ps_tree:.6e}\n")
            for line in _format_all_gelu_metric_lines(r):
                f.write(line + "\n")
            f.write("-" * 60 + "\n")


def save_coefficients(
    results: list[FitResult],
    path: str,
    C: float,
    truncation_threshold: float,
    extra_eval_x: tuple[float, float],
) -> None:
    ranked = sorted(results, key=lambda r: r.gelu_max_err)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GELU Chebyshev minimax 系数\n")
        f.write(f"# C = {C}\n")
        f.write(f"# {COEFF_BASIS_DOC}\n")
        f.write(f"# truncation_threshold = {truncation_threshold:.6e}\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"方案 {i}: {r.name} ({r.kind})\n")
            f.write(f"  max|Cheb coeff| = {r.max_cheb_coeff_abs:.12e}\n")
            f.write(f"  y(t) 最大误差 = {r.y_max_err:.12e}\n")
            for line in _format_all_gelu_metric_lines(r):
                f.write(line + "\n")
            if r.kind == "single":
                f.write("  f Chebyshev 系数 (T_0..T_n):\n")
                for j, c in enumerate(r.params["f_cheb_coeffs"]):
                    f.write(f"    c[{j}] = {c:.15e}\n")
                lo, hi = r.params["f_domain"]
                f.write(f"  f 定义域 x∈[{lo}, {hi}]\n")
            else:
                f.write("  f1 Chebyshev 系数 (T_0..T_n):\n")
                for j, c in enumerate(r.params["f1_cheb_coeffs"]):
                    f.write(f"    c1[{j}] = {c:.15e}\n")
                lo, hi = r.params["f1_domain"]
                f.write(f"  f1 定义域 t∈[{lo}, {hi}]\n")
                f.write("  f2 Chebyshev 系数 (T_0..T_n):\n")
                for j, c in enumerate(r.params["f2_cheb_coeffs"]):
                    f.write(f"    c2[{j}] = {c:.15e}\n")
                ulo, uhi = r.params["f2_domain"]
                f.write(f"  f2 定义域 u∈[{ulo:.15e}, {uhi:.15e}]\n")
            f.write("-" * 80 + "\n")

    json_path = os.path.splitext(path)[0] + ".json"
    payload = {
        "C": C,
        "eval_x_primary": list(eval_x_bounds(C)),
        "eval_x_extra": list(extra_eval_x),
        "truncation_threshold": truncation_threshold,
        "coeff_basis": "chebyshev",
        "coeff_order": "T0_to_Tn",
        "eval_methods": ["clenshaw", "ps_tree"],
        "note": COEFF_BASIS_DOC,
        "schemes": [
            {
                "rank": i,
                "name": r.name,
                "kind": r.kind,
                "C": r.C,
                "depth": r.depth,
                "depth_legacy_ps": r.depth_legacy_ps,
                "depth_he_clenshaw": r.depth_he_clenshaw,
                "depth_he_breakdown": r.depth_he_breakdown,
                "max_cheb_coeff_abs": r.max_cheb_coeff_abs,
                "y_max_err": r.y_max_err,
                "gelu_max_err": r.gelu_max_err,
                "eval_self_test": {
                    "clenshaw_vs_numpy": r.eval_self_test.clenshaw_vs_numpy,
                    "ps_tree_vs_numpy": r.eval_self_test.ps_tree_vs_numpy,
                    "clenshaw_vs_ps_tree": r.eval_self_test.clenshaw_vs_ps_tree,
                },
                "params": r.params,
            }
            for i, r in enumerate(ranked, 1)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2, ensure_ascii=False)


def run_eval_unit_tests(C: float) -> None:
    """全局求值器单元测试。"""
    print("\n[求值自检] Clenshaw / PS-tree / numpy.chebval ...")
    rng = np.random.default_rng(0)
    z = rng.uniform(-1.0, 1.0, 500)
    for deg in (7, 15, 31, 47):
        c = rng.standard_normal(deg + 1)
        mm = max_eval_mismatch(c, z, (-1.0, 1.0))
        ok = (
            mm["clenshaw_vs_numpy"] < EVAL_MATCH_ATOL
            and mm["ps_tree_vs_numpy"] < EVAL_MATCH_ATOL
            and mm["clenshaw_vs_ps_tree"] < EVAL_MATCH_ATOL
        )
        status = "OK" if ok else "FAIL"
        print(
            f"  degree={deg:2d}  Cl↔np={mm['clenshaw_vs_numpy']:.2e}  "
            f"PS↔np={mm['ps_tree_vs_numpy']:.2e}  "
            f"Cl↔PS={mm['clenshaw_vs_ps_tree']:.2e}  [{status}]"
        )
        if not ok:
            raise AssertionError(f"求值自检失败 degree={deg}")


def assert_scheme_eval_tests(results: list[FitResult]) -> None:
    for r in results:
        st = r.eval_self_test
        if st.clenshaw_vs_ps_tree > EVAL_MATCH_ATOL:
            raise AssertionError(
                f"{r.name}: Clenshaw vs PS-tree 偏差 {st.clenshaw_vs_ps_tree:.2e}"
            )


def main():
    parser = argparse.ArgumentParser(description="GELU Chebyshev minimax 拟合与误差测试")
    parser.add_argument("--C", type=float, default=DEFAULT_C_VALUE)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--coeffs", type=str, default=None)
    parser.add_argument(
        "--truncation-threshold", type=float, default=DEFAULT_TRUNCATION_THRESHOLD
    )
    parser.add_argument(
        "--eval-extra-x",
        type=str,
        default=f"{DEFAULT_EXTRA_EVAL_X_INTERVAL[0]},{DEFAULT_EXTRA_EVAL_X_INTERVAL[1]}",
    )
    parser.add_argument(
        "--skip-unit-test", action="store_true", help="跳过求值器单元测试"
    )
    args = parser.parse_args()

    C = args.C
    trunc = args.truncation_threshold
    extra_eval_x = parse_x_interval(args.eval_extra_x)

    report_path = resolve_output_path(C, "gelu_chebyshev_report.txt", args.report)
    coeffs_path = resolve_output_path(C, "gelu_chebyshev_coeffs.txt", args.coeffs)

    print("=" * 70)
    print("GELU Chebyshev minimax — Clenshaw + 平衡二叉树 T_k")
    print("=" * 70)
    print_C_note(C, extra_eval_x)
    print(f"截断阈值 = {trunc:.6e}")

    if not args.skip_unit_test:
        run_eval_unit_tests(C)

    results: list[FitResult] = []

    print("\n[1/2] 单段 Chebyshev minimax ...")
    for deg in SINGLE_DEGREES:
        print(f"  degree={deg} ...", end=" ", flush=True)
        r = fit_single_minimax(C, deg, trunc, extra_eval_x=extra_eval_x)
        results.append(r)
        print(
            f"GELU max={r.gelu_max_err:.4e}  "
            f"|c|max={r.max_cheb_coeff_abs:.4e}  "
            f"Cl↔PS={r.eval_self_test.clenshaw_vs_ps_tree:.2e}  "
            f"depth={r.depth}"
        )

    print("\n[2/2] 复合 Chebyshev minimax ...")
    for d1, d2 in COMPOSITE_CONFIGS:
        print(f"  ({d1},{d2}) ...", end=" ", flush=True)
        r = fit_composite_minimax(C, d1, d2, trunc, extra_eval_x=extra_eval_x)
        if r is None:
            print("失败")
            continue
        results.append(r)
        print(
            f"GELU max={r.gelu_max_err:.4e}  "
            f"|c|max={r.max_cheb_coeff_abs:.4e}  "
            f"Cl↔PS={r.eval_self_test.clenshaw_vs_ps_tree:.2e}  "
            f"depth={r.depth}"
        )

    if not results:
        print("无有效结果")
        return

    assert_scheme_eval_tests(results)

    ranked = sorted(results, key=lambda r: r.gelu_max_err)
    print("\n" + "=" * 70)
    ex_tag = f"[{extra_eval_x[0]:g},{extra_eval_x[1]:g}]"
    print(
        f"{'排名':<4} {'方案':<22} {'HE深度':<6} "
        f"{'max@C':<11} {'mean@C':<11} "
        f"{'max'+ex_tag:<11} {'mean'+ex_tag:<11} {'Cl↔PS':<10}"
    )
    print("-" * 105)
    for i, r in enumerate(ranked, 1):
        print(
            f"{i:<4} {r.name:<22} {r.depth:<6} "
            f"{r.gelu_max_err:<11.4e} {r.gelu_mean_err:<11.4e} "
            f"{r.gelu_extra_max_err:<11.4e} {r.gelu_extra_mean_err:<11.4e} "
            f"{r.eval_self_test.clenshaw_vs_ps_tree:<10.2e}"
        )

    save_report(results, report_path)
    save_coefficients(results, coeffs_path, C, trunc, extra_eval_x)
    print(f"\n报告：{report_path}")
    print(f"系数：{coeffs_path}")
    print(f"JSON：{os.path.splitext(coeffs_path)[0] + '.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
