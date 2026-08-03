"""
THOR 多项式 exp 精度测试：只评 Horner 段（除 shift / 平方还原之前）。

流程（与 eval_softmax_cuda 对齐到 /scale 之前）：
  x_scaled = x / (d1 * d2 * THOR_INPUT_SCALE)
  p        = Horner(coeffs, x_scaled)   # 逼近 exp(x/(d1·d2))
  对照     = exp(x / (d1 · d2))

不含：/ exp(shift/(d1·d2))、以及随后的 d1 次平方。

多项式设计域通常是 x_scaled ∈ [-1, 0]，即
  x ∈ [-THOR_INPUT_SCALE · d1 · d2, 0]
默认 d1=d2=2 → x ∈ [-32, 0]。改下面配置区即可扫别的区间。

用法（在 nolinear/ 下）：
  python3 softmax.py
"""
from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from softmax_poly import (  # noqa: E402
    THOR_EXP_POLY_COEFFS,
    THOR_INPUT_SCALE,
    _check_power_of_two,
)

# ===================== 可调配置 =====================
D1 = 2.0
D2 = 2.0

# 原始输入 x 的扫描区间（attention score 量级，未除 d1/d2/8）
# 默认取多项式设计域：[-INPUT_SCALE·d1·d2, 0]
X_MIN = -30  # d1=d2=2 → -32
X_MAX = 30

N_POINTS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64
# ==================================================


def thor_exp_poly_before_shift(
    x: torch.Tensor,
    *,
    delta1: float,
    delta2: float,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    """Horner(x/(d1·d2·8))，停在除 shift / 平方之前。"""
    d1 = float(delta1)
    d2 = float(delta2)
    _check_power_of_two("delta1", d1)
    _check_power_of_two("delta2", d2)

    x_scaled = x / d1 / d2 / THOR_INPUT_SCALE
    y = torch.zeros_like(x_scaled)
    for coeff in coeffs:
        y = y * x_scaled + coeff
    return y


def true_exp_target(x: torch.Tensor, *, delta1: float, delta2: float) -> torch.Tensor:
    return torch.exp(x / float(delta1) / float(delta2))


def error_stats(approx: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    abs_err = (approx - target).abs()
    rel_err = abs_err / target.abs().clamp(min=1e-30)
    return {
        "max_abs": float(abs_err.max()),
        "mean_abs": float(abs_err.mean()),
        "max_rel": float(rel_err.max()),
        "mean_rel": float(rel_err.mean()),
        "rmse": float(torch.sqrt((abs_err**2).mean())),
    }


def main() -> None:
    if X_MAX < X_MIN:
        raise ValueError(f"X_MAX ({X_MAX}) < X_MIN ({X_MIN})")

    device = torch.device(DEVICE)
    coeffs = torch.tensor(THOR_EXP_POLY_COEFFS, device=device, dtype=DTYPE)

    x = torch.linspace(X_MIN, X_MAX, N_POINTS, device=device, dtype=DTYPE)
    x_scaled = x / D1 / D2 / THOR_INPUT_SCALE

    approx = thor_exp_poly_before_shift(x, delta1=D1, delta2=D2, coeffs=coeffs)
    target = true_exp_target(x, delta1=D1, delta2=D2)
    st = error_stats(approx, target)

    abs_err = (approx - target).abs()
    i_abs = int(abs_err.argmax())
    rel_err = abs_err / target.abs().clamp(min=1e-30)
    i_rel = int(rel_err.argmax())

    design_lo = -THOR_INPUT_SCALE * D1 * D2
    design_hi = 0.0

    print("=" * 72)
    print("THOR poly exp (Horner only, before /shift and squaring)")
    print(f"  d1={D1:g}  d2={D2:g}  INPUT_SCALE={THOR_INPUT_SCALE:g}")
    print(f"  x ∈ [{X_MIN:g}, {X_MAX:g}]  n={N_POINTS}  dtype={DTYPE}  device={device}")
    print(
        f"  x_scaled ∈ [{float(x_scaled.min()):g}, {float(x_scaled.max()):g}]"
        f"  （设计域约 [-1, 0] ↔ x ∈ [{design_lo:g}, {design_hi:g}]）"
    )
    print(f"  target = exp(x/(d1·d2))")
    print("-" * 72)
    print(
        f"  max|err|   = {st['max_abs']:.6e}  @ x={float(x[i_abs]):.6g}"
        f"  approx={float(approx[i_abs]):.6e}  true={float(target[i_abs]):.6e}"
    )
    print(f"  mean|err|  = {st['mean_abs']:.6e}")
    print(f"  rmse       = {st['rmse']:.6e}")
    print(f"  max|rel|   = {100.0 * st['max_rel']:.4f}%  @ x={float(x[i_rel]):.6g}")
    print(f"  mean|rel|  = {100.0 * st['mean_rel']:.4f}%")
    print("=" * 72)

    for name, idx in (
        ("lo", 0),
        ("mid", N_POINTS // 2),
        ("hi", N_POINTS - 1),
    ):
        print(
            f"  [{name}] x={float(x[idx]):10.4g}  x_scaled={float(x_scaled[idx]):8.4g}  "
            f"approx={float(approx[idx]):.6e}  true={float(target[idx]):.6e}  "
            f"|err|={float(abs_err[idx]):.3e}  rel={100.0 * float(rel_err[idx]):.4f}%"
        )


if __name__ == "__main__":
    main()
