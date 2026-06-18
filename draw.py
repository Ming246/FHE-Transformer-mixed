"""绘制 Group B GeLU 三档多项式近似与原始 GELU 在 [-6, 6] 上的对比图。"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from gelu_poly import GELU_LEVEL_KEYS, GeluPolyEvaluator, _GROUP_B_LAYERS
from nolinear.gelu_chebyshev import gelu_exact

# Group B 各层方案相同，取代表层索引
REPRESENTATIVE_LAYER = min(_GROUP_B_LAYERS)
LEVELS = (0, 1, 2)
X_MIN, X_MAX = -10.0, 10.0
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nolinear",
    "gelu",
    "gelu_group_b_comparison.pdf",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GeLU low/mid/high vs exact GELU on [-6, 6]"
    )
    parser.add_argument("--x-min", type=float, default=X_MIN)
    parser.add_argument("--x-max", type=float, default=X_MAX)
    parser.add_argument("--n", type=int, default=2000, help="采样点数")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    x_np = np.linspace(args.x_min, args.x_max, args.n)
    y_exact = gelu_exact(x_np)

    x_t = torch.tensor(x_np, dtype=torch.float64)
    evaluators = {
        GELU_LEVEL_KEYS[level]: GeluPolyEvaluator(REPRESENTATIVE_LAYER, level)
        for level in LEVELS
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_curve = axes[0]
    ax_curve.plot(x_np, y_exact, "k-", linewidth=2.0, label="GELU (exact)")

    colors = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#2ca02c"}
    for level in LEVELS:
        name = GELU_LEVEL_KEYS[level]
        y_poly = evaluators[name](x_t).detach().cpu().numpy()
        ax_curve.plot(
            x_np,
            y_poly,
            color=colors[name],
            linewidth=1.5,
            linestyle="--",
            label=f"{name}",
        )

    ax_curve.set_xlim(args.x_min, args.x_max)
    ax_curve.set_xlabel("$x$")
    ax_curve.set_ylabel("$\\mathrm{GELU}(x)$")
    ax_curve.set_title(f"GeLU approximations vs exact  ($x \\in [{args.x_min}, {args.x_max}]$)")
    ax_curve.legend(loc="upper left")
    ax_curve.grid(True, alpha=0.3)

    ax_err = axes[1]
    for level in LEVELS:
        name = GELU_LEVEL_KEYS[level]
        y_poly = evaluators[name](x_t).detach().cpu().numpy()
        ax_err.plot(
            x_np,
            y_poly - y_exact,
            color=colors[name],
            linewidth=1.2,
            label=f"{name} error",
        )

    ax_err.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax_err.set_xlim(args.x_min, args.x_max)
    ax_err.set_xlabel("$x$")
    ax_err.set_ylabel("approx $-$ exact")
    ax_err.set_title("Absolute approximation error")
    ax_err.legend(loc="best")
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
