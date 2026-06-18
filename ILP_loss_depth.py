"""
在 cost 可分解（各位置 depth 累加）假设下，用整数线性规划分配 POLY_SCHEMES。

  min   Σ_j S_j,k
  s.t.  Σ_j cost_j,k ≤ C
        每个位置恰好选一个档位 k ∈ {0,1,2}（均为多项式近似，不含 original）

S 来自 sensitive_scores/{task}_sensitivity.csv（宽表）；
cost 沿用 cost.py 的 per-position depth。

TODO: cost 模型更新为非可分解形式后，需换求解器/建模方式。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from cost import (
    NUM_LAYERS,
    compute_scheme_cost,
    gelu_poly_depth,
    scheme_index,
    softmax_poly_depth,
    validate_scheme,
)
from gelu_poly import gelu_level_allowed

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]
# TASK_NAMES = ["rte"]
SENSITIVE_OUTPUT_DIR = "./results/sensitive_scores_1/"
POLY_LEVELS = (0, 1, 2)  # low, mid, high；24 个位置必须均为多项式近似

# 各任务 cost 预算（depth 累加上限）；None 表示不约束
COST_BUDGET_BY_TASK: dict[str, int | None] = {
    "mrpc": 520,
    "rte": 540,
    "sst2": 502,
}
# ILP 求解后是否在验证集上跑 poly_model_inference 对比推理
RUN_INFERENCE_AFTER_ILP = True
# ==================================================


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in ("softmax", "gelu")
    ]


def load_sensitivity_matrix(task_name: str) -> dict[tuple[int, str], dict[int, float]]:
    """读取宽表 CSV → (layer_idx, kind) → {level: S}。"""
    path = os.path.join(SENSITIVE_OUTPUT_DIR, f"{task_name}_sensitivity.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"敏感度文件不存在：{path}")

    out: dict[tuple[int, str], dict[int, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer_idx = int(row["layer_idx"])
            kind = row["kind"]
            out[(layer_idx, kind)] = {
                0: float(row["S_low"]),
                1: float(row["S_mid"]),
                2: float(row["S_high"]),
            }
    if len(out) != NUM_LAYERS * 2:
        raise ValueError(f"{task_name} 敏感度行数应为 24，当前为 {len(out)}")
    return out


def build_position_cost_matrix(task_name: str) -> dict[tuple[int, str], dict[int, int]]:
    """(layer_idx, kind) → {level: depth cost}，与 cost.py 一致。"""
    costs: dict[tuple[int, str], dict[int, int]] = {}
    for layer_idx, kind in iter_positions():
        level_cost: dict[int, int] = {}
        for level in POLY_LEVELS:
            if kind == "gelu" and not gelu_level_allowed(layer_idx, level):
                continue
            if kind == "softmax":
                level_cost[level] = softmax_poly_depth(task_name, layer_idx, level)
            else:
                level_cost[level] = gelu_poly_depth(layer_idx, level)
        costs[(layer_idx, kind)] = level_cost
    return costs


@dataclass
class ILPSolution:
    task_name: str
    scheme: list[int]
    total_s: float
    total_cost: int
    cost_budget: int | None
    status: str

    def format_scheme(self) -> str:
        return str(self.scheme)


def solve_scheme_ilp(
    task_name: str,
    cost_budget: int | None,
) -> ILPSolution:
    """
    整数线性规划求方案（每位置必选 k∈{0,1,2} 之一）。

    cost_budget=None 时不加 cost 约束（在纯多项式下通常趋向全 high）。
    """
    positions = iter_positions()
    n_pos = len(positions)
    levels = list(POLY_LEVELS)
    n_lev = len(levels)
    n_var = n_pos * n_lev

    s_mat = load_sensitivity_matrix(task_name)
    c_mat = build_position_cost_matrix(task_name)

    obj = np.zeros(n_var, dtype=np.float64)
    cost_row = np.zeros(n_var, dtype=np.float64)

    for j, (layer_idx, kind) in enumerate(positions):
        pos = (layer_idx, kind)
        for ki, level in enumerate(levels):
            idx = j * n_lev + ki
            if kind == "gelu" and not gelu_level_allowed(layer_idx, level):
                obj[idx] = 0.0
                cost_row[idx] = 0.0
                continue
            obj[idx] = s_mat[pos][level]
            cost_row[idx] = c_mat[pos][level]

    # 每个位置恰好选一个档位
    a_eq = np.zeros((n_pos, n_var), dtype=np.float64)
    for j in range(n_pos):
        for ki in range(n_lev):
            a_eq[j, j * n_lev + ki] = 1.0
    b_eq = np.ones(n_pos, dtype=np.float64)

    constraints = [LinearConstraint(a_eq, b_eq, b_eq)]
    if cost_budget is not None:
        a_ub = cost_row.reshape(1, -1)
        constraints.append(LinearConstraint(a_ub, -np.inf, np.array([cost_budget])))

    integrality = np.ones(n_var, dtype=np.int8)
    lb = np.zeros(n_var, dtype=np.float64)
    ub = np.ones(n_var, dtype=np.float64)
    for j, (layer_idx, kind) in enumerate(positions):
        for ki, level in enumerate(levels):
            idx = j * n_lev + ki
            if kind == "gelu" and not gelu_level_allowed(layer_idx, level):
                ub[idx] = 0.0
    bounds = Bounds(lb=lb, ub=ub)

    res = milp(
        c=obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        raise RuntimeError(f"ILP 未收敛：{task_name} status={res.message}")

    x = res.x
    if x is None:
        raise RuntimeError(f"ILP 无解：{task_name}")

    scheme = [0] * (NUM_LAYERS * 2)
    total_s = 0.0
    for j, (layer_idx, kind) in enumerate(positions):
        chosen = None
        for ki, level in enumerate(levels):
            if x[j * n_lev + ki] >= 0.5:
                chosen = level
                total_s += s_mat[(layer_idx, kind)][level]
                break
        if chosen is None:
            raise RuntimeError(f"位置 ({layer_idx}, {kind}) 未选中任何档位")
        scheme[scheme_index(layer_idx, kind)] = chosen

    validate_scheme(scheme, task_name)
    total_cost = int(compute_scheme_cost(task_name, scheme))

    return ILPSolution(
        task_name=task_name,
        scheme=scheme,
        total_s=total_s,
        total_cost=total_cost,
        cost_budget=cost_budget,
        status=str(res.message),
    )


def print_solution(sol: ILPSolution) -> None:
    budget_str = "∞" if sol.cost_budget is None else str(sol.cost_budget)
    print(f"\n{'=' * 60}")
    print(f"ILP 方案：{sol.task_name.upper()}  (cost 预算 ≤ {budget_str})")
    print(f"  状态：{sol.status}")
    print(f"  ΣS = {sol.total_s:.6e}")
    print(f"  cost(depth 累加) = {sol.total_cost}")
    print(f"  方案数组：")
    print(f"    {sol.format_scheme()}")
    print(f"{'=' * 60}")
    row_fmt = "  {layer:>5}  {kind:<8}  {level:<5}  {s:>12}  {depth:>5}"
    print(row_fmt.format(layer="layer", kind="kind", level="level", s="S", depth="depth"))
    print("  " + "-" * 48)
    s_mat = load_sensitivity_matrix(sol.task_name)
    c_mat = build_position_cost_matrix(sol.task_name)
    for layer_idx, kind in iter_positions():
        level = sol.scheme[scheme_index(layer_idx, kind)]
        level_name = {0: "low", 1: "mid", 2: "high"}[level]
        pos = (layer_idx, kind)
        print(
            row_fmt.format(
                layer=layer_idx,
                kind=kind,
                level=level_name,
                s=f"{s_mat[pos][level]:.4e}",
                depth=c_mat[pos][level],
            )
        )


def main() -> None:
    print("ILP 方案分配（可分解 depth cost，后续 cost 模型更新后需改建模）")
    inference_results = []

    for task_name in TASK_NAMES:
        budget = COST_BUDGET_BY_TASK.get(task_name)
        try:
            sol = solve_scheme_ilp(task_name, budget)
            print_solution(sol)
        except RuntimeError as e:
            print(f"\n{task_name.upper()} 求解失败：{e}")
            continue

        if not RUN_INFERENCE_AFTER_ILP:
            continue

        from poly_model_inference import (
            fmt_accuracy,
            fmt_accuracy_delta,
            run_task_with_comparison,
        )

        print(f"\n>>> 验证集推理测试：{task_name.upper()}（ILP 方案）")
        inference_results.append(run_task_with_comparison(task_name, sol.scheme))

    if RUN_INFERENCE_AFTER_ILP and inference_results:
        from poly_model_inference import fmt_accuracy, fmt_accuracy_delta

        print(f"\n{'=' * 60}")
        print("推理汇总（ILP 方案 vs 原始）")
        print(f"{'=' * 60}")
        print(
            f"{'task':<8} {'base_acc':<10} {'poly_acc':<10} "
            f"{'Δacc':<10} {'base_loss':<12} {'poly_loss':<12} {'Δloss':<10}"
        )
        print("-" * 78)
        for r in inference_results:
            print(
                f"{r['task']:<8} "
                f"{fmt_accuracy(r['baseline_accuracy']):<10} "
                f"{fmt_accuracy(r['poly_accuracy']):<10} "
                f"{fmt_accuracy_delta(r['accuracy_delta']):<10} "
                f"{r['baseline_loss']:.7f}{'':<4} "
                f"{r['poly_loss']:.7f}{'':<4} "
                f"{r['loss_delta']:+.7f}"
            )


if __name__ == "__main__":
    main()
