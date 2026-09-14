"""
POLY_SCHEMES 方案 cost。

方案数组：长度 48 = 12 层 × (softmax, ln1, gelu, ln2)，元素 0/1/2=多项式档位，3=原始函数（深度=0）。

- ``compute_scheme_cost``：乘法深度之和（占位 / 汇报列 ``total_depth``）
- ``compute_scheme_bts`` / ``compute_f_cost(..., cost_mode='bts')``：
  DualRail DP（``HE/thor/bts_ops.optimize_bootstrap``）给出的 **CT 加权 bts**。
  与 ``plan_mock`` + rem_guard 用的是同一求解器；budget 默认 14。

深度公式（GeLU Chebyshev / HE PS-tree）：
  取自 gelu_poly 方案配置 depth_he（含 GELU 还原 +1）
  Softmax:        SOFTMAX_DEPTH_BASE + asor_σ×ASOR_ITER + Σ_r(asor_Σy²_r)×ASOR_ITER
                  BASE = Stockmeyer(deg15 密路径 4) + δ1 平方 (log2(delta1)=1) = 5
                  ASOR_ITER = 1：HE ``he_asor_ct``（DeltaCt 把 kn 记进 delta，
                  每轮每轨仅 1 次 ct×ct；见 ``HE/thor/probe_asor_iter_depth.py``）
                  Σy² 轮数 = ceil(log2(exp_div))；第 1/2 轮 iters 可不同
  LayerNorm:      invsqrt_max_iters + 3（生产路径固定步数，无 α）
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from gelu_poly import GELU_LEVEL_KEYS, gelu_config_for_layer, gelu_level_allowed
from layernorm_poly import (
    LAYERNORM_LEVEL_KEYS,
    count_he_invsqrt_iters,
    layernorm_config_for_layer,
)
from softmax_poly import (
    LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK,
    LAYER_EXP_DIV_BY_TASK,
    SOFTMAX_LEVEL_KEYS,
)

NUM_LAYERS = 12
SCHEME_ORIGINAL = 3
SCHEME_SLOTS_PER_LAYER = 4
SCHEME_LEN = NUM_LAYERS * SCHEME_SLOTS_PER_LAYER

LAYERNORM_DEPTH_OVERHEAD = 3

# 进化搜索 depth 占位：f_cost = ceil(深度和 / COST_DEPTH_DIVISOR)
COST_DEPTH_DIVISOR = 1

# 主线深度占位用的预算（历史默认 15）。DualRail HE / 进化 bts 用
# ``HE/thor/bts_ops.BOOTSTRAP_DEPTH_BUDGET``（=14，与 plan_mock 一致）。
BOOTSTRAP_DEPTH_BUDGET = 15

# DualRail DP 的 bts 预算（与 smoke_thor_chain plan_mock 相同）。
BTS_DEPTH_BUDGET = 14
# thor exp：Stockmeyer(deg15) 关键路径 4（ct×ct；pt 系数乘在旁路不拉长关键路径）
# + δ1=2 的 1 次 square。旧值 7 为未精算占位。
SOFTMAX_DEPTH_BASE = 5
SOFTMAX_EXP_STOCKMEYER_DEPTH = 4
SOFTMAX_EXP_SQUARE_DEPTH = 1  # log2(THOR_DELTA1)；须与 SOFTMAX_DEPTH_BASE 之和一致
assert (
    SOFTMAX_EXP_STOCKMEYER_DEPTH + SOFTMAX_EXP_SQUARE_DEPTH == SOFTMAX_DEPTH_BASE
)
# Softmax aSOR 每轮 HE rem：1（DeltaCt 路径；明文 kn*b*b_temp 概念深度仍为 2）
SOFTMAX_ASOR_ITER_DEPTH = 1


def depth_sum_to_f_cost(depth: int) -> int:
    """乘法深度和 → 进化 f_cost（depth 占位模式）。"""
    return math.ceil(depth / COST_DEPTH_DIVISOR)


def depth_f_cost_label() -> str:
    return f"ceil(depth/{COST_DEPTH_DIVISOR})"


def f_cost_label(cost_mode: str) -> str:
    if cost_mode == "bts":
        return f"bts(budget={BTS_DEPTH_BUDGET}, DualRail DP)"
    if cost_mode == "depth":
        return depth_f_cost_label()
    raise ValueError(f"unknown cost_mode={cost_mode!r}")


def _thor_bts_ops():
    """Load ``HE/thor/bts_ops`` by path (avoid ``HE.bts_ops`` / circular import)."""
    import importlib.util

    thor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HE", "thor")
    path = os.path.join(thor_dir, "bts_ops.py")
    if thor_dir not in sys.path:
        sys.path.append(thor_dir)
    cached = sys.modules.get("wsm_thor_bts_ops")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("wsm_thor_bts_ops", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DualRail bts_ops from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wsm_thor_bts_ops"] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=4096)
def _scheme_bts_cached(
    task_name: str,
    scheme: tuple[int, ...],
    budget: int,
    phase: str,
) -> int:
    bts_ops = _thor_bts_ops()
    return int(
        bts_ops.compute_scheme_bts(
            task_name,
            list(scheme),
            budget=int(budget),
            phase=phase,
        )
    )


def compute_scheme_bts(
    task_name: str,
    scheme: list[int],
    *,
    budget: int | None = None,
    phase: str = "C",
) -> int:
    """最优 DualRail bootstrap 放置下的 CT 加权 bts（档位仅 0/1/2）。"""
    validate_scheme(scheme, task_name)
    for idx, level in enumerate(scheme):
        if level not in (0, 1, 2):
            raise ValueError(
                f"{task_name} bts cost 仅允许档位 0/1/2，"
                f"下标 {idx} 为 {level}（不支持 original）"
            )
    if task_name not in LAYER_EXP_DIV_BY_TASK:
        raise KeyError(f"未知任务：{task_name}")
    bgt = BTS_DEPTH_BUDGET if budget is None else int(budget)
    return _scheme_bts_cached(task_name, tuple(int(x) for x in scheme), bgt, phase)


def compute_scheme_bts_detail(
    task_name: str,
    scheme: list[int],
    *,
    budget: int | None = None,
    phase: str = "C",
) -> Any:
    """完整 ``BootstrapResult``（placements / rem_trace）。"""
    validate_scheme(scheme, task_name)
    bts_ops = _thor_bts_ops()
    kw: dict[str, Any] = {"phase": phase}
    if budget is not None:
        kw["budget"] = int(budget)
    else:
        kw["budget"] = BTS_DEPTH_BUDGET
    return bts_ops.optimize_bootstrap(task_name, scheme, **kw)


def compute_f_cost(
    task_name: str,
    scheme: list[int],
    *,
    cost_mode: str = "bts",
    budget: int | None = None,
    phase: str = "C",
) -> int:
    """进化目标 ``f_cost``：``bts`` = DP 总次数；``depth`` = ceil(深度和 / 除数)。"""
    if cost_mode == "bts":
        return compute_scheme_bts(
            task_name, scheme, budget=budget, phase=phase
        )
    if cost_mode == "depth":
        return depth_sum_to_f_cost(int(compute_scheme_cost(task_name, scheme)))
    raise ValueError(f"cost_mode must be 'depth' or 'bts', got {cost_mode!r}")


def log2_ceil(n: float) -> int:
    """log2(n) 向上取整；n 须 > 0。"""
    if n <= 0:
        raise ValueError(f"log2_ceil 要求 n > 0，当前 n={n}")
    return math.ceil(math.log2(n))


def scheme_index(layer_idx: int, kind: str) -> int:
    slot = {
        "softmax": 0,
        "ln1": 1,
        "gelu": 2,
        "ln2": 3,
    }.get(kind)
    if slot is None:
        raise ValueError(f"未知 kind：{kind}")
    return layer_idx * SCHEME_SLOTS_PER_LAYER + slot


def validate_scheme(scheme: list[int], task_name: str = "") -> None:
    if len(scheme) != SCHEME_LEN:
        raise ValueError(
            f"{task_name} 方案长度应为 {SCHEME_LEN}，当前为 {len(scheme)}"
        )
    for idx, level in enumerate(scheme):
        if level not in (0, 1, 2, SCHEME_ORIGINAL):
            raise ValueError(
                f"{task_name} 方案下标 {idx} 非法：{level}（仅允许 0/1/2/3）"
            )


def gelu_poly_depth(layer_idx: int, level: int) -> int:
    """单层 GeLU Chebyshev 方案 HE 乘法深度；禁止档位抛错。"""
    cfg = gelu_config_for_layer(layer_idx, level)
    return int(cfg["depth_he"])


def softmax_poly_depth(task_name: str, layer_idx: int, level: int) -> int:
    """单层 thor_softmax 乘法深度；level∈{0,1,2}。aSOR 固定 max_iters（无 α）。

    aSOR 每轮按 ``SOFTMAX_ASOR_ITER_DEPTH``（HE 实测 rem=1）计，不是明文公式的 2。
    """
    level_key = SOFTMAX_LEVEL_KEYS[level]
    iters_sigma = LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK[task_name][level_key][
        layer_idx
    ]
    iters_sq = [
        LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK[task_name][level_key][layer_idx]
    ]
    n2 = log2_ceil(LAYER_EXP_DIV_BY_TASK[task_name][layer_idx])
    if n2 >= 2:
        iters_sq.append(
            LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK[task_name][level_key][layer_idx]
        )
    if n2 > 2:
        raise ValueError(
            f"{task_name} 层{layer_idx} exp_div 对应 {n2} 轮 Σy²，暂只支持 ≤2"
        )
    d = SOFTMAX_ASOR_ITER_DEPTH
    return (
        SOFTMAX_DEPTH_BASE
        + int(iters_sigma) * d
        + sum(int(x) for x in iters_sq[:n2]) * d
    )


def layernorm_poly_depth(
    task_name: str, layer_idx: int, kind: str, level: int
) -> int:
    """单层 HE LayerNorm 乘法深度 = he_invsqrt 迭代次数 + 3。"""
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
    return iters + LAYERNORM_DEPTH_OVERHEAD


@dataclass
class LayerCostItem:
    layer_idx: int
    kind: str
    level: int
    depth: int


@dataclass
class SchemeCostResult:
    task_name: str
    total_depth: int
    softmax_depth: int
    gelu_depth: int
    layernorm_depth: int
    layers: list[LayerCostItem]

    def summary(self) -> str:
        return (
            f"total={self.total_depth} "
            f"(softmax={self.softmax_depth}, gelu={self.gelu_depth}, "
            f"layernorm={self.layernorm_depth})"
        )


def compute_scheme_cost(
    task_name: str,
    scheme: list[int],
    *,
    detailed: bool = False,
) -> int | SchemeCostResult:
    """
    输入 POLY_SCHEMES 方案，返回当前 cost（深度之和）。

    detailed=True 时返回 SchemeCostResult，含逐层明细。
    """
    validate_scheme(scheme, task_name)
    if task_name not in LAYER_EXP_DIV_BY_TASK:
        raise KeyError(f"未知任务：{task_name}")

    items: list[LayerCostItem] = []
    softmax_total = 0
    gelu_total = 0
    layernorm_total = 0

    for layer_idx in range(NUM_LAYERS):
        for kind in ("softmax", "ln1", "gelu", "ln2"):
            level = scheme[scheme_index(layer_idx, kind)]
            if level == SCHEME_ORIGINAL:
                depth = 0
            elif kind == "softmax":
                depth = softmax_poly_depth(task_name, layer_idx, level)
                softmax_total += depth
            elif kind == "gelu":
                depth = gelu_poly_depth(layer_idx, level)
                gelu_total += depth
            else:
                depth = layernorm_poly_depth(task_name, layer_idx, kind, level)
                layernorm_total += depth

            items.append(LayerCostItem(layer_idx, kind, level, depth))

    result = SchemeCostResult(
        task_name=task_name,
        total_depth=softmax_total + gelu_total + layernorm_total,
        softmax_depth=softmax_total,
        gelu_depth=gelu_total,
        layernorm_depth=layernorm_total,
        layers=items,
    )
    return result if detailed else result.total_depth


def print_scheme_cost(task_name: str, scheme: list[int]) -> SchemeCostResult:
    """打印方案 cost 明细。"""
    result = compute_scheme_cost(task_name, scheme, detailed=True)
    assert isinstance(result, SchemeCostResult)

    print(f"\n任务 {task_name.upper()}  方案 cost")
    print(f"  深度：{result.summary()}")
    try:
        bts = compute_scheme_bts(task_name, scheme)
        print(f"  bts：{bts}  ({f_cost_label('bts')})")
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"  bts：不可用（{exc}）")
    print(f"  {'层':>4} {'类型':<8} {'档位':>4} {'深度':>6}")
    print("  " + "-" * 28)
    for item in result.layers:
        if item.depth == 0 and item.level == SCHEME_ORIGINAL:
            level_str = "orig"
        elif item.kind in ("ln1", "ln2"):
            level_str = LAYERNORM_LEVEL_KEYS[item.level]
        else:
            keys = (
                SOFTMAX_LEVEL_KEYS
                if item.kind == "softmax"
                else GELU_LEVEL_KEYS
            )
            level_str = keys[item.level]
        print(
            f"  {item.layer_idx:4d} {item.kind:<8} {level_str:>4} {item.depth:6d}"
        )
    return result

if __name__ == "__main__":
    TASK_NAMES = ["mrpc", "rte", "sst2"]

    def _all_high() -> list[int]:
        out: list[int] = []
        for layer_idx in range(NUM_LAYERS):
            for kind in ("softmax", "ln1", "gelu", "ln2"):
                lv = 2
                if kind == "gelu" and not gelu_level_allowed(layer_idx, lv):
                    lv = 2
                out.append(lv)
        return out

    print("POLY_SCHEMES cost（深度和 + DualRail DP bts）")
    print("=" * 56)
    for task in TASK_NAMES:
        print_scheme_cost(task, _all_high())
