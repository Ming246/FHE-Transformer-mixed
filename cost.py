"""
POLY_SCHEMES 方案 cost 估算（当前仅用乘法深度占位，后续需更新完整 cost 模型）。

方案数组：长度 48 = 12 层 × (softmax, ln1, gelu, ln2)，元素 0/1/2=多项式档位，3=原始函数（cost=0）。

深度公式（GeLU Chebyshev / HE PS-tree）：
  取自 gelu_poly 方案配置 depth_he（含 GELU 还原 +1）
  Softmax:        SOFTMAX_DEPTH_BASE + gs_sigma*2 + ceil(log2(exp_div))*gs_sum_sq*2
                  BASE = Stockmeyer(deg15 关键路径 4) + δ1 平方 (log2(delta1)=1) = 5
  LayerNorm:      he_invsqrt 迭代次数 + 3（迭代次数由 variance JSON 的 min/max 与 alpha 确定）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from gelu_poly import GELU_LEVEL_KEYS, gelu_config_for_layer, gelu_level_allowed
from layernorm_poly import (
    LAYERNORM_LEVEL_KEYS,
    count_he_invsqrt_iters,
    layernorm_config_for_layer,
)
from softmax_poly import (
    LAYER_EXP_DIV_BY_TASK,
    LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK,
    LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK,
    SOFTMAX_LEVEL_KEYS,
)

NUM_LAYERS = 12
SCHEME_ORIGINAL = 3
SCHEME_SLOTS_PER_LAYER = 4
SCHEME_LEN = NUM_LAYERS * SCHEME_SLOTS_PER_LAYER

LAYERNORM_DEPTH_OVERHEAD = 3

# 进化搜索 depth 占位：f_cost = ceil(深度和 / COST_DEPTH_DIVISOR)
COST_DEPTH_DIVISOR = 5

# 单次 bootstrapping 恢复的乘法深度预算（bts 求解器输入；可调）
BOOTSTRAP_DEPTH_BUDGET = 15

# TODO: 后续更新 — 完整 cost 应包含非乘法项、数据通路、并行度等；当前仅深度求和。
# thor exp：Stockmeyer(deg15) 关键路径 4（ct×ct；pt 系数乘在旁路不拉长关键路径）
# + δ1=2 的 1 次 square。旧值 7 为未精算占位。
SOFTMAX_DEPTH_BASE = 5
SOFTMAX_EXP_STOCKMEYER_DEPTH = 4
SOFTMAX_EXP_SQUARE_DEPTH = 1  # log2(THOR_DELTA1)；须与 SOFTMAX_DEPTH_BASE 之和一致
assert (
    SOFTMAX_EXP_STOCKMEYER_DEPTH + SOFTMAX_EXP_SQUARE_DEPTH == SOFTMAX_DEPTH_BASE
)


def depth_sum_to_f_cost(depth: int) -> int:
    """乘法深度和 → 进化 f_cost（depth 占位模式）。"""
    return math.ceil(depth / COST_DEPTH_DIVISOR)


def depth_f_cost_label() -> str:
    return f"ceil(depth/{COST_DEPTH_DIVISOR})"


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
    """单层 thor_softmax 乘法深度；level∈{0,1,2}。"""
    level_key = SOFTMAX_LEVEL_KEYS[level]
    gs_sigma = LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK[task_name][level_key][
        layer_idx
    ]
    gs_sum_sq = LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK[task_name][level_key][
        layer_idx
    ]
    exp_div = LAYER_EXP_DIV_BY_TASK[task_name][layer_idx]
    return (
        SOFTMAX_DEPTH_BASE
        + gs_sigma * 2
        + log2_ceil(exp_div) * gs_sum_sq * 2
    )


def layernorm_poly_depth(
    task_name: str, layer_idx: int, kind: str, level: int
) -> int:
    """单层 HE LayerNorm 乘法深度 = he_invsqrt 迭代次数 + 3。"""
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(
        cfg["min_var"],
        cfg["max_var"],
        cfg["invsqrt_alpha"],
        cfg["invsqrt_max_iters"],
    )
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

    print(f"\n任务 {task_name.upper()}  方案 cost（深度占位，后续需更新）")
    print(f"  合计：{result.summary()}")
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

# TODO(临时): 后续更新 — 完整 cost 应为最佳bootstrapping分配方案中的bts次数；当前仅深度求和。
if __name__ == "__main__":
    TASK_NAMES = ["mrpc", "rte", "sst2"]
    # 本地示例方案（长度 24 = 12 层 × softmax/gelu；0/1/2=多项式档位，3=原始）

    POLY_SCHEMES: dict[str, list[int]] = {
        "mrpc": [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        "rte": [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
        "sst2": [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
    }

    print("POLY_SCHEMES cost（当前=乘法深度之和，后续需更新完整 cost 模型）")
    print("=" * 56)
    for task in TASK_NAMES:
        if task not in POLY_SCHEMES:
            continue
        print_scheme_cost(task, POLY_SCHEMES[task])
