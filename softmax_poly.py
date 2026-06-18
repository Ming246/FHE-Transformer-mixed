"""
thor_softmax 近似：参数 + 实现（来源 nolinear/eval_softmax_cuda.py）。

用法：
    from softmax_poly import ThorSoftmaxEvaluator, additive_attention_mask_to_key_valid
    probs = fn(scores, key_valid_mask=key_valid)
"""
from __future__ import annotations

import math

import torch

SOFTMAX_LEVEL_KEYS = ("low", "mid", "high")

# thor_softmax 全局常量
THOR_DELTA1 = 2.0
# x_scaled = x / delta1 / delta2 / THOR_INPUT_SCALE
THOR_INPUT_SCALE = 8.0

# exp 逼近 Horner 系数（常数项 → 最高次），已乘 1533
THOR_EXP_POLY_COEFFS: list[float] = [
    0.032855468333339584,
    0.05948672763856172,
    0.03881607331549499,
    0.0670090353368128,
    0.15202099984697098,
    0.20618261949210986,
    0.23721029007596767,
    0.26787311936472025,
    0.27220647178765545,
    0.2379982262906916,
    0.1780344447042791,
    0.11128698173597897,
    0.05566510463488879,
    0.020873931555133732,
    0.005218196900295354,
    0.0006522770224130905,
]
THOR_EXP_POLY_COEFFS = [c * 1533.0 for c in THOR_EXP_POLY_COEFFS]

# per-task × 12 层：exp shift / delta2（各精度档位共用）
LAYER_EXP_SHIFT_BY_TASK: dict[str, list[float]] = {
    "mrpc": [
        14, 18, 38.5, 20, 18.5, 18,
        18, 19, 23.5, 22.5, 19.5, 20.0,
    ],
    "rte": [
        16, 21, 40, 20, 18, 18.5,
        17, 20.0, 19.5, 19, 17, 15.5,
    ],
    "sst2": [
        14, 15.5, 35.5, 18.5, 15.5, 16.5,
        15.5, 16.5, 18.5, 16.5, 32.0, 15.5,
    ],
}

LAYER_EXP_DIV_BY_TASK: dict[str, list[float]] = {
    "mrpc": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "rte": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "sst2": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0],
}

# ---------------------------------------------------------------------------
# Goldschmidt 迭代次数：task → level → 12 层（layer 0 … layer 11）
# 低/高精度请直接改下面 "low" / "high" 里的列表；当前与中精度相同，作占位。
# gs  = 1/sum(exp_approx) 的 Goldschmidt 迭代次数
# gsq = 1/sum(y^2) 的 Goldschmidt 迭代次数（每 delta2 轮归一化各用一次）
# ---------------------------------------------------------------------------

LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK: dict[str, dict[str, list[int]]] = {
    "mrpc": {
        "low":  [4, 6, 5, 7, 6, 6, 6, 6, 8, 8, 8, 8],
        "mid":  [5, 7, 6, 8, 7, 7, 7, 7, 10, 10, 9, 9],
        "high": [6, 8, 7, 9, 8, 8, 8, 8, 12, 12, 10, 10],
        #"high": [6, 9, 7, 10, 9, 9, 9, 9, 13, 13, 10, 11],
    },
    "rte": {
        "low":  [5, 7, 7, 6, 5, 5, 5, 6, 6, 6, 5, 4],
        "mid":  [6, 8, 7, 7, 6, 6, 6, 7, 7, 7, 6, 5],
        "high": [7, 10, 8, 9, 7, 8, 7, 9, 8, 8, 7, 6],

    },
    "sst2": {
        "low":  [3, 3, 8, 5, 3, 4, 3, 3, 5, 4, 7, 4],
        "mid":  [4, 4, 8, 6, 4, 5, 4, 4, 6, 5, 7, 5],
        "high": [5, 5, 8, 7, 5, 6, 5, 5, 7, 6, 7, 6],
        #[5, 6, 8, 8, 6, 7, 6, 6, 8, 7, 7, 7],
        #[6, 7, 8, 9, 7, 8, 7, 7, 9, 8, 7, 8],
    },
}

LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK: dict[str, dict[str, list[int]]] = {
    "mrpc": {
        "low":  [4, 5, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4],
        "mid":  [5, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 5],
        "high": [7, 7, 7, 7, 7, 6, 6, 6, 8, 8, 7, 6],
        #[8, 8, 8, 8, 8, 7, 7, 7, 8, 7, 8, 7],
    },
    "rte": {
        "low":  [5, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6],
        "mid":  [6, 8, 7, 7, 7, 7, 6, 7, 7, 7, 7, 7],
        "high": [9, 9, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        
    },
    "sst2": {
        "low":  [5, 5, 6, 6, 5, 5, 6, 6, 5, 5, 4, 5],
        "mid":  [6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 5, 6],
        "high": [7, 8, 8, 8, 7, 7, 8, 8, 8, 7, 7, 7],
        #[8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 7, 8],
    },
}

# task → level → 层参数摘要（供 ThorSoftmaxEvaluator 查表）
SOFTMAX_CONFIG_BANK: dict[str, dict[str, dict]] = {}
for _task in LAYER_EXP_SHIFT_BY_TASK:
    SOFTMAX_CONFIG_BANK[_task] = {}
    for _level in SOFTMAX_LEVEL_KEYS:
        SOFTMAX_CONFIG_BANK[_task][_level] = {
            "delta1": THOR_DELTA1,
            "layer_exp_shift": LAYER_EXP_SHIFT_BY_TASK[_task],
            "layer_exp_div": LAYER_EXP_DIV_BY_TASK[_task],
            "goldschmidt_iterations_sigma": (
                LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK[_task][_level]
            ),
            "goldschmidt_iterations_sum_sq": (
                LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK[_task][_level]
            ),
        }

# HF BERT 加性 attention_mask：0=可 attend，约 -1e4=padding/不可 attend
ADDITIVE_MASK_ATTEND_MAX = -1.0

SOFTMAX_POLY_FUNC_NAMES: list[str] = [
    "poly_softmax_low",
    "poly_softmax_mid",
    "poly_softmax_high",
]


def additive_attention_mask_to_key_valid(
    additive_mask: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """加性 mask → key 维有效位（1=有效，0=被 mask）。"""
    return (additive_mask > ADDITIVE_MASK_ATTEND_MAX).to(dtype=dtype)


def _check_power_of_two(name: str, val: float) -> int:
    n = int(round(math.log2(val)))
    if 2**n != int(val) and abs(2**n - val) > 1e-6:
        raise ValueError(f"{name}={val} 须为 2 的整数幂")
    return n


def softmax_config_for_layer(task_name: str, layer_idx: int, level: int) -> dict:
    if task_name not in SOFTMAX_CONFIG_BANK:
        raise KeyError(f"任务 {task_name} 未配置 SOFTMAX_CONFIG_BANK")
    if level < 0 or level >= len(SOFTMAX_LEVEL_KEYS):
        raise ValueError(f"softmax level 非法：{level}")
    cfg = SOFTMAX_CONFIG_BANK[task_name][SOFTMAX_LEVEL_KEYS[level]]
    return {
        "delta1": cfg["delta1"],
        "shift": cfg["layer_exp_shift"][layer_idx],
        "delta2": cfg["layer_exp_div"][layer_idx],
        "goldschmidt_iterations_sigma": cfg["goldschmidt_iterations_sigma"][
            layer_idx
        ],
        "goldschmidt_iterations_sum_sq": cfg["goldschmidt_iterations_sum_sq"][
            layer_idx
        ],
    }


def goldschmidt_inverse_batched(x: torch.Tensor, iterations: int) -> torch.Tensor:
    y = 1.0 - x
    result = 2.0 - x
    for _ in range(iterations):
        y = y * y
        tmp = 1.0 + y
        result = result * tmp
    return result


def thor_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    goldschmidt_iterations_sigma: int,
    goldschmidt_iterations_sum_sq: int,
    exp_coeffs: torch.Tensor,
) -> torch.Tensor:
    """thor_softmax；x/mask: [N, L]，mask 为 1/0。"""
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)
    n2 = _check_power_of_two("delta2", d2)

    x_scaled = x_work / d1 / d2 / THOR_INPUT_SCALE
    exp_approx = torch.zeros_like(x_scaled)
    for coeff in exp_coeffs:
        exp_approx = exp_approx * x_scaled + coeff

    scale = math.exp(shift / d2 / d1)
    exp_approx = exp_approx / scale

    for _ in range(n1):
        exp_approx = exp_approx**2
    #exp_approx = exp_approx/8
    exp_approx = exp_approx * m
    sigma_exp = exp_approx.sum(dim=-1)
    inv_sigma = goldschmidt_inverse_batched(
        sigma_exp, goldschmidt_iterations_sigma
    )
    y = exp_approx * inv_sigma.unsqueeze(-1)

    for _ in range(n2):
        y_squared = y**2
        sum_y_sq = (y_squared * m).sum(dim=-1)
        inv_sum_sq = goldschmidt_inverse_batched(
            sum_y_sq, goldschmidt_iterations_sum_sq
        )
        y = y_squared * inv_sum_sq.unsqueeze(-1)

    return y * m


def _align_key_valid_mask(
    key_valid_mask: torch.Tensor, scores: torch.Tensor
) -> torch.Tensor:
    """将 HF 加性 mask 派生的 key_valid 对齐到 scores（如 [B,1,L,L] → [B,H,L,L]）。"""
    if key_valid_mask.shape == scores.shape:
        return key_valid_mask
    try:
        return key_valid_mask.expand_as(scores)
    except RuntimeError as e:
        raise ValueError(
            f"key_valid_mask 形状 {key_valid_mask.shape} 无法 broadcast 到 scores {scores.shape}"
        ) from e


class ThorSoftmaxEvaluator:
    """per-task × per-layer thor_softmax；exp 系数缓存到 x.device。"""

    __slots__ = (
        "task_name",
        "layer_idx",
        "level",
        "_cfg",
        "_cached_key",
        "_exp_coeffs",
    )

    def __init__(self, task_name: str, layer_idx: int, level: int):
        self.task_name = task_name
        self.layer_idx = layer_idx
        self.level = level
        self._cfg = softmax_config_for_layer(task_name, layer_idx, level)
        self._cached_key: tuple[torch.device, torch.dtype] | None = None
        self._exp_coeffs: torch.Tensor | None = None

    def _ensure_exp_coeffs(self, device: torch.device, dtype: torch.dtype) -> None:
        key = (device, dtype)
        if self._cached_key == key:
            return
        self._exp_coeffs = torch.tensor(
            THOR_EXP_POLY_COEFFS, device=device, dtype=dtype
        )
        self._cached_key = key

    def __call__(
        self,
        scores: torch.Tensor,
        dim: int = -1,
        key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dim != -1 and dim != scores.dim() - 1:
            raise ValueError("thor_softmax 仅支持在最后一维做 softmax")
        if key_valid_mask is None:
            raise ValueError("须显式传入 key_valid_mask（1=有效 key，0=被 mask）")
        key_valid_mask = _align_key_valid_mask(key_valid_mask, scores)
        self._ensure_exp_coeffs(scores.device, scores.dtype)
        orig_shape = scores.shape
        x = scores.reshape(-1, orig_shape[-1])
        mask = key_valid_mask.to(device=scores.device, dtype=scores.dtype).reshape(
            -1, orig_shape[-1]
        )
        y = thor_softmax(
            x,
            mask,
            self._cfg["shift"],
            self._cfg["delta1"],
            self._cfg["delta2"],
            self._cfg["goldschmidt_iterations_sigma"],
            self._cfg["goldschmidt_iterations_sum_sq"],
            self._exp_coeffs,
        )
        return y.reshape(orig_shape)


def poly_softmax_low(
    scores: torch.Tensor,
    layer_idx: int,
    key_valid_mask: torch.Tensor,
    dim: int = -1,
    *,
    task_name: str,
) -> torch.Tensor:
    return ThorSoftmaxEvaluator(task_name, layer_idx, 0)(
        scores, dim=dim, key_valid_mask=key_valid_mask
    )


def poly_softmax_mid(
    scores: torch.Tensor,
    layer_idx: int,
    key_valid_mask: torch.Tensor,
    dim: int = -1,
    *,
    task_name: str,
) -> torch.Tensor:
    return ThorSoftmaxEvaluator(task_name, layer_idx, 1)(
        scores, dim=dim, key_valid_mask=key_valid_mask
    )


def poly_softmax_high(
    scores: torch.Tensor,
    layer_idx: int,
    key_valid_mask: torch.Tensor,
    dim: int = -1,
    *,
    task_name: str,
) -> torch.Tensor:
    return ThorSoftmaxEvaluator(task_name, layer_idx, 2)(
        scores, dim=dim, key_valid_mask=key_valid_mask
    )


SOFTMAX_POLY_FUNCS: list = [
    poly_softmax_low,
    poly_softmax_mid,
    poly_softmax_high,
]
