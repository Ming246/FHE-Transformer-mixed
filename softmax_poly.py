"""
thor_softmax 近似：参数 + 实现（对齐 nolinear/eval_softmax_cuda.py）。

  - exp：THOR Horner 多项式 + δ1/δ2 平方还原
  - 倒数：aSOR（固定迭代次数，无 α；α 仅留在 eval_softmax_cuda）
  - 第一次 e0 = min(σ)/SIGMA_E0_DIVISOR；min(σ) 来自 nolinear/sigma_out/
  - δ2 轮 e0 = 上一轮结束 en / MAX_SEQ_LENGTH（与 eval 一致）
  - 高/中/低档由 max_iters 区分；Σy² 最多两轮各有独立上限

用法：
    from softmax_poly import ThorSoftmaxEvaluator, additive_attention_mask_to_key_valid
    probs = fn(scores, key_valid_mask=key_valid)
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import torch

SOFTMAX_LEVEL_KEYS = ("low", "mid", "high")
NUM_LAYERS = 12
MAX_SEQ_LENGTH = 128
MAX_SUM_SQ_ROUNDS = 2

# thor_softmax 全局常量
THOR_DELTA1 = 2.0
# x_scaled = x / delta1 / delta2 / THOR_INPUT_SCALE
THOR_INPUT_SCALE = 8.0
SIGMA_E0_DIVISOR = 2.0

_SIGMA_JSON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nolinear", "sigma_out"
)

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
        16, 21.5, 40, 20.5, 18.5, 19,
        17.5, 21.0, 20, 19.5, 17.5, 16.5,
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
# aSOR 迭代上限：task → level → 12 层
# σ 倒数 1 份；Σy² 第 1 / 第 2 轮各 1 份（d2=2 只用第 1 轮；d2=4 用两轮）
# high 对齐 eval_softmax_cuda 调参；mid/low 在 high 上递减
# ---------------------------------------------------------------------------

def _clamp_iters(vals: list[int], delta: int) -> list[int]:
    return [max(1, int(v) - delta) for v in vals]



LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK: dict[str, dict[str, list[int]]] = {
    "mrpc": {
        "low": [3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5],
        "mid": [3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5],
        "high": [4, 5, 5, 5, 5, 5, 5, 5, 7, 7, 6, 6],
    },
    "rte": {
        "low": [4, 6, 4, 5, 4, 5, 4, 5, 5, 5, 4, 4],
        "mid": [5, 7, 4, 6, 5, 6, 5, 6, 6, 6, 5, 5],
        "high": [5, 7, 4, 6, 5, 6, 5, 6, 6, 6, 5, 5],
    },
    "sst2": {
        "low": [4, 4, 4, 5, 4, 4, 5, 4, 5, 5, 5, 5],
        "mid": [4, 5, 5, 5, 4, 4, 5, 4, 5, 5, 5, 5],
        "high": [5, 5, 5, 6, 5, 5, 6, 5, 6, 6, 5, 6],
    },
}

LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK: dict[str, dict[str, list[int]]] = {
    "mrpc": {
        "low": [5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "mid": [6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        "high":[6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    },
    "rte": {
        "low": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "mid": [5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "high": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    },
    "sst2": {
        "low": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5],
        "mid": [6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 5, 6],
        "high": [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 5, 6],
    },
}

LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK: dict[str, dict[str, list[int]]] = {
    # 仅 δ2=4 的层（n2=2）使用第 2 轮；其余层填 0（不参与推理/深度）
    "mrpc": {
        "low": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "mid": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "high": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "rte": {
        "low": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "mid": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "high": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "sst2": {
        "low": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        "mid": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0],
        "high": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0],
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
            "asor_max_iters_sigma": (
                LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK[_task][_level]
            ),
            "asor_max_iters_sum_sq": (
                LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK[_task][_level]
            ),
            "asor_max_iters_sum_sq2": (
                LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK[_task][_level]
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


def sigma_json_path(task_name: str, json_dir: str = _SIGMA_JSON_DIR) -> str:
    return os.path.join(json_dir, f"{task_name}_sigma.json")


@lru_cache(maxsize=8)
def load_min_sigma_for_task(
    task_name: str, json_dir: str = _SIGMA_JSON_DIR
) -> tuple[float, ...]:
    """读 nolinear/sigma_out/{task}_sigma.json 的 min_sigma[12]。"""
    path = sigma_json_path(task_name, json_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"σ JSON 不存在：{path}；请先运行 "
            f"`python3 nolinear/eval_softmax_cuda.py --recompute-sigma`"
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    json_task = payload.get("task")
    if json_task and json_task != task_name:
        raise ValueError(
            f"{path} 中 task={json_task!r} 与请求任务 {task_name!r} 不一致"
        )
    if "min_sigma" not in payload:
        raise KeyError(f"{path} 缺少 min_sigma 字段")
    values = payload["min_sigma"]
    if len(values) != NUM_LAYERS:
        raise ValueError(f"{path} min_sigma 长度应为 {NUM_LAYERS}")
    out: list[float] = []
    for i, v in enumerate(values):
        vf = float(v)
        if not (vf > 0.0) or not math.isfinite(vf):
            raise ValueError(f"{task_name} 层{i} min_sigma={v} 须为有限正数")
        out.append(vf)
    return tuple(out)


def e0_sigma_from_min(min_sigma: float, divisor: float = SIGMA_E0_DIVISOR) -> float:
    return float(min_sigma) / float(divisor)


def thor_eps2_from_en(en: float, seq_len: int = MAX_SEQ_LENGTH) -> float:
    """与 eval_softmax_cuda 一致：e0_Σy² = en / seq_len。"""
    return float(en) / float(seq_len)


def asor_final_en(e0: float, max_iters: int) -> float:
    """固定 max_iters 步后的误差界 en（无 α）。"""
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    for _ in range(int(max_iters)):
        kn = 2.0 / (en + 1.0)
        en = kn * en * (2.0 - kn * en)
    return en


def softmax_config_for_layer(task_name: str, layer_idx: int, level: int) -> dict:
    if task_name not in SOFTMAX_CONFIG_BANK:
        raise KeyError(f"任务 {task_name} 未配置 SOFTMAX_CONFIG_BANK")
    if level < 0 or level >= len(SOFTMAX_LEVEL_KEYS):
        raise ValueError(f"softmax level 非法：{level}")
    cfg = SOFTMAX_CONFIG_BANK[task_name][SOFTMAX_LEVEL_KEYS[level]]
    min_sigma = float(load_min_sigma_for_task(task_name)[layer_idx])
    return {
        "delta1": cfg["delta1"],
        "shift": cfg["layer_exp_shift"][layer_idx],
        "delta2": cfg["layer_exp_div"][layer_idx],
        "min_sigma": min_sigma,
        "e0_sigma": e0_sigma_from_min(min_sigma),
        "asor_max_iters_sigma": int(cfg["asor_max_iters_sigma"][layer_idx]),
        "asor_max_iters_sum_sq": (
            int(cfg["asor_max_iters_sum_sq"][layer_idx]),
            int(cfg["asor_max_iters_sum_sq2"][layer_idx]),
        ),
    }


def asor_inverse_batched(
    denom: torch.Tensor,
    e0: float,
    max_iters: int,
) -> tuple[torch.Tensor, float]:
    """
    aSOR 求 1/denom（与 THOR he_inv 同形）。
    固定跑 max_iters 步（无 α 提前停止）。
    返回 (近似倒数, 结束时 en)。
    """
    a = torch.ones_like(denom)
    b = denom
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    for _ in range(max_iters):
        kn = 2.0 / (en + 1.0)
        b_temp = 2.0 - kn * b
        b = kn * b * b_temp
        a = kn * a * b_temp
        en = kn * en * (2.0 - kn * en)
    return a, en


def thor_softmax(
    x: torch.Tensor,
    mask: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    e0_sigma: float,
    asor_max_iters_sigma: int,
    asor_max_iters_sum_sq: tuple[int, int] | list[int],
    exp_coeffs: torch.Tensor,
) -> torch.Tensor:
    """thor_softmax + aSOR；x/mask: [N, L]，mask 为 1/0。"""
    if len(asor_max_iters_sum_sq) < MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"asor_max_iters_sum_sq 长度须 ≥ {MAX_SUM_SQ_ROUNDS}"
        )
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)
    n2 = _check_power_of_two("delta2", d2)
    if n2 > MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"delta2={d2} 对应 {n2} 轮 Σy²，超过 MAX_SUM_SQ_ROUNDS="
            f"{MAX_SUM_SQ_ROUNDS}"
        )

    x_scaled = x_work / d1 / d2 / THOR_INPUT_SCALE
    exp_approx = torch.zeros_like(x_scaled)
    for coeff in exp_coeffs:
        exp_approx = exp_approx * x_scaled + coeff

    scale = math.exp(shift / d2 / d1)
    exp_approx = exp_approx / scale

    for _ in range(n1):
        exp_approx = exp_approx**2

    exp_approx = exp_approx * m
    sigma_exp = exp_approx.sum(dim=-1)
    inv_sigma, en = asor_inverse_batched(
        sigma_exp, e0_sigma, asor_max_iters_sigma
    )
    y = exp_approx * inv_sigma.unsqueeze(-1)

    for r in range(n2):
        e0_2 = thor_eps2_from_en(en)
        y_squared = y**2
        sum_y_sq = (y_squared * m).sum(dim=-1)
        inv_sum_sq, en = asor_inverse_batched(
            sum_y_sq, e0_2, int(asor_max_iters_sum_sq[r])
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
            self._cfg["e0_sigma"],
            self._cfg["asor_max_iters_sigma"],
            self._cfg["asor_max_iters_sum_sq"],
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
