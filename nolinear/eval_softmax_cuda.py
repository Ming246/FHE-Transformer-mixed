"""
eval_softmax 的 GPU 批量版：thor_softmax + aSOR 倒数。

  - 第一次倒数：e0 = min(σ)/SIGMA_E0_DIVISOR
  - 后续 δ2 轮：e0 = 上一轮 aSOR 结束时的 en / MAX_SEQ_LENGTH / 2（THOR 公式）
  - 停止：en ≥ 1−α **或** iters ≥ max_iters（二者同时限制）
  - α / max_iters：按 task×层配置；σ 一轮，Σy² 最多两轮（d2=4 → 两轮）各有独立 α/max_iters
  - min(σ) 先整体统计写入 sigma_out/{task}_sigma.json，评估时读文件或内存缓存
    （接口供后续 softmax_poly 复用）

用法（在 nolinear/ 下）：
  python3 eval_softmax_cuda.py                      # 读 JSON，做误差评估
  python3 eval_softmax_cuda.py --recompute-sigma    # 重算 minσ、写 JSON、再评估
  python3 eval_softmax_cuda.py --sigma-only --recompute-sigma  # 仅统计
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
from datetime import datetime

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# ===================== 配置（可独立修改）=====================
#TASK_NAMES = ["mrpc", "rte", "sst2"]
TASK_NAMES = ["rte"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2

NUM_SAMPLES = None
RANDOM_SEED = 42
BATCH_SIZE = 8

SCORE_VALID_THRESHOLD = -1e4
DEFAULT_THOR_DELTA1 = 2.0
THOR_INPUT_SCALE = 8.0

# aSOR：停在 en ≥ 1−α 或 iters ≥ max_iters（二者同时限制）
# 第一次 e0 = minσ / SIGMA_E0_DIVISOR
# α / max_iters：σ 倒数 1 份；δ2 的 Σy² 倒数暂按最多 2 轮（d2=4→2 轮）各 1 份
SIGMA_E0_DIVISOR = 2.0
DEFAULT_ASOR_MAX_ITERS = 20
MAX_SUM_SQ_ROUNDS = 2  # δ2 轮数上限（暂按 d2≤4）

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(_SCRIPT_DIR, "sigma_out")

# task -> 12 层 min(σ)；load/recompute 后填充，供同进程复用
_MIN_SIGMA_CACHE: dict[str, list[float]] = {}


LAYER_ASOR_ALPHA_SIGMA_BY_TASK: dict[str, list[float]] = {
    "mrpc": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "rte": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "sst2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
}
# Σy² 第 1 / 第 2 轮（d2=2 只用第 1 轮；d2=4 用两轮）
LAYER_ASOR_ALPHA_SUM_SQ_BY_TASK: dict[str, list[float]] = {
    "mrpc": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "rte": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "sst2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
}
LAYER_ASOR_ALPHA_SUM_SQ2_BY_TASK: dict[str, list[float]] = {
    "mrpc": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "rte": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "sst2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
}
LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK: dict[str, list[int]] = {
    "mrpc": [4, 5, 5, 5, 5, 5, 5, 5, 7, 7, 6, 6],
    "rte": [5, 7, 4, 6, 5, 6, 5, 6, 6, 6, 5, 5],
    "sst2": [5, 5, 5, 6, 5, 5, 6, 5, 6, 6, 5, 6],
}
LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK: dict[str, list[int]] = {
    "mrpc":[6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "rte": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "sst2":  [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 5, 6],
}
LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK: dict[str, list[int]] = {
    # 仅 δ2=4 的层使用；其余为 0
    "mrpc": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "rte": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sst2": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 6, 0],
}
LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK["rte"]= [x+1  for x in LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK["rte"]]
#LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK["rte"] = [x+1  for x in LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK["rte"]]
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
    "mrpc": [2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "rte": [2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "sst2": [2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2],
}

MAX_VECTORS_FOR_ERROR_EVAL: int | None = None
ERROR_EVAL_CHUNK_SIZE = 4096

LAYER_ENABLE = [True] * NUM_LAYERS

# Stockmeyer(4) + δ1 square(1)；aSOR 每步按 ×2 计深度（与旧 Goldschmidt 占位一致）
SOFTMAX_DEPTH_BASE = 5
_THOR_P = (
    torch.tensor(
        [
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
        ],
        dtype=torch.float32,
    )
    * 1533.0
)
# =============================================================================


def layer_exp_shift_for_task(task_name: str) -> list[float]:
    if task_name not in LAYER_EXP_SHIFT_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_EXP_SHIFT_BY_TASK")
    shifts = LAYER_EXP_SHIFT_BY_TASK[task_name]
    if len(shifts) != NUM_LAYERS:
        raise ValueError(f"{task_name} shift 长度应为 {NUM_LAYERS}")
    return shifts


def layer_exp_div_for_task(task_name: str) -> list[float]:
    if task_name not in LAYER_EXP_DIV_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_EXP_DIV_BY_TASK")
    divs = LAYER_EXP_DIV_BY_TASK[task_name]
    if len(divs) != NUM_LAYERS:
        raise ValueError(f"{task_name} div 长度应为 {NUM_LAYERS}")
    for i, d in enumerate(divs):
        if d <= 0:
            raise ValueError(f"{task_name} 层{i} div 须 > 0")
    return divs


def _validate_alpha_list(task_name: str, name: str, alphas: list[float]) -> list[float]:
    if len(alphas) != NUM_LAYERS:
        raise ValueError(f"{task_name} {name} 长度应为 {NUM_LAYERS}")
    for i, a in enumerate(alphas):
        if not (0.0 < float(a) < 1.0):
            raise ValueError(f"{task_name} 层{i} {name}={a} 须在 (0,1)")
    return alphas


def layer_asor_alpha_sigma_for_task(task_name: str) -> list[float]:
    if task_name not in LAYER_ASOR_ALPHA_SIGMA_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_ALPHA_SIGMA_BY_TASK")
    return _validate_alpha_list(
        task_name, "asor_alpha_sigma", LAYER_ASOR_ALPHA_SIGMA_BY_TASK[task_name]
    )


def layer_asor_alpha_sum_sq_for_task(task_name: str) -> list[float]:
    """Σy² 第 1 轮 α（d2=2 / d2=4 均使用）。"""
    if task_name not in LAYER_ASOR_ALPHA_SUM_SQ_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_ALPHA_SUM_SQ_BY_TASK")
    return _validate_alpha_list(
        task_name, "asor_alpha_sum_sq", LAYER_ASOR_ALPHA_SUM_SQ_BY_TASK[task_name]
    )


def layer_asor_alpha_sum_sq2_for_task(task_name: str) -> list[float]:
    """Σy² 第 2 轮 α（仅 d2=4 时使用）。"""
    if task_name not in LAYER_ASOR_ALPHA_SUM_SQ2_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_ALPHA_SUM_SQ2_BY_TASK")
    return _validate_alpha_list(
        task_name, "asor_alpha_sum_sq2", LAYER_ASOR_ALPHA_SUM_SQ2_BY_TASK[task_name]
    )


def _validate_max_iters_list(
    task_name: str, name: str, max_iters_list: list[int], *, min_val: int = 1
) -> list[int]:
    if len(max_iters_list) != NUM_LAYERS:
        raise ValueError(f"{task_name} {name} 长度应为 {NUM_LAYERS}")
    out: list[int] = []
    for i, m in enumerate(max_iters_list):
        mi = int(m)
        if mi < min_val:
            raise ValueError(f"{task_name} 层{i} {name}={m} 须 ≥ {min_val}")
        out.append(mi)
    return out


def layer_asor_max_iters_sigma_for_task(task_name: str) -> list[int]:
    if task_name not in LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK")
    return _validate_max_iters_list(
        task_name,
        "asor_max_iters_sigma",
        LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK[task_name],
    )


def layer_asor_max_iters_sum_sq_for_task(task_name: str) -> list[int]:
    """Σy² 第 1 轮 max_iters。"""
    if task_name not in LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK")
    return _validate_max_iters_list(
        task_name,
        "asor_max_iters_sum_sq",
        LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK[task_name],
    )


def layer_asor_max_iters_sum_sq2_for_task(task_name: str) -> list[int]:
    """Σy² 第 2 轮 max_iters（仅 d2=4 时使用；未用层允许 0）。"""
    if task_name not in LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK")
    divs = layer_exp_div_for_task(task_name)
    raw = LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK[task_name]
    if len(raw) != NUM_LAYERS:
        raise ValueError(f"{task_name} asor_max_iters_sum_sq2 长度应为 {NUM_LAYERS}")
    out: list[int] = []
    for i, m in enumerate(raw):
        mi = int(m)
        n2 = _check_power_of_two("delta2", float(divs[i]))
        if n2 >= 2 and mi < 1:
            raise ValueError(
                f"{task_name} 层{i} d2={divs[i]} 需第 2 轮 Σy²，"
                f"asor_max_iters_sum_sq2={m} 须 ≥ 1"
            )
        if mi < 0:
            raise ValueError(f"{task_name} 层{i} asor_max_iters_sum_sq2={m} 须 ≥ 0")
        out.append(mi)
    return out



def sigma_json_path(task_name: str, json_dir: str = JSON_DIR) -> str:
    return os.path.join(json_dir, f"{task_name}_sigma.json")


def _validate_min_sigma_list(task_name: str, values: list[float]) -> list[float]:
    if len(values) != NUM_LAYERS:
        raise ValueError(f"{task_name} min_sigma 长度应为 {NUM_LAYERS}")
    out: list[float] = []
    for i, v in enumerate(values):
        vf = float(v)
        if not (vf > 0.0) or not math.isfinite(vf):
            raise ValueError(f"{task_name} 层{i} min_sigma={v} 须为有限正数")
        out.append(vf)
    return out


def save_min_sigma_json(
    task_name: str,
    min_sigma: list[float],
    json_dir: str = JSON_DIR,
    *,
    max_sigma: list[float] | None = None,
) -> str:
    """写入 nolinear/sigma_out/{task}_sigma.json；并更新内存缓存。"""
    min_sigma = _validate_min_sigma_list(task_name, min_sigma)
    os.makedirs(json_dir, exist_ok=True)
    path = sigma_json_path(task_name, json_dir)
    payload: dict = {
        "task": task_name,
        "min_sigma": min_sigma,
        "sigma_e0_divisor": SIGMA_E0_DIVISOR,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    if max_sigma is not None:
        if len(max_sigma) != NUM_LAYERS:
            raise ValueError(f"{task_name} max_sigma 长度应为 {NUM_LAYERS}")
        payload["max_sigma"] = [float(x) for x in max_sigma]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _MIN_SIGMA_CACHE[task_name] = list(min_sigma)
    return path


def load_min_sigma_from_json(
    task_name: str,
    json_dir: str = JSON_DIR,
    *,
    use_cache: bool = True,
) -> tuple[list[float], str]:
    """
    读 min(σ) 列表。优先内存缓存，否则读 JSON。
    返回 (min_sigma[12], source_path_or_cache)。
    """
    if use_cache and task_name in _MIN_SIGMA_CACHE:
        return list(_MIN_SIGMA_CACHE[task_name]), f"cache:{task_name}"

    path = sigma_json_path(task_name, json_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"σ JSON 不存在：{path}；请先运行 "
            f"`python3 eval_softmax_cuda.py --recompute-sigma`"
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
    min_sigma = _validate_min_sigma_list(task_name, payload["min_sigma"])
    _MIN_SIGMA_CACHE[task_name] = list(min_sigma)
    return list(min_sigma), path


def clear_min_sigma_cache(task_name: str | None = None) -> None:
    if task_name is None:
        _MIN_SIGMA_CACHE.clear()
    else:
        _MIN_SIGMA_CACHE.pop(task_name, None)


def e0_sigma_from_min(min_sigma: float, divisor: float = SIGMA_E0_DIVISOR) -> float:
    return float(min_sigma) / float(divisor)


def _check_power_of_two(name: str, val: float) -> int:
    n = int(round(math.log2(val)))
    if 2**n != int(val) and abs(2**n - val) > 1e-6:
        raise ValueError(f"{name}={val} 须为 2 的整数幂")
    return n


def count_asor_inv_iters(
    e0: float,
    alpha: float,
    max_iters: int = DEFAULT_ASOR_MAX_ITERS,
) -> int:
    """推进误差界 e，返回实际迭代次数（α 与 max_iters 同时限制）。"""
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    iters = 0
    while en < 1.0 - alpha and iters < max_iters:
        kn = 2.0 / (en + 1.0)
        en = kn * en * (2.0 - kn * en)
        iters += 1
    return iters


def asor_final_en(
    e0: float,
    alpha: float,
    max_iters: int = DEFAULT_ASOR_MAX_ITERS,
) -> float:
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    for _ in range(max_iters):
        if en >= 1.0 - alpha:
            break
        kn = 2.0 / (en + 1.0)
        en = kn * en * (2.0 - kn * en)
    return en


def thor_eps2_from_en(en: float, seq_len: int = MAX_SEQ_LENGTH) -> float:
    return float(en) / float(seq_len)


def softmax_layer_depth_asor(
    delta2: float, iters_sigma: int, iters_sum_sq: list[int]
) -> int:
    """单层深度：base + aSOR_σ×2 + Σ(各轮 aSOR_Σy²×2)。"""
    return (
        SOFTMAX_DEPTH_BASE
        + int(iters_sigma) * 2
        + sum(int(x) for x in iters_sum_sq) * 2
    )


def asor_inverse_batched(
    denom: torch.Tensor,
    e0: float,
    alpha: float,
    max_iters: int = DEFAULT_ASOR_MAX_ITERS,
) -> tuple[torch.Tensor, float, int]:
    """
    aSOR 求 1/denom（与 THOR he_inv / adaptive_goldschmidt 同形）。
    停止条件：en ≥ 1−α 或 iters ≥ max_iters。
    e 为公开误差界，全 batch 共用固定迭代次数。
    返回 (近似倒数, 结束时 en, 迭代次数)。
    """
    a = torch.ones_like(denom)
    b = denom
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha 须在 (0,1)，得到 {alpha}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    iters = 0
    while en < 1.0 - alpha and iters < max_iters:
        kn = 2.0 / (en + 1.0)
        b_temp = 2.0 - kn * b
        b = kn * b * b_temp
        a = kn * a * b_temp
        en = kn * en * (2.0 - kn * en)
        iters += 1
    return a, en, iters


def compute_sigma_exp_batched(
    x: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """thor 前半段：返回 σ=Σexp_approx，shape [N]。"""
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)

    x_scaled = x_work / d1 / d2 / THOR_INPUT_SCALE
    device, dtype = x.device, x.dtype
    p = _THOR_P.to(device=device, dtype=dtype)

    exp_approx = torch.zeros_like(x_scaled)
    for coeff in p:
        exp_approx = exp_approx * x_scaled + coeff

    scale = math.exp(shift / d2 / d1)
    exp_approx = exp_approx / scale
    for _ in range(n1):
        exp_approx = exp_approx**2

    exp_approx = exp_approx * m
    return exp_approx.sum(dim=-1)


@torch.no_grad()
def measure_sigma_range(
    scores: torch.Tensor,
    masks: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    chunk_size: int,
) -> tuple[float, float, int]:
    """返回 (min_σ, max_σ, n_vectors)。"""
    n = scores.shape[0]
    if n == 0:
        return float("nan"), float("nan"), 0
    min_sigma = torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)
    max_sigma = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sigma = compute_sigma_exp_batched(
            scores[start:end],
            shift,
            delta1,
            delta2,
            masks[start:end],
        )
        min_sigma = torch.minimum(min_sigma, sigma.min())
        max_sigma = torch.maximum(max_sigma, sigma.max())
    return float(min_sigma.cpu()), float(max_sigma.cpu()), n


def measure_min_sigma(
    scores: torch.Tensor,
    masks: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    chunk_size: int,
) -> float:
    """兼容旧接口；仅返回 min(σ)。"""
    mn, _, _ = measure_sigma_range(
        scores, masks, shift, delta1, delta2, chunk_size
    )
    return mn


def thor_softmax_batched(
    x: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    mask: torch.Tensor,
    e0_sigma: float,
    alpha_sigma: float,
    alpha_sum_sq: tuple[float, float] | list[float],
    max_iters_sigma: int = DEFAULT_ASOR_MAX_ITERS,
    max_iters_sum_sq: tuple[int, int] | list[int] = (
        DEFAULT_ASOR_MAX_ITERS,
        DEFAULT_ASOR_MAX_ITERS,
    ),
) -> tuple[torch.Tensor, dict]:
    """
    批量 thor_softmax + aSOR 倒数。
    第一次：e0=e0_sigma、α=alpha_sigma、max_iters=max_iters_sigma；
    之后最多 MAX_SUM_SQ_ROUNDS 轮 Σy²：第 r 轮用 alpha_sum_sq[r]、max_iters_sum_sq[r]。
    """
    if len(alpha_sum_sq) < MAX_SUM_SQ_ROUNDS or len(max_iters_sum_sq) < MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"alpha_sum_sq / max_iters_sum_sq 长度须 ≥ {MAX_SUM_SQ_ROUNDS}"
        )
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)
    n2 = _check_power_of_two("delta2", d2)
    if n2 > MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"delta2={d2} 对应 {n2} 轮 Σy²，超过暂定上限 MAX_SUM_SQ_ROUNDS="
            f"{MAX_SUM_SQ_ROUNDS}"
        )

    x_scaled = x_work / d1 / d2 / THOR_INPUT_SCALE
    device, dtype = x.device, x.dtype
    p = _THOR_P.to(device=device, dtype=dtype)

    exp_approx = torch.zeros_like(x_scaled)
    for coeff in p:
        exp_approx = exp_approx * x_scaled + coeff

    scale = math.exp(shift / d2 / d1)
    exp_approx = exp_approx / scale

    for _ in range(n1):
        exp_approx = exp_approx**2

    exp_approx = exp_approx * m
    sigma_exp = exp_approx.sum(dim=-1)
    inv_sigma, en, iters_sigma = asor_inverse_batched(
        sigma_exp,
        e0_sigma,
        alpha=alpha_sigma,
        max_iters=max_iters_sigma,
    )
    y = exp_approx * inv_sigma.unsqueeze(-1)

    iters_sum_sq: list[int] = []
    e0_sum_sq: list[float] = []
    alphas_used: list[float] = []
    max_iters_used: list[int] = []
    for r in range(n2):
        e0_2 = thor_eps2_from_en(en)
        e0_sum_sq.append(e0_2)
        alpha_r = float(alpha_sum_sq[r])
        max_r = int(max_iters_sum_sq[r])
        alphas_used.append(alpha_r)
        max_iters_used.append(max_r)
        y_squared = y**2
        sum_y_sq = (y_squared * m).sum(dim=-1)
        inv_sum_sq, en, iters_2 = asor_inverse_batched(
            sum_y_sq,
            e0_2,
            alpha=alpha_r,
            max_iters=max_r,
        )
        iters_sum_sq.append(iters_2)
        y = y_squared * inv_sum_sq.unsqueeze(-1)

    meta = {
        "e0_sigma": float(e0_sigma),
        "alpha_sigma": float(alpha_sigma),
        "alpha_sum_sq": (float(alpha_sum_sq[0]), float(alpha_sum_sq[1])),
        "max_iters_sigma": int(max_iters_sigma),
        "max_iters_sum_sq": (int(max_iters_sum_sq[0]), int(max_iters_sum_sq[1])),
        "alphas_sum_sq_used": alphas_used,
        "max_iters_sum_sq_used": max_iters_used,
        "iters_sigma": int(iters_sigma),
        "en_after_sigma": float(
            asor_final_en(e0_sigma, alpha_sigma, max_iters_sigma)
        ),
        "e0_sum_sq": e0_sum_sq,
        "iters_sum_sq": iters_sum_sq,
        "en_final": float(en),
    }
    return y * m, meta


def standard_softmax_batched(x: torch.Tensor) -> torch.Tensor:
    """x: [N, L]"""
    x_max = x.amax(dim=-1, keepdim=True)
    ex = torch.exp(x - x_max)
    denom = ex.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    return ex / denom


def scores_to_valid_mask_torch(scores: torch.Tensor) -> torch.Tensor:
    return (scores > SCORE_VALID_THRESHOLD).to(dtype=scores.dtype)


class GpuScaledScoreCollector:
    """在 device 上收集 [N, L] 分数与掩码。"""

    def __init__(self, layer_enable: list[bool], device: torch.device):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.device = device
        self.layer_enable = layer_enable
        self._scores: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(NUM_LAYERS) if layer_enable[i]
        }

    def add_scores(self, layer_idx: int, scores: torch.Tensor) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = scores.detach().reshape(-1, scores.shape[-1]).to(
            self.device, dtype=torch.float32
        )
        self._scores[layer_idx].append(rows)

    def stacked(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self._scores[layer_idx]
        if not parts:
            z = torch.zeros(0, 0, device=self.device, dtype=torch.float32)
            return z, z
        scores = torch.cat(parts, dim=0)
        masks = scores_to_valid_mask_torch(scores)
        return scores, masks

    def vector_counts(self) -> dict[int, int]:
        return {i: self.stacked(i)[0].shape[0] for i in self._scores}

    def release_layer(self, layer_idx: int) -> None:
        self._scores.pop(layer_idx, None)
        torch.cuda.empty_cache()


def register_scaled_score_hooks(model, collector: GpuScaledScoreCollector) -> list:
    hooks = []

    for layer_idx in range(NUM_LAYERS):
        if not collector.layer_enable[layer_idx]:
            continue

        layer = model.bert.encoder.layer[layer_idx]
        attn_self = layer.attention.self
        orig_forward = attn_self.forward

        def make_patched(orig_fn, lidx, coll, attn_module):
            @functools.wraps(orig_fn)
            def patched_forward(
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                **kwargs,
            ):
                mixed_query = attn_module.query(hidden_states)
                key_in = (
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )
                mixed_key = attn_module.key(key_in)

                head_dim = attn_module.attention_head_size
                num_heads = attn_module.num_attention_heads

                def transpose_for_scores(t):
                    new_shape = t.size()[:-1] + (num_heads, head_dim)
                    t = t.view(new_shape)
                    return t.permute(0, 2, 1, 3)

                query_layer = transpose_for_scores(mixed_query)
                key_layer = transpose_for_scores(mixed_key)

                attention_scores = torch.matmul(
                    query_layer, key_layer.transpose(-1, -2)
                )
                attention_scores = attention_scores / (head_dim**0.5)

                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]
                    attention_scores = attention_scores + attention_mask

                coll.add_scores(lidx, attention_scores)

                return orig_fn(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            return patched_forward

        attn_self.forward = make_patched(
            orig_forward, layer_idx, collector, attn_self
        )
        hooks.append((attn_self, orig_forward))

    return hooks


def restore_hooks(hooks: list) -> None:
    for attn_self, orig_forward in hooks:
        attn_self.forward = orig_forward


def get_preprocess_fn(task_name: str, tokenizer):
    if task_name == "sst2":

        def fn(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )

    else:

        def fn(examples):
            return tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )

    return fn


def select_validation_split(dataset, n: int | None, seed: int):
    if n is None:
        return dataset, len(dataset), "all"
    n = min(n, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return dataset.select(indices), n, "random_subset"


def _subsample_rows_gpu(
    scores: torch.Tensor,
    masks: torch.Tensor,
    max_rows: int | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = scores.shape[0]
    if max_rows is None or n <= max_rows:
        return scores, masks
    g = torch.Generator(device=scores.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=scores.device)[:max_rows]
    return scores[idx], masks[idx]


@torch.no_grad()
def evaluate_thor_vs_softmax_layer_gpu(
    scores: torch.Tensor,
    masks: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    e0_sigma: float,
    alpha_sigma: float,
    alpha_sum_sq: tuple[float, float] | list[float],
    max_iters_sigma: int,
    max_iters_sum_sq: tuple[int, int] | list[int],
    chunk_size: int,
    max_rows: int | None,
    seed: int,
) -> dict:
    scores, masks = _subsample_rows_gpu(scores, masks, max_rows, seed)
    n = scores.shape[0]
    alpha_pair = (float(alpha_sum_sq[0]), float(alpha_sum_sq[1]))
    max_pair = (int(max_iters_sum_sq[0]), int(max_iters_sum_sq[1]))
    if n == 0:
        return {
            "max_abs_err": float("nan"),
            "mean_abs_err": float("nan"),
            "num_vectors": 0,
            "iters_sigma": 0,
            "iters_sum_sq": [],
            "e0_sum_sq": [],
            "alpha_sigma": float(alpha_sigma),
            "alpha_sum_sq": alpha_pair,
            "max_iters_sigma": int(max_iters_sigma),
            "max_iters_sum_sq": max_pair,
        }

    max_abs = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    sum_abs = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    count = 0
    meta_ref: dict | None = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = scores[start:end]
        m = masks[start:end]

        y_ref = standard_softmax_batched(x)
        y_thor, meta = thor_softmax_batched(
            x,
            shift,
            delta1,
            delta2,
            m,
            e0_sigma,
            alpha_sigma,
            alpha_pair,
            max_iters_sigma,
            max_pair,
        )
        if meta_ref is None:
            meta_ref = meta

        valid = m > 0
        err = torch.abs(y_ref - y_thor)[valid]
        if err.numel() == 0:
            continue
        max_abs = torch.maximum(max_abs, err.max())
        sum_abs = sum_abs + err.sum()
        count += int(err.numel())

    if meta_ref is None:
        meta_ref = {
            "iters_sigma": count_asor_inv_iters(
                e0_sigma, alpha_sigma, max_iters_sigma
            ),
            "iters_sum_sq": [],
            "e0_sum_sq": [],
            "alpha_sigma": float(alpha_sigma),
            "alpha_sum_sq": alpha_pair,
            "max_iters_sigma": int(max_iters_sigma),
            "max_iters_sum_sq": max_pair,
        }

    out = {
        "iters_sigma": meta_ref["iters_sigma"],
        "iters_sum_sq": meta_ref["iters_sum_sq"],
        "e0_sum_sq": meta_ref["e0_sum_sq"],
        "alpha_sigma": float(alpha_sigma),
        "alpha_sum_sq": alpha_pair,
        "max_iters_sigma": int(max_iters_sigma),
        "max_iters_sum_sq": max_pair,
        "num_vectors": n,
    }
    if count == 0:
        out["max_abs_err"] = float("nan")
        out["mean_abs_err"] = float("nan")
        return out

    out["max_abs_err"] = float(max_abs.cpu())
    out["mean_abs_err"] = float((sum_abs / count).cpu())
    out["num_elems"] = count
    return out


def analyze_thor_softmax_errors_gpu(
    task_name: str,
    collector: GpuScaledScoreCollector,
    layer_shifts: list[float],
    layer_divs: list[float],
    layer_alpha_sigma: list[float],
    layer_alpha_sum_sq: list[float],
    layer_alpha_sum_sq2: list[float],
    layer_max_iters_sigma: list[int],
    layer_max_iters_sum_sq: list[int],
    layer_max_iters_sum_sq2: list[int],
    min_sigmas: list[float],
    eval_device: torch.device,
    *,
    sigma_only: bool = False,
) -> dict[int, dict]:
    results: dict[int, dict] = {}
    min_sigmas = _validate_min_sigma_list(task_name, min_sigmas)

    for layer_idx in sorted(list(collector._scores.keys())):
        scores, masks = collector.stacked(layer_idx)
        shift = layer_shifts[layer_idx]
        delta2 = layer_divs[layer_idx]
        alpha_sigma = layer_alpha_sigma[layer_idx]
        alpha_pair = (
            float(layer_alpha_sum_sq[layer_idx]),
            float(layer_alpha_sum_sq2[layer_idx]),
        )
        max_iters_sigma = layer_max_iters_sigma[layer_idx]
        max_pair = (
            int(layer_max_iters_sum_sq[layer_idx]),
            int(layer_max_iters_sum_sq2[layer_idx]),
        )
        min_sigma = float(min_sigmas[layer_idx])
        e0_sigma = e0_sigma_from_min(min_sigma)

        if sigma_only:
            results[layer_idx] = {
                "shift": shift,
                "delta1": DEFAULT_THOR_DELTA1,
                "delta2": delta2,
                "min_sigma": min_sigma,
                "e0_sigma": e0_sigma,
                "alpha_sigma": alpha_sigma,
                "alpha_sum_sq": alpha_pair,
                "max_iters_sigma": max_iters_sigma,
                "max_iters_sum_sq": max_pair,
                "num_vectors": int(scores.shape[0]),
            }
            collector.release_layer(layer_idx)
            continue

        err = evaluate_thor_vs_softmax_layer_gpu(
            scores,
            masks,
            shift,
            DEFAULT_THOR_DELTA1,
            delta2,
            e0_sigma,
            alpha_sigma,
            alpha_pair,
            max_iters_sigma,
            max_pair,
            ERROR_EVAL_CHUNK_SIZE,
            MAX_VECTORS_FOR_ERROR_EVAL,
            RANDOM_SEED + layer_idx,
        )
        depth = softmax_layer_depth_asor(
            delta2, err["iters_sigma"], err["iters_sum_sq"]
        )
        gsq_str = (
            ",".join(str(x) for x in err["iters_sum_sq"])
            if err["iters_sum_sq"]
            else "-"
        )
        results[layer_idx] = {
            **err,
            "shift": shift,
            "delta1": DEFAULT_THOR_DELTA1,
            "delta2": delta2,
            "min_sigma": min_sigma,
            "e0_sigma": e0_sigma,
            "gsq_iters_str": gsq_str,
            "depth": depth,
        }
        collector.release_layer(layer_idx)

    if sigma_only:
        print(f"\n--- {task_name}  min(σ)（来自已加载/新统计，未跑误差）---")
        print(
            f"{'层':>3} {'shift':>7} {'d2':>4} {'minσ':>12} {'e0=min/÷':>12} {'n':>10}"
        )
        for layer_idx in sorted(results.keys()):
            st = results[layer_idx]
            print(
                f"{layer_idx:3d} {st['shift']:7.4g} {st['delta2']:4g} "
                f"{st['min_sigma']:12.4e} {st['e0_sigma']:12.4e} "
                f"{st['num_vectors']:10d}"
            )
        return results

    print(f"\n--- {task_name}  thor_softmax(aSOR) vs standard_softmax [GPU] ---")
    cap = f"最多 {MAX_VECTORS_FOR_ERROR_EVAL} 条" if MAX_VECTORS_FOR_ERROR_EVAL else "全部"
    print(
        f"    评估向量: {cap}, chunk={ERROR_EVAL_CHUNK_SIZE}  "
        f"e0_σ=minσ/{SIGMA_E0_DIVISOR:g}  e0_Σy²=en/{MAX_SEQ_LENGTH}/2  "
        f"停条件: en≥1−α 或 iters≥max_iters"
    )
    print(
        f"{'层':>3} {'shift':>7} {'d2':>4} "
        f"{'itσ':>4} {'itΣy²':>8} {'depth':>5} "
        f"{'max_abs_err':>12} {'mean_abs_err':>12}"
    )
    for layer_idx in sorted(results.keys()):
        st = results[layer_idx]
        print(
            f"{layer_idx:3d} {st['shift']:7.4g} {st['delta2']:4g} "
            f"{st['iters_sigma']:4d} {st['gsq_iters_str']:>8} {st['depth']:5d} "
            f"{st['max_abs_err']:12.6g} {st['mean_abs_err']:12.6g}"
        )

    return results


def collect_min_sigma_for_task(
    task_name: str,
    collector: GpuScaledScoreCollector,
    layer_shifts: list[float],
    layer_divs: list[float],
) -> tuple[list[float], list[float]]:
    """遍历已收集分数，统计每层 min/max(σ)。不释放 collector。"""
    min_sigmas: list[float] = []
    max_sigmas: list[float] = []
    print(f"\n--- {task_name}  统计 σ=Σexp ---")
    print(f"{'层':>3} {'shift':>7} {'d2':>4} {'minσ':>12} {'maxσ':>12} {'n':>10}")
    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in collector._scores:
            raise ValueError(f"{task_name} 层{layer_idx} 无分数，无法统计 σ")
        scores, masks = collector.stacked(layer_idx)
        mn, mx, n = measure_sigma_range(
            scores,
            masks,
            layer_shifts[layer_idx],
            DEFAULT_THOR_DELTA1,
            layer_divs[layer_idx],
            ERROR_EVAL_CHUNK_SIZE,
        )
        min_sigmas.append(mn)
        max_sigmas.append(mx)
        print(
            f"{layer_idx:3d} {layer_shifts[layer_idx]:7.4g} {layer_divs[layer_idx]:4g} "
            f"{mn:12.4e} {mx:12.4e} {n:10d}"
        )
    return min_sigmas, max_sigmas


def run_task(
    task_name: str,
    device: torch.device,
    *,
    recompute_sigma: bool = False,
    sigma_only: bool = False,
    json_dir: str = JSON_DIR,
    num_samples: int | None = None,
) -> dict:
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    if num_samples is None:
        num_samples = NUM_SAMPLES
    val_set, _, _ = select_validation_split(
        dataset["validation"], num_samples, RANDOM_SEED
    )
    tokenized = val_set.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=val_set.column_names,
        desc=f"tokenize {task_name}",
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

    collector = GpuScaledScoreCollector(LAYER_ENABLE, device)
    hooks = register_scaled_score_hooks(model, collector)

    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)

    restore_hooks(hooks)

    layer_shifts = layer_exp_shift_for_task(task_name)
    layer_divs = layer_exp_div_for_task(task_name)
    layer_alpha_sigma = layer_asor_alpha_sigma_for_task(task_name)
    layer_alpha_sum_sq = layer_asor_alpha_sum_sq_for_task(task_name)
    layer_alpha_sum_sq2 = layer_asor_alpha_sum_sq2_for_task(task_name)
    layer_max_iters_sigma = layer_asor_max_iters_sigma_for_task(task_name)
    layer_max_iters_sum_sq = layer_asor_max_iters_sum_sq_for_task(task_name)
    layer_max_iters_sum_sq2 = layer_asor_max_iters_sum_sq2_for_task(task_name)

    max_sigmas: list[float] | None = None
    if recompute_sigma:
        min_sigmas, max_sigmas = collect_min_sigma_for_task(
            task_name, collector, layer_shifts, layer_divs
        )
        path = save_min_sigma_json(
            task_name, min_sigmas, json_dir, max_sigma=max_sigmas
        )
        sigma_source = path
        print(f"  已写入 {path}")
    else:
        min_sigmas, sigma_source = load_min_sigma_from_json(task_name, json_dir)
        print(f"  {task_name}: 已加载 min(σ) ← {sigma_source}")

    results = analyze_thor_softmax_errors_gpu(
        task_name,
        collector,
        layer_shifts,
        layer_divs,
        layer_alpha_sigma,
        layer_alpha_sum_sq,
        layer_alpha_sum_sq2,
        layer_max_iters_sigma,
        layer_max_iters_sum_sq,
        layer_max_iters_sum_sq2,
        min_sigmas,
        device,
        sigma_only=sigma_only,
    )
    return {
        "min_sigma": min_sigmas,
        "max_sigma": max_sigmas,
        "sigma_source": sigma_source,
        "results": results,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="thor Softmax aSOR 误差评估；min(σ) 读写 sigma_out/"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help="验证子集大小；默认配置区 NUM_SAMPLES（全量）",
    )
    parser.add_argument(
        "--json-dir",
        default=JSON_DIR,
        help=f"σ JSON 目录（默认 {JSON_DIR}）",
    )
    parser.add_argument(
        "--recompute-sigma",
        action="store_true",
        help="重新统计 min(σ) 并写入 JSON；默认从 JSON/缓存加载",
    )
    parser.add_argument(
        "--sigma-only",
        action="store_true",
        help="只处理 min(σ)（配合 --recompute-sigma 可只写 JSON）",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA，将使用 CPU（仍走批量 torch，但无 GPU 加速）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("eval_softmax_cuda.py — GPU 批量 thor_softmax (aSOR)")
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"σ JSON 目录：{args.json_dir}")
    print(
        f"min(σ)：{'重新统计' if args.recompute_sigma else '从 JSON/缓存加载'}  "
        f"e0_σ=minσ/{SIGMA_E0_DIVISOR:g}  e0_Σy²=en/{MAX_SEQ_LENGTH}/2"
    )
    if args.sigma_only and not args.recompute_sigma:
        print("提示：--sigma-only 未加 --recompute-sigma 时仅打印已加载的 min(σ)")

    for task_name in args.tasks:
        try:
            layer_exp_shift_for_task(task_name)
            layer_exp_div_for_task(task_name)
            layer_asor_alpha_sigma_for_task(task_name)
            layer_asor_alpha_sum_sq_for_task(task_name)
            layer_asor_alpha_sum_sq2_for_task(task_name)
            layer_asor_max_iters_sigma_for_task(task_name)
            layer_asor_max_iters_sum_sq_for_task(task_name)
            layer_asor_max_iters_sum_sq2_for_task(task_name)
            run_task(
                task_name,
                device,
                recompute_sigma=args.recompute_sigma,
                sigma_only=args.sigma_only,
                json_dir=args.json_dir,
                num_samples=args.samples,
            )
        except FileNotFoundError as e:
            print(f"跳过 {task_name}：{e}")

    print("完成。")


if __name__ == "__main__":
    main()
