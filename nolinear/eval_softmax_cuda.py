"""
eval_softmax 的 GPU 批量版：thor_softmax + aSOR 倒数。

  - 第一次倒数：e0 = min(σ)/SIGMA_E0_DIVISOR
  - 后续 δ2 轮：e0 = 上一轮 aSOR 结束时的 en / MAX_SEQ_LENGTH / 2（THOR 公式）
  - 停止：en ≥ 1−α **或** iters ≥ max_iters（二者同时限制）
  - α / max_iters：本文件按 task×层调参（σ 一轮；Σy² 最多两轮）
  - shift / δ2 / 多项式系数 / THOR 常量：统一来自 ../softmax_poly.py（勿在此重复维护）
  - min(σ)/max(σ)：calib∪validation 并集写入 sigma_out/{task}_sigma.json
    （min 取更小、max 取更大；接口供 softmax_poly 复用）
  - mnli / qnli：边 hook 边统计/评估（Streaming*Tracker），不缓存全部 attention scores

用法（在 nolinear/ 下）：
  python3 eval_softmax_cuda.py                         # 默认 all：校验集+验证集
  python3 eval_softmax_cuda.py --eval-split validation # 仅验证集
  python3 eval_softmax_cuda.py --eval-split calib      # 仅校验集
  python3 eval_softmax_cuda.py --recompute-sigma       # 两侧统计并集 → {task}_sigma.json
  python3 eval_softmax_cuda.py --sigma-only --recompute-sigma  # 仅统计

  --eval-split all：先 calib 再 validation，误差表每组两行（默认）
  --eval-split validation：官方 validation（可用 --samples 子采样）
  --eval-split calib：仅使用 coverage_metric 已写入的校验集索引 JSON
    （results/coverage_metrics/{task}_calib_indices_seed{seed}.json；禁止现场重建）
  aSOR 用的 min(σ)：始终读并集 sigma_out/{task}_sigma.json
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from softmax_poly import (  # noqa: E402
    LAYER_EXP_DIV_BY_TASK,
    LAYER_EXP_SHIFT_BY_TASK,
    MAX_SEQ_LENGTH as POLY_MAX_SEQ_LENGTH,
    MAX_SUM_SQ_ROUNDS as POLY_MAX_SUM_SQ_ROUNDS,
    NUM_LAYERS as POLY_NUM_LAYERS,
    SIGMA_E0_DIVISOR,
    THOR_DELTA1,
    THOR_EXP_POLY_COEFFS,
    THOR_INPUT_SCALE,
    _check_power_of_two,
)

# ===================== 配置（可独立修改）=====================
#TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
TASK_NAMES = ["mnli"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = POLY_MAX_SEQ_LENGTH
NUM_LAYERS = POLY_NUM_LAYERS
TASK_NUM_LABELS: dict[str, int] = {
    "mrpc": 2,
    "rte": 2,
    "sst2": 2,
    "cola": 2,
    "qnli": 2,
    "mnli": 3,
}

NUM_SAMPLES = None
RANDOM_SEED = 42
BATCH_SIZE = 128

# 校验集：只读 coverage_metric 落盘索引（与文件内容完全一致，不重建）
DEFAULT_EVAL_SPLIT = "all"  # "all" | "validation" | "calib"
CALIB_SEED = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_INDICES_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "results", "coverage_metrics")
)

SCORE_VALID_THRESHOLD = -1e4

# aSOR 调参：停在 en ≥ 1−α 或 iters ≥ max_iters（二者同时限制）
# 第一次 e0 = minσ / SIGMA_E0_DIVISOR（常数来自 softmax_poly）
# α / max_iters 仅在本文件维护；调完后把 max_iters 同步到 softmax_poly 的 high 档
DEFAULT_ASOR_MAX_ITERS = 20
MAX_SUM_SQ_ROUNDS = POLY_MAX_SUM_SQ_ROUNDS

JSON_DIR = os.path.join(_SCRIPT_DIR, "sigma_out")

# task -> 12 层 min(σ)；load/recompute 后填充，供同进程复用
_MIN_SIGMA_CACHE: dict[str, list[float]] = {}


_ASOR_ALPHA_ONES = [0.00001] * NUM_LAYERS
LAYER_ASOR_ALPHA_SIGMA_BY_TASK: dict[str, list[float]] = {
    t: list(_ASOR_ALPHA_ONES) for t in TASK_NUM_LABELS
}
# Σy² 第 1 / 第 2 轮（d2=2 只用第 1 轮；d2=4 用两轮）
LAYER_ASOR_ALPHA_SUM_SQ_BY_TASK: dict[str, list[float]] = {
    t: list(_ASOR_ALPHA_ONES) for t in TASK_NUM_LABELS
}
LAYER_ASOR_ALPHA_SUM_SQ2_BY_TASK: dict[str, list[float]] = {
    t: list(_ASOR_ALPHA_ONES) for t in TASK_NUM_LABELS
}

LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK: dict[str, list[int]] = {
        # #"mrpc": [3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5],
        # #"mrpc": [4, 4, 4, 5, 4, 4, 4, 4, 6, 6, 5, 5],
        # "mrpc": [4, 5, 5, 5, 5, 5, 5, 5, 7, 7, 6, 6],

        # #"rte": [4, 6, 4, 5, 4, 5, 4, 5, 5, 5, 4, 4],
        # #"rte": [5, 7, 4, 6, 5, 6, 5, 6, 6, 6, 5, 5],
        # "rte": [5, 7, 4, 6, 5, 6, 5, 6, 6, 6, 5, 5],

        # #"sst2": [4, 4, 4, 5, 4, 4, 5, 5, 5, 5, 8, 5],
        # #"sst2": [5, 5, 5, 6, 5, 5, 6, 5, 6, 5, 9, 6],
        # "sst2": [5, 5, 5, 7, 5, 5, 6, 6, 6, 6, 9, 6],
        # "cola": [5, 6, 4, 6, 6, 6, 6, 6, 6, 6, 9, 7],
        # "qnli": [5, 6, 4, 7, 6, 6, 6, 6, 7, 6, 7 ,6],


        "mnli": [5, 7, 5, 8, 7, 7, 7, 7, 7, 7, 8 ,6],
}
LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK: dict[str, list[int]] = {
    # #"mrpc": [5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    # #"mrpc": [5, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6],
    # "mrpc": [6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6],

    # #"rte": [5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    # #"rte": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    # "rte": [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],

    # #"sst2": [5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6, 5],
    # #"sst2": [5, 5, 5, 6, 5, 5, 5, 6, 5, 6, 6, 5],
    # "sst2": [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 7, 6],
    # "cola": [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    # "qnli": [6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],

    "mnli": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],

}
LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK: dict[str, list[int]] = {
    # "mrpc": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "rte": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "sst2": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "cola": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "qnli": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],

    "mnli": [0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}
# LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK["mnli"]= [x-1  for x in LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK["mnli"]]
# LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK["mnli"] = [x-1  for x in LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK["mnli"]]
LAYER_ENABLE = [True, True, True, True, True, True, 
                True, True, True, True, True, True]
MAX_VECTORS_FOR_ERROR_EVAL: int | None = None
ERROR_EVAL_CHUNK_SIZE = 4096
REL_ERR_DENOM_MIN = 1e-4  # |y_ref| 低于此值不参与相对误差



# Stockmeyer(4) + δ1 square(1)；aSOR 每步 HE rem=1（与 cost.SOFTMAX_ASOR_ITER_DEPTH 一致）
SOFTMAX_DEPTH_BASE = 5
try:
    from cost import SOFTMAX_ASOR_ITER_DEPTH as _ASOR_D
except ImportError:  # pragma: no cover
    _ASOR_D = 1
SOFTMAX_ASOR_ITER_DEPTH = int(_ASOR_D)
_THOR_P = torch.tensor(THOR_EXP_POLY_COEFFS, dtype=torch.float32)
# =============================================================================


def task_num_labels(task_name: str) -> int:
    if task_name not in TASK_NUM_LABELS:
        raise KeyError(f"未知任务 {task_name}，请在 TASK_NUM_LABELS 中配置")
    return TASK_NUM_LABELS[task_name]


def layer_exp_shift_for_task(task_name: str) -> list[float]:
    if task_name not in LAYER_EXP_SHIFT_BY_TASK:
        raise KeyError(
            f"任务 {task_name} 未配置 softmax_poly.LAYER_EXP_SHIFT_BY_TASK"
        )
    shifts = LAYER_EXP_SHIFT_BY_TASK[task_name]
    if len(shifts) != NUM_LAYERS:
        raise ValueError(f"{task_name} shift 长度应为 {NUM_LAYERS}")
    return shifts


def layer_exp_div_for_task(task_name: str) -> list[float]:
    if task_name not in LAYER_EXP_DIV_BY_TASK:
        raise KeyError(
            f"任务 {task_name} 未配置 softmax_poly.LAYER_EXP_DIV_BY_TASK"
        )
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


def merge_sigma_lists(
    a: list[float], b: list[float], *, how: str
) -> list[float]:
    """how='min' → 各层取更小；how='max' → 各层取更大。"""
    if len(a) != len(b):
        raise ValueError(f"merge_sigma_lists 长度不一致：{len(a)} vs {len(b)}")
    if how == "min":
        return [min(float(x), float(y)) for x, y in zip(a, b)]
    if how == "max":
        return [max(float(x), float(y)) for x, y in zip(a, b)]
    raise ValueError(f"未知 how={how!r}")


def save_min_sigma_json(
    task_name: str,
    min_sigma: list[float],
    json_dir: str = JSON_DIR,
    *,
    max_sigma: list[float] | None = None,
    sources: list[str] | None = None,
) -> str:
    """写入并集 σ JSON {task}_sigma.json，并更新内存缓存。"""
    min_sigma = _validate_min_sigma_list(task_name, min_sigma)
    os.makedirs(json_dir, exist_ok=True)
    path = sigma_json_path(task_name, json_dir)
    src = list(sources) if sources is not None else ["validation", "calib"]
    payload: dict = {
        "task": task_name,
        "eval_split": "union",
        "sources": src,
        "min_sigma": min_sigma,
        "sigma_e0_divisor": SIGMA_E0_DIVISOR,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "min_sigma/max_sigma 为 calib∪validation "
            "（min 取更小、max 取更大）"
        ),
    }
    if max_sigma is not None:
        if len(max_sigma) != NUM_LAYERS:
            raise ValueError(f"{task_name} max_sigma 长度应为 {NUM_LAYERS}")
        payload["max_sigma"] = [float(x) for x in max_sigma]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _MIN_SIGMA_CACHE[task_name] = list(min_sigma)
    return path


def load_min_sigma_from_json(
    task_name: str,
    json_dir: str = JSON_DIR,
    *,
    use_cache: bool = True,
) -> tuple[list[float], str]:
    """读并集 min(σ)；优先内存缓存。返回 (min_sigma[12], source)。"""
    if use_cache and task_name in _MIN_SIGMA_CACHE:
        return list(_MIN_SIGMA_CACHE[task_name]), f"cache:{task_name}"

    path = sigma_json_path(task_name, json_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"σ JSON 不存在：{path}；请先运行 "
            "`python3 eval_softmax_cuda.py --recompute-sigma`"
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
    """单层深度：base + aSOR_σ×d + Σ(各轮 aSOR_Σy²×d)；d=SOFTMAX_ASOR_ITER_DEPTH。"""
    d = SOFTMAX_ASOR_ITER_DEPTH
    return (
        SOFTMAX_DEPTH_BASE
        + int(iters_sigma) * d
        + sum(int(x) for x in iters_sum_sq) * d
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


def _compute_device_for(t: torch.Tensor) -> torch.device:
    if t.is_cuda:
        return t.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def measure_sigma_range(
    scores: torch.Tensor,
    masks: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    chunk_size: int,
) -> tuple[float, float, int]:
    """返回 (min_σ, max_σ, n_vectors)。分数可在 CPU；按 chunk 搬到 GPU 计算。"""
    n = scores.shape[0]
    if n == 0:
        return float("nan"), float("nan"), 0
    device = _compute_device_for(scores)
    min_sigma = float("inf")
    max_sigma = float("-inf")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sc = scores[start:end].to(device, non_blocking=True)
        mk = masks[start:end].to(device, non_blocking=True)
        sigma = compute_sigma_exp_batched(sc, shift, delta1, delta2, mk)
        min_sigma = min(min_sigma, float(sigma.min().item()))
        max_sigma = max(max_sigma, float(sigma.max().item()))
    return min_sigma, max_sigma, n


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
    """收集全部 [N, L] 分数与掩码（小任务用）。大任务请用 Streaming*Tracker。"""

    def __init__(
        self,
        layer_enable: list[bool],
        device: torch.device,
    ):
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
            dtype=torch.float32
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
        return {
            i: sum(int(p.shape[0]) for p in parts)
            for i, parts in self._scores.items()
        }

    def release_layer(self, layer_idx: int) -> None:
        self._scores.pop(layer_idx, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class StreamingSigmaTracker:
    """边 hook 边更新每层 min/max(σ)，不缓存 attention scores（mnli 等大集）。"""

    def __init__(
        self,
        layer_enable: list[bool],
        layer_shifts: list[float],
        layer_divs: list[float],
        *,
        chunk_size: int = ERROR_EVAL_CHUNK_SIZE,
    ):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.layer_enable = layer_enable
        self.layer_shifts = layer_shifts
        self.layer_divs = layer_divs
        self.chunk_size = int(chunk_size)
        self.min_sigma = [
            float("inf") if layer_enable[i] else float("nan")
            for i in range(NUM_LAYERS)
        ]
        self.max_sigma = [
            float("-inf") if layer_enable[i] else float("nan")
            for i in range(NUM_LAYERS)
        ]
        self.n_vectors = [0] * NUM_LAYERS

    @torch.no_grad()
    def add_scores(self, layer_idx: int, scores: torch.Tensor) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = scores.detach().reshape(-1, scores.shape[-1]).to(dtype=torch.float32)
        n = int(rows.shape[0])
        if n == 0:
            return
        shift = float(self.layer_shifts[layer_idx])
        delta2 = float(self.layer_divs[layer_idx])
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            sc = rows[start:end]
            mk = scores_to_valid_mask_torch(sc)
            sigma = compute_sigma_exp_batched(sc, shift, THOR_DELTA1, delta2, mk)
            self.min_sigma[layer_idx] = min(
                self.min_sigma[layer_idx], float(sigma.min().item())
            )
            self.max_sigma[layer_idx] = max(
                self.max_sigma[layer_idx], float(sigma.max().item())
            )
            self.n_vectors[layer_idx] += int(sigma.numel())


class StreamingErrorTracker:
    """边 hook 边累加 thor vs softmax 误差（或仅计向量数），不缓存 scores。"""

    def __init__(
        self,
        layer_enable: list[bool],
        layer_shifts: list[float],
        layer_divs: list[float],
        layer_alpha_sigma: list[float],
        layer_alpha_sum_sq: list[float],
        layer_alpha_sum_sq2: list[float],
        layer_max_iters_sigma: list[int],
        layer_max_iters_sum_sq: list[int],
        layer_max_iters_sum_sq2: list[int],
        min_sigmas: list[float],
        *,
        sigma_only: bool = False,
        chunk_size: int = ERROR_EVAL_CHUNK_SIZE,
    ):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.layer_enable = layer_enable
        self.layer_shifts = layer_shifts
        self.layer_divs = layer_divs
        self.layer_alpha_sigma = layer_alpha_sigma
        self.layer_alpha_sum_sq = layer_alpha_sum_sq
        self.layer_alpha_sum_sq2 = layer_alpha_sum_sq2
        self.layer_max_iters_sigma = layer_max_iters_sigma
        self.layer_max_iters_sum_sq = layer_max_iters_sum_sq
        self.layer_max_iters_sum_sq2 = layer_max_iters_sum_sq2
        self.min_sigmas = min_sigmas
        self.sigma_only = bool(sigma_only)
        self.chunk_size = int(chunk_size)
        self.n_vectors = [0] * NUM_LAYERS
        self._max_abs = [0.0] * NUM_LAYERS
        self._sum_abs = [0.0] * NUM_LAYERS
        self._err_count = [0] * NUM_LAYERS
        self._max_rel = [0.0] * NUM_LAYERS
        self._sum_rel = [0.0] * NUM_LAYERS
        self._rel_count = [0] * NUM_LAYERS
        self._meta: dict[int, dict] = {}

    @torch.no_grad()
    def add_scores(self, layer_idx: int, scores: torch.Tensor) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = scores.detach().reshape(-1, scores.shape[-1]).to(dtype=torch.float32)
        n = int(rows.shape[0])
        if n == 0:
            return
        self.n_vectors[layer_idx] += n
        if self.sigma_only:
            return

        shift = float(self.layer_shifts[layer_idx])
        delta2 = float(self.layer_divs[layer_idx])
        alpha_sigma = float(self.layer_alpha_sigma[layer_idx])
        alpha_pair = (
            float(self.layer_alpha_sum_sq[layer_idx]),
            float(self.layer_alpha_sum_sq2[layer_idx]),
        )
        max_iters_sigma = int(self.layer_max_iters_sigma[layer_idx])
        max_pair = (
            int(self.layer_max_iters_sum_sq[layer_idx]),
            int(self.layer_max_iters_sum_sq2[layer_idx]),
        )
        e0_sigma = e0_sigma_from_min(float(self.min_sigmas[layer_idx]))

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            x = rows[start:end]
            m = scores_to_valid_mask_torch(x)
            y_ref = standard_softmax_batched(x)
            y_thor, meta = thor_softmax_batched(
                x,
                shift,
                THOR_DELTA1,
                delta2,
                m,
                e0_sigma,
                alpha_sigma,
                alpha_pair,
                max_iters_sigma,
                max_pair,
            )
            if layer_idx not in self._meta:
                self._meta[layer_idx] = meta

            valid = m > 0
            if not valid.any():
                continue
            err = torch.abs(y_ref - y_thor)[valid]
            y_ref_v = y_ref[valid]
            self._max_abs[layer_idx] = max(
                self._max_abs[layer_idx], float(err.max().item())
            )
            self._sum_abs[layer_idx] += float(err.sum().item())
            self._err_count[layer_idx] += int(err.numel())

            y_ref_abs = torch.abs(y_ref_v)
            denom_ok = y_ref_abs >= REL_ERR_DENOM_MIN
            if denom_ok.any():
                rel_pct = err[denom_ok] / y_ref_abs[denom_ok] * 100.0
                self._max_rel[layer_idx] = max(
                    self._max_rel[layer_idx], float(rel_pct.max().item())
                )
                self._sum_rel[layer_idx] += float(rel_pct.sum().item())
                self._rel_count[layer_idx] += int(rel_pct.numel())

    def finalize(self, task_name: str) -> dict[int, dict]:
        min_sigmas = _validate_min_sigma_list(task_name, self.min_sigmas)
        results: dict[int, dict] = {}
        for layer_idx in range(NUM_LAYERS):
            if not self.layer_enable[layer_idx]:
                continue
            shift = float(self.layer_shifts[layer_idx])
            delta2 = float(self.layer_divs[layer_idx])
            alpha_sigma = float(self.layer_alpha_sigma[layer_idx])
            alpha_pair = (
                float(self.layer_alpha_sum_sq[layer_idx]),
                float(self.layer_alpha_sum_sq2[layer_idx]),
            )
            max_iters_sigma = int(self.layer_max_iters_sigma[layer_idx])
            max_pair = (
                int(self.layer_max_iters_sum_sq[layer_idx]),
                int(self.layer_max_iters_sum_sq2[layer_idx]),
            )
            min_sigma = float(min_sigmas[layer_idx])
            e0_sigma = e0_sigma_from_min(min_sigma)
            n = int(self.n_vectors[layer_idx])

            if self.sigma_only:
                results[layer_idx] = {
                    "shift": shift,
                    "delta1": THOR_DELTA1,
                    "delta2": delta2,
                    "min_sigma": min_sigma,
                    "e0_sigma": e0_sigma,
                    "alpha_sigma": alpha_sigma,
                    "alpha_sum_sq": alpha_pair,
                    "max_iters_sigma": max_iters_sigma,
                    "max_iters_sum_sq": max_pair,
                    "num_vectors": n,
                }
                continue

            meta = self._meta.get(layer_idx)
            if meta is None:
                meta = {
                    "iters_sigma": count_asor_inv_iters(
                        e0_sigma, alpha_sigma, max_iters_sigma
                    ),
                    "iters_sum_sq": [],
                    "e0_sum_sq": [],
                    "alpha_sigma": alpha_sigma,
                    "alpha_sum_sq": alpha_pair,
                    "max_iters_sigma": max_iters_sigma,
                    "max_iters_sum_sq": max_pair,
                }
            count = self._err_count[layer_idx]
            rel_count = self._rel_count[layer_idx]
            err = {
                "iters_sigma": meta["iters_sigma"],
                "iters_sum_sq": meta["iters_sum_sq"],
                "e0_sum_sq": meta["e0_sum_sq"],
                "alpha_sigma": alpha_sigma,
                "alpha_sum_sq": alpha_pair,
                "max_iters_sigma": max_iters_sigma,
                "max_iters_sum_sq": max_pair,
                "num_vectors": n,
            }
            if count == 0:
                err["max_abs_err"] = float("nan")
                err["mean_abs_err"] = float("nan")
                err["max_rel_pct"] = float("nan")
                err["mean_rel_pct"] = float("nan")
            else:
                err["max_abs_err"] = self._max_abs[layer_idx]
                err["mean_abs_err"] = self._sum_abs[layer_idx] / count
                err["num_elems"] = count
                err["max_rel_pct"] = (
                    self._max_rel[layer_idx] if rel_count > 0 else float("nan")
                )
                err["mean_rel_pct"] = (
                    self._sum_rel[layer_idx] / rel_count
                    if rel_count > 0
                    else float("nan")
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
                "delta1": THOR_DELTA1,
                "delta2": delta2,
                "min_sigma": min_sigma,
                "e0_sigma": e0_sigma,
                "gsq_iters_str": gsq_str,
                "depth": depth,
            }
        return results


def register_scaled_score_hooks(model, collector) -> list:
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
    if task_name in ("sst2", "cola"):

        def fn(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )

    elif task_name == "mnli":

        def fn(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )

    elif task_name == "qnli":

        def fn(examples):
            return tokenizer(
                examples["question"],
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


def _calib_indices_path(task_name: str, seed: int, indices_dir: str) -> str:
    return os.path.join(indices_dir, f"{task_name}_calib_indices_seed{seed}.json")


def load_calib_indices_json(
    task_name: str, seed: int, indices_dir: str
) -> tuple[list[int], dict]:
    """必须存在 coverage_metric 落盘索引；禁止现场重建。"""
    path = _calib_indices_path(task_name, seed, indices_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"校验集索引不存在：{path}；请先运行 coverage_metric.py 生成"
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "indices" not in payload or not payload["indices"]:
        raise ValueError(f"校验集索引为空或缺少 indices：{path}")
    indices = [int(i) for i in payload["indices"]]
    meta = {k: v for k, v in payload.items() if k != "indices"}
    meta["indices_path"] = path
    return indices, meta


def resolve_calib_eval_split(
    task_name: str,
    dataset,
    *,
    calib_seed: int,
    indices_dir: str,
):
    """从 train 取校验集：完整使用 coverage_metric JSON 中的全部 indices。"""
    if "train" not in dataset:
        raise KeyError(f"{task_name} 缺少 train，无法构造校验集")

    indices, meta = load_calib_indices_json(task_name, calib_seed, indices_dir)
    n_train = len(dataset["train"])
    bad = [i for i in indices if i < 0 or i >= n_train]
    if bad:
        raise IndexError(
            f"{task_name} 校验集索引越界（train n={n_train}），例：{bad[:5]}"
        )
    desc = (
        f"calib n={len(indices)} seed={calib_seed} "
        f"← {meta.get('indices_path')}"
        f" (file calib_size={meta.get('calib_size', '?')})"
    )
    return dataset["train"].select(indices), len(indices), desc


def resolve_eval_split(
    task_name: str,
    dataset,
    *,
    eval_split: str,
    num_samples: int | None,
    seed: int,
    calib_seed: int,
    indices_dir: str,
):
    """返回 (raw_split, n, 描述字符串)。"""
    if eval_split == "validation":
        split, n, mode = select_validation_split(
            dataset["validation"], num_samples, seed
        )
        desc = f"validation ({mode}, n={n})"
        return split, n, desc
    if eval_split == "calib":
        return resolve_calib_eval_split(
            task_name,
            dataset,
            calib_seed=calib_seed,
            indices_dir=indices_dir,
        )
    raise ValueError(
        f"未知 eval_split={eval_split!r}，可选 all / validation / calib"
    )


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
            "max_rel_pct": float("nan"),
            "mean_rel_pct": float("nan"),
            "num_vectors": 0,
            "iters_sigma": 0,
            "iters_sum_sq": [],
            "e0_sum_sq": [],
            "alpha_sigma": float(alpha_sigma),
            "alpha_sum_sq": alpha_pair,
            "max_iters_sigma": int(max_iters_sigma),
            "max_iters_sum_sq": max_pair,
        }

    device = _compute_device_for(scores)
    max_abs = torch.tensor(0.0, device=device, dtype=torch.float32)
    sum_abs = torch.tensor(0.0, device=device, dtype=torch.float32)
    max_rel = torch.tensor(0.0, device=device, dtype=torch.float32)
    sum_rel = torch.tensor(0.0, device=device, dtype=torch.float32)
    count = 0
    rel_count = 0
    meta_ref: dict | None = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = scores[start:end].to(device, non_blocking=True)
        m = masks[start:end].to(device, non_blocking=True)

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
        if not valid.any():
            continue
        err = torch.abs(y_ref - y_thor)[valid]
        y_ref_v = y_ref[valid]
        max_abs = torch.maximum(max_abs, err.max())
        sum_abs = sum_abs + err.sum()
        count += int(err.numel())

        y_ref_abs = torch.abs(y_ref_v)
        denom_ok = y_ref_abs >= REL_ERR_DENOM_MIN
        if denom_ok.any():
            rel_pct = err[denom_ok] / y_ref_abs[denom_ok] * 100.0
            max_rel = torch.maximum(max_rel, rel_pct.max())
            sum_rel = sum_rel + rel_pct.sum()
            rel_count += int(rel_pct.numel())

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
        out["max_rel_pct"] = float("nan")
        out["mean_rel_pct"] = float("nan")
        return out

    out["max_abs_err"] = float(max_abs.cpu())
    out["mean_abs_err"] = float((sum_abs / count).cpu())
    out["num_elems"] = count
    out["max_rel_pct"] = (
        float(max_rel.cpu()) if rel_count > 0 else float("nan")
    )
    out["mean_rel_pct"] = (
        float((sum_rel / rel_count).cpu()) if rel_count > 0 else float("nan")
    )
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
                "delta1": THOR_DELTA1,
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
            THOR_DELTA1,
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
            "delta1": THOR_DELTA1,
            "delta2": delta2,
            "min_sigma": min_sigma,
            "e0_sigma": e0_sigma,
            "gsq_iters_str": gsq_str,
            "depth": depth,
        }
        collector.release_layer(layer_idx)

    return results


def _fmt_rel_pct(v: float) -> str:
    return f"{v:10.4f}" if math.isfinite(v) else f"{'nan':>10}"


def _fmt_err_block(st: dict | None) -> str:
    if st is None:
        return f"{'nan':>12} {'nan':>12} {'nan':>10} {'nan':>10}"
    return (
        f"{st['max_abs_err']:12.6g} {st['mean_abs_err']:12.6g} "
        f"{_fmt_rel_pct(st['max_rel_pct'])} {_fmt_rel_pct(st['mean_rel_pct'])}"
    )


def _print_sigma_only_table(task_name: str, results: dict[int, dict]) -> None:
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


def _print_error_table(
    task_name: str,
    results: dict[int, dict],
    *,
    dual_results: dict[int, dict] | None = None,
) -> None:
    """dual 非空时：上行=校验集，下行=验证集（误差四列对齐）。"""
    print(f"\n--- {task_name}  thor_softmax(aSOR) vs standard_softmax [GPU] ---")
    cap = (
        f"最多 {MAX_VECTORS_FOR_ERROR_EVAL} 条"
        if MAX_VECTORS_FOR_ERROR_EVAL
        else "全部"
    )
    print(
        f"    评估向量: {cap}, chunk={ERROR_EVAL_CHUNK_SIZE}  "
        f"e0_σ=minσ/{SIGMA_E0_DIVISOR:g}  e0_Σy²=en/{MAX_SEQ_LENGTH}  "
        f"停条件: en≥1−α 或 iters≥max_iters"
    )
    print(
        f"    相对误差: |y−y_ref|/|y_ref|×100% "
        f"(仅统计 |y_ref|≥{REL_ERR_DENOM_MIN:g})"
    )
    err_hdr = (
        f"{'max_abs_err':>12} {'mean_abs_err':>12} "
        f"{'max_rel%':>10} {'mean_rel%':>10}"
    )
    meta_hdr = (
        f"{'层':>3} {'shift':>7} {'d2':>4} "
        f"{'itσ':>4} {'itΣy²':>8} {'depth':>5} "
    )
    val_indent = " " * len(meta_hdr)
    if dual_results is not None:
        print("    每组两行：上行=校验集(calib)，下行=验证集(validation)")
        print(f"{meta_hdr}{err_hdr}")
        keys = sorted(set(results.keys()) | set(dual_results.keys()))
        for layer_idx in keys:
            st_c = results.get(layer_idx)
            st_v = dual_results.get(layer_idx)
            st_meta = st_c or st_v
            assert st_meta is not None
            print(
                f"{layer_idx:3d} {st_meta['shift']:7.4g} {st_meta['delta2']:4g} "
                f"{st_meta['iters_sigma']:4d} {st_meta['gsq_iters_str']:>8} "
                f"{st_meta['depth']:5d} {_fmt_err_block(st_c)}"
            )
            print(f"{val_indent}{_fmt_err_block(st_v)}")
        return

    print(f"{meta_hdr}{err_hdr}")
    for layer_idx in sorted(results.keys()):
        st = results[layer_idx]
        print(
            f"{layer_idx:3d} {st['shift']:7.4g} {st['delta2']:4g} "
            f"{st['iters_sigma']:4d} {st['gsq_iters_str']:>8} {st['depth']:5d} "
            f"{_fmt_err_block(st)}"
        )


def _assert_split_results_independent(
    task_name: str,
    r_calib: dict,
    r_val: dict,
) -> None:
    """确认 all 模式两次评估未混用同一结果对象 / 同一批向量。"""
    if r_calib is r_val:
        raise RuntimeError(f"{task_name}: calib/val 返回了同一 dict（数据混用）")
    if r_calib["results"] is r_val["results"]:
        raise RuntimeError(f"{task_name}: calib/val results 为同一对象（数据混用）")
    if r_calib.get("eval_split") != "calib" or r_val.get("eval_split") != "validation":
        raise RuntimeError(
            f"{task_name}: eval_split 异常 "
            f"calib={r_calib.get('eval_split')!r} val={r_val.get('eval_split')!r}"
        )
    keys = sorted(set(r_calib["results"]) | set(r_val["results"]))
    if not keys:
        raise RuntimeError(f"{task_name}: 无误差结果")
    n_c = [int(r_calib["results"][k]["num_vectors"]) for k in keys if k in r_calib["results"]]
    n_v = [int(r_val["results"][k]["num_vectors"]) for k in keys if k in r_val["results"]]
    if not n_c or not n_v:
        raise RuntimeError(f"{task_name}: 缺少 calib 或 val 层结果")
    if n_c == n_v and all(
        r_calib["results"][k]["max_abs_err"] == r_val["results"][k]["max_abs_err"]
        and r_calib["results"][k]["mean_abs_err"] == r_val["results"][k]["mean_abs_err"]
        for k in keys
        if k in r_calib["results"] and k in r_val["results"]
    ):
        raise RuntimeError(
            f"{task_name}: calib/val 的 n 与全部误差完全一致，疑似未换数据 "
            f"(n_c={n_c[0]}, n_v={n_v[0]}, desc_c={r_calib.get('split_desc')!r}, "
            f"desc_v={r_val.get('split_desc')!r})"
        )


def collect_min_sigma_for_task(
    task_name: str,
    collector: GpuScaledScoreCollector,
    layer_shifts: list[float],
    layer_divs: list[float],
) -> tuple[list[float], list[float]]:
    """遍历已收集分数，统计每层 min/max(σ)。"""
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
            THOR_DELTA1,
            layer_divs[layer_idx],
            ERROR_EVAL_CHUNK_SIZE,
        )
        min_sigmas.append(mn)
        max_sigmas.append(mx)
        print(
            f"{layer_idx:3d} {layer_shifts[layer_idx]:7.4g} {layer_divs[layer_idx]:4g} "
            f"{mn:12.4e} {mx:12.4e} {n:10d}"
        )
        del scores, masks
        collector.release_layer(layer_idx)
    return min_sigmas, max_sigmas


def _uses_streaming_eval(task_name: str) -> bool:
    """大验证集任务：边 hook 边测，禁止先收集全部 scores。"""
    return task_name in ("mnli", "qnli")


def _prepare_eval_model_and_loader(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None = None,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[Any, DataLoader, str]:
    if eval_split not in ("validation", "calib"):
        raise ValueError(
            f"eval_split 须为 validation/calib，得到 {eval_split!r}"
        )
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=task_num_labels(task_name),
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    if num_samples is None:
        num_samples = NUM_SAMPLES
    eval_set, _, split_desc = resolve_eval_split(
        task_name,
        dataset,
        eval_split=eval_split,
        num_samples=num_samples,
        seed=RANDOM_SEED,
        calib_seed=calib_seed,
        indices_dir=calib_indices_dir,
    )
    tokenized = eval_set.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=eval_set.column_names,
        desc=f"tokenize {task_name}/{eval_split}",
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    return model, loader, split_desc


def _run_forward_with_hooks(model, loader, device, collector) -> None:
    hooks = register_scaled_score_hooks(model, collector)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    restore_hooks(hooks)


def stream_min_max_sigma_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    layer_shifts: list[float],
    layer_divs: list[float],
    num_samples: int | None = None,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[list[float], list[float], str]:
    """边 hook 边统计 σ，不落地分数。"""
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        eval_split=eval_split,
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    tracker = StreamingSigmaTracker(LAYER_ENABLE, layer_shifts, layer_divs)
    print(f"  σ 流式统计：{split_desc}")
    _run_forward_with_hooks(model, loader, device, tracker)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    label = f"{task_name}/{eval_split}"
    print(f"\n--- {label}  统计 σ=Σexp（streaming）---")
    print(f"{'层':>3} {'shift':>7} {'d2':>4} {'minσ':>12} {'maxσ':>12} {'n':>10}")
    for layer_idx in range(NUM_LAYERS):
        if not LAYER_ENABLE[layer_idx]:
            raise ValueError(f"{label} 层{layer_idx} 未启用，无法统计 σ")
        mn = tracker.min_sigma[layer_idx]
        mx = tracker.max_sigma[layer_idx]
        n = tracker.n_vectors[layer_idx]
        if n == 0 or not math.isfinite(mn):
            raise ValueError(f"{label} 层{layer_idx} 无有效 σ")
        print(
            f"{layer_idx:3d} {layer_shifts[layer_idx]:7.4g} {layer_divs[layer_idx]:4g} "
            f"{mn:12.4e} {mx:12.4e} {n:10d}"
        )
    return list(tracker.min_sigma), list(tracker.max_sigma), split_desc


def stream_error_eval_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    min_sigmas: list[float],
    sigma_only: bool = False,
    num_samples: int | None = None,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[dict[int, dict], str]:
    """边 hook 边算误差（或仅计向量），不落地分数。"""
    layer_shifts = layer_exp_shift_for_task(task_name)
    layer_divs = layer_exp_div_for_task(task_name)
    tracker = StreamingErrorTracker(
        LAYER_ENABLE,
        layer_shifts,
        layer_divs,
        layer_asor_alpha_sigma_for_task(task_name),
        layer_asor_alpha_sum_sq_for_task(task_name),
        layer_asor_alpha_sum_sq2_for_task(task_name),
        layer_asor_max_iters_sigma_for_task(task_name),
        layer_asor_max_iters_sum_sq_for_task(task_name),
        layer_asor_max_iters_sum_sq2_for_task(task_name),
        min_sigmas,
        sigma_only=sigma_only,
    )
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        eval_split=eval_split,
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    _run_forward_with_hooks(model, loader, device, tracker)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tracker.finalize(task_name), split_desc


def load_model_and_collect_scores(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None = None,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[Any, GpuScaledScoreCollector, str]:
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        eval_split=eval_split,
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    collector = GpuScaledScoreCollector(LAYER_ENABLE, device)
    _run_forward_with_hooks(model, loader, device, collector)
    return model, collector, split_desc


def recompute_union_sigma(
    task_name: str,
    device: torch.device,
    *,
    num_samples: int | None = None,
    json_dir: str = JSON_DIR,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> dict[str, Any]:
    """分别统计 calib / validation，并集写入 {task}_sigma.json。"""
    print(f"\n=== {task_name}: 重算 σ（calib ∪ validation）===")
    layer_shifts = layer_exp_shift_for_task(task_name)
    layer_divs = layer_exp_div_for_task(task_name)

    if _uses_streaming_eval(task_name):
        print(f"  {task_name}: 使用 streaming（边 hook 边统计，不缓存 scores）")
        calib_mins, calib_maxs, desc_c = stream_min_max_sigma_for_split(
            task_name,
            device,
            eval_split="calib",
            layer_shifts=layer_shifts,
            layer_divs=layer_divs,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  σ 统计数据：{desc_c}")
        val_mins, val_maxs, desc_v = stream_min_max_sigma_for_split(
            task_name,
            device,
            eval_split="validation",
            layer_shifts=layer_shifts,
            layer_divs=layer_divs,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  σ 统计数据：{desc_v}")
    else:
        _, coll_c, desc_c = load_model_and_collect_scores(
            task_name,
            device,
            eval_split="calib",
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  σ 统计数据：{desc_c}")
        calib_mins, calib_maxs = collect_min_sigma_for_task(
            f"{task_name}/calib", coll_c, layer_shifts, layer_divs
        )
        del coll_c
        if device.type == "cuda":
            torch.cuda.empty_cache()

        _, coll_v, desc_v = load_model_and_collect_scores(
            task_name,
            device,
            eval_split="validation",
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  σ 统计数据：{desc_v}")
        val_mins, val_maxs = collect_min_sigma_for_task(
            f"{task_name}/validation", coll_v, layer_shifts, layer_divs
        )
        del coll_v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    union_mins = merge_sigma_lists(calib_mins, val_mins, how="min")
    union_maxs = merge_sigma_lists(calib_maxs, val_maxs, how="max")
    print(f"\n--- {task_name}/union  minσ/maxσ ---")
    print(f"{'层':>3} {'minσ':>12} {'maxσ':>12}")
    for i in range(NUM_LAYERS):
        print(f"{i:3d} {union_mins[i]:12.4e} {union_maxs[i]:12.4e}")

    path = save_min_sigma_json(
        task_name,
        union_mins,
        json_dir,
        max_sigma=union_maxs,
        sources=["validation", "calib"],
    )
    print(f"  已写入并集 σ {path}")
    return {
        "min_sigma": union_mins,
        "max_sigma": union_maxs,
        "sigma_path": path,
        "calib_min_sigma": calib_mins,
        "validation_min_sigma": val_mins,
    }


def run_task(
    task_name: str,
    device: torch.device,
    *,
    sigma_only: bool = False,
    json_dir: str = JSON_DIR,
    num_samples: int | None = None,
    eval_split: str = "validation",
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    print_errors: bool = True,
    min_sigmas: list[float] | None = None,
    sigma_source: str | None = None,
) -> dict:
    if eval_split not in ("validation", "calib"):
        raise ValueError(
            f"run_task 的 eval_split 须为 validation/calib，得到 {eval_split!r}"
        )

    if min_sigmas is None:
        min_sigmas, sigma_source = load_min_sigma_from_json(task_name, json_dir)
    assert sigma_source is not None
    print(f"  {task_name}: aSOR 用并集 min(σ) ← {sigma_source}")

    if _uses_streaming_eval(task_name):
        print(f"  {task_name}: 使用 streaming（边 hook 边评估，不缓存 scores）")
        results, split_desc = stream_error_eval_for_split(
            task_name,
            device,
            eval_split=eval_split,
            min_sigmas=min_sigmas,
            sigma_only=sigma_only,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  评估数据：{split_desc}")
    else:
        model, collector, split_desc = load_model_and_collect_scores(
            task_name,
            device,
            eval_split=eval_split,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  评估数据：{split_desc}")

        layer_shifts = layer_exp_shift_for_task(task_name)
        layer_divs = layer_exp_div_for_task(task_name)
        results = analyze_thor_softmax_errors_gpu(
            task_name,
            collector,
            layer_shifts,
            layer_divs,
            layer_asor_alpha_sigma_for_task(task_name),
            layer_asor_alpha_sum_sq_for_task(task_name),
            layer_asor_alpha_sum_sq2_for_task(task_name),
            layer_asor_max_iters_sigma_for_task(task_name),
            layer_asor_max_iters_sum_sq_for_task(task_name),
            layer_asor_max_iters_sum_sq2_for_task(task_name),
            min_sigmas,
            device,
            sigma_only=sigma_only,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if sigma_only and print_errors:
        _print_sigma_only_table(task_name, results)
    elif not sigma_only and print_errors:
        _print_error_table(task_name, results)

    return {
        "min_sigma": min_sigmas,
        "sigma_source": sigma_source,
        "eval_split": eval_split,
        "split_desc": split_desc,
        "results": results,
    }


def run_task_all(
    task_name: str,
    device: torch.device,
    *,
    json_dir: str = JSON_DIR,
    num_samples: int | None = None,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    min_sigmas: list[float] | None = None,
    sigma_source: str | None = None,
) -> dict[str, Any]:
    """先独立跑 calib，再独立跑 validation；仅打印时并排。aSOR 用并集 σ。"""
    if min_sigmas is None:
        min_sigmas, sigma_source = load_min_sigma_from_json(task_name, json_dir)
    common = dict(
        sigma_only=False,
        json_dir=json_dir,
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
        print_errors=False,
        min_sigmas=min_sigmas,
        sigma_source=sigma_source,
    )
    # 两次各 load 模型 + 各收集分数；collector 不跨 split 复用
    r_calib = run_task(task_name, device, eval_split="calib", **common)
    r_val = run_task(task_name, device, eval_split="validation", **common)
    _assert_split_results_independent(task_name, r_calib, r_val)
    _print_error_table(
        task_name,
        r_calib["results"],
        dual_results=r_val["results"],
    )
    return {
        "eval_split": "all",
        "min_sigma": min_sigmas,
        "sigma_source": sigma_source,
        "calib": r_calib,
        "validation": r_val,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="thor Softmax aSOR 误差评估；σ 并集读写 sigma_out/"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument(
        "--eval-split",
        choices=("all", "validation", "calib"),
        default=DEFAULT_EVAL_SPLIT,
        help=(
            "评估数据：all=校验集+验证集（默认）；"
            "validation=官方验证集；calib=train 分层校验集"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help="仅 validation：子集大小；默认 NUM_SAMPLES（全量）。calib 忽略此项",
    )
    parser.add_argument(
        "--calib-seed",
        type=int,
        default=CALIB_SEED,
        help=(
            f"仅 calib：读取 {{task}}_calib_indices_seed{{seed}}.json "
            f"（默认 {CALIB_SEED}）；使用文件内全部 indices，禁止重建"
        ),
    )
    parser.add_argument(
        "--calib-indices-dir",
        default=CALIB_INDICES_DIR,
        help="calib 索引 JSON 目录（coverage_metric 输出，只读）",
    )
    parser.add_argument(
        "--json-dir",
        default=JSON_DIR,
        help=f"σ JSON 目录（默认 {JSON_DIR}）",
    )
    parser.add_argument(
        "--recompute-sigma",
        action="store_true",
        help="重算 calib+validation σ 并集，写入 {task}_sigma.json",
    )
    parser.add_argument(
        "--sigma-only",
        action="store_true",
        help="仅统计/打印 σ（配合 --recompute-sigma 可只写 JSON）",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA，将使用 CPU（仍走批量 torch，但无 GPU 加速）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("eval_softmax_cuda.py — GPU 批量 thor_softmax (aSOR)")
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"σ JSON 目录：{args.json_dir}")
    print(f"eval-split：{args.eval_split}")
    if args.eval_split == "all":
        print("  all：先评估校验集，再评估验证集；误差表每组两行（上 calib / 下 val）")
    print("aSOR σ：始终读 {task}_sigma.json（calib∪validation 并集）")
    if args.eval_split in ("calib", "all") or args.recompute_sigma:
        print(
            f"  calib indices: seed={args.calib_seed} "
            f"dir={args.calib_indices_dir}（只读文件，不重建）"
        )
    if args.recompute_sigma:
        print("  --recompute-sigma：写并集 {task}_sigma.json")
    print(
        f"min(σ)：{'重新统计并写入' if args.recompute_sigma else '从 JSON/缓存加载'}  "
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

            min_sigmas = None
            sigma_source = None
            if args.recompute_sigma:
                out = recompute_union_sigma(
                    task_name,
                    device,
                    num_samples=args.samples,
                    json_dir=args.json_dir,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                )
                min_sigmas = out["min_sigma"]
                sigma_source = out["sigma_path"]
            if args.sigma_only:
                if not args.recompute_sigma:
                    mins, src = load_min_sigma_from_json(task_name, args.json_dir)
                    print(f"\n--- {task_name}  已加载并集 min(σ) ← {src} ---")
                    for i, v in enumerate(mins):
                        print(f"  L{i}: minσ={v:.6g}  e0={e0_sigma_from_min(v):.6g}")
                continue

            if args.eval_split == "all":
                run_task_all(
                    task_name,
                    device,
                    json_dir=args.json_dir,
                    num_samples=args.samples,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                    min_sigmas=min_sigmas,
                    sigma_source=sigma_source,
                )
            else:
                run_task(
                    task_name,
                    device,
                    json_dir=args.json_dir,
                    num_samples=args.samples,
                    eval_split=args.eval_split,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                    min_sigmas=min_sigmas,
                    sigma_source=sigma_source,
                )
        except (FileNotFoundError, ValueError, IndexError, KeyError) as e:
            print(f"跳过 {task_name}：{e}")

    print("完成。")


if __name__ == "__main__":
    main()
