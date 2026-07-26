"""
HE LayerNorm 近似：参数 + 实现（来源 nolinear/layernorm.py）。

M 缩放 + he_invsqrt；不含 CKKS 编码 bookend。
方差区间来自 nolinear/variance_out/{task}_variance.json；
he_invsqrt 固定跑 max_iters 步（无 α；α 仅留在 nolinear/layernorm.py 调参）；
高中低精度档由 LAYER_INV_SQRT_MAX_ITERS_BY_TASK 区分。

用法：
    from layernorm_poly import HeLayerNormPolyEvaluator, layernorm_config_for_layer
    y = HeLayerNormPolyEvaluator("mrpc", layer_idx=0, kind="ln1", level=1)(
        x, gamma, beta, eps
    )
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np
import torch

LAYERNORM_LEVEL_KEYS = ("low", "mid", "high")
LAYERNORM_KIND_KEYS = ("ln1", "ln2")  # ln1=attn output LN, ln2=FFN output LN

LAYERNORM_POLY_FUNC_NAMES: list[str] = [
    "poly_layernorm_low",
    "poly_layernorm_mid",
    "poly_layernorm_high",
]

TASK_NAMES = ["mrpc", "rte", "sst2"]
NUM_LAYERS = 12

W_BUFFER = 1

_VARIANCE_JSON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nolinear", "variance_out"
)


# task → level → {ln1, ln2} → 12 层 he_invsqrt 迭代次数（固定步数，无 α）
LAYER_INV_SQRT_MAX_ITERS_BY_TASK: dict[str, dict[str, dict[str, list[int]]]] = {
    "mrpc": {
        "low": {
            "ln1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
            "ln2": [2, 2, 4, 3, 3, 3, 3, 3, 3, 5, 5, 2],
        },
        "mid": {
            "ln1": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4],
            "ln2": [3, 3, 5, 4, 4, 4, 4, 4, 4, 6, 6, 3],
        },
        "high": {
            "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5],
            "ln2": [4, 4, 6, 5, 5, 5, 5, 5, 5, 7, 7, 4],
        },
    },
    "rte": {
        "low": {
            "ln1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4],
            "ln2": [2, 2, 4, 3, 3, 3, 3, 3, 3, 5, 5, 2],
        },
        "mid": {
            "ln1": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5],
            "ln2": [3, 3, 5, 4, 4, 4, 4, 4, 4, 6, 6, 3],
        },
        "high": {
            "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 6],
            "ln2": [4, 4, 6, 5, 5, 5, 5, 5, 5, 7, 7, 4],
        },
    },
    "sst2": {
        "low": {
            "ln1": [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3],
            "ln2": [2, 2, 4, 3, 3, 3, 3, 3, 3, 5, 5, 1],
        },
        "mid": {
            "ln1": [3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4],
            "ln2": [3, 3, 5, 4, 4, 4, 4, 4, 4, 6, 6, 2],
        },
        "high": {
            "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5],
            "ln2": [4, 4, 6, 5, 5, 5, 5, 5, 5, 7, 7, 3],
        },
    },
}


def variance_json_path(task_name: str, json_dir: str = _VARIANCE_JSON_DIR) -> str:
    return os.path.join(json_dir, f"{task_name}_variance.json")


def _parse_variance_json(payload: dict) -> dict[str, list[list[float]]]:
    if "ln1" in payload and "ln2" in payload:
        return {
            "ln1": [[float(a), float(b)] for a, b in payload["ln1"]],
            "ln2": [[float(a), float(b)] for a, b in payload["ln2"]],
        }
    if "LAYER_VAR_RANGE" in payload:
        table = payload["LAYER_VAR_RANGE"]
        return {
            "ln1": [[float(a), float(b)] for a, b in table["ln1"]],
            "ln2": [[float(a), float(b)] for a, b in table["ln2"]],
        }
    layers = payload.get("layers", {})
    ln1: list[list[float]] = []
    ln2: list[list[float]] = []
    for layer_idx in range(NUM_LAYERS):
        e1 = layers.get(f"layer{layer_idx}_ln1")
        e2 = layers.get(f"layer{layer_idx}_ln2")
        if e1 is None or e2 is None:
            raise ValueError(
                f"layers 缺少 layer{layer_idx}_ln1 或 layer{layer_idx}_ln2"
            )
        ln1.append([float(e1["var_range"][0]), float(e1["var_range"][1])])
        ln2.append([float(e2["var_range"][0]), float(e2["var_range"][1])])
    return {"ln1": ln1, "ln2": ln2}


def load_var_ranges_from_json(
    task_name: str,
    json_dir: str = _VARIANCE_JSON_DIR,
) -> tuple[dict[str, list[list[float]]], str]:
    path = variance_json_path(task_name, json_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"方差 JSON 不存在：{path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    json_task = payload.get("task")
    if json_task and json_task != task_name:
        raise ValueError(
            f"{path} 中 task={json_task!r} 与请求任务 {task_name!r} 不一致"
        )

    table = _parse_variance_json(payload)
    for kind in LAYERNORM_KIND_KEYS:
        if len(table[kind]) != NUM_LAYERS:
            raise ValueError(f"{path} 中 {kind} 长度应为 {NUM_LAYERS}")
        for i, pair in enumerate(table[kind]):
            min_v, max_v = float(pair[0]), float(pair[1])
            if min_v <= 0 or max_v <= 0 or min_v >= max_v:
                raise ValueError(
                    f"{task_name} 层{i} {kind} 须满足 0 < min_var < max_var"
                )
    return table, path


def layer_invsqrt_max_iters_for_task(
    task_name: str, kind: str, level_key: str
) -> list[int]:
    if task_name not in LAYER_INV_SQRT_MAX_ITERS_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_INV_SQRT_MAX_ITERS_BY_TASK")
    if level_key not in LAYERNORM_LEVEL_KEYS:
        raise ValueError(f"layernorm level 非法：{level_key}")
    table = LAYER_INV_SQRT_MAX_ITERS_BY_TASK[task_name][level_key]
    if kind not in table:
        raise KeyError(f"未知 LayerNorm 类型：{kind}")
    max_iters_list = table[kind]
    if len(max_iters_list) != NUM_LAYERS:
        raise ValueError(
            f"{task_name} {level_key} {kind} max_iters 长度应为 {NUM_LAYERS}"
        )
    out: list[int] = []
    for i, m in enumerate(max_iters_list):
        m_i = int(m)
        if m_i < 1:
            raise ValueError(
                f"{task_name} 层{i} {level_key} {kind} max_iters 须 ≥ 1"
            )
        out.append(m_i)
    return out


@lru_cache(maxsize=None)
def _var_ranges_for_task(task_name: str) -> dict[str, list[list[float]]]:
    table, _ = load_var_ranges_from_json(task_name)
    return table


def layernorm_config_for_layer(
    task_name: str,
    layer_idx: int,
    kind: str,
    level: int,
) -> dict:
    if task_name not in LAYER_INV_SQRT_MAX_ITERS_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_INV_SQRT_MAX_ITERS_BY_TASK")
    if kind not in LAYERNORM_KIND_KEYS:
        raise ValueError(f"未知 LayerNorm 类型：{kind}")
    if level < 0 or level >= len(LAYERNORM_LEVEL_KEYS):
        raise ValueError(f"layernorm level 非法：{level}")
    if not (0 <= layer_idx < NUM_LAYERS):
        raise ValueError(f"layer_idx 非法：{layer_idx}")

    level_key = LAYERNORM_LEVEL_KEYS[level]
    min_var, max_var = _var_ranges_for_task(task_name)[kind][layer_idx]
    invsqrt_max_iters = layer_invsqrt_max_iters_for_task(
        task_name, kind, level_key
    )[layer_idx]
    return {
        "task_name": task_name,
        "layer_idx": layer_idx,
        "kind": kind,
        "level": level,
        "level_name": level_key,
        "min_var": float(min_var),
        "max_var": float(max_var),
        "w_buffer": W_BUFFER,
        "invsqrt_max_iters": invsqrt_max_iters,
    }


def _he_invsqrt_kn(en: float) -> float:
    coeffs = [1.0 - en**3, 6.0 * en**2 - 6.0, 9.0 - 9.0 * en]
    roots = np.roots(coeffs)
    return float(roots[1].real)


def count_he_invsqrt_iters(max_iters: int) -> int:
    """生产路径固定跑 max_iters 步，深度直接取该值。"""
    mi = int(max_iters)
    if mi < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    return mi


def he_invsqrt_batched(
    variance: torch.Tensor,
    e_init: float,
    max_iters: int,
) -> torch.Tensor:
    """he_invsqrt；固定 max_iters 步（无 α）。"""
    an = variance.clamp(min=1e-30)
    bn = torch.ones_like(an)
    en = float(e_init)
    if en <= 0.0:
        raise ValueError(f"e_init 须 > 0，得到 {e_init}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    for _ in range(max_iters):
        kn = _he_invsqrt_kn(en)
        inv_kn = 3.0 / kn
        bn = bn * (kn ** (3.0 / 2.0) / 2.0) * (inv_kn - an)
        an = an * (kn**3 / 4.0) * (inv_kn - an) ** 2
        an = an.clamp(min=1e-30)
        bn = bn.clamp(min=1e-30, max=1e30)
        en = kn * en * (3.0 - kn * en) ** 2 / 4.0
    return bn


def he_layernorm_from_config(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    cfg: dict,
) -> torch.Tensor:
    """HE LayerNorm；x: [..., D]，gamma/beta: [D]。"""
    min_var = cfg["min_var"]
    max_var = cfg["max_var"]
    max_iters = int(cfg["invsqrt_max_iters"])
    w_buffer = float(cfg.get("w_buffer", W_BUFFER))

    n = float(x.shape[-1])
    max_for_denominator = (max_var * w_buffer + eps) * (n**2)
    scale = 1.0 / math.sqrt(max_for_denominator)
    x_scaled = x * scale

    sum_x = x_scaled.sum(dim=-1, keepdim=True)
    numerator = n * x_scaled - sum_x

    sum_x_flat = sum_x.squeeze(-1)
    sigma_x2 = (x_scaled * x_scaled).sum(dim=-1)
    variance = n * sigma_x2 - sum_x_flat * sum_x_flat + eps / max_for_denominator

    inv_sqrt = he_invsqrt_batched(
        variance,
        e_init=min_var / max_var,
        max_iters=max_iters,
    )

    gamma = gamma.to(device=x.device, dtype=x.dtype)
    beta = beta.to(device=x.device, dtype=x.dtype)
    return numerator * inv_sqrt.unsqueeze(-1) * gamma + beta


class HeLayerNormPolyEvaluator:
    """per-task × per-layer × ln1/ln2 的 HE LayerNorm 求值器。"""

    __slots__ = ("task_name", "layer_idx", "kind", "level", "_cfg")

    def __init__(self, task_name: str, layer_idx: int, kind: str, level: int):
        self.task_name = task_name
        self.layer_idx = layer_idx
        self.kind = kind
        self.level = level
        self._cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)

    def __call__(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return he_layernorm_from_config(x, gamma, beta, eps, self._cfg)


def poly_layernorm_low(
    x: torch.Tensor,
    layer_idx: int,
    kind: str,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    *,
    task_name: str,
) -> torch.Tensor:
    return HeLayerNormPolyEvaluator(task_name, layer_idx, kind, 0)(
        x, gamma, beta, eps
    )


def poly_layernorm_mid(
    x: torch.Tensor,
    layer_idx: int,
    kind: str,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    *,
    task_name: str,
) -> torch.Tensor:
    return HeLayerNormPolyEvaluator(task_name, layer_idx, kind, 1)(
        x, gamma, beta, eps
    )


def poly_layernorm_high(
    x: torch.Tensor,
    layer_idx: int,
    kind: str,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    *,
    task_name: str,
) -> torch.Tensor:
    return HeLayerNormPolyEvaluator(task_name, layer_idx, kind, 2)(
        x, gamma, beta, eps
    )


LAYERNORM_POLY_FUNCS: list = [
    poly_layernorm_low,
    poly_layernorm_mid,
    poly_layernorm_high,
]
