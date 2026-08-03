"""
LayerNorm HE 近似评估：方差区间统计 + 与标准 LayerNorm 的逐层误差。

算法（M 缩放 + he_invsqrt；不含 CKKS 编码 bookend）：
  0. x' ← x / √((max_var·W_BUFFER+ε)·n²)
  1. numerator ← n·x' − Σx'
  2. variance ← n·Σ(x')² − (Σx')² + ε / max_for_denominator
  3. inv_sqrt ← he_invsqrt(variance, e₀=min_var/max_var, alpha)
  4. y ← numerator ⊗ γ ⊗ inv_sqrt ⊕ β

  - α / max_iters：本文件按 task×层调参（带 α 停止）
  - W_BUFFER / VAR_*_SCALE：统一来自 ../layernorm_poly.py（勿在此重复维护）
  - 调完 max_iters 后写回 layernorm_poly 对应任务的 high 档
  - mnli / qnli：边 hook 边统计/评估（Streaming*Tracker），不缓存全部 LN 输入

用法（在 nolinear/ 下）：
  python3 layernorm.py                         # 默认 all：校验集+验证集都评估
  python3 layernorm.py --eval-split validation # 仅验证集
  python3 layernorm.py --eval-split calib      # 仅校验集
  python3 layernorm.py --recompute-variance    # 两侧统计后取并集 → {task}_variance.json
  python3 layernorm.py --variance-only --recompute-variance  # 仅统计方差
  python3 layernorm.py --task mrpc --samples 32

  --eval-split all：先 calib 再 validation，误差表每组两行（默认）
  --eval-split validation：官方 validation（可用 --samples 子采样）
  --eval-split calib：仅使用 coverage_metric 已写入的校验集索引 JSON
    （results/coverage_metrics/{task}_calib_indices_seed{seed}.json；禁止现场重建）

方差 JSON：
  {task}_variance.json — HE 用；calib∪validation 各位置 min 取更小、max 取更大
  存原始 [var_min, var_max]；使用时 scaled = [min×VAR_MIN_SCALE, max×VAR_MAX_SCALE]
  （尺度来自 layernorm_poly）。
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
import torch.nn.functional as F
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

from layernorm_poly import (  # noqa: E402
    NUM_LAYERS as POLY_NUM_LAYERS,
    VAR_MAX_SCALE,
    VAR_MIN_SCALE,
    W_BUFFER,
    scaled_var_range,
)

# ===================== 配置 =====================
#TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
TASK_NAMES = ["cola", "qnli", "mnli"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = 128
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
BATCH_SIZE = 8

# 校验集：只读 coverage_metric 落盘索引（与文件内容完全一致，不重建）
DEFAULT_EVAL_SPLIT = "all"  # "all" | "validation" | "calib"
CALIB_SEED = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_INDICES_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "results", "coverage_metrics")
)

MAX_VECTORS_FOR_ERROR_EVAL: int | None = None
ERROR_EVAL_CHUNK_SIZE = 4096
STATS_CHUNK_SIZE = 4096
REL_ERR_DENOM_MIN = 1e-4  # |y_ref| 低于此值的点不参与相对误差统计


JSON_DIR = os.path.join(_SCRIPT_DIR, "variance_out")

LAYER_INV_SQRT_ALPHA_BY_TASK: dict[str, dict[str, list[float]]] = {
    task: {
        "ln1": [0.00001]*12,
        "ln2": [0.00001]*12,
    }
    for task in TASK_NUM_LABELS
}

# 单档 max_iters（带 α）；调完后同步到 layernorm_poly 的 high 档
LAYER_INV_SQRT_MAX_ITERS_BY_TASK: dict[str, dict[str, list[int]]] = {
    # "mrpc": {
    #     "ln1": [4,4,4,4,4,4,4,4,4,4,5,5],
    #     "ln2": [4,4,6,5,5,5,5,5,5,7,7,4],# mrpc 0.00005
    # },
    # "rte": {
    #     "ln1": [4,4,4,4,4,4,4,4,4,4,5,6],
    #     "ln2":[4,4,6,5,5,5,5,5,5,7,7,4],
    # },
    # "sst2": {
    #     "ln1": [4,4,4,4,4,4,4,4,4,5,5,5],
    #     "ln2": [4,4,6,5,5,5,5,5,5,7,7,4],
    # },
    # cola/qnli/mnli：初值抄 mrpc，跑通后按误差表再调
    "cola": {
        "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5],
        "ln2": [4, 4, 5, 5, 5, 5, 5, 4, 5, 7, 7, 3],
    },
    "qnli": {
        "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6],
        "ln2": [4, 4, 6, 5, 5, 5, 5, 5, 5, 7, 7, 4],
    },
    "mnli": {
        "ln1": [4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6],
        "ln2": [4, 4, 6, 5, 5, 5, 5, 5, 5, 7, 7, 4],
    },

}
LAYER_ENABLE = [True, True, True, True, True, True, 
                True, True, True, True, True, True]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["cola"]["ln1"] = [x-2  for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["cola"]["ln1"]]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["cola"]["ln2"] = [x-2  for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["cola"]["ln2"]]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["qnli"]["ln1"]  = [x-2 for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["qnli"]["ln1"]]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["qnli"]["ln2"]  = [x-2  for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["qnli"]["ln2"]]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["mnli"]["ln1"] = [x-2  for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["mnli"]["ln1"]]
LAYER_INV_SQRT_MAX_ITERS_BY_TASK["mnli"]["ln2"] = [x-2 for x in LAYER_INV_SQRT_MAX_ITERS_BY_TASK["mnli"]["ln2"]]

# =============================================================================

LayerNormKind = str  # "ln1" | "ln2"


def task_num_labels(task_name: str) -> int:
    if task_name not in TASK_NUM_LABELS:
        raise KeyError(f"未知任务 {task_name}，请在 TASK_NUM_LABELS 中配置")
    return TASK_NUM_LABELS[task_name]


def format_var_range(rng: list[float]) -> str:
    return f"[{rng[0]:.6g}, {rng[1]:.6g}]"


def variance_json_path(task_name: str, json_dir: str = JSON_DIR) -> str:
    return os.path.join(json_dir, f"{task_name}_variance.json")


def _parse_variance_json(payload: dict) -> dict[str, list[list[float]]]:
    """解析 JSON；兼容新格式（顶层 ln1/ln2）与旧格式。"""
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


def _validate_var_ranges(
    task_name: str, table: dict[str, list[list[float]]]
) -> dict[str, list[list[float]]]:
    for kind in ("ln1", "ln2"):
        ranges = table[kind]
        if len(ranges) != NUM_LAYERS:
            raise ValueError(f"{task_name} {kind} 方差区间长度应为 {NUM_LAYERS}")
        for i, pair in enumerate(ranges):
            if len(pair) != 2:
                raise ValueError(f"{task_name} 层{i} {kind} 须为 [min_var, max_var]")
            min_v, max_v = float(pair[0]), float(pair[1])
            if min_v <= 0 or max_v <= 0 or min_v >= max_v:
                raise ValueError(
                    f"{task_name} 层{i} {kind} 须满足 0 < min_var < max_var"
                )
    return table


def load_var_ranges_from_json(
    task_name: str,
    json_dir: str = JSON_DIR,
) -> tuple[dict[str, list[list[float]]], str]:
    path = variance_json_path(task_name, json_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"方差 JSON 不存在：{path}；请先运行 "
            "`python3 layernorm.py --recompute-variance`"
        )

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    json_task = payload.get("task")
    if json_task and json_task != task_name:
        raise ValueError(
            f"{path} 中 task={json_task!r} 与请求任务 {task_name!r} 不一致"
        )

    table = _validate_var_ranges(task_name, _parse_variance_json(payload))
    return table, path


def merge_var_ranges(
    a: dict[str, list[list[float]]],
    b: dict[str, list[list[float]]],
) -> dict[str, list[list[float]]]:
    """各位置 min 取更小、max 取更大。"""
    out: dict[str, list[list[float]]] = {"ln1": [], "ln2": []}
    for kind in ("ln1", "ln2"):
        if len(a[kind]) != len(b[kind]):
            raise ValueError(f"merge {kind}: 长度不一致")
        for ra, rb in zip(a[kind], b[kind]):
            out[kind].append(
                [min(float(ra[0]), float(rb[0])), max(float(ra[1]), float(rb[1]))]
            )
    return out


def save_var_ranges_json(
    task_name: str,
    var_ranges: dict[str, list[list[float]]],
    json_dir: str = JSON_DIR,
    *,
    sources: list[str] | None = None,
    note: str | None = None,
) -> str:
    os.makedirs(json_dir, exist_ok=True)
    path = variance_json_path(task_name, json_dir)
    src = list(sources) if sources is not None else ["validation", "calib"]
    if note is None:
        note = (
            "ln1/ln2 为 calib∪validation 原始 [var_min, var_max]"
            "（各位置 min 取更小、max 取更大）；使用时再 × var_*_scale"
        )
    payload = {
        "task": task_name,
        "eval_split": "union",
        "sources": src,
        "scaled": False,
        "var_min_scale": VAR_MIN_SCALE,
        "var_max_scale": VAR_MAX_SCALE,
        "ln1": var_ranges["ln1"],
        "ln2": var_ranges["ln2"],
        "updated": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def layer_invsqrt_alpha_for_task(
    task_name: str, kind: LayerNormKind
) -> list[float]:
    if task_name not in LAYER_INV_SQRT_ALPHA_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_INV_SQRT_ALPHA_BY_TASK")
    table = LAYER_INV_SQRT_ALPHA_BY_TASK[task_name]
    if kind not in table:
        raise KeyError(f"未知 LayerNorm 类型：{kind}")
    alphas = table[kind]
    if len(alphas) != NUM_LAYERS:
        raise ValueError(f"{task_name} {kind} alpha 长度应为 {NUM_LAYERS}")
    for i, a in enumerate(alphas):
        a_f = float(a)
        if not (0.0 < a_f < 1.0):
            raise ValueError(f"{task_name} 层{i} {kind} alpha 须满足 0 < alpha < 1")
    return [float(a) for a in alphas]


def layer_invsqrt_max_iters_for_task(
    task_name: str, kind: LayerNormKind
) -> list[int]:
    if task_name not in LAYER_INV_SQRT_MAX_ITERS_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_INV_SQRT_MAX_ITERS_BY_TASK")
    table = LAYER_INV_SQRT_MAX_ITERS_BY_TASK[task_name]
    if kind not in table:
        raise KeyError(f"未知 LayerNorm 类型：{kind}")
    max_iters_list = table[kind]
    if len(max_iters_list) != NUM_LAYERS:
        raise ValueError(f"{task_name} {kind} max_iters 长度应为 {NUM_LAYERS}")
    out: list[int] = []
    for i, m in enumerate(max_iters_list):
        m_i = int(m)
        if m_i < 1:
            raise ValueError(f"{task_name} 层{i} {kind} max_iters 须 ≥ 1")
        out.append(m_i)
    return out


def _he_invsqrt_kn(en: float) -> float:
    coeffs = [1.0 - en**3, 6.0 * en**2 - 6.0, 9.0 - 9.0 * en]
    roots = np.roots(coeffs)
    return float(roots[1].real)


def he_invsqrt_batched(
    variance: torch.Tensor,
    e_init: float,
    alpha: float,
    max_iters: int,
) -> tuple[torch.Tensor, int]:
    an = variance.clamp(min=1e-30)
    bn = torch.ones_like(an)
    en = float(e_init)
    iters = 0
    while en < 1.0 - alpha and iters < max_iters:
        kn = _he_invsqrt_kn(en)
        inv_kn = 3.0 / kn
        bn = bn * (kn ** (3.0 / 2.0) / 2.0) * (inv_kn - an)
        an = an * (kn**3 / 4.0) * (inv_kn - an) ** 2
        an = an.clamp(min=1e-30)
        bn = bn.clamp(min=1e-30, max=1e30)
        en = kn * en * (3.0 - kn * en) ** 2 / 4.0
        iters += 1
    return bn, iters


def he_layernorm_batched(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    var_e: float,
    min_var: float,
    max_var: float,
    invsqrt_alpha: float,
    invsqrt_max_iters: int,
) -> tuple[torch.Tensor, int]:
    n = float(x.shape[-1])

    max_for_denominator = (max_var * W_BUFFER + var_e) * (n**2)
    scale = 1.0 / math.sqrt(max_for_denominator)
    x_scaled = x * scale

    sum_x = x_scaled.sum(dim=-1, keepdim=True)
    numerator = n * x_scaled - sum_x

    sum_x_flat = sum_x.squeeze(-1)
    sigma_x2 = (x_scaled * x_scaled).sum(dim=-1)
    variance = n * sigma_x2 - sum_x_flat * sum_x_flat + var_e / max_for_denominator

    inv_sqrt, inv_iters = he_invsqrt_batched(
        variance,
        e_init=min_var / max_var,
        alpha=invsqrt_alpha,
        max_iters=invsqrt_max_iters,
    )

    out = numerator * inv_sqrt.unsqueeze(-1) * gamma + beta
    return out, inv_iters


def reference_layernorm_batched(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=gamma, bias=beta, eps=eps)


class GpuLayerNormCollector:
    """收集全部 LayerNorm 输入向量 [N, D]（小任务用）。大任务请用 Streaming*Tracker。"""

    def __init__(self, layer_enable: list[bool], device: torch.device):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.device = device
        self.layer_enable = layer_enable
        self._inputs: dict[tuple[int, LayerNormKind], list[torch.Tensor]] = {}

    def _key(self, layer_idx: int, kind: LayerNormKind) -> tuple[int, LayerNormKind]:
        return layer_idx, kind

    def add_input(
        self,
        layer_idx: int,
        kind: LayerNormKind,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = _valid_ln_input_rows(hidden, attention_mask)
        if rows is None:
            return
        key = self._key(layer_idx, kind)
        self._inputs.setdefault(key, []).append(rows)

    def stacked(self, layer_idx: int, kind: LayerNormKind) -> torch.Tensor:
        key = self._key(layer_idx, kind)
        parts = self._inputs.get(key, [])
        if not parts:
            return torch.zeros(0, 0, device=self.device, dtype=torch.float32)
        return torch.cat(parts, dim=0)

    def release(self, layer_idx: int, kind: LayerNormKind) -> None:
        self._inputs.pop(self._key(layer_idx, kind), None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _valid_ln_input_rows(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """取有效 token 的 LN 输入行；无有效行时返回 None。不跨 batch 缓存。"""
    rows = hidden.detach().reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
    if attention_mask is None:
        return rows
    mask = attention_mask.reshape(-1)
    valid = mask > 0
    if not valid.any():
        return None
    return rows[valid]


class StreamingVarianceTracker:
    """边 hook 边更新各层 ln1/ln2 的 min/max Var(x)，不缓存输入。"""

    def __init__(
        self,
        layer_enable: list[bool],
        *,
        chunk_size: int = STATS_CHUNK_SIZE,
    ):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.layer_enable = layer_enable
        self.chunk_size = int(chunk_size)
        self.var_min: dict[tuple[int, LayerNormKind], float] = {}
        self.var_max: dict[tuple[int, LayerNormKind], float] = {}
        self.n_vectors: dict[tuple[int, LayerNormKind], int] = {}
        for i in range(NUM_LAYERS):
            if not layer_enable[i]:
                continue
            for kind in ("ln1", "ln2"):
                key = (i, kind)
                self.var_min[key] = float("inf")
                self.var_max[key] = float("-inf")
                self.n_vectors[key] = 0

    @torch.no_grad()
    def add_input(
        self,
        layer_idx: int,
        kind: LayerNormKind,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = _valid_ln_input_rows(hidden, attention_mask)
        if rows is None:
            return
        key = (layer_idx, kind)
        n = int(rows.shape[0])
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            v = population_var_batched(rows[start:end])
            self.var_min[key] = min(self.var_min[key], float(v.min().item()))
            self.var_max[key] = max(self.var_max[key], float(v.max().item()))
            self.n_vectors[key] += int(v.numel())

    def finalize(
        self, task_label: str
    ) -> tuple[
        dict[str, list[list[float]]],
        list[tuple[int, LayerNormKind, int, list[float]]],
    ]:
        var_ranges: dict[str, list[list[float]]] = {"ln1": [], "ln2": []}
        variance_rows: list[tuple[int, LayerNormKind, int, list[float]]] = []
        for layer_idx in range(NUM_LAYERS):
            if not self.layer_enable[layer_idx]:
                raise ValueError(f"{task_label} 层{layer_idx} 未启用，无法统计方差")
            for kind in ("ln1", "ln2"):
                key = (layer_idx, kind)
                n = self.n_vectors[key]
                mn = self.var_min[key]
                mx = self.var_max[key]
                if n == 0 or not math.isfinite(mn) or not math.isfinite(mx):
                    raise ValueError(f"{task_label} 层{layer_idx} {kind} 无有效方差")
                raw_range = [mn, mx]
                var_ranges[kind].append(raw_range)
                variance_rows.append((layer_idx, kind, n, raw_range))
        return _validate_var_ranges(task_label, var_ranges), variance_rows


class StreamingErrorTracker:
    """边 hook 边累加 HE vs 标准 LayerNorm 误差，不缓存输入。"""

    def __init__(
        self,
        layer_enable: list[bool],
        he_var_ranges: dict[str, list[list[float]]],
        ln1_alphas: list[float],
        ln2_alphas: list[float],
        ln1_max_iters: list[int],
        ln2_max_iters: list[int],
        ln_params: dict[tuple[int, LayerNormKind], tuple[float, torch.Tensor, torch.Tensor]],
        *,
        chunk_size: int = ERROR_EVAL_CHUNK_SIZE,
    ):
        if len(layer_enable) != NUM_LAYERS:
            raise ValueError(f"LAYER_ENABLE 长度应为 {NUM_LAYERS}")
        self.layer_enable = layer_enable
        self.he_var_ranges = he_var_ranges
        self.alphas = {"ln1": ln1_alphas, "ln2": ln2_alphas}
        self.max_iters = {"ln1": ln1_max_iters, "ln2": ln2_max_iters}
        self.ln_params = ln_params
        self.chunk_size = int(chunk_size)
        self.n_vectors: dict[tuple[int, LayerNormKind], int] = {}
        self._max_abs: dict[tuple[int, LayerNormKind], float] = {}
        self._sum_abs: dict[tuple[int, LayerNormKind], float] = {}
        self._err_count: dict[tuple[int, LayerNormKind], int] = {}
        self._max_rel: dict[tuple[int, LayerNormKind], float] = {}
        self._sum_rel: dict[tuple[int, LayerNormKind], float] = {}
        self._rel_count: dict[tuple[int, LayerNormKind], int] = {}
        self._invsqrt_iters: dict[tuple[int, LayerNormKind], int] = {}
        self._hidden_dim: dict[tuple[int, LayerNormKind], int] = {}
        for i in range(NUM_LAYERS):
            if not layer_enable[i]:
                continue
            for kind in ("ln1", "ln2"):
                key = (i, kind)
                self.n_vectors[key] = 0
                self._max_abs[key] = 0.0
                self._sum_abs[key] = 0.0
                self._err_count[key] = 0
                self._max_rel[key] = 0.0
                self._sum_rel[key] = 0.0
                self._rel_count[key] = 0
                self._invsqrt_iters[key] = 0

    @torch.no_grad()
    def add_input(
        self,
        layer_idx: int,
        kind: LayerNormKind,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        if not self.layer_enable[layer_idx]:
            return
        rows = _valid_ln_input_rows(hidden, attention_mask)
        if rows is None:
            return
        key = (layer_idx, kind)
        n = int(rows.shape[0])
        self.n_vectors[key] += n
        self._hidden_dim[key] = int(rows.shape[-1])

        eps, gamma, beta = self.ln_params[key]
        gamma = gamma.to(device=rows.device, dtype=rows.dtype)
        beta = beta.to(device=rows.device, dtype=rows.dtype)
        raw_he = self.he_var_ranges[kind][layer_idx]
        var_range = scaled_var_range(float(raw_he[0]), float(raw_he[1]))
        min_var, max_var = float(var_range[0]), float(var_range[1])
        invsqrt_alpha = float(self.alphas[kind][layer_idx])
        invsqrt_max_iters = int(self.max_iters[kind][layer_idx])

        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            chunk = rows[start:end]
            y_ref = reference_layernorm_batched(chunk, gamma, beta, eps)
            y_he, inv_iters = he_layernorm_batched(
                chunk,
                gamma,
                beta,
                eps,
                min_var,
                max_var,
                invsqrt_alpha,
                invsqrt_max_iters,
            )
            self._invsqrt_iters[key] = inv_iters

            err = torch.abs(y_ref - y_he)
            self._max_abs[key] = max(self._max_abs[key], float(err.max().item()))
            self._sum_abs[key] += float(err.sum().item())
            self._err_count[key] += int(err.numel())

            y_ref_abs = torch.abs(y_ref)
            denom_ok = y_ref_abs >= REL_ERR_DENOM_MIN
            if denom_ok.any():
                rel_pct = err[denom_ok] / y_ref_abs[denom_ok] * 100.0
                self._max_rel[key] = max(
                    self._max_rel[key], float(rel_pct.max().item())
                )
                self._sum_rel[key] += float(rel_pct.sum().item())
                self._rel_count[key] += int(rel_pct.numel())

    def finalize(self) -> dict[tuple[int, LayerNormKind], dict]:
        results: dict[tuple[int, LayerNormKind], dict] = {}
        for layer_idx in range(NUM_LAYERS):
            if not self.layer_enable[layer_idx]:
                continue
            for kind in ("ln1", "ln2"):
                key = (layer_idx, kind)
                n = self.n_vectors[key]
                if n == 0:
                    continue
                eps, _, _ = self.ln_params[key]
                raw_he = self.he_var_ranges[kind][layer_idx]
                var_range = scaled_var_range(float(raw_he[0]), float(raw_he[1]))
                min_var, max_var = float(var_range[0]), float(var_range[1])
                invsqrt_alpha = float(self.alphas[kind][layer_idx])
                invsqrt_max_iters = int(self.max_iters[kind][layer_idx])
                count = self._err_count[key]
                rel_count = self._rel_count[key]
                err: dict[str, Any] = {
                    "num_vectors": n,
                    "invsqrt_iters": self._invsqrt_iters[key],
                    "eps": eps,
                    "hidden_dim": self._hidden_dim.get(key, 0),
                    "min_var": min_var,
                    "max_var": max_var,
                    "invsqrt_alpha": invsqrt_alpha,
                    "invsqrt_max_iters": invsqrt_max_iters,
                }
                if count == 0:
                    err["max_abs_err"] = float("nan")
                    err["mean_abs_err"] = float("nan")
                    err["max_rel_pct"] = float("nan")
                    err["mean_rel_pct"] = float("nan")
                else:
                    err["max_abs_err"] = self._max_abs[key]
                    err["mean_abs_err"] = self._sum_abs[key] / count
                    err["num_elems"] = count
                    err["max_rel_pct"] = (
                        self._max_rel[key] if rel_count > 0 else float("nan")
                    )
                    err["mean_rel_pct"] = (
                        self._sum_rel[key] / rel_count
                        if rel_count > 0
                        else float("nan")
                    )
                results[key] = err
        return results


def register_layernorm_hooks(model, collector) -> list:
    hooks: list = []

    for layer_idx in range(NUM_LAYERS):
        if not collector.layer_enable[layer_idx]:
            continue

        layer = model.bert.encoder.layer[layer_idx]

        for kind, ln_module in (
            ("ln1", layer.attention.output.LayerNorm),
            ("ln2", layer.output.LayerNorm),
        ):
            orig_forward = ln_module.forward

            def make_patched(orig_fn, lidx, knd, coll, ln_mod):
                @functools.wraps(orig_fn)
                def patched_forward(hidden_states, *args, **kwargs):
                    attn_mask = getattr(ln_mod, "_eval_ln_attn_mask", None)
                    coll.add_input(lidx, knd, hidden_states, attn_mask)
                    return orig_fn(hidden_states, *args, **kwargs)

                return patched_forward

            ln_module.forward = make_patched(
                orig_forward, layer_idx, kind, collector, ln_module
            )
            hooks.append((ln_module, orig_forward))

    return hooks


def restore_layernorm_hooks(hooks: list) -> None:
    for ln_module, orig_forward in hooks:
        ln_module.forward = orig_forward
        if hasattr(ln_module, "_eval_ln_attn_mask"):
            delattr(ln_module, "_eval_ln_attn_mask")


def attach_attention_mask_to_layernorms(model, attention_mask: torch.Tensor) -> None:
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        for ln_module in (
            layer.attention.output.LayerNorm,
            layer.output.LayerNorm,
        ):
            ln_module._eval_ln_attn_mask = attention_mask


def clear_attention_mask_on_layernorms(model) -> None:
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        for ln_module in (
            layer.attention.output.LayerNorm,
            layer.output.LayerNorm,
        ):
            if hasattr(ln_module, "_eval_ln_attn_mask"):
                delattr(ln_module, "_eval_ln_attn_mask")


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


@torch.no_grad()
def population_var_batched(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    return ((x - mu) ** 2).mean(dim=-1)


@torch.no_grad()
def variance_stats_gpu(
    x: torch.Tensor,
    chunk_size: int = STATS_CHUNK_SIZE,
) -> dict[str, float]:
    if x.shape[0] == 0:
        return {
            "num_vectors": 0,
            "var_min": float("nan"),
            "var_max": float("nan"),
        }

    var_min = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
    var_max = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
    count = 0

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        v = population_var_batched(x[start:end])
        var_min = torch.minimum(var_min, v.min())
        var_max = torch.maximum(var_max, v.max())
        count += int(v.numel())

    return {
        "num_vectors": count,
        "var_min": float(var_min.cpu()),
        "var_max": float(var_max.cpu()),
    }


def _uses_streaming_eval(task_name: str) -> bool:
    """大验证集任务：边 hook 边测，禁止先收集全部 LN 输入。"""
    return task_name in ("mnli", "qnli")


def _extract_ln_params(
    model,
) -> dict[tuple[int, LayerNormKind], tuple[float, torch.Tensor, torch.Tensor]]:
    """(eps, gamma, beta)；gamma/beta 为 CPU float32，评估时再搬到输入设备。"""
    out: dict[tuple[int, LayerNormKind], tuple[float, torch.Tensor, torch.Tensor]] = {}
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        for kind, ln_module in (
            ("ln1", layer.attention.output.LayerNorm),
            ("ln2", layer.output.LayerNorm),
        ):
            out[(layer_idx, kind)] = (
                float(ln_module.eps),
                ln_module.weight.detach().to(dtype=torch.float32).cpu(),
                ln_module.bias.detach().to(dtype=torch.float32).cpu(),
            )
    return out


def _prepare_eval_model_and_loader(
    task_name: str,
    device: torch.device,
    num_samples: int | None,
    *,
    eval_split: str = "validation",
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[Any, DataLoader, str]:
    if eval_split not in ("validation", "calib"):
        raise ValueError(f"eval_split 须为 validation/calib，得到 {eval_split!r}")
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


def _run_forward_with_ln_hooks(model, loader, device, collector) -> None:
    hooks = register_layernorm_hooks(model, collector)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attach_attention_mask_to_layernorms(model, attn_mask)
            model(**batch)
            clear_attention_mask_on_layernorms(model)
    restore_layernorm_hooks(hooks)


def load_model_and_collect_inputs(
    task_name: str,
    device: torch.device,
    num_samples: int | None,
    *,
    eval_split: str = "validation",
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[Any, Any, str]:
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        num_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    collector = GpuLayerNormCollector(LAYER_ENABLE, device)
    _run_forward_with_ln_hooks(model, loader, device, collector)
    return model, collector, split_desc


def stream_var_ranges_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None = NUM_SAMPLES,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[
    dict[str, list[list[float]]],
    list[tuple[int, LayerNormKind, int, list[float]]],
    str,
]:
    """边 hook 边统计方差，不落地 LN 输入。"""
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        num_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    tracker = StreamingVarianceTracker(LAYER_ENABLE)
    print(f"  方差流式统计：{split_desc}")
    _run_forward_with_ln_hooks(model, loader, device, tracker)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    label = f"{task_name}/{eval_split}"
    var_ranges, variance_rows = tracker.finalize(label)
    return var_ranges, variance_rows, split_desc


def stream_error_eval_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    he_var_ranges: dict[str, list[list[float]]],
    num_samples: int | None = NUM_SAMPLES,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[dict[tuple[int, LayerNormKind], dict], str]:
    """边 hook 边算误差，不落地 LN 输入。"""
    model, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        num_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    tracker = StreamingErrorTracker(
        LAYER_ENABLE,
        he_var_ranges,
        layer_invsqrt_alpha_for_task(task_name, "ln1"),
        layer_invsqrt_alpha_for_task(task_name, "ln2"),
        layer_invsqrt_max_iters_for_task(task_name, "ln1"),
        layer_invsqrt_max_iters_for_task(task_name, "ln2"),
        _extract_ln_params(model),
    )
    _run_forward_with_ln_hooks(model, loader, device, tracker)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return tracker.finalize(), split_desc


def _subsample_rows_gpu(
    x: torch.Tensor, max_rows: int | None, seed: int
) -> torch.Tensor:
    n = x.shape[0]
    if max_rows is None or n <= max_rows:
        return x
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g, device=x.device)[:max_rows]
    return x[idx]


@torch.no_grad()
def evaluate_he_vs_reference_layernorm_gpu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    min_var: float,
    max_var: float,
    invsqrt_alpha: float,
    invsqrt_max_iters: int,
    chunk_size: int,
    max_rows: int | None,
    seed: int,
) -> dict[str, float]:
    x = _subsample_rows_gpu(x, max_rows, seed)
    n = x.shape[0]
    if n == 0:
        return {
            "max_abs_err": float("nan"),
            "mean_abs_err": float("nan"),
            "max_rel_pct": float("nan"),
            "mean_rel_pct": float("nan"),
            "num_vectors": 0,
            "invsqrt_iters": 0,
        }

    gamma = gamma.to(device=x.device, dtype=x.dtype)
    beta = beta.to(device=x.device, dtype=x.dtype)

    max_abs = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    sum_abs = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    max_rel = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    sum_rel = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    count = 0
    rel_count = 0
    last_invsqrt_iters = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = x[start:end]

        y_ref = reference_layernorm_batched(chunk, gamma, beta, eps)
        y_he, inv_iters = he_layernorm_batched(
            chunk,
            gamma,
            beta,
            eps,
            min_var,
            max_var,
            invsqrt_alpha,
            invsqrt_max_iters,
        )
        last_invsqrt_iters = inv_iters

        err = torch.abs(y_ref - y_he)
        max_abs = torch.maximum(max_abs, err.max())
        sum_abs = sum_abs + err.sum()
        count += int(err.numel())

        y_ref_abs = torch.abs(y_ref)
        denom_ok = y_ref_abs >= REL_ERR_DENOM_MIN
        if denom_ok.any():
            rel_pct = err[denom_ok] / y_ref_abs[denom_ok] * 100.0
            max_rel = torch.maximum(max_rel, rel_pct.max())
            sum_rel = sum_rel + rel_pct.sum()
            rel_count += int(rel_pct.numel())

    mean_rel_pct = float((sum_rel / rel_count).cpu()) if rel_count > 0 else float("nan")
    max_rel_pct = float(max_rel.cpu()) if rel_count > 0 else float("nan")

    return {
        "max_abs_err": float(max_abs.cpu()),
        "mean_abs_err": float((sum_abs / count).cpu()),
        "max_rel_pct": max_rel_pct,
        "mean_rel_pct": mean_rel_pct,
        "num_vectors": n,
        "num_elems": count,
        "invsqrt_iters": last_invsqrt_iters,
    }


def _print_variance_table(
    task_name: str,
    variance_rows: list[tuple[int, LayerNormKind, int, list[float]]],
) -> None:
    print(f"\n--- {task_name}  LayerNorm 输入方差 Var(x) 统计（原始，写入 JSON）---")
    print(f"{'层':>3} {'LN':>4} {'向量数':>10} {'var_range(raw)':>28}")
    for layer_idx, kind, num_vectors, var_range in variance_rows:
        print(
            f"{layer_idx:3d} {kind:>4} {num_vectors:10d} "
            f"{format_var_range(var_range):>28}"
        )


def _fmt_rel_pct(v: float) -> str:
    return f"{v:10.4f}" if math.isfinite(v) else f"{'nan':>10}"


def _fmt_err_block(st: dict | None) -> str:
    if st is None:
        return f"{'nan':>14} {'nan':>14} {'nan':>10} {'nan':>10}"
    return (
        f"{st['max_abs_err']:14.6g} {st['mean_abs_err']:14.6g} "
        f"{_fmt_rel_pct(st['max_rel_pct'])} {_fmt_rel_pct(st['mean_rel_pct'])}"
    )


def _print_error_table(
    task_name: str,
    error_results: dict[tuple[int, LayerNormKind], dict],
    *,
    dual_error_results: dict[tuple[int, LayerNormKind], dict] | None = None,
) -> None:
    """打印误差表。dual 非空时：上行=校验集，下行=验证集（数字列对齐）。"""
    dual = dual_error_results is not None
    print(f"\n--- {task_name}  HE LayerNorm vs standard LayerNorm [GPU 批量] ---")
    cap = (
        f"最多 {MAX_VECTORS_FOR_ERROR_EVAL} 条"
        if MAX_VECTORS_FOR_ERROR_EVAL
        else "全部"
    )
    print(f"    评估向量: {cap}, chunk={ERROR_EVAL_CHUNK_SIZE}")
    print(
        f"    相对误差: |y_he−y_ref|/|y_ref|×100% "
        f"(仅统计 |y_ref|≥{REL_ERR_DENOM_MIN:g})"
    )
    err_hdr = (
        f"{'max_abs_err':>14} {'mean_abs_err':>14} "
        f"{'max_rel%':>10} {'mean_rel%':>10}"
    )
    meta_hdr = f"{'层':>3} {'LN':>4} {'max_i':>5} {'isqrt':>5} "
    # "  0  ln1     4     4 " → 与数字列左对齐的空白前缀
    val_indent = " " * len(meta_hdr)
    if dual:
        print("    每组两行：上行=校验集(calib)，下行=验证集(validation)")
        print(f"{meta_hdr}{err_hdr}")
        keys = sorted(
            set(error_results.keys()) | set(dual_error_results.keys())
        )
        for key in keys:
            st_c = error_results.get(key)
            st_v = dual_error_results.get(key)
            st_meta = st_c or st_v
            assert st_meta is not None
            layer_idx, kind = key
            print(
                f"{layer_idx:3d} {kind:>4} "
                f"{st_meta['invsqrt_max_iters']:5d} "
                f"{st_meta['invsqrt_iters']:5d} "
                f"{_fmt_err_block(st_c)}"
            )
            print(f"{val_indent}{_fmt_err_block(st_v)}")
        return

    print(f"{meta_hdr}{err_hdr}")
    for (layer_idx, kind) in sorted(error_results.keys()):
        st = error_results[(layer_idx, kind)]
        print(
            f"{layer_idx:3d} {kind:>4} "
            f"{st['invsqrt_max_iters']:5d} {st['invsqrt_iters']:5d} "
            f"{_fmt_err_block(st)}"
        )


def collect_var_ranges_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None = NUM_SAMPLES,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> tuple[
    dict[str, list[list[float]]],
    list[tuple[int, LayerNormKind, int, list[float]]],
    str,
]:
    """在指定 split 上统计各层 ln1/ln2 原始 [var_min, var_max]。"""
    if eval_split not in ("validation", "calib"):
        raise ValueError(f"collect 仅支持 validation/calib，得到 {eval_split!r}")

    if _uses_streaming_eval(task_name):
        print(f"  {task_name}: 使用 streaming（边 hook 边统计，不缓存 LN 输入）")
        return stream_var_ranges_for_split(
            task_name,
            device,
            eval_split=eval_split,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )

    model, collector, split_desc = load_model_and_collect_inputs(
        task_name,
        device,
        num_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    print(f"  方差统计数据：{split_desc}")
    var_ranges: dict[str, list[list[float]]] = {"ln1": [], "ln2": []}
    variance_rows: list[tuple[int, LayerNormKind, int, list[float]]] = []
    for layer_idx in range(NUM_LAYERS):
        if not collector.layer_enable[layer_idx]:
            continue
        for kind in ("ln1", "ln2"):
            inputs = collector.stacked(layer_idx, kind)
            if inputs.shape[0] == 0:
                continue
            stats = variance_stats_gpu(inputs)
            raw_range = [float(stats["var_min"]), float(stats["var_max"])]
            var_ranges[kind].append(raw_range)
            variance_rows.append(
                (layer_idx, kind, stats["num_vectors"], raw_range)
            )
            collector.release(layer_idx, kind)
    if len(var_ranges["ln1"]) != NUM_LAYERS or len(var_ranges["ln2"]) != NUM_LAYERS:
        raise ValueError(f"{task_name}/{eval_split} 方差统计不完整")
    var_ranges = _validate_var_ranges(task_name, var_ranges)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return var_ranges, variance_rows, split_desc


def recompute_union_variance(
    task_name: str,
    device: torch.device,
    *,
    num_samples: int | None = NUM_SAMPLES,
    json_dir: str = JSON_DIR,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
) -> dict[str, Any]:
    """分别统计 calib / validation，并集写入 {task}_variance.json。"""
    print(f"\n=== {task_name}: 重算方差（calib ∪ validation）===")
    calib_ranges, calib_rows, _ = collect_var_ranges_for_split(
        task_name,
        device,
        eval_split="calib",
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    _print_variance_table(f"{task_name}/calib", calib_rows)

    val_ranges, val_rows, _ = collect_var_ranges_for_split(
        task_name,
        device,
        eval_split="validation",
        num_samples=num_samples,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    _print_variance_table(f"{task_name}/validation", val_rows)

    union = _validate_var_ranges(task_name, merge_var_ranges(calib_ranges, val_ranges))
    union_rows: list[tuple[int, LayerNormKind, int, list[float]]] = []
    for kind in ("ln1", "ln2"):
        for layer_idx, rng in enumerate(union[kind]):
            union_rows.append((layer_idx, kind, -1, rng))
    _print_variance_table(f"{task_name}/union", union_rows)
    union_path = save_var_ranges_json(
        task_name,
        union,
        json_dir,
        sources=["validation", "calib"],
    )
    print(
        f"    已写入并集原始区间 {union_path} "
        f"(使用时 ×[{VAR_MIN_SCALE}, {VAR_MAX_SCALE}])"
    )
    return {
        "calib_ranges": calib_ranges,
        "validation_ranges": val_ranges,
        "union_ranges": union,
        "union_path": union_path,
    }


def run_task(
    task_name: str,
    device: torch.device,
    *,
    num_samples: int | None = NUM_SAMPLES,
    json_dir: str = JSON_DIR,
    eval_split: str = "validation",
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    print_errors: bool = True,
    he_var_ranges: dict[str, list[list[float]]] | None = None,
    he_var_source: str | None = None,
) -> dict[str, Any]:
    if eval_split not in ("validation", "calib"):
        raise ValueError(
            f"run_task 的 eval_split 须为 validation/calib，得到 {eval_split!r}"
        )

    if he_var_ranges is None:
        he_var_ranges, he_var_source = load_var_ranges_from_json(
            task_name, json_dir
        )
    assert he_var_source is not None
    print(f"  {task_name}: HE 用并集方差 ← {he_var_source}")

    if _uses_streaming_eval(task_name):
        print(f"  {task_name}: 使用 streaming（边 hook 边评估，不缓存 LN 输入）")
        error_results, split_desc = stream_error_eval_for_split(
            task_name,
            device,
            eval_split=eval_split,
            he_var_ranges=he_var_ranges,
            num_samples=num_samples,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  评估数据：{split_desc}")
    else:
        model, collector, split_desc = load_model_and_collect_inputs(
            task_name,
            device,
            num_samples,
            eval_split=eval_split,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        print(f"  评估数据：{split_desc}")

        ln1_alphas = layer_invsqrt_alpha_for_task(task_name, "ln1")
        ln2_alphas = layer_invsqrt_alpha_for_task(task_name, "ln2")
        ln1_max_iters = layer_invsqrt_max_iters_for_task(task_name, "ln1")
        ln2_max_iters = layer_invsqrt_max_iters_for_task(task_name, "ln2")

        error_results: dict[tuple[int, LayerNormKind], dict] = {}

        for layer_idx in range(NUM_LAYERS):
            if not collector.layer_enable[layer_idx]:
                continue
            layer = model.bert.encoder.layer[layer_idx]
            for kind, ln_module, alphas, max_iters_list in (
                ("ln1", layer.attention.output.LayerNorm, ln1_alphas, ln1_max_iters),
                ("ln2", layer.output.LayerNorm, ln2_alphas, ln2_max_iters),
            ):
                inputs = collector.stacked(layer_idx, kind)
                if inputs.shape[0] == 0:
                    continue

                raw_he = he_var_ranges[kind][layer_idx]
                var_range = scaled_var_range(float(raw_he[0]), float(raw_he[1]))

                eps = float(ln_module.eps)
                gamma = ln_module.weight.detach()
                beta = ln_module.bias.detach()
                min_var, max_var = float(var_range[0]), float(var_range[1])
                invsqrt_alpha = alphas[layer_idx]
                invsqrt_max_iters = max_iters_list[layer_idx]

                err = evaluate_he_vs_reference_layernorm_gpu(
                    inputs,
                    gamma,
                    beta,
                    eps,
                    min_var,
                    max_var,
                    invsqrt_alpha,
                    invsqrt_max_iters,
                    ERROR_EVAL_CHUNK_SIZE,
                    MAX_VECTORS_FOR_ERROR_EVAL,
                    RANDOM_SEED + layer_idx * 2 + (0 if kind == "ln1" else 1),
                )
                error_results[(layer_idx, kind)] = {
                    **err,
                    "eps": eps,
                    "hidden_dim": int(inputs.shape[-1]),
                    "min_var": min_var,
                    "max_var": max_var,
                    "invsqrt_alpha": invsqrt_alpha,
                    "invsqrt_max_iters": invsqrt_max_iters,
                }

                collector.release(layer_idx, kind)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if print_errors:
        _print_error_table(task_name, error_results)

    return {
        "var_ranges": he_var_ranges,
        "var_source": he_var_source,
        "he_var_source": he_var_source,
        "eval_split": eval_split,
        "split_desc": split_desc,
        "error_results": error_results,
    }


def run_task_all(
    task_name: str,
    device: torch.device,
    *,
    num_samples: int | None = NUM_SAMPLES,
    json_dir: str = JSON_DIR,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    he_var_ranges: dict[str, list[list[float]]] | None = None,
    he_var_source: str | None = None,
) -> dict[str, Any]:
    """先 calib 再 validation，合并打印误差表。HE 始终读并集 {task}_variance.json。"""
    if he_var_ranges is None:
        he_var_ranges, he_var_source = load_var_ranges_from_json(
            task_name, json_dir
        )
    common = dict(
        num_samples=num_samples,
        json_dir=json_dir,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
        print_errors=False,
        he_var_ranges=he_var_ranges,
        he_var_source=he_var_source,
    )
    r_calib = run_task(task_name, device, eval_split="calib", **common)
    r_val = run_task(task_name, device, eval_split="validation", **common)
    _print_error_table(
        task_name,
        r_calib["error_results"],
        dual_error_results=r_val["error_results"],
    )
    return {
        "eval_split": "all",
        "he_var_source": he_var_source,
        "calib": r_calib,
        "validation": r_val,
        "error_results_calib": r_calib["error_results"],
        "error_results_validation": r_val["error_results"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LayerNorm 方差区间统计 + HE 近似误差评估"
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="任务名（可多次指定）；默认跑 TASK_NAMES 全部",
    )
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
        default=None,
        help=(
            f"仅 validation：子集大小；默认 {NUM_SAMPLES!r}（全量）。"
            "calib 忽略此项"
        ),
    )
    parser.add_argument(
        "--calib-seed",
        type=int,
        default=CALIB_SEED,
        help=(
            f"仅 calib：读取 {{task}}_calib_indices_seed{{seed}}.json "
            f"（默认 {CALIB_SEED}）"
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
        help=f"方差 JSON 目录（默认 {JSON_DIR}）",
    )
    parser.add_argument(
        "--recompute-variance",
        action="store_true",
        help=(
            "重算 calib+validation 方差并集，写入 {task}_variance.json"
        ),
    )
    parser.add_argument(
        "--variance-only",
        action="store_true",
        help="仅统计/更新方差区间，不做误差评估",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA，将使用 CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = args.tasks if args.tasks else list(TASK_NAMES)

    print("layernorm.py — LayerNorm 方差区间 + HE 近似误差评估")
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"方差 JSON 目录：{args.json_dir}")
    print(f"eval-split：{args.eval_split}")
    if args.eval_split == "all":
        print("  all：先评估校验集，再评估验证集；误差表每组两行（上 calib / 下 val）")
    print("HE 方差：始终读 {task}_variance.json（calib∪validation 并集）")
    if args.eval_split in ("calib", "all") or args.recompute_variance:
        print(
            f"  calib indices: seed={args.calib_seed} "
            f"dir={args.calib_indices_dir}（只读文件，不重建）"
        )
    if args.recompute_variance:
        print("  --recompute-variance：写并集 {task}_variance.json")
    print(
        f"方差区间：{'重新统计并写入' if args.recompute_variance else '从 JSON 加载'}"
    )
    if args.variance_only:
        print("模式：仅方差统计")
        if not args.recompute_variance:
            print("提示：--variance-only 未加 --recompute-variance 时不做统计写入")

    for task_name in tasks:
        try:
            he_ranges = None
            he_source = None
            if args.recompute_variance:
                out = recompute_union_variance(
                    task_name,
                    device,
                    num_samples=args.samples,
                    json_dir=args.json_dir,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                )
                he_ranges = out["union_ranges"]
                he_source = out["union_path"]
            if args.variance_only:
                continue
            if args.eval_split == "all":
                run_task_all(
                    task_name,
                    device,
                    num_samples=args.samples,
                    json_dir=args.json_dir,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                    he_var_ranges=he_ranges,
                    he_var_source=he_source,
                )
            else:
                run_task(
                    task_name,
                    device,
                    num_samples=args.samples,
                    json_dir=args.json_dir,
                    eval_split=args.eval_split,
                    calib_seed=args.calib_seed,
                    calib_indices_dir=args.calib_indices_dir,
                    he_var_ranges=he_ranges,
                    he_var_source=he_source,
                )
        except FileNotFoundError as e:
            msg = str(e)
            if "模型不存在" in msg:
                print(f"跳过 {task_name}：{e}")
            else:
                raise

    print("完成。")


if __name__ == "__main__":
    main()
