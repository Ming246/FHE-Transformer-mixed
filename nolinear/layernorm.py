"""
LayerNorm HE 近似评估：方差区间统计 + 与标准 LayerNorm 的逐层误差。

算法（M 缩放 + he_invsqrt；不含 CKKS 编码 bookend）：
  0. x' ← x / √((max_var·1.05+ε)·n²)
  1. numerator ← n·x' − Σx'
  2. variance ← n·Σ(x')² − (Σx')² + ε / max_for_denominator
  3. inv_sqrt ← he_invsqrt(variance, e₀=min_var/max_var, alpha)
  4. y ← numerator ⊗ γ ⊗ inv_sqrt ⊕ β

用法（在 nolinear/ 下）：
  python3 layernorm.py                         # 读 JSON，做误差评估
  python3 layernorm.py --recompute-variance    # 重算方差区间、写 JSON、再评估
  python3 layernorm.py --task mrpc --samples 32 --recompute-variance
  python3 layernorm.py --variance-only --recompute-variance  # 仅统计方差
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
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

# ===================== 配置 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2

NUM_SAMPLES = None
RANDOM_SEED = 42
BATCH_SIZE = 8

MAX_VECTORS_FOR_ERROR_EVAL: int | None = None
ERROR_EVAL_CHUNK_SIZE = 4096
STATS_CHUNK_SIZE = 4096

LAYER_ENABLE = [True] * NUM_LAYERS

INV_SQRT_MAX_ITERS = 50
W_BUFFER = 1.05
DEFAULT_INV_SQRT_ALPHA = 0.001

VAR_MIN_SCALE = 0.9
VAR_MAX_SCALE = 1.1

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(_SCRIPT_DIR, "variance_out")

LAYER_INV_SQRT_ALPHA_BY_TASK: dict[str, dict[str, list[float]]] = {
    task: {
        "ln1": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
        "ln2": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
    }
    for task in TASK_NAMES
}
# =============================================================================

LayerNormKind = str  # "ln1" | "ln2"


def scaled_var_range(var_min: float, var_max: float) -> list[float]:
    """输出用方差区间：min×0.9，max×1.1。"""
    return [var_min * VAR_MIN_SCALE, var_max * VAR_MAX_SCALE]


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
        raise FileNotFoundError(f"方差 JSON 不存在：{path}")

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    json_task = payload.get("task")
    if json_task and json_task != task_name:
        raise ValueError(
            f"{path} 中 task={json_task!r} 与请求任务 {task_name!r} 不一致"
        )

    table = _validate_var_ranges(task_name, _parse_variance_json(payload))
    return table, path


def save_var_ranges_json(
    task_name: str,
    var_ranges: dict[str, list[list[float]]],
    json_dir: str = JSON_DIR,
) -> str:
    os.makedirs(json_dir, exist_ok=True)
    path = variance_json_path(task_name, json_dir)
    payload = {
        "task": task_name,
        "ln1": var_ranges["ln1"],
        "ln2": var_ranges["ln2"],
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
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


def _he_invsqrt_kn(en: float) -> float:
    coeffs = [1.0 - en**3, 6.0 * en**2 - 6.0, 9.0 - 9.0 * en]
    roots = np.roots(coeffs)
    return float(roots[1].real)


def he_invsqrt_batched(
    variance: torch.Tensor,
    e_init: float,
    alpha: float,
    max_iters: int = INV_SQRT_MAX_ITERS,
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
    """收集各层 LayerNorm 输入向量 [N, D]。"""

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
        rows = hidden.detach().reshape(-1, hidden.shape[-1]).to(
            self.device, dtype=torch.float32
        )
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).to(self.device)
            valid = mask > 0
            if valid.any():
                rows = rows[valid]
            else:
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


def register_layernorm_hooks(model, collector: GpuLayerNormCollector) -> list:
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


def load_model_and_collect_inputs(
    task_name: str, device: torch.device, num_samples: int | None
) -> tuple[Any, GpuLayerNormCollector]:
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

    collector = GpuLayerNormCollector(LAYER_ENABLE, device)
    hooks = register_layernorm_hooks(model, collector)
    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            attn_mask = batch.get("attention_mask")
            if attn_mask is not None:
                attach_attention_mask_to_layernorms(model, attn_mask)
            model(**batch)
            clear_attention_mask_on_layernorms(model)

    restore_layernorm_hooks(hooks)
    return model, collector


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
            "num_vectors": 0,
            "invsqrt_iters": 0,
        }

    gamma = gamma.to(device=x.device, dtype=x.dtype)
    beta = beta.to(device=x.device, dtype=x.dtype)

    max_abs = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    sum_abs = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    count = 0
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
        )
        last_invsqrt_iters = inv_iters

        err = torch.abs(y_ref - y_he)
        max_abs = torch.maximum(max_abs, err.max())
        sum_abs = sum_abs + err.sum()
        count += int(err.numel())

    return {
        "max_abs_err": float(max_abs.cpu()),
        "mean_abs_err": float((sum_abs / count).cpu()),
        "num_vectors": n,
        "num_elems": count,
        "invsqrt_iters": last_invsqrt_iters,
    }


def _print_variance_table(
    task_name: str,
    variance_rows: list[tuple[int, LayerNormKind, int, list[float]]],
) -> None:
    print(f"\n--- {task_name}  LayerNorm 输入方差 Var(x) 统计 ---")
    print(f"{'层':>3} {'LN':>4} {'向量数':>10} {'var_range':>28}")
    for layer_idx, kind, num_vectors, var_range in variance_rows:
        print(
            f"{layer_idx:3d} {kind:>4} {num_vectors:10d} "
            f"{format_var_range(var_range):>28}"
        )


def _print_error_table(
    task_name: str,
    error_results: dict[tuple[int, LayerNormKind], dict],
) -> None:
    print(f"\n--- {task_name}  HE LayerNorm vs standard LayerNorm [GPU 批量] ---")
    cap = (
        f"最多 {MAX_VECTORS_FOR_ERROR_EVAL} 条"
        if MAX_VECTORS_FOR_ERROR_EVAL
        else "全部"
    )
    print(f"    评估向量: {cap}, chunk={ERROR_EVAL_CHUNK_SIZE}")
    print(
        f"{'层':>3} {'LN':>4} {'D':>4} {'min_v':>7} {'max_v':>7} "
        f"{'alpha':>7} {'isqrt':>5} "
        f"{'向量数':>10} {'max_abs_err':>14} {'mean_abs_err':>14}"
    )
    for (layer_idx, kind) in sorted(error_results.keys()):
        st = error_results[(layer_idx, kind)]
        print(
            f"{layer_idx:3d} {kind:>4} {st['hidden_dim']:4d} "
            f"{st['min_var']:7.3g} {st['max_var']:7.3g} "
            f"{st['invsqrt_alpha']:7.4g} {st['invsqrt_iters']:5d} "
            f"{st['num_vectors']:10d} {st['max_abs_err']:14.6g} "
            f"{st['mean_abs_err']:14.6g}"
        )


def run_task(
    task_name: str,
    device: torch.device,
    *,
    recompute_variance: bool = False,
    num_samples: int | None = NUM_SAMPLES,
    json_dir: str = JSON_DIR,
    variance_only: bool = False,
) -> dict[str, Any]:
    model, collector = load_model_and_collect_inputs(task_name, device, num_samples)

    if recompute_variance:
        var_ranges: dict[str, list[list[float]]] = {"ln1": [], "ln2": []}
        var_source = "recomputed"
    else:
        var_ranges, var_source = load_var_ranges_from_json(task_name, json_dir)
        print(f"  {task_name}: 已加载方差区间 {var_source}")

    ln1_alphas = layer_invsqrt_alpha_for_task(task_name, "ln1")
    ln2_alphas = layer_invsqrt_alpha_for_task(task_name, "ln2")

    variance_rows: list[tuple[int, LayerNormKind, int, list[float]]] = []
    error_results: dict[tuple[int, LayerNormKind], dict] = {}

    for layer_idx in range(NUM_LAYERS):
        if not collector.layer_enable[layer_idx]:
            continue
        layer = model.bert.encoder.layer[layer_idx]
        for kind, ln_module, alphas in (
            ("ln1", layer.attention.output.LayerNorm, ln1_alphas),
            ("ln2", layer.output.LayerNorm, ln2_alphas),
        ):
            inputs = collector.stacked(layer_idx, kind)
            if inputs.shape[0] == 0:
                continue

            if recompute_variance:
                stats = variance_stats_gpu(inputs)
                var_range = scaled_var_range(stats["var_min"], stats["var_max"])
                var_ranges[kind].append(var_range)
                variance_rows.append(
                    (layer_idx, kind, stats["num_vectors"], var_range)
                )
            else:
                var_range = var_ranges[kind][layer_idx]

            if not variance_only:
                eps = float(ln_module.eps)
                gamma = ln_module.weight.detach()
                beta = ln_module.bias.detach()
                min_var, max_var = float(var_range[0]), float(var_range[1])
                invsqrt_alpha = alphas[layer_idx]

                err = evaluate_he_vs_reference_layernorm_gpu(
                    inputs,
                    gamma,
                    beta,
                    eps,
                    min_var,
                    max_var,
                    invsqrt_alpha,
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
                }

            collector.release(layer_idx, kind)

    if recompute_variance:
        if len(var_ranges["ln1"]) != NUM_LAYERS or len(var_ranges["ln2"]) != NUM_LAYERS:
            raise ValueError(f"{task_name} 方差统计不完整，无法写入 JSON")
        var_ranges = _validate_var_ranges(task_name, var_ranges)
        _print_variance_table(task_name, variance_rows)
        path = save_var_ranges_json(task_name, var_ranges, json_dir)
        print(f"    已写入 {path}")

    if not variance_only:
        _print_error_table(task_name, error_results)

    return {
        "var_ranges": var_ranges,
        "var_source": var_source if not recompute_variance else "recomputed",
        "variance_rows": variance_rows,
        "error_results": error_results,
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
        "--samples",
        type=int,
        default=None,
        help=f"验证集抽样数（默认 {NUM_SAMPLES!r} 表示全量）",
    )
    parser.add_argument(
        "--json-dir",
        default=JSON_DIR,
        help=f"方差 JSON 目录（默认 {JSON_DIR}）",
    )
    parser.add_argument(
        "--recompute-variance",
        action="store_true",
        help="重新统计方差区间并写入 JSON；默认从 JSON 加载",
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
    print(
        f"方差区间：{'重新统计' if args.recompute_variance else '从 JSON 加载'}"
    )
    if args.variance_only:
        print("模式：仅方差统计")

    for task_name in tasks:
        try:
            run_task(
                task_name,
                device,
                recompute_variance=args.recompute_variance,
                num_samples=args.samples,
                json_dir=args.json_dir,
                variance_only=args.variance_only,
            )
        except FileNotFoundError as e:
            if "模型不存在" in str(e):
                print(f"跳过 {task_name}：{e}")
            else:
                raise

    print("完成。")


if __name__ == "__main__":
    main()
