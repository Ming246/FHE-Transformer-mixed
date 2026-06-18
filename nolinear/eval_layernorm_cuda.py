"""
eval_layernorm 的 GPU 批量版：按 HE LayerNorm 流程评估近似误差。

算法（对齐 he_layernorm / he_invsqrt，忽略 CKKS 工程细节）：
  0. x' ← x / (√((max_var·1.05+ε)·n²) · input_extra_div)
  1. numerator ← n·x' − Σx'
  2. variance ← n·Σ(x')² − (Σx')² + ε / max_for_denominator
  3. inv_sqrt ← he_invsqrt(variance, e₀=min_var/max_var, alpha)
  4. y ← numerator ⊗ γ ⊗ inv_sqrt ⊕ β（ln2 含 HE 编码固定的 ×2）

参考实现为标准 LayerNorm（F.layer_norm）。

用法：python3 nolinear/eval_layernorm_cuda.py
"""
from __future__ import annotations

import functools
import math
import os
from datetime import datetime

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

LAYER_ENABLE = [True] * NUM_LAYERS

# he_invsqrt 停止阈值 alpha：当 e_n >= 1 - alpha 时结束迭代
# alpha 越小 → 要求 e_n 更接近 1 → 迭代更多 → 1/sqrt 精度越高
INV_SQRT_MAX_ITERS = 50
W_BUFFER = 1.05
DEFAULT_INV_SQRT_ALPHA = 0.001

# ln1 = attention.output.LayerNorm, ln2 = output.LayerNorm
# 每层 [min_var, max_var]：InvertSqrt 方差预期区间（见 he_layernorm1/2/3）
_HE_LN1_VAR: list[float] = [0.15, 10.0]
_HE_LN2_VAR: list[float] = [0.2, 150.0]

LAYER_VAR_RANGE_BY_TASK: dict[str, dict[str, list[list[float]]]] = {
    "mrpc": {
        "ln1": [_HE_LN1_VAR] * NUM_LAYERS,
        "ln2": [_HE_LN2_VAR] * NUM_LAYERS,
    },
    "rte": {
        "ln1": [_HE_LN1_VAR] * NUM_LAYERS,
        "ln2": [_HE_LN2_VAR] * NUM_LAYERS,
    },
    "sst2": {
        "ln1": [_HE_LN1_VAR] * NUM_LAYERS,
        "ln2": [_HE_LN2_VAR] * NUM_LAYERS,
    },
}

# 每层 he_invsqrt 的 alpha（停止精度；越小精度越高、迭代越多）
LAYER_INV_SQRT_ALPHA_BY_TASK: dict[str, dict[str, list[float]]] = {
    "mrpc": {
        "ln1": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
        "ln2": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
    },
    "rte": {
        "ln1": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
        "ln2": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
    },
    "sst2": {
        "ln1": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
        "ln2": [DEFAULT_INV_SQRT_ALPHA] * NUM_LAYERS,
    },
}
# =============================================================================


LayerNormKind = str  # "ln1" | "ln2"


def layer_var_range_for_task(
    task_name: str, kind: LayerNormKind
) -> list[list[float]]:
    if task_name not in LAYER_VAR_RANGE_BY_TASK:
        raise KeyError(f"任务 {task_name} 未配置 LAYER_VAR_RANGE_BY_TASK")
    table = LAYER_VAR_RANGE_BY_TASK[task_name]
    if kind not in table:
        raise KeyError(f"未知 LayerNorm 类型：{kind}")
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
    return ranges


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


def _he_encoding_scale(kind: LayerNormKind) -> float:
    """HE 编码约定：ln2/ln3 输入 mask/2、输出 out+out，ln1 无额外缩放。"""
    return 2.0 if kind == "ln2" else 1.0


def _he_invsqrt_kn(en: float) -> float:
    """与 he_invsqrt 中 np.roots(...)[1] 一致。"""
    coeffs = [1.0 - en**3, 6.0 * en**2 - 6.0, 9.0 - 9.0 * en]
    roots = np.roots(coeffs)
    return float(roots[1].real)


def he_invsqrt_batched(
    variance: torch.Tensor,
    e_init: float,
    alpha: float,
    max_iters: int = INV_SQRT_MAX_ITERS,
) -> tuple[torch.Tensor, int]:
    """
    自适应 he_invsqrt：求 1/sqrt(variance)。
    variance: [N] 正标量；e_init = min_var/max_var。
    alpha: 停止阈值，当 e_n >= 1 - alpha 时结束（越小精度越高）。
    返回 (inv_sqrt, 迭代次数)。
    """
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
    kind: LayerNormKind,
    invsqrt_alpha: float,
) -> tuple[torch.Tensor, int]:
    """
    HE LayerNorm，x: [N, D]；gamma, beta: [D]。
    ln2 的 HE 编码缩放（输入 /2、输出 ×2）由 kind 固定，不在配置里调节。
    """
    n = float(x.shape[-1])
    enc_scale = _he_encoding_scale(kind)

    max_for_denominator = (max_var * W_BUFFER + var_e) * (n**2)
    scale = 1.0 / math.sqrt(max_for_denominator) / enc_scale
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
    if enc_scale != 1.0:
        out = out * enc_scale
    return out, inv_iters


def reference_layernorm_batched(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """标准 LayerNorm，与 nn.LayerNorm 一致。"""
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
        # hidden: [B, S, D]
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


def register_layernorm_hooks(
    model,
    collector: GpuLayerNormCollector,
) -> list:
    """在 LayerNorm 前截取输入；通过 patched forward 传入 attention_mask。"""
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
    """将当前 batch 的 attention_mask 挂到各 LayerNorm 模块上供 hook 过滤 padding。"""
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
    kind: LayerNormKind,
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
            kind,
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


def analyze_layernorm_errors_gpu(
    task_name: str,
    model,
    collector: GpuLayerNormCollector,
    ln1_var_ranges: list[list[float]],
    ln2_var_ranges: list[list[float]],
    ln1_alphas: list[float],
    ln2_alphas: list[float],
) -> dict[tuple[int, LayerNormKind], dict]:
    results: dict[tuple[int, LayerNormKind], dict] = {}

    for layer_idx in range(NUM_LAYERS):
        if not collector.layer_enable[layer_idx]:
            continue
        layer = model.bert.encoder.layer[layer_idx]
        for kind, ln_module, var_ranges, alphas in (
            ("ln1", layer.attention.output.LayerNorm, ln1_var_ranges, ln1_alphas),
            ("ln2", layer.output.LayerNorm, ln2_var_ranges, ln2_alphas),
        ):
            inputs = collector.stacked(layer_idx, kind)
            if inputs.shape[0] == 0:
                continue

            eps = float(ln_module.eps)
            gamma = ln_module.weight.detach()
            beta = ln_module.bias.detach()
            min_var, max_var = var_ranges[layer_idx]
            invsqrt_alpha = alphas[layer_idx]

            err = evaluate_he_vs_reference_layernorm_gpu(
                inputs,
                gamma,
                beta,
                eps,
                min_var,
                max_var,
                kind,
                invsqrt_alpha,
                ERROR_EVAL_CHUNK_SIZE,
                MAX_VECTORS_FOR_ERROR_EVAL,
                RANDOM_SEED + layer_idx * 2 + (0 if kind == "ln1" else 1),
            )
            results[(layer_idx, kind)] = {
                **err,
                "eps": eps,
                "hidden_dim": int(inputs.shape[-1]),
                "min_var": min_var,
                "max_var": max_var,
                "invsqrt_alpha": invsqrt_alpha,
            }
            collector.release(layer_idx, kind)

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
    for (layer_idx, kind) in sorted(results.keys()):
        st = results[(layer_idx, kind)]
        print(
            f"{layer_idx:3d} {kind:>4} {st['hidden_dim']:4d} "
            f"{st['min_var']:7.3g} {st['max_var']:7.3g} "
            f"{st['invsqrt_alpha']:7.4g} {st['invsqrt_iters']:5d} "
            f"{st['num_vectors']:10d} {st['max_abs_err']:14.6g} "
            f"{st['mean_abs_err']:14.6g}"
        )

    return results


def run_task(task_name: str, device: torch.device) -> dict:
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
        dataset["validation"], NUM_SAMPLES, RANDOM_SEED
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

    ln1_var_ranges = layer_var_range_for_task(task_name, "ln1")
    ln2_var_ranges = layer_var_range_for_task(task_name, "ln2")
    ln1_alphas = layer_invsqrt_alpha_for_task(task_name, "ln1")
    ln2_alphas = layer_invsqrt_alpha_for_task(task_name, "ln2")

    return analyze_layernorm_errors_gpu(
        task_name,
        model,
        collector,
        ln1_var_ranges,
        ln2_var_ranges,
        ln1_alphas,
        ln2_alphas,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA，将使用 CPU（仍走批量 torch，但无 GPU 加速）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("eval_layernorm_cuda.py — GPU 批量 HE LayerNorm 误差评估")
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for task_name in TASK_NAMES:
        try:
            layer_var_range_for_task(task_name, "ln1")
            layer_var_range_for_task(task_name, "ln2")
            layer_invsqrt_alpha_for_task(task_name, "ln1")
            layer_invsqrt_alpha_for_task(task_name, "ln2")
            run_task(task_name, device)
        except FileNotFoundError as e:
            print(f"跳过 {task_name}：{e}")

    print("完成。")


if __name__ == "__main__":
    main()
