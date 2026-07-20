"""
eval_softmax 的 GPU 批量版：thor_softmax 全向量化，固定步数 Goldschmidt 求倒数。

与 eval_softmax.py 逻辑一致，但：
  - 分数/掩码留在 CUDA 上拼接为 [N, L]
  - thor / reference softmax 按批在 GPU 上计算
  - 倒数使用 goldschmidt_inverse（固定 iterations），不用 adaptive

原始 CPU 单行实现见 eval_softmax.py（未修改）。

用法：python3 nolinear/eval_softmax_cuda.py
"""
from __future__ import annotations

import functools
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

# ===================== 配置（与 eval_softmax.py 对齐，可独立修改）=====================
# TASK_NAMES = ["mrpc", "rte", "sst2"]
TASK_NAMES = ["mrpc"]
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
        14, 15.5, 34, 18.5, 15.5, 16.5,
        15.5, 16.5, 18.5, 16.5, 32.0, 15.5,
    ],
}

LAYER_EXP_DIV_BY_TASK: dict[str, list[float]] = {
    "mrpc": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "rte": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    "sst2": [2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0],
}

# 每层 Goldschmidt 固定迭代次数（按数据集 × 层，与 delta2 配置方式相同）
LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK: dict[str, list[int]] = {
    "mrpc": [6, 9, 7, 10, 9, 9, 9, 9, 13, 13, 10, 11],
    #"sst2": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
    "rte": [6, 9, 7, 8, 6, 7, 6, 8, 7, 7, 6, 5],
    "sst2": [5, 6, 8, 8, 6, 7, 6, 6, 8, 7, 7, 7],
}

LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK: dict[str, list[int]] = {
    "mrpc": [8, 8, 8, 8, 8, 7, 7, 7, 8, 7, 8, 7],
    "rte": [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    "sst2": [7, 7, 7, 8, 7, 7, 8, 8, 7, 7, 7, 7],
}
LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["mrpc"] = [x-2  for x in LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["mrpc"]]
LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["mrpc"] = [x-2 for x in LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["mrpc"]]

# LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["rte"] = [x for x in LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["rte"]]
# LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["rte"] = [x for x in LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["rte"]]

# LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["sst2"] = [x  for x in LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK["sst2"]]
# LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["sst2"] = [x for x in LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK["sst2"]]

MAX_VECTORS_FOR_ERROR_EVAL: int | None = None
# GPU 上误差评估分块大小（防 OOM）
ERROR_EVAL_CHUNK_SIZE = 4096

LAYER_ENABLE = [True] * NUM_LAYERS

# thor exp 多项式系数（与 eval_softmax.py / softmax.py 一致）
# 与 cost.SOFTMAX_DEPTH_BASE 一致：Stockmeyer(4) + δ1 square(1)
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


def layer_goldschmidt_sigma_iters_for_task(task_name: str) -> list[int]:
    if task_name not in LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK:
        raise KeyError(
            f"任务 {task_name} 未配置 LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK"
        )
    iters = LAYER_GOLDSCHMIDT_ITERATIONS_SIGMA_BY_TASK[task_name]
    if len(iters) != NUM_LAYERS:
        raise ValueError(f"{task_name} goldschmidt_sigma 长度应为 {NUM_LAYERS}")
    for i, n in enumerate(iters):
        if n < 1:
            raise ValueError(f"{task_name} 层{i} goldschmidt_sigma 迭代次数须 >= 1")
    return iters


def layer_goldschmidt_sum_sq_iters_for_task(task_name: str) -> list[int]:
    if task_name not in LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK:
        raise KeyError(
            f"任务 {task_name} 未配置 LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK"
        )
    iters = LAYER_GOLDSCHMIDT_ITERATIONS_SUM_SQ_BY_TASK[task_name]
    if len(iters) != NUM_LAYERS:
        raise ValueError(f"{task_name} goldschmidt_sum_sq 长度应为 {NUM_LAYERS}")
    for i, n in enumerate(iters):
        if n < 1:
            raise ValueError(f"{task_name} 层{i} goldschmidt_sum_sq 迭代次数须 >= 1")
    return iters


def _check_power_of_two(name: str, val: float) -> int:
    n = int(round(math.log2(val)))
    if 2**n != int(val) and abs(2**n - val) > 1e-6:
        raise ValueError(f"{name}={val} 须为 2 的整数幂")
    return n


def log2_ceil(n: float) -> int:
    if n <= 0:
        raise ValueError(f"log2_ceil 要求 n > 0，当前 n={n}")
    return math.ceil(math.log2(n))


def softmax_layer_depth(
    delta2: float, goldschmidt_iterations_sigma: int, goldschmidt_iterations_sum_sq: int
) -> int:
    """单层 thor_softmax HE 乘法深度（与 cost.softmax_poly_depth 公式一致）。"""
    return (
        SOFTMAX_DEPTH_BASE
        + goldschmidt_iterations_sigma * 2
        + log2_ceil(delta2) * goldschmidt_iterations_sum_sq * 2
    )


def goldschmidt_inverse_batched(x: torch.Tensor, iterations: int) -> torch.Tensor:
    """
    固定迭代次数的 Goldschmidt 倒数（向量化）。
    x: 任意形状，逐元素 1/x 近似；常用 shape [N]（每行 sigma 或 sum(y^2)）。
    与 eval_softmax.py 中 goldschmidt_inverse 标量版公式一致。
    """
    y = 1.0 - x
    result = 2.0 - x
    for _ in range(iterations):
        y = y * y
        tmp = 1.0 + y
        result = result * tmp
    return result


def thor_softmax_batched(
    x: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    mask: torch.Tensor,
    goldschmidt_iterations_sigma: int,
    goldschmidt_iterations_sum_sq: int,
) -> torch.Tensor:
    """
    批量 thor_softmax。
    x, mask: [N, L]，mask 为 1/0（float）。
    """
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)
    n2 = _check_power_of_two("delta2", d2)

    x_scaled = x_work / d1 / d2 / 8.0
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
    inv_sigma = goldschmidt_inverse_batched(sigma_exp, goldschmidt_iterations_sigma)
    y = exp_approx * inv_sigma.unsqueeze(-1)

    for _ in range(n2):
        y_squared = y**2
        sum_y_sq = (y_squared * m).sum(dim=-1)
        inv_sum_sq = goldschmidt_inverse_batched(
            sum_y_sq, goldschmidt_iterations_sum_sq
        )
        y = y_squared * inv_sum_sq.unsqueeze(-1)

    return y * m


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
    goldschmidt_iterations_sigma: int,
    goldschmidt_iterations_sum_sq: int,
    chunk_size: int,
    max_rows: int | None,
    seed: int,
) -> dict[str, float]:
    scores, masks = _subsample_rows_gpu(scores, masks, max_rows, seed)
    n = scores.shape[0]
    if n == 0:
        return {
            "max_abs_err": float("nan"),
            "mean_abs_err": float("nan"),
            "num_vectors": 0,
        }

    max_abs = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    sum_abs = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    count = 0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        x = scores[start:end]
        m = masks[start:end]

        y_ref = standard_softmax_batched(x)
        y_thor = thor_softmax_batched(
            x,
            shift,
            delta1,
            delta2,
            m,
            goldschmidt_iterations_sigma,
            goldschmidt_iterations_sum_sq,
        )

        valid = m > 0
        err = torch.abs(y_ref - y_thor)[valid]
        if err.numel() == 0:
            continue
        max_abs = torch.maximum(max_abs, err.max())
        sum_abs = sum_abs + err.sum()
        count += int(err.numel())

    if count == 0:
        return {
            "max_abs_err": float("nan"),
            "mean_abs_err": float("nan"),
            "num_vectors": n,
        }

    return {
        "max_abs_err": float(max_abs.cpu()),
        "mean_abs_err": float((sum_abs / count).cpu()),
        "num_vectors": n,
        "num_elems": count,
    }


def analyze_thor_softmax_errors_gpu(
    task_name: str,
    collector: GpuScaledScoreCollector,
    layer_shifts: list[float],
    layer_divs: list[float],
    layer_gs_iters: list[int],
    layer_gsq_iters: list[int],
    eval_device: torch.device,
) -> dict[int, dict]:
    results: dict[int, dict] = {}

    for layer_idx in sorted(list(collector._scores.keys())):
        scores, masks = collector.stacked(layer_idx)
        shift = layer_shifts[layer_idx]
        delta2 = layer_divs[layer_idx]
        gs_iters = layer_gs_iters[layer_idx]
        gsq_iters = layer_gsq_iters[layer_idx]

        err = evaluate_thor_vs_softmax_layer_gpu(
            scores,
            masks,
            shift,
            DEFAULT_THOR_DELTA1,
            delta2,
            gs_iters,
            gsq_iters,
            ERROR_EVAL_CHUNK_SIZE,
            MAX_VECTORS_FOR_ERROR_EVAL,
            RANDOM_SEED + layer_idx,
        )
        depth = softmax_layer_depth(delta2, gs_iters, gsq_iters)
        results[layer_idx] = {
            **err,
            "shift": shift,
            "delta1": DEFAULT_THOR_DELTA1,
            "delta2": delta2,
            "goldschmidt_iterations_sigma": gs_iters,
            "goldschmidt_iterations_sum_sq": gsq_iters,
            "depth": depth,
        }
        collector.release_layer(layer_idx)

    print(f"\n--- {task_name}  thor_softmax vs standard_softmax [GPU 批量] ---")
    cap = f"最多 {MAX_VECTORS_FOR_ERROR_EVAL} 条" if MAX_VECTORS_FOR_ERROR_EVAL else "全部"
    print(f"    评估向量: {cap}, chunk={ERROR_EVAL_CHUNK_SIZE}")
    print(
        f"{'层':>3} {'shift':>8} {'d1':>4} {'d2':>4} {'gs':>3} {'gsq':>3} {'depth':>5} "
        f"{'向量数':>10} {'max_abs_err':>14} {'mean_abs_err':>14}"
    )
    for layer_idx in sorted(results.keys()):
        st = results[layer_idx]
        print(
            f"{layer_idx:3d} {st['shift']:8.4g} {st['delta1']:4g} {st['delta2']:4g} "
            f"{st['goldschmidt_iterations_sigma']:3d} {st['goldschmidt_iterations_sum_sq']:3d} "
            f"{st['depth']:5d} "
            f"{st['num_vectors']:10d} {st['max_abs_err']:14.6g} {st['mean_abs_err']:14.6g}"
        )

    return results


def run_task(task_name: str, device: torch.device) -> dict[int, dict]:
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    # print(f"\n{'=' * 60}")
    # sample_desc = "全部" if NUM_SAMPLES is None else f"随机 {NUM_SAMPLES}"
    # print(f"任务：{task_name.upper()}  |  {sample_desc}  |  设备：{device}")
    # print("评估：GPU thor_softmax + 每层 Goldschmidt 迭代（见配置表）")
    # print(f"{'=' * 60}")

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

    collector = GpuScaledScoreCollector(LAYER_ENABLE, device)
    hooks = register_scaled_score_hooks(model, collector)

    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    #print(f"推理 {len(tokenized)} 条...")
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
            # if step % 5 == 0 or step == len(loader):
            #     print(f"  batch {step}/{len(loader)}")

    restore_hooks(hooks)
    # print(f"各层向量条数：{collector.vector_counts()}")

    layer_shifts = layer_exp_shift_for_task(task_name)
    layer_divs = layer_exp_div_for_task(task_name)
    layer_gs_iters = layer_goldschmidt_sigma_iters_for_task(task_name)
    layer_gsq_iters = layer_goldschmidt_sum_sq_iters_for_task(task_name)

    return analyze_thor_softmax_errors_gpu(
        task_name,
        collector,
        layer_shifts,
        layer_divs,
        layer_gs_iters,
        layer_gsq_iters,
        device,
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA，将使用 CPU（仍走批量 torch，但无 GPU 加速）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("eval_softmax_cuda.py — GPU 批量 thor_softmax 误差评估")
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for task_name in TASK_NAMES:
        try:
            layer_exp_shift_for_task(task_name)
            layer_exp_div_for_task(task_name)
            layer_goldschmidt_sigma_iters_for_task(task_name)
            layer_goldschmidt_sum_sq_iters_for_task(task_name)
            run_task(task_name, device)
        except FileNotFoundError as e:
            print(f"跳过 {task_name}：{e}")

    print("完成。")


if __name__ == "__main__":
    main()
