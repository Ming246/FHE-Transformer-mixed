"""
校验集（从 train 分层抽样）校准各层非线性 [min,max]，
在验证集（评估集）上统计覆盖率。

安全：默认不运行任何任务；必须显式传入 --tasks（如 --tasks mrpc），
避免误跑覆盖已落盘、已被下游复用的校验集索引。

抽样（默认）：
  1) 按 label × 长度分层，先抽 pool = calib_size × OVERSAMPLE_RATIO
  2) 在 pool 上前向，标出各 slot 贡献全局 min/max 的样本（强制保留）
  3) 剩余名额再分层补足到 calib_size（极值样本过多时可略大于目标）

默认校验集规模：各任务见 TASK_CALIB_DEFAULTS。

支持 --calib-seeds 多随机种子稳定性报告。
"""
from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import json
import math
import os
import random

# DataLoader fork 前禁用 tokenizers 并行，避免刷屏警告（不影响抽样/前向逻辑）
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import Dataset, disable_progress_bar, load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    logging as hf_logging,
)

disable_progress_bar()
hf_logging.set_verbosity_error()

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
LOCAL_DATA_ROOT = "./glue_datasets/"
FINETUNED_MODEL_ROOT = "./finetuned_weight/"
OUTPUT_DIR = "./results/coverage_metrics/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
TASK_NUM_LABELS: dict[str, int] = {
    "mrpc": 2,
    "rte": 2,
    "sst2": 2,
    "cola": 2,
    "qnli": 2,
    "mnli": 3,
}

NONLINEAR_NAMES = ["Softmax", "LayerNorm1", "GeLU", "LayerNorm2"]
N_LEN_BINS = 5
DEFAULT_CALIB_SEED = 42
SCORE_VALID_THRESHOLD = -1e4

# 验证集覆盖率放宽系数（仅 validation 判定，不改变 calib 区间统计）
VAL_SOFTMAX_MIN_SCALE = 1.2   # 验证下界 = calib_min × scale
VAL_SOFTMAX_MAX_SCALE = 1.2   # 验证上界 = calib_max × scale
VAL_GELU_ABS_MAX_SCALE = 1.2  # 验证 bound = max(|calib_min|, |calib_max|) × scale，比较 |x|
VAL_LN_VAR_MIN_SCALE = 0.9    # 验证方差下界 = calib_min × scale
VAL_LN_VAR_MAX_SCALE = 1.2    # 验证方差上界 = calib_max × scale

TASK_CALIB_DEFAULTS: dict[str, dict[str, int]] = {
    "sst2": {"calib_size": 512},
    "mrpc": {"calib_size": 256},
    "rte": {"calib_size": 256},
    "cola": {"calib_size": 256},
    "qnli": {"calib_size": 512},
    "mnli": {"calib_size": 512},
}
# 先抽更大 pool，再保留极值样本并缩到 calib_size；1.0 = 关闭过采样
DEFAULT_CALIB_OVERSAMPLE_RATIO = 4.0
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = device.type == "cuda"
if USE_CUDA:
    torch.backends.cudnn.benchmark = True

BATCH_SIZE = 16 if USE_CUDA else 8


def _dataloader_kwargs() -> dict:
    kw: dict = {"shuffle": False}
    if USE_CUDA:
        kw["pin_memory"] = True
        kw["num_workers"] = 2
    return kw


def _batch_to_device(batch: dict) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=USE_CUDA)
        else:
            out[k] = v
    return out


def _transpose_for_scores(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    new_shape = x.size()[:-1] + (num_heads, head_dim)
    return x.view(new_shape).permute(0, 2, 1, 3)


def compute_masked_attention_scores(
    attn_module,
    hidden_states: torch.Tensor,
    *,
    encoder_hidden_states: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """QK^T / sqrt(d)，并加上 attention_mask（与 eval_softmax_cuda 一致）。"""
    mixed_query = attn_module.query(hidden_states)
    key_input = (
        encoder_hidden_states
        if encoder_hidden_states is not None
        else hidden_states
    )
    mixed_key = attn_module.key(key_input)
    head_dim = attn_module.attention_head_size
    num_heads = attn_module.num_attention_heads
    query_layer = _transpose_for_scores(mixed_query, num_heads, head_dim)
    key_layer = _transpose_for_scores(mixed_key, num_heads, head_dim)
    scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    scores = scores / (head_dim**0.5)
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]
        scores = scores + attention_mask
    return scores


def scores_valid_mask(scores: torch.Tensor) -> torch.Tensor:
    return scores > SCORE_VALID_THRESHOLD


def validation_softmax_bounds(lo: float, hi: float) -> tuple[float, float]:
    return lo * VAL_SOFTMAX_MIN_SCALE, hi * VAL_SOFTMAX_MAX_SCALE


def validation_gelu_abs_bound(lo: float, hi: float) -> float:
    return max(abs(lo), abs(hi)) * VAL_GELU_ABS_MAX_SCALE


def validation_ln_var_bounds(lo: float, hi: float) -> tuple[float, float]:
    return lo * VAL_LN_VAR_MIN_SCALE, hi * VAL_LN_VAR_MAX_SCALE


def update_softmax_element_range(
    ranges: dict[str, dict[str, float]],
    key: str,
    scores: torch.Tensor,
    query_mask_2d: torch.Tensor | None = None,
) -> None:
    """仅在有效 query × 有效 key 位置上更新 Softmax 分数区间。"""
    s = scores.detach()
    valid = scores_valid_mask(s)
    if query_mask_2d is not None:
        b, h, q, k = s.shape
        q_mask = query_mask_2d[:, :q].to(device=s.device).bool()
        query_row_valid = q_mask.unsqueeze(1).expand(-1, h, -1).reshape(-1)
        valid_rows = valid.reshape(-1, k)
        valid = (valid_rows & query_row_valid.unsqueeze(-1)).reshape_as(valid)
    if not bool(valid.any()):
        return
    vals = s[valid]
    ranges[key]["min"] = min(ranges[key]["min"], float(vals.min().item()))
    ranges[key]["max"] = max(ranges[key]["max"], float(vals.max().item()))


def _load_layernorm_utils():
    ln_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nolinear", "layernorm.py"
    )
    spec = importlib.util.spec_from_file_location("layernorm_eval", ln_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {ln_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LN = _load_layernorm_utils()


# ===================== 校验集选取 =====================


def task_calib_defaults(task_name: str) -> dict[str, int]:
    if task_name not in TASK_CALIB_DEFAULTS:
        raise KeyError(f"未知任务 {task_name}，请在 TASK_CALIB_DEFAULTS 中配置")
    return dict(TASK_CALIB_DEFAULTS[task_name])


def compute_valid_lengths(tokenized_ds: Dataset) -> list[int]:
    if "attention_mask" not in tokenized_ds.column_names:
        raise KeyError("tokenized 数据集缺少 attention_mask")
    return [int(sum(mask)) for mask in tokenized_ds["attention_mask"]]


def _length_bin(valid_len: int, edges: list[int]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= valid_len < edges[i + 1]:
            return i
    return len(edges) - 2


def stratified_sample_indices(
    n: int,
    labels: list[int],
    valid_lens: list[int],
    seed: int,
    n_len_bins: int = N_LEN_BINS,
) -> list[int]:
    """按 label × 有效 token 长度分桶分层抽样。"""
    if n <= 0:
        return []
    n_total = len(labels)
    if n >= n_total:
        return list(range(n_total))

    rng = random.Random(seed)
    sorted_lens = sorted(valid_lens)
    edges = [sorted_lens[0]]
    for i in range(1, n_len_bins):
        pos = min(n_total - 1, int(i * n_total / n_len_bins))
        edges.append(sorted_lens[pos])
    edges.append(sorted_lens[-1] + 1)

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (lab, vlen) in enumerate(zip(labels, valid_lens)):
        lb = _length_bin(vlen, edges)
        strata[(int(lab), lb)].append(idx)

    non_empty = {k: v for k, v in strata.items() if v}
    if not non_empty:
        return rng.sample(range(n_total), n)

    alloc: dict[tuple[int, int], int] = {}
    remaining = n
    for key in non_empty:
        alloc[key] = 1
        remaining -= 1
    if remaining < 0:
        keys = list(non_empty.keys())
        rng.shuffle(keys)
        for key in keys[: -remaining]:
            alloc[key] = 0
        remaining = 0

    if remaining > 0:
        weights = {k: len(v) for k, v in non_empty.items()}
        w_sum = sum(weights.values())
        extras = {k: int(remaining * weights[k] / w_sum) for k in non_empty}
        for k, extra in extras.items():
            alloc[k] += extra
        assigned = sum(extras.values())
        leftover = remaining - assigned
        if leftover > 0:
            keys = sorted(non_empty.keys(), key=lambda k: weights[k], reverse=True)
            for key in keys:
                if leftover <= 0:
                    break
                alloc[key] += 1
                leftover -= 1

    chosen: list[int] = []
    for key, count in alloc.items():
        if count <= 0:
            continue
        pool = non_empty[key]
        take = min(count, len(pool))
        chosen.extend(rng.sample(pool, take))

    if len(chosen) < n:
        rest = sorted(set(range(n_total)) - set(chosen))
        need = n - len(chosen)
        chosen.extend(rng.sample(rest, min(need, len(rest))))
    elif len(chosen) > n:
        chosen = rng.sample(chosen, n)
    return sorted(chosen)


def _get_labels(tokenized_train: Dataset) -> list[int]:
    col = "labels" if "labels" in tokenized_train.column_names else "label"
    if col not in tokenized_train.column_names:
        raise KeyError("tokenized 数据集缺少 label/labels 列")
    return [int(x) for x in tokenized_train[col]]


def build_calibration_indices(
    tokenized_train: Dataset,
    calib_size: int,
    seed: int,
) -> tuple[list[int], dict]:
    """仅分层抽样（无过采样 / 无极值保留）。"""
    labels = _get_labels(tokenized_train)
    valid_lens = compute_valid_lengths(tokenized_train)
    indices = stratified_sample_indices(calib_size, labels, valid_lens, seed=seed)
    meta = {
        "seed": seed,
        "calib_size": calib_size,
        "total_calib": len(indices),
        "oversample_ratio": 1.0,
        "pool_size": len(indices),
        "n_extreme_kept": 0,
    }
    return indices, meta


def resolve_pool_size(calib_size: int, n_train: int, oversample_ratio: float) -> int:
    if calib_size <= 0:
        return 0
    if oversample_ratio <= 1.0:
        return min(calib_size, n_train)
    return min(n_train, max(calib_size, int(math.ceil(calib_size * oversample_ratio))))


@dataclass
class SlotExtremeState:
    min_val: float = field(default_factory=lambda: float("inf"))
    max_val: float = field(default_factory=lambda: float("-inf"))
    min_sample: int | None = None
    max_sample: int | None = None


@dataclass
class ExtremeTracker:
    """在 pool（0..n_pool-1）上跟踪各 slot 的全局 min/max 及贡献样本。"""

    slots: dict[str, SlotExtremeState] = field(default_factory=dict)

    def _slot(self, key: str) -> SlotExtremeState:
        if key not in self.slots:
            self.slots[key] = SlotExtremeState()
        return self.slots[key]

    def consider(self, key: str, value: float, sample_idx: int) -> None:
        if not math.isfinite(value):
            return
        st = self._slot(key)
        if value < st.min_val:
            st.min_val = value
            st.min_sample = sample_idx
        if value > st.max_val:
            st.max_val = value
            st.max_sample = sample_idx

    def consider_sample_minmax(
        self, key: str, sample_min: float, sample_max: float, sample_idx: int
    ) -> None:
        self.consider(key, sample_min, sample_idx)
        self.consider(key, sample_max, sample_idx)

    def extreme_sample_indices(self) -> set[int]:
        out: set[int] = set()
        for st in self.slots.values():
            if st.min_sample is not None:
                out.add(int(st.min_sample))
            if st.max_sample is not None:
                out.add(int(st.max_sample))
        return out

    def to_ranges(self) -> dict[str, dict[str, float]]:
        ranges = create_empty_ranges()
        for key, st in self.slots.items():
            if key not in ranges:
                continue
            if math.isfinite(st.min_val):
                ranges[key]["min"] = st.min_val
            if math.isfinite(st.max_val):
                ranges[key]["max"] = st.max_val
        return ranges


def _update_softmax_extremes(
    tracker: ExtremeTracker,
    ranges: dict[str, dict[str, float]],
    key: str,
    scores: torch.Tensor,
    query_mask_2d: torch.Tensor | None,
    batch_offset: int,
) -> None:
    s = scores.detach()
    valid = scores_valid_mask(s)
    if query_mask_2d is not None:
        b, h, q, k = s.shape
        q_mask = query_mask_2d[:, :q].to(device=s.device).bool()
        query_row_valid = q_mask.unsqueeze(1).expand(-1, h, -1)
        valid = valid & query_row_valid.unsqueeze(-1)
    if bool(valid.any()):
        vals = s[valid]
        ranges[key]["min"] = min(ranges[key]["min"], float(vals.min().item()))
        ranges[key]["max"] = max(ranges[key]["max"], float(vals.max().item()))
    bsz = int(s.shape[0])
    for bi in range(bsz):
        vb = valid[bi]
        if not bool(vb.any()):
            continue
        sample_vals = s[bi][vb]
        tracker.consider_sample_minmax(
            key,
            float(sample_vals.min().item()),
            float(sample_vals.max().item()),
            batch_offset + bi,
        )


def _update_gelu_extremes(
    tracker: ExtremeTracker,
    ranges: dict[str, dict[str, float]],
    key: str,
    tensor: torch.Tensor,
    batch_offset: int,
) -> None:
    x = tensor.detach()
    ranges[key]["min"] = min(ranges[key]["min"], float(x.min().item()))
    ranges[key]["max"] = max(ranges[key]["max"], float(x.max().item()))
    bsz = int(x.shape[0])
    flat = x.reshape(bsz, -1)
    for bi in range(bsz):
        row = flat[bi]
        tracker.consider_sample_minmax(
            key,
            float(row.min().item()),
            float(row.max().item()),
            batch_offset + bi,
        )


def _update_ln_extremes(
    tracker: ExtremeTracker,
    ranges: dict[str, dict[str, float]],
    key: str,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    batch_offset: int,
) -> None:
    x = hidden_states.detach()
    bsz = int(x.shape[0])
    for bi in range(bsz):
        rows = x[bi]
        if attention_mask is not None:
            m = attention_mask[bi].to(device=rows.device).bool()
            rows = rows[m]
        if rows.numel() == 0:
            continue
        vars_t = _LN.population_var_batched(rows)
        vmin = float(vars_t.min().item())
        vmax = float(vars_t.max().item())
        ranges[key]["min"] = min(ranges[key]["min"], vmin)
        ranges[key]["max"] = max(ranges[key]["max"], vmax)
        tracker.consider_sample_minmax(key, vmin, vmax, batch_offset + bi)


def register_extreme_calib_hooks(
    model,
    ranges: dict[str, dict[str, float]],
    tracker: ExtremeTracker,
) -> HookState:
    """前向时同步更新区间，并记录贡献 min/max 的 pool 内样本下标。"""
    state = HookState()

    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]

        softmax_key = slot_key(layer_idx, "Softmax")
        attn_self = layer.attention.self
        orig_forward = attn_self.forward

        def make_softmax_patched(orig_fn, key, attn_module, bert_model):
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
                scores = compute_masked_attention_scores(
                    attn_module,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                offset = int(getattr(bert_model, "_extreme_batch_offset", 0))
                query_mask = getattr(attn_module, "_coverage_query_mask", None)
                _update_softmax_extremes(
                    tracker, ranges, key, scores, query_mask, offset
                )
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

        attn_self.forward = make_softmax_patched(
            orig_forward, softmax_key, attn_self, model
        )
        state.forward_restores.append((attn_self, orig_forward))

        gelu_key = slot_key(layer_idx, "GeLU")
        act_fn = layer.intermediate.intermediate_act_fn
        if isinstance(act_fn, torch.nn.Module):

            def make_gelu_hook(key, bert_model):
                def hook_fn(module, inp, out):
                    offset = int(getattr(bert_model, "_extreme_batch_offset", 0))
                    _update_gelu_extremes(tracker, ranges, key, inp[0], offset)

                return hook_fn

            state.module_hooks.append(
                act_fn.register_forward_hook(make_gelu_hook(gelu_key, model))
            )
        else:
            intermediate = layer.intermediate
            orig_inter = intermediate.forward

            def make_gelu_inter(orig_fn, key, bert_model):
                @functools.wraps(orig_fn)
                def patched(*args, **kwargs):
                    original_gelu = F.gelu

                    def tracking_gelu(input, approximate="none"):
                        offset = int(getattr(bert_model, "_extreme_batch_offset", 0))
                        _update_gelu_extremes(tracker, ranges, key, input, offset)
                        return original_gelu(input, approximate=approximate)

                    F.gelu = tracking_gelu
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        F.gelu = original_gelu

                return patched

            intermediate.forward = make_gelu_inter(orig_inter, gelu_key, model)
            state.forward_restores.append((intermediate, orig_inter))

        for kind, ln_module, ln_name in (
            ("ln1", layer.attention.output.LayerNorm, "LayerNorm1"),
            ("ln2", layer.output.LayerNorm, "LayerNorm2"),
        ):
            key = slot_key(layer_idx, ln_name)
            orig_ln = ln_module.forward

            def make_ln_patched(orig_fn, key_v, ln_mod, bert_model):
                @functools.wraps(orig_fn)
                def patched_forward(hidden_states, *args, **kwargs):
                    offset = int(getattr(bert_model, "_extreme_batch_offset", 0))
                    attn_mask = getattr(ln_mod, "_eval_ln_attn_mask", None)
                    _update_ln_extremes(
                        tracker, ranges, key_v, hidden_states, attn_mask, offset
                    )
                    return orig_fn(hidden_states, *args, **kwargs)

                return patched_forward

            ln_module.forward = make_ln_patched(orig_ln, key, ln_module, model)
            state.forward_restores.append((ln_module, orig_ln))

    return state


@torch.no_grad()
def collect_pool_extremes(
    model, tokenized_pool: Dataset, collator
) -> tuple[set[int], dict[str, dict[str, float]]]:
    """在过采样 pool 上前向：返回极值样本（pool 内下标）与区间。"""
    ranges = create_empty_ranges()
    tracker = ExtremeTracker()
    hook_state = register_extreme_calib_hooks(model, ranges, tracker)
    loader = DataLoader(
        tokenized_pool,
        batch_size=BATCH_SIZE,
        collate_fn=collator,
        **_dataloader_kwargs(),
    )
    try:
        offset = 0
        for batch in loader:
            batch = _batch_to_device(batch)
            attn_mask = batch.get("attention_mask")
            model._extreme_batch_offset = offset
            if attn_mask is not None:
                _LN.attach_attention_mask_to_layernorms(model, attn_mask)
                attach_coverage_query_mask(model, attn_mask)
            model(**batch)
            _LN.clear_attention_mask_on_layernorms(model)
            clear_coverage_query_mask(model)
            offset += int(batch["labels"].shape[0])
    finally:
        restore_hooks(hook_state)
        if hasattr(model, "_extreme_batch_offset"):
            delattr(model, "_extreme_batch_offset")
    return tracker.extreme_sample_indices(), ranges


def select_pool_indices_keep_extremes(
    pool_train_indices: list[int],
    extreme_local: set[int],
    *,
    calib_size: int,
    labels: list[int],
    valid_lens: list[int],
    seed: int,
) -> tuple[list[int], dict]:
    """
    从 pool 缩到目标大小：先保留极值样本，再用分层抽样补足。
    返回 train 下标列表与统计 meta。
    """
    n_pool = len(pool_train_indices)
    must_local = sorted(i for i in extreme_local if 0 <= i < n_pool)
    n_extreme = len(must_local)

    if n_extreme >= calib_size:
        # 极值已覆盖目标：全部保留（可略大于 calib_size）
        chosen_local = must_local
    else:
        need = calib_size - n_extreme
        remaining_local = [i for i in range(n_pool) if i not in extreme_local]
        if need >= len(remaining_local):
            chosen_local = sorted(set(must_local) | set(remaining_local))
        else:
            rem_labels = [labels[pool_train_indices[i]] for i in remaining_local]
            rem_lens = [valid_lens[pool_train_indices[i]] for i in remaining_local]
            # stratified_sample_indices 返回的是 remaining_local 内的下标
            pick = stratified_sample_indices(
                need, rem_labels, rem_lens, seed=seed + 17
            )
            fill_local = [remaining_local[j] for j in pick]
            chosen_local = sorted(set(must_local) | set(fill_local))

    train_indices = [pool_train_indices[i] for i in chosen_local]
    meta = {
        "n_extreme_kept": n_extreme,
        "n_fill": max(0, len(train_indices) - n_extreme),
        "final_size": len(train_indices),
        "exceeded_target": len(train_indices) > calib_size,
    }
    return train_indices, meta


def build_calibration_indices_oversampled(
    model,
    tokenized_train: Dataset,
    collator,
    *,
    calib_size: int,
    seed: int,
    oversample_ratio: float = DEFAULT_CALIB_OVERSAMPLE_RATIO,
) -> tuple[list[int], dict, dict[str, dict[str, float]] | None]:
    """
    过采样 pool → 标极值 → 缩到 calib_size。
    返回 (train_indices, meta, pool_ranges_or_None)。
    ratio<=1 时退化为普通分层抽样（不跑极值前向）。
    """
    labels = _get_labels(tokenized_train)
    valid_lens = compute_valid_lengths(tokenized_train)
    n_train = len(tokenized_train)
    pool_size = resolve_pool_size(calib_size, n_train, oversample_ratio)

    pool_indices = stratified_sample_indices(
        pool_size, labels, valid_lens, seed=seed
    )
    meta: dict = {
        "seed": seed,
        "calib_size": calib_size,
        "oversample_ratio": float(oversample_ratio),
        "pool_size": len(pool_indices),
    }

    if oversample_ratio <= 1.0 or pool_size <= calib_size:
        meta.update(
            {
                "total_calib": len(pool_indices),
                "n_extreme_kept": 0,
                "n_fill": len(pool_indices),
                "final_size": len(pool_indices),
                "exceeded_target": False,
            }
        )
        return pool_indices, meta, None

    print(
        f"  过采样 pool={len(pool_indices)} "
        f"(target={calib_size}, ratio={oversample_ratio:g})，扫描极值样本…"
    )
    pool_ds = tokenized_train.select(pool_indices)
    extreme_local, pool_ranges = collect_pool_extremes(model, pool_ds, collator)
    final_indices, sel_meta = select_pool_indices_keep_extremes(
        pool_indices,
        extreme_local,
        calib_size=calib_size,
        labels=labels,
        valid_lens=valid_lens,
        seed=seed,
    )
    meta.update(sel_meta)
    meta["total_calib"] = len(final_indices)
    print(
        f"  极值保留 {sel_meta['n_extreme_kept']} + 分层补足 "
        f"{sel_meta['n_fill']} → 最终 {len(final_indices)}"
        + ("（略大于目标）" if sel_meta["exceeded_target"] else "")
    )
    return final_indices, meta, pool_ranges


def save_calib_indices(
    task_name: str, indices: list[int], meta: dict, output_dir: str, seed: int
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{task_name}_calib_indices_seed{seed}.json")
    payload = {"task": task_name, "indices": indices, **meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


# ===================== 区间 / 覆盖率统计 =====================


def slot_key(layer_idx: int, name: str) -> str:
    return f"Layer{layer_idx}_{name}"


def create_empty_ranges() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            out[slot_key(layer_idx, name)] = {
                "min": float("inf"),
                "max": float("-inf"),
            }
    return out


def update_element_range(ranges: dict[str, dict[str, float]], key: str, tensor: torch.Tensor) -> None:
    t = tensor.detach()
    ranges[key]["min"] = min(ranges[key]["min"], float(t.min().item()))
    ranges[key]["max"] = max(ranges[key]["max"], float(t.max().item()))


@dataclass
class SampleCoverageState:
    """验证集逐样本覆盖率：每个 slot 维护 length=n_samples 的 bool 数组。"""

    n_samples: int
    slot_pass: dict[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.slot_pass = {
            slot_key(i, n): np.ones(self.n_samples, dtype=bool)
            for i in range(NUM_LAYERS)
            for n in NONLINEAR_NAMES
        }

    def apply_batch(self, key: str, batch_offset: int, ok: torch.Tensor) -> None:
        n = int(ok.numel())
        if n <= 0:
            return
        self.slot_pass[key][batch_offset : batch_offset + n] &= (
            ok.detach().cpu().numpy().astype(bool)
        )

    def slot_pct(self, key: str) -> float:
        return pct(self.slot_numerator(key), self.n_samples)

    def slot_numerator(self, key: str) -> int:
        return int(self.slot_pass[key].sum())

    def all_layers_all_slots_pct(self) -> float:
        return pct(self.all_layers_numerator(), self.n_samples)

    def all_layers_numerator(self) -> int:
        mat = np.stack(
            [
                self.slot_pass[slot_key(i, n)]
                for i in range(NUM_LAYERS)
                for n in NONLINEAR_NAMES
            ],
            axis=1,
        )
        return int(mat.all(axis=1).sum())


@dataclass
class HookState:
    module_hooks: list = field(default_factory=list)
    forward_restores: list[tuple[object, object]] = field(default_factory=list)


def _gelu_sample_ok_batch(
    tensor: torch.Tensor,
    lo: float,
    hi: float,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """逐样本：有效 token 的 |GeLU 输入| <= max(|min|,|max|)×VAL_GELU_ABS_MAX_SCALE。"""
    abs_bound = validation_gelu_abs_bound(lo, hi)
    x = tensor.detach()
    in_range = x.abs() <= abs_bound
    if attention_mask is not None:
        m = attention_mask.to(device=x.device).bool().unsqueeze(-1)
        check = torch.where(m, in_range, torch.ones_like(in_range, dtype=torch.bool))
    else:
        check = in_range
    return check.all(dim=(1, 2))


def _softmax_sample_ok_batch(
    scores: torch.Tensor,
    lo: float,
    hi: float,
    query_mask_2d: torch.Tensor | None,
) -> torch.Tensor:
    """逐样本：有效 query × 有效 key 的 score 落在放宽后的 [lo, hi] 内。"""
    val_lo, val_hi = validation_softmax_bounds(lo, hi)
    valid = scores_valid_mask(scores.detach())
    in_range = (scores >= val_lo) & (scores <= val_hi)
    if query_mask_2d is not None:
        qv = query_mask_2d.to(device=scores.device).bool().unsqueeze(1).unsqueeze(-1)
    else:
        qv = torch.ones(
            scores.shape[0],
            1,
            scores.shape[2],
            1,
            dtype=torch.bool,
            device=scores.device,
        )
    need = valid & qv
    ok_all = in_range | ~need
    return ok_all.all(dim=(1, 2, 3))


def _ln_sample_ok_batch(
    tensor: torch.Tensor,
    attention_mask: torch.Tensor | None,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """逐样本：有效 token 的 LayerNorm 输入方差落在放宽后的 [lo, hi] 内。"""
    val_lo, val_hi = validation_ln_var_bounds(lo, hi)
    x = tensor.detach()
    var = x.var(dim=-1, unbiased=False)
    in_range = (var >= val_lo) & (var <= val_hi)
    if attention_mask is not None:
        m = attention_mask.to(device=x.device).bool()
        check = torch.where(m, in_range, torch.ones_like(in_range, dtype=torch.bool))
    else:
        check = in_range
    return check.all(dim=1)


def _coverage_batch_context(model) -> tuple[SampleCoverageState | None, int, torch.Tensor | None]:
    state = getattr(model, "_coverage_state", None)
    offset = int(getattr(model, "_coverage_batch_offset", 0))
    attn_mask = getattr(model, "_coverage_attn_mask_2d", None)
    return state, offset, attn_mask


def _apply_sample_coverage(
    model, key: str, ok: torch.Tensor, state: SampleCoverageState | None, offset: int
) -> None:
    if state is not None:
        state.apply_batch(key, offset, ok)


def register_calib_hooks(model, ranges: dict[str, dict[str, float]]) -> HookState:
    state = HookState()

    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]

        softmax_key = slot_key(layer_idx, "Softmax")
        attn_self = layer.attention.self
        orig_forward = attn_self.forward

        def make_softmax_patched(orig_fn, key, attn_module):
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
                scores = compute_masked_attention_scores(
                    attn_module,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                query_mask = getattr(attn_module, "_coverage_query_mask", None)
                update_softmax_element_range(ranges, key, scores, query_mask_2d=query_mask)
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

        attn_self.forward = make_softmax_patched(orig_forward, softmax_key, attn_self)
        state.forward_restores.append((attn_self, orig_forward))

        gelu_key = slot_key(layer_idx, "GeLU")
        act_fn = layer.intermediate.intermediate_act_fn
        if isinstance(act_fn, torch.nn.Module):

            def make_gelu_hook(key):
                def hook_fn(module, inp, out):
                    update_element_range(ranges, key, inp[0])

                return hook_fn

            state.module_hooks.append(
                act_fn.register_forward_hook(make_gelu_hook(gelu_key))
            )
        else:
            intermediate = layer.intermediate
            orig_inter = intermediate.forward

            def make_gelu_inter(orig_fn, key):
                @functools.wraps(orig_fn)
                def patched(*args, **kwargs):
                    original_gelu = F.gelu

                    def tracking_gelu(input, approximate="none"):
                        update_element_range(ranges, key, input)
                        return original_gelu(input, approximate=approximate)

                    F.gelu = tracking_gelu
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        F.gelu = original_gelu

                return patched

            intermediate.forward = make_gelu_inter(orig_inter, gelu_key)
            state.forward_restores.append((intermediate, orig_inter))

    return state


def register_validation_hooks(
    model,
    ranges: dict[str, dict[str, float]],
) -> HookState:
    state = HookState()

    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]

        softmax_key = slot_key(layer_idx, "Softmax")
        lo, hi = ranges[softmax_key]["min"], ranges[softmax_key]["max"]
        attn_self = layer.attention.self
        orig_forward = attn_self.forward

        def make_softmax_patched(orig_fn, key, lo_v, hi_v, attn_module, bert_model):
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
                scores = compute_masked_attention_scores(
                    attn_module,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                cov_state, offset, _ = _coverage_batch_context(bert_model)
                query_mask = getattr(attn_module, "_coverage_query_mask", None)
                ok = _softmax_sample_ok_batch(scores, lo_v, hi_v, query_mask)
                _apply_sample_coverage(bert_model, key, ok, cov_state, offset)
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

        attn_self.forward = make_softmax_patched(
            orig_forward, softmax_key, lo, hi, attn_self, model
        )
        state.forward_restores.append((attn_self, orig_forward))

        gelu_key = slot_key(layer_idx, "GeLU")
        lo, hi = ranges[gelu_key]["min"], ranges[gelu_key]["max"]
        act_fn = layer.intermediate.intermediate_act_fn
        if isinstance(act_fn, torch.nn.Module):

            def make_gelu_hook(key, lo_v, hi_v, bert_model):
                def hook_fn(module, inp, out):
                    cov_state, offset, attn_mask = _coverage_batch_context(bert_model)
                    ok = _gelu_sample_ok_batch(inp[0], lo_v, hi_v, attn_mask)
                    _apply_sample_coverage(bert_model, key, ok, cov_state, offset)

                return hook_fn

            state.module_hooks.append(
                act_fn.register_forward_hook(make_gelu_hook(gelu_key, lo, hi, model))
            )
        else:
            intermediate = layer.intermediate
            orig_inter = intermediate.forward

            def make_gelu_inter(orig_fn, key, lo_v, hi_v):
                @functools.wraps(orig_fn)
                def patched(*args, **kwargs):
                    original_gelu = F.gelu
                    bert_model = model

                    def tracking_gelu(input, approximate="none"):
                        cov_state, offset, attn_mask = _coverage_batch_context(bert_model)
                        ok = _gelu_sample_ok_batch(input, lo_v, hi_v, attn_mask)
                        _apply_sample_coverage(bert_model, key, ok, cov_state, offset)
                        return original_gelu(input, approximate=approximate)

                    F.gelu = tracking_gelu
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        F.gelu = original_gelu

                return patched

            intermediate.forward = make_gelu_inter(orig_inter, gelu_key, lo, hi)
            state.forward_restores.append((intermediate, orig_inter))

        for _kind, ln_module, ln_name in (
            ("ln1", layer.attention.output.LayerNorm, "LayerNorm1"),
            ("ln2", layer.output.LayerNorm, "LayerNorm2"),
        ):
            key = slot_key(layer_idx, ln_name)
            lo = ranges[key]["min"]
            hi = ranges[key]["max"]
            orig_ln = ln_module.forward

            def make_ln_patched(orig_fn, key_v, lo_v, hi_v, ln_mod, bert_model):
                @functools.wraps(orig_fn)
                def patched_forward(hidden_states, *args, **kwargs):
                    cov_state, offset, attn_mask = _coverage_batch_context(bert_model)
                    if attn_mask is None:
                        attn_mask = getattr(ln_mod, "_eval_ln_attn_mask", None)
                    ok = _ln_sample_ok_batch(hidden_states, attn_mask, lo_v, hi_v)
                    _apply_sample_coverage(bert_model, key_v, ok, cov_state, offset)
                    return orig_fn(hidden_states, *args, **kwargs)

                return patched_forward

            ln_module.forward = make_ln_patched(
                orig_ln, key, lo, hi, ln_module, model
            )
            state.forward_restores.append((ln_module, orig_ln))

    return state


def attach_coverage_query_mask(model, attention_mask: torch.Tensor) -> None:
    """为 Softmax 覆盖率统计挂载 batch 级 [B, L] query mask。"""
    for layer in model.bert.encoder.layer:
        layer.attention.self._coverage_query_mask = attention_mask


def clear_coverage_query_mask(model) -> None:
    for layer in model.bert.encoder.layer:
        if hasattr(layer.attention.self, "_coverage_query_mask"):
            delattr(layer.attention.self, "_coverage_query_mask")


def restore_hooks(state: HookState) -> None:
    for h in state.module_hooks:
        h.remove()
    for module, orig_forward in state.forward_restores:
        module.forward = orig_forward


@torch.no_grad()
def run_forward(
    model,
    tokenized_split,
    data_collator,
    coverage_state: SampleCoverageState | None = None,
) -> None:
    loader = DataLoader(
        tokenized_split,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        **_dataloader_kwargs(),
    )
    sample_offset = 0
    for batch in loader:
        batch = _batch_to_device(batch)
        attn_mask = batch.get("attention_mask")
        bs = int(batch["labels"].shape[0]) if coverage_state is not None else 0
        if coverage_state is not None:
            model._coverage_state = coverage_state
            model._coverage_batch_offset = sample_offset
            model._coverage_attn_mask_2d = attn_mask
        if attn_mask is not None:
            _LN.attach_attention_mask_to_layernorms(model, attn_mask)
            attach_coverage_query_mask(model, attn_mask)
        model(**batch)
        _LN.clear_attention_mask_on_layernorms(model)
        clear_coverage_query_mask(model)
        if coverage_state is not None:
            sample_offset += bs
    if coverage_state is not None:
        for attr in ("_coverage_state", "_coverage_batch_offset", "_coverage_attn_mask_2d"):
            if hasattr(model, attr):
                delattr(model, attr)


def finalize_layernorm_calib_ranges(
    ln_collector, ranges: dict[str, dict[str, float]]
) -> None:
    kind_to_name = {"ln1": "LayerNorm1", "ln2": "LayerNorm2"}
    for layer_idx in range(NUM_LAYERS):
        for kind, name in kind_to_name.items():
            key = slot_key(layer_idx, name)
            inputs = ln_collector.stacked(layer_idx, kind)
            if inputs.shape[0] == 0:
                ranges[key]["min"] = float("nan")
                ranges[key]["max"] = float("nan")
                continue
            st = _LN.variance_stats_gpu(inputs)
            ranges[key]["min"] = st["var_min"]
            ranges[key]["max"] = st["var_max"]
            ln_collector.release(layer_idx, kind)


def task_num_labels(task_name: str) -> int:
    if task_name not in TASK_NUM_LABELS:
        raise KeyError(f"未知任务 {task_name}，请在 TASK_NUM_LABELS 中配置")
    return TASK_NUM_LABELS[task_name]


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


def tokenize_split(split, task_name: str, tokenizer):
    drop_cols = [c for c in split.column_names if c != "label"]
    return split.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=drop_cols,
    )


def load_model_and_data(task_name: str):
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    if "train" not in dataset:
        raise KeyError(f"{task_name} 数据集缺少 train split")
    if "validation" not in dataset:
        raise KeyError(f"{task_name} 数据集缺少 validation split")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=task_num_labels(task_name),
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    tokenized_train = tokenize_split(dataset["train"], task_name, tokenizer)
    tokenized_val = tokenize_split(dataset["validation"], task_name, tokenizer)
    return model, tokenized_train, tokenized_val, collator


def collect_calib_ranges(
    model, tokenized_calib: Dataset, collator
) -> dict[str, dict[str, float]]:
    ranges = create_empty_ranges()
    ln_collector = _LN.GpuLayerNormCollector([True] * NUM_LAYERS, device)
    ln_hooks = _LN.register_layernorm_hooks(model, ln_collector)
    hook_state = register_calib_hooks(model, ranges)
    try:
        run_forward(model, tokenized_calib, collator)
        finalize_layernorm_calib_ranges(ln_collector, ranges)
    finally:
        restore_hooks(hook_state)
        _LN.restore_layernorm_hooks(ln_hooks)
    return ranges


def collect_validation_coverage(
    model,
    tokenized_val,
    collator,
    ranges: dict[str, dict[str, float]],
) -> SampleCoverageState:
    coverage_state = SampleCoverageState(len(tokenized_val))
    hook_state = register_validation_hooks(model, ranges)
    try:
        run_forward(model, tokenized_val, collator, coverage_state=coverage_state)
    finally:
        restore_hooks(hook_state)
    return coverage_state


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den > 0 else float("nan")


# (字段名, 列宽, 对齐, 浮点小数位；None 表示按整数/字符串输出)
ColumnSpec = tuple[str, int, str, int | None]

COVERAGE_TABLE_COLUMNS: list[ColumnSpec] = [
    ("task", 6, "left", None),
    ("layer", 5, "right", None),
    ("nonlinear", 12, "left", None),
    ("calib_count", 11, "right", None),
    ("seed", 5, "right", None),
    ("calib_min", 10, "right", 2),
    ("calib_max", 10, "right", 2),
    ("metric", 32, "left", None),
    ("value_pct", 10, "right", 2),
    ("numerator", 10, "right", None),
    ("denominator", 11, "right", None),
]

CALIB_RANGE_TABLE_COLUMNS: list[ColumnSpec] = [
    ("layer", 5, "right", None),
    ("nonlinear", 12, "left", None),
    ("calib_min", 10, "right", 2),
    ("calib_max", 10, "right", 2),
    ("val_min", 10, "right", 2),
    ("val_max", 10, "right", 2),
]


def _format_table_cell(
    value: object, width: int, align: str, float_decimals: int | None
) -> str:
    if float_decimals is not None and isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            text = "nan"
        else:
            text = f"{float(value):.{float_decimals}f}"
    elif isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value)
    if len(text) > width:
        text = text[:width]
    return text.rjust(width) if align == "right" else text.ljust(width)


def _format_table_row(row: dict, columns: list[ColumnSpec]) -> str:
    return " ".join(
        _format_table_cell(row[key], width, align, dec)
        for key, width, align, dec in columns
    )


def write_aligned_table(
    path: str, columns: list[ColumnSpec], rows: list[dict]
) -> None:
    header_row = {spec[0]: spec[0] for spec in columns}
    lines = [_format_table_row(header_row, columns)]
    lines.extend(_format_table_row(row, columns) for row in rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def aggregate_sample_fully_coverage(coverage_state: SampleCoverageState) -> float:
    vals = [
        coverage_state.slot_pct(slot_key(layer_idx, name))
        for layer_idx in range(NUM_LAYERS)
        for name in NONLINEAR_NAMES
    ]
    finite = [v for v in vals if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def build_result_rows(
    task_name: str,
    ranges: dict[str, dict[str, float]],
    coverage_state: SampleCoverageState,
    *,
    calib_count: int,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            key = slot_key(layer_idx, name)
            ref = ranges[key]
            rows.append(
                {
                    "task": task_name,
                    "layer": layer_idx,
                    "nonlinear": name,
                    "calib_count": calib_count,
                    "seed": seed,
                    "calib_min": ref["min"],
                    "calib_max": ref["max"],
                    "metric": "sample_in_range_pct",
                    "value_pct": coverage_state.slot_pct(key),
                    "numerator": coverage_state.slot_numerator(key),
                    "denominator": coverage_state.n_samples,
                }
            )
    rows.append(
        {
            "task": task_name,
            "layer": "all",
            "nonlinear": "ALL",
            "calib_count": calib_count,
            "seed": seed,
            "calib_min": float("nan"),
            "calib_max": float("nan"),
            "metric": "sample_all_layers_in_range_pct",
            "value_pct": coverage_state.all_layers_all_slots_pct(),
            "numerator": coverage_state.all_layers_numerator(),
            "denominator": coverage_state.n_samples,
        }
    )
    return rows


def save_task_outputs(
    task_name: str,
    ranges: dict[str, dict[str, float]],
    rows: list[dict],
    output_dir: str,
    calib_meta: dict,
    val_ranges: dict[str, dict[str, float]] | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    range_path = os.path.join(output_dir, f"{task_name}_calib_ranges.csv")
    range_rows = []
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            key = slot_key(layer_idx, name)
            row = {
                "layer": layer_idx,
                "nonlinear": name,
                "calib_min": ranges[key]["min"],
                "calib_max": ranges[key]["max"],
                "val_min": float("nan"),
                "val_max": float("nan"),
            }
            if val_ranges is not None and key in val_ranges:
                row["val_min"] = val_ranges[key]["min"]
                row["val_max"] = val_ranges[key]["max"]
            range_rows.append(row)
    write_aligned_table(range_path, CALIB_RANGE_TABLE_COLUMNS, range_rows)

    cov_path = os.path.join(output_dir, f"{task_name}_validation_coverage.csv")
    write_aligned_table(cov_path, COVERAGE_TABLE_COLUMNS, rows)

    meta_path = os.path.join(output_dir, f"{task_name}_calib_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(calib_meta, f, indent=2)

    print(f"  已保存校验集区间：{range_path}")
    print(f"  已保存验证集覆盖率：{cov_path}")
    print(f"  校验集元信息：{meta_path}")


def run_stability_study(
    task_name: str,
    model,
    tokenized_train: Dataset,
    tokenized_val: Dataset,
    collator,
    output_dir: str,
    calib_size: int,
    n_seeds: int,
    base_seed: int,
    *,
    oversample_ratio: float = DEFAULT_CALIB_OVERSAMPLE_RATIO,
) -> None:
    rows: list[dict] = []
    for i in range(n_seeds):
        seed = base_seed + i
        indices, meta, _ = build_calibration_indices_oversampled(
            model,
            tokenized_train,
            collator,
            calib_size=calib_size,
            seed=seed,
            oversample_ratio=oversample_ratio,
        )
        calib_ds = tokenized_train.select(indices)
        ranges = collect_calib_ranges(model, calib_ds, collator)
        coverage_state = collect_validation_coverage(model, tokenized_val, collator, ranges)
        mean_cov = coverage_state.all_layers_all_slots_pct()
        rows.append(
            {
                "task": task_name,
                "seed": seed,
                "calib_total_n": len(indices),
                "n_extreme_kept": meta.get("n_extreme_kept", 0),
                "mean_sample_fully_in_range_pct": mean_cov,
                **{f"width_{k}": ranges[k]["max"] - ranges[k]["min"] for k in sorted(ranges)[:4]},
            }
        )
        print(f"    seed={seed} total={len(indices)} mean_sample_cov={mean_cov:.2f}%")

    path = os.path.join(output_dir, f"{task_name}_calib_stability.csv")
    os.makedirs(output_dir, exist_ok=True)
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  稳定性报告：{path}")


@dataclass
class CalibConfig:
    calib_size: int
    seed: int = DEFAULT_CALIB_SEED
    oversample_ratio: float = DEFAULT_CALIB_OVERSAMPLE_RATIO


def resolve_calib_config(task_name: str, args) -> CalibConfig:
    defaults = task_calib_defaults(task_name)
    return CalibConfig(
        calib_size=args.calib_size if args.calib_size is not None else defaults["calib_size"],
        seed=args.calib_seed,
        oversample_ratio=float(args.oversample_ratio),
    )


def process_task(task_name: str, output_dir: str, args) -> list[dict]:
    print(f"\n>>> {task_name.upper()}")
    model, tokenized_train, tokenized_val, collator = load_model_and_data(task_name)
    cfg = resolve_calib_config(task_name, args)

    if args.calib_seeds > 1:
        print(f"  多 seed 稳定性：{args.calib_seeds} 次（base_seed={cfg.seed}）")
        run_stability_study(
            task_name,
            model,
            tokenized_train,
            tokenized_val,
            collator,
            output_dir,
            cfg.calib_size,
            args.calib_seeds,
            cfg.seed,
            oversample_ratio=cfg.oversample_ratio,
        )

    print(
        f"  校验集：目标 n={cfg.calib_size}（seed={cfg.seed}, "
        f"oversample×{cfg.oversample_ratio:g}）"
    )
    indices, meta, _pool_ranges = build_calibration_indices_oversampled(
        model,
        tokenized_train,
        collator,
        calib_size=cfg.calib_size,
        seed=cfg.seed,
        oversample_ratio=cfg.oversample_ratio,
    )
    idx_path = save_calib_indices(task_name, indices, meta, output_dir, cfg.seed)
    print(f"  校验集索引：{idx_path}（共 {len(indices)} 条）")

    calib_ds = tokenized_train.select(indices)
    ranges = collect_calib_ranges(model, calib_ds, collator)

    print(f"  评估集（validation）样本数：{len(tokenized_val)}")
    print("  统计验证集原始区间（与校验集同定义，便于对比）…")
    val_ranges = collect_calib_ranges(model, tokenized_val, collator)
    coverage_state = collect_validation_coverage(model, tokenized_val, collator, ranges)
    rows = build_result_rows(
        task_name,
        ranges,
        coverage_state,
        calib_count=len(indices),
        seed=cfg.seed,
    )
    calib_meta = {
        **meta,
        "task": task_name,
        "indices_path": idx_path,
        "eval_split": "validation",
    }
    save_task_outputs(
        task_name, ranges, rows, output_dir, calib_meta, val_ranges=val_ranges
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "校验集（train 分层抽样）定区间 + validation 覆盖率。"
            "必须显式指定 --tasks，否则不运行任何数据集。"
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=(
            "要处理的任务名（必须显式指定；省略则不运行）。"
            f"可选：{' '.join(TASK_NAMES)}"
        ),
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--calib-size",
        type=int,
        default=None,
        help="分层随机样本数（默认各任务见 TASK_CALIB_DEFAULTS）",
    )
    parser.add_argument("--calib-seed", type=int, default=DEFAULT_CALIB_SEED)
    parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=DEFAULT_CALIB_OVERSAMPLE_RATIO,
        help=(
            "先分层抽 calib_size×ratio 的 pool，前向保留各 slot min/max 样本后再缩到目标；"
            "设为 1 关闭过采样（旧行为）"
        ),
    )
    parser.add_argument(
        "--calib-seeds",
        type=int,
        default=1,
        help=">1 时额外输出多 seed 稳定性 CSV",
    )
    args = parser.parse_args()

    if not args.tasks:
        print(
            "未指定 --tasks：默认不运行任何数据集"
            "（避免覆盖已有校验集索引）。"
        )
        print(f"示例：python3 coverage_metric.py --tasks mrpc")
        print(f"可选任务：{' '.join(TASK_NAMES)}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if USE_CUDA:
        print(f"device={device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"device={device}")
    print(f"任务：{args.tasks}")
    print(f"输出目录：{args.output_dir}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_rows: list[dict] = []
    for task_name in args.tasks:
        try:
            all_rows.extend(process_task(task_name, args.output_dir, args))
        except (FileNotFoundError, KeyError) as exc:
            print(f"  跳过 {task_name.upper()}：{exc}")

    merged_path = os.path.join(args.output_dir, "all_tasks_validation_coverage.csv")
    if all_rows:
        write_aligned_table(merged_path, COVERAGE_TABLE_COLUMNS, all_rows)
        print(f"\n合并覆盖率表：{merged_path}")


if __name__ == "__main__":
    main()
