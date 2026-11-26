"""LayerNorm 输入的有效 token 方差统计与 hook。

供 ``coverage_metric.py`` 与 ``3.Infer_Fintuned.py`` 使用。口径与离线评估脚本一致：
按 hidden 向量逐行算总体方差，只统计 attention mask 上的有效 token。
"""
from __future__ import annotations

import functools

import torch

from layernorm_poly import NUM_LAYERS

STATS_CHUNK_SIZE = 4096
LayerNormKind = str  # "ln1" | "ln2"


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


class GpuLayerNormCollector:
    """收集全部 LayerNorm 输入向量 [N, D]。"""

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
