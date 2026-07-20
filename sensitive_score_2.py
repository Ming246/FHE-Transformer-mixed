"""
试验：层内完整二阶 Taylor 敏感度（vs sensitive_score.py 的对角近似）。

理论（固定层输出 y，其余层精确，扰动 Δy = y_poly − y_exact）：

  ΔL ≈ g^T Δy + ½ Δy^T H Δy
  g = ∂L/∂y，  H = ∂²L/∂y²

完整二阶含所有 element 交叉项 ½ Σ_{i≠j} H_{ij} Δy_i Δy_j，无需构造 d×d Hessian，
用 HVP（Pearlmutter）：v=Δy 时  ∇_y(g^T v) = H v，故  Δy^T H Δy = (Δy · HVP)。

sensitive_score.py 的「二阶项」实为对角代理（非真 Hessian）：
  Σ_j (g_j Δy_j + ½ g_j² Δy_j²)  等价于假设 H_{ij}≈0 (i≠j) 且 H_{jj}≈g_j²

本脚本固定使用完整二阶：S = |g^T Δy + ½ Δy^T H Δy|。
三种方法的 CSV 对比见 score_compare.py。

局限：单层扰动、忽略跨 slot 交叉；full 仅在该层 y 上二阶准确。
"""
from __future__ import annotations

import functools
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from gelu_poly import GELU_LEVEL_KEYS, GeluPolyEvaluator, gelu_level_allowed
from layernorm_poly import (
    LAYERNORM_KIND_KEYS,
    LAYERNORM_LEVEL_KEYS,
    HeLayerNormPolyEvaluator,
)
from softmax_poly import (
    SOFTMAX_LEVEL_KEYS,
    ThorSoftmaxEvaluator,
    additive_attention_mask_to_key_valid,
)

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]

LOCAL_DATA_ROOT = "./glue_datasets/"
FINETUNED_MODEL_ROOT = "./finetuned_weight/"
SENSITIVE_OUTPUT_DIR = "./results/sensitive_scores_2/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2
BATCH_SIZE = 1
NUM_PRECISION_LEVELS = 3
MAX_CALIB_SAMPLES = None

SCHEME_KINDS_PER_LAYER = ("softmax", "ln1", "gelu", "ln2")
SCHEME_SLOTS_PER_LAYER = len(SCHEME_KINDS_PER_LAYER)
SCHEME_LEN = NUM_LAYERS * SCHEME_SLOTS_PER_LAYER
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ORIGINAL_EAGER_ATTENTION_FORWARD = None
_attn_module_layer_map: dict[int, int] = {}
_gelu_hook_handles: list = []
_ln_hook_handles: list = []

# layer_idx -> dict with y_exact, grad via retain_grad, and inputs for Δy
_softmax_capture: dict[int, dict] = {}
_gelu_capture: dict[int, dict] = {}
_ln_capture: dict[tuple[int, str], dict] = {}


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


def position_label(layer_idx: int, kind: str, level: int) -> str:
    if kind == "softmax":
        prec = SOFTMAX_LEVEL_KEYS[level]
    elif kind in LAYERNORM_KIND_KEYS:
        prec = LAYERNORM_LEVEL_KEYS[level]
    else:
        prec = GELU_LEVEL_KEYS[level]
    return f"layer{layer_idx}_{kind}_{prec}"


def bind_attention_layer_indices(model) -> None:
    global _attn_module_layer_map
    _attn_module_layer_map = {}
    for layer_idx in range(NUM_LAYERS):
        attn_self = model.bert.encoder.layer[layer_idx].attention.self
        attn_self.layer_idx = layer_idx
        _attn_module_layer_map[id(attn_self)] = layer_idx


def install_softmax_output_capture() -> None:
    """捕获 attention 概率（dropout 前）、原 scores 与 key_valid_mask。"""
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return

    import transformers.models.bert.modeling_bert as bert_modeling

    _ORIGINAL_EAGER_ATTENTION_FORWARD = bert_modeling.eager_attention_forward

    @functools.wraps(_ORIGINAL_EAGER_ATTENTION_FORWARD)
    def capturing_eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=None,
        dropout=0.0,
        **kwargs,
    ):
        if scaling is None:
            scaling = query.size(-1) ** -0.5

        attn_scores = torch.matmul(query, key.transpose(2, 3)) * scaling

        key_valid_mask = None
        additive_mask = None
        if attention_mask is not None:
            additive_mask = attention_mask[:, :, :, : key.shape[-2]]
            key_valid_mask = additive_attention_mask_to_key_valid(
                additive_mask, attn_scores.dtype
            )
            attn_scores_for_exact = attn_scores + additive_mask
        else:
            attn_scores_for_exact = attn_scores
            key_valid_mask = torch.ones_like(attn_scores)

        attn_probs = F.softmax(attn_scores_for_exact, dim=-1)
        attn_probs.retain_grad()

        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            layer_idx = _attn_module_layer_map.get(id(module))
        if layer_idx is not None:
            _softmax_capture[layer_idx] = {
                "y_exact": attn_probs,
                "scores": attn_scores,
                "key_valid_mask": key_valid_mask,
            }

        attn_probs = F.dropout(attn_probs, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_probs

    bert_modeling.eager_attention_forward = capturing_eager_attention_forward


def register_gelu_capture_hooks(model) -> None:
    """捕获 GeLU 输入与激活后输出（输出 retain_grad）。"""
    global _gelu_hook_handles
    for h in _gelu_hook_handles:
        h.remove()
    _gelu_hook_handles.clear()

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, output):
            output.retain_grad()
            gelu_in = inp[0]
            _gelu_capture[layer_idx] = {
                "y_exact": output,
                "gelu_input": gelu_in,
            }

        return hook_fn

    for layer_idx in range(NUM_LAYERS):
        act_fn = model.bert.encoder.layer[layer_idx].intermediate.intermediate_act_fn
        h = act_fn.register_forward_hook(make_hook(layer_idx))
        _gelu_hook_handles.append(h)


def register_layernorm_capture_hooks(model) -> None:
    """捕获 ln1/ln2 输入与输出（输出 retain_grad）。"""
    global _ln_hook_handles
    for h in _ln_hook_handles:
        h.remove()
    _ln_hook_handles.clear()

    def make_hook(layer_idx: int, kind: str):
        def hook_fn(module, inp, output):
            output.retain_grad()
            _ln_capture[(layer_idx, kind)] = {
                "y_exact": output,
                "ln_input": inp[0],
            }

        return hook_fn

    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        for kind, ln_module in (
            ("ln1", layer.attention.output.LayerNorm),
            ("ln2", layer.output.LayerNorm),
        ):
            h = ln_module.register_forward_hook(make_hook(layer_idx, kind))
            _ln_hook_handles.append(h)


def _build_poly_evaluators(task_name: str) -> tuple[dict, dict, dict]:
    softmax_eval = {
        layer_idx: [
            ThorSoftmaxEvaluator(task_name, layer_idx, level)
            for level in range(NUM_PRECISION_LEVELS)
        ]
        for layer_idx in range(NUM_LAYERS)
    }
    gelu_eval = {
        layer_idx: {
            level: GeluPolyEvaluator(layer_idx, level)
            for level in range(NUM_PRECISION_LEVELS)
            if gelu_level_allowed(layer_idx, level)
        }
        for layer_idx in range(NUM_LAYERS)
    }
    layernorm_eval = {
        (layer_idx, kind): {
            level: HeLayerNormPolyEvaluator(task_name, layer_idx, kind, level)
            for level in range(NUM_PRECISION_LEVELS)
        }
        for layer_idx in range(NUM_LAYERS)
        for kind in LAYERNORM_KIND_KEYS
    }
    return softmax_eval, gelu_eval, layernorm_eval


def _full_taylor2_batch_sum(
    y: torch.Tensor,
    loss: torch.Tensor,
    delta_y: torch.Tensor,
    batch_size: int,
) -> float:
    """Σ_m | g^T Δy + ½ Δy^T H Δy |，HVP 实现完整层内二阶项。"""
    g = torch.autograd.grad(
        loss,
        y,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return 0.0

    hvp = torch.autograd.grad(
        g,
        y,
        grad_outputs=delta_y,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if hvp is None:
        hvp = torch.zeros_like(delta_y)

    taylor2 = g.detach() * delta_y + 0.5 * delta_y * hvp.detach()
    return taylor2.reshape(batch_size, -1).sum(dim=1).abs().sum().item()


def _accumulate_softmax_scores_for_batch(
    softmax_evaluators: dict[int, list[ThorSoftmaxEvaluator]],
    accum: dict[tuple[int, int], float],
    batch_size: int,
    *,
    loss: torch.Tensor,
) -> None:
    for layer_idx in range(NUM_LAYERS):
        cap = _softmax_capture.get(layer_idx)
        if cap is None:
            continue
        y_exact = cap["y_exact"]
        scores = cap["scores"]
        key_valid = cap["key_valid_mask"]

        for level in range(NUM_PRECISION_LEVELS):
            with torch.no_grad():
                y_poly = softmax_evaluators[layer_idx][level](
                    scores, dim=-1, key_valid_mask=key_valid
                )
                delta_y = y_poly - y_exact.detach()
            key = (layer_idx, level)
            accum[key] = accum.get(key, 0.0) + _full_taylor2_batch_sum(
                y_exact, loss, delta_y, batch_size
            )


def _accumulate_gelu_scores_for_batch(
    gelu_evaluators: dict[int, dict[int, GeluPolyEvaluator]],
    accum: dict[tuple[int, int], float],
    batch_size: int,
    *,
    loss: torch.Tensor,
) -> None:
    for layer_idx in range(NUM_LAYERS):
        cap = _gelu_capture.get(layer_idx)
        if cap is None:
            continue
        y_exact = cap["y_exact"]
        gelu_in = cap["gelu_input"]

        for level, evaluator in gelu_evaluators[layer_idx].items():
            with torch.no_grad():
                y_poly = evaluator(gelu_in)
                delta_y = y_poly - y_exact.detach()
            key = (layer_idx, level)
            accum[key] = accum.get(key, 0.0) + _full_taylor2_batch_sum(
                y_exact, loss, delta_y, batch_size
            )


def _accumulate_layernorm_scores_for_batch(
    model,
    layernorm_evaluators: dict[tuple[int, str], dict[int, HeLayerNormPolyEvaluator]],
    accum: dict[tuple[int, str, int], float],
    batch_size: int,
    *,
    loss: torch.Tensor,
) -> None:
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        for kind, ln_module in (
            ("ln1", layer.attention.output.LayerNorm),
            ("ln2", layer.output.LayerNorm),
        ):
            cap = _ln_capture.get((layer_idx, kind))
            if cap is None:
                continue
            y_exact = cap["y_exact"]
            ln_input = cap["ln_input"]
            gamma = ln_module.weight
            beta = ln_module.bias
            eps = float(ln_module.eps)

            for level, evaluator in layernorm_evaluators[(layer_idx, kind)].items():
                with torch.no_grad():
                    y_poly = evaluator(ln_input, gamma, beta, eps)
                    delta_y = y_poly - y_exact.detach()
                key = (layer_idx, kind, level)
                accum[key] = accum.get(key, 0.0) + _full_taylor2_batch_sum(
                    y_exact, loss, delta_y, batch_size
                )


def compute_sensitivity_for_task(
    task_name: str,
) -> tuple[dict[str, float], int]:
    finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"模型不存在：{finetuned_model_path}")

    install_softmax_output_capture()
    softmax_evaluators, gelu_evaluators, layernorm_evaluators = _build_poly_evaluators(
        task_name
    )

    print(f"\n{'=' * 60}")
    print(f"Taylor 敏感度（完整二阶 HVP）：{task_name.upper()}")
    print(f"{'=' * 60}")
    print("S = |g^T Δy + ½ Δy^T H Δy|；HVP 含层内交叉项")
    print("Δy = y_poly − y_exact；三档 low/mid/high")

    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_path,
        num_labels=NUM_LABELS,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    bind_attention_layer_indices(model)
    register_gelu_capture_hooks(model)
    register_layernorm_capture_hooks(model)

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    calib = dataset["validation"]
    if MAX_CALIB_SAMPLES is not None:
        calib = calib.select(range(min(MAX_CALIB_SAMPLES, len(calib))))

    tokenized = calib.map(get_preprocess_fn(task_name, tokenizer), batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    softmax_accum: dict[tuple[int, int], float] = {}
    gelu_accum: dict[tuple[int, int], float] = {}
    layernorm_accum: dict[tuple[int, str, int], float] = {}
    sample_count = 0

    print(f"校准样本数 M = {len(tokenized)}，batch_size = {BATCH_SIZE}")

    for step, batch in enumerate(loader):
        global _softmax_capture, _gelu_capture, _ln_capture
        _softmax_capture = {}
        _gelu_capture = {}
        _ln_capture = {}

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label_key = "labels" if "labels" in batch else "label"
        labels = batch[label_key].to(device)
        batch_size = labels.size(0)

        model.zero_grad(set_to_none=True)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss * batch_size
        _accumulate_softmax_scores_for_batch(
            softmax_evaluators,
            softmax_accum,
            batch_size,
            loss=loss,
        )
        _accumulate_gelu_scores_for_batch(
            gelu_evaluators,
            gelu_accum,
            batch_size,
            loss=loss,
        )
        _accumulate_layernorm_scores_for_batch(
            model,
            layernorm_evaluators,
            layernorm_accum,
            batch_size,
            loss=loss,
        )

        sample_count += batch_size
        if (step + 1) % 20 == 0:
            print(f"  已处理 {sample_count}/{len(tokenized)} 样本...")

    scores: dict[str, float] = {}
    for layer_idx in range(NUM_LAYERS):
        for kind in SCHEME_KINDS_PER_LAYER:
            for level in range(NUM_PRECISION_LEVELS):
                if kind == "softmax":
                    s_val = softmax_accum.get((layer_idx, level), 0.0) / sample_count
                elif kind == "gelu":
                    if gelu_level_allowed(layer_idx, level):
                        s_val = gelu_accum.get((layer_idx, level), 0.0) / sample_count
                    else:
                        s_val = 0.0
                else:
                    s_val = (
                        layernorm_accum.get((layer_idx, kind, level), 0.0)
                        / sample_count
                    )
                scores[position_label(layer_idx, kind, level)] = s_val

    return scores, sample_count


def print_scores(task_name: str, scores: dict[str, float], sample_count: int) -> None:
    print(f"\n任务 {task_name.upper()}  |  M = {sample_count}")
    print(f"{'位置':<28} {'S':>14}")
    print("-" * 44)
    for layer_idx in range(NUM_LAYERS):
        for kind in SCHEME_KINDS_PER_LAYER:
            for level in range(NUM_PRECISION_LEVELS):
                key = position_label(layer_idx, kind, level)
                print(f"{key:<28} {scores[key]:>14.6e}")
    print("-" * 44)


def save_scores_csv(task_name: str, scores: dict[str, float]) -> str:
    """宽表：每 (layer, kind) 一行，列为 S_low / S_mid / S_high（48 行/任务）。"""
    os.makedirs(SENSITIVE_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(SENSITIVE_OUTPUT_DIR, f"{task_name}_sensitivity.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("layer_idx,kind,S_low,S_mid,S_high\n")
        for layer_idx in range(NUM_LAYERS):
            for kind in SCHEME_KINDS_PER_LAYER:
                s = [
                    scores[position_label(layer_idx, kind, level)]
                    for level in range(NUM_PRECISION_LEVELS)
                ]
                f.write(
                    f"{layer_idx},{kind},"
                    f"{s[0]:.10e},{s[1]:.10e},{s[2]:.10e}\n"
                )
    return path


def main():
    print(f"设备：{device}")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    task_timings: list[tuple[str, float]] = []
    t_all = time.perf_counter()
    for task_name in TASK_NAMES:
        t0 = time.perf_counter()
        scores, m = compute_sensitivity_for_task(task_name)
        print_scores(task_name, scores, m)
        csv_path = save_scores_csv(task_name, scores)
        elapsed = time.perf_counter() - t0
        task_timings.append((task_name, elapsed))
        print(f"已保存：{csv_path}")
        print(f"耗时：{elapsed:.2f}s")
        all_results[task_name] = scores

    total_elapsed = time.perf_counter() - t_all
    print(f"\n{'=' * 60}")
    print(
        "三任务完成（CSV 宽表 48 行/任务："
        "layer × softmax|ln1|gelu|ln2 × S_low|S_mid|S_high）"
    )
    print(f"{'=' * 60}")
    for task_name in TASK_NAMES:
        top = sorted(all_results[task_name].items(), key=lambda x: x[1], reverse=True)[:5]
        print(
            f"{task_name.upper()} Top-5：",
            ", ".join(f"{k}={v:.4e}" for k, v in top),
        )
    print("耗时统计：")
    for task_name, elapsed in task_timings:
        print(f"  {task_name.upper()}：{elapsed:.2f}s")
    print(f"  合计：{total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
