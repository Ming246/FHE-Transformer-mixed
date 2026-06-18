"""
计算 BERT 各非线性输出位置的 Taylor 型敏感度 S_l（三档近似误差）。

对每个位置 l、精度档 k∈{low,mid,high}，在验证集上：

  S_{l,j}^{(k)} = |g_{l,j}·Δy_{l,j}^{(k)} + ½·g_{l,j}²·(Δy_{l,j}^{(k)})²|
  S_l^{(k)}     = Σ_j S_{l,j}^{(k)}
  S̄_l^{(k)}     = (1/M) Σ_m S_{l,m}^{(k)}

- g：原模型（精确 Softmax/GeLU）反传得到的一阶梯度（保留符号）
- y：Softmax=attention 概率（dropout 前）；GeLU=激活后 hidden
- Δy = y_poly − y_exact（近似输出减精确输出，保留符号）
- 每个 batch：1 次精确 forward+backward；Δy 在缓存的中间结果上逐层串行、
  同层三档并行用 gelu_poly / softmax_poly 计算
"""
from __future__ import annotations

import functools
import os
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
from softmax_poly import (
    SOFTMAX_LEVEL_KEYS,
    ThorSoftmaxEvaluator,
    additive_attention_mask_to_key_valid,
)

# ===================== 配置区 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]

LOCAL_DATA_ROOT = "./glue_datasets/"
FINETUNED_MODEL_ROOT = "./finetuned_weight/"
SENSITIVE_OUTPUT_DIR = "./results/sensitive_scores/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2
BATCH_SIZE = 1
NUM_PRECISION_LEVELS = 3
MAX_CALIB_SAMPLES = None
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ORIGINAL_EAGER_ATTENTION_FORWARD = None
_attn_module_layer_map: dict[int, int] = {}
_gelu_hook_handles: list = []

# layer_idx -> dict with y_exact, grad via retain_grad, and inputs for Δy
_softmax_capture: dict[int, dict] = {}
_gelu_capture: dict[int, dict] = {}


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
    """捕获 GeLU 输入与激活后输出（均 retain_grad 作用于输出）。"""
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


def _build_poly_evaluators(task_name: str) -> tuple[dict, dict]:
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
    return softmax_eval, gelu_eval


def _taylor_s_batch_sum(
    grad: torch.Tensor, delta_y: torch.Tensor, batch_size: int
) -> float:
    """Σ_m Σ_j |g·Δy + ½·g²·Δy²|，对当前 batch。"""
    term = grad * delta_y + 0.5 * (grad ** 2) * (delta_y ** 2)
    return term.abs().reshape(batch_size, -1).sum(dim=1).sum().item()


def _accumulate_softmax_scores_for_batch(
    softmax_evaluators: dict[int, list[ThorSoftmaxEvaluator]],
    accum: dict[tuple[int, int], float],
    batch_size: int,
) -> None:
    for layer_idx in range(NUM_LAYERS):
        cap = _softmax_capture.get(layer_idx)
        if cap is None:
            continue
        y_exact = cap["y_exact"]
        if y_exact.grad is None:
            continue
        grad = y_exact.grad.detach()
        scores = cap["scores"]
        key_valid = cap["key_valid_mask"]

        with torch.no_grad():
            for level in range(NUM_PRECISION_LEVELS):
                y_poly = softmax_evaluators[layer_idx][level](
                    scores, dim=-1, key_valid_mask=key_valid
                )
                delta_y = y_poly - y_exact.detach()
                key = (layer_idx, level)
                accum[key] = accum.get(key, 0.0) + _taylor_s_batch_sum(
                    grad, delta_y, batch_size
                )


def _accumulate_gelu_scores_for_batch(
    gelu_evaluators: dict[int, dict[int, GeluPolyEvaluator]],
    accum: dict[tuple[int, int], float],
    batch_size: int,
) -> None:
    for layer_idx in range(NUM_LAYERS):
        cap = _gelu_capture.get(layer_idx)
        if cap is None:
            continue
        y_exact = cap["y_exact"]
        if y_exact.grad is None:
            continue
        grad = y_exact.grad.detach()
        gelu_in = cap["gelu_input"]

        with torch.no_grad():
            for level, evaluator in gelu_evaluators[layer_idx].items():
                y_poly = evaluator(gelu_in)
                delta_y = y_poly - y_exact.detach()
                key = (layer_idx, level)
                accum[key] = accum.get(key, 0.0) + _taylor_s_batch_sum(
                    grad, delta_y, batch_size
                )


def compute_sensitivity_for_task(
    task_name: str,
) -> tuple[dict[str, float], int]:
    finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"模型不存在：{finetuned_model_path}")

    install_softmax_output_capture()
    softmax_evaluators, gelu_evaluators = _build_poly_evaluators(task_name)

    print(f"\n{'=' * 60}")
    print(f"Taylor 敏感度：{task_name.upper()}")
    print(f"{'=' * 60}")
    print(
        "S = Σ_j |g_j·Δy_j + ½·g_j²·Δy_j²|；"
        "Δy = y_poly − y_exact；三档 low/mid/high"
    )

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
    sample_count = 0

    print(f"校准样本数 M = {len(tokenized)}，batch_size = {BATCH_SIZE}")

    for step, batch in enumerate(loader):
        global _softmax_capture, _gelu_capture
        _softmax_capture = {}
        _gelu_capture = {}

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
        loss.backward()

        _accumulate_softmax_scores_for_batch(
            softmax_evaluators, softmax_accum, batch_size
        )
        _accumulate_gelu_scores_for_batch(
            gelu_evaluators, gelu_accum, batch_size
        )

        sample_count += batch_size
        if (step + 1) % 20 == 0:
            print(f"  已处理 {sample_count}/{len(tokenized)} 样本...")

    scores: dict[str, float] = {}
    for layer_idx in range(NUM_LAYERS):
        for level in range(NUM_PRECISION_LEVELS):
            s_soft = softmax_accum.get((layer_idx, level), 0.0) / sample_count
            if gelu_level_allowed(layer_idx, level):
                s_gelu = gelu_accum.get((layer_idx, level), 0.0) / sample_count
            else:
                s_gelu = 0.0
            scores[position_label(layer_idx, "softmax", level)] = s_soft
            scores[position_label(layer_idx, "gelu", level)] = s_gelu

    return scores, sample_count


def print_scores(task_name: str, scores: dict[str, float], sample_count: int) -> None:
    print(f"\n任务 {task_name.upper()}  |  M = {sample_count}")
    print(f"{'位置':<28} {'S':>14}")
    print("-" * 44)
    for layer_idx in range(NUM_LAYERS):
        for kind in ("softmax", "gelu"):
            for level in range(NUM_PRECISION_LEVELS):
                key = position_label(layer_idx, kind, level)
                print(f"{key:<28} {scores[key]:>14.6e}")
    print("-" * 44)


def save_scores_csv(task_name: str, scores: dict[str, float]) -> str:
    """宽表：每 (layer, kind) 一行，列为 S_low / S_mid / S_high（24 行/任务）。"""
    os.makedirs(SENSITIVE_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(SENSITIVE_OUTPUT_DIR, f"{task_name}_sensitivity.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("layer_idx,kind,S_low,S_mid,S_high\n")
        for layer_idx in range(NUM_LAYERS):
            for kind in ("softmax", "gelu"):
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
    for task_name in TASK_NAMES:
        scores, m = compute_sensitivity_for_task(task_name)
        print_scores(task_name, scores, m)
        csv_path = save_scores_csv(task_name, scores)
        print(f"已保存：{csv_path}")
        all_results[task_name] = scores

    print(f"\n{'=' * 60}")
    print("三任务完成（CSV 宽表 24 行/任务：layer × softmax|gelu × S_low|S_mid|S_high）")
    print(f"{'=' * 60}")
    for task_name in TASK_NAMES:
        top = sorted(all_results[task_name].items(), key=lambda x: x[1], reverse=True)[:5]
        print(
            f"{task_name.upper()} Top-5：",
            ", ".join(f"{k}={v:.4e}" for k, v in top),
        )


if __name__ == "__main__":
    main()
