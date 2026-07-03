"""
多项式近似 BERT 推理
按 per-dataset 方案数组替换各层 Softmax / GeLU / LayerNorm，在验证集上评估并与原始模型对比。

方案数组顺序（长度 48 = 12 层 × 4）：
  [layer0_softmax, layer0_ln1, layer0_gelu, layer0_ln2,
   layer1_softmax, layer1_ln1, layer1_gelu, layer1_ln2, ...]
元素 0 / 1 / 2 / 3 分别表示低 / 中 / 高多项式方案 / 原始函数。
"""
import os
import types
import functools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)

# ===================== 配置区 =====================
# TASK_NAMES = ["mrpc", "rte", "sst2"]
TASK_NAMES = ["mrpc"]
LOCAL_DATA_ROOT = "./glue_datasets/"
FINETUNED_MODEL_ROOT = "./finetuned_weight/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2
SCHEME_SLOTS_PER_LAYER = 4
SCHEME_LEN = NUM_LAYERS * SCHEME_SLOTS_PER_LAYER

# per-dataset 方案：key=数据集名，value=长度 48 的 0/1/2/3 数组
SCHEME_ORIGINAL = 3
SCHEME_LEVEL_NAMES: dict[int, str] = {
    0: "low",
    1: "mid",
    2: "high",
    SCHEME_ORIGINAL: "original",
}

# 以下为占位示例，请按搜索/分配结果替换
def _example_scheme(level: int = 1) -> list[int]:
    return [level] * SCHEME_LEN


def _expand_scheme_24_to_48(
    scheme24: list[int], ln_level: int = SCHEME_ORIGINAL
) -> list[int]:
    """将旧版 24 维方案扩展为 48 维（顺序：softmax, ln1, gelu, ln2）。"""
    if len(scheme24) != NUM_LAYERS * 2:
        raise ValueError(f"输入方案长度应为 {NUM_LAYERS * 2}，当前为 {len(scheme24)}")
    out: list[int] = []
    for layer_idx in range(NUM_LAYERS):
        out.append(scheme24[layer_idx * 2])
        out.append(ln_level)
        out.append(scheme24[layer_idx * 2 + 1])
        out.append(ln_level)
    return out


def normalize_scheme(
    scheme: list[int],
    *,
    ln_level: int = SCHEME_ORIGINAL,
    task_name: str = "",
    validate: bool = True,
) -> list[int]:
    """接受 24 或 48 维方案，统一返回 48 维。"""
    if len(scheme) == SCHEME_LEN:
        out = list(scheme)
    elif len(scheme) == NUM_LAYERS * 2:
        out = _expand_scheme_24_to_48(scheme, ln_level=ln_level)
    else:
        raise ValueError(
            f"{task_name} 方案长度应为 {SCHEME_LEN} 或 {NUM_LAYERS * 2}，"
            f"当前为 {len(scheme)}"
        )
    if validate:
        validate_scheme(out, task_name)
    return out


def _init_poly_schemes() -> dict[str, list[int]]:
    raw: dict[str, list[int]] = {
        # "mrpc": [
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 0, 1,   1, 1, 0, 1,   1, 1, 1, 1,
        # ],
        # "rte": [
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 0, 1,   1, 1, 0, 1,   1, 1, 1, 1,
        # ],
        # "sst2": [
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 1, 1,   1, 1, 1, 1,   1, 1, 1, 1,
        #     1, 1, 0, 1,   1, 1, 0, 1,   1, 1, 1, 1,
        # ],
        "mrpc": [2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2],
        "rte": [0]*48,
        "sst2": [0]*48,
    }
    return {
        task: normalize_scheme(scheme, task_name=task, validate=True)
        for task, scheme in raw.items()
    }


POLY_SCHEMES: dict[str, list[int]] = {}
# ==================================================


from gelu_poly import (
    GELU_LEVEL_KEYS,
    GELU_POLY_FUNCS,
    GeluPolyEvaluator,
    gelu_level_allowed,
)
from layernorm_poly import HeLayerNormPolyEvaluator
from cost import compute_scheme_cost
from softmax_poly import (
    SOFTMAX_POLY_FUNCS,
    ThorSoftmaxEvaluator,
    additive_attention_mask_to_key_valid,
)


# ===================== 方案工具 =====================

def validate_scheme(scheme: list[int], task_name: str = "") -> None:
    if len(scheme) != SCHEME_LEN:
        raise ValueError(
            f"{task_name} 方案长度应为 {SCHEME_LEN}，当前为 {len(scheme)}"
        )
    for idx, level in enumerate(scheme):
        if level not in SCHEME_LEVEL_NAMES:
            raise ValueError(
                f"{task_name} 方案下标 {idx} 取值非法：{level}（仅允许 0/1/2/3）"
            )
        slot = idx % SCHEME_SLOTS_PER_LAYER
        if slot == 2 and not is_original_scheme(level):
            layer_idx = idx // SCHEME_SLOTS_PER_LAYER
            if not gelu_level_allowed(layer_idx, level):
                allowed = [
                    GELU_LEVEL_KEYS[lvl]
                    for lvl in (0, 1, 2)
                    if gelu_level_allowed(layer_idx, lvl)
                ]
                raise ValueError(
                    f"{task_name} layer {layer_idx} GeLU 档位 "
                    f"{GELU_LEVEL_KEYS[level]} 禁止；允许：{allowed}"
                )


def is_original_scheme(level: int) -> bool:
    return level == SCHEME_ORIGINAL


def scheme_level_name(level: int) -> str:
    return SCHEME_LEVEL_NAMES[level]


def scheme_index(layer_idx: int, nonlinear: str) -> int:
    """nonlinear: 'softmax' | 'ln1' | 'gelu' | 'ln2'"""
    slot = {
        "softmax": 0,
        "ln1": 1,
        "gelu": 2,
        "ln2": 3,
    }.get(nonlinear)
    if slot is None:
        raise ValueError(f"未知非线性类型：{nonlinear}")
    return layer_idx * SCHEME_SLOTS_PER_LAYER + slot


POLY_SCHEMES.update(_init_poly_schemes())


def bind_poly_with_layer(poly_fn, layer_idx: int, task_name: str):
    """
    将带 layer_idx 的多项式函数绑定到指定层。
    返回的可调用对象：
      - Softmax：bound(scores, dim=-1)
      - GeLU：   bound(x)
    """
    if poly_fn in GELU_POLY_FUNCS:
        level = GELU_POLY_FUNCS.index(poly_fn)
        if not gelu_level_allowed(layer_idx, level):
            raise ValueError(
                f"layer {layer_idx} 禁止 GeLU 档位 {GELU_LEVEL_KEYS[level]}"
            )
        return GeluPolyEvaluator(layer_idx, level)
    if poly_fn in SOFTMAX_POLY_FUNCS:
        level = SOFTMAX_POLY_FUNCS.index(poly_fn)
        return ThorSoftmaxEvaluator(task_name, layer_idx, level)
    return functools.partial(poly_fn, layer_idx=layer_idx, task_name=task_name)


# ===================== 模型替换 =====================

_ORIGINAL_EAGER_ATTENTION_FORWARD = None


def install_eager_attention_poly_patch() -> None:
    """全局 patch eager attention：若模块带 _poly_softmax_fn 则用之，否则走精确 softmax。"""
    global _ORIGINAL_EAGER_ATTENTION_FORWARD
    if _ORIGINAL_EAGER_ATTENTION_FORWARD is not None:
        return

    import transformers.models.bert.modeling_bert as bert_modeling

    _ORIGINAL_EAGER_ATTENTION_FORWARD = bert_modeling.eager_attention_forward

    @functools.wraps(_ORIGINAL_EAGER_ATTENTION_FORWARD)
    def patched_eager_attention_forward(
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

        attn_scores = torch.matmul(query, key.transpose(2 , 3)) * scaling

        key_valid_mask = None
        additive_mask = None
        if attention_mask is not None:
            additive_mask = attention_mask[:, :, :, : key.shape[-2]]
            key_valid_mask = additive_attention_mask_to_key_valid(
                additive_mask, attn_scores.dtype
            )

        poly_fn = getattr(module, "_poly_softmax_fn", None)
        if poly_fn is not None:
            if key_valid_mask is None:
                key_valid_mask = torch.ones_like(attn_scores)
            attn_weights = poly_fn(
                attn_scores, dim=-1, key_valid_mask=key_valid_mask
            )
        else:
            if additive_mask is not None:
                attn_scores = attn_scores + additive_mask
            attn_weights = F.softmax(attn_scores, dim=-1)

        attn_weights = F.dropout(
            attn_weights, p=dropout, training=module.training
        )
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights

    bert_modeling.eager_attention_forward = patched_eager_attention_forward


def _patch_gelu_module(gelu_module: torch.nn.Module, poly_fn) -> None:
    """将 GELUActivation 替换为指定多项式函数。"""
    if not hasattr(gelu_module, "_orig_gelu_forward"):
        gelu_module._orig_gelu_forward = gelu_module.forward

    @functools.wraps(gelu_module.forward)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return poly_fn(input)

    gelu_module.forward = types.MethodType(forward, gelu_module)
    gelu_module._poly_gelu_fn = poly_fn


def _restore_gelu_module(gelu_module: torch.nn.Module) -> None:
    """恢复 GeLU 为模块原始 forward（精确 GELU）。"""
    orig = getattr(gelu_module, "_orig_gelu_forward", None)
    if orig is not None:
        gelu_module.forward = orig
    gelu_module._poly_gelu_fn = None
    gelu_module._poly_gelu_level = SCHEME_ORIGINAL


def _restore_layernorm_module(ln_module: torch.nn.Module) -> None:
    """恢复 LayerNorm 为模块原始 forward。"""
    orig = getattr(ln_module, "_orig_ln_forward", None)
    if orig is not None:
        ln_module.forward = orig
    ln_module._poly_ln_fn = None
    ln_module._poly_ln_level = SCHEME_ORIGINAL


def _patch_layernorm_module(
    ln_module: torch.nn.Module, poly_evaluator: HeLayerNormPolyEvaluator
) -> None:
    """将 LayerNorm 替换为 HE 近似。"""
    if not hasattr(ln_module, "_orig_ln_forward"):
        ln_module._orig_ln_forward = ln_module.forward

    @functools.wraps(ln_module.forward)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return poly_evaluator(
            hidden_states, self.weight, self.bias, float(self.eps)
        )

    ln_module.forward = types.MethodType(forward, ln_module)
    ln_module._poly_ln_fn = poly_evaluator
    ln_module._poly_ln_level = poly_evaluator.level


def apply_polynomial_scheme(
    model, scheme: list[int], task_name: str
) -> None:
    """
    按方案为 12 层注册 Softmax / GeLU / LayerNorm 替换。
    档位 0/1/2 用多项式近似；3 保留原始算子。
    需已设置 attn_implementation='eager' 且已调用 install_eager_attention_poly_patch()。
    """
    validate_scheme(scheme)

    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        softmax_level = scheme[scheme_index(layer_idx, "softmax")]
        gelu_level = scheme[scheme_index(layer_idx, "gelu")]
        ln1_level = scheme[scheme_index(layer_idx, "ln1")]
        ln2_level = scheme[scheme_index(layer_idx, "ln2")]

        attn_self = layer.attention.self
        if is_original_scheme(softmax_level):
            if hasattr(attn_self, "_poly_softmax_fn"):
                delattr(attn_self, "_poly_softmax_fn")
        else:
            attn_self._poly_softmax_fn = bind_poly_with_layer(
                SOFTMAX_POLY_FUNCS[softmax_level], layer_idx, task_name
            )
        attn_self._poly_softmax_level = softmax_level
        attn_self._poly_layer_idx = layer_idx

        gelu_module = layer.intermediate.intermediate_act_fn
        if is_original_scheme(gelu_level):
            _restore_gelu_module(gelu_module)
        else:
            _patch_gelu_module(
                gelu_module,
                bind_poly_with_layer(GELU_POLY_FUNCS[gelu_level], layer_idx, task_name),
            )
            gelu_module._poly_gelu_level = gelu_level
        gelu_module._poly_layer_idx = layer_idx

        ln1_module = layer.attention.output.LayerNorm
        if is_original_scheme(ln1_level):
            _restore_layernorm_module(ln1_module)
        else:
            _patch_layernorm_module(
                ln1_module,
                HeLayerNormPolyEvaluator(task_name, layer_idx, "ln1", ln1_level),
            )

        ln2_module = layer.output.LayerNorm
        if is_original_scheme(ln2_level):
            _restore_layernorm_module(ln2_module)
        else:
            _patch_layernorm_module(
                ln2_module,
                HeLayerNormPolyEvaluator(task_name, layer_idx, "ln2", ln2_level),
            )


# ===================== 数据与推理 =====================

def fmt_accuracy(acc: float) -> str:
    """准确率小数 → 百分比字符串。"""
    return f"{acc * 100:.2f}%"


def fmt_accuracy_delta(delta: float) -> str:
    """准确率差值（小数）→ 百分点字符串。"""
    return f"{delta * 100:+.2f}%"


def fmt_metric_delta(delta: float) -> str:
    """指标差值（小数）→ 百分点字符串。"""
    return f"{delta * 100:+.2f}%"


def mrpc_glue_score(accuracy: float, f1: float) -> float:
    """MRPC GLUE 任务分：accuracy 与 F1 的非加权平均。"""
    return (accuracy + f1) / 2.0


def per_sample_output_kl(
    logits_ref: torch.Tensor, logits_approx: torch.Tensor
) -> torch.Tensor:
    """Forward KL(p_ref || p_approx)，按样本返回 shape [N]。"""
    log_p = F.log_softmax(logits_ref.float(), dim=-1)
    log_q = F.log_softmax(logits_approx.float(), dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1)


def mean_output_kl(
    logits_ref: torch.Tensor,
    logits_approx: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """验证集平均 Output KL：KL(p_baseline || p_poly)；可传入 valid_mask 排除无效样本。"""
    kl = per_sample_output_kl(logits_ref, logits_approx)
    if valid_mask is None:
        return float(kl.mean().item())
    m = valid_mask.to(dtype=torch.bool, device=kl.device)
    if not bool(m.any()):
        return float("nan")
    return float(kl[m].mean().item())


def per_sample_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """逐样本交叉熵，shape [N]。"""
    return F.cross_entropy(logits.float(), labels.long(), reduction="none")


def abnormal_output_mask(logits: torch.Tensor) -> torch.Tensor:
    """逐样本：最终 logits 含 NaN/Inf 则视为异常。"""
    return torch.isnan(logits).any(dim=1) | torch.isinf(logits).any(dim=1)


def fmt_oor_pct(oor_pct: float) -> str:
    """输出异常样本占比（0–100 百分点）→ 百分比字符串。"""
    if not math.isfinite(oor_pct):
        return "nan"
    return f"{oor_pct:.2f}%"


def compute_flips_pct(
    baseline_preds: np.ndarray,
    poly_preds: np.ndarray,
    labels: np.ndarray,
) -> float:
    """错误→正确 与 正确→错误 的样本占比（百分点，0–100）。"""
    baseline_ok = baseline_preds == labels
    poly_ok = poly_preds == labels
    wrong_to_right = int((~baseline_ok & poly_ok).sum())
    right_to_wrong = int((baseline_ok & ~poly_ok).sum())
    n = len(labels)
    return 100.0 * (wrong_to_right + right_to_wrong) / n if n > 0 else 0.0


def evaluate_logits_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_name: str,
    *,
    baseline_logits: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    由验证集 logits 计算指标；排除输出异常（NaN/Inf）样本。
    若提供 baseline_logits，delta 与 flips 仅在 poly 有效子集上相对 baseline 计算。
    """
    labels_long = labels.long()
    labels_np = labels_long.numpy()
    preds = logits.argmax(dim=-1).numpy()
    n_total = int(labels.shape[0])

    invalid = abnormal_output_mask(logits).numpy()
    valid = ~invalid
    n_abnormal = int(invalid.sum())
    n_used = int(valid.sum())

    if n_used > 0:
        valid_t = torch.from_numpy(valid)
        acc = float(accuracy_score(labels_np[valid], preds[valid]))
        loss = float(
            per_sample_cross_entropy(logits, labels_long)[valid_t].mean().item()
        )
        if task_name == "mrpc":
            f1 = float(f1_score(labels_np[valid], preds[valid]))
            glue = mrpc_glue_score(acc, f1)
        else:
            f1 = float("nan")
            glue = float("nan")
    else:
        acc = float("nan")
        loss = float("nan")
        f1 = float("nan")
        glue = float("nan")

    out: dict[str, float] = {
        "val_accuracy": acc,
        "val_loss": loss,
        "oor_count": float(n_abnormal),
        "oor_pct": 100.0 * n_abnormal / n_total if n_total > 0 else float("nan"),
        "n_eval_used": float(n_used),
        "flips_pct": float("nan"),
    }
    if task_name == "mrpc":
        out["val_f1"] = f1
        out["val_glue"] = glue

    if baseline_logits is not None and n_used > 0:
        base_preds = baseline_logits.argmax(dim=-1).numpy()
        valid_t = torch.from_numpy(valid)
        base_acc = float(accuracy_score(labels_np[valid], base_preds[valid]))
        base_loss = float(
            per_sample_cross_entropy(baseline_logits, labels_long)[valid_t]
            .mean()
            .item()
        )
        out["accuracy_delta"] = acc - base_acc
        out["loss_delta"] = loss - base_loss
        out["flips_pct"] = compute_flips_pct(
            base_preds[valid], preds[valid], labels_np[valid]
        )
        if task_name == "mrpc":
            base_f1 = float(f1_score(labels_np[valid], base_preds[valid]))
            out["f1_delta"] = f1 - base_f1
            out["glue_delta"] = (
                mrpc_glue_score(acc, f1) - mrpc_glue_score(base_acc, base_f1)
            )
    return out


def _make_compute_metrics(task_name: str):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        metrics = {"accuracy": float(accuracy_score(labels, predictions))}
        if task_name == "mrpc":
            f1 = float(f1_score(labels, predictions))
            metrics["f1"] = f1
            metrics["glue"] = mrpc_glue_score(metrics["accuracy"], f1)
        return metrics

    return compute_metrics


def _parse_validation_metrics(task_name: str, val_results: dict) -> dict[str, float]:
    out = {
        "val_accuracy": float(val_results["eval_accuracy"]),
        "val_loss": float(val_results["eval_loss"]),
    }
    if task_name == "mrpc":
        out["val_f1"] = float(val_results["eval_f1"])
        out["val_glue"] = float(val_results["eval_glue"])
    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _load_tokenizer_and_model(task_name: str):
    finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(finetuned_model_path):
        raise FileNotFoundError(f"模型不存在：{finetuned_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_path,
        num_labels=NUM_LABELS,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()
    return tokenizer, model, finetuned_model_path


def _load_tokenized_validation(task_name: str, tokenizer):
    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    validation = dataset["validation"]
    drop_cols = [c for c in validation.column_names if c != "label"]
    return validation.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=drop_cols,
    )


def evaluate_on_validation(
    model,
    tokenizer,
    task_name: str,
    tokenized_validation=None,
) -> dict[str, float]:
    if tokenized_validation is None:
        tokenized_validation = _load_tokenized_validation(task_name, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=_make_compute_metrics(task_name),
    )

    val_results = trainer.evaluate(tokenized_validation)
    return _parse_validation_metrics(task_name, val_results)


@torch.no_grad()
def collect_output_logits(
    model,
    tokenized_validation,
    tokenizer,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """验证集分类 logits 与 labels，顺序与 tokenized_validation 一致。"""
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized_validation,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for batch in loader:
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        logit_chunks.append(model(**batch).logits.detach().cpu())
        label_chunks.append(labels)
    return torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)


def run_baseline_inference(task_name: str) -> dict:
    # """原始 Softmax / GeLU，验证集评估。"""
    # print(f"\n{'=' * 60}")
    # print(f"原始模型推理：{task_name.upper()}")
    # print(f"{'=' * 60}")

    tokenizer, model, model_path = _load_tokenizer_and_model(task_name)
    # print(f"加载模型：{model_path}")
    # print("验证集评估中...")
    metrics = evaluate_on_validation(model, tokenizer, task_name)
    # print(f"验证集准确率：{fmt_accuracy(metrics['val_accuracy'])}")
    # print(f"验证集损失：  {metrics['val_loss']:.4f}")

    return {"task": task_name, **metrics}


def _warmup_poly_kernels(model, scheme: list[int]) -> None:
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    gelu_warmup = torch.zeros(1, device=model_device, dtype=model_dtype)
    hidden_size = model.config.hidden_size
    ln_warmup = torch.zeros(1, hidden_size, device=model_device, dtype=model_dtype)
    attn_warmup = torch.zeros(
        1, 1, 1, MAX_SEQ_LENGTH, device=model_device, dtype=model_dtype
    )
    key_valid_warmup = torch.ones_like(attn_warmup)
    for layer_idx in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[layer_idx]
        if not is_original_scheme(scheme[scheme_index(layer_idx, "gelu")]):
            layer.intermediate.intermediate_act_fn.forward(gelu_warmup)
        if not is_original_scheme(scheme[scheme_index(layer_idx, "softmax")]):
            layer.attention.self._poly_softmax_fn(
                attn_warmup, dim=-1, key_valid_mask=key_valid_warmup
            )
        if not is_original_scheme(scheme[scheme_index(layer_idx, "ln1")]):
            layer.attention.output.LayerNorm.forward(ln_warmup)
        if not is_original_scheme(scheme[scheme_index(layer_idx, "ln2")]):
            layer.output.LayerNorm.forward(ln_warmup)


def run_poly_inference(task_name: str, scheme: list[int]) -> dict:
    """加载微调模型 → 应用多项式方案 → 验证集评估。"""
    scheme = normalize_scheme(scheme, task_name=task_name)

    print(f"{task_name.upper()}：{scheme}")

    install_eager_attention_poly_patch()

    tokenizer, model, model_path = _load_tokenizer_and_model(task_name)
    tokenized_validation = _load_tokenized_validation(task_name, tokenizer)

    apply_polynomial_scheme(model, scheme, task_name)
    with torch.no_grad():
        _warmup_poly_kernels(model, scheme)
        logits, labels = collect_output_logits(
            model, tokenized_validation, tokenizer
        )
    metrics = evaluate_logits_metrics(logits, labels, task_name)
    total_depth = int(compute_scheme_cost(task_name, scheme))

    return {
        "task": task_name,
        "scheme": scheme,
        "total_depth": total_depth,
        **metrics,
    }


def run_task_with_comparison(task_name: str, scheme: list[int]) -> dict:
    scheme = normalize_scheme(scheme, task_name=task_name)
    print(f"{task_name.upper()}：{scheme}")

    install_eager_attention_poly_patch()
    tokenizer, model, _ = _load_tokenizer_and_model(task_name)
    tokenized_validation = _load_tokenized_validation(task_name, tokenizer)

    with torch.no_grad():
        logits_baseline, labels = collect_output_logits(
            model, tokenized_validation, tokenizer
        )
    baseline = evaluate_logits_metrics(logits_baseline, labels, task_name)
    baseline["task"] = task_name

    apply_polynomial_scheme(model, scheme, task_name)
    with torch.no_grad():
        _warmup_poly_kernels(model, scheme)
        logits_poly, _ = collect_output_logits(
            model, tokenized_validation, tokenizer
        )
    poly = evaluate_logits_metrics(
        logits_poly, labels, task_name, baseline_logits=logits_baseline
    )
    total_depth = int(compute_scheme_cost(task_name, scheme))
    valid_mask = ~abnormal_output_mask(logits_poly)
    output_kl = mean_output_kl(
        logits_baseline, logits_poly, valid_mask=valid_mask
    )

    poly.update(
        {
            "task": task_name,
            "scheme": scheme,
            "total_depth": total_depth,
            "val_output_kl": output_kl,
        }
    )

    result = {
        "task": task_name,
        "baseline_accuracy": baseline["val_accuracy"],
        "baseline_loss": baseline["val_loss"],
        "poly_accuracy": poly["val_accuracy"],
        "poly_loss": poly["val_loss"],
        "accuracy_delta": poly.get("accuracy_delta", float("nan")),
        "loss_delta": poly.get("loss_delta", float("nan")),
        "output_kl": output_kl,
        "flips_pct": poly.get("flips_pct", float("nan")),
        "oor_count": int(poly["oor_count"]),
        "oor_pct": poly["oor_pct"],
        "n_eval_used": int(poly["n_eval_used"]),
        "total_depth": total_depth,
        "scheme": scheme,
    }
    if task_name == "mrpc":
        result.update(
            {
                "baseline_f1": baseline["val_f1"],
                "poly_f1": poly["val_f1"],
                "f1_delta": poly.get("f1_delta", float("nan")),
                "baseline_glue": baseline["val_glue"],
                "poly_glue": poly["val_glue"],
                "glue_delta": poly.get("glue_delta", float("nan")),
            }
        )
    return result


def main():
    results = []
    for task_name in TASK_NAMES:
        if task_name not in POLY_SCHEMES:
            print(f"未配置方案，跳过：{task_name}")
            continue
        scheme = normalize_scheme(POLY_SCHEMES[task_name], task_name=task_name)
        results.append(run_task_with_comparison(task_name, scheme))

    print(f"\n{'=' * 60}")
    print("汇总（近似 vs 原始）")
    print(f"{'=' * 60}")
    print(
        f"{'任务':<6} {'深度和':<6} "
        f"{'Δ准确率':<10} {'Δ损失':<10} "
        f"{'Output KL':<12} {'异常%':<8} {'flips%':<8}"
    )
    print("-" * 110)
    for r in results:
        flips_s = (
            f"{r['flips_pct']:.2f}%"
            if math.isfinite(r["flips_pct"])
            else "nan"
        )
        print(
            f"{r['task']:<8} "
            f"{r['total_depth']:<6} "
            f"{fmt_accuracy_delta(r['accuracy_delta']):<13} "
            f"{r['loss_delta']:+.7f}{'':<3} "
            f"{r['output_kl']:.6f}{'':<4} "
            f"{fmt_oor_pct(r['oor_pct']):<8} "
            f"{flips_s}"
        )
        if r["task"] == "mrpc":
            print(
                f"         {'F1':<6} "
                # f"{fmt_accuracy(r['baseline_f1']):<13} "
                # f"{fmt_accuracy(r['poly_f1']):<13} "
                f"{fmt_metric_delta(r['f1_delta'])}"
            )
            # print(
            #     f"         {'GLUE':<6} "
            #     f"{fmt_accuracy(r['baseline_glue']):<13} "
            #     f"{fmt_accuracy(r['poly_glue']):<13} "
            #     f"{fmt_metric_delta(r['glue_delta'])}  "
            #     f"(acc+f1)/2"
            # )



if __name__ == "__main__":
    main()
