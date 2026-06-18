"""
多项式近似 BERT 推理
按 per-dataset 方案数组替换各层 Softmax / GeLU，在验证集上评估并与原始模型对比。

方案数组顺序（长度 24 = 12 层 × 2）：
  [layer0_softmax, layer0_gelu, layer1_softmax, layer1_gelu, ...]
元素 0 / 1 / 2 / 3 分别表示低 / 中 / 高多项式方案 / 原始函数。
"""
import os
import types
import functools
import torch
import evaluate
import torch.nn.functional as F
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

# per-dataset 方案：key=数据集名，value=长度 24 的 0/1/2/3 数组
SCHEME_ORIGINAL = 3
SCHEME_LEVEL_NAMES: dict[int, str] = {
    0: "low",
    1: "mid",
    2: "high",
    SCHEME_ORIGINAL: "original",
}

# 以下为占位示例，请按搜索/分配结果替换
def _example_scheme(level: int = 1) -> list[int]:
    return [level] * (NUM_LAYERS * 2)


POLY_SCHEMES: dict[str, list[int]] = {
    "mrpc":  [0, 0, 1, 0, 2, 0, 2, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0],
    "rte":  [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "sst2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1, 0, 1, 0, 2, 0, 0, 0, 0, 0],

    # "mrpc": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    # "rte":  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    # "sst2": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],

    # "mrpc": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    # "rte":  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    # "sst2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],

    # "mrpc": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "rte":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "sst2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    
}
# ==================================================


from gelu_poly import (
    GELU_LEVEL_KEYS,
    GELU_POLY_FUNCS,
    GeluPolyEvaluator,
    gelu_level_allowed,
)
from cost import compute_scheme_cost
from softmax_poly import (
    SOFTMAX_POLY_FUNCS,
    ThorSoftmaxEvaluator,
    additive_attention_mask_to_key_valid,
)


# ===================== 方案工具 =====================

def validate_scheme(scheme: list[int], task_name: str = "") -> None:
    expected_len = NUM_LAYERS * 2
    if len(scheme) != expected_len:
        raise ValueError(
            f"{task_name} 方案长度应为 {expected_len}，当前为 {len(scheme)}"
        )
    for idx, level in enumerate(scheme):
        if level not in SCHEME_LEVEL_NAMES:
            raise ValueError(
                f"{task_name} 方案下标 {idx} 取值非法：{level}（仅允许 0/1/2/3）"
            )
        if idx % 2 == 1 and not is_original_scheme(level):
            layer_idx = idx // 2
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
    """nonlinear: 'softmax' | 'gelu'"""
    if nonlinear == "softmax":
        return layer_idx * 2
    if nonlinear == "gelu":
        return layer_idx * 2 + 1
    raise ValueError(f"未知非线性类型：{nonlinear}")


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


def apply_polynomial_scheme(
    model, scheme: list[int], task_name: str
) -> None:
    """
    按方案为 12 层注册 Softmax / GeLU 替换。
    档位 0/1/2 用多项式近似；3 保留原始 Softmax / GeLU。
    需已设置 attn_implementation='eager' 且已调用 install_eager_attention_poly_patch()。
    """
    validate_scheme(scheme)

    for layer_idx in range(NUM_LAYERS):
        softmax_level = scheme[scheme_index(layer_idx, "softmax")]
        gelu_level = scheme[scheme_index(layer_idx, "gelu")]

        attn_self = model.bert.encoder.layer[layer_idx].attention.self
        if is_original_scheme(softmax_level):
            if hasattr(attn_self, "_poly_softmax_fn"):
                delattr(attn_self, "_poly_softmax_fn")
        else:
            attn_self._poly_softmax_fn = bind_poly_with_layer(
                SOFTMAX_POLY_FUNCS[softmax_level], layer_idx, task_name
            )
        attn_self._poly_softmax_level = softmax_level
        attn_self._poly_layer_idx = layer_idx

        gelu_module = model.bert.encoder.layer[layer_idx].intermediate.intermediate_act_fn
        if is_original_scheme(gelu_level):
            _restore_gelu_module(gelu_module)
        else:
            _patch_gelu_module(
                gelu_module,
                bind_poly_with_layer(GELU_POLY_FUNCS[gelu_level], layer_idx, task_name),
            )
            gelu_module._poly_gelu_level = gelu_level
        gelu_module._poly_layer_idx = layer_idx


# ===================== 数据与推理 =====================

def fmt_accuracy(acc: float) -> str:
    """准确率小数 → 百分比字符串。"""
    return f"{acc * 100:.2f}%"


def fmt_accuracy_delta(delta: float) -> str:
    """准确率差值（小数）→ 百分点字符串。"""
    return f"{delta * 100:+.2f}%"


metric = evaluate.load("accuracy")
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


def evaluate_on_validation(
    model,
    tokenizer,
    task_name: str,
) -> dict[str, float]:
    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    tokenized_validation = dataset["validation"].map(
        get_preprocess_fn(task_name, tokenizer), batched=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    val_results = trainer.evaluate(tokenized_validation)
    return {
        "val_accuracy": float(val_results["eval_accuracy"]),
        "val_loss": float(val_results["eval_loss"]),
    }


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


def run_poly_inference(task_name: str, scheme: list[int]) -> dict:
    """加载微调模型 → 应用多项式方案 → 验证集评估。"""
    validate_scheme(scheme, task_name)

    # print(f"\n{'=' * 60}")
    # print(f"多项式近似推理：{task_name.upper()}")
    # print(f"{'=' * 60}")
    print(f"{task_name.upper()}：{scheme}")

    install_eager_attention_poly_patch()

    tokenizer, model, model_path = _load_tokenizer_and_model(task_name)
    #print(f"加载模型：{model_path}")

    apply_polynomial_scheme(model, scheme, task_name)
    n_poly_softmax = sum(
        1
        for i in range(NUM_LAYERS)
        if not is_original_scheme(scheme[scheme_index(i, "softmax")])
    )
    n_poly_gelu = sum(
        1
        for i in range(NUM_LAYERS)
        if not is_original_scheme(scheme[scheme_index(i, "gelu")])
    )
    # print(
    #     f"已应用方案：Softmax 近似 {n_poly_softmax}/{NUM_LAYERS} 层，"
    #     f"GeLU 近似 {n_poly_gelu}/{NUM_LAYERS} 层（3=原始函数）"
    # )
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        gelu_warmup = torch.zeros(1, device=model_device, dtype=model_dtype)
        attn_warmup = torch.zeros(
            1, 1, 1, MAX_SEQ_LENGTH, device=model_device, dtype=model_dtype
        )
        key_valid_warmup = torch.ones_like(attn_warmup)
        for layer_idx in range(NUM_LAYERS):
            if not is_original_scheme(scheme[scheme_index(layer_idx, "gelu")]):
                gelu_module = model.bert.encoder.layer[
                    layer_idx
                ].intermediate.intermediate_act_fn
                gelu_module.forward(gelu_warmup)
            if not is_original_scheme(scheme[scheme_index(layer_idx, "softmax")]):
                attn_self = model.bert.encoder.layer[layer_idx].attention.self
                attn_self._poly_softmax_fn(
                    attn_warmup, dim=-1, key_valid_mask=key_valid_warmup
                )

    metrics = evaluate_on_validation(model, tokenizer, task_name)
    total_depth = int(compute_scheme_cost(task_name, scheme))
    # print(f"验证集准确率：{fmt_accuracy(metrics['val_accuracy'])}")
    # print(f"验证集损失：  {metrics['val_loss']:.4f}")

    return {
        "task": task_name,
        "scheme": scheme,
        "total_depth": total_depth,
        **metrics,
    }


def run_task_with_comparison(task_name: str, scheme: list[int]) -> dict:
    baseline = run_baseline_inference(task_name)
    poly = run_poly_inference(task_name, scheme)

    acc_delta = poly["val_accuracy"] - baseline["val_accuracy"]
    loss_delta = poly["val_loss"] - baseline["val_loss"]

    # print(f"\n--- {task_name.upper()} 近似 vs 原始 ---")
    # print(
    #     f"{'':<12} {'准确率':>10} {'损失':>10}\n"
    #     f"{'原始':<12} {fmt_accuracy(baseline['val_accuracy']):>10} "
    #     f"{baseline['val_loss']:10.4f}\n"
    #     f"{'多项式近似':<12} {fmt_accuracy(poly['val_accuracy']):>10} "
    #     f"{poly['val_loss']:10.4f}\n"
    #     f"{'差值(近似-原始)':<12} {fmt_accuracy_delta(acc_delta):>10} "
    #     f"{loss_delta:+10.4f}"
    # )

    return {
        "task": task_name,
        "baseline_accuracy": baseline["val_accuracy"],
        "baseline_loss": baseline["val_loss"],
        "poly_accuracy": poly["val_accuracy"],
        "poly_loss": poly["val_loss"],
        "accuracy_delta": acc_delta,
        "loss_delta": loss_delta,
        "total_depth": poly["total_depth"],
        "scheme": scheme,
    }


def main():
    results = []
    for task_name in TASK_NAMES:
        if task_name not in POLY_SCHEMES:
            print(f"未配置方案，跳过：{task_name}")
            continue
        scheme = POLY_SCHEMES[task_name]
        results.append(run_task_with_comparison(task_name, scheme))

    print(f"\n{'=' * 60}")
    print("汇总（近似 vs 原始）")
    print(f"{'=' * 60}")
    print(
        f"{'任务':<6} {'深度和':<6} {'原始准确率':<8} {'近似准确率':<8} "
        f"{'Δ准确率':<8} {'原始损失':<8} {'近似损失':<8} {'Δ损失':<8}"
    )
    print("-" * 86)
    for r in results:
        print(
            f"{r['task']:<8} "
            f"{r['total_depth']:<6} "
            f"{fmt_accuracy(r['baseline_accuracy']):<13} "
            f"{fmt_accuracy(r['poly_accuracy']):<12} "
            f"{fmt_accuracy_delta(r['accuracy_delta']):<12} "
            f"{r['baseline_loss']:.7f}{'':<6} "
            f"{r['poly_loss']:.7f}{'':<4} "
            f"{r['loss_delta']:+.7f}"
        )



if __name__ == "__main__":
    main()
