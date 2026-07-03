"""
微调模型推理 - MRPC / RTE / SST2
使用钩子统计 12 层 × 4 个非线性函数的输入范围。
  Softmax / GeLU：输入张量元素 min/max
  LayerNorm1/2：与 nolinear/layernorm.py 相同口径，统计有效 token 上 Var(x) 的
                真实 min/max（按 hidden 向量逐行算总体方差，不做 0.9/1.1 缩放）
输出：每个任务的准确率 + 合并的 48×6 CSV
"""
import importlib.util
import math
import os
import csv
import torch
from sklearn.metrics import accuracy_score
import functools
import torch.nn.functional as F
from datetime import datetime
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

# ===================== 【核心配置区】 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]

LOCAL_DATA_ROOT = "./glue_datasets/"
FINETUNED_MODEL_ROOT = "./finetuned_weight/"
INFER_OUTPUT_ROOT = "./finetuned_infer/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2

NONLINEAR_NAMES = ["Softmax", "LayerNorm1", "GeLU", "LayerNorm2"]

# GLUE 测试集标签均为 -1（未公开），设为 False 则只评估验证集
EVALUATE_TEST_SET = False
# ======================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device}")
os.makedirs(INFER_OUTPUT_ROOT, exist_ok=True)


def _load_layernorm_utils():
    """加载 nolinear/layernorm.py 中的方差统计工具（与 variance 评估一致）。"""
    ln_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nolinear", "layernorm.py")
    spec = importlib.util.spec_from_file_location("layernorm_eval", ln_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {ln_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LN = _load_layernorm_utils()


# ===================== 钩子工具函数 =====================

def create_empty_stats():
    """创建 48 个条目的空统计字典"""
    stats = {}
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            key = f"Layer{layer_idx}_{name}"
            stats[key] = {"min": float("inf"), "max": float("-inf")}
    return stats


def update_stats(stats, key, tensor):
    """更新某个 key 的 min/max 统计"""
    val_min = tensor.detach().min().item()
    val_max = tensor.detach().max().item()
    stats[key]["min"] = min(stats[key]["min"], val_min)
    stats[key]["max"] = max(stats[key]["max"], val_max)


def make_input_hook(stats, key):
    """为 GeLU 等 Module 创建 forward hook（统计输入元素 min/max）。"""
    def hook_fn(module, inp, out):
        update_stats(stats, key, inp[0])
    return hook_fn


def register_hooks(model, stats):
    """
    对 BERT 12 层注册钩子，统计 Softmax / GeLU 的输入元素 min/max。
    LayerNorm 方差统计见 register_layernorm_variance_hooks。
    """
    hooks = []

    for i in range(NUM_LAYERS):
        layer = model.bert.encoder.layer[i]

        # -------- 1. Softmax 输入（attention_scores）--------
        softmax_key = f"Layer{i}_Softmax"
        attn_self = layer.attention.self
        _orig_forward = attn_self.forward

        def make_patched_forward(orig_fn, stats_dict, skey, attn_module):
            """
            替换 BertSelfAttention.forward，在 softmax 调用前截取 attention_scores。
            兼容 eager / sdpa 两种 attention 实现。
            """
            @functools.wraps(orig_fn)
            def patched_forward(
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                **kwargs
            ):
                # --- 手动计算 attention_scores（与 BertSelfAttention 逻辑一致）---
                mixed_query = attn_module.query(hidden_states)
                key_layer_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                mixed_key = attn_module.key(key_layer_input)

                batch_size = mixed_query.size(0)
                head_dim = attn_module.attention_head_size
                num_heads = attn_module.num_attention_heads

                def transpose_for_scores(x):
                    new_shape = x.size()[:-1] + (num_heads, head_dim)
                    x = x.view(new_shape)
                    return x.permute(0, 2, 1, 3)

                query_layer = transpose_for_scores(mixed_query)
                key_layer = transpose_for_scores(mixed_key)

                # attention_scores: [batch, heads, seq_q, seq_k]
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / (head_dim ** 0.5)

                # 统计 softmax 输入（加 mask 之前是纯 attention score）
                update_stats(stats_dict, skey, attention_scores)

                # --- 调用原始 forward 获得正确输出 ---
                return orig_fn(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    **kwargs
                )
            return patched_forward

        attn_self.forward = make_patched_forward(
            _orig_forward, stats, softmax_key, attn_self
        )

        # -------- 2. GeLU（intermediate.intermediate_act_fn）--------
        gelu_key = f"Layer{i}_GeLU"
        act_fn = layer.intermediate.intermediate_act_fn
        if isinstance(act_fn, torch.nn.Module):
            h = act_fn.register_forward_hook(make_input_hook(stats, gelu_key))
            hooks.append(h)
        else:
            intermediate_module = layer.intermediate
            _orig_inter_forward = intermediate_module.forward

            def make_patched_inter(orig_fn, stats_dict, gkey):
                @functools.wraps(orig_fn)
                def patched(*args, **kwargs):
                    original_gelu = F.gelu
                    def tracking_gelu(input, approximate='none'):
                        update_stats(stats_dict, gkey, input)
                        return original_gelu(input, approximate=approximate)
                    F.gelu = tracking_gelu
                    try:
                        result = orig_fn(*args, **kwargs)
                    finally:
                        F.gelu = original_gelu
                    return result
                return patched

            intermediate_module.forward = make_patched_inter(
                _orig_inter_forward, stats, gelu_key
            )

    return hooks


def register_layernorm_variance_hooks(model):
    """LayerNorm 输入方差收集（有效 token，与 layernorm.py 一致）。"""
    collector = _LN.GpuLayerNormCollector([True] * NUM_LAYERS, device)
    ln_hooks = _LN.register_layernorm_hooks(model, collector)
    return collector, ln_hooks


def finalize_layernorm_variance_stats(collector, stats) -> None:
    """将 collector 中各层 ln1/ln2 输入的真实 Var(x) min/max 写入 stats。"""
    kind_to_name = {"ln1": "LayerNorm1", "ln2": "LayerNorm2"}
    for layer_idx in range(NUM_LAYERS):
        for kind, name in kind_to_name.items():
            key = f"Layer{layer_idx}_{name}"
            inputs = collector.stacked(layer_idx, kind)
            if inputs.shape[0] == 0:
                stats[key]["min"] = float("nan")
                stats[key]["max"] = float("nan")
                continue
            st = _LN.variance_stats_gpu(inputs)
            stats[key]["min"] = st["var_min"]
            stats[key]["max"] = st["var_max"]
            collector.release(layer_idx, kind)


@torch.no_grad()
def evaluate_split(model, tokenized_split, data_collator):
    """在指定 split 上前向，返回 accuracy；期间 LayerNorm collector 持续收集。"""
    loader = DataLoader(
        tokenized_split,
        batch_size=8,
        shuffle=False,
        collate_fn=data_collator,
    )
    pred_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for batch in loader:
        labels = batch["labels"]
        batch = {k: v.to(device) for k, v in batch.items()}
        attn_mask = batch.get("attention_mask")
        if attn_mask is not None:
            _LN.attach_attention_mask_to_layernorms(model, attn_mask)
        outputs = model(**batch)
        _LN.clear_attention_mask_on_layernorms(model)
        pred_chunks.append(outputs.logits.argmax(dim=-1).cpu())
        label_chunks.append(labels)

    preds = torch.cat(pred_chunks, dim=0).numpy()
    labels_np = torch.cat(label_chunks, dim=0).numpy()
    return float(accuracy_score(labels_np, preds))


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ===================== 单任务推理 =====================

def get_preprocess_fn(task_name, tokenizer):
    """根据任务类型返回预处理函数"""
    if task_name == "sst2":
        def fn(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True, max_length=MAX_SEQ_LENGTH
            )
    else:  # mrpc, rte（双句任务）
        def fn(examples):
            return tokenizer(
                examples["sentence1"], examples["sentence2"],
                truncation=True, max_length=MAX_SEQ_LENGTH
            )
    return fn


def infer_single_task(task_name):
    """对单个数据集推理，返回准确率 + 非线性统计"""
    finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)

    if not os.path.exists(finetuned_model_path):
        print(f"⚠️  模型权重不存在，跳过 {task_name}：{finetuned_model_path}")
        return None, None

    print(f"\n{'='*60}")
    print(f"🔄 推理任务：{task_name.upper()}")
    print(f"{'='*60}")

    # 1. 加载模型（强制 eager attention，确保 softmax 可观测）
    print(f"✅ 加载模型：{finetuned_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        finetuned_model_path,
        num_labels=NUM_LABELS,
        attn_implementation="eager"  # 关键：禁用 SDPA
    )
    model = model.to(device)
    model.eval()

    # 2. 注册钩子（Softmax/GeLU 元素范围 + LayerNorm 方差收集）
    stats = create_empty_stats()
    hooks = register_hooks(model, stats)
    ln_collector, ln_hooks = register_layernorm_variance_hooks(model)
    print(
        f"✅ 已注册 {NUM_LAYERS}层 × {len(NONLINEAR_NAMES)}函数；"
        f"LayerNorm 按 Var(x) 区间统计（与 layernorm.py 一致）"
    )

    # 3. 加载并预处理数据集
    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    print(f"✅ 加载数据集：{data_path}")
    dataset = load_from_disk(data_path)

    preprocess_fn = get_preprocess_fn(task_name, tokenizer)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

    def tokenize_split(split):
        drop_cols = [c for c in split.column_names if c != "label"]
        return split.map(
            preprocess_fn,
            batched=True,
            remove_columns=drop_cols,
        )

    tokenized_validation = tokenize_split(dataset["validation"])

    # 4. 验证集评估（同时收集 LayerNorm 输入）
    print(f"📊 验证集评估中...")
    val_acc = evaluate_split(model, tokenized_validation, data_collator)
    print(f"✅ 验证集准确率：{val_acc:.4f}")

    # 5. 测试集评估（可选）
    test_acc = None
    if EVALUATE_TEST_SET and "test" in dataset:
        print(f"📊 测试集评估中...")
        try:
            tokenized_test = tokenize_split(dataset["test"])
            test_acc = evaluate_split(model, tokenized_test, data_collator)
            print(f"✅ 测试集准确率：{test_acc:.4f}")
        except Exception as e:
            print(f"⚠️  测试集评估失败：{e}")
    else:
        print(f"ℹ️  跳过测试集（GLUE 测试集标签未公开）")

    finalize_layernorm_variance_stats(ln_collector, stats)

    # 6. 统计预览
    print(f"\n📈 非线性统计预览（{task_name}；LN 为 Var(x) 真实 min/max）：")
    print(f"   {'函数':<25} {'min':<20} {'max':<20}")
    print(f"   {'-'*65}")
    for layer_idx in [0, 5, 11]:
        for name in NONLINEAR_NAMES:
            key = f"Layer{layer_idx}_{name}"
            s = stats[key]
            if name.startswith("LayerNorm"):
                min_str = (
                    f"{s['min']:.6g}"
                    if isinstance(s["min"], float) and not math.isnan(s["min"])
                    else "N/A"
                )
                max_str = (
                    f"{s['max']:.6g}"
                    if isinstance(s["max"], float) and not math.isnan(s["max"])
                    else "N/A"
                )
            else:
                min_str = f"{s['min']:.6f}" if s["min"] != float("inf") else "N/A"
                max_str = f"{s['max']:.6f}" if s["max"] != float("-inf") else "N/A"
            print(f"   {key:<25} {min_str:<20} {max_str:<20}")

    # 7. 清理钩子
    remove_hooks(hooks)
    _LN.restore_layernorm_hooks(ln_hooks)
    print(f"✅ 已移除所有钩子")

    # 8. 保存准确率
    output_file = os.path.join(INFER_OUTPUT_ROOT, f"{task_name}_res.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 微调模型推理结果 - {task_name.upper()}\n")
        f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# 模型路径：{finetuned_model_path}\n\n")
        f.write(f"验证集准确率 (val_accuracy):  {val_acc:.4f}\n")
        if test_acc is not None:
            f.write(f"测试集准确率 (test_accuracy): {test_acc:.4f}\n")
        else:
            f.write(f"测试集准确率 (test_accuracy): N/A (GLUE测试集标签未公开)\n")
    print(f"💾 准确率结果已保存：{output_file}")

    # 9. 单任务 CSV
    csv_path = os.path.join(INFER_OUTPUT_ROOT, f"{task_name}_nonlinear_range.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "NonLinear", "min", "max", "stat_kind"])
        for layer_idx in range(NUM_LAYERS):
            for name in NONLINEAR_NAMES:
                key = f"Layer{layer_idx}_{name}"
                kind = "variance" if name.startswith("LayerNorm") else "element"
                writer.writerow([
                    layer_idx, name,
                    stats[key]["min"], stats[key]["max"],
                    kind,
                ])
    print(f"💾 单任务统计已保存：{csv_path}")

    return {
        "task": task_name,
        "val_acc": val_acc,
        "test_acc": test_acc
    }, stats


# ===================== 合并 CSV 生成 =====================

def save_merged_csv(all_stats, csv_path):
    """
    保存合并表格：48 行 × (3 + 3×2) 列
    LayerNorm 行 min/max 为 Var(x) 真实 min/max；其余为元素 min/max。
    """
    header = ["Layer", "NonLinear", "stat_kind"]
    for task_name in TASK_NAMES:
        header.extend([f"{task_name}_min", f"{task_name}_max"])

    # ---- 构建数据行 ----
    rows = []
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            key = f"Layer{layer_idx}_{name}"
            stat_kind = "variance" if name.startswith("LayerNorm") else "element"
            row = [layer_idx, name, stat_kind]
            for task_name in TASK_NAMES:
                if task_name in all_stats and key in all_stats[task_name]:
                    s = all_stats[task_name][key]
                    min_v = s["min"] if s["min"] != float("inf") else "N/A"
                    max_v = s["max"] if s["max"] != float("-inf") else "N/A"
                else:
                    min_v, max_v = "N/A", "N/A"
                row.extend([min_v, max_v])
            rows.append(row)

    # ---- 写入文件 ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n💾 合并统计表已保存：{csv_path}")
    print(f"   {len(rows)} 行（{NUM_LAYERS}层 × {len(NONLINEAR_NAMES)}函数）")
    print(f"   {len(header)} 列（Layer + NonLinear + {len(TASK_NAMES)}数据集 × min/max）")


# ===================== 主循环 =====================

print(f"\n🚀 开始推理 {len(TASK_NAMES)} 个数据集：{TASK_NAMES}")
print(f"   时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = []
all_stats = {}

for task_name in TASK_NAMES:
    result, stats = infer_single_task(task_name)
    if result is not None:
        results.append(result)
        all_stats[task_name] = stats

# ===================== 保存合并 CSV =====================

merged_csv_path = os.path.join(INFER_OUTPUT_ROOT, "nonlinear_input_range_merged.csv")
save_merged_csv(all_stats, merged_csv_path)

# ===================== 汇总报告 =====================

print(f"\n{'='*60}")
print(f"🎯 所有任务推理完成！")
print(f"{'='*60}")

print(f"\n{'任务':<10} {'验证集准确率':<15} {'测试集准确率':<15}")
print("-" * 45)
for r in results:
    test_str = f"{r['test_acc']:.4f}" if r['test_acc'] is not None else "N/A"
    print(f"{r['task']:<10} {r['val_acc']:.4f}{'':<10} {test_str}")

# 保存汇总报告
summary_file = os.path.join(INFER_OUTPUT_ROOT, "summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"# 微调模型推理汇总报告\n")
    f.write(f"# 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"{'任务':<10} {'验证集准确率':<15} {'测试集准确率':<15}\n")
    f.write("-" * 45 + "\n")
    for r in results:
        test_str = f"{r['test_acc']:.4f}" if r['test_acc'] is not None else "N/A"
        f.write(f"{r['task']:<10} {r['val_acc']:.4f}{'':<10} {test_str}\n")
    f.write(f"\n# 非线性函数输入范围统计见：nonlinear_input_range_merged.csv\n")

print(f"\n💾 汇总报告已保存：{summary_file}")
print(f"💾 合并统计表：{merged_csv_path}")
print(f"📁 所有结果保存在：{INFER_OUTPUT_ROOT}")
print(f"\n🎉 全部完成！")