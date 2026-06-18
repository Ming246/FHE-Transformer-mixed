"""
微调模型推理 - MRPC / RTE / SST2
使用钩子统计12层×4个非线性函数的输入范围
输出：每个任务的准确率 + 合并的 48×6 CSV
"""
import os
import csv
import torch
import evaluate
import functools
import torch.nn.functional as F
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
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

metric = evaluate.load("accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device}")
os.makedirs(INFER_OUTPUT_ROOT, exist_ok=True)


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
    """为 LayerNorm 等标准 Module 创建 forward hook"""
    def hook_fn(module, inp, out):
        update_stats(stats, key, inp[0])
    return hook_fn


def register_hooks(model, stats):
    """
    对 BERT 12 层注册钩子，统计 4 个非线性函数的输入范围。

    针对 Softmax：
    - 新版 transformers 可能使用 SDPA（F.softmax 不会被调用）
    - 解决方案：直接 hook BertSelfAttention，在 forward 中
      拦截 attention_scores（softmax 的输入）
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

        # -------- 2. LayerNorm1（attention.output.LayerNorm）--------
        ln1_key = f"Layer{i}_LayerNorm1"
        h = layer.attention.output.LayerNorm.register_forward_hook(
            make_input_hook(stats, ln1_key)
        )
        hooks.append(h)

        # -------- 3. GeLU（intermediate.intermediate_act_fn）--------
        gelu_key = f"Layer{i}_GeLU"
        act_fn = layer.intermediate.intermediate_act_fn
        if isinstance(act_fn, torch.nn.Module):
            # 新版 transformers: GELUActivation 是 nn.Module
            h = act_fn.register_forward_hook(make_input_hook(stats, gelu_key))
            hooks.append(h)
        else:
            # 旧版: 函数式 gelu，用猴子补丁
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

        # -------- 4. LayerNorm2（output.LayerNorm）--------
        ln2_key = f"Layer{i}_LayerNorm2"
        h = layer.output.LayerNorm.register_forward_hook(
            make_input_hook(stats, ln2_key)
        )
        hooks.append(h)

    return hooks


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

    # 2. 注册钩子
    stats = create_empty_stats()
    hooks = register_hooks(model, stats)
    print(f"✅ 已注册 {NUM_LAYERS}层 × {len(NONLINEAR_NAMES)}函数 = "
          f"{NUM_LAYERS * len(NONLINEAR_NAMES)} 个监控点")

    # 3. 加载并预处理数据集
    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    print(f"✅ 加载数据集：{data_path}")
    dataset = load_from_disk(data_path)

    preprocess_fn = get_preprocess_fn(task_name, tokenizer)
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. 评估函数
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. 验证集评估
    print(f"📊 验证集评估中...")
    val_results = trainer.evaluate(tokenized_dataset["validation"])
    val_acc = val_results['eval_accuracy']
    print(f"✅ 验证集准确率：{val_acc:.4f}")

    # 7. 测试集评估（可选）
    test_acc = None
    if EVALUATE_TEST_SET and "test" in tokenized_dataset:
        print(f"📊 测试集评估中...")
        try:
            test_results = trainer.evaluate(tokenized_dataset["test"])
            test_acc = test_results.get('eval_accuracy', None)
            if test_acc is not None:
                print(f"✅ 测试集准确率：{test_acc:.4f}")
        except Exception as e:
            print(f"⚠️  测试集评估失败：{e}")
    else:
        print(f"ℹ️  跳过测试集（GLUE 测试集标签未公开）")

    # 8. 统计预览
    print(f"\n📈 非线性函数输入范围预览（{task_name}）：")
    print(f"   {'函数':<25} {'最小值':<20} {'最大值':<20}")
    print(f"   {'-'*65}")
    for layer_idx in [0, 5, 11]:
        for name in NONLINEAR_NAMES:
            key = f"Layer{layer_idx}_{name}"
            s = stats[key]
            min_str = f"{s['min']:.6f}" if s['min'] != float('inf') else "N/A"
            max_str = f"{s['max']:.6f}" if s['max'] != float('-inf') else "N/A"
            print(f"   {key:<25} {min_str:<20} {max_str:<20}")

    # 9. 清理钩子
    remove_hooks(hooks)
    print(f"✅ 已移除所有钩子")

    # 10. 保存准确率
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

    # 11. 单任务 CSV（可选保留）
    csv_path = os.path.join(INFER_OUTPUT_ROOT, f"{task_name}_nonlinear_range.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Layer", "NonLinear", "min", "max"])
        for layer_idx in range(NUM_LAYERS):
            for name in NONLINEAR_NAMES:
                key = f"Layer{layer_idx}_{name}"
                writer.writerow([
                    layer_idx, name,
                    stats[key]["min"], stats[key]["max"]
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
    保存合并表格：48 行 × (2 + 3×2) 列
    行：12层 × 4函数
    列：Layer, NonLinear, mrpc_min, mrpc_max, rte_min, rte_max, sst2_min, sst2_max
    """
    # ---- 构建表头 ----
    header = ["Layer", "NonLinear"]
    for task_name in TASK_NAMES:
        header.extend([f"{task_name}_min", f"{task_name}_max"])

    # ---- 构建数据行 ----
    rows = []
    for layer_idx in range(NUM_LAYERS):
        for name in NONLINEAR_NAMES:
            key = f"Layer{layer_idx}_{name}"
            row = [layer_idx, name]
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