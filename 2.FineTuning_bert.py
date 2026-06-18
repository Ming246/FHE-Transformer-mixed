"""
BERT 微调 - MRPC / RTE / SST2
统一微调脚本，通过 TASK_NAME 切换任务
"""
import os
import sys
import torch
import evaluate
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)

# ===================== 配置区 =====================
# 修改这里切换任务，或通过命令行传入：python finetune.py mrpc
TASK_NAME = sys.argv[1] if len(sys.argv) > 1 else "mrpc"
assert TASK_NAME in ["mrpc", "rte", "sst2"], f"不支持的任务：{TASK_NAME}"

LOCAL_MODEL_PATH = "./bert_weight/"
LOCAL_DATA_ROOT = "./glue_datasets/"
SAVE_FINETUNED_PATH = f"./finetuned_weight/{TASK_NAME}/"

# 训练参数
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01          # 【修复】添加权重衰减
WARMUP_RATIO = 0.06          # 【修复】添加学习率预热
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LOG_STEP = 10
NUM_LABELS = 2

# 任务描述映射
TASK_DESCRIPTIONS = {
    "mrpc": "句子对语义相似度判断",
    "rte":  "句子对蕴含判断",
    "sst2": "情感分析（单句）",
}
# ============================================

metric = evaluate.load("accuracy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device}")
print(f"✅ 当前任务：{TASK_NAME.upper()} - {TASK_DESCRIPTIONS[TASK_NAME]}")

# 1. 加载模型和分词器
print(f"\n✅ 加载本地预训练BERT模型：{LOCAL_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    LOCAL_MODEL_PATH, num_labels=NUM_LABELS
)

# 2. 加载数据集
data_path = os.path.join(LOCAL_DATA_ROOT, TASK_NAME)
print(f"✅ 加载本地数据集：{data_path}")
dataset = load_from_disk(data_path)

# 3. 数据预处理（根据任务类型自动切换）
def preprocess_function(examples):
    if TASK_NAME == "sst2":
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )
    else:  # mrpc, rte（双句任务）
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

tokenized_dataset = dataset.map(preprocess_function, batched=True)
# 【修复】不再手动 remove_columns，Trainer 会自动处理

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. 评估函数
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric.compute(predictions=predictions, references=labels)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir=SAVE_FINETUNED_PATH,
    num_train_epochs=TRAIN_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,         # 【修复】
    # warmup_ratio=WARMUP_RATIO,         # 【修复】
    warmup_steps=int(0.06 * len(tokenized_dataset["train"])),
    logging_steps=LOG_STEP,
    logging_first_step=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # overwrite_output_dir=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    disable_tqdm=False,
    seed=42,                           # 【新增】明确随机种子
)

# 6. 回调：打印训练进度
class PrintProgressCallback(TrainerCallback):
    def __init__(self):
        super().__init__()             # 【修复】调用父类 init
        self.is_training = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.is_training = True

    def on_train_end(self, args, state, control, **kwargs):
        self.is_training = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.is_training and logs and 'loss' in logs:
            loss = logs.get('loss', 'N/A')
            lr = logs.get('learning_rate', 'N/A')
            loss_str = f"{float(loss):.4f}" if isinstance(loss, (int, float)) else str(loss)
            lr_str = f"{float(lr):.2e}" if isinstance(lr, (int, float)) else str(lr)
            print(f"  Step {state.global_step} | loss: {loss_str} | lr: {lr_str}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.is_training:
            print(f"\n✅ Epoch {state.epoch} 完成！")

# 7. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    # tokenizer=tokenizer,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[PrintProgressCallback()],  # 【修复】传实例而非类
)

# 8. 评估初始模型
print("\n📊 评估【未微调】的初始模型性能...")
initial_results = trainer.evaluate()
initial_acc = initial_results['eval_accuracy']
print(f"初始验证集准确率：{initial_acc:.4f}")

# 9. 开始训练
print(f"\n🚀 开始微调 BERT on {TASK_NAME.upper()} 数据集...")
print(f"   有效批大小：{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   训练样本数：{len(tokenized_dataset['train'])}")
print(f"   验证样本数：{len(tokenized_dataset['validation'])}")
trainer.train()

# 10. 保存最优模型
trainer.save_model(SAVE_FINETUNED_PATH)
tokenizer.save_pretrained(SAVE_FINETUNED_PATH)  # 【新增】确保 tokenizer 也保存
print(f"\n🎉 微调完成！最优模型已保存至：{SAVE_FINETUNED_PATH}")

# 11. 最终验证集评估（使用 load_best_model_at_end 加载的最优模型）
print(f"\n📊 在 {TASK_NAME.upper()} 验证集上评估最终最优模型...")
val_results = trainer.evaluate()
val_acc = val_results['eval_accuracy']
val_loss = val_results['eval_loss']
best_val_acc = trainer.state.best_metric
print(f"✅ 最终验证集准确率：{val_acc:.4f}")
print(f"✅ 最终验证集损失：  {val_loss:.4f}")

# 12. 测试集说明
# 【修复】GLUE 三个数据集的 test set 标签全部是 -1，不做评估
print(f"\nℹ️  跳过测试集评估（GLUE 测试集标签均为 -1，未公开）")
print(f"   如需提交测试集预测，请使用 GLUE Benchmark 官网提交。")

# 13. 计算提升幅度
acc_improvement = val_acc - initial_acc
acc_improvement_pct = (acc_improvement / max(initial_acc, 1e-8)) * 100

print(f"\n{'='*60}")
print(f"📈 微调效果对比 - {TASK_NAME.upper()}")
print(f"{'='*60}")
print(f"   微调前验证集准确率：{initial_acc:.4f}")
print(f"   微调后验证集准确率：{val_acc:.4f}")
print(f"   绝对提升：          +{acc_improvement:.4f}")
print(f"   相对提升：          +{acc_improvement_pct:.2f}%")
print(f"{'='*60}")

# 14. 生成详细 README 报告（包含微调前后对比）
readme_content = f"""# BERT 微调结果报告 - {TASK_NAME.upper()} 数据集

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、基础信息

| 项目 | 值 |
|------|-----|
| 基础模型 | bert-base-uncased |
| 微调任务 | {TASK_NAME.upper()} - {TASK_DESCRIPTIONS[TASK_NAME]} |
| 分类类别 | {NUM_LABELS} 分类 |
| 硬件 | RTX 4060 8G 笔记本显卡 |
| 混合精度 | FP16 |

---

## 二、训练超参数

| 参数 | 值 |
|------|-----|
| 训练轮数 (Epochs) | {TRAIN_EPOCHS} |
| 学习率 (Learning Rate) | {LEARNING_RATE} |
| 权重衰减 (Weight Decay) | {WEIGHT_DECAY} |
| 预热比例 (Warmup Ratio) | {WARMUP_RATIO} |
| 最大序列长度 | {MAX_SEQ_LENGTH} |
| 单卡批大小 | {BATCH_SIZE} |
| 梯度累积步数 | {GRADIENT_ACCUMULATION_STEPS} |
| 有效批大小 | {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} |
| 随机种子 | 42 |

---

## 三、数据集统计

| 数据集划分 | 样本数 |
|-----------|--------|
| 训练集 (train) | {len(tokenized_dataset['train'])} |
| 验证集 (validation) | {len(tokenized_dataset['validation'])} |
| 测试集 (test) | {len(tokenized_dataset['test']) if 'test' in tokenized_dataset else 'N/A'} |

---

## 四、⭐ 微调前后性能对比（核心结果）

| 指标 | 微调前 | 微调后 | 提升 |
|------|--------|--------|------|
| 验证集准确率 | {initial_acc:.4f} | {val_acc:.4f} | **+{acc_improvement:.4f}** (+{acc_improvement_pct:.2f}%) |
| 验证集损失 | - | {val_loss:.4f} | - |
| 训练最优轮次 | - | Best (自动选择) | - |

### 关键结论

- 微调前准确率：**{initial_acc:.4f}**（预训练 BERT 直接推理，分类头随机初始化）
- 微调后准确率：**{val_acc:.4f}**
- **绝对提升：+{acc_improvement:.4f}**
- **相对提升：+{acc_improvement_pct:.2f}%**

---

## 五、测试集说明

⚠️ GLUE Benchmark 的测试集标签**均未公开**（标签值为 -1）。
如需获取测试集分数，请将预测结果提交至：
👉 https://gluebenchmark.com/submit

---

## 六、训练策略说明

- 使用 `load_best_model_at_end=True`，自动保存验证集上最优的模型
- 评估指标：`accuracy`
- 每个 epoch 结束后进行一次验证集评估
- FP16 混合精度训练，减少显存占用
- 梯度累积 {GRADIENT_ACCUMULATION_STEPS} 步，等效增大批大小

---

## 七、文件清单

```
{SAVE_FINETUNED_PATH}
├── config.json              # 模型配置
├── model.safetensors        # 模型权重
├── tokenizer.json           # 分词器
├── tokenizer_config.json    # 分词器配置
├── special_tokens_map.json  # 特殊 token 映射
├── vocab.txt                # 词表
└── README.md                # 本报告
```
"""

readme_path = os.path.join(SAVE_FINETUNED_PATH, "README.md")
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print(f"\n📄 详细训练报告已生成：{readme_path}")

# 15. 额外保存一份纯文本摘要（方便脚本解析）
summary_path = os.path.join(SAVE_FINETUNED_PATH, "result_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"task={TASK_NAME}\n")
    f.write(f"initial_val_acc={initial_acc:.6f}\n")
    f.write(f"final_val_acc={val_acc:.6f}\n")
    f.write(f"final_val_loss={val_loss:.6f}\n")
    f.write(f"acc_improvement={acc_improvement:.6f}\n")
    f.write(f"acc_improvement_pct={acc_improvement_pct:.2f}\n")
    f.write(f"best_val_acc={best_val_acc:.6f}\n")
    f.write(f"epochs={TRAIN_EPOCHS}\n")
    f.write(f"learning_rate={LEARNING_RATE}\n")
    f.write(f"weight_decay={WEIGHT_DECAY}\n")
    f.write(f"warmup_ratio={WARMUP_RATIO}\n")
    f.write(f"batch_size={BATCH_SIZE}\n")
    f.write(f"grad_accum={GRADIENT_ACCUMULATION_STEPS}\n")
    f.write(f"timestamp={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print(f"📄 结果摘要已保存：{summary_path}")

print(f"\n{'='*60}")
print(f"🎉 {TASK_NAME.upper()} 微调全部完成！")
print(f"   模型权重：{SAVE_FINETUNED_PATH}")
print(f"   验证准确率：{initial_acc:.4f} → {val_acc:.4f} (+{acc_improvement:.4f})")
print(f"{'='*60}")
