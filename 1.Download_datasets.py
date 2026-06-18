import os
from datasets import load_dataset

# ===================== 配置 =====================
# 要下载的3个GLUE数据集（BERT微调最常用）
DATASET_NAMES = ["mrpc", "rte", "sst2"]
# 根保存目录（所有数据集存在这里）
ROOT_SAVE_DIR = "./glue_datasets/"
# ================================================

# 循环下载+保存每个数据集
for task in DATASET_NAMES:
    print(f"\n========================================")
    print(f"正在下载并保存 GLUE-{task.upper()} 数据集...")
    
    # 1. 创建当前数据集的专属目录
    save_path = os.path.join(ROOT_SAVE_DIR, task)
    os.makedirs(save_path, exist_ok=True)

    # 2. 从HuggingFace加载数据集（自动下载 train/validation/test）
    # GLUE 格式：load_dataset("glue", 任务名)
    dataset = load_dataset("glue", task)

    # 3. 保存到本地（完整保存训练/验证/测试集，可直接加载）
    dataset.save_to_disk(save_path)

    # 打印数据集信息
    print(f"✅ {task.upper()} 数据集保存完成！路径：{save_path}")
    print(f"   包含：训练集({len(dataset['train'])}条) | 验证集({len(dataset['validation'])}条) | 测试集({len(dataset['test'])}条)")

print(f"\n🎉 所有数据集下载完成！统一存放于：{ROOT_SAVE_DIR}")