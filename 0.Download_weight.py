import os
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ===================== 配置参数 =====================
# 模型名称（HuggingFace官方bert-base-uncased）
MODEL_NAME = "google-bert/bert-base-uncased"
# 本地保存路径（当前目录下的bert_weight文件夹）
SAVE_PATH = "./bert_weight/"
# ====================================================

# 1. 创建本地保存目录（如果不存在则自动创建）
os.makedirs(SAVE_PATH, exist_ok=True)

# 2. 从HuggingFace下载 分词器 + 模型
print("正在从HuggingFace下载 bert-base-uncased 模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

# 3. 保存到本地指定路径
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"下载完成！模型和分词器已保存到：{SAVE_PATH}")