"""下载 GLUE 数据集到 ./glue_datasets/{task}/。

MNLI：HuggingFace Hub 不可达时，回退到 Facebook GLUE 官方 MNLI.zip，
并将 validation_matched 别名为 validation（列名对齐 HF：premise/hypothesis）。
"""
from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# ===================== 配置 =====================
# Phase1：仅下载 MNLI
DATASET_NAMES = ["mnli"]
ROOT_SAVE_DIR = "./glue_datasets/"
MNLI_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip"
# ================================================

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def split_size_line(dataset: DatasetDict) -> str:
    return " | ".join(f"{name}({len(split)}条)" for name, split in dataset.items())


def _read_mnli_tsv(zf: zipfile.ZipFile, member: str, *, has_label: bool) -> Dataset:
    """解析 GLUE MNLI TSV（sentence1/sentence2）→ HF 列名 premise/hypothesis。"""
    # parse 树字段可能超过默认 128KB 限制
    csv.field_size_limit(min(sys.maxsize, 16 * 1024 * 1024))
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        premises, hypotheses, labels, idxs = [], [], [], []
        for i, row in enumerate(reader):
            # 个别坏行缺字段时跳过
            s1 = row.get("sentence1")
            s2 = row.get("sentence2")
            if s1 is None or s2 is None:
                continue
            if has_label:
                lab = (row.get("gold_label") or "").strip()
                if lab not in LABEL2ID:
                    continue  # 跳过 "-" 等无效标签
                labels.append(LABEL2ID[lab])
            else:
                labels.append(-1)
            premises.append(s1)
            hypotheses.append(s2)
            idxs.append(i)
    cols = {
        "premise": premises,
        "hypothesis": hypotheses,
        "label": labels,
        "idx": idxs,
    }
    return Dataset.from_dict(cols)


def load_mnli_from_fb_zip(zip_path: str | None = None) -> DatasetDict:
    """从 Facebook GLUE MNLI.zip 构建 DatasetDict。"""
    if zip_path is None:
        cache = Path(tempfile.gettempdir()) / "MNLI.zip"
        if not cache.exists() or cache.stat().st_size < 100_000_000:
            print(f"  从 Facebook 下载 MNLI.zip → {cache} ...")
            urllib.request.urlretrieve(MNLI_ZIP_URL, cache)
        zip_path = str(cache)
        print(f"  使用本地 zip：{zip_path} ({Path(zip_path).stat().st_size} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".tsv")}
        needed = {
            "train": ("train.tsv", True),
            "validation": ("dev_matched.tsv", True),
            "validation_mismatched": ("dev_mismatched.tsv", True),
            "test": ("test_matched.tsv", False),
            "test_mismatched": ("test_mismatched.tsv", False),
        }
        out = {}
        for split, (fname, has_label) in needed.items():
            if fname not in members:
                raise FileNotFoundError(f"MNLI.zip 缺少 {fname}；有：{sorted(members)}")
            print(f"  解析 {fname} → {split} ...")
            out[split] = _read_mnli_tsv(zf, members[fname], has_label=has_label)
    return DatasetDict(out)


def prepare_glue_dataset(task: str) -> DatasetDict:
    if task != "mnli":
        return load_dataset("glue", task)

    # 优先 HF；失败则用 FB zip（当前环境 Hub DNS 常失败）
    try:
        raw = load_dataset("glue", "mnli")
        return DatasetDict(
            {
                "train": raw["train"],
                "validation": raw["validation_matched"],
                "validation_mismatched": raw["validation_mismatched"],
                "test": raw["test_matched"],
                "test_mismatched": raw["test_mismatched"],
            }
        )
    except Exception as exc:
        print(f"  HuggingFace load_dataset 失败（{type(exc).__name__}），回退 Facebook MNLI.zip")
        return load_mnli_from_fb_zip()


for task in DATASET_NAMES:
    print(f"\n========================================")
    print(f"正在下载并保存 GLUE-{task.upper()} 数据集...")

    save_path = os.path.join(ROOT_SAVE_DIR, task)
    os.makedirs(save_path, exist_ok=True)

    dataset = prepare_glue_dataset(task)
    dataset.save_to_disk(save_path)

    print(f"✅ {task.upper()} 数据集保存完成！路径：{save_path}")
    print(f"   包含：{split_size_line(dataset)}")
    if task == "mnli":
        print(f"   validation(=matched) 列：{dataset['validation'].column_names}")
        print(f"   label 示例：{dataset['validation'][0]['label']} / {dataset['train'].features}")

print(f"\n🎉 所有数据集下载完成！统一存放于：{ROOT_SAVE_DIR}")
