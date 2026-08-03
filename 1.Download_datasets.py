"""下载 GLUE 数据集到 ./glue_datasets/{task}/。

支持：cola / qnli / mnli（及其他 glue 子任务名）。
HuggingFace Hub 不可达时：
  - MNLI → Facebook MNLI.zip（validation_matched 别名为 validation）
  - CoLA → Facebook CoLA.zip
  - QNLI → Facebook QNLIv2.zip
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

# 部分环境 HTTP(S)_PROXY 会导致 FB/HF 下载 403，下载前清除
for _k in list(os.environ):
    if "proxy" in _k.lower():
        del os.environ[_k]

# ===================== 配置 =====================
DATASET_NAMES = ["cola", "qnli"]
ROOT_SAVE_DIR = "./glue_datasets/"
MNLI_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip"
COLA_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip"
QNLI_ZIP_URL = "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip"
# ================================================

MNLI_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
QNLI_LABEL2ID = {"entailment": 0, "not_entailment": 1}


def split_size_line(dataset: DatasetDict) -> str:
    return " | ".join(f"{name}({len(split)}条)" for name, split in dataset.items())


def _read_mnli_tsv(zf: zipfile.ZipFile, member: str, *, has_label: bool) -> Dataset:
    """解析 GLUE MNLI TSV（sentence1/sentence2）→ HF 列名 premise/hypothesis。"""
    csv.field_size_limit(min(sys.maxsize, 16 * 1024 * 1024))
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        premises, hypotheses, labels, idxs = [], [], [], []
        for i, row in enumerate(reader):
            s1 = row.get("sentence1")
            s2 = row.get("sentence2")
            if s1 is None or s2 is None:
                continue
            if has_label:
                lab = (row.get("gold_label") or "").strip()
                if lab not in MNLI_LABEL2ID:
                    continue
                labels.append(MNLI_LABEL2ID[lab])
            else:
                labels.append(-1)
            premises.append(s1)
            hypotheses.append(s2)
            idxs.append(i)
    return Dataset.from_dict(
        {
            "premise": premises,
            "hypothesis": hypotheses,
            "label": labels,
            "idx": idxs,
        }
    )


def load_mnli_from_fb_zip(zip_path: str | None = None) -> DatasetDict:
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


def _read_mnli_tsv(zf: zipfile.ZipFile, member: str, *, has_label: bool) -> Dataset:
    csv.field_size_limit(min(sys.maxsize, 16 * 1024 * 1024))
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        premises, hypotheses, labels, idxs = [], [], [], []
        for i, row in enumerate(reader):
            s1 = row.get("sentence1")
            s2 = row.get("sentence2")
            if s1 is None or s2 is None:
                continue
            if has_label:
                lab = (row.get("gold_label") or "").strip()
                if lab not in MNLI_LABEL2ID:
                    continue
                labels.append(MNLI_LABEL2ID[lab])
            else:
                labels.append(-1)
            premises.append(s1)
            hypotheses.append(s2)
            idxs.append(i)
    return Dataset.from_dict(
        {
            "premise": premises,
            "hypothesis": hypotheses,
            "label": labels,
            "idx": idxs,
        }
    )


def _read_cola_tsv(zf: zipfile.ZipFile, member: str, *, has_label: bool) -> Dataset:
    """CoLA TSV：无表头，列 = source / label / author / sentence。"""
    csv.field_size_limit(min(sys.maxsize, 16 * 1024 * 1024))
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.reader(text, delimiter="\t")
        sentences, labels, idxs = [], [], []
        for i, row in enumerate(reader):
            if has_label:
                if len(row) < 4:
                    continue
                try:
                    lab = int(row[1])
                except ValueError:
                    continue
                sentences.append(row[3])
                labels.append(lab)
            else:
                # test.tsv: id \t sentence
                if len(row) < 2:
                    continue
                sentences.append(row[-1])
                labels.append(-1)
            idxs.append(i)
    return Dataset.from_dict({"sentence": sentences, "label": labels, "idx": idxs})


def load_cola_from_fb_zip(zip_path: str | None = None) -> DatasetDict:
    if zip_path is None:
        cache = Path(tempfile.gettempdir()) / "CoLA.zip"
        if not cache.exists() or cache.stat().st_size < 100_000:
            print(f"  从 Facebook 下载 CoLA.zip → {cache} ...")
            urllib.request.urlretrieve(COLA_ZIP_URL, cache)
        zip_path = str(cache)
        print(f"  使用本地 zip：{zip_path} ({Path(zip_path).stat().st_size} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".tsv")}
        needed = {
            "train": ("train.tsv", True),
            "validation": ("dev.tsv", True),
            "test": ("test.tsv", False),
        }
        out = {}
        for split, (fname, has_label) in needed.items():
            if fname not in members:
                raise FileNotFoundError(f"CoLA.zip 缺少 {fname}；有：{sorted(members)}")
            print(f"  解析 {fname} → {split} ...")
            out[split] = _read_cola_tsv(zf, members[fname], has_label=has_label)
    return DatasetDict(out)


def _read_qnli_tsv(zf: zipfile.ZipFile, member: str, *, has_label: bool) -> Dataset:
    csv.field_size_limit(min(sys.maxsize, 16 * 1024 * 1024))
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8")
        reader = csv.DictReader(text, delimiter="\t")
        questions, sentences, labels, idxs = [], [], [], []
        for i, row in enumerate(reader):
            q = row.get("question")
            s = row.get("sentence")
            if q is None or s is None:
                continue
            if has_label:
                lab = (row.get("label") or "").strip()
                if lab not in QNLI_LABEL2ID:
                    continue
                labels.append(QNLI_LABEL2ID[lab])
            else:
                labels.append(-1)
            questions.append(q)
            sentences.append(s)
            idxs.append(i)
    return Dataset.from_dict(
        {
            "question": questions,
            "sentence": sentences,
            "label": labels,
            "idx": idxs,
        }
    )


def load_qnli_from_fb_zip(zip_path: str | None = None) -> DatasetDict:
    if zip_path is None:
        cache = Path(tempfile.gettempdir()) / "QNLIv2.zip"
        if not cache.exists() or cache.stat().st_size < 1_000_000:
            print(f"  从 Facebook 下载 QNLIv2.zip → {cache} ...")
            urllib.request.urlretrieve(QNLI_ZIP_URL, cache)
        zip_path = str(cache)
        print(f"  使用本地 zip：{zip_path} ({Path(zip_path).stat().st_size} bytes)")

    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist() if n.endswith(".tsv")}
        needed = {
            "train": ("train.tsv", True),
            "validation": ("dev.tsv", True),
            "test": ("test.tsv", False),
        }
        out = {}
        for split, (fname, has_label) in needed.items():
            if fname not in members:
                raise FileNotFoundError(f"QNLIv2.zip 缺少 {fname}；有：{sorted(members)}")
            print(f"  解析 {fname} → {split} ...")
            out[split] = _read_qnli_tsv(zf, members[fname], has_label=has_label)
    return DatasetDict(out)


def prepare_glue_dataset(task: str) -> DatasetDict:
    if task == "mnli":
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

    if task == "cola":
        try:
            return load_dataset("glue", "cola")
        except Exception as exc:
            print(f"  HuggingFace load_dataset 失败（{type(exc).__name__}），回退 Facebook CoLA.zip")
            return load_cola_from_fb_zip()

    if task == "qnli":
        try:
            return load_dataset("glue", "qnli")
        except Exception as exc:
            print(f"  HuggingFace load_dataset 失败（{type(exc).__name__}），回退 Facebook QNLIv2.zip")
            return load_qnli_from_fb_zip()

    return load_dataset("glue", task)


for task in DATASET_NAMES:
    print(f"\n========================================")
    print(f"正在下载并保存 GLUE-{task.upper()} 数据集...")

    save_path = os.path.join(ROOT_SAVE_DIR, task)
    os.makedirs(save_path, exist_ok=True)

    dataset = prepare_glue_dataset(task)
    dataset.save_to_disk(save_path)

    print(f"✅ {task.upper()} 数据集保存完成！路径：{save_path}")
    print(f"   包含：{split_size_line(dataset)}")
    print(f"   列名：{dataset['validation'].column_names}")


print(f"\n🎉 所有数据集下载完成！统一存放于：{ROOT_SAVE_DIR}")
