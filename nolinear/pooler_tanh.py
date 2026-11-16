"""
统计 BertPooler.dense 输出（即 tanh 输入）在各 GLUE 任务上的区间。

口径：
  pooled = dense(h_CLS)   # [B, 768]，精确 tanh 的输入
  与 HuggingFace BertPooler / THOR ThorBertPooler 相同。
  THOR 密文 tanh 假定 |z| ≤ 40（dense 后 ×1/40）。

用法（在 nolinear/ 下）：
  python3 pooler_tanh.py                         # 默认 all：校验集+验证集并集
  python3 pooler_tanh.py --eval-split validation
  python3 pooler_tanh.py --eval-split calib
  python3 pooler_tanh.py --task mrpc --samples 32

  --eval-split all：先 calib 再 validation，JSON 写并集（min 更小、max 更大）
  --eval-split calib：只读 coverage_metric 索引，不重建

输出：tanh_out/{task}_tanh.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ===================== 配置 =====================
TASK_NAMES = ["mrpc", "rte", "sst2", "cola", "qnli", "mnli"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = 128
TASK_NUM_LABELS: dict[str, int] = {
    "mrpc": 2,
    "rte": 2,
    "sst2": 2,
    "cola": 2,
    "qnli": 2,
    "mnli": 3,
}

NUM_SAMPLES = None
RANDOM_SEED = 42
BATCH_SIZE = 32

DEFAULT_EVAL_SPLIT = "all"  # "all" | "validation" | "calib"
CALIB_SEED = 42
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_INDICES_DIR = os.path.abspath(
    os.path.join(_SCRIPT_DIR, "..", "results", "coverage_metrics")
)
JSON_DIR = os.path.join(_SCRIPT_DIR, "tanh_out")

# THOR he_tanh 的定义域半宽（dense 后 ×1/scale）
THOR_TANH_SCALE = 40.0
PERCENTILES = (0.1, 1.0, 50.0, 99.0, 99.9)
# ======================================================================


def task_num_labels(task_name: str) -> int:
    if task_name not in TASK_NUM_LABELS:
        raise KeyError(f"未知任务 {task_name}，请在 TASK_NUM_LABELS 中配置")
    return TASK_NUM_LABELS[task_name]


def get_preprocess_fn(task_name, tokenizer):
    if task_name in ("sst2", "cola"):
        def fn(examples):
            return tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
    elif task_name == "qnli":
        def fn(examples):
            return tokenizer(
                examples["question"],
                examples["sentence"],
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
    elif task_name == "mnli":
        def fn(examples):
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
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


def select_validation_split(dataset, n: int | None, seed: int):
    if n is None:
        return dataset, len(dataset), "all"
    n = min(n, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return dataset.select(indices), n, "random_subset"


def _calib_indices_path(task_name: str, seed: int, indices_dir: str) -> str:
    return os.path.join(indices_dir, f"{task_name}_calib_indices_seed{seed}.json")


def load_calib_indices_json(
    task_name: str, seed: int, indices_dir: str
) -> tuple[list[int], dict]:
    path = _calib_indices_path(task_name, seed, indices_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"校验集索引不存在：{path}；请先运行 coverage_metric.py 生成"
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if "indices" not in payload or not payload["indices"]:
        raise ValueError(f"校验集索引为空或缺少 indices：{path}")
    indices = [int(i) for i in payload["indices"]]
    meta = {k: v for k, v in payload.items() if k != "indices"}
    meta["indices_path"] = path
    return indices, meta


def resolve_calib_eval_split(
    task_name: str,
    dataset,
    *,
    calib_seed: int,
    indices_dir: str,
):
    if "train" not in dataset:
        raise KeyError(f"{task_name} 缺少 train，无法构造校验集")
    indices, meta = load_calib_indices_json(task_name, calib_seed, indices_dir)
    n_train = len(dataset["train"])
    bad = [i for i in indices if i < 0 or i >= n_train]
    if bad:
        raise IndexError(
            f"{task_name} 校验集索引越界（train n={n_train}），例：{bad[:5]}"
        )
    desc = (
        f"calib n={len(indices)} seed={calib_seed} "
        f"← {meta.get('indices_path')}"
        f" (file calib_size={meta.get('calib_size', '?')})"
    )
    return dataset["train"].select(indices), len(indices), desc


def validation_raw_split(dataset, task_name: str):
    if "validation" in dataset:
        return dataset["validation"]
    if task_name == "mnli" and "validation_matched" in dataset:
        return dataset["validation_matched"]
    raise KeyError(
        f"{task_name} 缺少 validation / validation_matched：{list(dataset.keys())}"
    )


def resolve_eval_split(
    task_name: str,
    dataset,
    *,
    eval_split: str,
    num_samples: int | None,
    seed: int,
    calib_seed: int,
    indices_dir: str,
):
    if eval_split == "validation":
        split, n, mode = select_validation_split(
            validation_raw_split(dataset, task_name), num_samples, seed
        )
        desc = f"validation ({mode}, n={n})"
        return split, n, desc
    if eval_split == "calib":
        return resolve_calib_eval_split(
            task_name,
            dataset,
            calib_seed=calib_seed,
            indices_dir=indices_dir,
        )
    raise ValueError(
        f"未知 eval_split={eval_split!r}，可选 all / validation / calib"
    )


class TanhInputTracker:
    """流式 min/max + 收集 CLS 向量供分位数。"""

    def __init__(self, scale: float = THOR_TANH_SCALE):
        self.scale = float(scale)
        self.min_v = float("inf")
        self.max_v = float("-inf")
        self.n_samples = 0
        self.n_elem = 0
        self.n_elem_abs_gt = 0
        self.n_samples_any_abs_gt = 0
        self._chunks: list[torch.Tensor] = []

    def update(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError(f"tanh 输入应为 [B, H]，得到 {tuple(z.shape)}")
        z32 = z.detach().to(dtype=torch.float32)
        zmin = float(z32.min().item())
        zmax = float(z32.max().item())
        self.min_v = min(self.min_v, zmin)
        self.max_v = max(self.max_v, zmax)
        abs_z = z32.abs()
        gt = abs_z > self.scale
        self.n_elem_abs_gt += int(gt.sum().item())
        self.n_samples_any_abs_gt += int(gt.any(dim=-1).sum().item())
        self.n_samples += int(z32.shape[0])
        self.n_elem += int(z32.numel())
        self._chunks.append(z32.cpu())

    def stacked(self) -> torch.Tensor:
        if not self._chunks:
            raise RuntimeError("未收集到 pooler.dense 输出")
        return torch.cat(self._chunks, dim=0)

    def finalize(self) -> dict[str, Any]:
        stacked = self.stacked()
        flat = stacked.reshape(-1)
        qs = {
            str(p): float(torch.quantile(flat, p / 100.0).item())
            for p in PERCENTILES
        }
        abs_max = max(abs(self.min_v), abs(self.max_v))
        return {
            "n_samples": self.n_samples,
            "n_elements": self.n_elem,
            "hidden_size": int(self._chunks[0].shape[-1]),
            "min": self.min_v,
            "max": self.max_v,
            "abs_max": abs_max,
            "percentiles": qs,
            "thor_scale": self.scale,
            "frac_abs_gt_scale": (
                self.n_elem_abs_gt / self.n_elem if self.n_elem else 0.0
            ),
            "n_elem_abs_gt_scale": self.n_elem_abs_gt,
            "n_samples_any_abs_gt_scale": self.n_samples_any_abs_gt,
        }


def register_pooler_dense_hook(model, tracker: TanhInputTracker):
    dense = model.bert.pooler.dense

    def hook(_module, _inp, out):
        tracker.update(out)

    return dense.register_forward_hook(hook)


def _prepare_eval_model_and_loader(
    task_name: str,
    device: torch.device,
    num_samples: int | None,
    *,
    eval_split: str = "validation",
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    model=None,
    tokenizer=None,
):
    if eval_split not in ("validation", "calib"):
        raise ValueError(f"eval_split 须为 validation/calib，得到 {eval_split!r}")
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=task_num_labels(task_name),
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    eval_set, _, split_desc = resolve_eval_split(
        task_name,
        dataset,
        eval_split=eval_split,
        num_samples=num_samples,
        seed=RANDOM_SEED,
        calib_seed=calib_seed,
        indices_dir=calib_indices_dir,
    )
    tokenized = eval_set.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=eval_set.column_names,
        desc=f"tokenize {task_name}/{eval_split}",
    )
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )
    return model, tokenizer, loader, split_desc


@torch.no_grad()
def collect_tanh_input_for_split(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None = NUM_SAMPLES,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    model=None,
    tokenizer=None,
) -> tuple[dict[str, Any], torch.Tensor, str, Any, Any]:
    model, tokenizer, loader, split_desc = _prepare_eval_model_and_loader(
        task_name,
        device,
        num_samples,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
        model=model,
        tokenizer=tokenizer,
    )
    tracker = TanhInputTracker(THOR_TANH_SCALE)
    handle = register_pooler_dense_hook(model, tracker)
    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    finally:
        handle.remove()
    stats = tracker.finalize()
    stats["eval_split"] = eval_split
    stats["split_desc"] = split_desc
    return stats, tracker.stacked(), split_desc, model, tokenizer


def merge_split_stats(
    a: dict[str, Any],
    b: dict[str, Any],
    stacked: torch.Tensor,
) -> dict[str, Any]:
    """并集：min/max 取更极端；分位数由两侧 CLS 向量拼接后重算。"""
    n_elem = int(a["n_elements"]) + int(b["n_elements"])
    n_gt = int(a["n_elem_abs_gt_scale"]) + int(b["n_elem_abs_gt_scale"])
    min_v = min(float(a["min"]), float(b["min"]))
    max_v = max(float(a["max"]), float(b["max"]))
    flat = stacked.reshape(-1)
    qs = {
        str(p): float(torch.quantile(flat, p / 100.0).item()) for p in PERCENTILES
    }
    return {
        "n_samples": int(a["n_samples"]) + int(b["n_samples"]),
        "n_elements": n_elem,
        "hidden_size": int(a["hidden_size"]),
        "min": min_v,
        "max": max_v,
        "abs_max": max(abs(min_v), abs(max_v)),
        "thor_scale": float(a["thor_scale"]),
        "frac_abs_gt_scale": n_gt / n_elem if n_elem else 0.0,
        "n_elem_abs_gt_scale": n_gt,
        "n_samples_any_abs_gt_scale": (
            int(a["n_samples_any_abs_gt_scale"])
            + int(b["n_samples_any_abs_gt_scale"])
        ),
        "percentiles": qs,
    }


def save_tanh_json(
    task_name: str,
    payload: dict[str, Any],
    json_dir: str = JSON_DIR,
) -> str:
    os.makedirs(json_dir, exist_ok=True)
    path = os.path.join(json_dir, f"{task_name}_tanh.json")
    payload = dict(payload)
    payload["task"] = task_name
    payload["updated"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def _print_stats(title: str, s: dict[str, Any]) -> None:
    qs = s.get("percentiles") or {}
    qtxt = ""
    if qs:
        qtxt = "  " + "  ".join(f"p{k}={v:.4g}" for k, v in qs.items())
    print(
        f"  {title}: n={s['n_samples']}  "
        f"min={s['min']:.6g}  max={s['max']:.6g}  abs_max={s['abs_max']:.6g}  "
        f"|z|>{s['thor_scale']:g}: elem={s['frac_abs_gt_scale']:.4%} "
        f"samples={s['n_samples_any_abs_gt_scale']}"
        f"{qtxt}",
        flush=True,
    )


def run_task(
    task_name: str,
    device: torch.device,
    *,
    eval_split: str,
    num_samples: int | None,
    calib_seed: int,
    calib_indices_dir: str,
    json_dir: str,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}\n任务 {task_name}  device={device}\n{'=' * 60}", flush=True)
    splits_to_run = (
        ["calib", "validation"] if eval_split == "all" else [eval_split]
    )
    per_split: dict[str, dict[str, Any]] = {}
    stacked_parts: list[torch.Tensor] = []
    model = None
    tokenizer = None
    for sp in splits_to_run:
        stats, stacked, desc, model, tokenizer = collect_tanh_input_for_split(
            task_name,
            device,
            eval_split=sp,
            num_samples=num_samples if sp == "validation" else None,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
            model=model,
            tokenizer=tokenizer,
        )
        per_split[sp] = stats
        stacked_parts.append(stacked)
        _print_stats(desc, stats)

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if eval_split == "all":
        union = merge_split_stats(
            per_split["calib"],
            per_split["validation"],
            torch.cat(stacked_parts, dim=0),
        )
        union["eval_split"] = "union"
        union["sources"] = ["validation", "calib"]
        union["splits"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "split_desc"}
            for k, v in per_split.items()
        }
        union["note"] = (
            "BertPooler.dense(h_CLS) = tanh 输入；"
            f"thor_scale={THOR_TANH_SCALE:g} 为 THOR he_tanh 半宽"
        )
        payload = union
        _print_stats("union", union)
    else:
        sp = splits_to_run[0]
        payload = dict(per_split[sp])
        payload["sources"] = [sp]
        payload["note"] = (
            "BertPooler.dense(h_CLS) = tanh 输入；"
            f"thor_scale={THOR_TANH_SCALE:g} 为 THOR he_tanh 半宽"
        )

    path = save_tanh_json(task_name, payload, json_dir)
    print(f"💾 {path}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="统计 pooler tanh 输入区间")
    p.add_argument(
        "--task",
        nargs="*",
        default=None,
        help="任务名，默认配置块 TASK_NAMES",
    )
    p.add_argument(
        "--eval-split",
        choices=("all", "validation", "calib"),
        default=DEFAULT_EVAL_SPLIT,
    )
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--calib-seed", type=int, default=CALIB_SEED)
    p.add_argument("--calib-indices-dir", default=CALIB_INDICES_DIR)
    p.add_argument("--json-dir", default=JSON_DIR)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    global BATCH_SIZE
    BATCH_SIZE = int(args.batch_size)
    tasks = list(args.task) if args.task else list(TASK_NAMES)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(
        f"pooler tanh 输入统计  tasks={tasks}  split={args.eval_split}  "
        f"device={device}  thor_scale={THOR_TANH_SCALE:g}",
        flush=True,
    )
    rows = []
    for task in tasks:
        payload = run_task(
            task,
            device,
            eval_split=args.eval_split,
            num_samples=args.samples,
            calib_seed=args.calib_seed,
            calib_indices_dir=args.calib_indices_dir,
            json_dir=args.json_dir,
        )
        rows.append((task, payload))

    print("\n汇总（tanh 输入 = pooler.dense(CLS)）", flush=True)
    print(
        f"{'task':<8} {'n':>7} {'min':>12} {'max':>12} {'abs_max':>12} "
        f"{'|z|>40 elem':>14} {'samples>40':>11}",
        flush=True,
    )
    for task, s in rows:
        print(
            f"{task:<8} {s['n_samples']:7d} {s['min']:12.6g} {s['max']:12.6g} "
            f"{s['abs_max']:12.6g} {s['frac_abs_gt_scale']:14.4%} "
            f"{s['n_samples_any_abs_gt_scale']:11d}",
            flush=True,
        )


if __name__ == "__main__":
    main()
