"""进化搜索与 Pareto 汇报共用的推理评估（validation / calib）。"""
from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from cost import SCHEME_LEN, compute_scheme_cost
from poly_model_inference import (
    CALIB_INDICES_DIR,
    CALIB_SEED,
    FINETUNED_MODEL_ROOT,
    SCHEME_ORIGINAL,
    _load_raw_eval_split,
    _warmup_poly_kernels,
    abnormal_output_mask,
    apply_polynomial_scheme,
    compute_flips_pct,
    get_preprocess_fn,
    install_eager_attention_poly_patch,
    mean_output_kl,
    per_sample_cross_entropy,
    validate_scheme,
    _load_tokenizer_and_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_BATCH_SIZE = 64


def format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {secs:.0f}s"


def save_search_timings(output_dir: str, timings: list[tuple[str, float]]) -> str:
    """合并写入各任务进化搜索用时（秒）。

    已有 CSV：同名 task 覆盖，新 task 追加；不写 total 行。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "search_timings.csv")

    merged: dict[str, float] = {}
    order: list[str] = []
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                task = (row.get("task") or "").strip()
                if not task or task == "total":
                    continue
                try:
                    seconds = float(row["search_seconds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if task not in merged:
                    order.append(task)
                merged[task] = seconds

    for task_name, elapsed_s in timings:
        if task_name == "total":
            continue
        if task_name not in merged:
            order.append(task_name)
        merged[task_name] = float(elapsed_s)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "search_seconds", "search_time"])
        for task_name in order:
            elapsed_s = merged[task_name]
            writer.writerow(
                [task_name, f"{elapsed_s:.2f}", format_elapsed(elapsed_s)]
            )
    return path


def fmt_flips_pct(flips_pct: float) -> str:
    """反转率（已是 0–100 百分点）→ 百分比字符串。"""
    return f"{flips_pct:.2f}%"


@dataclass
class SchemeEvalResult:
    output_kl: float
    accuracy: float
    loss: float
    accuracy_delta: float
    loss_delta: float
    flips_pct: float = 0.0
    oor_count: int = 0
    oor_pct: float = 0.0
    n_eval_used: int = 0
    f1: float | None = None
    f1_delta: float | None = None


class SchemeEvaluator:
    """固定评估子集 + 单模型；scheme→KL / acc / loss 带缓存。

    eval_split:
      - validation：官方验证集
      - calib：coverage_metric 落盘索引对应的 train 校验子集
    """

    def __init__(
        self,
        task_name: str,
        *,
        eval_samples: int | None = 256,
        seed: int = 42,
        eval_split: str = "validation",
        calib_seed: int = CALIB_SEED,
        calib_indices_dir: str = CALIB_INDICES_DIR,
    ) -> None:
        if eval_split not in ("validation", "calib"):
            raise ValueError(
                f"eval_split 须为 validation/calib，得到 {eval_split!r}"
            )
        self.task_name = task_name
        self.eval_samples = eval_samples
        self.eval_split = eval_split
        self._cache: dict[tuple[int, ...], SchemeEvalResult] = {}
        self._acc_cache: dict[tuple[int, ...], float] = {}

        finetuned_model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
        if not os.path.exists(finetuned_model_path):
            raise FileNotFoundError(f"模型不存在：{finetuned_model_path}")

        install_eager_attention_poly_patch()
        self.tokenizer, self.model, _ = _load_tokenizer_and_model(task_name)

        split, split_desc = _load_raw_eval_split(
            task_name,
            eval_split=eval_split,
            calib_seed=calib_seed,
            calib_indices_dir=calib_indices_dir,
        )
        n_total = len(split)

        if eval_samples is not None and eval_samples < n_total:
            rng = random.Random(seed)
            indices = sorted(rng.sample(range(n_total), eval_samples))
            split = split.select(indices)
            self.eval_desc = (
                f"{eval_split} {eval_samples}/{n_total} 条"
                f"（seed={seed} 固定；{split_desc}）"
            )
        else:
            self.eval_desc = f"{eval_split} 全部 {n_total} 条（{split_desc}）"

        drop_cols = [c for c in split.column_names if c != "label"]
        self.eval_dataset = split.map(
            get_preprocess_fn(task_name, self.tokenizer),
            batched=True,
            remove_columns=drop_cols,
            desc=None,
        )
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        all_orig = [SCHEME_ORIGINAL] * SCHEME_LEN
        self._init_baseline(all_orig)

    @torch.no_grad()
    def _eval_pass(
        self,
    ) -> tuple[
        torch.Tensor,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float | None,
        int,
        float,
        int,
        np.ndarray,
    ]:
        loader = DataLoader(
            self.eval_dataset,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=self.collator,
        )
        logits_chunks: list[torch.Tensor] = []
        pred_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        loss_chunks: list[torch.Tensor] = []
        invalid_chunks: list[np.ndarray] = []
        n_abnormal = 0
        n_total = 0

        for batch in loader:
            labels = batch["labels"]
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)
            logits = outputs.logits.detach().cpu()
            labels_cpu = labels.long()
            bs = labels.size(0)
            n_total += bs

            batch_invalid = abnormal_output_mask(logits)
            n_abnormal += int(batch_invalid.sum().item())

            per_loss = per_sample_cross_entropy(logits, labels_cpu)
            valid = ~batch_invalid & torch.isfinite(per_loss)
            if bool(valid.any()):
                loss_chunks.append(per_loss[valid])

            logits_chunks.append(logits)
            pred_chunks.append(logits.argmax(dim=-1))
            label_chunks.append(labels)
            invalid_chunks.append(batch_invalid.numpy())

        logits = torch.cat(logits_chunks, dim=0)
        preds = torch.cat(pred_chunks, dim=0).numpy()
        labels_np = torch.cat(label_chunks, dim=0).numpy()
        invalid_np = np.concatenate(invalid_chunks)
        valid_np = ~invalid_np
        n_used = int(valid_np.sum())

        if n_used > 0:
            acc = float(accuracy_score(labels_np[valid_np], preds[valid_np]))
            loss = float(torch.cat(loss_chunks).mean().item())
            if self.task_name == "mrpc":
                f1 = float(f1_score(labels_np[valid_np], preds[valid_np]))
            else:
                f1 = None
        else:
            acc = float("nan")
            loss = float("nan")
            f1 = None

        abnormal_pct = 100.0 * n_abnormal / n_total if n_total > 0 else 0.0
        return (
            logits,
            preds,
            labels_np,
            acc,
            loss,
            f1,
            n_abnormal,
            abnormal_pct,
            n_used,
            valid_np,
        )

    @torch.no_grad()
    def _init_baseline(self, scheme: list[int]) -> None:
        validate_scheme(scheme, self.task_name)
        apply_polynomial_scheme(self.model, scheme, self.task_name)
        _warmup_poly_kernels(self.model, scheme)
        self.model.eval()

        (
            self.baseline_logits,
            self.baseline_preds,
            self.labels_np,
            self.baseline_acc,
            self.baseline_loss,
            self.baseline_f1,
            _,
            _,
            self.baseline_n_used,
            self.baseline_valid_mask,
        ) = self._eval_pass()
        self.baseline_output_kl = 0.0
        self._cache[tuple(scheme)] = SchemeEvalResult(
            output_kl=0.0,
            accuracy=self.baseline_acc,
            loss=self.baseline_loss,
            accuracy_delta=0.0,
            loss_delta=0.0,
            flips_pct=0.0,
            oor_count=0,
            oor_pct=0.0,
            n_eval_used=self.baseline_n_used,
            f1=self.baseline_f1,
            f1_delta=0.0 if self.baseline_f1 is not None else None,
        )

    @torch.no_grad()
    def evaluate_for_scheme(self, scheme: list[int]) -> SchemeEvalResult:
        key = tuple(scheme)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        validate_scheme(scheme, self.task_name)
        apply_polynomial_scheme(self.model, scheme, self.task_name)
        _warmup_poly_kernels(self.model, scheme)
        self.model.eval()

        (
            logits_poly,
            poly_preds,
            _,
            acc,
            loss,
            f1,
            abnormal_count,
            abnormal_pct,
            n_used,
            valid_mask,
        ) = self._eval_pass()
        valid_t = torch.from_numpy(valid_mask)
        output_kl = mean_output_kl(
            self.baseline_logits, logits_poly, valid_mask=valid_t
        )
        if bool(valid_mask.any()):
            base_acc_valid = float(
                accuracy_score(
                    self.labels_np[valid_mask], self.baseline_preds[valid_mask]
                )
            )
            base_loss_valid = float(
                per_sample_cross_entropy(
                    self.baseline_logits, torch.from_numpy(self.labels_np).long()
                )[valid_t]
                .mean()
                .item()
            )
            if self.task_name == "mrpc" and self.baseline_f1 is not None:
                base_f1_valid = float(
                    f1_score(
                        self.labels_np[valid_mask], self.baseline_preds[valid_mask]
                    )
                )
                f1_delta = f1 - base_f1_valid
            else:
                f1_delta = None
            flips_pct = compute_flips_pct(
                self.baseline_preds[valid_mask],
                poly_preds[valid_mask],
                self.labels_np[valid_mask],
            )
            acc_delta = acc - base_acc_valid
            loss_delta = loss - base_loss_valid
        else:
            acc_delta = float("nan")
            loss_delta = float("nan")
            f1_delta = float("nan") if self.baseline_f1 is not None else None
            flips_pct = float("nan")

        cached = SchemeEvalResult(
            output_kl=output_kl,
            accuracy=acc,
            loss=loss,
            accuracy_delta=acc_delta,
            loss_delta=loss_delta,
            flips_pct=flips_pct,
            oor_count=abnormal_count,
            oor_pct=abnormal_pct,
            n_eval_used=n_used,
            f1=f1,
            f1_delta=f1_delta,
        )
        self._cache[key] = cached
        return cached

    @torch.no_grad()
    def accuracy_for_scheme(self, scheme: list[int]) -> float:
        """仅返回准确率（单遍前向，不算 KL）。"""
        key = tuple(scheme)
        cached = self._acc_cache.get(key)
        if cached is not None:
            return cached

        validate_scheme(scheme, self.task_name)
        apply_polynomial_scheme(self.model, scheme, self.task_name)
        _warmup_poly_kernels(self.model, scheme)
        self.model.eval()
        _, _, _, acc, _, _, _, _, _, _ = self._eval_pass()
        self._acc_cache[key] = acc
        return acc


def enrich_pareto_report(task_name: str, archive, scheme_eval: SchemeEvaluator) -> None:
    """在 scheme_eval 对应评估集上单遍推理，刷新 Pareto 解汇报指标。"""
    unique_schemes: dict[tuple[int, ...], list[int]] = {}
    for ind in archive.items:
        key = tuple(ind.scheme)
        if key not in unique_schemes:
            unique_schemes[key] = ind.scheme

    n_unique = len(unique_schemes)
    print(
        f"  汇报评估 {n_unique} 个 Pareto 方案"
        f"（{scheme_eval.eval_desc}）..."
    )
    metrics_by_key: dict[tuple[int, ...], SchemeEvalResult] = {}
    for i, (key, scheme) in enumerate(unique_schemes.items(), start=1):
        metrics_by_key[key] = scheme_eval.evaluate_for_scheme(scheme)
        if i % max(1, n_unique // 5) == 0 or i == n_unique:
            print(f"    已评估 {i}/{n_unique}")

    for ind in archive.items:
        metrics = metrics_by_key[tuple(ind.scheme)]
        ind.total_depth = int(compute_scheme_cost(task_name, ind.scheme))
        ind.accuracy_delta = metrics.accuracy_delta
        ind.loss_delta = metrics.loss_delta
        ind.f1_delta = metrics.f1_delta
        ind.output_kl = metrics.output_kl
        ind.flips_pct = metrics.flips_pct
        ind.oor_count = metrics.oor_count
        ind.oor_pct = metrics.oor_pct
        ind.n_eval_used = metrics.n_eval_used
        if hasattr(ind, "f_output_kl"):
            # 汇报重算若出现非有限 KL，保留搜索期目标值，避免污染 archive
            if math.isfinite(metrics.output_kl):
                ind.f_output_kl = metrics.output_kl
            elif not math.isfinite(getattr(ind, "f_output_kl", float("nan"))):
                ind.f_output_kl = float("inf")
        if hasattr(ind, "f_acc_delta"):
            ind.f_acc_delta = abs(metrics.accuracy_delta)

    # 丢掉重算后仍无有效 KL 的个体
    if hasattr(archive, "items") and archive.items and hasattr(
        archive.items[0], "f_output_kl"
    ):
        before = len(archive.items)
        archive.items = [
            ind for ind in archive.items if math.isfinite(ind.f_output_kl)
        ]
        dropped = before - len(archive.items)
        if dropped:
            print(f"  警告：汇报后剔除 {dropped} 个非有限 KL 的 Pareto 解")
