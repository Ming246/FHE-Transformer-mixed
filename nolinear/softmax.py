"""
统计 thor Softmax 中间量级（按 task × layer）：

  1) 第一部分 σ = Σ exp_approx
  2) 各 δ2 轮的 s = Σ y²，并与 THOR 公式 e_n/128/2 比较
     （e_n 为上一轮 aSOR 结束时的 en≈1−α；保守 ⟺ min(s) ≥ e_n/128/2）

参数来自 softmax_poly.py。验证集上截获 attention scores。

用法（在 nolinear/ 下）：
  python3 softmax.py
  python3 softmax.py --tasks mrpc --samples 64
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import sys

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

from softmax_poly import (  # noqa: E402
    LAYER_EXP_DIV_BY_TASK,
    LAYER_EXP_SHIFT_BY_TASK,
    THOR_DELTA1,
    THOR_EXP_POLY_COEFFS,
    THOR_INPUT_SCALE,
    _check_power_of_two,
)

# ===================== 配置 =====================
TASK_NAMES = ["mrpc", "rte", "sst2"]
LOCAL_DATA_ROOT = "../glue_datasets/"
FINETUNED_MODEL_ROOT = "../finetuned_weight/"

MAX_SEQ_LENGTH = 128
NUM_LAYERS = 12
NUM_LABELS = 2
BATCH_SIZE = 8
SCORE_VALID_THRESHOLD = -1e4

NUM_SAMPLES: int | None = None
RANDOM_SEED = 42
SPLIT = "validation"

# 与 THOR he_softmax 一致：第一次 inv 用 alpha=internal_alpha/10=0.01
ASOR_ALPHA = 0.01
# 第一次 e0 参照（THOR he_softmax1）
THOR_INV_EPSILON_REF = 2 ** (-11)
# ================================================


def asor_final_en(e0: float, alpha: float, max_iters: int = 128) -> float:
    """仅更新误差界 e（与 THOR he_inv 中 en 轨迹一致），返回结束时的 en。"""
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    for _ in range(max_iters):
        if en >= 1.0 - alpha:
            break
        kn = 2.0 / (en + 1.0)
        en = kn * en * (2.0 - kn * en)
    return en


def thor_eps2_from_en(en: float, seq_len: int = MAX_SEQ_LENGTH) -> float:
    """THOR update_inv_D: epsilon2 = output_precision / 128 / 2。"""
    return float(en) / float(seq_len) / 2.0


def _empty_scalar_stats() -> dict[str, float]:
    return {
        "n": 0,
        "min": float("nan"),
        "max": float("nan"),
        "mean": float("nan"),
        "p01": float("nan"),
        "p50": float("nan"),
    }


def _reduce_scalar_chunks(
    chunks: list[torch.Tensor],
) -> dict[str, float]:
    if not chunks:
        return _empty_scalar_stats()
    all_v = torch.cat([c.detach().cpu().reshape(-1) for c in chunks], dim=0)
    n = int(all_v.numel())
    if n == 0:
        return _empty_scalar_stats()
    return {
        "n": n,
        "min": float(all_v.min()),
        "max": float(all_v.max()),
        "mean": float(all_v.double().mean()),
        "p01": float(torch.quantile(all_v, 0.01)),
        "p50": float(torch.quantile(all_v, 0.50)),
    }


@torch.no_grad()
def collect_sigma_and_sum_ysq(
    x: torch.Tensor,
    mask: torch.Tensor,
    shift: float,
    delta1: float,
    delta2: float,
    exp_coeffs: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    与 softmax_poly.thor_softmax 同结构。
    第一次归一化用精确 1/σ（便于看 Σy² 几何量级）；
    返回 (σ[N], [每轮 Σy²[N], ...])，轮数 = log2(δ2)。
    """
    m = (mask > 0).to(dtype=x.dtype)
    x_work = torch.where(m > 0, x, torch.zeros_like(x))

    d1 = float(delta1)
    d2 = float(delta2)
    n1 = _check_power_of_two("delta1", d1)
    n2 = _check_power_of_two("delta2", d2)

    x_scaled = x_work / d1 / d2 / THOR_INPUT_SCALE
    exp_approx = torch.zeros_like(x_scaled)
    for coeff in exp_coeffs:
        exp_approx = exp_approx * x_scaled + coeff

    scale = math.exp(shift / d2 / d1)
    exp_approx = exp_approx / scale
    for _ in range(n1):
        exp_approx = exp_approx**2

    exp_approx = exp_approx * m
    sigma = exp_approx.sum(dim=-1)
    y = exp_approx / sigma.clamp(min=1e-30).unsqueeze(-1)

    sum_ysq_rounds: list[torch.Tensor] = []
    for _ in range(n2):
        y_sq = y * y
        s = (y_sq * m).sum(dim=-1)
        sum_ysq_rounds.append(s)
        y = y_sq * (1.0 / s.clamp(min=1e-30)).unsqueeze(-1)

    return sigma, sum_ysq_rounds


class ScoreCollector:
    def __init__(self, device: torch.device):
        self.device = device
        self._scores: dict[int, list[torch.Tensor]] = {
            i: [] for i in range(NUM_LAYERS)
        }

    def add(self, layer_idx: int, scores: torch.Tensor) -> None:
        rows = scores.detach().reshape(-1, scores.shape[-1]).to(
            self.device, dtype=torch.float32
        )
        l = rows.shape[-1]
        if l < MAX_SEQ_LENGTH:
            pad = torch.full(
                (rows.shape[0], MAX_SEQ_LENGTH - l),
                float(SCORE_VALID_THRESHOLD),
                device=self.device,
                dtype=torch.float32,
            )
            rows = torch.cat([rows, pad], dim=-1)
        elif l > MAX_SEQ_LENGTH:
            rows = rows[:, :MAX_SEQ_LENGTH]
        self._scores[layer_idx].append(rows)

    def stacked(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self._scores[layer_idx]
        if not parts:
            z = torch.zeros(0, 0, device=self.device, dtype=torch.float32)
            return z, z
        scores = torch.cat(parts, dim=0)
        masks = (scores > SCORE_VALID_THRESHOLD).to(dtype=scores.dtype)
        return scores, masks

    def release(self, layer_idx: int) -> None:
        self._scores.pop(layer_idx, None)


def register_score_hooks(model, collector: ScoreCollector) -> list:
    hooks: list = []
    for layer_idx in range(NUM_LAYERS):
        attn_self = model.bert.encoder.layer[layer_idx].attention.self
        orig_forward = attn_self.forward

        def make_patched(orig_fn, lidx, coll, attn_module):
            @functools.wraps(orig_fn)
            def patched_forward(
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_value=None,
                output_attentions=False,
                **kwargs,
            ):
                mixed_query = attn_module.query(hidden_states)
                key_in = (
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )
                mixed_key = attn_module.key(key_in)
                head_dim = attn_module.attention_head_size
                num_heads = attn_module.num_attention_heads

                def transpose_for_scores(t):
                    new_shape = t.size()[:-1] + (num_heads, head_dim)
                    return t.view(new_shape).permute(0, 2, 1, 3)

                query_layer = transpose_for_scores(mixed_query)
                key_layer = transpose_for_scores(mixed_key)
                attention_scores = torch.matmul(
                    query_layer, key_layer.transpose(-1, -2)
                )
                attention_scores = attention_scores / (head_dim**0.5)
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]
                    attention_scores = attention_scores + attention_mask
                coll.add(lidx, attention_scores)
                return orig_fn(
                    hidden_states,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    **kwargs,
                )

            return patched_forward

        attn_self.forward = make_patched(
            orig_forward, layer_idx, collector, attn_self
        )
        hooks.append((attn_self, orig_forward))
    return hooks


def restore_hooks(hooks: list) -> None:
    for attn_self, orig_forward in hooks:
        attn_self.forward = orig_forward


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


def select_split(dataset, n: int | None, seed: int):
    if n is None:
        return dataset
    n = min(n, len(dataset))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    return dataset.select(indices)


@torch.no_grad()
def layer_intermediate_stats(
    scores: torch.Tensor,
    masks: torch.Tensor,
    shift: float,
    delta2: float,
    exp_coeffs: torch.Tensor,
    chunk_size: int = 4096,
) -> dict:
    n = scores.shape[0]
    n2 = _check_power_of_two("delta2", float(delta2))
    if n == 0:
        return {
            "sigma": _empty_scalar_stats(),
            "sum_ysq": [_empty_scalar_stats() for _ in range(n2)],
        }

    sigma_chunks: list[torch.Tensor] = []
    ysq_chunks: list[list[torch.Tensor]] = [[] for _ in range(n2)]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sigma, rounds = collect_sigma_and_sum_ysq(
            scores[start:end],
            masks[start:end],
            shift,
            THOR_DELTA1,
            delta2,
            exp_coeffs,
        )
        sigma_chunks.append(sigma)
        for r, s in enumerate(rounds):
            ysq_chunks[r].append(s)

    return {
        "sigma": _reduce_scalar_chunks(sigma_chunks),
        "sum_ysq": [_reduce_scalar_chunks(ysq_chunks[r]) for r in range(n2)],
    }


def run_task(task_name: str, device: torch.device, num_samples: int | None) -> None:
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")

    shifts = LAYER_EXP_SHIFT_BY_TASK[task_name]
    divs = LAYER_EXP_DIV_BY_TASK[task_name]
    if len(shifts) != NUM_LAYERS or len(divs) != NUM_LAYERS:
        raise ValueError(f"{task_name} shift/div 长度应为 {NUM_LAYERS}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    split = select_split(dataset[SPLIT], num_samples, RANDOM_SEED)
    tokenized = split.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=split.column_names,
    )
    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    collector = ScoreCollector(device)
    hooks = register_score_hooks(model, collector)
    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    finally:
        restore_hooks(hooks)

    exp_coeffs = torch.tensor(
        THOR_EXP_POLY_COEFFS, device=device, dtype=torch.float32
    )

    en_after_thor_e0 = asor_final_en(THOR_INV_EPSILON_REF, ASOR_ALPHA)
    eps2_from_thor_e0 = thor_eps2_from_en(en_after_thor_e0)

    print(f"\n{'=' * 88}")
    print(
        f"{task_name.upper()}  split={SPLIT}  n_samples={len(split)}  "
        f"δ1={THOR_DELTA1}  参数=softmax_poly"
    )
    print(
        f"THOR 参照：e0_σ={THOR_INV_EPSILON_REF:.4e} → aSOR 终态 en={en_after_thor_e0:.6f} "
        f"→ en/128/2={eps2_from_thor_e0:.4e}  (α={ASOR_ALPHA})"
    )
    print(
        "Σy² 在精确 1/σ 归一化之后统计；ok ⟺ min(Σy²) ≥ en/128/2（足够保守）"
    )

    print(f"\n--- 第一部分 σ=Σexp ---")
    print(
        f"{'层':>3} {'shift':>7} {'δ2':>4} {'n':>10} "
        f"{'minσ':>11} {'p01σ':>11} {'meanσ':>11} {'maxσ':>11}"
    )

    layer_stats: list[dict] = []
    for layer_idx in range(NUM_LAYERS):
        scores, masks = collector.stacked(layer_idx)
        st = layer_intermediate_stats(
            scores,
            masks,
            shifts[layer_idx],
            divs[layer_idx],
            exp_coeffs,
        )
        layer_stats.append(st)
        sg = st["sigma"]
        print(
            f"{layer_idx:3d} {shifts[layer_idx]:7.4g} {divs[layer_idx]:4g} "
            f"{sg['n']:10d} "
            f"{sg['min']:11.4e} {sg['p01']:11.4e} "
            f"{sg['mean']:11.4e} {sg['max']:11.4e}"
        )
        collector.release(layer_idx)

    print(f"\n--- 第二部分 Σy² vs en/128/2（e0←本层实测 minσ）---")
    print(
        f"{'层':>3} {'轮':>3} {'minΣy²':>11} {'p01':>11} {'mean':>11} {'max':>11} "
        f"{'en_fin':>8} {'en/128/2':>11} {'min/eps2':>9} {'ok?':>4}"
    )
    n_fail = 0
    n_cmp = 0
    for layer_idx in range(NUM_LAYERS):
        st = layer_stats[layer_idx]
        min_sigma = st["sigma"]["min"]
        if not math.isfinite(min_sigma) or min_sigma <= 0:
            en_fin = float("nan")
            eps2 = float("nan")
        else:
            en_fin = asor_final_en(min_sigma, ASOR_ALPHA)
            eps2 = thor_eps2_from_en(en_fin)

        for r, ys in enumerate(st["sum_ysq"]):
            ratio = (
                ys["min"] / eps2
                if math.isfinite(eps2) and eps2 > 0 and math.isfinite(ys["min"])
                else float("nan")
            )
            ok = bool(math.isfinite(ratio) and ratio >= 1.0)
            n_cmp += 1
            if not ok:
                n_fail += 1
            print(
                f"{layer_idx:3d} {r:3d} "
                f"{ys['min']:11.4e} {ys['p01']:11.4e} "
                f"{ys['mean']:11.4e} {ys['max']:11.4e} "
                f"{en_fin:8.5f} {eps2:11.4e} {ratio:9.3f} "
                f"{'OK' if ok else 'LOW'}"
            )
            if math.isfinite(eps2) and eps2 > 0:
                en_fin = asor_final_en(eps2, ASOR_ALPHA)
                eps2 = thor_eps2_from_en(en_fin)

    print(
        f"\n--- 对照：全层共用 THOR e0={THOR_INV_EPSILON_REF:.4e} "
        f"→ en/128/2={eps2_from_thor_e0:.4e} ---"
    )
    print(
        f"{'层':>3} {'轮':>3} {'minΣy²':>11} {'thor_eps2':>11} {'min/eps2':>9} {'ok?':>4}"
    )
    n_fail_thor = 0
    for layer_idx in range(NUM_LAYERS):
        st = layer_stats[layer_idx]
        en_fin = en_after_thor_e0
        eps2 = eps2_from_thor_e0
        for r, ys in enumerate(st["sum_ysq"]):
            ratio = ys["min"] / eps2 if eps2 > 0 else float("nan")
            ok = bool(math.isfinite(ratio) and ratio >= 1.0)
            if not ok:
                n_fail_thor += 1
            print(
                f"{layer_idx:3d} {r:3d} "
                f"{ys['min']:11.4e} {eps2:11.4e} {ratio:9.3f} "
                f"{'OK' if ok else 'LOW'}"
            )
            en_fin = asor_final_en(eps2, ASOR_ALPHA)
            eps2 = thor_eps2_from_en(en_fin)

    print(
        f"\n汇总：以实测 minσ→en/128/2 判定 LOW={n_fail}/{n_cmp}；"
        f"以 THOR 固定 e0 判定 LOW={n_fail_thor}/{n_cmp}"
    )
    print(
        f"理想均匀下界 1/{MAX_SEQ_LENGTH}={1 / MAX_SEQ_LENGTH:.4e}；"
        f"1/(2·{MAX_SEQ_LENGTH})={1 / (2 * MAX_SEQ_LENGTH):.4e}"
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 thor Softmax σ 与 Σy²，并对照 en/128/2"
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_NAMES)
    parser.add_argument(
        "--samples",
        type=int,
        default=NUM_SAMPLES,
        help="验证子集大小；默认全量",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    for task_name in args.tasks:
        run_task(task_name, device, args.samples)


if __name__ == "__main__":
    main()
