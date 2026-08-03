"""
准备 MRPC 校验集缓存：嵌入层输出 + 预编码权重 + MaskBank。

解决 bert_base 明文仿真瓶颈：
  - encode_weight_* 约 100s/矩阵
  - MaskBank scatter 建表约 140s
缓存一次后，对拍只跑 rotate/mult/add。

用法（仓库根）::

  # 仅层 0 注意力权重 + MaskBank + calib 嵌入（推荐先跑）
  python3 HE/prepare_mrpc_cache.py --layers 0 --stages embed,maskbank,attn

  # 层 0 含 FF
  python3 HE/prepare_mrpc_cache.py --layers 0 --stages all

  # 烟雾：用缓存跑一层并对拍 NumPy
  python3 HE/prepare_mrpc_cache.py --smoke

输出目录默认：``HE/cache/mrpc/``
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import slot_decode as sd
import thor_encoder_linear_core as core
from poly_model_inference import (
    CALIB_INDICES_DIR,
    CALIB_SEED,
    FINETUNED_MODEL_ROOT,
    LOCAL_DATA_ROOT,
    MAX_SEQ_LENGTH,
    get_preprocess_fn,
    load_calib_indices_json,
)

DEFAULT_CACHE_DIR = os.path.join(_HERE, "cache", "mrpc")


def _cfg_to_dict(cfg: core.ThorConfig) -> dict[str, int]:
    return {
        "seq_len": cfg.seq_len,
        "hidden_dim": cfg.hidden_dim,
        "num_heads": cfg.num_heads,
        "ffn_dim": cfg.ffn_dim,
        "num_slots": cfg.num_slots,
    }


def _cfg_from_dict(d: dict[str, int]) -> core.ThorConfig:
    return core.ThorConfig(
        seq_len=int(d["seq_len"]),
        hidden_dim=int(d["hidden_dim"]),
        num_heads=int(d["num_heads"]),
        ffn_dim=int(d["ffn_dim"]),
        num_slots=int(d["num_slots"]),
    )


def extract_layer_raw_weights(model, layer_idx: int) -> dict[str, np.ndarray]:
    """HuggingFace BERT 层 → core 用 (out, in) float64。"""
    layer = model.bert.encoder.layer[layer_idx]
    attn = layer.attention.self
    return {
        "w_q": attn.query.weight.detach().cpu().numpy().astype(np.float64),
        "w_k": attn.key.weight.detach().cpu().numpy().astype(np.float64),
        "w_v": attn.value.weight.detach().cpu().numpy().astype(np.float64),
        "w_o": layer.attention.output.dense.weight.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "w1": layer.intermediate.dense.weight.detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "w2": layer.output.dense.weight.detach().cpu().numpy().astype(np.float64),
    }


@torch.no_grad()
def collect_calib_embeddings(
    task_name: str = "mrpc",
    *,
    calib_seed: int = CALIB_SEED,
    calib_indices_dir: str = CALIB_INDICES_DIR,
    batch_size: int = 16,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """返回 embeddings (N,S,H)、attn_mask (N,S)、labels (N,)、meta。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(FINETUNED_MODEL_ROOT, task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, attn_implementation="eager"
    )
    model.to(device)
    model.eval()

    data_path = os.path.join(LOCAL_DATA_ROOT, task_name)
    dataset = load_from_disk(data_path)
    indices, calib_meta = load_calib_indices_json(
        task_name, calib_seed, calib_indices_dir
    )
    calib = dataset["train"].select(indices)
    cols_to_remove = [c for c in calib.column_names if c != "label"]
    tokenized = calib.map(
        get_preprocess_fn(task_name, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
        desc=f"tokenize {task_name}/calib",
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )
    loader = DataLoader(
        tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator
    )

    emb_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for batch in loader:
        labels = batch["labels"] if "labels" in batch else batch["label"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        emb = model.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        emb_chunks.append(emb.detach().cpu().numpy().astype(np.float32))
        mask_chunks.append(attention_mask.detach().cpu().numpy().astype(np.int64))
        label_chunks.append(labels.numpy().astype(np.int64))

    embeddings = np.concatenate(emb_chunks, axis=0)
    attn_mask = np.concatenate(mask_chunks, axis=0)
    labels_np = np.concatenate(label_chunks, axis=0)
    meta = {
        "task": task_name,
        "n": int(embeddings.shape[0]),
        "seq_len": int(embeddings.shape[1]),
        "hidden_dim": int(embeddings.shape[2]),
        "calib_seed": calib_seed,
        "indices": indices,
        "calib_meta": {k: v for k, v in calib_meta.items() if k != "indices"},
        "note": "bert.embeddings 输出（含 token/pos/type + emb LayerNorm）",
    }
    return embeddings, attn_mask, labels_np, meta


def encode_attn_weights(
    raw: dict[str, np.ndarray], cfg: core.ThorConfig
) -> dict[str, Any]:
    """编码 Q/K/V/W_O plaintext 打包。"""
    out: dict[str, Any] = {}
    for key, n_out, block in (
        ("w_q", cfg.head_dim, (cfg.head_dim, cfg.seq_len)),
        ("w_k", cfg.head_dim, (cfg.head_dim, cfg.seq_len)),
        ("w_v", cfg.head_dim, (cfg.head_dim, cfg.seq_len)),
        ("w_o", cfg.seq_len, (cfg.head_dim, cfg.seq_len)),
    ):
        t0 = time.perf_counter()
        out[key] = core.encode_weight_upper_diagonals(
            raw[key], cfg, cfg.n_in_slot, n_out, block
        )
        print(f"    encode {key}: {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def encode_ff_weights(
    raw: dict[str, np.ndarray], cfg: core.ThorConfig
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    t0 = time.perf_counter()
    out["w1"] = core.encode_ff_weight(
        raw["w1"], cfg, split="vertical", n_splits=cfg.ff_n_splits
    )
    print(f"    encode w1: {time.perf_counter() - t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    out["w2"] = core.encode_ff_weight(
        raw["w2"], cfg, split="horizontal", n_splits=cfg.ff_n_splits
    )
    print(f"    encode w2: {time.perf_counter() - t0:.1f}s", flush=True)
    return out


def save_maskbank(path: str, masks: core.MaskBank) -> None:
    payload = {
        "cfg": _cfg_to_dict(masks.cfg),
        "cc_score_scatter_pairs": masks.cc_score_scatter_pairs,
        "cc_context_scatter_pairs": masks.cc_context_scatter_pairs,
        "ct_ct_score": masks.ct_ct_score,
        "ct_ct_context": masks.ct_ct_context,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_maskbank(path: str, cfg: core.ThorConfig | None = None) -> core.MaskBank:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if cfg is None:
        cfg = _cfg_from_dict(payload["cfg"])
    return core.MaskBank.from_precomputed(
        cfg,
        cc_score_scatter_pairs=payload["cc_score_scatter_pairs"],
        cc_context_scatter_pairs=payload["cc_context_scatter_pairs"],
        ct_ct_score=payload.get("ct_ct_score"),
        ct_ct_context=payload.get("ct_ct_context"),
    )


def run_layer0_attn_cached(
    x: np.ndarray,
    raw: dict[str, np.ndarray],
    encoded: dict[str, Any],
    cfg: core.ThorConfig,
    masks: core.MaskBank,
    *,
    apply_softmax: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """用缓存权重跑层 0 注意力+残差（FF 占位跳过），对拍 NumPy。"""
    x_enc = core.encode_input_lower_diagonals(x, cfg)

    t0 = time.perf_counter()
    q = core.pc_matmul_from_encoded(x_enc, encoded["w_q"], cfg)
    k = core.pc_matmul_from_encoded(x_enc, encoded["w_k"], cfg)
    v = core.pc_matmul_from_encoded(x_enc, encoded["w_v"], cfg)
    t_qkv = time.perf_counter() - t0

    t0 = time.perf_counter()
    score = core.compute_attention_score_slots(q, k, cfg, masks, backend="geometric")
    ctx = core.compute_attention_context_slots(
        score, v, cfg, masks, apply_softmax=apply_softmax, backend="geometric"
    )
    t_cc = time.perf_counter() - t0

    # W_O：先 layout bridge 再 PC-MM（与 compute_output_projection_slots 一致）
    t0 = time.perf_counter()
    x_wo = core.pc_ctx_vecs_to_input_lower_slots(ctx, cfg)
    wo = core.pc_matmul_from_encoded(x_wo, encoded["w_o"], cfg)
    wo_in = core.pc_output_to_input_lower_slots(wo, cfg)
    attn_res = core.slot_vec_add(x_enc, wo_in)
    t_wo = time.perf_counter() - t0

    y = sd.decode_input_from_lower_diagonals(attn_res, cfg)
    # NumPy ref：仅到 attn residual
    q_np = x @ raw["w_q"].T
    k_np = x @ raw["w_k"].T
    v_np = x @ raw["w_v"].T
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.stack(
        [
            (sd.split_heads(k_np, cfg)[h] @ sd.split_heads(q_np, cfg)[h].T)
            * scale
            for h in range(cfg.num_heads)
        ],
        axis=0,
    )
    ctx_np = sd.attention_context_ref_from_scores(
        scores, v_np, cfg, apply_softmax=apply_softmax
    )
    y_ref = x + ctx_np @ raw["w_o"].T
    err = float(np.max(np.abs(y - y_ref)))
    timings = {"qkv_s": t_qkv, "cc_s": t_cc, "wo_s": t_wo, "err_attn_res": err}
    return y, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="MRPC HE 明文缓存：嵌入 + 预编码权重")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--layers",
        default="0",
        help="逗号分隔层号，或 all（12 层）",
    )
    parser.add_argument(
        "--stages",
        default="embed,maskbank,attn",
        help="embed,maskbank,attn,ff,all 逗号分隔",
    )
    parser.add_argument("--smoke", action="store_true", help="加载缓存跑层0注意力对拍")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--calib-seed", type=int, default=CALIB_SEED)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    cfg = core.bert_base()
    cfg.validate()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    if "all" in stages:
        stages = {"embed", "maskbank", "attn", "ff"}

    if args.layers.strip().lower() == "all":
        layer_ids = list(range(12))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]

    meta_path = os.path.join(args.cache_dir, "meta.json")

    if "embed" in stages and not args.smoke:
        print("=== calib embeddings ===", flush=True)
        t0 = time.perf_counter()
        emb, mask, labels, emb_meta = collect_calib_embeddings(
            calib_seed=args.calib_seed
        )
        np.save(os.path.join(args.cache_dir, "calib_embeddings.npy"), emb)
        np.save(os.path.join(args.cache_dir, "calib_attn_mask.npy"), mask)
        np.save(os.path.join(args.cache_dir, "calib_labels.npy"), labels)
        with open(os.path.join(args.cache_dir, "calib_meta.json"), "w", encoding="utf-8") as f:
            json.dump(emb_meta, f, ensure_ascii=False, indent=2)
        print(
            f"  saved embeddings {emb.shape} in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    if "maskbank" in stages and not args.smoke:
        print("=== MaskBank ===", flush=True)
        mb_path = os.path.join(args.cache_dir, "maskbank.pkl")
        t0 = time.perf_counter()
        masks = core.MaskBank(cfg)
        save_maskbank(mb_path, masks)
        print(f"  saved {mb_path} in {time.perf_counter() - t0:.1f}s", flush=True)

    need_weights = ("attn" in stages or "ff" in stages) and not args.smoke
    if need_weights:
        model_path = os.path.join(FINETUNED_MODEL_ROOT, "mrpc")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=2, attn_implementation="eager"
        )
        model.to(device)
        model.eval()
        layers_dir = os.path.join(args.cache_dir, "layers")
        os.makedirs(layers_dir, exist_ok=True)
        for li in layer_ids:
            print(f"=== layer {li} weights ===", flush=True)
            layer_dir = os.path.join(layers_dir, f"{li:02d}")
            os.makedirs(layer_dir, exist_ok=True)
            raw = extract_layer_raw_weights(model, li)
            np.savez_compressed(
                os.path.join(layer_dir, "raw.npz"),
                **{k: v for k, v in raw.items()},
            )
            encoded: dict[str, Any] = {}
            if "attn" in stages:
                encoded.update(encode_attn_weights(raw, cfg))
            if "ff" in stages:
                encoded.update(encode_ff_weights(raw, cfg))
            with open(os.path.join(layer_dir, "encoded.pkl"), "wb") as f:
                pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  saved {layer_dir}", flush=True)

    # write/update meta
    meta = {
        "task": "mrpc",
        "cfg": _cfg_to_dict(cfg),
        "layers": layer_ids,
        "stages": sorted(stages),
        "cache_dir": os.path.abspath(args.cache_dir),
    }
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        old.update(meta)
        meta = old
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.smoke:
        print("=== smoke: cached layer0 attn ===", flush=True)
        emb = np.load(os.path.join(args.cache_dir, "calib_embeddings.npy"))
        raw = dict(
            np.load(os.path.join(args.cache_dir, "layers", "00", "raw.npz"))
        )
        with open(
            os.path.join(args.cache_dir, "layers", "00", "encoded.pkl"), "rb"
        ) as f:
            encoded = pickle.load(f)
        masks = load_maskbank(os.path.join(args.cache_dir, "maskbank.pkl"), cfg)
        x = emb[args.sample_idx].astype(np.float64)
        t0 = time.perf_counter()
        _, timings = run_layer0_attn_cached(x, raw, encoded, cfg, masks)
        print(
            f"  sample={args.sample_idx}  total={time.perf_counter() - t0:.2f}s  "
            f"qkv={timings['qkv_s']:.2f}s cc={timings['cc_s']:.2f}s "
            f"wo={timings['wo_s']:.2f}s  err_attn_res={timings['err_attn_res']:.3e}",
            flush=True,
        )
        if timings["err_attn_res"] > 1e-5:
            raise SystemExit("smoke FAIL: attn residual error too large")
        print("  smoke PASS", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
