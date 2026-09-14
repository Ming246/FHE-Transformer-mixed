#!/usr/bin/env python3
"""
Verify repo Softmax (``slot_softmax_he`` / poly params) on **real** MRPC data.

Uses finetuned ``finetuned_weight/mrpc`` + one GLUE MRPC sample:
  embedding → dense Q,K → score packs → exact slot_softmax vs slot_softmax_he.

This is the mainline 8-pack layout (layout unchanged through Softmax).
Does **not** exercise THOR DualRail 128-copy (that is Branch-A context).
"""
from __future__ import annotations

import argparse
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

_HE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
if _HE not in sys.path:
    sys.path.insert(0, _HE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    encode_key_valid_mask_packs,
    slot_softmax_he,
)
from thor_encoder_linear_core import bert_base, slot_softmax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from linear_eval import (  # noqa: E402
    decode_score_packs,
    dense_qk_scores,
    encode_score_packs_from_dense,
)

# reuse tokenize helper from sensitive_score_1
sys.path.insert(0, _REPO)
from sensitive_score_1 import get_preprocess_fn  # noqa: E402


def _one_layer_input(
    task: str,
    sample: int,
    layer: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_in, key_valid, w_q, w_k) for encoder layer ``layer``."""
    model_path = os.path.join(_REPO, "finetuned_weight", task)
    data_path = os.path.join(_REPO, "glue_datasets", task)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2, attn_implementation="eager"
    )
    model.to(device)
    model.eval()
    ds = load_from_disk(data_path)["validation"]
    row = ds.select([sample])
    cols = [c for c in row.column_names if c != "label"]
    tok = row.map(
        get_preprocess_fn(task, tokenizer),
        batched=True,
        remove_columns=cols,
    )
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=128
    )
    batch = next(iter(DataLoader(tok, batch_size=1, collate_fn=collator)))
    batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        out = model.bert(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            output_hidden_states=True,
        )
        # hidden_states[L] = input to encoder layer L
        x = out.hidden_states[layer].detach().cpu().numpy()[0].astype(np.float64)
    mask = batch["attention_mask"][0].detach().cpu().numpy().astype(np.float64)
    bl = model.bert.encoder.layer[layer]
    w_q = bl.attention.self.query.weight.detach().cpu().numpy().astype(np.float64)
    w_k = bl.attention.self.key.weight.detach().cpu().numpy().astype(np.float64)
    return x, mask, w_q, w_k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    cfg = bert_base()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"load layer-input task={args.task} sample={args.sample} "
        f"layer={args.layer} ...",
        flush=True,
    )
    x, attn_mask, w_q, w_k = _one_layer_input(
        args.task, args.sample, args.layer, device
    )
    key_valid = (attn_mask > 0).astype(np.float64)
    print(
        f"  x={x.shape} n_valid={int(key_valid.sum())} device={device}",
        flush=True,
    )

    q = x @ w_q.T
    k = x @ w_k.T
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = dense_qk_scores(q, k, cfg, scale=scale)
    for h in range(cfg.num_heads):
        for query in range(cfg.seq_len):
            scores[h, :, query] = np.where(
                key_valid > 0, scores[h, :, query], -1e4
            )

    packs = encode_score_packs_from_dense(scores, cfg)
    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)

    print("exact slot_softmax ...", flush=True)
    exact = slot_softmax(packs, cfg)
    for pi in range(len(exact)):
        exact[pi] = exact[pi] * mask_packs[pi]

    print(
        f"slot_softmax_he (task={args.task} L{args.layer} level={args.level}) ...",
        flush=True,
    )
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    he = slot_softmax_he(packs, mask_packs, cfg, params)

    err = max(float(np.max(np.abs(he[i] - exact[i]))) for i in range(len(he)))
    he_d = decode_score_packs(he, cfg)
    ex_d = decode_score_packs(exact, cfg)
    for h in range(cfg.num_heads):
        for query in range(cfg.seq_len):
            he_d[h, key_valid < 1, query] = 0.0
            ex_d[h, key_valid < 1, query] = 0.0
    err_d = float(np.max(np.abs(he_d - ex_d)))
    print(
        f"  pack max|err|={err:.3e}  dense max|err|={err_d:.3e}  "
        f"max|he|={float(np.max(np.abs(he_d))):.3e}  "
        f"max|exact|={float(np.max(np.abs(ex_d))):.3e}",
        flush=True,
    )
    ok = err_d < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
