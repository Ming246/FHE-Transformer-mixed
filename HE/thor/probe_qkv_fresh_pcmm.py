#!/usr/bin/env python3
"""Fresh DualRail Q PC-MM vs mainline (isolates pickle/ThorAttention)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    encode_qkv_upper_from_dense,
    encode_weight_qkv_dualrail,
    encrypt_embedding_dualrail,
    mainline_qkv_ref,
    make_qkv_evaluator,
    prepare_linear_keys,
    qkv_pcmm_he,
)

sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from thor_encoder_linear_core import bert_base  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import load_from_disk  # noqa: E402


def main() -> int:
    local = os.path.join(_REPO, "finetuned_weight", "mrpc")
    model = AutoModelForSequenceClassification.from_pretrained(
        local, local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(local, local_files_only=True)
    ds = load_from_disk(os.path.join(_REPO, "glue_datasets", "mrpc"))["validation"]
    row = ds[0]
    enc = tok(
        row["sentence1"],
        row["sentence2"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        h = (
            model.bert(
                **{
                    k: v
                    for k, v in enc.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                },
                output_hidden_states=True,
            )
            .hidden_states[0][0]
            .numpy()
            .astype(np.float64)
        )
        W = (
            model.bert.encoder.layer[0]
            .attention.self.query.weight.detach()
            .numpy()
            .astype(np.float64)
        )
        b = (
            model.bert.encoder.layer[0]
            .attention.self.query.bias.detach()
            .numpy()
            .astype(np.float64)
        )

    cfg = bert_base()
    ref = mainline_qkv_ref(h, W, scale=1.0, out_packs=4)
    w_only = encode_qkv_upper_from_dense(h @ W.T, cfg)
    err_ml = max(float(np.max(np.abs(ref[i] - w_only[i]))) for i in range(4))
    print(f"mainline vs encode(X@W.T): max|err|={err_ml:.3e}", flush=True)

    print("engine...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_linear_keys(engine, sk)
    ev = make_qkv_evaluator(engine)
    from bts_ops import DEPTH_QKV, remaining_to_level_calc
    lc = remaining_to_level_calc(DEPTH_QKV + 1, int(engine.num_levels))
    print(f"working lc={lc}", flush=True)
    print("encrypt + encode W...", flush=True)
    x = encrypt_embedding_dualrail(engine, h, pk, level=lc)
    wpt = encode_weight_qkv_dualrail(engine, W, level=lc, scale=1.0, out_packs=4)
    print("HE pcmm...", flush=True)
    qct = qkv_pcmm_he(engine, ev, x, wpt, real_extract=True)
    he = [
        np.asarray(engine.decrode(qct[i], sk, is_real=True), dtype=np.float64).ravel()
        for i in range(4)
    ]
    err = max(float(np.max(np.abs(he[i] - ref[i]))) for i in range(4))
    a = np.concatenate(he)
    b = np.concatenate(ref)
    g = float(np.dot(a, b) / np.dot(a, a))
    corr = float(np.corrcoef(a, b)[0, 1])
    print(
        f"HE vs mainline: max|err|={err:.3e} ls_gain={g:.6g} corr={corr:.6f}",
        flush=True,
    )
    wb = encode_qkv_upper_from_dense(h @ W.T + b.reshape(1, -1), cfg)
    err_b = max(float(np.max(np.abs(he[i] - wb[i]))) for i in range(4))
    print(f"HE (no bias) vs encode(X@W.T+b): max|err|={err_b:.3e}", flush=True)
    print("done", flush=True)
    return 0 if err < 1e-2 and abs(g - 1.0) < 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
