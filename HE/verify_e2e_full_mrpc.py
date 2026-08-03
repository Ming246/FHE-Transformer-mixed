"""
完整 12 层 HE 同构 BERT 推理正确性（按层流式加载，避免 ~120GB 同时驻留）：

  embeddings → 12×Encoder(HE) → decode → 明文 Pooler(tanh)+Classifier → logits

与同参数多项式 NumPy 全流程对拍。

用法::

  python3 HE/verify_e2e_full_mrpc.py --sample 0 --level 2
"""
from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time

import numpy as np
from transformers import AutoModelForSequenceClassification

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import prepare_mrpc_cache as cache
import slot_decode as sd
import thor_encoder_linear_core as core
from slot_gelu_he import GeluHeParams
from slot_layernorm_he import LayerNormHeParams, encode_affine_to_input_lower
from slot_softmax_he import SoftmaxHeParams, encode_key_valid_mask_packs
from verify_e2e_layer_mrpc import _numpy_poly_layer_ref

FINETUNED_ROOT = os.path.join(_REPO, "finetuned_weight")
NUM_ENCODER_LAYERS = 12


def plaintext_classify(
    hidden: np.ndarray,
    head: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    cls = hidden[0]
    pooled = np.tanh(cls @ head["pool_w"].T + head["pool_b"])
    logits = pooled @ head["clf_w"].T + head["clf_b"]
    return logits, pooled


def _load_head(cache_dir: str, model) -> dict[str, np.ndarray]:
    path = os.path.join(cache_dir, "head.npz")
    if os.path.isfile(path):
        return {k: v.astype(np.float64) for k, v in dict(np.load(path)).items()}
    pool, clf = model.bert.pooler.dense, model.classifier
    payload = {
        "pool_w": pool.weight.detach().cpu().numpy().astype(np.float64),
        "pool_b": pool.bias.detach().cpu().numpy().astype(np.float64),
        "clf_w": clf.weight.detach().cpu().numpy().astype(np.float64),
        "clf_b": clf.bias.detach().cpu().numpy().astype(np.float64),
    }
    np.savez_compressed(path, **payload)
    return payload


def _layer_meta(model, layer_idx: int, task: str, level: int, cfg: core.ThorConfig) -> dict:
    bl = model.bert.encoder.layer[layer_idx]
    ln1, ln2 = bl.attention.output.LayerNorm, bl.output.LayerNorm
    g1 = ln1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = ln1.bias.detach().cpu().numpy().astype(np.float64)
    g2 = ln2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = ln2.bias.detach().cpu().numpy().astype(np.float64)
    return {
        "softmax_he_params": SoftmaxHeParams.from_softmax_poly(
            task, layer_idx, level=level
        ),
        "gelu_he_params": GeluHeParams.from_gelu_poly(layer_idx, level=level),
        "ln1_he_params": LayerNormHeParams.from_layernorm_poly(
            task, layer_idx, "ln1", level=level
        ),
        "ln2_he_params": LayerNormHeParams.from_layernorm_poly(
            task, layer_idx, "ln2", level=level
        ),
        "ln1_gamma_beta": encode_affine_to_input_lower(g1, b1, cfg),
        "ln2_gamma_beta": encode_affine_to_input_lower(g2, b2, cfg),
        "gamma1": g1,
        "beta1": b1,
        "gamma2": g2,
        "beta2": b2,
        "eps": float(ln1.eps),
    }


def _load_encoded(enc_path: str) -> dict:
    t0 = time.perf_counter()
    with open(enc_path, "rb") as f:
        encoded = pickle.load(f)
    print(f"    encoded loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    need = [k for k in ("w_q", "w_k", "w_v", "w_o", "w1", "w2") if k not in encoded]
    if need:
        raise FileNotFoundError(f"{enc_path} 缺少 {need}；请先 prepare attn,ff")
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full 12-layer HE BERT + plaintext head (streaming)"
    )
    parser.add_argument(
        "--cache-dir", default=os.path.join(_HERE, "cache", "mrpc")
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--task", default="mrpc")
    parser.add_argument("--num-layers", type=int, default=NUM_ENCODER_LAYERS)
    parser.add_argument("--tol-hidden", type=float, default=1e-7)
    parser.add_argument("--tol-logits", type=float, default=1e-6)
    args = parser.parse_args()
    if not (1 <= args.num_layers <= 12):
        raise ValueError("--num-layers 须在 1..12")

    emb_path = os.path.join(args.cache_dir, "calib_embeddings.npy")
    mask_path = os.path.join(args.cache_dir, "calib_attn_mask.npy")
    mb_path = os.path.join(args.cache_dir, "maskbank.pkl")
    for p in (emb_path, mask_path, mb_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    x0 = np.load(emb_path)[args.sample].astype(np.float64)
    key_valid = (np.load(mask_path)[args.sample] > 0).astype(np.float64)
    labels_path = os.path.join(args.cache_dir, "calib_labels.npy")
    label = int(np.load(labels_path)[args.sample]) if os.path.isfile(labels_path) else -1

    cfg = core.bert_base()
    cfg.validate()
    print("loading MaskBank...", flush=True)
    masks = cache.load_maskbank(mb_path, cfg)

    print("loading HF model (LN γ/β + head)...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(FINETUNED_ROOT, args.task),
        num_labels=2,
        attn_implementation="eager",
    )
    model.eval()
    head = _load_head(args.cache_dir, model)

    metas = [
        _layer_meta(model, li, args.task, args.level, cfg)
        for li in range(args.num_layers)
    ]
    del model
    gc.collect()

    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)
    ln_eps = metas[0]["eps"]

    # --- HE path: stream one encoded layer at a time ---
    print(
        f"=== HE stream {args.num_layers} layers→decode→head "
        f"sample={args.sample} level={args.level} ===",
        flush=True,
    )
    t0 = time.perf_counter()
    x_enc = core.encode_input_lower_diagonals(x0, cfg)
    for li in range(args.num_layers):
        enc_path = os.path.join(
            args.cache_dir, "layers", f"{li:02d}", "encoded.pkl"
        )
        print(f"  HE layer {li}/{args.num_layers - 1} ...", flush=True)
        encoded = _load_encoded(enc_path)
        m = metas[li]
        t_l = time.perf_counter()
        x_enc = core.run_encoder_layer_he_cached(
            x_enc,
            encoded,
            cfg,
            masks,
            softmax_he_params=m["softmax_he_params"],
            softmax_mask_packs=mask_packs,
            gelu_he_params=m["gelu_he_params"],
            ln1_he_params=m["ln1_he_params"],
            ln2_he_params=m["ln2_he_params"],
            ln1_gamma_beta=m["ln1_gamma_beta"],
            ln2_gamma_beta=m["ln2_gamma_beta"],
            ln_eps=ln_eps,
        )
        print(f"    layer wall {time.perf_counter() - t_l:.1f}s", flush=True)
        del encoded
        gc.collect()

    hidden_he = sd.decode_input_from_lower_diagonals(x_enc, cfg)
    logits_he, pooled_he = plaintext_classify(hidden_he, head)
    print(f"  HE+head total {time.perf_counter() - t0:.1f}s", flush=True)

    # --- NumPy poly reference (only raw.npz, small) ---
    print("=== NumPy poly stack + plaintext head ===", flush=True)
    t0 = time.perf_counter()
    y = x0
    for li in range(args.num_layers):
        raw_path = os.path.join(
            args.cache_dir, "layers", f"{li:02d}", "raw.npz"
        )
        raw = {k: v.astype(np.float64) for k, v in dict(np.load(raw_path)).items()}
        m = metas[li]
        y = _numpy_poly_layer_ref(
            y,
            raw,
            key_valid,
            task=args.task,
            layer=li,
            level=args.level,
            gamma1=m["gamma1"],
            beta1=m["beta1"],
            gamma2=m["gamma2"],
            beta2=m["beta2"],
            eps=m["eps"],
            cfg=cfg,
        )
    hidden_ref = y
    logits_ref, pooled_ref = plaintext_classify(hidden_ref, head)
    print(f"  ref wall {time.perf_counter() - t0:.1f}s", flush=True)

    err_h = float(np.max(np.abs(hidden_he - hidden_ref)))
    err_p = float(np.max(np.abs(pooled_he - pooled_ref)))
    err_l = float(np.max(np.abs(logits_he - logits_ref)))
    pred_he = int(np.argmax(logits_he))
    pred_ref = int(np.argmax(logits_ref))

    print(
        f"hidden max|Δ|={err_h:.3e}  pooled max|Δ|={err_p:.3e}  "
        f"logits max|Δ|={err_l:.3e}",
        flush=True,
    )
    print(f"logits_he={logits_he}  logits_ref={logits_ref}", flush=True)
    print(
        f"pred_he={pred_he} pred_ref={pred_ref}"
        + (f" label={label}" if label >= 0 else ""),
        flush=True,
    )

    ok = (
        err_h < args.tol_hidden
        and err_l < args.tol_logits
        and pred_he == pred_ref
    )
    print(
        f"PASS={ok}  (tol_hidden={args.tol_hidden} tol_logits={args.tol_logits})",
        flush=True,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
