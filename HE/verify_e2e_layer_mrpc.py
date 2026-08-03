"""
MRPC 端到端：HE 线性 + Softmax/GeLU/LN1/LN2 同构路径 vs 同参数多项式 NumPy 参考。

用法::

  python3 HE/verify_e2e_layer_mrpc.py
  python3 HE/verify_e2e_layer_mrpc.py --sample 0 --level 2
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
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
from gelu_poly import GeluPolyEvaluator
from layernorm_poly import HeLayerNormPolyEvaluator
from slot_gelu_he import GeluHeParams
from slot_layernorm_he import LayerNormHeParams, encode_affine_to_input_lower
from slot_softmax_he import SoftmaxHeParams, encode_key_valid_mask_packs
from softmax_poly import ThorSoftmaxEvaluator

FINETUNED_ROOT = os.path.join(_REPO, "finetuned_weight")


def _ensure_ff_encoded(
    encoded: dict, raw: dict[str, np.ndarray], cfg: core.ThorConfig, enc_path: str
) -> dict:
    need = [k for k in ("w1", "w2") if k not in encoded]
    if not need:
        return encoded
    print(f"encoding missing FF weights {need} ...", flush=True)
    ff = cache.encode_ff_weights(raw, cfg)
    encoded = dict(encoded)
    encoded.update(ff)
    with open(enc_path, "wb") as f:
        pickle.dump(encoded, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved FF into {enc_path}", flush=True)
    return encoded


def _numpy_poly_layer_ref(
    x: np.ndarray,
    raw: dict[str, np.ndarray],
    key_valid: np.ndarray,
    *,
    task: str,
    layer: int,
    level: int,
    gamma1: np.ndarray,
    beta1: np.ndarray,
    gamma2: np.ndarray,
    beta2: np.ndarray,
    eps: float,
    cfg: core.ThorConfig,
) -> np.ndarray:
    """同参数多项式非线性的矩阵参考（排除 HE 打包噪声）。"""
    w_q, w_k, w_v, w_o = raw["w_q"], raw["w_k"], raw["w_v"], raw["w_o"]
    w1, w2 = raw["w1"], raw["w2"]
    q = x @ w_q.T
    k = x @ w_k.T
    v = x @ w_v.T
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores_hkq = np.stack(
        [
            (sd.split_heads(k, cfg)[h] @ sd.split_heads(q, cfg)[h].T) * scale
            for h in range(cfg.num_heads)
        ],
        axis=0,
    )
    scores_hqk = np.transpose(scores_hkq, (0, 2, 1))
    sm = ThorSoftmaxEvaluator(task, layer, level=level)
    mask_t = (
        torch.from_numpy(key_valid)
        .double()
        .view(1, 1, -1)
        .expand(cfg.num_heads, cfg.seq_len, cfg.seq_len)
    )
    with torch.no_grad():
        alpha_hqk = sm(
            torch.from_numpy(scores_hqk).double(), key_valid_mask=mask_t
        ).numpy()
    alpha_hkq = np.transpose(alpha_hqk, (0, 2, 1))
    v_h = sd.split_heads(v, cfg)
    ctx_heads = np.stack(
        [alpha_hkq[h].T @ v_h[h] for h in range(cfg.num_heads)], axis=0
    )
    ctx = sd.merge_heads(ctx_heads, cfg)
    attn_res = x + ctx @ w_o.T

    ln1 = HeLayerNormPolyEvaluator(task, layer, "ln1", level)
    gelu = GeluPolyEvaluator(layer, level)
    ln2 = HeLayerNormPolyEvaluator(task, layer, "ln2", level)
    with torch.no_grad():
        attn_ln = ln1(
            torch.from_numpy(attn_res).double(),
            torch.from_numpy(gamma1).double(),
            torch.from_numpy(beta1).double(),
            eps,
        ).numpy()
        fc1 = attn_ln @ w1.T
        fc1_g = gelu(torch.from_numpy(fc1).double()).numpy()
        ff = fc1_g @ w2.T
        out = ln2(
            torch.from_numpy(attn_ln + ff).double(),
            torch.from_numpy(gamma2).double(),
            torch.from_numpy(beta2).double(),
            eps,
        ).numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HE e2e encoder layer vs poly NumPy")
    parser.add_argument(
        "--cache-dir", default=os.path.join(_HERE, "cache", "mrpc")
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--task", default="mrpc")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="HE slot vs poly NumPy 最大绝对误差容差",
    )
    args = parser.parse_args()

    emb_path = os.path.join(args.cache_dir, "calib_embeddings.npy")
    mask_path = os.path.join(args.cache_dir, "calib_attn_mask.npy")
    mb_path = os.path.join(args.cache_dir, "maskbank.pkl")
    layer_dir = os.path.join(args.cache_dir, "layers", f"{args.layer:02d}")
    enc_path = os.path.join(layer_dir, "encoded.pkl")
    raw_path = os.path.join(layer_dir, "raw.npz")
    for p in (emb_path, mask_path, mb_path, enc_path, raw_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    x = np.load(emb_path)[args.sample].astype(np.float64)
    key_valid = (np.load(mask_path)[args.sample] > 0).astype(np.float64)
    raw = {k: v.astype(np.float64) for k, v in dict(np.load(raw_path)).items()}

    print("loading encoded weights (large)...", flush=True)
    t0 = time.perf_counter()
    with open(enc_path, "rb") as f:
        encoded = pickle.load(f)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s  keys={sorted(encoded.keys())}", flush=True)

    cfg = core.bert_base()
    cfg.validate()
    encoded = _ensure_ff_encoded(encoded, raw, cfg, enc_path)
    masks = cache.load_maskbank(mb_path, cfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(FINETUNED_ROOT, args.task),
        num_labels=2,
        attn_implementation="eager",
    )
    bl = model.bert.encoder.layer[args.layer]
    ln1 = bl.attention.output.LayerNorm
    ln2 = bl.output.LayerNorm
    g1 = ln1.weight.detach().cpu().numpy().astype(np.float64)
    b1 = ln1.bias.detach().cpu().numpy().astype(np.float64)
    g2 = ln2.weight.detach().cpu().numpy().astype(np.float64)
    b2 = ln2.bias.detach().cpu().numpy().astype(np.float64)
    eps = float(ln1.eps)

    sm_p = SoftmaxHeParams.from_softmax_poly(args.task, args.layer, level=args.level)
    gelu_p = GeluHeParams.from_gelu_poly(args.layer, level=args.level)
    ln1_p = LayerNormHeParams.from_layernorm_poly(
        args.task, args.layer, "ln1", level=args.level
    )
    ln2_p = LayerNormHeParams.from_layernorm_poly(
        args.task, args.layer, "ln2", level=args.level
    )
    print(
        f"sample={args.sample} level={args.level} n_valid={int(key_valid.sum())} "
        f"sm_shift={sm_p.shift} gelu={gelu_p.scheme_name} "
        f"ln1_iters={ln1_p.invsqrt_max_iters} ln2_iters={ln2_p.invsqrt_max_iters}",
        flush=True,
    )

    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)
    ln1_gb = encode_affine_to_input_lower(g1, b1, cfg)
    ln2_gb = encode_affine_to_input_lower(g2, b2, cfg)

    print("=== HE cached full layer ===", flush=True)
    t0 = time.perf_counter()
    x_enc = core.encode_input_lower_diagonals(x, cfg)
    y_packs = core.run_encoder_layer_he_cached(
        x_enc,
        encoded,
        cfg,
        masks,
        softmax_he_params=sm_p,
        softmax_mask_packs=mask_packs,
        gelu_he_params=gelu_p,
        ln1_he_params=ln1_p,
        ln2_he_params=ln2_p,
        ln1_gamma_beta=ln1_gb,
        ln2_gamma_beta=ln2_gb,
        ln_eps=eps,
    )
    y_he = sd.decode_input_from_lower_diagonals(y_packs, cfg)
    t_he = time.perf_counter() - t0
    print(f"  HE wall {t_he:.2f}s", flush=True)

    print("=== NumPy poly reference ===", flush=True)
    t0 = time.perf_counter()
    y_ref = _numpy_poly_layer_ref(
        x,
        raw,
        key_valid,
        task=args.task,
        layer=args.layer,
        level=args.level,
        gamma1=g1,
        beta1=b1,
        gamma2=g2,
        beta2=b2,
        eps=eps,
        cfg=cfg,
    )
    print(f"  ref wall {time.perf_counter() - t0:.2f}s", flush=True)

    abs_err = np.abs(y_he - y_ref)
    max_abs = float(np.max(abs_err))
    mean_abs = float(np.mean(abs_err))
    # 仅有效 token 行（可选）
    tok_m = key_valid.astype(bool)
    max_valid = float(np.max(abs_err[tok_m])) if tok_m.any() else max_abs
    print(
        f"HE vs poly-NumPy  max|Δ|={max_abs:.3e}  max|Δ|(valid tok)={max_valid:.3e}  "
        f"mean|Δ|={mean_abs:.3e}",
        flush=True,
    )

    ok = max_abs < args.tol
    print(f"PASS={ok}  (tol={args.tol})", flush=True)
    if not ok:
        i = int(np.argmax(abs_err))
        t, d = divmod(i, cfg.hidden_dim)
        print(
            f"  max at token={t} dim={d} he={y_he[t, d]:.8g} ref={y_ref[t, d]:.8g}",
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
