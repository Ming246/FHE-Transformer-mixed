#!/usr/bin/env python3
"""
Plaintext DualRail twins for ``smoke_thor_repro`` sites vs HF BERT (layer 0).

No Liberate. Reports abs max|err| / corr at:

  after_att_score / sftmx_in   (score; bootstrap N/A in plain)
  att_context                  (exact Softmax α + DualRail context island)
  ln1_out                      (W_O DualRail + resid + exact LN)
  ln2_out                      (exact GeLU + FF DualRail + resid + exact LN)

Production score contract: KEY bake ``1/(64·δ1·δ2)``;
score decode bake ``8·δ1·δ2`` (DualRail island already ×8 vs naive QK).

Usage::

  python3 HE/thor/audit_repro_sites_plain.py --sample 0
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from linear_eval import (  # noqa: E402
    complexify_v_plain,
    compute_attention_context_dualrail_island,
    compute_attention_score_dualrail_island,
    compute_ff_dualrail_plain,
    compute_wo_dualrail_plain,
    decode_qkv_upper_to_dense,
    decode_score_packs,
    encode_qkv_upper_from_dense,
    encode_score_packs_from_dense,
    mainline_qkv_ref,
)

sys.path.insert(0, _HE)
from slot_decode import (  # noqa: E402
    decode_input_from_lower_diagonals,
    merge_heads,
)
from thor_encoder_linear_core import (  # noqa: E402
    bert_base,
    encode_input_lower_diagonals,
)


def _stats(name: str, got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    g = np.asarray(got, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    err = float(np.max(np.abs(g - r)))
    if np.std(g) > 1e-15 and np.std(r) > 1e-15:
        corr = float(np.corrcoef(g, r)[0, 1])
    else:
        corr = float("nan")
    print(
        f"  [{name}] max|err|={err:.3e}  corr={corr:.6f}  "
        f"max|got|={float(np.max(np.abs(g))):.4g}  max|ref|={float(np.max(np.abs(r))):.4g}",
        flush=True,
    )
    return err, corr


def _transpose_for_scores(x: torch.Tensor, attention_m) -> torch.Tensor:
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attention_m.attention_head_size)
    return x.view(*hidden_shape).transpose(1, 2)


def _load_sample(dataset: str, sample: int, model_dir: str):
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_dir)
    ds = load_from_disk(os.path.join(_REPO, "glue_datasets", dataset))
    ex = ds["validation"][int(sample)]
    if "sentence1" in ex:
        enc = tok(
            ex["sentence1"],
            ex["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    else:
        enc = tok(
            ex["sentence"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    return model, enc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rest-div", type=float, default=1.0)
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    layer_idx = int(args.layer)
    rest_div = float(args.rest_div)
    sys.path.insert(0, _REPO)
    from softmax_poly import k_key_encode_scale, score_hf_decode_bake  # noqa: E402

    bake = score_hf_decode_bake(args.dataset, layer_idx)
    k_scale = k_key_encode_scale(args.dataset, layer_idx)

    model_dir = os.path.join(_REPO, "finetuned_weight", args.dataset)
    print(
        f"=== repro sites PLAIN twin (L{layer_idx} sample={args.sample}) ===",
        flush=True,
    )
    print(
        f"score: KEY×{k_scale:g}, scale=1, rest_div={rest_div:g}, bake×{bake:g}",
        flush=True,
    )

    t0 = time.time()
    model, batch = _load_sample(args.dataset, args.sample, model_dir)
    layer = model.bert.encoder.layer[layer_idx]
    att = layer.attention.self
    with torch.no_grad():
        hs = model.bert.embeddings(
            input_ids=batch["input_ids"],
            token_type_ids=batch.get("token_type_ids"),
        )
        # Walk to layer_idx if needed
        h = hs
        for li in range(layer_idx):
            h = model.bert.encoder.layer[li](h)[0]
        q = _transpose_for_scores(att.query(h), att)
        k = _transpose_for_scores(att.key(h), att)
        v = _transpose_for_scores(att.value(h), att)
        scores_hf = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            att.attention_head_size
        )
        # mask for softmax only
        am = batch["attention_mask"]
        ext = (1.0 - am[:, None, None, :].to(dtype=h.dtype)) * torch.finfo(
            h.dtype
        ).min
        sftmx_in = scores_hf + ext
        sftmx_out = torch.nn.functional.softmax(sftmx_in, dim=-1)
        att_ctx = torch.matmul(sftmx_out, v)
        ctx_merge = (
            att_ctx.permute(0, 2, 1, 3)
            .contiguous()
            .view(att_ctx.size(0), att_ctx.size(2), -1)
        )
        dense_wo = layer.attention.output.dense(ctx_merge)
        ln1_in = dense_wo + h
        ln1_out = layer.attention.output.LayerNorm(ln1_in)
        gelu_in = layer.intermediate.dense(ln1_out)
        gelu_out = layer.intermediate.intermediate_act_fn(gelu_in)
        dense2 = layer.output.dense(gelu_out)
        ln2_in = dense2 + ln1_out
        ln2_out = layer.output.LayerNorm(ln2_in)

    x = h.cpu().numpy().squeeze().astype(np.float64)
    q_np = q.cpu().numpy().squeeze().astype(np.float64)  # (h,seq,d)
    k_np = k.cpu().numpy().squeeze().astype(np.float64)
    v_np = v.cpu().numpy().squeeze().astype(np.float64)
    scores_ref = scores_hf.cpu().numpy().squeeze().astype(np.float64)  # (h,q,k)
    alpha_hf = sftmx_out.cpu().numpy().squeeze().astype(np.float64)  # (h,q,k)
    ctx_hf = att_ctx.cpu().numpy().squeeze().astype(np.float64)  # (h,q,d)
    ln1_ref = ln1_out.cpu().numpy().squeeze().astype(np.float64)
    ln2_ref = ln2_out.cpu().numpy().squeeze().astype(np.float64)
    gelu_out_np = gelu_out.cpu().numpy().squeeze().astype(np.float64)
    print(f"  HF refs ready ({time.time()-t0:.1f}s)", flush=True)

    Wq = att.query.weight.detach().cpu().numpy().astype(np.float64)
    Wk = att.key.weight.detach().cpu().numpy().astype(np.float64)
    Wv = att.value.weight.detach().cpu().numpy().astype(np.float64)
    bq = att.query.bias.detach().cpu().numpy().astype(np.float64)
    bk = att.key.bias.detach().cpu().numpy().astype(np.float64)
    bv = att.value.bias.detach().cpu().numpy().astype(np.float64)
    Wo = layer.attention.output.dense.weight.detach().cpu().numpy().astype(np.float64)
    bo = layer.attention.output.dense.bias.detach().cpu().numpy().astype(np.float64)
    W1 = layer.intermediate.dense.weight.detach().cpu().numpy().astype(np.float64)
    b1 = layer.intermediate.dense.bias.detach().cpu().numpy().astype(np.float64)
    W2 = layer.output.dense.weight.detach().cpu().numpy().astype(np.float64)
    b2 = layer.output.dense.bias.detach().cpu().numpy().astype(np.float64)
    ln1_w = layer.attention.output.LayerNorm.weight.detach().cpu().numpy().astype(
        np.float64
    )
    ln1_b = layer.attention.output.LayerNorm.bias.detach().cpu().numpy().astype(
        np.float64
    )
    ln2_w = layer.output.LayerNorm.weight.detach().cpu().numpy().astype(np.float64)
    ln2_b = layer.output.LayerNorm.bias.detach().cpu().numpy().astype(np.float64)
    eps1 = float(layer.attention.output.LayerNorm.eps)
    eps2 = float(layer.output.LayerNorm.eps)

    def exact_ln(y, weight, bias, eps):
        mu = y.mean(axis=-1, keepdims=True)
        var = y.var(axis=-1, keepdims=True)
        return weight * (y - mu) / np.sqrt(var + eps) + bias

    # ------------------------------------------------------------------
    # [1] after_att_score / sftmx_in (plain; bts is HE-only → same tensor)
    # ------------------------------------------------------------------
    print("\n--- after_att_score / sftmx_in (plain; bts N/A) ---", flush=True)
    t1 = time.time()
    # A: HF Q/K with KEY bake (includes bias already in HF Q/K)
    q_dense = merge_heads(q_np, cfg)
    k_dense = merge_heads(k_np * k_scale, cfg)  # bake into K values
    # Bias: HF K already has bk; scaling K by k_scale scales activations including
    # the linear map — THOR bakes scale into W and b. Match THOR: K = X@(sW).T + s*b
    # HF k = X@W.T+b → we need s*(X@W.T+b) = s*k_hf. Yes k_np * k_scale is correct.
    qp = encode_qkv_upper_from_dense(q_dense, cfg)
    kp = encode_qkv_upper_from_dense(k_dense, cfg)
    island = compute_attention_score_dualrail_island(
        qp, kp, cfg, scale=1.0, gain=1.0, rest_div=rest_div
    )
    scores_got = decode_score_packs(island, cfg) * bake  # (h,key,query)
    scores_thor = np.transpose(scores_ref, (0, 2, 1))
    e_a, c_a = _stats("A HF-Q/K→score×bake vs HF att_score", scores_got, scores_thor)

    # B: emb → pure-real PC-MM Q/K (+ bias) → score (full linear spine)
    q_pc = mainline_qkv_ref(x, Wq, scale=1.0, out_packs=4)
    k_pc = mainline_qkv_ref(x, Wk, scale=k_scale, out_packs=4)
    # add bias into upper packs via dense round-trip
    q_pc_d = decode_qkv_upper_to_dense(q_pc, cfg) + bq
    k_pc_d = decode_qkv_upper_to_dense(k_pc, cfg) + (k_scale * bk)
    qp_b = encode_qkv_upper_from_dense(q_pc_d, cfg)
    kp_b = encode_qkv_upper_from_dense(k_pc_d, cfg)
    island_b = compute_attention_score_dualrail_island(
        qp_b, kp_b, cfg, scale=1.0, gain=1.0, rest_div=rest_div
    )
    scores_b = decode_score_packs(island_b, cfg) * bake
    e_b, c_b = _stats(
        "B emb→PC-MM+Q/K→score×bake vs HF att_score", scores_b, scores_thor
    )
    # PC-MM Q vs HF Q (sanity)
    _stats("  (sanity) PC-MM Q+bq vs HF Q", q_pc_d, q_dense)
    print(f"  score block ({time.time()-t1:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # [2] att_context — exact HF Softmax α + DualRail context island
    # ------------------------------------------------------------------
    print("\n--- att_context (exact Softmax α + DualRail context) ---", flush=True)
    t2 = time.time()
    v_dense = merge_heads(v_np, cfg)
    # alpha DualRail: (h,key,query); HF is (h,query,key)
    alpha_thor = np.transpose(alpha_hf, (0, 2, 1))
    a_packs = encode_score_packs_from_dense(alpha_thor, cfg)
    vp = encode_qkv_upper_from_dense(v_dense, cfg)
    ctx_packs = compute_attention_context_dualrail_island(
        a_packs, vp, cfg, scale=1.0, gain=1.0, alpha_copy_mode="make_copies"
    )
    ctx_dense = decode_qkv_upper_to_dense(ctx_packs, cfg)  # (seq, hidden)
    ctx_ref = merge_heads(ctx_hf, cfg)  # (seq, hidden) from (h,q,d)
    e_c, c_c = _stats("att_context island vs HF", ctx_dense, ctx_ref)
    print(f"  context block ({time.time()-t2:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # [3] ln1_out — DualRail W_O + bias + resid + exact LN
    # ------------------------------------------------------------------
    print("\n--- ln1_out (DualRail W_O + resid + exact LN) ---", flush=True)
    t3 = time.time()
    # Use HF context for fair W_O-only; also report chain from island context
    ctx_upper = encode_qkv_upper_from_dense(ctx_ref, cfg)
    ctx_cplx = complexify_v_plain(ctx_upper)
    wo_packs = compute_wo_dualrail_plain(ctx_cplx, Wo, cfg, scale=1.0, out_packs=8)
    wo_dense = decode_input_from_lower_diagonals(wo_packs, cfg) + bo
    ln1_in_p = wo_dense + x
    ln1_p = exact_ln(ln1_in_p, ln1_w, ln1_b, eps1)
    e_l1, c_l1 = _stats("ln1 (HF-ctx→W_O) vs HF", ln1_p, ln1_ref)

    ctx_upper_i = encode_qkv_upper_from_dense(ctx_dense, cfg)
    wo_i = compute_wo_dualrail_plain(
        complexify_v_plain(ctx_upper_i), Wo, cfg, scale=1.0, out_packs=8
    )
    ln1_chain = exact_ln(
        decode_input_from_lower_diagonals(wo_i, cfg) + bo + x, ln1_w, ln1_b, eps1
    )
    e_l1c, c_l1c = _stats("ln1 (island-ctx→W_O chain) vs HF", ln1_chain, ln1_ref)
    print(f"  ln1 block ({time.time()-t3:.1f}s)", flush=True)

    # ------------------------------------------------------------------
    # [4] ln2 — DualRail FF packing (no GeLU/bias) + dense exact ln2 sanity
    # ------------------------------------------------------------------
    print(
        "\n--- ln2_out (FF DualRail packing + dense exact GeLU sanity) ---",
        flush=True,
    )
    t4 = time.time()
    ln1_packs = list(encode_input_lower_diagonals(ln1_ref, cfg))
    # Same contract as audit_linear_ff_gate: no GeLU, no bias.
    ff_no_gelu = compute_ff_dualrail_plain(
        ln1_packs, W1, W2, cfg, scale1=1.0, scale2=1.0, out_packs=8
    )
    ff_dense_no_gelu = decode_input_from_lower_diagonals(ff_no_gelu, cfg)
    dense_no_gelu = (ln1_ref @ W1.T) @ W2.T
    e_ff, c_ff = _stats(
        "FF DualRail (no GeLU/bias) vs dense (ln1@W1.T)@W2.T",
        ff_dense_no_gelu,
        dense_no_gelu,
    )

    ln2_dense = exact_ln(
        gelu_out_np @ W2.T + b2 + ln1_ref, ln2_w, ln2_b, eps2
    )
    e_l2d, c_l2d = _stats("ln2 dense(exact GeLU+FC2+LN) vs HF", ln2_dense, ln2_ref)
    print(
        "  note: full ln2 DualRail needs GeLU between dense1/dense2; "
        "HE poly GeLU is approx — packing checked without GeLU.",
        flush=True,
    )
    print(f"  ln2 block ({time.time()-t4:.1f}s)", flush=True)

    print("\n=== summary (PLAIN twin vs HF / dense) ===", flush=True)
    print(
        f"  after_att_score A (HF Q/K→score×{bake:g}): "
        f"max|err|={e_a:.3e}  corr={c_a:.6f}",
        flush=True,
    )
    print(
        f"  after_att_score B (PC-MM→score×{bake:g}): "
        f"max|err|={e_b:.3e}  corr={c_b:.6f}",
        flush=True,
    )
    print(
        "  sftmx_in after DualRail bts: N/A in plain "
        "(identical to after_att_score)",
        flush=True,
    )
    print(
        f"  att_context (exact α + DualRail): max|err|={e_c:.3e}  corr={c_c:.6f}",
        flush=True,
    )
    print(
        f"  ln1_out (W_O DualRail + exact LN): max|err|={e_l1:.3e}  corr={c_l1:.6f}",
        flush=True,
    )
    print(
        f"  ln1_out chain (ctx island→W_O): max|err|={e_l1c:.3e}  corr={c_l1c:.6f}",
        flush=True,
    )
    print(
        f"  FF DualRail packing (no GeLU): max|err|={e_ff:.3e}  corr={c_ff:.6f}",
        flush=True,
    )
    print(
        f"  ln2 dense sanity (exact GeLU): max|err|={e_l2d:.3e}  corr={c_l2d:.6f}",
        flush=True,
    )

    ok = (
        e_a < 1e-4
        and e_b < 1e-4
        and e_c < 1e-4
        and e_l1 < 1e-4
        and e_l1c < 1e-4
        and e_ff < 1e-4
        and e_l2d < 1e-4
    )
    print("PASS" if ok else "FAIL (see errs above)", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
