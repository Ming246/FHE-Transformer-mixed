#!/usr/bin/env python3
"""
Real-data att_context probe vs ``poly_model_inference`` intermediates.

Previous ``probe_att_context_rescale.py`` used **random** α/V. This script:

  1. Load GLUE sample + finetuned BERT
  2. ``apply_polynomial_scheme`` + L0 poly forward → ``sftmx_out``, ``v``, ``att_context``
  3. Encrypt poly α (make_copies) + V; run smoke_thor_repro rescale → context path
  4. Decrypt HE ``att_context`` vs poly ``att_context`` (and vs poly α@V)

Usage::

  python3 HE/thor/probe_att_context_vs_poly.py
  python3 HE/thor/probe_att_context_vs_poly.py --dataset mrpc --sample 0 --layer 0
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
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
if _HE in sys.path:
    sys.path.remove(_HE)
sys.path.insert(0, _HE)

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    bind_att_context_mask_complement,
    decode_qkv_upper_to_dense,
    dense_alpha_v_context,
    encode_qkv_upper_from_dense,
    encode_score_packs_from_dense,
    ensure_ct_ct_matmul_masks,
    make_att_context_module,
    make_qkv_evaluator,
    prepare_att_context_keys,
    thor_context_to_mainline_packs,
)

sys.path.insert(0, _HE)
from slot_decode import merge_heads  # noqa: E402
from softmax_he_ct import align_bypass_to_main  # noqa: E402
from thor_encoder_linear_core import bert_base, make_copies  # noqa: E402


def _stats(he: np.ndarray, ref: np.ndarray) -> dict:
    a = np.asarray(he, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    denom = float(np.dot(a, a))
    if denom > 1e-30:
        g = float(np.dot(a, b) / denom)
        err_ls = float(np.max(np.abs(a * g - b)))
    else:
        g = float("nan")
        err_ls = float("inf")
    nz = np.abs(a) > 1e-6
    med = float(np.median(b[nz] / a[nz])) if np.any(nz) else float("nan")
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if np.std(a) > 1e-15 and np.std(b) > 1e-15
        else float("nan")
    )
    return {
        "err_gain1": float(np.max(np.abs(a - b))),
        "ls_gain": g,
        "err_ls": err_ls,
        "corr": corr,
        "median_ratio": med,
        "max_abs_he": float(np.max(np.abs(a))),
        "max_abs_ref": float(np.max(np.abs(b))),
    }


def _print(tag: str, s: dict) -> None:
    print(
        f"  [{tag}]\n"
        f"    err@1={s['err_gain1']:.3e}  ls_gain={s['ls_gain']:.6g}  "
        f"err@LS={s['err_ls']:.3e}  corr={s['corr']:.6f}\n"
        f"    median(ref/he)={s['median_ratio']:.6g}  "
        f"max|he|={s['max_abs_he']:.4g}  max|ref|={s['max_abs_ref']:.4g}",
        flush=True,
    )


def _ok(s: dict, *, err_tol: float = 5e-2) -> bool:
    return (
        np.isfinite(s["corr"])
        and s["corr"] > 0.99
        and abs(s["ls_gain"] - 1.0) < 0.05
        and s["err_gain1"] < err_tol
    )


def _transpose_for_scores(x, attention_m):
    input_shape = x.shape[:-1]
    hidden_shape = (*input_shape, -1, attention_m.attention_head_size)
    return x.view(*hidden_shape).transpose(1, 2)


def _poly_l0_refs_via_encoder(model, enc, *, task: str, scheme_level: int):
    """L0 poly intermediates (same Softmax path as poly_model_inference)."""
    from poly_model_inference import (  # noqa: WPS433
        SCHEME_LEN,
        apply_polynomial_scheme,
        install_eager_attention_poly_patch,
    )

    scheme = [int(scheme_level)] * SCHEME_LEN
    install_eager_attention_poly_patch()
    apply_polynomial_scheme(model, scheme, task)

    with torch.no_grad():
        hs = model.bert.embeddings(
            input_ids=enc["input_ids"],
            token_type_ids=enc.get("token_type_ids"),
        )
        layer = model.bert.encoder.layer[0]
        attention_m = layer.attention.self
        q = _transpose_for_scores(attention_m.query(hs), attention_m)
        k = _transpose_for_scores(attention_m.key(hs), attention_m)
        v = _transpose_for_scores(attention_m.value(hs), attention_m)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            attention_m.attention_head_size
        )
        poly_fn = getattr(attention_m, "_poly_softmax_fn", None)
        if poly_fn is None:
            raise RuntimeError("missing _poly_softmax_fn after apply_polynomial_scheme")
        key_valid = (enc["attention_mask"] > 0).to(dtype=attention_scores.dtype)
        key_valid = key_valid[:, None, None, :].expand_as(attention_scores)
        sftmx_out = poly_fn(attention_scores, dim=-1, key_valid_mask=key_valid)
        att_context = torch.matmul(sftmx_out, v)

    return {
        "sftmx_out": sftmx_out.detach().cpu().numpy().squeeze(0).astype(np.float64),
        "v": v.detach().cpu().numpy().squeeze(0).astype(np.float64),
        "att_context": att_context.detach().cpu().numpy()
        .squeeze(0)
        .astype(np.float64),
        "attention_mask": enc["attention_mask"].detach().cpu().numpy().reshape(-1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--layer", type=int, default=0, choices=(0,))
    ap.add_argument("--softmax-level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--rem", type=int, default=12)
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    cfg = bert_base()
    print(
        f"=== att_context vs poly  {args.dataset} L{args.layer} "
        f"sample={args.sample} sm_level={args.softmax_level} rem={args.rem} ===",
        flush=True,
    )
    print(
        "(prior probe_att_context_rescale used random α/V — this uses real GLUE)",
        flush=True,
    )

    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    local_model = os.path.join(_REPO, "finetuned_weight", args.dataset)
    print("load HF + poly scheme ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model, local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
    ds = load_from_disk(os.path.join(_REPO, "glue_datasets", args.dataset))[
        "validation"
    ]
    row = ds[int(args.sample)]
    if args.dataset in ("mrpc", "rte"):
        enc = tok(
            row["sentence1"],
            row["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    else:
        text = row.get("sentence") or row.get("premise") or row["sentence"]
        enc = tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

    refs = _poly_l0_refs_via_encoder(
        model, enc, task=args.dataset, scheme_level=args.softmax_level
    )
    # HF: sftmx_out [H,Q,K], v [H,Q,D], att_context [H,Q,D]
    alpha_hqk = refs["sftmx_out"]
    alpha_hkq = np.transpose(alpha_hqk, (0, 2, 1))
    v_h = refs["v"]
    ctx_h = refs["att_context"]
    v_dense = merge_heads(v_h, cfg)
    ctx_poly = merge_heads(ctx_h, cfg)
    ctx_matmul = dense_alpha_v_context(alpha_hkq, v_dense, cfg)

    s_poly_internal = _stats(ctx_poly, ctx_matmul)
    print("\npoly sanity: att_context vs sftmx_out@V", flush=True)
    _print("poly att_context vs dense α@V", s_poly_internal)

    kv = refs["attention_mask"] > 0.5
    row_sums = []
    for h in range(cfg.num_heads):
        for q in range(cfg.seq_len):
            row_sums.append(float(alpha_hqk[h, q, kv].sum()))
    row_sums = np.asarray(row_sums)
    print(
        f"poly Softmax row-sum (valid keys): mean={row_sums.mean():.6f}  "
        f"max|sum−1|={float(np.max(np.abs(row_sums - 1.0))):.3e}",
        flush=True,
    )
    print(
        f"|α|_∞={float(np.max(np.abs(alpha_hqk))):.4g}  "
        f"|V|_∞={float(np.max(np.abs(v_dense))):.4g}  "
        f"|ctx|_∞={float(np.max(np.abs(ctx_poly))):.4g}",
        flush=True,
    )

    a_packs = encode_score_packs_from_dense(alpha_hkq, cfg)
    a_copies_pt = make_copies(a_packs, cfg)
    v_packs = encode_qkv_upper_from_dense(v_dense, cfg)

    print("create engine ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_att_context_keys(engine, sk)
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.rem)
    print(f"  ready ({time.time()-t0:.1f}s) enc_lc={enc_lc}", flush=True)

    v_ct = np.full((4,), None, dtype=object)
    for i in range(4):
        v_ct[i] = engine.encodecrypt(
            np.asarray(v_packs[i], dtype=np.float64), pk, level=enc_lc
        )
    sftmx_out = np.full((128,), None, dtype=object)
    for i in range(128):
        sftmx_out[i] = engine.encodecrypt(
            np.asarray(a_copies_pt[i], dtype=np.float64), pk, level=enc_lc
        )

    v_cplx = np.full((2,), None, dtype=object)
    for i in range(2):
        v_cplx[i] = engine.cc_add(v_ct[i], engine.imult(v_ct[i + 2]))
    for j in range(2):
        v_cplx[j] = align_bypass_to_main(engine, sftmx_out[0], v_cplx[j])

    # pre-rescale α decrypt spot-check
    he_a0 = np.asarray(
        engine.decrode(sftmx_out[0], sk, is_real=True), dtype=np.float64
    )
    s_a = _stats(he_a0, a_copies_pt[0])
    print("\n=== pre-rescale α decrypt ===", flush=True)
    _print("α[0] vs poly make_copies[0]", s_a)

    for i in range(2):
        v_cplx[i] = engine.rescale(v_cplx[i])
    sftmx_rs = np.full((128,), None, dtype=object)
    for j in range(128):
        sftmx_rs[j] = engine.rescale(sftmx_out[j])

    print("\n=== calculate_attention_context (outer-rescale path) ===", flush=True)
    evaluator = make_qkv_evaluator(engine)
    ensure_ct_ct_matmul_masks(
        evaluator, engine, n_max=128, level=int(v_cplx[0].level_calc)
    )
    attn = make_att_context_module(engine, evaluator)
    bind_att_context_mask_complement(attn)
    t1 = time.time()
    att_ct = attn.calculate_attention_context(v_cplx, sftmx_rs, rescale=False)
    print(f"  done ({time.time()-t1:.1f}s)", flush=True)

    cplx_dec = [
        np.asarray(engine.decrode(att_ct[i], sk), dtype=np.complex128)
        for i in range(2)
    ]
    mainline = thor_context_to_mainline_packs(cplx_dec, gain=1.0)
    he_ctx = decode_qkv_upper_to_dense(mainline, cfg)

    print("\n=== HE att_context vs poly ===", flush=True)
    s_vs_poly = _stats(he_ctx, ctx_poly)
    _print("HE vs poly att_context (seq,hidden)", s_vs_poly)
    s_vs_matmul = _stats(he_ctx, ctx_matmul)
    _print("HE vs poly α@V matmul", s_vs_matmul)

    ok = _ok(s_vs_poly, err_tol=args.tol) and _ok(s_a, err_tol=1e-2)
    print("\n=== verdict ===", flush=True)
    print(
        f"  pre-α decrypt={'PASS' if _ok(s_a, err_tol=1e-2) else 'FAIL'}  "
        f"HE vs poly att_context={'PASS' if _ok(s_vs_poly, err_tol=args.tol) else 'FAIL'}  "
        f"ls_gain={s_vs_poly['ls_gain']:.6g} err@1={s_vs_poly['err_gain1']:.3e}",
        flush=True,
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
