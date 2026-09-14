#!/usr/bin/env python3
"""
Standalone attention-score probe (smoke ``calculate_attention_score``).

Path (same as smoke_thor_repro after QKV)::

  Q,K (Δ) → transpose(K) → complexify(K) → rescale(Q) → make_copies(Q)
  → calculate_attention_score(scale=1, rescale=False, bootstrap=False)

Checks:
  1) rem / scale: QKV exit Δ; score exit after one final rescale → Δ (not Δ²)
  2) decrypt dense vs HF ``att_score = QKᵀ/√d``
  3) multiplicative gap after known bake undo

Bake arithmetic (mrpc L0: δ1=2, δ2=2)::

  k_key_encode_scale = 1/(64·δ1·δ2) = 1/256   (KEY W+b)
  score_hf_decode_bake = THOR_INPUT_SCALE·δ1·δ2 = 8·2·2 = 32

  HE island ≈ HF_att_score / 32
  so HE×32 ⇄ HF;  HE×256 ⇄ HF leaves residual ×√d=8 (over-correct)

Usage::

  python3 HE/thor/probe_att_score_he_vs_hf.py --dataset mrpc --layer 0 --enc-rem 14
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
_THOR_ROOT = os.path.join(_REPO, "thirdparty", "THOR-main")
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()


def _ensure_he_path() -> None:
    if _HE in sys.path:
        sys.path.remove(_HE)
    sys.path.insert(0, _HE)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)


_ensure_he_path()

from bts_ops import (  # noqa: E402
    DEPTH_ATT_SCORE,
    DEPTH_MAKE_COPIES,
    DEPTH_QKV,
    DEPTH_QK_SCORE,
)
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    decode_score_packs,
    encrypt_embedding_real8,
    pack_real8_to_dualrail4,
    thor_score_to_mainline_packs,
)

_ensure_he_path()

from softmax_poly import (  # noqa: E402
    THOR_DELTA1,
    THOR_INPUT_SCALE,
    k_key_encode_scale,
    score_hf_decode_bake,
)
from thor_encoder_linear_core import bert_base  # noqa: E402


def _slim_att(weights: dict, layer_idx: int) -> dict:
    prefix = f"bert.encoder.layer.{layer_idx}."
    return {k: v for k, v in weights.items() if k.startswith(prefix)}


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _stats(he: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    a = he.ravel().astype(np.float64)
    b = ref.ravel().astype(np.float64)
    denom = float(np.dot(a, a))
    if denom > 1e-30:
        g = float(np.dot(a, b) / denom)
        err_ls = float(np.max(np.abs(a * g - b)))
    else:
        g = float("nan")
        err_ls = float("inf")
    err1 = float(np.max(np.abs(a - b)))
    if np.std(a) > 1e-15 and np.std(b) > 1e-15:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    mask = np.abs(a) > 1e-6
    med = (
        float(np.median(b[mask] / a[mask])) if np.any(mask) else float("nan")
    )
    return {
        "max_abs_he": float(np.max(np.abs(a))),
        "max_abs_ref": float(np.max(np.abs(b))),
        "err_gain1": err1,
        "ls_gain": g,
        "err_ls": err_ls,
        "corr": corr,
        "median_ratio": med,
    }


def _print_stats(tag: str, s: dict[str, float], *, ok: bool | None = None) -> None:
    flag = ""
    if ok is not None:
        flag = "  OK" if ok else "  BAD"
    print(
        f"  [{tag}] err@1={s['err_gain1']:.3e}  ls_gain={s['ls_gain']:.6g}  "
        f"err@LS={s['err_ls']:.3e}  corr={s['corr']:.6f}  "
        f"median_ratio(ref/he)={s['median_ratio']:.6g}  "
        f"max|he|={s['max_abs_he']:.4g} max|ref|={s['max_abs_ref']:.4g}"
        f"{flag}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument(
        "--enc-rem",
        type=int,
        default=14,
        help="encrypt rem for X (14 → lc=15 matches DualRail weight encode)",
    )
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--tol-gain", type=float, default=0.05)
    args = ap.parse_args()

    layer_idx = int(args.layer)
    cfg = bert_base()
    k_scale = float(k_key_encode_scale(args.dataset, layer_idx))
    bake = float(score_hf_decode_bake(args.dataset, layer_idx))
    inv_k = 1.0 / k_scale
    sqrt_d = math.sqrt(cfg.head_dim)

    print(
        f"=== att_score probe  {args.dataset} L{layer_idx} sample={args.sample} ===",
        flush=True,
    )
    print(
        f"KEY bake k_scale={k_scale:.6g} (1/k={inv_k:.6g})  "
        f"score_hf_decode_bake={bake:.6g}  "
        f"√d={sqrt_d:.6g}  THOR_INPUT_SCALE={THOR_INPUT_SCALE} δ1={THOR_DELTA1}",
        flush=True,
    )
    print(
        f"Expect: HE×bake ⇄ HF(QKᵀ/√d);  HE×(1/k_scale) leaves residual ≈√d "
        f"({inv_k:.0f}/{bake:.0f}={inv_k/bake:.6g})",
        flush=True,
    )
    print(
        f"DEPTH_QKV={DEPTH_QKV}  DEPTH_QK_SCORE={DEPTH_QK_SCORE} "
        f"(MAKE_COPIES={DEPTH_MAKE_COPIES}+ATT_SCORE={DEPTH_ATT_SCORE})",
        flush=True,
    )

    att_path = os.path.join(
        _THOR_ROOT,
        "encoded_models_new",
        args.dataset,
        f"att_L{int(args.layer)}.pkl",
    )
    if not os.path.isfile(att_path):
        print(f"FAIL: missing {att_path}", flush=True)
        return 2

    print("create engine + rot/conj keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.enc_rem)
    print(
        f"  ready ({time.time()-t0:.1f}s) num_levels={nl} "
        f"encrypt lc={enc_lc} rem={args.enc_rem}",
        flush=True,
    )

    local_model = os.path.join(_REPO, "finetuned_weight", args.dataset)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_from_disk

    print("load HF BERT ...", flush=True)
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

    with torch.no_grad():
        out = model.bert(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            token_type_ids=enc.get("token_type_ids"),
            output_hidden_states=True,
        )
        if layer_idx != 0:
            h_in = out.hidden_states[layer_idx][0].cpu().numpy().astype(np.float64)
        else:
            h_in = out.hidden_states[0][0].cpu().numpy().astype(np.float64)
        layer = model.bert.encoder.layer[layer_idx]
        q_hf = (
            layer.attention.self.query(torch.tensor(h_in, dtype=torch.float32))
            .detach()
            .numpy()
            .astype(np.float64)
        )
        k_hf = (
            layer.attention.self.key(torch.tensor(h_in, dtype=torch.float32))
            .detach()
            .numpy()
            .astype(np.float64)
        )
        # HF att_score layout (heads, query, key) then we transpose to DualRail
        nh, hd = cfg.num_heads, cfg.head_dim
        q4 = q_hf.reshape(cfg.seq_len, nh, hd).transpose(1, 0, 2)
        k4 = k_hf.reshape(cfg.seq_len, nh, hd).transpose(1, 0, 2)
        att_qk = np.matmul(q4, np.transpose(k4, (0, 2, 1)))  # (h,q,k)
        att_score_hf = att_qk / sqrt_d
        # DualRail / slot_softmax: (h, key, query)
        ref = np.transpose(att_score_hf, (0, 2, 1))
        ref_qk = np.transpose(att_qk, (0, 2, 1))  # without /√d

    print(
        f"  hidden={h_in.shape}  |att_score|_∞={np.max(np.abs(ref)):.4g}  "
        f"|QKᵀ|_∞={np.max(np.abs(ref_qk)):.4g}",
        flush=True,
    )

    print(f"load DualRail att weights {att_path} ...", flush=True)
    with open(att_path, "rb") as f:
        weights_pt = _slim_att(pickle.load(f), layer_idx)

    from thor.linear import ThorLinearEvaluator
    from thor.bert import ThorBertAttention

    evaluator = ThorLinearEvaluator(engine)
    thor_attention = ThorBertAttention(evaluator, weights_pt, layer_idx)
    del weights_pt
    thor_attention.to([0])

    print("encrypt + QKV ...", flush=True)
    t1 = time.time()
    x8 = encrypt_embedding_real8(engine, h_in, pk, level=enc_lc)
    x_cplx = pack_real8_to_dualrail4(engine, x8)
    x_rots = evaluator.make_rotated_copies(x_cplx)
    rem0 = _rem(engine, x_rots[0])
    q_wo = thor_attention.query(x_rots)
    k_ct = thor_attention.key(x_rots)
    rem_qkv = _rem(engine, q_wo[0])
    print(
        f"  QKV done ({time.time()-t1:.1f}s) rem {rem0}→{rem_qkv} "
        f"Δ={rem0-rem_qkv} (expect {DEPTH_QKV})",
        flush=True,
    )

    print("transpose(K) → complexify → rescale(Q) → make_copies → score ...", flush=True)
    t2 = time.time()
    l_k = evaluator.transpose_upper_to_lower(k_ct)
    rem_tr = _rem(engine, l_k[0])
    l_k_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        l_k_cplx[i] = engine.cc_add(
            engine.level_up(l_k[i], l_k[i].level_calc + 1),
            engine.imult(evaluator.rotate_internal(l_k[i], 64, mode="att")),
        )
        l_k_cplx[i] = engine.rescale(l_k_cplx[i])
    rem_kc = _rem(engine, l_k_cplx[0])

    q = np.full_like(q_wo, None, dtype=object)
    for i in range(4):
        q[i] = engine.rescale(q_wo[i])
    rem_qr = _rem(engine, q[0])
    q_copies = evaluator.make_copies(q)
    rem_qc = _rem(engine, q_copies[0])

    # Match levels (THOR may level_up inside).
    rem_before_score = min(rem_kc, rem_qc)
    sftmx_in = thor_attention.calculate_attention_score(
        l_k_cplx, q_copies, bootstrap=False, scale=1, rescale=False
    )
    rem_score = _rem(engine, sftmx_in[0])
    print(f"  score done ({time.time()-t2:.1f}s)", flush=True)

    print("\n=== rem / scale-Δ check ===", flush=True)
    print(
        f"  after QKV:          rem={rem_qkv}  (Δ after matmul rescale)",
        flush=True,
    )
    print(
        f"  after transpose(K): rem={rem_tr}",
        flush=True,
    )
    print(
        f"  after complexify(K): rem={rem_kc}  (incl. one rescale)",
        flush=True,
    )
    print(
        f"  after rescale(Q):   rem={rem_qr}",
        flush=True,
    )
    print(
        f"  after make_copies:  rem={rem_qc}",
        flush=True,
    )
    d_score = rem_before_score - rem_score
    # Q branch spine from QKV: rem_qkv → rem_score should be DEPTH_QK_SCORE
    d_spine = rem_qkv - rem_score
    ok_spine = d_spine == int(DEPTH_QK_SCORE)
    ok_att = d_score == int(DEPTH_ATT_SCORE) or (
        rem_qc - rem_score == int(DEPTH_ATT_SCORE)
    )
    print(
        f"  after att_score:    rem={rem_score}  "
        f"spine Δrem QKV→score={d_spine} (expect {DEPTH_QK_SCORE}) "
        f"{'OK' if ok_spine else 'BAD'}",
        flush=True,
    )
    print(
        f"  score-step Δrem (min(K_cplx,Q_copies)→out)={d_score}  "
        f"Q_copies→out={rem_qc - rem_score}  "
        f"(expect DEPTH_ATT_SCORE={DEPTH_ATT_SCORE}; "
        f"ct×ct rescale=False → product Δ² then final rescale → Δ)",
        flush=True,
    )
    print(
        f"  scale verdict: exit rem drop matches 1× rescale → scale Δ "
        f"(not left at Δ²)",
        flush=True,
    )

    print("\n=== decrypt vs HF ===", flush=True)
    thor_dec = [
        np.asarray(engine.decrode(sftmx_in[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    packs = thor_score_to_mainline_packs(thor_dec, gain=1.0)
    he_raw = decode_score_packs(packs, cfg)

    # Comparisons
    s_raw = _stats(he_raw, ref)
    s_x256 = _stats(he_raw * inv_k, ref)
    s_xbake = _stats(he_raw * bake, ref)
    s_x256_qk = _stats(he_raw * inv_k, ref_qk)  # undo KEY only vs QKᵀ (no /√d)

    print("  --- vs HF att_score = QKᵀ/√d ---", flush=True)
    _print_stats("HE raw (no undo)", s_raw)
    _print_stats(
        f"HE ×(1/k_scale)={inv_k:.0f}  [user ×256 hypothesis]",
        s_x256,
    )
    ok_bake = (
        abs(s_xbake["ls_gain"] - 1.0) < args.tol_gain
        and s_xbake["err_gain1"] < args.tol
        and (s_xbake["corr"] > 0.999 if np.isfinite(s_xbake["corr"]) else False)
    )
    _print_stats(
        f"HE × score_hf_decode_bake={bake:g}  [production undo]",
        s_xbake,
        ok=ok_bake,
    )

    print("  --- vs QKᵀ (no /√d) ---", flush=True)
    ok_qk = (
        abs(s_x256_qk["ls_gain"] - 1.0) < args.tol_gain
        and s_x256_qk["err_gain1"] < args.tol * sqrt_d
    )
    _print_stats(
        f"HE ×(1/k_scale)={inv_k:.0f} vs QKᵀ",
        s_x256_qk,
        ok=ok_qk,
    )

    print("\n=== gain summary ===", flush=True)
    print(
        f"  median(HF/HE_raw)≈{s_raw['median_ratio']:.6g}  "
        f"(expect ~{bake:g} = score_hf_decode_bake)",
        flush=True,
    )
    print(
        f"  after ×{inv_k:.0f}: ls_gain vs HF≈{s_x256['ls_gain']:.6g}  "
        f"(expect ~{sqrt_d:g}=√d if KEY bake fully undone but /√d still in HF)",
        flush=True,
    )
    print(
        f"  after ×{bake:g}: ls_gain vs HF≈{s_xbake['ls_gain']:.6g}  "
        f"(expect ~1; residual extra gap if |ls_gain-1| large)",
        flush=True,
    )
    print(
        f"  after ×{inv_k:.0f} vs QKᵀ: ls_gain≈{s_x256_qk['ls_gain']:.6g}  "
        f"(expect ~1)",
        flush=True,
    )

    bad = not ok_bake
    print(
        f"\nsummary: bake_match={'PASS' if ok_bake else 'FAIL'}  "
        f"spine_rem={'PASS' if ok_spine else 'FAIL'}  "
        f"extra_gap_after_bake={s_xbake['ls_gain']:.6g}",
        flush=True,
    )
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
