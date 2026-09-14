#!/usr/bin/env python3
"""
Standalone QKV probe (smoke_thor_repro query/key/value).

Checks:
  1) rem drop across Q/K/V (== DEPTH_QKV=2 ⇒ one matmul rescale → scale Δ)
  2) decrypt → dense vs HF Linear, after undoing known DualRail bakes
  3) whether a residual global gain remains (ls_gain) after bake undo

Known bakes (should cancel or be undone explicitly):
  - W and bias encode ``/2``; ``cc_add(·, conjugate(·))`` = 2·Re → cancels
  - KEY only: ``k_key_encode_scale = 1/(64·δ1·δ2)`` on W+b → undo by ``HE/k_scale``

Usage::

  python3 HE/thor/probe_qkv_he_vs_hf.py --dataset mrpc --layer 0 --sample 0
"""
from __future__ import annotations

import argparse
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

from bts_ops import DEPTH_QKV  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    decode_qkv_upper_to_dense,
    encrypt_embedding_real8,
    pack_real8_to_dualrail4,
)

_ensure_he_path()

from softmax_poly import k_key_encode_scale  # noqa: E402
from thor_encoder_linear_core import bert_base  # noqa: E402


def _slim_att(weights: dict, layer_idx: int) -> dict:
    prefix = f"bert.encoder.layer.{layer_idx}."
    return {k: v for k, v in weights.items() if k.startswith(prefix)}


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
    return {
        "max_abs_he": float(np.max(np.abs(a))),
        "max_abs_ref": float(np.max(np.abs(b))),
        "err_gain1": err1,
        "ls_gain": g,
        "err_ls": err_ls,
        "corr": corr,
        "median_ratio": float(
            np.median(b[np.abs(a) > 1e-6] / a[np.abs(a) > 1e-6])
        )
        if np.any(np.abs(a) > 1e-6)
        else float("nan"),
    }


def _decrypt_qkv_dense(engine, sk, cts, cfg) -> np.ndarray:
    packs = []
    for i in range(4):
        packs.append(
            np.asarray(
                engine.decrode(cts[i], sk, is_real=True), dtype=np.float64
            ).ravel()
        )
    return decode_qkv_upper_to_dense(packs, cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--enc-rem", type=int, default=16, help="encrypt rem for X")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--tol-gain", type=float, default=0.05, help="|ls_gain-1| ok")
    args = ap.parse_args()

    layer_idx = int(args.layer)
    cfg = bert_base()
    k_scale = float(k_key_encode_scale(args.dataset, layer_idx))
    print(
        f"=== QKV probe  {args.dataset} L{layer_idx} sample={args.sample} ===",
        flush=True,
    )
    print(
        f"DEPTH_QKV={DEPTH_QKV}  KEY bake k_scale={k_scale:.6g}  "
        f"(undo HE_K /= k_scale vs HF)",
        flush=True,
    )
    print(
        "DualRail W+b encode /2; +conjugate → 2·Re (should cancel, no undo)",
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
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    print("load HF BERT ...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model, local_files_only=True, attn_implementation="eager"
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    # Same GLUE sample path as smoke (validation[sample]).
    from datasets import load_from_disk

    ds_path = os.path.join(_REPO, "glue_datasets", args.dataset)
    ds = load_from_disk(ds_path)["validation"]
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
        # L0 entry = embedding output = hidden_states[0]
        h0 = out.hidden_states[0][0].cpu().numpy().astype(np.float64)
        layer = model.bert.encoder.layer[layer_idx]
        # For layer>0 would need hidden_states[layer]; this probe defaults L0.
        if layer_idx != 0:
            h_in = out.hidden_states[layer_idx][0].cpu().numpy().astype(np.float64)
        else:
            h_in = h0
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
        v_hf = (
            layer.attention.self.value(torch.tensor(h_in, dtype=torch.float32))
            .detach()
            .numpy()
            .astype(np.float64)
        )

    print(
        f"  hidden in shape={h_in.shape}  "
        f"|q|_∞={np.max(np.abs(q_hf)):.4g} |k|_∞={np.max(np.abs(k_hf)):.4g} "
        f"|v|_∞={np.max(np.abs(v_hf)):.4g}",
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
    thor_attention.to([0])  # CPU pickle tensors → cuda:0 (THOR Module.to)

    print("encrypt embedding (8 real → 4 cplx → rotated) ...", flush=True)
    x8 = encrypt_embedding_real8(engine, h_in, pk, level=enc_lc)
    x_cplx = pack_real8_to_dualrail4(engine, x8)
    rem_entry = nl - int(x_cplx[0].level_calc)
    x_rots = evaluator.make_rotated_copies(x_cplx)
    rem_rots = nl - int(x_rots[0].level_calc)
    print(
        f"  entry_cplx rem={rem_entry}  after_make_rotated rem={rem_rots} "
        f"(rotate should not burn rem)",
        flush=True,
    )

    print("HE query / key / value ...", flush=True)
    t1 = time.time()
    q_ct = thor_attention.query(x_rots)
    rem_q = nl - int(q_ct[0].level_calc)
    k_ct = thor_attention.key(x_rots)
    rem_k = nl - int(k_ct[0].level_calc)
    v_ct = thor_attention.value(x_rots)
    rem_v = nl - int(v_ct[0].level_calc)
    print(f"  done ({time.time()-t1:.1f}s)", flush=True)

    print("\n=== rem / scale-Δ check ===", flush=True)
    for name, rem in (("Q", rem_q), ("K", rem_k), ("V", rem_v)):
        d = rem_rots - rem
        ok = d == int(DEPTH_QKV)
        print(
            f"  {name}: rem {rem_rots}→{rem}  Δrem={d}  "
            f"declared DEPTH_QKV={DEPTH_QKV}  "
            f"{'OK (matmul rescale→Δ)' if ok else 'BAD'}",
            flush=True,
        )

    print("\n=== decrypt vs HF (after bake undo) ===", flush=True)
    q_packs = [
        np.asarray(engine.decrode(q_ct[i], sk, is_real=True), dtype=np.float64).ravel()
        for i in range(4)
    ]
    k_packs = [
        np.asarray(engine.decrode(k_ct[i], sk, is_real=True), dtype=np.float64).ravel()
        for i in range(4)
    ]
    v_packs = [
        np.asarray(engine.decrode(v_ct[i], sk, is_real=True), dtype=np.float64).ravel()
        for i in range(4)
    ]
    q_he = decode_qkv_upper_to_dense(q_packs, cfg)
    k_he = decode_qkv_upper_to_dense(k_packs, cfg)
    v_he = decode_qkv_upper_to_dense(v_packs, cfg)

    from linear_eval import encode_qkv_upper_from_dense  # noqa: WPS433

    q_ref_packs = encode_qkv_upper_from_dense(q_hf, cfg)
    k_ref_packs = encode_qkv_upper_from_dense(k_hf, cfg)
    v_ref_packs = encode_qkv_upper_from_dense(v_hf, cfg)
    # KEY bake: HF packs * k_scale ≡ HE packs (if /2 cancel OK)
    k_ref_packs_baked = [p * k_scale for p in k_ref_packs]

    print("  --- pack-level (no dense decode) ---", flush=True)
    for name, he_ps, ref_ps, note in (
        ("Q", q_packs, q_ref_packs, "vs encode(HF_Q)"),
        ("K", k_packs, k_ref_packs_baked, f"vs encode(HF_K)*k_scale"),
        ("K_vs_HFraw", k_packs, k_ref_packs, "vs encode(HF_K) no bake"),
        ("V", v_packs, v_ref_packs, "vs encode(HF_V)"),
    ):
        # concat packs for one LS fit
        he_c = np.concatenate(he_ps)
        ref_c = np.concatenate(ref_ps)
        s = _stats(he_c, ref_c)
        print(
            f"  [pack {name}] {note}: err@1={s['err_gain1']:.3e}  "
            f"ls_gain={s['ls_gain']:.6g}  err@LS={s['err_ls']:.3e}  "
            f"corr={s['corr']:.6f}  median_ratio(ref/he)={s['median_ratio']:.6g}",
            flush=True,
        )

    # Undo KEY encode scale only for dense path.
    k_he_undo = k_he / k_scale

    rows = [
        ("Q", q_he, q_hf, "none (/2 cancelled by +conj)"),
        ("K", k_he_undo, k_hf, f"HE /= k_scale ({k_scale:.6g})"),
        ("V", v_he, v_hf, "none (/2 cancelled by +conj)"),
    ]
    # Also show K WITHOUT undo to make the bake visible.
    print("  --- dense decode (secondary) ---", flush=True)
    print(
        "  [K raw, no undo] vs HF:",
        flush=True,
    )
    s_raw = _stats(k_he, k_hf)
    print(
        f"    err@1={s_raw['err_gain1']:.3e}  ls_gain={s_raw['ls_gain']:.6g}  "
        f"median_ratio(ref/he)={s_raw['median_ratio']:.6g}  "
        f"(if HE=k_scale*HF expect ratio≈{1.0/k_scale:.4g})",
        flush=True,
    )

    n_bad = 0
    for name, he, ref, bake_note in rows:
        s = _stats(he, ref)
        gain_ok = abs(s["ls_gain"] - 1.0) <= float(args.tol_gain)
        ok = gain_ok and s["err_gain1"] <= float(args.tol)
        if not ok:
            n_bad += 1
        flag = "OK" if ok else "BAD"
        print(
            f"  [{flag}] {name} dense: bake_undo={bake_note}",
            flush=True,
        )
        print(
            f"    max|HE|={s['max_abs_he']:.4g}  max|HF|={s['max_abs_ref']:.4g}",
            flush=True,
        )
        print(
            f"    err@gain1={s['err_gain1']:.3e}  ls_gain={s['ls_gain']:.6g}  "
            f"err@LS={s['err_ls']:.3e}  corr={s['corr']:.6f}  "
            f"median_ratio(HF/HE)={s['median_ratio']:.6g}",
            flush=True,
        )
        if not gain_ok:
            print(
                f"    → residual gain after bake undo: ls_gain={s['ls_gain']:.6g} "
                f"(≠1 within tol={args.tol_gain})",
                flush=True,
            )

    # Primary verdict from pack-level Q/K_baked/V
    pack_bad = 0
    for name, he_ps, ref_ps in (
        ("Q", q_packs, q_ref_packs),
        ("K", k_packs, k_ref_packs_baked),
        ("V", v_packs, v_ref_packs),
    ):
        s = _stats(np.concatenate(he_ps), np.concatenate(ref_ps))
        if abs(s["ls_gain"] - 1.0) > float(args.tol_gain) or s["err_gain1"] > float(
            args.tol
        ):
            pack_bad += 1

    print(
        f"\nsummary: pack_bad={pack_bad}/3 dense_bad={n_bad}/3  "
        f"(remΔ={DEPTH_QKV} already checked; focus pack_bad for bake/gain)",
        flush=True,
    )
    return 0 if pack_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
