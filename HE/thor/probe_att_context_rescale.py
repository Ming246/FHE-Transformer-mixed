#!/usr/bin/env python3
"""
Probe ``smoke_thor_repro`` att-context rescale quirk::

  v_cplx = pack(V); align → sftmx_out
  rescale(v_cplx); rescale(sftmx_out)     # outer rescale (decrypt of α/V often BAD)
  att_context = calculate_attention_context(v, α_rs, rescale=False)

Compares HE ``att_context`` decrypt to dense α@V from **pre-rescale**
decrypt/decode of α and V (plain matmul), as requested.

Also reports whether α/V decrypt OK before vs after the outer rescale.

Usage::

  python3 HE/thor/probe_att_context_rescale.py
  python3 HE/thor/probe_att_context_rescale.py --rem 12
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

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
from softmax_he_ct import align_bypass_to_main  # noqa: E402
from thor_encoder_linear_core import bert_base, make_copies  # noqa: E402


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--rem",
        type=int,
        default=12,
        help="encrypt rem (need ≥ DEPTH_ATT_CONTEXT + outer rescale + margin)",
    )
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    v_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    logits = rng.normal(0.0, 0.5, (cfg.num_heads, cfg.seq_len, cfg.seq_len))
    m = logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits - m)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    v_packs = encode_qkv_upper_from_dense(v_dense, cfg)
    a_packs = encode_score_packs_from_dense(alpha, cfg)
    # Softmax DualRail exit leaves 128 copies; geometric make_copies is the
    # plaintext twin of that broadcast used by context BSGS (smoke_att_context).
    a_copies_pt = make_copies(a_packs, cfg)
    ctx_ref = dense_alpha_v_context(alpha, v_dense, cfg)

    print(
        f"=== att_context outer-rescale probe  rem={args.rem} seed={args.seed} ===",
        flush=True,
    )
    print(
        f"dense |α@V|_∞={float(np.max(np.abs(ctx_ref))):.4g}  "
        f"|α|_∞={float(np.max(np.abs(alpha))):.4g}  "
        f"|V|_∞={float(np.max(np.abs(v_dense))):.4g}",
        flush=True,
    )

    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_att_context_keys(engine, sk)
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.rem)
    print(
        f"  ready ({time.time()-t0:.1f}s) nl={nl} enc_lc={enc_lc} rem={args.rem}",
        flush=True,
    )

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

    # --- pack V complex (NO rescale yet; match smoke_thor_repro) ---
    v_cplx = np.full((2,), None, dtype=object)
    for i in range(2):
        v_cplx[i] = engine.cc_add(v_ct[i], engine.imult(v_ct[i + 2]))
    for j in range(2):
        v_cplx[j] = align_bypass_to_main(engine, sftmx_out[0], v_cplx[j])

    print("\n=== (1) decrypt BEFORE outer rescale ===", flush=True)
    he_a0 = np.asarray(
        engine.decrode(sftmx_out[0], sk, is_real=True), dtype=np.float64
    )
    s_a = _stats(he_a0, a_copies_pt[0])
    _print("α[0] vs plain make_copies[0]", s_a)

    he_v0 = np.asarray(engine.decrode(v_cplx[0], sk), dtype=np.complex128)
    v_plain0 = np.asarray(v_packs[0], dtype=np.float64) + 1j * np.asarray(
        v_packs[2], dtype=np.float64
    )
    s_vr = _stats(np.real(he_v0), np.real(v_plain0))
    s_vi = _stats(np.imag(he_v0), np.imag(v_plain0))
    _print("v_cplx[0].re vs V pack0", s_vr)
    _print("v_cplx[0].im vs V pack2", s_vi)

    # Pre-rescale plain matmul ref (user request): dense from true α,V
    # (equivalent to decoding good pre-rescale messages → α@V)
    print(
        "\nplain ref = dense α@V from pre-rescale messages "
        "(same as decrypt-OK α packs × V)",
        flush=True,
    )

    rem_before = _rem(engine, sftmx_out[0])
    print(
        f"\n=== (2) outer rescale (smoke_thor_repro)  rem={rem_before} ===",
        flush=True,
    )
    for i in range(2):
        v_cplx[i] = engine.rescale(v_cplx[i])
    sftmx_rs = np.full((128,), None, dtype=object)
    for j in range(128):
        sftmx_rs[j] = engine.rescale(sftmx_out[j])
    rem_after = _rem(engine, sftmx_rs[0])
    print(f"  rem α {rem_before}→{rem_after}", flush=True)

    print("\n=== (3) decrypt AFTER outer rescale (expect BAD for α/V) ===", flush=True)
    he_a0_rs = np.asarray(
        engine.decrode(sftmx_rs[0], sk, is_real=True), dtype=np.float64
    )
    s_a_rs = _stats(he_a0_rs, a_copies_pt[0])
    _print("rescale(α[0]) vs plain copy[0]", s_a_rs)

    he_v0_rs = np.asarray(engine.decrode(v_cplx[0], sk), dtype=np.complex128)
    s_vr_rs = _stats(np.real(he_v0_rs), np.real(v_plain0))
    _print("rescale(v_cplx[0]).re vs V pack0", s_vr_rs)

    print("\n=== (4) calculate_attention_context(..., rescale=False) ===", flush=True)
    evaluator = make_qkv_evaluator(engine)
    score_level = int(v_cplx[0].level_calc)
    ensure_ct_ct_matmul_masks(evaluator, engine, n_max=128, level=score_level)
    attn = make_att_context_module(engine, evaluator)
    bind_att_context_mask_complement(attn)

    t1 = time.time()
    att_context = attn.calculate_attention_context(
        v_cplx, sftmx_rs, rescale=False
    )
    rem_ctx = _rem(engine, att_context[0])
    print(
        f"  done ({time.time()-t1:.1f}s) rem={rem_ctx}  "
        f"lc={int(att_context[0].level_calc)}  n={len(att_context)}",
        flush=True,
    )

    print("\n=== (5) decrypt att_context vs dense α@V ===", flush=True)
    cplx_dec = [
        np.asarray(engine.decrode(att_context[i], sk), dtype=np.complex128)
        for i in range(2)
    ]
    # gain=1: geometric / make_copies path (smoke_att_context default)
    mainline = thor_context_to_mainline_packs(cplx_dec, gain=1.0)
    he_ctx = decode_qkv_upper_to_dense(mainline, cfg)
    s_ctx = _stats(he_ctx, ctx_ref)
    _print("att_context decrypt vs dense α@V (gain=1)", s_ctx)

    # also try if a constant scale appears
    if np.isfinite(s_ctx["ls_gain"]) and abs(s_ctx["ls_gain"] - 1.0) > 0.05:
        print(
            f"  note: ls_gain={s_ctx['ls_gain']:.6g} ≠ 1 — "
            f"possible residual bake; err@LS={s_ctx['err_ls']:.3e}",
            flush=True,
        )

    ok_pre_a = _ok(s_a, err_tol=1e-2)
    ok_pre_v = _ok(s_vr, err_tol=1e-2) and _ok(s_vi, err_tol=1e-2)
    bad_post_a = not _ok(s_a_rs, err_tol=1e-2)
    bad_post_v = not _ok(s_vr_rs, err_tol=1e-2)
    ok_ctx = _ok(s_ctx, err_tol=args.tol)

    print("\n=== verdict ===", flush=True)
    print(
        f"  pre-rescale α decrypt={'OK' if ok_pre_a else 'BAD'}  "
        f"pre-rescale V decrypt={'OK' if ok_pre_v else 'BAD'}",
        flush=True,
    )
    print(
        f"  post-rescale α decrypt={'BAD (as expected)' if bad_post_a else 'UNEXPECTEDLY OK'}  "
        f"post-rescale V={'BAD (as expected)' if bad_post_v else 'UNEXPECTEDLY OK'}",
        flush=True,
    )
    print(
        f"  att_context vs pre-rescale α@V={'PASS' if ok_ctx else 'FAIL'}  "
        f"(err@1={s_ctx['err_gain1']:.3e}, ls_gain={s_ctx['ls_gain']:.6g})",
        flush=True,
    )
    return 0 if ok_ctx else 2


if __name__ == "__main__":
    raise SystemExit(main())
