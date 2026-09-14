#!/usr/bin/env python3
"""
Probe Softmax σ-aSOR (``he_asor_ct`` → ``inv_d``) decrypt vs poly twin.

Spine (same as ``he_softmax_poly`` up through σ-inv)::

  scores → thor_exp_ct → rescale(pt×mask) → Σpacks + rotsum → sigma_exp
  enc_one = head_mask
  inv_d, d_delta, en = he_asor_ct(enc_one, sigma_exp, e0_σ, iters_σ)

Compares in the rem≤14 working zone. References:
  - plain Stockmeyer+mask+aggregate (sigma)
  - ``plain_he_asor_ct`` stored ``inv_v`` / logical ``inv_v/d_delta``
  - ``asor_inverse_slots`` (poly logical inv, head slots)

Usage::

  python3 HE/thor/probe_softmax_asor_sigma.py
  python3 HE/thor/probe_softmax_asor_sigma.py --start-rem 14 --level 2
  python3 HE/thor/probe_softmax_asor_sigma.py --start-rem 13 --level 0
"""
from __future__ import annotations

import argparse
import math
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


def _ensure_he_path() -> None:
    if _HE in sys.path:
        sys.path.remove(_HE)
    sys.path.insert(0, _HE)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)


_ensure_he_path()

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import dense_qk_scores, encode_score_packs_from_dense  # noqa: E402

_ensure_he_path()

from plain_softmax_delta_sim import head_mask, plain_he_asor_ct  # noqa: E402
from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    asor_inverse_slots,
    rotsum,
    stockmeyer_poly_slots,
)
from softmax_he_ct import he_asor_ct, thor_exp_ct  # noqa: E402
from softmax_poly import k_key_encode_scale  # noqa: E402
from thor_encoder_linear_core import add_vec, bert_base  # noqa: E402


def _encode_attention_mask_pt(engine, attention_mask: np.ndarray, *, level: int):
    attention_mask = np.asarray(attention_mask, dtype=np.float64).reshape(-1)
    n_tokens = int(np.count_nonzero(attention_mask))
    out = np.full((8,), None, dtype=object)
    for i in range(8):
        msg = np.zeros((2**15,), dtype=np.float64)
        for j in range(16):
            temp = j * (2**11)
            diag_index = i * 16 + j
            for t in range(128):
                col_index = (diag_index + t) % 128
                is_token = 1.0 if col_index < n_tokens else 0.0
                for head in range(12):
                    msg[temp + t * 16 + head] = is_token
        out[i] = engine.encode(msg, int(level))
    return out


def _mask_packs_numpy(attention_mask: np.ndarray, cfg) -> list[np.ndarray]:
    attention_mask = np.asarray(attention_mask, dtype=np.float64).reshape(-1)
    n_tokens = int(np.count_nonzero(attention_mask))
    packs = []
    for i in range(8):
        msg = np.zeros((cfg.num_slots,), dtype=np.float64)
        for j in range(16):
            temp = j * (2**11)
            diag_index = i * 16 + j
            for t in range(128):
                col_index = (diag_index + t) % 128
                is_token = 1.0 if col_index < n_tokens else 0.0
                for head in range(12):
                    msg[temp + t * 16 + head] = is_token
        packs.append(msg)
    return packs


def _plain_exp_masked(packs, mask_np, params) -> list[np.ndarray]:
    d1 = float(params.delta1)
    d2 = float(params.delta2)
    n1 = int(round(math.log2(d1)))
    scale = math.exp(params.shift / d2 / d1)
    baked = np.array(tuple(reversed(params.exp_coeffs)), dtype=np.float64) / scale
    out = []
    for p, m in zip(packs, mask_np):
        y = stockmeyer_poly_slots(baked, np.asarray(p, dtype=np.float64))
        for _ in range(n1):
            y = y * y
        out.append(y * m)
    return out


def _plain_sigma(exp_u: list[np.ndarray], cfg) -> np.ndarray:
    sigma = np.asarray(exp_u[0], dtype=np.float64).copy()
    for i in range(1, 8):
        sigma = add_vec(sigma, exp_u[i])
    return rotsum(sigma, cfg.slot_stride)


def _stats(he: np.ndarray, ref: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    a = he.ravel().astype(np.float64)
    b = ref.ravel().astype(np.float64)
    if mask is not None:
        m = mask.ravel().astype(bool)
        a, b = a[m], b[m]
    denom = float(np.dot(a, a))
    if denom > 1e-30:
        g = float(np.dot(a, b) / denom)
        err_ls = float(np.max(np.abs(a * g - b)))
    else:
        g = float("nan")
        err_ls = float("inf")
    nz = np.abs(a) > 1e-8
    med = float(np.median(b[nz] / a[nz])) if np.any(nz) else float("nan")
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if a.size > 1 and np.std(a) > 1e-15 and np.std(b) > 1e-15
        else float("nan")
    )
    return {
        "err_gain1": float(np.max(np.abs(a - b))) if a.size else float("nan"),
        "ls_gain": g,
        "err_ls": err_ls,
        "corr": corr,
        "median_ratio": med,
        "max_abs_he": float(np.max(np.abs(a))) if a.size else 0.0,
        "max_abs_ref": float(np.max(np.abs(b))) if a.size else 0.0,
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


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _ok(s: dict, *, err_tol: float = 5e-2) -> bool:
    return (
        np.isfinite(s["corr"])
        and s["corr"] > 0.99
        and abs(s["ls_gain"] - 1.0) < 0.05
        and s["err_gain1"] < err_tol
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-rem", type=int, default=14)
    ap.add_argument("--n-pad", type=int, default=20)
    args = ap.parse_args()

    cfg = bert_base()
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    rng = np.random.default_rng(args.seed)
    q = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    scores = np.clip(
        dense_qk_scores(q, k, cfg, scale=1.0 / np.sqrt(cfg.head_dim)), -8.0, 8.0
    )
    scores = scores * k_key_encode_scale(args.task, args.layer)
    packs = encode_score_packs_from_dense(scores, cfg)

    n_valid = cfg.seq_len - int(args.n_pad)
    attn = np.zeros(cfg.seq_len, dtype=np.float64)
    attn[:n_valid] = 1.0
    mask_np = _mask_packs_numpy(attn, cfg)

    plain_exp_u = _plain_exp_masked(packs, mask_np, params)
    plain_sigma = _plain_sigma(plain_exp_u, cfg)
    hm = head_mask(int(cfg.num_slots), pad=0.0)
    head_m = hm > 0.5

    plain_inv_v, plain_d_delta, plain_en = plain_he_asor_ct(
        hm,
        plain_sigma,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
    )
    plain_inv_log = plain_inv_v / float(plain_d_delta)
    poly_inv, poly_en = asor_inverse_slots(
        plain_sigma,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
    )

    sigma_head = plain_sigma[head_m]
    print(
        f"=== Softmax σ-aSOR probe  {args.task} L{args.layer} "
        f"level={args.level} start_rem={args.start_rem} ===",
        flush=True,
    )
    print(
        f"delta1={params.delta1} delta2={params.delta2}  "
        f"e0_sigma={params.e0_sigma:.6g}  iters_sigma={params.asor_max_iters_sigma}  "
        f"n_valid={n_valid}",
        flush=True,
    )
    print(
        f"plain sigma (head): min={float(np.min(sigma_head)):.6g}  "
        f"max={float(np.max(sigma_head)):.6g}  "
        f"mean={float(np.mean(sigma_head)):.6g}  "
        f"|σ|_∞={float(np.max(np.abs(plain_sigma))):.6g}",
        flush=True,
    )
    print(
        f"plain_he_asor: d_delta={plain_d_delta:.6g}  en={plain_en:.6g}  "
        f"max|inv_v|={float(np.max(np.abs(plain_inv_v))):.4g}  "
        f"max|inv_log|={float(np.max(np.abs(plain_inv_log))):.4g}",
        flush=True,
    )
    unit_err = float(np.max(np.abs(plain_inv_log[head_m] * plain_sigma[head_m] - 1.0)))
    print(
        f"plain logical: max|inv·σ−1| (heads)={unit_err:.3e}  "
        f"poly_en={poly_en:.6g}",
        flush=True,
    )

    print("create engine ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.start_rem)
    print(
        f"  ready ({time.time()-t0:.1f}s) nl={nl} enc_lc={enc_lc} "
        f"engine.scale={float(engine.scale):.6g}",
        flush=True,
    )

    x_ct = np.full((8,), None, dtype=object)
    for i in range(8):
        x_ct[i] = engine.encodecrypt(
            np.asarray(packs[i], dtype=np.float64), pk, level=enc_lc
        )
    rem_in = _rem(engine, x_ct[0])
    mask_pt = _encode_attention_mask_pt(
        engine, attn, level=int(x_ct[0].level_calc)
    )

    print("thor_exp_ct + mask + aggregate → sigma_exp ...", flush=True)
    t1 = time.time()
    exp_cts = thor_exp_ct(engine, list(x_ct), params, bts_hook=None)
    rem_exp = _rem(engine, exp_cts[0])
    exp_u = np.full((8,), None, dtype=object)
    for i in range(8):
        exp_u[i] = engine.rescale(engine.pt_ct_mult(mask_pt[i], exp_cts[i]))
    rem_u = _rem(engine, exp_u[0])

    sigma_exp = exp_u[0]
    for i in range(1, 8):
        sigma_exp = engine.cc_add(sigma_exp, exp_u[i])
    sigma_exp = engine.rotsum(sigma_exp, interval=2**11)
    rem_sigma = _rem(engine, sigma_exp)
    print(
        f"  done ({time.time()-t1:.1f}s)  rem entry→exp→mask→σ  "
        f"{rem_in}→{rem_exp}→{rem_u}→{rem_sigma}",
        flush=True,
    )

    print("\n=== decrypt sigma_exp ===", flush=True)
    he_sigma = np.asarray(
        engine.decrode(sigma_exp, sk, is_real=True), dtype=np.float64
    )
    s_sigma = _stats(he_sigma, plain_sigma, mask=head_m)
    _print("sigma_exp vs plain (heads)", s_sigma)
    print(
        f"  HE sigma (heads): min={float(np.min(he_sigma[head_m])):.6g}  "
        f"max={float(np.max(he_sigma[head_m])):.6g}",
        flush=True,
    )

    # Match production enc_one level: nl - ct.level (rem bookkeeping).
    enc_one_lc = int(engine.num_levels) - int(sigma_exp.level)
    enc_one = engine.encodecrypt(hm, pk, level=enc_one_lc)
    print(
        f"\nenc_one lc={enc_one_lc} rem={_rem(engine, enc_one)}  "
        f"(sigma rem={rem_sigma} level={int(sigma_exp.level)} "
        f"lc={int(sigma_exp.level_calc)})",
        flush=True,
    )
    need_rem = int(params.asor_max_iters_sigma)
    if rem_sigma < need_rem:
        print(
            f"  WARN: rem_sigma={rem_sigma} < iters_sigma={need_rem} "
            f"(may exhaust rem mid-aSOR; expect BAD decrypt)",
            flush=True,
        )

    print("he_asor_ct (σ-inv, bts_hook=None) ...", flush=True)
    t2 = time.time()
    inv_d, d_delta, en = he_asor_ct(
        engine,
        enc_one,
        sigma_exp,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
        bts_hook=None,
        rem_probe=None,
    )
    rem_inv = _rem(engine, inv_d)
    print(
        f"  done ({time.time()-t2:.1f}s)  rem σ→inv {rem_sigma}→{rem_inv}  "
        f"Δrem={rem_sigma - rem_inv}  d_delta={d_delta:.6g}  en={en:.6g}",
        flush=True,
    )
    print(
        f"  plain twin d_delta={plain_d_delta:.6g}  "
        f"(HE/plain ratio={d_delta / plain_d_delta if plain_d_delta else float('nan'):.6g})",
        flush=True,
    )

    print("\n=== decrypt inv_d (stored message) ===", flush=True)
    he_inv_v = np.asarray(engine.decrode(inv_d, sk, is_real=True), dtype=np.float64)
    s_stored = _stats(he_inv_v, plain_inv_v, mask=head_m)
    _print("inv_d decrypt vs plain_he_asor stored inv_v (heads)", s_stored)

    he_inv_log = he_inv_v / float(d_delta)
    s_log = _stats(he_inv_log, plain_inv_log, mask=head_m)
    _print("inv_d/d_delta vs plain logical (heads)", s_log)

    s_poly = _stats(he_inv_log, poly_inv, mask=head_m)
    _print("inv_d/d_delta vs asor_inverse_slots (heads)", s_poly)

    # Direct decrypt without /delta — does it already match logical poly?
    s_direct_poly = _stats(he_inv_v, poly_inv, mask=head_m)
    _print("inv_d raw decrypt vs asor_inverse_slots (heads)", s_direct_poly)

    he_unit = float(np.max(np.abs(he_inv_log[head_m] * he_sigma[head_m] - 1.0)))
    print(
        f"\n  HE logical max|inv·σ−1| (heads)={he_unit:.3e}  "
        f"(plain twin={unit_err:.3e})",
        flush=True,
    )

    ok_sigma = _ok(s_sigma, err_tol=1e-2)
    ok_stored = _ok(s_stored, err_tol=5e-2)
    ok_log = _ok(s_log, err_tol=5e-2)
    ok_poly = _ok(s_poly, err_tol=5e-2)
    # "direct decode usable" = stored decrypt tracks plain stored, or
    # logical after /delta tracks poly.
    print("\n=== verdict ===", flush=True)
    print(
        f"  sigma_exp decrypt={'PASS' if ok_sigma else 'FAIL'}  "
        f"inv stored vs plain_Δ={'PASS' if ok_stored else 'FAIL'}  "
        f"inv/d_delta vs plain_log={'PASS' if ok_log else 'FAIL'}  "
        f"inv/d_delta vs poly_asor={'PASS' if ok_poly else 'FAIL'}",
        flush=True,
    )
    print(
        f"  note: DualRail keeps stored inv_d + d_delta separately; "
        f"logical inv = decrypt(inv_d)/d_delta "
        f"(d_delta={d_delta:.6g}, not folded into CT).",
        flush=True,
    )
    return 0 if (ok_sigma and ok_log and ok_poly) else 2


if __name__ == "__main__":
    raise SystemExit(main())
