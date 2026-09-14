#!/usr/bin/env python3
"""
Probe ``dualrail_softmax_exit`` decrypt (128 α copies) vs plain twin / Softmax y.

Assumes post-``update_inv_D_fixed`` state is already correct. Continuity::

  scores → … → Σy² → (exp_2u, inv_d, d_delta)
  → dualrail_softmax_exit → 128 CTs

Decode rules (do **not** treat all 128 as independent Softmax packs)::

  layout:  cplx_softmax1[64] + cplx_softmax2[64]
           for i in 0..3, j in 0..15:
             α[i*16+j]     = 2·Re(rotsum(exp_cplx[i]·inv_rot[j]))
             α[64+i*16+j] = 2·Im(...)
  After Softmax, inv is rotsum-replicated → the 16 ``j`` copies are **identical**.
  Recover 8 packs: take α[i*16] and α[64+i*16], divide by extract_gain=2.

Usage::

  python3 HE/thor/probe_softmax_dualrail_exit.py
  python3 HE/thor/probe_softmax_dualrail_exit.py --start-rem 14 --land-rem 14
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
from linear_eval import (  # noqa: E402
    decode_score_packs,
    dense_qk_scores,
    encode_score_packs_from_dense,
)

_ensure_he_path()

from plain_softmax_delta_sim import (  # noqa: E402
    decode_alpha_128_to_packs8,
    head_mask,
    plain_dualrail_softmax_exit,
    plain_he_asor_ct,
    plain_update_inv_D_fixed,
)
from slot_softmax_he import SoftmaxHeParams, rotsum, stockmeyer_poly_slots  # noqa: E402
from softmax_he_ct import (  # noqa: E402
    dualrail_softmax_exit,
    he_asor_ct,
    thor_exp_ct,
    update_inv_D_fixed,
)
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
    if a.size == 0:
        return {
            "err_gain1": float("nan"),
            "ls_gain": float("nan"),
            "err_ls": float("nan"),
            "corr": float("nan"),
            "median_ratio": float("nan"),
            "max_abs_he": 0.0,
            "max_abs_ref": 0.0,
        }
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


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _ok(s: dict, *, err_tol: float = 5e-2) -> bool:
    if not np.isfinite(s["ls_gain"]) or abs(s["ls_gain"] - 1.0) >= 0.05:
        return False
    if s["err_gain1"] >= err_tol:
        return False
    if np.isfinite(s["corr"]) and s["corr"] > 0.99:
        return True
    return np.isfinite(s["median_ratio"]) and abs(s["median_ratio"] - 1.0) < 0.05


def _j_copy_spread(alpha128: list[np.ndarray]) -> float:
    """max |α[i*16]-α[i*16+j]| over Re/Im groups (expect ~0)."""
    err = 0.0
    for base in (0, 64):
        for i in range(4):
            ref = np.asarray(alpha128[base + i * 16], dtype=np.float64)
            for j in range(1, 16):
                cur = np.asarray(alpha128[base + i * 16 + j], dtype=np.float64)
                err = max(err, float(np.max(np.abs(ref - cur))))
    return err


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-rem", type=int, default=14)
    ap.add_argument("--land-rem", type=int, default=14)
    ap.add_argument("--n-pad", type=int, default=20)
    args = ap.parse_args()

    cfg = bert_base()
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    d2 = float(params.delta2)
    n2 = int(round(math.log2(d2)))
    iters_r0 = int(params.asor_max_iters_sum_sq[0])
    final_inv = n2 == 1

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
    hm = head_mask(int(cfg.num_slots), pad=0.0)

    # ---- plain through Σy², then DualRail exit + y shortcut ----
    plain_exp = _plain_exp_masked(packs, mask_np, params)
    plain_sigma = _plain_sigma(plain_exp, cfg)
    plain_inv_v, plain_d_delta, plain_en = plain_he_asor_ct(
        hm,
        plain_sigma,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
    )
    plain_exp2, plain_inv2_v, plain_d2, _plain_en2 = plain_update_inv_D_fixed(
        plain_exp,
        mask_np,
        plain_inv_v,
        plain_d_delta,
        plain_en,
        max_iters=iters_r0,
        final_inv=final_inv,
        cfg=cfg,
    )
    if final_inv:
        short = np.array(([1.0] * 12 + [0.0] * 4) * (2**7), dtype=np.float64)
        he_style = np.zeros(int(cfg.num_slots), dtype=np.float64)
        he_style[: short.size] = short
        plain_inv2_v = plain_inv2_v * he_style
    plain_y = [
        p * (plain_inv2_v / float(plain_d2)) * m
        for p, m in zip(plain_exp2, mask_np)
    ]
    plain128 = plain_dualrail_softmax_exit(
        plain_exp2, plain_inv2_v, plain_d2, cfg
    )
    scale_k = int(1.0 / (2.0 * float(plain_d2))) + 1
    expected_pack_over_y = float(scale_k) * float(plain_d2)

    print(
        f"=== DualRail Softmax exit probe  {args.task} L{args.layer} "
        f"level={args.level} start_rem={args.start_rem} land_rem={args.land_rem} ===",
        flush=True,
    )
    print(
        f"d_delta={plain_d2:.6g}  scale_k={scale_k}  "
        f"expect decode8/y ≈ scale_k·d_delta={expected_pack_over_y:.6g}  "
        f"(then ÷extract_gain is already in decode)",
        flush=True,
    )

    print("create engine ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.start_rem)
    land_lc = nl - int(args.land_rem)
    print(f"  ready ({time.time()-t0:.1f}s) nl={nl}", flush=True)

    # ---- HE spine → σ-aSOR ----
    x_ct = np.full((8,), None, dtype=object)
    for i in range(8):
        x_ct[i] = engine.encodecrypt(
            np.asarray(packs[i], dtype=np.float64), pk, level=enc_lc
        )
    mask_pt = _encode_attention_mask_pt(
        engine, attn, level=int(x_ct[0].level_calc)
    )

    print("HE → σ-aSOR ...", flush=True)
    t1 = time.time()
    exp_cts = thor_exp_ct(engine, list(x_ct), params, bts_hook=None)
    exp_u = np.full((8,), None, dtype=object)
    for i in range(8):
        exp_u[i] = engine.rescale(engine.pt_ct_mult(mask_pt[i], exp_cts[i]))
    sigma_exp = exp_u[0]
    for i in range(1, 8):
        sigma_exp = engine.cc_add(sigma_exp, exp_u[i])
    sigma_exp = engine.rotsum(sigma_exp, interval=2**11)
    enc_one = engine.encodecrypt(
        hm, pk, level=int(engine.num_levels) - int(sigma_exp.level)
    )
    inv_d, d_delta, en = he_asor_ct(
        engine,
        enc_one,
        sigma_exp,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
        bts_hook=None,
    )
    he_exp_msgs = [
        np.asarray(engine.decrode(exp_u[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    he_inv_msg = np.asarray(
        engine.decrode(inv_d, sk, is_real=True), dtype=np.float64
    )
    print(f"  done ({time.time()-t1:.1f}s) rem(inv)={_rem(engine, inv_d)}", flush=True)

    # ---- re-land → Σy² ----
    print(f"re-land rem={args.land_rem} → update_inv_D_fixed ...", flush=True)
    t2 = time.time()
    exp_u2 = np.full((8,), None, dtype=object)
    for i in range(8):
        exp_u2[i] = engine.encodecrypt(he_exp_msgs[i], pk, level=land_lc)
    inv_d2 = engine.encodecrypt(he_inv_msg, pk, level=land_lc)
    mask_pt2 = _encode_attention_mask_pt(engine, attn, level=land_lc)
    exp_2u, inv_new, d_delta_new, en_new = update_inv_D_fixed(
        engine,
        exp_u2,
        mask_pt2,
        inv_d2,
        d_delta,
        en,
        max_iters=iters_r0,
        final_inv=final_inv,
        bts_hook=None,
    )
    rem_inv = _rem(engine, inv_new)
    rem_exp = _rem(engine, exp_2u[0])
    print(
        f"  done ({time.time()-t2:.1f}s) rem exp2/inv={rem_exp}/{rem_inv}  "
        f"d_delta={d_delta_new:.6g}",
        flush=True,
    )
    if rem_inv < 1:
        print(
            f"  WARN: inv rem={rem_inv} < DEPTH_SOFTMAX_EXIT=1; exit may fail",
            flush=True,
        )

    # ---- DualRail exit (no mid-bts; rem budget on inv) ----
    print("dualrail_softmax_exit ...", flush=True)
    t3 = time.time()
    out_ct = dualrail_softmax_exit(
        engine,
        exp_2u,
        inv_new,
        d_delta=d_delta_new,
        rescale=False,
        bts_hook=None,
    )
    rem_out = _rem(engine, out_ct[0])
    print(
        f"  done ({time.time()-t3:.1f}s) len={len(out_ct)}  "
        f"rem {rem_inv}→{rem_out}  Δrem={rem_inv - rem_out}",
        flush=True,
    )

    print("\n=== decrypt 128 α ===", flush=True)
    t4 = time.time()
    he128 = [
        np.asarray(engine.decrode(out_ct[i], sk, is_real=True), dtype=np.float64)
        for i in range(128)
    ]
    print(f"  decrypt done ({time.time()-t4:.1f}s)", flush=True)

    j_he = _j_copy_spread(he128)
    j_pt = _j_copy_spread(plain128)
    print(
        f"  j-copy spread: HE={j_he:.3e}  plain={j_pt:.3e}  "
        f"|HE−plain|spread={abs(j_he - j_pt):.3e}",
        flush=True,
    )
    print(
        "  note: with final_inv short mask, rotated inv is NOT fully "
        "rotsum-replicated → j copies differ (by design); "
        "decode uses only j=0 (α[i*16] / α[64+i*16]).",
        flush=True,
    )

    s128 = _stats(np.concatenate(he128), np.concatenate(plain128))
    _print("HE128 vs plain_dualrail_softmax_exit (all slots)", s128)

    print(
        f"  α[0] max|he|={float(np.max(np.abs(he128[0]))):.4g}  "
        f"max|plain|={float(np.max(np.abs(plain128[0]))):.4g}",
        flush=True,
    )

    print("\n=== decode 128→8 (take j=0 copies, ÷ extract_gain=2) ===", flush=True)
    p8_he = decode_alpha_128_to_packs8(he128, extract_gain=2.0)
    p8_pt = decode_alpha_128_to_packs8(plain128, extract_gain=2.0)
    s8 = _stats(np.concatenate(p8_he), np.concatenate(p8_pt))
    _print("decode8 HE vs plain exit decode8", s8)

    # vs logical Softmax y: on both-nz slots, decode8 = (scale_k·d_delta)·y
    both_lists = []
    for a, b in zip(p8_he, plain_y):
        both = (np.abs(a) > 1e-10) & (np.abs(b) > 1e-10)
        if np.any(both):
            both_lists.append((a[both], b[both]))
    if both_lists:
        he_b = np.concatenate([t[0] for t in both_lists])
        y_b = np.concatenate([t[1] for t in both_lists])
        s_vs_y = _stats(he_b, y_b * expected_pack_over_y)
        _print(
            f"decode8 vs (scale_k·Δ)·y on both-nz "
            f"(expect gain1; scale_k·Δ={expected_pack_over_y:.6g})",
            s_vs_y,
        )
        med_raw = float(np.median(he_b / y_b))
        print(
            f"  median(decode8/y)={med_raw:.6g}  "
            f"(expect ≈ scale_k·d_delta={expected_pack_over_y:.6g})",
            flush=True,
        )
    else:
        s_vs_y = {
            "err_gain1": float("inf"),
            "ls_gain": float("nan"),
            "corr": float("nan"),
            "median_ratio": float("nan"),
        }
        print("  no overlapping nz slots for decode8 vs y", flush=True)

    ok128 = _ok(s128, err_tol=5e-2)
    ok8 = _ok(s8, err_tol=5e-2)
    ok_j_match = abs(j_he - j_pt) < 1e-6 or (
        j_pt > 0 and abs(j_he - j_pt) / j_pt < 0.01
    )
    scale_ok = _ok(s_vs_y, err_tol=5e-2)
    print("\n=== verdict ===", flush=True)
    print(
        f"  decrypt128 vs plain_exit={'PASS' if ok128 else 'FAIL'}  "
        f"j-spread HE≡plain={'PASS' if ok_j_match else 'FAIL'}  "
        f"decode8 vs plain={'PASS' if ok8 else 'FAIL'}  "
        f"decode8=(scale_k·Δ)·y={'PASS' if scale_ok else 'FAIL'}",
        flush=True,
    )
    return 0 if (ok128 and ok8 and ok_j_match and scale_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
