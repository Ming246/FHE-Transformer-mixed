#!/usr/bin/env python3
"""
Probe Softmax attn-mask step scale + decrypt error::

  exp_cts = thor_exp_ct(...)          # after square, no post-rescale
  exp_u   = rescale(pt_ct_mult(mask, exp_cts))

Reports rem/level, decrypt vs plain Stockmeyer+δ1-square (and masked),
and residual gain (Δ-scale mismatch shows up as huge ls_gain).

  python3 HE/thor/probe_softmax_mask_scale.py --task mrpc --layer 0 --start-rem 25
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

from slot_softmax_he import SoftmaxHeParams, stockmeyer_poly_slots  # noqa: E402
from softmax_he_ct import thor_exp_ct  # noqa: E402
from softmax_poly import k_key_encode_scale, score_hf_decode_bake  # noqa: E402
from thor_encoder_linear_core import bert_base  # noqa: E402


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
    """Same layout as encode_attention_mask, as float packs."""
    attention_mask = np.asarray(attention_mask, dtype=np.float64).reshape(-1)
    n_tokens = int(np.count_nonzero(attention_mask))
    packs = []
    for i in range(8):
        msg = np.zeros((cfg.num_slots,), dtype=np.float64)
        for j in range(16):
            temp = j * (2**11)
            diag_index = i * 16 + j
            for t in range(cfg.seq_len):
                col_index = (diag_index + t) % cfg.seq_len
                is_token = 1.0 if col_index < n_tokens else 0.0
                for head in range(cfg.num_heads):
                    msg[temp + t * 16 + head] = is_token
        packs.append(msg)
    return packs


def _plain_exp_cts(packs: list[np.ndarray], params: SoftmaxHeParams) -> list[np.ndarray]:
    """Mirror thor_exp_ct (prescaled): baked coeffs → Stockmeyer → δ1 squares."""
    d1 = float(params.delta1)
    d2 = float(params.delta2)
    n1 = int(round(math.log2(d1)))
    if 2**n1 != int(d1):
        raise ValueError(f"delta1={d1} not power of two")
    scale = math.exp(params.shift / d2 / d1)
    baked = np.array(tuple(reversed(params.exp_coeffs)), dtype=np.float64) / scale
    out = []
    for p in packs:
        y = stockmeyer_poly_slots(baked, np.asarray(p, dtype=np.float64))
        for _ in range(n1):
            y = y * y
        out.append(y)
    return out


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
    mask = np.abs(a) > 1e-8
    med = (
        float(np.median(b[mask] / a[mask])) if np.any(mask) else float("nan")
    )
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


def _print(tag: str, s: dict[str, float]) -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start-rem", type=int, default=25)
    ap.add_argument("--n-pad", type=int, default=20, help="padding tokens at end")
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

    # realistic key mask: first (128-n_pad) tokens valid
    n_valid = cfg.seq_len - int(args.n_pad)
    attn = np.zeros(cfg.seq_len, dtype=np.float64)
    attn[:n_valid] = 1.0
    mask_np = _mask_packs_numpy(attn, cfg)

    plain_exp = _plain_exp_cts(packs, params)
    plain_masked = [e * m for e, m in zip(plain_exp, mask_np)]

    print(
        f"=== Softmax mask-scale probe  {args.task} L{args.layer} "
        f"level={args.level} start_rem={args.start_rem} ===",
        flush=True,
    )
    bake = float(score_hf_decode_bake(args.task, args.layer))
    print(
        f"delta1={params.delta1} delta2={params.delta2}  "
        f"score_hf_decode_bake={bake:g}  n_valid={n_valid}  "
        f"plain |exp|_∞={max(np.max(np.abs(p)) for p in plain_exp):.4g}",
        flush=True,
    )

    print("create engine ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    nl = int(engine.num_levels)
    enc_lc = nl - int(args.start_rem)
    eng_scale = float(getattr(engine, "scale", float("nan")))
    print(
        f"  ready ({time.time()-t0:.1f}s) nl={nl} enc_lc={enc_lc} "
        f"engine.scale={eng_scale:.6g} (≈Δ)",
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

    print("thor_exp_ct ...", flush=True)
    t1 = time.time()
    exp_cts = thor_exp_ct(engine, list(x_ct), params, bts_hook=None)
    rem_exp = _rem(engine, exp_cts[0])
    print(
        f"  done ({time.time()-t1:.1f}s)  rem entry→exp {rem_in}→{rem_exp}  "
        f"Δrem={rem_in - rem_exp}  level_calc={exp_cts[0].level_calc}",
        flush=True,
    )

    print("\n=== decrypt exp_cts (pre-mask) ===", flush=True)
    he_exp = [
        np.asarray(engine.decrode(exp_cts[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    s_exp = _stats(np.concatenate(he_exp), np.concatenate(plain_exp))
    _print("exp_cts vs plain Stockmeyer+square", s_exp)

    # One extra rescale: if exp_cts were left at Δ² unread, this may help/hurt
    print("  --- control: one extra rescale(exp_cts) then decrypt ---", flush=True)
    he_exp_rs = []
    rem_rs = rem_exp
    for i in range(8):
        rs = engine.rescale(exp_cts[i])
        if i == 0:
            rem_rs = _rem(engine, rs)
        he_exp_rs.append(
            np.asarray(engine.decrode(rs, sk, is_real=True), dtype=np.float64)
        )
    s_rs = _stats(np.concatenate(he_exp_rs), np.concatenate(plain_exp))
    _print(f"rescale(exp_cts) rem {rem_exp}→{rem_rs} vs plain", s_rs)

    print("\n=== mask step: rescale(pt_ct_mult(mask, exp_cts)) ===", flush=True)
    exp_u = np.full((8,), None, dtype=object)
    # also capture pre-rescale product for scale diagnosis
    prod = np.full((8,), None, dtype=object)
    for i in range(8):
        prod[i] = engine.pt_ct_mult(mask_pt[i], exp_cts[i])
        exp_u[i] = engine.rescale(prod[i])
    rem_prod = _rem(engine, prod[0])
    rem_u = _rem(engine, exp_u[0])
    print(
        f"  rem: exp_cts={rem_exp}  pt_ct_mult(no rs)={rem_prod}  "
        f"after rescale={rem_u}  (expect rem drop 1 on rescale only)",
        flush=True,
    )

    print("\n=== decrypt prod (pt×ct, no rescale) ===", flush=True)
    he_prod = [
        np.asarray(engine.decrode(prod[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    s_prod = _stats(np.concatenate(he_prod), np.concatenate(plain_masked))
    _print("prod vs plain masked exp", s_prod)
    s_prod32 = _stats(
        np.concatenate(he_prod),
        np.concatenate([p * bake for p in plain_masked]),
    )
    _print(
        f"prod vs plain_masked × score_hf_decode_bake(={bake:g})",
        s_prod32,
    )

    print("\n=== decrypt exp_u (after rescale) ===", flush=True)
    he_u = [
        np.asarray(engine.decrode(exp_u[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    s_u = _stats(np.concatenate(he_u), np.concatenate(plain_masked))
    _print("exp_u vs plain masked exp", s_u)

    print("\n=== scale verdict ===", flush=True)
    print(
        f"  rem: exp_cts={rem_exp} → pt×ct={rem_prod} (no drop) → "
        f"rescale={rem_u} (drop 1)",
        flush=True,
    )
    print(
        f"  engine.scale(Δ)={eng_scale:.6g}  score_hf_decode_bake={bake:g}",
        flush=True,
    )
    print(
        f"  exp_cts direct decrypt: max|he|={s_exp['max_abs_he']:.4g} "
        f"(noise-like if ≪ plain); ls_gain={s_exp['ls_gain']:.6g}",
        flush=True,
    )
    print(
        f"  prod (no rs): corr={s_prod['corr']:.6f}  "
        f"med(ref/he)={s_prod['median_ratio']:.6g}  "
        f"vs plain×{bake:g}: ls_gain={s_prod32['ls_gain']:.6g} "
        f"err@1={s_prod32['err_gain1']:.3e}",
        flush=True,
    )
    print(
        f"  exp_u=rs(prod): max|he|={s_u['max_abs_he']:.4g}  "
        f"ls_gain={s_u['ls_gain']:.6g}  "
        f"(if noise again ⇒ outer rescale kills message at this scale state)",
        flush=True,
    )

    # PASS only if mask+rescale lands on plain; current Liberate path often fails that.
    ok_prod32 = (
        abs(s_prod32["ls_gain"] - 1.0) < 0.05
        and s_prod32["corr"] > 0.99
        and s_prod32["err_gain1"] < 5e-2
    )
    ok_u = abs(s_u["ls_gain"] - 1.0) < 0.05 and s_u["corr"] > 0.99
    print(
        f"\nsummary: rem_drop_ok={rem_u == rem_exp - 1}  "
        f"prod_vs_plain×{bake:g}={'PASS' if ok_prod32 else 'FAIL'}  "
        f"exp_u_vs_plain={'PASS' if ok_u else 'FAIL'}",
        flush=True,
    )
    return 0 if (ok_prod32 and ok_u) else 2


if __name__ == "__main__":
    raise SystemExit(main())
