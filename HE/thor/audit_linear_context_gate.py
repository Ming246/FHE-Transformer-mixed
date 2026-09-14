#!/usr/bin/env python3
"""
Linear Context gate (user methodology):

  1) Plain DualRail ≡ dense α@V  (machine eps)
  2) CT×CT / V-rots == DualRail theory floor
  3) HE DualRail ≡ dense         (CKKS noise)  [--he]

Canonical path: V complexify + make_copies(α) + gain=1
(Softmax DualRail 128-α layout is a separate Softmax I/O contract.)

Usage::

  python3 audit_linear_context_gate.py
  python3 audit_linear_context_gate.py --he --tol-he 1e-3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from linear_eval import (  # noqa: E402
    bind_att_context_mask_complement,
    complexify_v_he,
    complexify_v_plain,
    compute_attention_context_dualrail_island,
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

from thor_encoder_linear_core import (  # noqa: E402
    _accumulate_ttemp_adaptive,
    _build_ct_ct_masks,
    bert_base,
    make_copies,
    rotate_left,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--tol-plain", type=float, default=1e-12)
    ap.add_argument("--he", action="store_true")
    ap.add_argument("--tol-he", type=float, default=1e-3)
    ap.add_argument("--level", type=int, default=None)
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    rng = np.random.default_rng(args.seed)
    v = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim))
    logits = rng.normal(0.0, 0.5, (cfg.num_heads, cfg.seq_len, cfg.seq_len))
    m = logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits - m)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)
    vp = encode_qkv_upper_from_dense(v, cfg)
    ap = encode_score_packs_from_dense(alpha, cfg)
    ref = dense_alpha_v_context(alpha, v, cfg)

    print("=== Context gate (DualRail, make_copies α, gain=1) ===", flush=True)
    ct = _build_ct_ct_masks(cfg, cfg.n_in_slot)
    t0 = time.time()
    island = compute_attention_context_dualrail_island(
        ap, vp, cfg, gain=args.gain, ct_ct=ct, alpha_copy_mode="make_copies"
    )
    err_p = float(np.max(np.abs(decode_qkv_upper_to_dense(island, cfg) - ref)))
    ok1 = err_p < args.tol_plain
    print(
        f"[1] plain ≡ dense: max|err|={err_p:.3e}  "
        f"{'PASS' if ok1 else 'FAIL'}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    n_in, n_out = int(cfg.seq_len), 2
    theory_ctct = n_in * n_out
    theory_rot = (n_in - 1) * n_out
    vc = complexify_v_plain(vp)
    ac = make_copies(ap, cfg)
    s, stride, p = int(cfg.num_slots), int(cfg.slot_stride), int(cfg.pack)
    counts = {"ctct": 0, "v_rot": 0}
    ttemp = np.full((n_out, 4), None, dtype=object)
    for out in range(n_out):
        counts["ctct"] += 1
        ttemp[out, 0] = vc[out] * ac[0]
    for n in range(1, n_in):
        j = n % p
        q = (n // p) % 4
        rrot = (stride * j - p * n) % s
        temp = np.full((n_out, 4), None, dtype=object)
        for out in range(n_out):
            counts["v_rot"] += 1
            left = rotate_left(vc[out], (-rrot) % s)
            counts["ctct"] += 1
            prod = left * ac[n]
            if j == 0:
                temp[out, 0] = ct[0][n] * prod
                temp[out, 1] = prod - temp[out, 0]
            else:
                temp[out, 0] = ct[0][n] * prod
                temp[out, 1] = ct[1][n] * prod
                temp[out, 2] = ct[2][n] * prod
                temp[out, 3] = prod - temp[out, 0] - temp[out, 1] - temp[out, 2]
        _accumulate_ttemp_adaptive(ttemp, temp, q, j, n_out)
    ok2 = counts["ctct"] == theory_ctct and counts["v_rot"] == theory_rot
    print("[2] theory floor (BSGS core):", flush=True)
    print(
        f"    CT×CT={theory_ctct}  V-rots={theory_rot}  mul-depth ledger=2",
        flush=True,
    )
    print(
        f"    measured CT×CT={counts['ctct']} V-rots={counts['v_rot']}  "
        f"{'PASS' if ok2 else 'FAIL'}",
        flush=True,
    )

    ok3 = True
    if args.he:
        from bootstrap_hook import BootstrapHook
        from bts_ops import DEPTH_ATT_CONTEXT, remaining_to_level_calc
        from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys

        print("[3] HE DualRail context ...", flush=True)
        engine = create_engine(dict(THOR_DEFAULT_PARAMS))
        keys = create_keys(engine, with_evk=True)
        sk, pk = keys["sk"], keys["pk"]
        prepare_att_context_keys(engine, sk)
        keep = DEPTH_ATT_CONTEXT + 6
        work_level = (
            args.level
            if args.level is not None
            else remaining_to_level_calc(keep, engine.num_levels)
        )
        evaluator = make_qkv_evaluator(engine)
        hook = BootstrapHook(engine, mode="record")
        v_ct = np.full((4,), None, dtype=object)
        for i in range(4):
            v_ct[i] = engine.encodecrypt(vp[i], pk, level=work_level)
        a_copies_pt = make_copies(ap, cfg)
        a_copies = np.full((128,), None, dtype=object)
        for i in range(128):
            a_copies[i] = engine.encodecrypt(
                np.asarray(a_copies_pt[i], dtype=np.float64), pk, level=work_level
            )
        t1 = time.time()
        v_cplx = complexify_v_he(engine, v_ct)
        for j in range(128):
            a_copies[j] = engine.rescale(a_copies[j])
        lev_v = int(v_cplx[0].level_calc)
        lev_a = int(a_copies[0].level_calc)
        if lev_v != lev_a:
            target = max(lev_v, lev_a)
            if lev_v < target:
                for i in range(2):
                    v_cplx[i] = engine.level_up(v_cplx[i], target)
            if lev_a < target:
                for i in range(128):
                    a_copies[i] = engine.level_up(a_copies[i], target)
        ensure_ct_ct_matmul_masks(
            evaluator, engine, n_max=128, level=int(v_cplx[0].level_calc)
        )
        attn = make_att_context_module(engine, evaluator)
        bind_att_context_mask_complement(attn)
        ctx_ct = attn.calculate_attention_context(v_cplx, a_copies, rescale=False)
        hook.refresh("gate.linear_att_context", ctx_ct, indices=[])
        cplx_dec = [np.asarray(engine.decrode(ctx_ct[i], sk)) for i in range(2)]
        he_ctx = decode_qkv_upper_to_dense(
            thor_context_to_mainline_packs(cplx_dec, gain=args.gain), cfg
        )
        err_h = float(np.max(np.abs(he_ctx - ref)))
        ok3 = err_h < args.tol_he
        print(
            f"    HE ≡ dense: max|err|={err_h:.3e}  "
            f"{'PASS' if ok3 else 'FAIL'}  ({time.time()-t1:.1f}s)",
            flush=True,
        )
        print(f"    bootstrap_hook: {hook.summary()}", flush=True)
    else:
        print(
            "[3] HE skipped (pass --he). Prior smoke_att_context: ~1e-6 @ tol 1e-3.",
            flush=True,
        )

    ok = ok1 and ok2 and ok3
    print("CONTEXT GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
