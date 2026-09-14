#!/usr/bin/env python3
"""
Linear Score gate (user methodology):

  1) Plain DualRail ≡ dense K@Q.T  (machine eps only)
  2) CT×CT / K-rots / merge-rots / mul-depth == DualRail theory floor
  3) HE DualRail ≡ dense               (CKKS noise only)  [--he]

Canonical path: n≥1 BSGS masks ×(1/rest_div) with rest_div=8, unpack gain=1
(0 extra mul-depth vs THOR native). Softmax must use dense-scale ranges.
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
if _HE not in sys.path:
    sys.path.insert(0, _HE)

from path_setup import ensure_thor_path

ensure_thor_path()
sys.path.insert(0, _HE)
sys.path.insert(0, _HERE)

from linear_eval import (  # noqa: E402
    bind_att_score_mask_complement,
    complexify_k_he,
    complexify_k_plain,
    compute_attention_score_dualrail_island,
    decode_score_packs,
    dense_qk_scores,
    encode_qkv_upper_from_dense,
    ensure_ct_ct_matmul_masks,
    ensure_make_copies_masks,
    ensure_rot_internal_att_masks,
    ensure_transpose_masks,
    make_att_score_module,
    make_qkv_evaluator,
    numpy_make_copies_dualrail,
    prep_q_for_copies_he,
    prepare_att_score_keys,
    thor_score_to_mainline_packs,
)

sys.path.insert(0, _HE)

from thor_encoder_linear_core import (  # noqa: E402
    _accumulate_ttemp_adaptive,
    _build_ct_ct_masks,
    bert_base,
    rotate_left,
    transpose_upper_to_lower,
)


def _plain_and_counts(cfg, q_packs, k_packs, scale: float, rest_div: float) -> tuple:
    ct = _build_ct_ct_masks(cfg, cfg.head_dim)
    island = compute_attention_score_dualrail_island(
        q_packs, k_packs, cfg, scale=scale, gain=1.0, rest_div=rest_div, ct_ct=ct
    )
    # dense ref built by caller
    n_in, n_out = int(cfg.head_dim), 4
    theory = {
        "ctct": n_in * n_out,
        "k_rot": (n_in - 1) * n_out,
        "merge_rot": 2 * n_out,
        "depth": 1,
    }
    k_lower = transpose_upper_to_lower(k_packs, cfg)
    kc = complexify_k_plain(k_lower, cfg)
    qc = numpy_make_copies_dualrail(q_packs)
    s, stride, p = int(cfg.num_slots), int(cfg.slot_stride), int(cfg.pack)
    counts = {"ctct": 0, "k_rot": 0, "merge_rot": 0}
    ttemp = np.full((n_out, 4), None, dtype=object)
    sp = np.full(s, float(scale))
    inv = 1.0 / float(rest_div)
    for out in range(n_out):
        counts["ctct"] += 1
        ttemp[out, 0] = sp * (kc[out] * qc[0])
    for n in range(1, n_in):
        j = n % p
        qg = n // p
        rrot = (stride * j - p * n) % s
        temp = np.full((n_out, 4), None, dtype=object)
        for out in range(n_out):
            counts["k_rot"] += 1
            left = rotate_left(kc[out], (-rrot) % s)
            counts["ctct"] += 1
            prod = left * qc[n]
            if j == 0:
                temp[out, 0] = (inv * ct[0][n]) * prod
                temp[out, 1] = (inv * ct[1][n]) * prod
            else:
                temp[out, 0] = (inv * ct[0][n]) * prod
                temp[out, 1] = (inv * ct[1][n]) * prod
                temp[out, 2] = (inv * ct[2][n]) * prod
                temp[out, 3] = (inv * ct[3][n]) * prod
        _accumulate_ttemp_adaptive(ttemp, temp, qg, j, n_out)
    for _ in range(n_out):
        counts["merge_rot"] += 2
    return island, theory, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--rest-div",
        type=float,
        default=8.0,
        help="n≥1 mask /rest_div (canonical 8); unpack gain=1",
    )
    ap.add_argument("--tol-plain", type=float, default=1e-12)
    ap.add_argument("--he", action="store_true", help="also run HE CT score smoke path")
    ap.add_argument("--tol-he", type=float, default=1e-3)
    ap.add_argument("--level", type=int, default=None)
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    scale = 1.0 / np.sqrt(float(cfg.head_dim))
    rng = np.random.default_rng(args.seed)
    q = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim))
    qp = encode_qkv_upper_from_dense(q, cfg)
    kp = encode_qkv_upper_from_dense(k, cfg)
    ref = dense_qk_scores(q, k, cfg, scale=scale)

    print("=== Score gate (DualRail island, rest_div mask /8) ===", flush=True)
    print(f"rest_div={args.rest_div} gain=1 scale={scale:.6f}", flush=True)

    t0 = time.time()
    island, theory, counts = _plain_and_counts(cfg, qp, kp, scale, args.rest_div)
    err_p = float(np.max(np.abs(decode_score_packs(island, cfg) - ref)))
    ok1 = err_p < args.tol_plain
    print(
        f"[1] plain ≡ dense: max|err|={err_p:.3e}  "
        f"{'PASS' if ok1 else 'FAIL'}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    ok2 = (
        counts["ctct"] == theory["ctct"]
        and counts["k_rot"] == theory["k_rot"]
        and counts["merge_rot"] == theory["merge_rot"]
    )
    print("[2] theory floor (BSGS core):", flush=True)
    print(
        f"    CT×CT={theory['ctct']}  K-rots={theory['k_rot']}  "
        f"merge-rots={theory['merge_rot']}  mul-depth={theory['depth']}",
        flush=True,
    )
    print(
        f"    measured CT×CT={counts['ctct']} K-rots={counts['k_rot']} "
        f"merge-rots={counts['merge_rot']}  {'PASS' if ok2 else 'FAIL'}",
        flush=True,
    )
    print(
        "    (prep outside core: transpose + complexify×4 + DualRail make_copies; "
        "geometric place would be ~8192 CT×CT)",
        flush=True,
    )

    ok3 = True
    if args.he:
        from bootstrap_hook import BootstrapHook
        from bts_ops import (
            DEPTH_ATT_SCORE,
            DEPTH_MAKE_COPIES,
            DEPTH_TRANSPOSE_K,
            remaining_to_level_calc,
        )
        from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys

        print("[3] HE DualRail score ...", flush=True)
        engine = create_engine(dict(THOR_DEFAULT_PARAMS))
        keys = create_keys(engine, with_evk=True)
        sk, pk = keys["sk"], keys["pk"]
        prepare_att_score_keys(engine, sk)
        keep = DEPTH_TRANSPOSE_K + DEPTH_MAKE_COPIES + DEPTH_ATT_SCORE + 6
        work_level = (
            args.level
            if args.level is not None
            else remaining_to_level_calc(keep, engine.num_levels)
        )
        evaluator = make_qkv_evaluator(engine)
        ensure_transpose_masks(evaluator, engine)
        ensure_make_copies_masks(evaluator, engine)
        ensure_rot_internal_att_masks(evaluator, engine, deltas=[64])
        hook = BootstrapHook(engine, mode="record")
        q_ct = np.full((4,), None, dtype=object)
        k_ct = np.full((4,), None, dtype=object)
        for i in range(4):
            q_ct[i] = engine.encodecrypt(qp[i], pk, level=work_level)
            k_ct[i] = engine.encodecrypt(kp[i], pk, level=work_level)
        t1 = time.time()
        k_lower = evaluator.transpose_upper_to_lower(k_ct)
        k_cplx = complexify_k_he(engine, evaluator, k_lower)
        q_prep = prep_q_for_copies_he(engine, q_ct)
        q_copies = evaluator.make_copies(q_prep)
        score_level = max(int(k_cplx[0].level_calc), int(q_copies[0].level_calc))
        ensure_ct_ct_matmul_masks(
            evaluator,
            engine,
            n_max=64,
            level=score_level,
            bank="ct_ct_matmul_score",
            value_scale=1.0 / float(args.rest_div),
        )
        attn = make_att_score_module(engine, evaluator)
        bind_att_score_mask_complement(attn, rest_div=float(args.rest_div))
        att = attn.calculate_attention_score(
            k_cplx, q_copies, bootstrap=False, scale=scale, rescale=False
        )
        hook.refresh("gate.linear_att_score", att, indices=[])
        thor_dec = [
            np.asarray(engine.decrode(att[i], sk, is_real=True), dtype=np.float64)
            for i in range(8)
        ]
        he_scores = decode_score_packs(
            thor_score_to_mainline_packs(thor_dec, gain=1.0), cfg
        )
        err_h = float(np.max(np.abs(he_scores - ref)))
        ok3 = err_h < args.tol_he
        print(
            f"    HE ≡ dense: max|err|={err_h:.3e}  "
            f"{'PASS' if ok3 else 'FAIL'}  ({time.time()-t1:.1f}s)  "
            f"tol={args.tol_he}",
            flush=True,
        )
        print(f"    bootstrap_hook: {hook.summary()}", flush=True)
    else:
        print(
            "[3] HE skipped (pass --he). Prior smoke_att_score: ~3e-6 @ tol 1e-3.",
            flush=True,
        )

    ok = ok1 and ok2 and ok3
    print("SCORE GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
