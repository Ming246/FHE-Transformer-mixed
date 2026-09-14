#!/usr/bin/env python3
"""
Chunk smoke: DualRail attention score vs dense ``K @ Q.T``.

Oracle is ordinary matmul (not a DualRail BSGS twin):
  1. Random dense Q,K → upper-diagonal packs (same layout as attention geometry)
  2. scores_ref[h,key,query] = scale * (K_h @ Q_h.T)
  3. Sanity: mainline geometric packs → decode ≈ scores_ref
  4. HE THOR score → decrypt → DualRail remap → decode ≈ scores_ref

QKV PC-MM is out of scope here (covered by ``smoke_qkv_pcmm.py``).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_HE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _HE not in sys.path:
    sys.path.insert(0, _HE)
from path_setup import ensure_thor_path

ensure_thor_path()
if _HE not in sys.path:
    sys.path.append(_HE)

from bootstrap_hook import BootstrapHook  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys  # noqa: E402
from linear_eval import (  # noqa: E402
    bind_att_score_mask_complement,
    complexify_k_he,
    decode_score_packs,
    dense_qk_scores,
    encode_qkv_upper_from_dense,
    ensure_ct_ct_matmul_masks,
    ensure_make_copies_masks,
    ensure_rot_internal_att_masks,
    ensure_transpose_masks,
    make_att_score_module,
    make_qkv_evaluator,
    prep_q_for_copies_he,
    prepare_att_score_keys,
    thor_score_to_mainline_packs,
)

# dualrail_encode → ensure_thor_path may drop HE; re-append before mainline imports
if _HE not in sys.path:
    sys.path.append(_HE)
from thor_encoder_linear_core import (  # noqa: E402
    MaskBank,
    bert_base,
    compute_attention_score_slots,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="vs dense; rest_div=8 → expect CKKS noise floor",
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument(
        "--scale",
        type=float,
        default=None,
        help="score scale; default 1/sqrt(head_dim)",
    )
    ap.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="decrypt /gain; dense contract uses 1",
    )
    ap.add_argument(
        "--rest-div",
        type=float,
        default=8.0,
        help="n≥1 mask value scale 1/rest_div (canonical 8)",
    )
    args = ap.parse_args()

    cfg = bert_base()
    scale = (
        float(args.scale)
        if args.scale is not None
        else 1.0 / np.sqrt(cfg.head_dim)
    )
    rng = np.random.default_rng(args.seed)
    q_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    k_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)

    print("pack Q/K (upper-diag) + dense K@Q.T ref ...", flush=True)
    t0 = time.time()
    q_packs = encode_qkv_upper_from_dense(q_dense, cfg)
    k_packs = encode_qkv_upper_from_dense(k_dense, cfg)
    scores_ref = dense_qk_scores(q_dense, k_dense, cfg, scale=scale)
    print(
        f"  ready ({time.time()-t0:.1f}s) "
        f"max|K@Q.T|={float(np.max(np.abs(scores_ref))):.3e}",
        flush=True,
    )

    print("sanity: mainline geometric decode vs dense ...", flush=True)
    masks = MaskBank(cfg)
    geo = compute_attention_score_slots(
        q_packs, k_packs, cfg, masks, scale=scale, backend="geometric"
    )
    geo_dec = decode_score_packs(geo, cfg)
    geo_err = float(np.max(np.abs(geo_dec - scores_ref)))
    print(f"  geometric vs dense max|err|={geo_err:.3e}", flush=True)
    if geo_err > 1e-8:
        print("FAIL: geometric decode ≠ dense (pack/decode bug)", flush=True)
        return 2

    print("creating engine ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_att_score_keys(engine, sk)

    from bts_ops import (  # noqa: E402
        DEPTH_ATT_SCORE,
        DEPTH_MAKE_COPIES,
        DEPTH_TRANSPOSE_K,
        remaining_to_level_calc,
    )

    keep = DEPTH_TRANSPOSE_K + DEPTH_MAKE_COPIES + DEPTH_ATT_SCORE + 6
    work_level = (
        args.level
        if args.level is not None
        else remaining_to_level_calc(keep, engine.num_levels)
    )
    print(f"working level_calc={work_level} (keep={keep})", flush=True)

    evaluator = make_qkv_evaluator(engine)
    ensure_transpose_masks(evaluator, engine)
    ensure_make_copies_masks(evaluator, engine)
    ensure_rot_internal_att_masks(evaluator, engine, deltas=[64])

    hook = BootstrapHook(engine, mode="record")
    q_ct = np.full((4,), None, dtype=object)
    k_ct = np.full((4,), None, dtype=object)
    for i in range(4):
        q_ct[i] = engine.encodecrypt(q_packs[i], pk, level=work_level)
        k_ct[i] = engine.encodecrypt(k_packs[i], pk, level=work_level)

    print(
        f"HE transpose(K) → complexify → prep_q → make_copies → score "
        f"(rest_div={args.rest_div}, gain={args.gain}) ...",
        flush=True,
    )
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
        k_cplx,
        q_copies,
        bootstrap=False,
        scale=scale,
        rescale=False,
    )
    hook.refresh("L0.linear_att_score", att, indices=[])
    print(
        f"  HE done ({time.time()-t1:.1f}s) level={att[0].level_calc}",
        flush=True,
    )

    thor_dec = [
        np.asarray(engine.decrode(att[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]
    mainline_order = thor_score_to_mainline_packs(thor_dec, gain=float(args.gain))
    he_scores = decode_score_packs(mainline_order, cfg)
    err = float(np.max(np.abs(he_scores - scores_ref)))
    corr = float(np.corrcoef(he_scores.ravel(), scores_ref.ravel())[0, 1])
    print(
        f"  HE decode vs dense K@Q.T max|err|={err:.3e}  corr={corr:.6f}  "
        f"max|he|={float(np.max(np.abs(he_scores))):.3e}  "
        f"max|ref|={float(np.max(np.abs(scores_ref))):.3e}",
        flush=True,
    )
    print("bootstrap_hook:", hook.summary())
    ok = err < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
