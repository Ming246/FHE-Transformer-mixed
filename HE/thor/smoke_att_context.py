#!/usr/bin/env python3
"""
Chunk smoke: DualRail attention context (α@V) vs dense matmul.

Skip Softmax (nonlinear / data-dependent approx — out of scope):
  1. Random dense V + row-normalized alpha
  2. Pack V (upper-diag) + alpha (score packs)
  3. Sanity: mainline geometric context → decode ≈ dense α@V
  4. HE: make_copies(α)(128) + V→2 complex → calculate_attention_context
     → decrypt remap (/gain=1) → vs dense

Default alpha path matches plaintext DualRail island (``make_copies`` + gain=1).
``--copy-mode final|identity`` keeps the Softmax DualRail I/O twin for A/B.
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
    bind_att_context_mask_complement,
    complexify_v_he,
    decode_qkv_upper_to_dense,
    dense_alpha_v_context,
    dualrail_alpha_copies_softmax_layout,
    encode_qkv_upper_from_dense,
    encode_score_packs_from_dense,
    ensure_ct_ct_matmul_masks,
    make_att_context_module,
    make_qkv_evaluator,
    prepare_att_context_keys,
    thor_context_to_mainline_packs,
)

if _HE not in sys.path:
    sys.path.append(_HE)
from thor_encoder_linear_core import (  # noqa: E402
    MaskBank,
    bert_base,
    compute_attention_context_slots,
    make_copies,
)


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="vs dense; make_copies+gain1 expect CKKS noise floor",
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="decrypt /gain; make_copies path uses 1 (Softmax twin ~32)",
    )
    ap.add_argument(
        "--copy-mode",
        choices=("make_copies", "final", "identity"),
        default="make_copies",
        help="make_copies: ≡ geometric; final/identity: Softmax DualRail I/O twin",
    )
    args = ap.parse_args()

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    v_dense = rng.normal(0.0, 0.1, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    logits = rng.normal(0.0, 0.5, (cfg.num_heads, cfg.seq_len, cfg.seq_len))
    m = logits.max(axis=1, keepdims=True)
    alpha = np.exp(logits - m)
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    print("pack V + alpha + dense α@V ref ...", flush=True)
    t0 = time.time()
    v_packs = encode_qkv_upper_from_dense(v_dense, cfg)
    a_packs = encode_score_packs_from_dense(alpha, cfg)
    ctx_ref = dense_alpha_v_context(alpha, v_dense, cfg)
    print(
        f"  ready ({time.time()-t0:.1f}s) "
        f"max|α@V|={float(np.max(np.abs(ctx_ref))):.3e}",
        flush=True,
    )

    print("sanity: mainline geometric context vs dense ...", flush=True)
    masks = MaskBank(cfg)
    geo = compute_attention_context_slots(
        a_packs, v_packs, cfg, masks, apply_softmax=False, backend="geometric"
    )
    geo_dec = decode_qkv_upper_to_dense(geo, cfg)
    geo_err = float(np.max(np.abs(geo_dec - ctx_ref)))
    print(f"  geometric vs dense max|err|={geo_err:.3e}", flush=True)
    if geo_err > 1e-8:
        print("FAIL: geometric decode ≠ dense (pack/decode bug)", flush=True)
        return 2

    print(f"alpha copies (128, mode={args.copy_mode}) ...", flush=True)
    if args.copy_mode == "make_copies":
        a_copies_pt = make_copies(a_packs, cfg)
    else:
        a_copies_pt = dualrail_alpha_copies_softmax_layout(
            a_packs, extract_gain=2.0, mode=args.copy_mode
        )
    assert len(a_copies_pt) == 128
    max_c = float(np.max(np.abs(a_copies_pt[0])))
    j_err = float(np.max(np.abs(a_copies_pt[0] - a_copies_pt[1])))
    print(f"  max|copy[0]|={max_c:.3e}  |copy[0]-copy[1]|={j_err:.3e}", flush=True)

    print("creating engine ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = create_keys(engine, with_evk=True)
    sk, pk = keys["sk"], keys["pk"]
    prepare_att_context_keys(engine, sk)

    from bts_ops import DEPTH_ATT_CONTEXT, remaining_to_level_calc  # noqa: E402

    keep = DEPTH_ATT_CONTEXT + 6
    work_level = (
        args.level
        if args.level is not None
        else remaining_to_level_calc(keep, engine.num_levels)
    )
    print(f"working level_calc={work_level} (keep={keep})", flush=True)

    evaluator = make_qkv_evaluator(engine)
    hook = BootstrapHook(engine, mode="record")

    print("encrypt V(4) + alpha copies(128) ...", flush=True)
    v_ct = np.full((4,), None, dtype=object)
    for i in range(4):
        v_ct[i] = engine.encodecrypt(v_packs[i], pk, level=work_level)
    a_copies = np.full((128,), None, dtype=object)
    for i in range(128):
        a_copies[i] = engine.encodecrypt(
            np.asarray(a_copies_pt[i], dtype=np.float64), pk, level=work_level
        )
        if (i + 1) % 32 == 0:
            print(f"  encrypted alpha copies {i+1}/128", flush=True)

    print(
        f"HE complexify(V) + rescale(α) → context BSGS (gain={args.gain}) ...",
        flush=True,
    )
    t1 = time.time()
    # Match bert.forward: complexify+rescale(V), rescale(att_prob), then align.
    v_cplx = complexify_v_he(engine, v_ct)
    for j in range(128):
        a_copies[j] = engine.rescale(a_copies[j])
    lev_v = int(v_cplx[0].level_calc)
    lev_a = int(a_copies[0].level_calc)
    if lev_v != lev_a:
        target = max(lev_v, lev_a)
        print(f"  level align v={lev_v} a={lev_a} → {target}", flush=True)
        if lev_v < target:
            for i in range(2):
                v_cplx[i] = engine.level_up(v_cplx[i], target)
        if lev_a < target:
            for i in range(128):
                a_copies[i] = engine.level_up(a_copies[i], target)
    score_level = int(v_cplx[0].level_calc)
    ensure_ct_ct_matmul_masks(evaluator, engine, n_max=128, level=score_level)
    attn = make_att_context_module(engine, evaluator)
    bind_att_context_mask_complement(attn)
    ctx_ct = attn.calculate_attention_context(v_cplx, a_copies, rescale=False)
    hook.refresh("L0.linear_att_context", ctx_ct, indices=[])
    print(
        f"  HE done ({time.time()-t1:.1f}s) level={ctx_ct[0].level_calc} "
        f"n_out={len(ctx_ct)}",
        flush=True,
    )

    cplx_dec = [_to_numpy(engine.decrode(ctx_ct[i], sk)) for i in range(2)]
    mainline = thor_context_to_mainline_packs(cplx_dec, gain=args.gain)
    he_ctx = decode_qkv_upper_to_dense(mainline, cfg)
    err = float(np.max(np.abs(he_ctx - ctx_ref)))
    corr = float(np.corrcoef(he_ctx.ravel(), ctx_ref.ravel())[0, 1])
    print(
        f"  HE decode vs dense α@V max|err|={err:.3e}  corr={corr:.6f}  "
        f"max|he|={float(np.max(np.abs(he_ctx))):.3e}  "
        f"max|ref|={float(np.max(np.abs(ctx_ref))):.3e}  gain={args.gain}",
        flush=True,
    )
    print("bootstrap_hook:", hook.summary())
    ok = err < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
