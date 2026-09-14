#!/usr/bin/env python3
"""
Standalone test: Δ-bookkeeping plaintext twin of ``softmax_he_ct`` vs references.

Compares on the same 8-pack HE layout (prescaled score entry):
  1. ``he_softmax_delta_sim`` — mirrors ``he_asor_ct`` / ``update_inv_D_fixed`` / exit
  2. ``slot_softmax_he`` — logical slot sim without separate Δ
  3. ``softmax_poly.thor_softmax`` — dense row-softmax reference

Reports pack / dense max|err| without LS gain; also σ-aSOR ``d_delta`` and inv logical error.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
if _HE not in sys.path:
    sys.path.insert(0, _HE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linear_eval import decode_score_packs, dense_qk_scores, encode_score_packs_from_dense  # noqa: E402
from plain_softmax_delta_sim import (  # noqa: E402
    decode_alpha_128_to_packs8,
    he_softmax_delta_sim,
    head_mask,
    pack_ls_gain,
    pack_max_abs_diff,
    plain_he_asor_ct,
)
from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    encode_key_valid_mask_packs,
    slot_softmax_he,
    sum_over_keys,
    thor_exp_slots,
)
from softmax_poly import k_key_encode_scale, thor_softmax  # noqa: E402
from thor_encoder_linear_core import ThorConfig, bert_base  # noqa: E402

def _thor_dense_ref(
    scores: np.ndarray,
    key_valid: np.ndarray,
    params: SoftmaxHeParams,
) -> np.ndarray:
    """``thor_softmax`` per head → ``scores[h,key,query]`` layout."""
    seq = scores.shape[1]
    kv = torch.tensor(key_valid, dtype=torch.float64)
    coeffs = torch.tensor(list(params.exp_coeffs), dtype=torch.float64)
    out = np.zeros_like(scores, dtype=np.float64)
    for h in range(scores.shape[0]):
        x = torch.tensor(scores[h].T, dtype=torch.float64)
        m = kv.unsqueeze(0).expand(seq, seq)
        y = thor_softmax(
            x,
            m,
            float(params.shift),
            float(params.delta1),
            float(params.delta2),
            float(params.e0_sigma),
            int(params.asor_max_iters_sigma),
            params.asor_max_iters_sum_sq,
            coeffs,
        )
        out[h] = y.detach().cpu().numpy().T
    return out


def _random_case(
    cfg: ThorConfig,
    rng: np.random.Generator,
    *,
    prescale_he: bool,
    task: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    q = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.clip(dense_qk_scores(q, k, cfg, scale=scale), -8.0, 8.0)
    key_valid = np.ones(cfg.seq_len, dtype=np.float64)
    if prescale_he:
        scores = scores * k_key_encode_scale(task, layer)
    packs = encode_score_packs_from_dense(scores, cfg)
    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)
    return scores, key_valid, packs, mask_packs


def _sigma_probe(
    exp_u: list[np.ndarray],
    cfg: ThorConfig,
    params: SoftmaxHeParams,
) -> tuple[float, float, float]:
    """Return (d_delta, max|inv_logical·σ−1|, max|inv_logical·σ|)."""
    sigma = sum_over_keys(exp_u, cfg)
    enc_one = head_mask(int(cfg.num_slots), pad=0.0)
    inv_v, d_delta, _en = plain_he_asor_ct(
        enc_one,
        sigma,
        float(params.e0_sigma),
        int(params.asor_max_iters_sigma),
    )
    inv_log = inv_v / float(d_delta)
    prod = sigma * inv_log
    err_unit = float(np.max(np.abs(prod - 1.0)))
    max_prod = float(np.max(np.abs(prod)))
    return float(d_delta), err_unit, max_prod


def main() -> int:
    ap = argparse.ArgumentParser(description="Plaintext Δ Softmax twin vs references")
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol-pack", type=float, default=0.05)
    ap.add_argument("--tol-dense", type=float, default=0.05)
    ap.add_argument("--no-prescale", action="store_true", help="raw HF scores (exp_input_prescaled=False)")
    args = ap.parse_args()

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    prescale = not args.no_prescale
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    exp_prescaled = prescale

    scores, key_valid, packs, mask_packs = _random_case(
        cfg, rng, prescale_he=prescale, task=args.task, layer=args.layer
    )

    print(
        f"=== plain Δ softmax sim  task={args.task} L{args.layer} level={args.level} "
        f"prescale_he={prescale} seed={args.seed} ===",
        flush=True,
    )

    exp_u = thor_exp_slots(
        packs, mask_packs, params, exp_input_prescaled=exp_prescaled
    )
    d_delta, err_sigma, max_sigma_prod = _sigma_probe(exp_u, cfg, params)
    print(
        f"  σ-aSOR: d_delta={d_delta:.6g}  max|inv·σ−1|={err_sigma:.3e}  "
        f"max|inv·σ|={max_sigma_prod:.3e}",
        flush=True,
    )

    y_delta = he_softmax_delta_sim(
        packs, mask_packs, cfg, params, exp_input_prescaled=exp_prescaled
    )
    y_slot = slot_softmax_he(
        packs, mask_packs, cfg, params, exp_input_prescaled=exp_prescaled
    )

    err_pack = pack_max_abs_diff(y_delta, y_slot)
    d_delta_out = decode_score_packs(y_delta, cfg)
    d_slot = decode_score_packs(y_slot, cfg)
    for h in range(cfg.num_heads):
        for q in range(cfg.seq_len):
            d_delta_out[h, key_valid < 1, q] = 0.0
            d_slot[h, key_valid < 1, q] = 0.0
    err_dense_slot = float(np.max(np.abs(d_delta_out - d_slot)))

    scores_hf = (
        scores / k_key_encode_scale(args.task, args.layer) if prescale else scores
    )
    thor_ref = _thor_dense_ref(scores_hf, key_valid, params)
    err_dense_thor = float(np.max(np.abs(d_delta_out - thor_ref)))
    err_slot_thor = float(np.max(np.abs(d_slot - thor_ref)))

    print(
        f"  Δ-sim vs slot_softmax_he:  pack max|err|={err_pack:.3e}  "
        f"dense max|err|={err_dense_slot:.3e}",
        flush=True,
    )
    print(
        f"  vs thor_softmax:           Δ-sim dense={err_dense_thor:.3e}  "
        f"slot dense={err_slot_thor:.3e}",
        flush=True,
    )
    print(
        f"  max|α|  Δ-sim={float(np.max(d_delta_out)):.3e}  "
        f"slot={float(np.max(d_slot)):.3e}  thor={float(np.max(thor_ref)):.3e}",
        flush=True,
    )

    # --- full HE: DualRail exit → decrypt unpack → decode ---
    alpha_128 = he_softmax_delta_sim(
        packs,
        mask_packs,
        cfg,
        params,
        exp_input_prescaled=exp_prescaled,
        full_he_exit=True,
    )
    j_spread = max(
        float(np.max(np.abs(alpha_128[i * 16] - alpha_128[i * 16 + 1])))
        for i in range(4)
    )
    packs8_dec = decode_alpha_128_to_packs8(alpha_128, extract_gain=2.0)
    d_full = decode_score_packs(packs8_dec, cfg)
    for h in range(cfg.num_heads):
        for q in range(cfg.seq_len):
            d_full[h, key_valid < 1, q] = 0.0
    err_full_raw = float(np.max(np.abs(d_full - thor_ref)))
    gain_full = pack_ls_gain(packs8_dec, y_slot)
    d_full_scaled = d_full * gain_full
    err_full_ls = float(np.max(np.abs(d_full_scaled - thor_ref)))
    err_full_vs_8 = float(np.max(np.abs(d_full - d_delta_out)))
    print(
        f"  full HE exit (128→8 decode): 128-copy j-spread={j_spread:.3e}  "
        f"vs thor raw={err_full_raw:.3e}  ls_gain={gain_full:.6g}  "
        f"vs thor scaled={err_full_ls:.3e}  vs 8pack-shortcut={err_full_vs_8:.3e}",
        flush=True,
    )
    print(
        f"  max|α| full decode={float(np.max(d_full)):.3e}  "
        f"×gain={float(np.max(d_full_scaled)):.3e}",
        flush=True,
    )

    ok = (
        err_dense_slot < args.tol_dense
        and err_dense_thor < args.tol_dense
        and err_full_ls < args.tol_dense
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
