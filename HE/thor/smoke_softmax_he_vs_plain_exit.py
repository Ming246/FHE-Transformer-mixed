#!/usr/bin/env python3
"""
Compare real HE ``he_softmax_poly`` (128 DualRail exit) decrypt vs plaintext full exit.

Same prescaled 8-pack score input; reports pack / dense error and row-sum checks.
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

from bootstrap_hook import BootstrapHook  # noqa: E402
from dualrail_bts import install_mock_bootstrap  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from linear_eval import decode_score_packs, dense_qk_scores, encode_score_packs_from_dense  # noqa: E402
from plain_softmax_delta_sim import (  # noqa: E402
    decode_alpha_128_to_packs8,
    he_softmax_delta_sim,
    pack_ls_gain,
    pack_max_abs_diff,
)
from slot_softmax_he import (  # noqa: E402
    SoftmaxHeParams,
    encode_key_valid_mask_packs,
)
from softmax_he_ct import he_softmax_poly  # noqa: E402
from softmax_poly import k_key_encode_scale  # noqa: E402
from thor_encoder_linear_core import bert_base  # noqa: E402


def _encode_attention_mask_pt(
    engine,
    attention_mask: np.ndarray,
    *,
    level: int,
) -> np.ndarray:
    """THOR ``ThorDataEncryptor.encode_attention_mask`` (PT only)."""
    attention_mask = np.asarray(attention_mask, dtype=np.float64).reshape(-1)
    if attention_mask.shape != (128,):
        raise ValueError(f"attention_mask must be (128,), got {attention_mask.shape}")
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


def _random_packs(
    cfg,
    rng: np.random.Generator,
    *,
    task: str,
    layer: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    q = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.clip(dense_qk_scores(q, k, cfg, scale=scale), -8.0, 8.0)
    scores = scores * k_key_encode_scale(task, layer)
    key_valid = np.ones(cfg.seq_len, dtype=np.float64)
    packs = encode_score_packs_from_dense(scores, cfg)
    mask_packs = encode_key_valid_mask_packs(key_valid, cfg)
    return packs, mask_packs


def _decrypt_he128(engine, sk, out_ct) -> list[np.ndarray]:
    dec: list[np.ndarray] = []
    for i in range(128):
        msg = np.asarray(
            engine.decrode(out_ct[i], sk, is_real=True), dtype=np.float64
        ).ravel()
        dec.append(msg)
    return dec


def _row_sum_stats(alpha_dense: np.ndarray, key_valid: np.ndarray) -> tuple[float, float]:
    kv = key_valid.reshape(-1) > 0
    sums = []
    for h in range(alpha_dense.shape[0]):
        for q in range(alpha_dense.shape[2]):
            sums.append(float(alpha_dense[h, kv, q].sum()))
    s = np.asarray(sums, dtype=np.float64)
    return float(np.max(np.abs(s - 1.0))), float(np.mean(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock-land-rem", type=int, default=14)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    packs, mask_packs = _random_packs(cfg, rng, task=args.task, layer=args.layer)

    plain128 = he_softmax_delta_sim(
        packs, mask_packs, cfg, params, full_he_exit=True
    )

    print("create engine + keys (rot + mock bts) ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    land = int(engine.num_levels) - int(args.mock_land_rem)
    mock_info = install_mock_bootstrap(engine, sk, pk, land_level=land)
    bts_hook = BootstrapHook(engine, mode="always")
    print(
        f"  engine ready ({time.time()-t0:.1f}s) num_levels={engine.num_levels} "
        f"mock_land_lc={mock_info['land_level']} rem≈{args.mock_land_rem}",
        flush=True,
    )

    x_ct = np.full((8,), None, dtype=object)
    enc_lc = int(mock_info["land_level"])
    for i in range(8):
        ct = engine.encodecrypt(
            np.asarray(packs[i], dtype=np.float64), pk, level=enc_lc
        )
        x_ct[i] = engine.bootstrap(ct)
    mask_ct = _encode_attention_mask_pt(
        engine,
        np.ones(cfg.seq_len, dtype=np.float64),
        level=int(x_ct[0].level_calc),
    )

    print("HE he_softmax_poly (mock bts always) ...", flush=True)
    t1 = time.time()
    try:
        out_ct = he_softmax_poly(
            engine,
            x_ct,
            mask_ct,
            params,
            layer_idx=int(args.layer),
            rescale=False,
            bts_hook=bts_hook,
        )
    except (RuntimeError, Exception) as exc:
        print(f"FAIL: HE softmax raised: {exc}", flush=True)
        return 2
    print(
        f"  done ({time.time()-t1:.1f}s) out len={len(out_ct)} "
        f"out[0].lc={int(out_ct[0].level_calc)} rem="
        f"{engine.num_levels - int(out_ct[0].level_calc)}",
        flush=True,
    )

    he128 = _decrypt_he128(engine, sk, out_ct)

    err128 = pack_max_abs_diff(he128, plain128)
    g128 = pack_ls_gain(he128, plain128)
    he128_s = [np.asarray(v, dtype=np.float64) * g128 for v in he128]
    err128_ls = pack_max_abs_diff(he128_s, plain128)

    j_he = max(
        float(np.max(np.abs(he128[i * 16] - he128[i * 16 + 1]))) for i in range(4)
    )
    j_pt = max(
        float(np.max(np.abs(plain128[i * 16] - plain128[i * 16 + 1])))
        for i in range(4)
    )

    p8_he = decode_alpha_128_to_packs8(he128, extract_gain=2.0)
    p8_pt = decode_alpha_128_to_packs8(plain128, extract_gain=2.0)
    d_he = decode_score_packs(p8_he, cfg)
    d_pt = decode_score_packs(p8_pt, cfg)
    g8 = pack_ls_gain(p8_he, p8_pt)
    err8 = float(np.max(np.abs(d_he - d_pt)))
    err8_ls = float(np.max(np.abs(d_he * g8 - d_pt)))

    kv = np.ones(cfg.seq_len, dtype=np.float64)
    rs_he, mean_he = _row_sum_stats(d_he, kv)
    rs_pt, mean_pt = _row_sum_stats(d_pt, kv)

    print(
        f"=== HE vs plain full exit  task={args.task} L{args.layer} "
        f"level={args.level} seed={args.seed} ===",
        flush=True,
    )
    print(
        f"  128-pack: max|err|={err128:.3e}  ls_gain={g128:.6g}  "
        f"max|err|@gain={err128_ls:.3e}",
        flush=True,
    )
    print(
        f"  j-spread HE={j_he:.3e}  plain={j_pt:.3e}",
        flush=True,
    )
    print(
        f"  8pack decode: max|err|={err8:.3e}  ls_gain={g8:.6g}  "
        f"max|err|@gain={err8_ls:.3e}",
        flush=True,
    )
    print(
        f"  row-sum max|Σ−1|  HE={rs_he:.3e} (mean={mean_he:.4f})  "
        f"plain={rs_pt:.3e} (mean={mean_pt:.4f})",
        flush=True,
    )
    print(
        f"  max|α| decode  HE={float(np.max(np.abs(d_he))):.3e}  "
        f"plain={float(np.max(np.abs(d_pt))):.3e}",
        flush=True,
    )

    ok = err128_ls < args.tol or err8_ls < args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
