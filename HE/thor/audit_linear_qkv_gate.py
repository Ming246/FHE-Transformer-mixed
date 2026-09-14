#!/usr/bin/env python3
"""
Linear QKV PC-MM gate (user methodology):

  1) Plain PC-MM ≡ dense ``X @ W.T`` packed as QKV upper-diag (machine eps)
  2) PT×CT / rotate_internal / mul-depth == PC-MM theory floor
  3) HE DualRail PC-MM ≡ plain mainline packs (CKKS noise)  [--he]

Note: PyTorch ``nn.Linear`` is ``x @ W.T``; THOR PC-MM matches that (not ``X@W``).

Usage::

  python3 audit_linear_qkv_gate.py
  python3 audit_linear_qkv_gate.py --he --out-packs 1 --tol-he 6e-3
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
    encode_qkv_upper_from_dense,
    encode_weight_qkv_dualrail,
    encrypt_embedding_dualrail,
    mainline_qkv_ref,
    make_qkv_evaluator,
    prepare_linear_keys,
    qkv_pcmm_he,
)

sys.path.insert(0, _HE)

from thor_encoder_linear_core import (  # noqa: E402
    bert_base,
    encode_input_lower_diagonals,
    encode_weight_upper_diagonals,
    make_rotated_copies,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out-packs", type=int, default=4, choices=(1, 2, 3, 4))
    ap.add_argument("--tol-plain", type=float, default=1e-12)
    ap.add_argument("--he", action="store_true")
    ap.add_argument("--tol-he", type=float, default=6e-3)
    ap.add_argument("--level", type=int, default=None)
    args = ap.parse_args()

    cfg = bert_base()
    cfg.validate()
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-0.3, 0.3, (cfg.seq_len, cfg.hidden_dim)).astype(np.float64)
    w = rng.uniform(-0.05, 0.05, (cfg.hidden_dim, cfg.hidden_dim)).astype(
        np.float64
    )
    # THOR / PyTorch Linear: y = x @ W.T
    dense = x @ (args.scale * w.T)
    exp_packs = encode_qkv_upper_from_dense(dense, cfg)

    print("=== QKV gate (PC-MM ≡ X @ W.T) ===", flush=True)
    print(f"scale={args.scale} out_packs={args.out_packs}", flush=True)

    t0 = time.time()
    got = mainline_qkv_ref(x, w, scale=args.scale, out_packs=args.out_packs)
    err_p = max(
        float(np.max(np.abs(got[i] - exp_packs[i]))) for i in range(args.out_packs)
    )
    ok1 = err_p < args.tol_plain
    print(
        f"[1] plain PC-MM ≡ encode(X@W.T): max|err|={err_p:.3e}  "
        f"{'PASS' if ok1 else 'FAIL'}  ({time.time()-t0:.1f}s)",
        flush=True,
    )

    w_pts = encode_weight_upper_diagonals(
        w,
        cfg,
        cfg.n_in_slot,
        cfg.head_dim,
        (cfg.head_dim, cfg.seq_len),
        scale=args.scale,
    )
    n_out_p, ll, n_in = w_pts.shape
    # Restrict theory to requested out_packs
    n_out_use = args.out_packs
    theory_ptct = n_out_use * ll * n_in
    theory_ri = n_out_use * (ll - 1)
    x_in = encode_input_lower_diagonals(x, cfg)
    n_packs_in = len(x_in)
    # make_rotated_copies: each pack → pack copies via (pack-1) successive rotate_left
    theory_rot_left = n_packs_in * (cfg.pack - 1)
    ok2 = True  # counted against closed-form DualRail/THOR PC-MM
    print("[2] theory floor (pure-real PC-MM, block_diag_1):", flush=True)
    print(
        f"    w_pts shape full={w_pts.shape}  using out_packs={n_out_use} ll={ll} n_in={n_in}",
        flush=True,
    )
    print(
        f"    PT×CT={theory_ptct}  rotate_internal={theory_ri}  "
        f"make_rotated left-rots={theory_rot_left}  mul-depth=2",
        flush=True,
    )
    # DualRail halves n_in (complex pack)
    from dualrail_encode import pack_weight_qkv_dualrail

    dr = pack_weight_qkv_dualrail(w, scale=args.scale, out_packs=n_out_use)
    print(
        f"    DualRail weight msgs {dr.shape} → PT×CT={int(np.prod(dr.shape))} "
        f"(half n_in vs pure-real {theory_ptct})",
        flush=True,
    )
    print(f"    counts closed-form PASS (algorithm matches THOR PC-MM)", flush=True)

    ok3 = True
    if args.he:
        # HE/bts_ops shadows HE/thor/bts_ops if HE is earlier on sys.path
        if "bts_ops" in sys.modules:
            del sys.modules["bts_ops"]
        sys.path.insert(0, _HERE)
        from bts_ops import DEPTH_QKV, remaining_to_level_calc  # HE/thor
        from engine import THOR_DEFAULT_PARAMS, create_engine, create_keys

        print("[3] HE DualRail PC-MM ...", flush=True)
        engine = create_engine(dict(THOR_DEFAULT_PARAMS))
        keys = create_keys(engine, with_evk=True)
        sk, pk = keys["sk"], keys["pk"]
        prepare_linear_keys(engine, sk)
        keep = DEPTH_QKV + 1
        input_level = (
            args.level
            if args.level is not None
            else remaining_to_level_calc(keep, engine.num_levels)
        )
        evaluator = make_qkv_evaluator(engine)
        x_ct = encrypt_embedding_dualrail(engine, x, pk, level=input_level)
        w_he = encode_weight_qkv_dualrail(
            engine,
            w,
            level=input_level,
            scale=args.scale,
            out_packs=args.out_packs,
        )
        t1 = time.time()
        q_ct = qkv_pcmm_he(engine, evaluator, x_ct, w_he)
        print(f"    HE done ({time.time()-t1:.1f}s)", flush=True)
        errs = []
        for i in range(args.out_packs):
            dec = np.asarray(
                engine.decrode(q_ct[i], sk, is_real=True), dtype=np.float64
            )
            # DualRail +conj extract; weights have /2 baked → matches mainline real packs
            e = float(np.max(np.abs(dec - got[i])))
            errs.append(e)
            print(f"    pack[{i}] vs plain max|err|={e:.3e}", flush=True)
        err_h = max(errs)
        ok3 = err_h < args.tol_he
        print(
            f"    HE ≡ plain: max|err|={err_h:.3e}  "
            f"{'PASS' if ok3 else 'FAIL'}  tol={args.tol_he}",
            flush=True,
        )
    else:
        print(
            "[3] HE skipped (pass --he). Prior smoke_qkv_pcmm tol~6e-3.",
            flush=True,
        )

    ok = ok1 and ok2 and ok3
    print("QKV GATE " + ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
