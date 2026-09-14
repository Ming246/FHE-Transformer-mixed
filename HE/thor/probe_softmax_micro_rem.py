#!/usr/bin/env python3
"""
Standalone Softmax micro-event rem probe (no mid-segment bootstrap).

Encrypt score CTs at high enough remaining multiplicative depth so the full
poly Softmax path runs without plan bts / local align boots. Compare per-event
Δrem against ``bts_ops`` phase-C Event.depth.

  python3 HE/thor/probe_softmax_micro_rem.py --task mrpc --layer 0 --level 2
  python3 HE/thor/probe_softmax_micro_rem.py --start-rem 25 --tol 0
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


def _ensure_he_path() -> None:
    """``ensure_thor_path`` / ``linear_eval``→``dualrail_encode`` drop ``HE/``."""
    # insert(0) so a later ensure_thor_path removal is easy to undo
    if _HE in sys.path:
        sys.path.remove(_HE)
    sys.path.insert(0, _HE)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)


_ensure_he_path()

from bts_ops import build_bootstrap_events  # noqa: E402
from cost import NUM_LAYERS  # noqa: E402
from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402
from linear_eval import dense_qk_scores, encode_score_packs_from_dense  # noqa: E402

_ensure_he_path()

from slot_softmax_he import SoftmaxHeParams, encode_key_valid_mask_packs  # noqa: E402
from softmax_he_ct import he_softmax_poly  # noqa: E402
from softmax_poly import k_key_encode_scale  # noqa: E402
from thor_encoder_linear_core import bert_base  # noqa: E402


def _scheme(*, softmax: int, ln: int, gelu: int) -> list[int]:
    out: list[int] = []
    for li in range(NUM_LAYERS):
        g = gelu if gelu_level_allowed(li, gelu) else 2
        out.extend([softmax, ln, g, ln])
    return out


def _encode_attention_mask_pt(engine, attention_mask: np.ndarray, *, level: int):
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


def _forbid_bootstrap(engine) -> list[str]:
    """Replace ``engine.bootstrap`` so accidental mid-bts fail loudly."""
    hits: list[str] = []
    real = engine.bootstrap

    def _blocked(*args, **kwargs):
        hits.append("bootstrap")
        raise RuntimeError(
            "unexpected bootstrap during no-bts Softmax rem probe "
            f"(args={type(args[0]).__name__ if args else None})"
        )

    engine.bootstrap = _blocked  # type: ignore[method-assign]
    engine._probe_bootstrap_real = real  # type: ignore[attr-defined]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--start-rem",
        type=int,
        default=0,
        help="encrypt rem; 0 → auto = DP Softmax Σdepth + 3",
    )
    ap.add_argument("--tol", type=int, default=0)
    ap.add_argument("--no-forbid-bts", action="store_true")
    args = ap.parse_args()

    scheme = _scheme(softmax=args.level, ln=args.level, gelu=args.level)
    events = [
        e
        for e in build_bootstrap_events(args.task, scheme, phase="C")
        if e.layer_idx == int(args.layer)
        and e.name.split(".", 1)[1].startswith("softmax_")
    ]
    dp_depths = {e.name.split(".", 1)[1]: int(e.depth) for e in events}
    dp_order = [e.name.split(".", 1)[1] for e in events]
    declared_sum = sum(dp_depths.values())
    start_rem = int(args.start_rem) if args.start_rem > 0 else declared_sum + 3

    print(
        f"=== Softmax micro rem probe  task={args.task} L{args.layer} "
        f"level={args.level} start_rem={start_rem} (DP Σdepth={declared_sum}) ===",
        flush=True,
    )
    print("DP Softmax events:", flush=True)
    for name in dp_order:
        print(f"  {name:36s} depth={dp_depths[name]}", flush=True)

    cfg = bert_base()
    rng = np.random.default_rng(args.seed)
    params = SoftmaxHeParams.from_softmax_poly(
        args.task, args.layer, level=args.level
    )
    q = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    k = rng.normal(0.0, 0.05, (cfg.seq_len, cfg.hidden_dim))
    scale = 1.0 / np.sqrt(cfg.head_dim)
    scores = np.clip(dense_qk_scores(q, k, cfg, scale=scale), -8.0, 8.0)
    scores = scores * k_key_encode_scale(args.task, args.layer)
    packs = encode_score_packs_from_dense(scores, cfg)
    _ = encode_key_valid_mask_packs(np.ones(cfg.seq_len, dtype=np.float64), cfg)

    print("create engine + rot keys (no bootstrap key) ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    del sk
    nl = int(engine.num_levels)
    if start_rem >= nl:
        raise SystemExit(f"start_rem={start_rem} >= num_levels={nl}")
    enc_lc = nl - int(start_rem)
    print(
        f"  ready ({time.time()-t0:.1f}s) num_levels={nl} "
        f"encrypt level_calc={enc_lc} → rem={start_rem}",
        flush=True,
    )

    boot_hits: list[str] = []
    if not args.no_forbid_bts:
        boot_hits = _forbid_bootstrap(engine)

    x_ct = np.full((8,), None, dtype=object)
    for i in range(8):
        x_ct[i] = engine.encodecrypt(
            np.asarray(packs[i], dtype=np.float64), pk, level=enc_lc
        )
    mask_ct = _encode_attention_mask_pt(
        engine,
        np.ones(cfg.seq_len, dtype=np.float64),
        level=int(x_ct[0].level_calc),
    )

    probes: list[tuple[str, int]] = []

    def rem_probe(name: str, rem: int) -> None:
        probes.append((str(name), int(rem)))
        print(f"  [probe] {name:36s} rem={rem}", flush=True)

    print("HE Softmax (bts_hook=None) ...", flush=True)
    t1 = time.time()
    try:
        out = he_softmax_poly(
            engine,
            x_ct,
            mask_ct,
            params,
            layer_idx=int(args.layer),
            rescale=False,
            bts_hook=None,
            rem_probe=rem_probe,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}", flush=True)
        if boot_hits:
            print(f"  bootstrap attempts={len(boot_hits)}", flush=True)
        return 2
    print(
        f"  done ({time.time()-t1:.1f}s) out_rem="
        f"{nl - int(out[0].level_calc)} probes={len(probes)}",
        flush=True,
    )

    rem_by: dict[str, int] = {}
    for name, rem in probes:
        rem_by[name] = rem  # last write wins

    print("\n=== Δrem vs DP depth ===", flush=True)
    ok = bad = miss = 0
    prev_name = "entry"
    if "entry" not in rem_by:
        print("FAIL: missing entry probe", flush=True)
        return 2
    prev_rem = rem_by["entry"]
    print(f"  entry rem={prev_rem}", flush=True)

    for name in dp_order:
        if name not in rem_by:
            print(f"  [MISS] {name:36s} declared={dp_depths[name]}", flush=True)
            miss += 1
            continue
        cur = rem_by[name]
        measured = prev_rem - cur
        declared = dp_depths[name]
        diff = measured - declared
        flag = "OK" if abs(diff) <= int(args.tol) else "BAD"
        if flag == "OK":
            ok += 1
        else:
            bad += 1
        print(
            f"  [{flag}] {name:36s} HE_Δ={measured:3d} declared={declared:3d} "
            f"diff={diff:+d}  (rem {prev_rem}→{cur})",
            flush=True,
        )
        prev_name, prev_rem = name, cur

    he_sum = rem_by["entry"] - rem_by.get(
        "softmax_dualrail_exit", rem_by[prev_name]
    )
    print(
        f"\nsummary: ok={ok} bad={bad} miss={miss}  "
        f"HE_total_Δ={he_sum} DP_Σ={declared_sum}  "
        f"boot_hits={len(boot_hits)}",
        flush=True,
    )
    return 0 if bad == 0 and miss == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
