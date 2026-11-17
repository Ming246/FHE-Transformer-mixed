#!/usr/bin/env python3
"""Re-encrypt saved L11 ln2 DualRail slots → HE pooler+classifier; compare logits."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from path_setup import ensure_thor_path

ensure_thor_path()

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from head_he_ct import HeBertHead, decrypt_logits_slot0  # noqa: E402
from research_unified_entry_slots import decode_layer_output_to_matrix  # noqa: E402
from tanh_poly import TanhPolyEvaluator  # noqa: E402
from thor.linear import ThorLinearEvaluator  # noqa: E402

DEFAULT_SLOTS = os.path.join(
    _REPO,
    "results",
    "he_chain",
    "mrpc_scheme_s1_sigma_atomic_slots",
    "L11_ln2_slots.npz",
)


def _plain_logits(hidden_2d, model, *, poly_tanh: bool, C: float) -> np.ndarray:
    x = torch.as_tensor(hidden_2d, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        dense = model.bert.pooler.dense(x[:, 0].to(next(model.parameters()).device))
        if poly_tanh:
            pooled = TanhPolyEvaluator(C=C)(dense)
        else:
            pooled = torch.tanh(dense)
        logits = model.classifier(pooled)
    return np.asarray(logits.detach().cpu().numpy(), dtype=np.float64).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default=DEFAULT_SLOTS)
    ap.add_argument("--dataset", default="mrpc")
    ap.add_argument("--he-head-c", type=float, default=5.0)
    ap.add_argument(
        "--bootstrap",
        choices=("real", "mock"),
        default="real",
        help="real=CKKS bootstrap in head; mock=decrypt→encodecrypt (debug)",
    )
    args = ap.parse_args()
    slot_path = args.slots if os.path.isabs(args.slots) else os.path.join(_REPO, args.slots)
    z = np.load(slot_path)
    packs = np.asarray(z["packs"], dtype=np.float64)
    if packs.shape != (8, 32768):
        raise SystemExit(f"packs {packs.shape}, expected (8, 32768)")
    lc = int(z["level_calc"])
    rem = int(z["rem"])
    layer = int(z["layer"])
    h = decode_layer_output_to_matrix([packs[i] for i in range(8)])
    print(
        f"slots {slot_path} L{layer} rem={rem} level_calc={lc} "
        f"decoded max|h|={float(np.max(np.abs(h))):.4g}",
        flush=True,
    )

    os.chdir(os.path.join(_REPO, "thirdparty", "THOR-main"))
    print("create engine + keys0 ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(
        engine,
        with_bootstrap=args.bootstrap == "real",
        with_rot_deltas=True,
    )
    sk, pk = keys["sk"], keys["pk"]
    print(f"  engine ready ({time.time() - t0:.1f}s) num_levels={engine.num_levels}", flush=True)
    if args.bootstrap == "mock":
        from dualrail_bts import install_mock_bootstrap

        install_mock_bootstrap(engine, sk, pk, rem_after=14)
        print("  mock bootstrap installed", flush=True)

    print("ThorLinearEvaluator (block_diag masks) ...", flush=True)
    evaluator = ThorLinearEvaluator(engine)
    enc_level = lc
    if int(z["num_levels"]) != int(engine.num_levels):
        print(
            f"  warn saved num_levels={z['num_levels']} != engine {engine.num_levels}; "
            f"encrypt at saved level_calc={enc_level}",
            flush=True,
        )
    print(f"re-encrypt 8 packs at level_calc={enc_level} (saved rem={rem})", flush=True)
    x = np.empty(8, dtype=object)
    for i in range(8):
        x[i] = engine.encode_and_encrypt(packs[i], pk, level=enc_level)

    head = HeBertHead(
        evaluator,
        dataset=args.dataset,
        tanh_C=float(args.he_head_c),
    )
    t1 = time.time()
    logits_ct = head.forward(x)
    he_logits = decrypt_logits_slot0(engine, sk, logits_ct)
    print(f"  HE head done ({time.time() - t1:.1f}s)", flush=True)

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(_REPO, "finetuned_weight", args.dataset),
        local_files_only=True,
    )
    model.eval()
    poly_logits = _plain_logits(h, model, poly_tanh=True, C=float(args.he_head_c))
    exact_logits = _plain_logits(h, model, poly_tanh=False, C=float(args.he_head_c))

    def _fmt(name, a):
        pred = int(np.argmax(a))
        s = ", ".join(f"{v:.6g}" for v in a)
        return f"{name:18s} logits=[{s}] pred={pred}"

    print(_fmt("HE", he_logits), flush=True)
    print(_fmt(f"poly-tanh C={args.he_head_c:g}", poly_logits), flush=True)
    print(_fmt("exact tanh", exact_logits), flush=True)
    d_poly = np.abs(he_logits - poly_logits)
    d_ex = np.abs(he_logits - exact_logits)
    print(
        f"  |HE−poly|  max={d_poly.max():.4e} mean={d_poly.mean():.4e} {d_poly.tolist()}",
        flush=True,
    )
    print(
        f"  |HE−exact| max={d_ex.max():.4e} mean={d_ex.mean():.4e} {d_ex.tolist()}",
        flush=True,
    )
    same_poly = int(np.argmax(he_logits)) == int(np.argmax(poly_logits))
    print(f"  pred match poly={same_poly}", flush=True)
    return 0 if same_poly and d_poly.max() < 0.5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
