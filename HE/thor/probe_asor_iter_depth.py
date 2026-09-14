#!/usr/bin/env python3
"""
Measure HE rem burn of one Softmax aSOR iteration (``he_asor_ct`` graph).

Encrypt at high rem, run several iters, print rem after each CT op.

Usage::

  python3 HE/thor/probe_asor_iter_depth.py
"""
from __future__ import annotations

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

from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys  # noqa: E402
from softmax_he_ct import _DeltaCt  # noqa: E402


def _rem(engine, ct) -> int:
    return int(engine.num_levels) - int(ct.level_calc)


def _log(tag: str, engine, an: _DeltaCt, bn: _DeltaCt) -> None:
    print(
        f"  {tag:40s}  an.lc={int(an.ciphertext.level_calc):3d} rem={_rem(engine, an.ciphertext):2d}  "
        f"bn.lc={int(bn.ciphertext.level_calc):3d} rem={_rem(engine, bn.ciphertext):2d}  "
        f"δa={an.delta:.4g} δb={bn.delta:.4g}",
        flush=True,
    )


def main() -> int:
    print("create engine + keys (rot only) ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    sk, pk = keys["sk"], keys["pk"]
    print(
        f"  ready ({time.time() - t0:.1f}s) num_levels={engine.num_levels} "
        f"slots={engine.num_slots}",
        flush=True,
    )

    # Land with plenty of rem (same land as mock bts ≈ lc15 rem14).
    land = int(engine.num_levels) - 14
    n = int(engine.num_slots)
    # Positive denom in (0,1]-ish Softmax σ range; num=1 for inv≈1/denom.
    rng = np.random.default_rng(0)
    denom_msg = 0.2 + 0.5 * rng.random(n)
    num_msg = np.ones(n, dtype=np.float64)
    an = _DeltaCt(engine.encodecrypt(num_msg, pk, level=land), 1.0)
    bn = _DeltaCt(engine.encodecrypt(denom_msg, pk, level=land), 1.0)
    en = float(np.min(denom_msg) / 2.0)  # e0 style
    print(
        f"\nstart land_lc={land} rem={_rem(engine, an.ciphertext)} e0={en:.6g}",
        flush=True,
    )
    _log("entry", engine, an, bn)

    max_iters = 4
    rem_an = []
    rem_bn = []
    for i in range(max_iters):
        rem_an.append(_rem(engine, an.ciphertext))
        rem_bn.append(_rem(engine, bn.ciphertext))
        print(f"\n=== aSOR iter {i} (start rem an={rem_an[-1]} bn={rem_bn[-1]}) ===", flush=True)

        kn = 2.0 / (en + 1.0)
        # Match he_asor_ct body exactly (no mid-bts).
        b_temp = _DeltaCt(
            engine.negate(
                engine.add_scalar(bn.ciphertext, -2.0 / kn * bn.delta)
            ),
            bn.delta,
        )
        _log("after b_temp=add_scalar/negate", engine, an, b_temp)

        an1 = _DeltaCt(
            engine.auto_ct_ct_mult(an.ciphertext, b_temp.ciphertext),
            an.delta * b_temp.delta,
        )
        _log("after an ← an×b_temp (ct×ct)", engine, an1, bn)

        an1 = _DeltaCt(an1.ciphertext, an1.delta / (kn**2))
        _log("after an.delta /= kn² (no CT)", engine, an1, bn)

        bn1 = _DeltaCt(
            engine.auto_ct_ct_mult(bn.ciphertext, b_temp.ciphertext),
            bn.delta * b_temp.delta,
        )
        _log("after bn ← bn×b_temp (ct×ct)", engine, an1, bn1)

        bn1 = _DeltaCt(bn1.ciphertext, bn1.delta / (kn**2))
        en = kn * en * (2.0 - kn * en)

        scale_adjust = int(1.0 / bn1.delta / 2**8)
        if scale_adjust > 1:
            conj_an = engine.conjugate(an1.ciphertext)
            an1 = _DeltaCt(engine.add(an1.ciphertext, conj_an), an1.delta * 2.0)
            conj_bn = engine.conjugate(bn1.ciphertext)
            bn1 = _DeltaCt(engine.add(bn1.ciphertext, conj_bn), bn1.delta * 2.0)
            scale_adjust = int(1.0 / bn1.delta / 2**8)
            _log("after conj-fold (scale_adjust>1)", engine, an1, bn1)
        else:
            scale_adjust = 1
        bn1 = _DeltaCt(
            engine.mult_int_scalar(bn1.ciphertext, scale_adjust),
            bn1.delta * scale_adjust,
        )
        an1 = _DeltaCt(
            engine.mult_int_scalar(an1.ciphertext, scale_adjust),
            an1.delta * scale_adjust,
        )
        _log(f"after mult_int_scalar(×{scale_adjust})", engine, an1, bn1)

        an, bn = an1, bn1
        d_an = rem_an[-1] - _rem(engine, an.ciphertext)
        d_bn = rem_bn[-1] - _rem(engine, bn.ciphertext)
        print(
            f"  >> iter {i} Δrem an={d_an} bn={d_bn}  "
            f"(end rem an={_rem(engine, an.ciphertext)} bn={_rem(engine, bn.ciphertext)})",
            flush=True,
        )

    print("\n=== summary (an rem per iter start) ===", flush=True)
    rem_an.append(_rem(engine, an.ciphertext))
    for i in range(max_iters):
        print(
            f"  iter {i}: rem {rem_an[i]} → {rem_an[i+1]}  Δ={rem_an[i]-rem_an[i+1]}",
            flush=True,
        )
    deltas = [rem_an[i] - rem_an[i + 1] for i in range(max_iters)]
    print(f"  per-iter Δrem = {deltas}  (mean={sum(deltas)/len(deltas):.2f})", flush=True)
    print(
        "  plaintext formula kn*b*b_temp suggests depth 2; "
        "HE DeltaCt path may burn differently — see Δ above.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
