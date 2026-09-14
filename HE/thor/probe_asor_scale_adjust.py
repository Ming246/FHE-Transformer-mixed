#!/usr/bin/env python3
"""Probe whether softmax_he_ct scale_adjust block (170-186) burns HE rem."""
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


def _state(tag: str, engine, an: _DeltaCt, bn: _DeltaCt) -> None:
    print(
        f"  {tag:36s}  an lc={int(an.ciphertext.level_calc):3d} rem={_rem(engine, an.ciphertext):2d}  "
        f"bn lc={int(bn.ciphertext.level_calc):3d} rem={_rem(engine, bn.ciphertext):2d}  "
        f"δa={an.delta:.6g} δb={bn.delta:.6g}",
        flush=True,
    )


def _fresh_pair(engine, pk, *, land: int, bn_delta: float) -> tuple[_DeltaCt, _DeltaCt]:
    n = int(engine.num_slots)
    rng = np.random.default_rng(42)
    an = _DeltaCt(engine.encodecrypt(rng.random(n), pk, level=land), 1.0)
    bn = _DeltaCt(engine.encodecrypt(0.2 + 0.5 * rng.random(n), pk, level=land), bn_delta)
    return an, bn


def _run_scale_block(
    engine,
    an: _DeltaCt,
    bn: _DeltaCt,
    *,
    force_scale: int,
    use_conj_fold: bool,
) -> tuple[_DeltaCt, _DeltaCt]:
    """Mirror softmax_he_ct.py 170-186 with forced scale_adjust."""
    scale_adjust = int(force_scale)
    if use_conj_fold:
        conj_an = engine.conjugate(an.ciphertext)
        an = _DeltaCt(engine.add(an.ciphertext, conj_an), an.delta * 2.0)
        conj_bn = engine.conjugate(bn.ciphertext)
        bn = _DeltaCt(engine.add(bn.ciphertext, conj_bn), bn.delta * 2.0)
    bn = _DeltaCt(
        engine.mult_int_scalar(bn.ciphertext, scale_adjust),
        bn.delta * scale_adjust,
    )
    an = _DeltaCt(
        engine.mult_int_scalar(an.ciphertext, scale_adjust),
        an.delta * scale_adjust,
    )
    return an, bn


def _case(
    engine,
    pk,
    *,
    title: str,
    force_scale: int,
    use_conj_fold: bool,
    bn_delta: float,
) -> None:
    land = int(engine.num_levels) - 14
    an, bn = _fresh_pair(engine, pk, land=land, bn_delta=bn_delta)
    print(f"\n=== {title} (land rem={_rem(engine, an.ciphertext)}, force_scale={force_scale}) ===", flush=True)
    _state("before", engine, an, bn)
    an2, bn2 = _run_scale_block(
        engine, an, bn, force_scale=force_scale, use_conj_fold=use_conj_fold
    )
    _state("after block", engine, an2, bn2)
    d_an = _rem(engine, an.ciphertext) - _rem(engine, an2.ciphertext)
    d_bn = _rem(engine, bn.ciphertext) - _rem(engine, bn.ciphertext)
    print(
        f"  >> Δrem an={d_an} bn={d_bn}  (lc Δ an={int(an.ciphertext.level_calc)-int(an2.ciphertext.level_calc)} "
        f"bn={int(bn.ciphertext.level_calc)-int(bn2.ciphertext.level_calc)})",
        flush=True,
    )


def main() -> int:
    print("create engine + keys ...", flush=True)
    t0 = time.time()
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False, with_rot_deltas=True)
    pk = keys["pk"]
    print(f"  ready ({time.time() - t0:.1f}s) num_levels={engine.num_levels}", flush=True)

    # Case A: scale_adjust=1 only (no conj fold) — else branch
    _case(
        engine,
        pk,
        title="scale_adjust=1, no conj fold",
        force_scale=1,
        use_conj_fold=False,
        bn_delta=1.0,
    )

    # Case B: scale_adjust=10 only (no conj fold)
    _case(
        engine,
        pk,
        title="scale_adjust=10, no conj fold",
        force_scale=10,
        use_conj_fold=False,
        bn_delta=1.0,
    )

    # Case C: production path when delta tiny — conj fold then scale_adjust=10
    _case(
        engine,
        pk,
        title="conj fold + scale_adjust=10 (prod-like)",
        force_scale=10,
        use_conj_fold=True,
        bn_delta=1.0 / (2**10),
    )

    # Case D: natural prod trigger — run full formula from he_asor_ct
    land = int(engine.num_levels) - 14
    an, bn = _fresh_pair(engine, pk, land=land, bn_delta=1.0 / (2**10))
    print("\n=== natural scale_adjust from 1/bn.delta/2**8 ===", flush=True)
    _state("before", engine, an, bn)
    scale_adjust = int(1.0 / bn.delta / 2**8)
    print(f"  computed scale_adjust={scale_adjust}", flush=True)
    if scale_adjust > 1:
        conj_an = engine.conjugate(an.ciphertext)
        an = _DeltaCt(engine.add(an.ciphertext, conj_an), an.delta * 2.0)
        conj_bn = engine.conjugate(bn.ciphertext)
        bn = _DeltaCt(engine.add(bn.ciphertext, conj_bn), bn.delta * 2.0)
        scale_adjust = int(1.0 / bn.delta / 2**8)
        print(f"  after conj fold scale_adjust={scale_adjust}", flush=True)
    else:
        scale_adjust = 1
    an, bn = _run_scale_block(
        engine, an, bn, force_scale=scale_adjust, use_conj_fold=False
    )
    _state("after block", engine, an, bn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
