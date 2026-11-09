"""Pre/post message scaling around Liberate ``engine.bootstrap``.

Goal: amplify plaintext before bootstrap (better absolute SNR), then restore.

- **Pre** ``×BTS_MSG_SCALE`` via ``mult_int_scalar`` — no rescale, **0 rem**.
- **Post** ``×BTS_MSG_INV`` via ``mult_scalar`` (rescale) — **1 rem**.
  - ``direct``: applied immediately after bootstrap.
  - ``real_pack``: fused into DualRail unpack ``×½`` → single ``×(½·INV)``.

Both pack modes thus land at ``budget − 1`` after a full refresh.
"""
from __future__ import annotations

from typing import Any

# Integer amplify before bts (mult_int_scalar, no depth).
BTS_MSG_SCALE: int = 128
# Restore after bts (mult_scalar + rescale → 1 rem), or fused into DualRail ×½.
BTS_MSG_INV: float = 1.0/128.0
# Rem tax paid once per bootstrap refresh (direct postscale or real_pack unpack).
BTS_POST_DEPTH: int = 1
DUALRAIL_HALF: float = 0.5


def dualrail_unpack_scale(half: float = DUALRAIL_HALF) -> float:
    """Fused DualRail unpack ×½ and message ×INV (one rescale / 1 rem)."""
    return float(half) * float(BTS_MSG_INV)


def bts_prescale(engine: Any, ct: Any) -> Any:
    """``×BTS_MSG_SCALE`` with no depth (integer scalar)."""
    return engine.mult_int_scalar(ct, int(BTS_MSG_SCALE))


def bts_postscale(engine: Any, ct: Any) -> Any:
    """``×BTS_MSG_INV`` with rescale (1 rem)."""
    return engine.mult_scalar(ct, float(BTS_MSG_INV))


def bootstrap_prescaled(engine: Any, ct: Any, *, postscale: bool = True) -> Any:
    """
    Prescale → bootstrap → optional postscale.

    ``postscale=False`` when DualRail unpack will apply ``dualrail_unpack_scale``.
    """
    ct = bts_prescale(engine, ct)
    ct = engine.bootstrap(ct)
    if postscale:
        ct = bts_postscale(engine, ct)
    return ct
