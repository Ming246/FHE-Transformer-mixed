"""
Backward-compatible DualRail ``f_cost`` wrapper.

Prefer ``from cost import compute_f_cost, compute_scheme_bts``.
This module re-exports the root API so ``smoke_cost_bts.py`` /
``evolution_*_bts.py`` keep working.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cost import (  # noqa: E402
    BTS_DEPTH_BUDGET,
    COST_DEPTH_DIVISOR,
    compute_f_cost,
    compute_scheme_bts,
    compute_scheme_bts_detail,
    depth_f_cost_label,
    depth_sum_to_f_cost,
    f_cost_label,
    validate_scheme,
)
from bts_ops import BOOTSTRAP_DEPTH_BUDGET  # noqa: E402

assert int(BOOTSTRAP_DEPTH_BUDGET) == int(BTS_DEPTH_BUDGET)

__all__ = [
    "BOOTSTRAP_DEPTH_BUDGET",
    "BTS_DEPTH_BUDGET",
    "COST_DEPTH_DIVISOR",
    "compute_f_cost",
    "compute_scheme_bts",
    "compute_scheme_bts_detail",
    "depth_f_cost_label",
    "depth_sum_to_f_cost",
    "f_cost_label",
    "validate_scheme",
]
