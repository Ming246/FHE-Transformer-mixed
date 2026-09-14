#!/usr/bin/env python3
"""
Alias for root ``evolution_score.py`` (now has DualRail ``cost_mode=bts``).

  python3 HE/thor/evolution_score_bts.py --tasks mrpc
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import cost_bts  # noqa: E402
import evolution_score as _ev  # noqa: E402

# ``evaluate_individual`` resolves ``compute_f_cost`` via this module global.
_ev.compute_f_cost = cost_bts.compute_f_cost  # type: ignore[assignment]


if __name__ == "__main__":
    _ev.main()
