"""Ciphertext pooler tanh: p2(p1(z/C)) with two bootstraps."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from tanh_poly import DEFAULT_COEFF_JSON, load_tanh_fit
from .polynomial import evaluate_polynomial_stockmeyer


def _pad16(coeffs: Sequence[float]) -> np.ndarray:
    c = np.asarray(list(coeffs), dtype=np.float64).ravel()
    if c.size < 16:
        c = np.pad(c, (0, 16 - int(c.size)))
    if c.size != 16:
        raise ValueError(f"Stockmeyer 需要 16 个升幂系数，得到 {c.size}")
    return c


def _tanh_coeffs(coeff_path: str | None, C: float | None) -> tuple[float, np.ndarray, np.ndarray]:
    path = coeff_path or DEFAULT_COEFF_JSON
    payload = load_tanh_fit(path)
    scale = float(C) if C is not None else float(payload["C"])
    p1 = _pad16(payload["p1_increasing"])
    p2 = _pad16(payload["p2_increasing"])
    return scale, p1, p2


def he_tanh_ct(engine, ct, *, p1: np.ndarray, p2: np.ndarray):
    """THOR he_tanh：bts → Stockmeyer(p1) → Stockmeyer(p2) → bts。"""
    ct = engine.bootstrap(ct)
    ct = evaluate_polynomial_stockmeyer(engine, p1, ct)
    ct = evaluate_polynomial_stockmeyer(engine, p2, ct)
    ct = engine.bootstrap(ct)
    return ct
