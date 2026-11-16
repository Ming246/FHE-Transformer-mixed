"""
BertPooler tanh 多项式：THOR 结构 ŷ = p2(p1(z/C))，普通基（升幂）。

默认 C=5，系数 nolinear/tanh_out/tanh_fit_C5_d15_15.json。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COEFF_JSON = os.path.join(
    _SCRIPT_DIR, "nolinear", "tanh_out", "tanh_fit_C5_d15_15.json"
)
TANH_C = 5.0
_POLY_EVAL_DTYPE = torch.float64


@lru_cache(maxsize=4)
def load_tanh_fit(path: str = DEFAULT_COEFF_JSON) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("kind") != "composite_p2_of_p1":
        raise ValueError(f"{path} kind 须为 composite_p2_of_p1")
    if payload.get("basis") != "monomial_increasing":
        raise ValueError(f"{path} basis 须为 monomial_increasing")
    return payload


def _horner_increasing(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    for k in range(int(coeffs.numel()) - 1, -1, -1):
        y = y * x + coeffs[k]
    return y


class TanhPolyEvaluator:
    """ŷ = p2(p1(z/C))，系数缓存在 x.device 上。"""

    __slots__ = ("C", "_p1", "_p2", "_cached_device")

    def __init__(self, coeff_path: str = DEFAULT_COEFF_JSON, C: float | None = None):
        payload = load_tanh_fit(coeff_path)
        self.C = float(C) if C is not None else float(payload["C"])
        self._p1 = torch.tensor(
            payload["p1_increasing"], dtype=_POLY_EVAL_DTYPE
        )
        self._p2 = torch.tensor(
            payload["p2_increasing"], dtype=_POLY_EVAL_DTYPE
        )
        self._cached_device: torch.device | None = None

    def _ensure(self, device: torch.device) -> None:
        if self._cached_device == device:
            return
        self._p1 = self._p1.to(device)
        self._p2 = self._p2.to(device)
        self._cached_device = device

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self._ensure(z.device)
        z64 = z.to(_POLY_EVAL_DTYPE)
        t = z64 / self.C
        u = _horner_increasing(t, self._p1)
        y = _horner_increasing(u, self._p2)
        return y.to(dtype=z.dtype)


def poly_pooler_tanh(z: torch.Tensor) -> torch.Tensor:
    return TanhPolyEvaluator()(z)
