"""
GeLU 多项式近似：Chebyshev 系数 + Clenshaw 求值。

系数来源：nolinear/gelu/gelu_chebyshev_coeffs_C{C}.json
GELU(x) = x * (0.5 + y(t))，t = x/C；复合 y ≈ f1(t)+f2(f1(t))，单段 y ≈ f(t)。

用法：
    from gelu_poly import GeluPolyEvaluator, gelu_config_for_layer
    y = GeluPolyEvaluator(layer_idx=0, level=2)(x)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

GELU_LEVEL_KEYS = ("low", "mid", "high")

GELU_POLY_FUNC_NAMES: list[str] = [
    "poly_gelu_low",
    "poly_gelu_mid",
    "poly_gelu_high",
]

# 层组方案：(C, scheme_name)
_GROUP_A_LAYERS = frozenset({0, 1, 6, 7, 8, 11})
_GROUP_B_LAYERS = frozenset({2, 3, 4, 5})
_GROUP_C_LAYERS = frozenset({9, 10})

_GELU_LAYER_SCHEME: dict[int, dict[int, tuple[int, str]]] = {}

for _layer in _GROUP_A_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        2: (40, "composite_15_23"),
        1: (40, "composite_13_15"),
        0: (40, "composite_7_15"),
    }
for _layer in _GROUP_B_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        2: (80, "composite_27_31"),
        1: (80, "composite_15_23"),
        0: (80, "composite_15_15"),
    }
for _layer in _GROUP_C_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        2: (160, "composite_31_31"),
        0: (160, "composite_15_31"),
    }

GELU_FORBIDDEN_LEVELS: dict[int, frozenset[int]] = {
    **{layer: frozenset({1}) for layer in _GROUP_C_LAYERS},
}

_CHEB_JSON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nolinear", "gelu"
)

_POLY_EVAL_DTYPE = torch.float64


def gelu_level_allowed(layer_idx: int, level: int) -> bool:
    """该层是否允许使用该 GeLU 档位（0/1/2 = low/mid/high）。"""
    if level not in (0, 1, 2):
        return False
    return level not in GELU_FORBIDDEN_LEVELS.get(layer_idx, frozenset())


@lru_cache(maxsize=None)
def _load_cheb_json(C: int) -> dict:
    path = os.path.join(_CHEB_JSON_DIR, f"gelu_chebyshev_coeffs_C{int(C)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chebyshev 系数文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _scheme_config(C: int, scheme_name: str) -> dict:
    data = _load_cheb_json(C)
    for item in data["schemes"]:
        if item["name"] == scheme_name:
            return _normalize_scheme(item)
    raise KeyError(f"C={C} 中未找到方案 {scheme_name!r}")


def _normalize_scheme(item: dict) -> dict:
    p = item["params"]
    cfg: dict = {
        "basis": "chebyshev",
        "kind": item["kind"],
        "C": float(item["C"]),
        "scheme_name": item["name"],
        "depth_he": int(item["depth"]),
        "gelu_max_err": float(item["gelu_max_err"]),
    }
    if item["kind"] == "composite":
        cfg.update(
            {
                "d1": int(p["d1"]),
                "d2": int(p["d2"]),
                "f1_cheb_coeffs": list(p["f1_cheb_coeffs"]),
                "f1_domain": tuple(float(x) for x in p["f1_domain"]),
                "f2_cheb_coeffs": list(p["f2_cheb_coeffs"]),
                "f2_domain": tuple(float(x) for x in p["f2_domain"]),
            }
        )
        # Liberate PS-tree depth (combo rescale + f2 affine); JSON ``depth`` may lag.
        from nolinear.gelu_chebyshev import gelu_he_depth_breakdown

        bd = gelu_he_depth_breakdown(
            cfg["d1"],
            cfg["d2"],
            eval_method="ps_tree",
            f2_domain=cfg["f2_domain"],
        )
        cfg["depth_he"] = int(bd["total"])
        cfg["depth_he_breakdown"] = bd
    else:
        cfg.update(
            {
                "degree": int(p["degree"]),
                "f_cheb_coeffs": list(p["f_cheb_coeffs"]),
                "f_domain": tuple(float(x) for x in p["f_domain"]),
            }
        )
        from nolinear.gelu_chebyshev import gelu_he_depth_breakdown

        bd = gelu_he_depth_breakdown(cfg["degree"], eval_method="ps_tree")
        cfg["depth_he"] = int(bd["total"])
        cfg["depth_he_breakdown"] = bd
    return cfg


def gelu_config_for_layer(layer_idx: int, level: int) -> dict:
    if not gelu_level_allowed(layer_idx, level):
        allowed = sorted(_GELU_LAYER_SCHEME.get(layer_idx, {}).keys())
        raise ValueError(
            f"layer {layer_idx} 禁止 GeLU 档位 {GELU_LEVEL_KEYS[level]}；"
            f"允许档位：{[GELU_LEVEL_KEYS[l] for l in allowed]}"
        )
    C, name = _GELU_LAYER_SCHEME[layer_idx][level]
    return _scheme_config(C, name)


def gelu_scale_for_layer(layer_idx: int) -> float:
    """该层 high 档使用的 C（兼容旧接口）。"""
    if layer_idx in _GELU_LAYER_SCHEME and 2 in _GELU_LAYER_SCHEME[layer_idx]:
        return float(_GELU_LAYER_SCHEME[layer_idx][2][0])
    raise KeyError(f"layer {layer_idx} 无 high 档配置")


# THOR / HE encode scale (FC1): see ``gelu_ff1_encode_scale``.
# KEY encode / score decode: ``softmax_poly.k_key_encode_scale`` /
# ``softmax_poly.score_hf_decode_bake`` (``1/(64·δ1·δ2)`` per layer).

from softmax_poly import k_key_encode_scale, score_hf_decode_bake  # noqa: F401


def gelu_ff1_encode_scale(layer_idx: int) -> float:
    """FC1 W/bias encode scale ``1/C`` for layer ``layer_idx``."""
    return 1.0 / gelu_scale_for_layer(layer_idx)


@torch.jit.script
def clenshaw_cheb_jit(coeffs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Clenshaw 求 sum_k c_k T_k(z)。"""
    b_kp2 = torch.zeros_like(z)
    b_kp1 = torch.zeros_like(z)
    n = coeffs.numel()
    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2.0 * z * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k
    return coeffs[0] + z * b_kp1 - b_kp2


def _affine_to_z(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    return (2.0 * x - a - b) / (b - a)


def _cheb_eval(
    coeffs_f64: torch.Tensor,
    x: torch.Tensor,
    domain: tuple[float, float],
) -> torch.Tensor:
    a, b = domain
    z = _affine_to_z(x, a, b)
    return clenshaw_cheb_jit(coeffs_f64, z)


def _coeffs_to_f64(coeffs: list, device: torch.device) -> torch.Tensor:
    return torch.tensor(coeffs, device=device, dtype=_POLY_EVAL_DTYPE)


def _composite_gelu_cheb(
    x: torch.Tensor,
    cfg: dict,
    f1_f64: torch.Tensor,
    f2_f64: torch.Tensor,
) -> torch.Tensor:
    x64 = x.to(_POLY_EVAL_DTYPE)
    t = x64 / cfg["C"]
    f1 = _cheb_eval(f1_f64, t, cfg["f1_domain"])
    y = f1 + _cheb_eval(f2_f64, f1, cfg["f2_domain"])
    return (x64 * (0.5 + y)).to(x.dtype)


def _single_gelu_cheb(
    x: torch.Tensor, cfg: dict, f_f64: torch.Tensor
) -> torch.Tensor:
    x64 = x.to(_POLY_EVAL_DTYPE)
    t = x64 / cfg["C"]
    y = _cheb_eval(f_f64, t, cfg["f_domain"])
    return (x64 * (0.5 + y)).to(x.dtype)


def gelu_eval_from_config(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    device = x.device
    if cfg["kind"] == "composite":
        f1_f64 = _coeffs_to_f64(cfg["f1_cheb_coeffs"], device)
        f2_f64 = _coeffs_to_f64(cfg["f2_cheb_coeffs"], device)
        return _composite_gelu_cheb(x, cfg, f1_f64, f2_f64)
    f_f64 = _coeffs_to_f64(cfg["f_cheb_coeffs"], device)
    return _single_gelu_cheb(x, cfg, f_f64)


class GeluPolyEvaluator:
    """单层 GeLU Chebyshev 多项式；系数在首次 forward 时缓存到 x.device。"""

    __slots__ = ("layer_idx", "level", "_cfg", "_cached_key", "_f", "_f1", "_f2")

    def __init__(self, layer_idx: int, level: int):
        self.layer_idx = layer_idx
        self.level = level
        self._cfg = gelu_config_for_layer(layer_idx, level)
        self._cached_key: tuple[torch.device, torch.dtype] | None = None
        self._f: torch.Tensor | None = None
        self._f1: torch.Tensor | None = None
        self._f2: torch.Tensor | None = None

    def _ensure_coeff_tensors(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        key = (device, dtype)
        if self._cached_key == key:
            return
        if self._cfg["kind"] == "composite":
            self._f1 = _coeffs_to_f64(self._cfg["f1_cheb_coeffs"], device)
            self._f2 = _coeffs_to_f64(self._cfg["f2_cheb_coeffs"], device)
            self._f = None
        else:
            self._f = _coeffs_to_f64(self._cfg["f_cheb_coeffs"], device)
            self._f1 = None
            self._f2 = None
        self._cached_key = key

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_coeff_tensors(x.device, x.dtype)
        if self._cfg["kind"] == "composite":
            return _composite_gelu_cheb(x, self._cfg, self._f1, self._f2)
        return _single_gelu_cheb(x, self._cfg, self._f)


def poly_gelu_low(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 0))


def poly_gelu_mid(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 1))


def poly_gelu_high(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 2))


GELU_POLY_FUNCS: list = [
    poly_gelu_low,
    poly_gelu_mid,
    poly_gelu_high,
]
