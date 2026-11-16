"""
GeLU 多项式近似：Chebyshev 系数 + Clenshaw 求值。

系数来源：
  - high 复合：nolinear/gelu/gelu_chebyshev_coeffs_C{C}.json（层组共用）
  - low 直接 GELU：nolinear/gelu/gelu_direct_layer_coeffs.json（每层独立次数/拟合区间）

复合：GELU(x) = x * (0.5 + y(t))，t = x/C，y ≈ f1(t)+f2(f1(t))
直接：ĝ(x) = Chebyshev(x) ≈ GELU(x)，f_domain=LAYER_FIT_X（无 /C、无 reconstruct）

用法：
    from gelu_poly import GeluPolyEvaluator, gelu_config_for_layer
    y = GeluPolyEvaluator(layer_idx=0, level=1)(x)
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import torch

GELU_LEVEL_KEYS = ("low", "high")  # 档位 0/1；旧 mid 方案保留在下方注释

GELU_POLY_FUNC_NAMES: list[str] = [
    "poly_gelu_low",
    "poly_gelu_high",
]

# high：(C, scheme_name) 仍按层组共用；low 每层独立，见 gelu_direct_layer_coeffs.json
_GROUP_A_LAYERS = frozenset({0, 1, 6, 7, 8, 11})
_GROUP_B_LAYERS = frozenset({2, 3, 4, 5})
_GROUP_C_LAYERS = frozenset({9, 10})

_GELU_LAYER_SCHEME: dict[int, dict[int, tuple[int, str] | None]] = {}

for _layer in _GROUP_A_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        1: (40, "composite_15_23"),  # high（层组共用）
        0: None,  # low：每层独立 direct
    }
for _layer in _GROUP_B_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        1: (80, "composite_27_31"),
        0: None,
    }
for _layer in _GROUP_C_LAYERS:
    _GELU_LAYER_SCHEME[_layer] = {
        1: (160, "composite_31_31"),
        0: None,
    }

_DIRECT_LAYER_JSON = "gelu_direct_layer_coeffs.json"

# 历史：组 C 曾禁止 mid；现仅 low/high，无额外禁止档
GELU_FORBIDDEN_LEVELS: dict[int, frozenset[int]] = {}

_CHEB_JSON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nolinear", "gelu"
)

_POLY_EVAL_DTYPE = torch.float64


def gelu_level_allowed(layer_idx: int, level: int) -> bool:
    """该层是否允许使用该 GeLU 档位（0/1 = low/high）。"""
    if level not in (0, 1):
        return False
    if level not in _GELU_LAYER_SCHEME.get(layer_idx, {}):
        return False
    return level not in GELU_FORBIDDEN_LEVELS.get(layer_idx, frozenset())


_SCHEME_JSON_STEMS = (
    "gelu_chebyshev_coeffs",
    "gelu_direct_coeffs",
)


@lru_cache(maxsize=None)
def _load_scheme_json(C: int, stem: str) -> dict:
    path = os.path.join(_CHEB_JSON_DIR, f"{stem}_C{int(C)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chebyshev 系数文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _load_cheb_json(C: int) -> dict:
    return _load_scheme_json(C, "gelu_chebyshev_coeffs")


@lru_cache(maxsize=None)
def _scheme_config(C: int, scheme_name: str) -> dict:
    seen = []
    for stem in _SCHEME_JSON_STEMS:
        try:
            data = _load_scheme_json(C, stem)
        except FileNotFoundError:
            continue
        seen.append(stem)
        for item in data["schemes"]:
            if item["name"] == scheme_name:
                return _normalize_scheme(item)
    raise KeyError(
        f"C={C} 中未找到方案 {scheme_name!r}（已查 {seen or _SCHEME_JSON_STEMS}）"
    )


@lru_cache(maxsize=None)
def _load_direct_layer_json() -> dict:
    path = os.path.join(_CHEB_JSON_DIR, _DIRECT_LAYER_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(f"每层 low GeLU 系数文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _low_config_for_layer(layer_idx: int) -> dict:
    data = _load_direct_layer_json()
    for item in data.get("schemes", []):
        if int(item.get("layer", -1)) == int(layer_idx):
            return _normalize_scheme(item)
    raise KeyError(
        f"{_DIRECT_LAYER_JSON} 中未找到 layer={layer_idx} 的 low 方案"
    )


def _normalize_scheme(item: dict) -> dict:
    p = item["params"]
    kind = str(item["kind"])
    cfg: dict = {
        "basis": "chebyshev",
        "kind": kind,
        "scheme_name": item["name"],
        "depth_he": int(item["depth"]),
        "gelu_max_err": float(item["gelu_max_err"]),
    }
    if kind != "direct":
        cfg["C"] = float(item["C"])
    if kind == "composite":
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
        if kind == "direct":
            # ĝ(x)=Cheb(x) on f_domain；无 /C、无 reconstruct。定义域非 [-1,1] 时 affine_to_z +1。
            from nolinear.gelu_chebyshev import _domain_is_unit_depth

            affine = 0 if _domain_is_unit_depth(cfg["f_domain"]) else 1
            total = (
                int(bd["y_eval"]) + int(bd["cheb_combo_rescale"]) + affine
            )
            cfg["depth_he"] = total
            cfg["depth_he_breakdown"] = {
                "y_eval": int(bd["y_eval"]),
                "cheb_combo_rescale": int(bd["cheb_combo_rescale"]),
                "domain_affine": affine,
                "gelu_reconstruct": 0,
                "total": total,
                "eval_method": bd["eval_method"],
            }
        else:
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
    if level == 0:
        return _low_config_for_layer(layer_idx)
    spec = _GELU_LAYER_SCHEME[layer_idx][level]
    if spec is None:
        raise ValueError(f"layer {layer_idx} 档位 {level} 无方案")
    C, name = spec
    return _scheme_config(C, name)


def gelu_scale_for_layer(layer_idx: int) -> float:
    """该层 high 档使用的 C（兼容旧接口）。"""
    if layer_idx in _GELU_LAYER_SCHEME and 1 in _GELU_LAYER_SCHEME[layer_idx]:
        return float(_GELU_LAYER_SCHEME[layer_idx][1][0])
    raise KeyError(f"layer {layer_idx} 无 high 档配置")


# THOR / HE encode scale (FC1): ``gelu_ff1_encode_scale(layer, level)``.
# KEY encode / score decode: ``softmax_poly.k_key_encode_scale`` /
# ``softmax_poly.score_hf_decode_bake`` (``1/(64·δ1·δ2)`` per layer).

from softmax_poly import k_key_encode_scale, score_hf_decode_bake  # noqa: F401


def gelu_ff1_encode_scale(layer_idx: int, level: int = 1) -> float:
    """FC1 W/bias encode scale for GeLU ``level`` (0=low, 1=high, 3=original).

    high：``1/C``（复合 Cheb 在 ``t=x/C``）；low / original：``1``（直接 Cheb 在 ``x``）。
    """
    if int(level) in (0, 3):
        return 1.0
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


def _direct_gelu_cheb(
    x: torch.Tensor, cfg: dict, f_f64: torch.Tensor
) -> torch.Tensor:
    x64 = x.to(_POLY_EVAL_DTYPE)
    return _cheb_eval(f_f64, x64, cfg["f_domain"]).to(x.dtype)


def gelu_eval_from_config(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    device = x.device
    if cfg["kind"] == "composite":
        f1_f64 = _coeffs_to_f64(cfg["f1_cheb_coeffs"], device)
        f2_f64 = _coeffs_to_f64(cfg["f2_cheb_coeffs"], device)
        return _composite_gelu_cheb(x, cfg, f1_f64, f2_f64)
    f_f64 = _coeffs_to_f64(cfg["f_cheb_coeffs"], device)
    if cfg["kind"] == "direct":
        return _direct_gelu_cheb(x, cfg, f_f64)
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
        if self._cfg["kind"] == "direct":
            return _direct_gelu_cheb(x, self._cfg, self._f)
        return _single_gelu_cheb(x, self._cfg, self._f)


def poly_gelu_low(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 0))


def poly_gelu_high(x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    return gelu_eval_from_config(x, gelu_config_for_layer(layer_idx, 1))


GELU_POLY_FUNCS: list = [
    poly_gelu_low,
    poly_gelu_high,
]
