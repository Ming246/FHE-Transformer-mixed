"""
HE 同构 GeLU（纯实数 slot 向量）：Chebyshev + PS-tree。

与 ``depth_he`` / cost 一致：平衡二叉树构造 T_k（深度 ~⌈log₂ d⌉），
再明文系数线性组合；复合 f1→f2 后 ``x·(0.5+y)``（+1）。
Clenshaw 仅作数值旁路参考，不在本路径使用。

原语：pt×ct / ct×ct（逐 slot）/ add / 明文常量。逐元素，无需 rotate。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from thor_encoder_linear_core import add_vec, pt_mult  # noqa: E402

from gelu_poly import gelu_config_for_layer  # noqa: E402


def _scale(v: np.ndarray, s: float) -> np.ndarray:
    return pt_mult(np.full_like(v, float(s), dtype=np.float64), v)


def _add_scalar(v: np.ndarray, s: float) -> np.ndarray:
    return add_vec(v, np.full_like(v, float(s), dtype=np.float64))


@dataclass(frozen=True)
class GeluHeParams:
    """与档位无关的 GeLU HE 参数包（切比雪夫）。"""

    kind: str
    C: float
    depth_he: int
    scheme_name: str
    f_cheb_coeffs: tuple[float, ...] | None = None
    f_domain: tuple[float, float] | None = None
    f1_cheb_coeffs: tuple[float, ...] | None = None
    f1_domain: tuple[float, float] | None = None
    f2_cheb_coeffs: tuple[float, ...] | None = None
    f2_domain: tuple[float, float] | None = None

    @staticmethod
    def from_gelu_poly(layer_idx: int, *, level: int = 2) -> "GeluHeParams":
        cfg = gelu_config_for_layer(layer_idx, level)
        if cfg["kind"] == "composite":
            return GeluHeParams(
                kind="composite",
                C=float(cfg["C"]),
                depth_he=int(cfg["depth_he"]),
                scheme_name=str(cfg["scheme_name"]),
                f1_cheb_coeffs=tuple(float(c) for c in cfg["f1_cheb_coeffs"]),
                f1_domain=(float(cfg["f1_domain"][0]), float(cfg["f1_domain"][1])),
                f2_cheb_coeffs=tuple(float(c) for c in cfg["f2_cheb_coeffs"]),
                f2_domain=(float(cfg["f2_domain"][0]), float(cfg["f2_domain"][1])),
            )
        return GeluHeParams(
            kind="single",
            C=float(cfg["C"]),
            depth_he=int(cfg["depth_he"]),
            scheme_name=str(cfg["scheme_name"]),
            f_cheb_coeffs=tuple(float(c) for c in cfg["f_cheb_coeffs"]),
            f_domain=(float(cfg["f_domain"][0]), float(cfg["f_domain"][1])),
        )


def affine_to_z_slots(
    x: np.ndarray, domain: tuple[float, float]
) -> np.ndarray:
    """z = (2x - a - b) / (b - a)（仅 pt×ct + 加常量）。"""
    a, b = float(domain[0]), float(domain[1])
    denom = b - a
    if denom == 0.0:
        raise ValueError(f"无效 Chebyshev domain {domain}")
    return _add_scalar(_scale(x, 2.0 / denom), -(a + b) / denom)


def _build_Tk_balanced_tree_slots(z: np.ndarray, max_k: int) -> list[np.ndarray]:
    """
    HE 同构：T_{a+b} = 2 T_a T_b - T_{|a-b|}（ct×ct），T_0=1，T_1=z。
    与 ``nolinear/gelu_chebyshev._build_Tk_balanced_tree`` 同构。
    """
    z = np.asarray(z, dtype=np.float64)
    T: list[np.ndarray | None] = [None] * (max_k + 1)

    def build(k: int) -> np.ndarray:
        cached = T[k]
        if cached is not None:
            return cached
        if k == 0:
            out = np.ones_like(z, dtype=np.float64)
        elif k == 1:
            out = z.copy()
        else:
            a = k // 2
            b = k - a
            ta = build(a)
            tb = build(b)
            tab = build(abs(a - b))
            # 2*Ta*Tb - Tab
            out = add_vec(_scale(pt_mult(ta, tb), 2.0), _scale(tab, -1.0))
        T[k] = out
        return out

    for k in range(max_k + 1):
        build(k)
    return [T[k] for k in range(max_k + 1)]  # type: ignore[misc]


def cheb_eval_ps_tree_slots(
    coeffs: tuple[float, ...] | list[float],
    z: np.ndarray,
) -> np.ndarray:
    """sum_k c_k T_k(z)：先 PS-tree 造 T_k，再明文系数 pt×ct 累加。"""
    c = [float(v) for v in coeffs]
    z = np.asarray(z, dtype=np.float64)
    if not c:
        return np.zeros_like(z)
    if len(c) == 1:
        return np.full_like(z, c[0])
    tk = _build_Tk_balanced_tree_slots(z, len(c) - 1)
    out = np.zeros_like(z)
    for k, ck in enumerate(c):
        if ck != 0.0:
            out = add_vec(out, _scale(tk[k], ck))
    return out


def cheb_eval_slots(
    coeffs: tuple[float, ...] | list[float],
    x: np.ndarray,
    domain: tuple[float, float],
) -> np.ndarray:
    return cheb_eval_ps_tree_slots(coeffs, affine_to_z_slots(x, domain))


def gelu_poly_slots(x: np.ndarray, params: GeluHeParams) -> np.ndarray:
    """单条 slot 向量上的 Chebyshev GeLU（PS-tree）：GELU≈ x·(0.5+y)。"""
    x = np.asarray(x, dtype=np.float64)
    t = _scale(x, 1.0 / float(params.C))
    if params.kind == "composite":
        assert params.f1_cheb_coeffs is not None and params.f1_domain is not None
        assert params.f2_cheb_coeffs is not None and params.f2_domain is not None
        f1 = cheb_eval_slots(params.f1_cheb_coeffs, t, params.f1_domain)
        y = add_vec(
            f1,
            cheb_eval_slots(params.f2_cheb_coeffs, f1, params.f2_domain),
        )
    else:
        assert params.f_cheb_coeffs is not None and params.f_domain is not None
        y = cheb_eval_slots(params.f_cheb_coeffs, t, params.f_domain)
    return pt_mult(x, _add_scalar(y, 0.5))


def slot_gelu_he(
    fc1_vecs: list[list[np.ndarray]] | list[np.ndarray],
    params: GeluHeParams,
) -> list[list[np.ndarray]] | list[np.ndarray]:
    """
    对 FC1 输出 slot 做 GeLU（layout 不变，PS-tree）。

    支持两路 ``[[rep0...], [rep1...]]`` 或单路 ``[vec, ...]``。
    """
    if not fc1_vecs:
        return fc1_vecs
    if isinstance(fc1_vecs[0], list):
        return [
            [gelu_poly_slots(v, params) for v in rep]  # type: ignore[arg-type]
            for rep in fc1_vecs  # type: ignore[assignment]
        ]
    return [gelu_poly_slots(v, params) for v in fc1_vecs]  # type: ignore[arg-type]
