"""
HE 同构 Softmax（纯实数 slot 向量）：thor exp (Stockmeyer) + aSOR。

原语仅：rotate / add / pt×ct / ct×ct（逐 slot 乘）/ 明文常量。
exp 多项式与 ``cost.SOFTMAX_DEPTH_BASE`` 一致：THOR Stockmeyer（deg15→深度4）
+ δ1 平方还原；不用 Horner（Horner 为 O(d)，与记账不符）。

参数通用（shift / δ1 / δ2 / e0 / max_iters）；档位只是参数组合。
测试默认 high 档以压低近似误差。
"""
from __future__ import annotations

import math
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

from thor_encoder_linear_core import (  # noqa: E402
    ThorConfig,
    _slot_index,
    add_vec,
    pt_mult,
    rotate_left,
)

from softmax_poly import (  # noqa: E402
    MAX_SUM_SQ_ROUNDS,
    THOR_EXP_POLY_COEFFS,
    THOR_INPUT_SCALE,
    _check_power_of_two,
    softmax_config_for_layer,
    thor_eps2_from_en,
)


def _scale(v: np.ndarray, s: float) -> np.ndarray:
    return pt_mult(np.full_like(v, float(s), dtype=np.float64), v)


def _add_scalar(v: np.ndarray, s: float) -> np.ndarray:
    return add_vec(v, np.full_like(v, float(s), dtype=np.float64))


def _const_like(v: np.ndarray, s: float) -> np.ndarray:
    return np.full_like(v, float(s), dtype=np.float64)


@dataclass(frozen=True)
class SoftmaxHeParams:
    """与档位无关的 Softmax HE 参数包。"""

    shift: float
    delta1: float
    delta2: float
    e0_sigma: float
    asor_max_iters_sigma: int
    asor_max_iters_sum_sq: tuple[int, int]
    exp_coeffs: tuple[float, ...]

    @staticmethod
    def from_softmax_poly(
        task_name: str,
        layer_idx: int,
        *,
        level: int = 2,
    ) -> "SoftmaxHeParams":
        """level: 0/1/2 = low/mid/high；正确性测试默认 high。"""
        cfg = softmax_config_for_layer(task_name, layer_idx, level)
        return SoftmaxHeParams(
            shift=float(cfg["shift"]),
            delta1=float(cfg["delta1"]),
            delta2=float(cfg["delta2"]),
            e0_sigma=float(cfg["e0_sigma"]),
            asor_max_iters_sigma=int(cfg["asor_max_iters_sigma"]),
            asor_max_iters_sum_sq=(
                int(cfg["asor_max_iters_sum_sq"][0]),
                int(cfg["asor_max_iters_sum_sq"][1]),
            ),
            exp_coeffs=tuple(float(c) for c in THOR_EXP_POLY_COEFFS),
        )


def rotsum(vec: np.ndarray, interval: int) -> np.ndarray:
    """THOR ``rotsum``：按 ``interval`` 倍旋转累加。"""
    s = len(vec)
    if s % interval != 0:
        raise ValueError(f"len={s} 须能被 interval={interval} 整除")
    rep = int(math.log2(s // interval))
    temp = np.asarray(vec, dtype=np.float64).copy()
    for i in range(rep):
        temp = add_vec(temp, rotate_left(temp, interval * (2**i)))
    return temp


def _sum_pack_vectors(vecs: list[np.ndarray]) -> np.ndarray:
    acc = np.asarray(vecs[0], dtype=np.float64).copy()
    for v in vecs[1:]:
        acc = add_vec(acc, np.asarray(v, dtype=np.float64))
    return acc


def sum_over_keys(packs: list[np.ndarray], cfg: ThorConfig) -> np.ndarray:
    """
    Σ_key：先对各对角 pack 逐 slot 相加，再 ``rotsum(slot_stride)`` 折叠 subpack。
    """
    if len(packs) != cfg.n_input_packs:
        raise ValueError(
            f"期望 {cfg.n_input_packs} 个 score pack，得到 {len(packs)}"
        )
    return rotsum(_sum_pack_vectors(packs), cfg.slot_stride)


def asor_inverse_slots(
    denom: np.ndarray,
    e0: float,
    max_iters: int,
) -> tuple[np.ndarray, float]:
    """aSOR 求 1/denom（每步 2×ct×ct，与 cost 的 iters*2 一致）。"""
    a = np.ones_like(denom, dtype=np.float64)
    b = np.asarray(denom, dtype=np.float64).copy()
    en = float(e0)
    if en <= 0.0:
        raise ValueError(f"e0 须 > 0，得到 {e0}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    for _ in range(max_iters):
        kn = 2.0 / (en + 1.0)
        b_temp = add_vec(
            np.full_like(b, 2.0),
            pt_mult(np.full_like(b, -kn), b),
        )
        b = pt_mult(np.full_like(b, kn), pt_mult(b, b_temp))
        a = pt_mult(np.full_like(a, kn), pt_mult(a, b_temp))
        en = kn * en * (2.0 - kn * en)
    return a, en


def stockmeyer_poly_slots(
    coeffs: tuple[float, ...] | list[float] | np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """
    THOR ``evaluate_polynomial_stockmeyer`` 的 slot 同构（幂基，升幂系数）。

    deg15（16 系数）：baby 宽 4 + x4/x8 巨步 → 关键路径 ct×ct 深度 4
   （对齐 ``cost.SOFTMAX_EXP_STOCKMEYER_DEPTH``）。
    """
    p = np.asarray(coeffs, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return _const_like(x, float(p[0]))
    if n % 4 != 0:
        raise ValueError(
            f"Stockmeyer 系数长度须为 4 的倍数（THOR baby=4），得到 {n}"
        )
    if n < 16:
        raise ValueError(
            f"当前仅实现 THOR 路径 len≥16（got {n}）；thor exp 为 16 系数"
        )

    x2 = pt_mult(x, x)
    # x3 在 THOR 中预计算；baby 用 x2*(c3*x)，与 c3*x3 同构

    def evaluate_baby_poly(chunk: np.ndarray) -> np.ndarray:
        c = np.asarray(chunk, dtype=np.float64).ravel()
        if c.size == 1:
            return _const_like(x, float(c[0]))
        result = _scale(x, float(c[1]))
        if c.size == 2:
            return _add_scalar(result, float(c[0]))
        if c.size == 3:
            result = add_vec(result, _scale(x2, float(c[2])))
            return _add_scalar(result, float(c[0]))
        result = add_vec(result, _scale(x2, float(c[2])))
        ax3 = pt_mult(x2, _scale(x, float(c[3])))
        result = add_vec(result, ax3)
        return _add_scalar(result, float(c[0]))

    x4 = pt_mult(x2, x2)
    x8 = pt_mult(x4, x4)

    if n == 16:
        subpolys = np.split(p, 4)
        babies = [evaluate_baby_poly(sp) for sp in subpolys]
        gs1 = [
            add_vec(babies[0], pt_mult(babies[1], x4)),
            add_vec(babies[2], pt_mult(babies[3], x4)),
        ]
        return add_vec(gs1[0], pt_mult(gs1[1], x8))

    x16 = pt_mult(x8, x8)
    n_chunks = n // 4
    subpolys = np.split(p, n_chunks)
    babies = [evaluate_baby_poly(sp) for sp in subpolys]
    gs1: list[np.ndarray] = []
    for i in range(0, len(babies), 2):
        if i + 1 < len(babies):
            gs1.append(add_vec(babies[i], pt_mult(babies[i + 1], x4)))
        else:
            gs1.append(babies[i])
    gs2: list[np.ndarray] = []
    for i in range(0, len(gs1), 2):
        if i + 1 < len(gs1):
            gs2.append(add_vec(gs1[i], pt_mult(gs1[i + 1], x8)))
        else:
            gs2.append(gs1[i])
    if len(gs2) == 1:
        return gs2[0]
    if len(gs2) != 2:
        raise ValueError(f"Stockmeyer gs2 长度异常: {len(gs2)}（n={n}）")
    return add_vec(gs2[0], pt_mult(gs2[1], x16))


def thor_exp_slots(
    x_packs: list[np.ndarray],
    mask_packs: list[np.ndarray],
    params: SoftmaxHeParams,
) -> list[np.ndarray]:
    """Stockmeyer exp 多项式 + δ1 平方；再 × mask。"""
    d1 = float(params.delta1)
    d2 = float(params.delta2)
    n1 = _check_power_of_two("delta1", d1)
    scale = math.exp(params.shift / d2 / d1)
    inv_scale = 1.0 / scale
    inv_in = 1.0 / (d1 * d2 * THOR_INPUT_SCALE)

    out: list[np.ndarray] = []
    for x, m in zip(x_packs, mask_packs):
        x = np.asarray(x, dtype=np.float64)
        m = np.asarray(m, dtype=np.float64)
        x_work = pt_mult(m, x)
        x_scaled = pt_mult(np.full_like(x_work, inv_in), x_work)
        # 与 THOR he_exp1 一致：字面量是降幂，Stockmeyer 要升幂 → reverse
        # （本仓库 Horner 不 reverse，与 reverse+Stockmeyer 为同一多项式）
        coeffs_asc = tuple(reversed(params.exp_coeffs))
        exp_approx = stockmeyer_poly_slots(coeffs_asc, x_scaled)
        exp_approx = pt_mult(np.full_like(exp_approx, inv_scale), exp_approx)
        for _ in range(n1):
            exp_approx = pt_mult(exp_approx, exp_approx)
        out.append(pt_mult(m, exp_approx))
    return out


def encode_key_valid_mask_packs(
    key_valid: np.ndarray,
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """
    ``key_valid`` shape ``(seq,)``：1=有效 key，0=padding。
    写入与 score 相同的对角 pack 布局；pad head 槽位恒为 0。
    """
    key_valid = np.asarray(key_valid, dtype=np.float64).reshape(-1)
    if key_valid.shape[0] != cfg.seq_len:
        raise ValueError(
            f"key_valid 长度须为 seq_len={cfg.seq_len}，得到 {key_valid.shape[0]}"
        )
    packs = [
        np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)
    ]
    for pack_i in range(cfg.n_input_packs):
        for j in range(cfg.pack):
            diag = pack_i * cfg.pack + j
            if diag >= cfg.seq_len:
                continue
            for t in range(cfg.seq_len):
                key = (diag + t) % cfg.seq_len
                kv = float(key_valid[key])
                for h in range(cfg.num_heads):
                    packs[pack_i][_slot_index(cfg, j, t, h)] = kv
    return packs


def slot_softmax_he(
    score_packs: list[np.ndarray],
    mask_packs: list[np.ndarray],
    cfg: ThorConfig,
    params: SoftmaxHeParams,
) -> list[np.ndarray]:
    """
    HE 同构 Softmax：layout 不变，返回与 ``score_packs`` 同形状的概率 packs。

    步骤：Stockmeyer exp → Σ_key → aSOR(1/σ) → y；再按 δ2 做 Σy² 归一化轮。
    """
    if len(score_packs) != len(mask_packs):
        raise ValueError("score_packs 与 mask_packs 长度须一致")
    d2 = float(params.delta2)
    n2 = _check_power_of_two("delta2", d2)
    if n2 > MAX_SUM_SQ_ROUNDS:
        raise ValueError(
            f"delta2={d2} 对应 {n2} 轮 Σy²，超过 MAX_SUM_SQ_ROUNDS={MAX_SUM_SQ_ROUNDS}"
        )

    exp_packs = thor_exp_slots(score_packs, mask_packs, params)
    sigma = sum_over_keys(exp_packs, cfg)
    inv_sigma, en = asor_inverse_slots(
        sigma, params.e0_sigma, params.asor_max_iters_sigma
    )
    y_packs = [pt_mult(e, inv_sigma) for e in exp_packs]
    y_packs = [pt_mult(m, y) for y, m in zip(y_packs, mask_packs)]

    for r in range(n2):
        e0_2 = thor_eps2_from_en(en, seq_len=cfg.seq_len)
        y2_packs = [pt_mult(y, y) for y in y_packs]
        y2_masked = [pt_mult(m, y2) for y2, m in zip(y2_packs, mask_packs)]
        sum_sq = sum_over_keys(y2_masked, cfg)
        iters_r = int(params.asor_max_iters_sum_sq[r])
        if iters_r < 1:
            raise ValueError(
                f"Σy² 第 {r} 轮 asor_max_iters={iters_r} 无效（delta2={d2}）"
            )
        inv_sq, en = asor_inverse_slots(sum_sq, e0_2, iters_r)
        y_packs = [pt_mult(y2, inv_sq) for y2 in y2_packs]
        y_packs = [pt_mult(m, y) for y, m in zip(y_packs, mask_packs)]

    return y_packs
