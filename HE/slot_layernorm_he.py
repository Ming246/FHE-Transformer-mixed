"""
HE 同构 LayerNorm（纯实数 slot 向量）：M 缩放 + he_invsqrt。

对齐 ``layernorm_poly.he_layernorm_from_config`` 数值；布局操作为
THOR 风格：pack 求和 + ``rotsum(slot_stride)`` + head 轴 1/2/4 折叠/广播
（仅用 rotate / add / pt×ct / ct×ct）。

``input_lower`` 中 head 0..5 与 6..11 为同一列块的重复写入；求和前 mask 掉
h≥6，避免双计。``depth`` 与 cost：``invsqrt_max_iters + 3``。

参数通用（min_var/max_var/iters/w_buffer）；档位只是参数组合。测试默认 high。
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

from layernorm_poly import (  # noqa: E402
    W_BUFFER,
    _he_invsqrt_kn,
    layernorm_config_for_layer,
)


def _scale(v: np.ndarray, s: float) -> np.ndarray:
    return pt_mult(np.full_like(v, float(s), dtype=np.float64), v)


def _add_scalar(v: np.ndarray, s: float) -> np.ndarray:
    return add_vec(v, np.full_like(v, float(s), dtype=np.float64))


@dataclass(frozen=True)
class LayerNormHeParams:
    """与档位无关的 LayerNorm HE 参数包。"""

    min_var: float
    max_var: float
    invsqrt_max_iters: int
    w_buffer: float
    kind: str
    layer_idx: int
    level: int

    @staticmethod
    def from_layernorm_poly(
        task_name: str,
        layer_idx: int,
        kind: str,
        *,
        level: int = 2,
    ) -> "LayerNormHeParams":
        cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
        return LayerNormHeParams(
            min_var=float(cfg["min_var"]),
            max_var=float(cfg["max_var"]),
            invsqrt_max_iters=int(cfg["invsqrt_max_iters"]),
            w_buffer=float(cfg.get("w_buffer", W_BUFFER)),
            kind=str(cfg["kind"]),
            layer_idx=int(cfg["layer_idx"]),
            level=int(cfg["level"]),
        )


def rotsum(vec: np.ndarray, interval: int) -> np.ndarray:
    s = len(vec)
    if s % interval != 0:
        raise ValueError(f"len={s} 须能被 interval={interval} 整除")
    rep = int(math.log2(s // interval))
    temp = np.asarray(vec, dtype=np.float64).copy()
    for i in range(rep):
        temp = add_vec(temp, rotate_left(temp, interval * (2**i)))
    return temp


def _sum_packs(packs: list[np.ndarray]) -> np.ndarray:
    acc = np.asarray(packs[0], dtype=np.float64).copy()
    for p in packs[1:]:
        acc = add_vec(acc, np.asarray(p, dtype=np.float64))
    return acc


def _head_unique_mask(cfg: ThorConfig) -> np.ndarray:
    """仅保留 h∈[0, n_hidden_col_blocks)（去重列块；pad/重复头为 0）。"""
    nb = cfg.n_hidden_col_blocks
    mask = np.zeros(cfg.num_slots, dtype=np.float64)
    for j in range(cfg.pack):
        for t in range(cfg.seq_len):
            for h in range(nb):
                mask[_slot_index(cfg, j, t, h)] = 1.0
    return mask


def _head_lane0_mask(cfg: ThorConfig) -> np.ndarray:
    """每个 (j,t) 组内仅保留 h=0（折叠后的汇总槽）。"""
    mask = np.zeros(cfg.num_slots, dtype=np.float64)
    for j in range(cfg.pack):
        for t in range(cfg.seq_len):
            mask[_slot_index(cfg, j, t, 0)] = 1.0
    return mask


def _rotsum_heads_partial(vec: np.ndarray, cfg: ThorConfig) -> np.ndarray:
    """
    在 head_slot_pack 内用 1/2/4 旋转求和（THOR LN）。
    调用前须 mask 掉 h≥n_hidden_col_blocks，使 pad 为 0，从而和 = Σ_{h<nb}。
    """
    temp = np.asarray(vec, dtype=np.float64).copy()
    # 覆盖 nb=6：1+2+4 折叠 8 槽；多余槽为 0
    for step in (1, 2, 4):
        temp = add_vec(temp, rotate_left(temp, step))
    return temp


def _broadcast_heads_partial(vec: np.ndarray, cfg: ThorConfig) -> np.ndarray:
    """把 h=0 的值广播到 h=0..7（再由业务 mask 截到 nb 或复制到重复头）。"""
    temp = np.asarray(vec, dtype=np.float64).copy()
    for step in (1, 2, 4):
        temp = add_vec(temp, rotate_left(temp, -step))
    return temp


def sum_over_hidden(
    packs: list[np.ndarray],
    cfg: ThorConfig,
    *,
    unique_mask: np.ndarray | None = None,
    lane0_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Σ_{hidden}：pack 加 → rotsum(stride) → mask 唯一头 → head 1/2/4 和 → 留 h=0。

    返回与 packs[0] 同形；有效汇总在各 (j,t) 的 h=0 槽（rotsum 后各 j 已相同）。
    """
    if len(packs) != cfg.n_input_packs:
        raise ValueError(f"期望 {cfg.n_input_packs} packs，得到 {len(packs)}")
    if unique_mask is None:
        unique_mask = _head_unique_mask(cfg)
    if lane0_mask is None:
        lane0_mask = _head_lane0_mask(cfg)

    s = rotsum(_sum_packs(packs), cfg.slot_stride)
    s = pt_mult(unique_mask, s)
    s = _rotsum_heads_partial(s, cfg)
    return pt_mult(lane0_mask, s)


def broadcast_hidden_stat(
    lane0_vec: np.ndarray,
    cfg: ThorConfig,
    *,
    unique_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    将 h=0 统计量广播到所有唯一头槽，并复制到重复头 h=nb..2*nb-1（与 encode 一致）。
    """
    nb = cfg.n_hidden_col_blocks
    if unique_mask is None:
        unique_mask = _head_unique_mask(cfg)
    spread = _broadcast_heads_partial(lane0_vec, cfg)
    spread = pt_mult(unique_mask, spread)
    # 复制到 h+nb（重复写入的 head）
    dup = np.zeros_like(spread)
    for j in range(cfg.pack):
        for t in range(cfg.seq_len):
            for h in range(nb):
                src = _slot_index(cfg, j, t, h)
                dst = _slot_index(cfg, j, t, h + nb)
                dup[dst] = spread[src]
    return add_vec(spread, dup)


def he_invsqrt_slots(
    variance: np.ndarray,
    e_init: float,
    max_iters: int,
) -> np.ndarray:
    """
    ``layernorm_poly.he_invsqrt_batched`` 的 slot 同构（无 float clamp；HE 路径）。

    an←variance，bn←1；每步 kn 明文标量。
    """
    an = np.asarray(variance, dtype=np.float64).copy()
    bn = np.ones_like(an)
    en = float(e_init)
    if en <= 0.0:
        raise ValueError(f"e_init 须 > 0，得到 {e_init}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters 须 ≥ 1，得到 {max_iters}")
    for _ in range(max_iters):
        kn = _he_invsqrt_kn(en)
        inv_kn = 3.0 / kn
        # diff = inv_kn - an
        diff = _add_scalar(_scale(an, -1.0), inv_kn)
        # bn = bn * (kn**(3/2)/2) * diff
        bn = pt_mult(
            _scale(bn, (kn ** 1.5) / 2.0),
            diff,
        )
        # an = an * (kn**3/4) * diff^2
        diff2 = pt_mult(diff, diff)
        an = pt_mult(_scale(an, (kn**3) / 4.0), diff2)
        en = kn * en * (3.0 - kn * en) ** 2 / 4.0
    return bn


def encode_affine_to_input_lower(
    gamma: np.ndarray,
    beta: np.ndarray,
    cfg: ThorConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    将 gamma/beta ``(hidden,)`` 编成与 ``encode_input_lower_diagonals`` 相同布局的明文 packs。
    """
    gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    if gamma.shape[0] != cfg.hidden_dim or beta.shape[0] != cfg.hidden_dim:
        raise ValueError(
            f"gamma/beta 长度须为 hidden_dim={cfg.hidden_dim}"
        )
    # 复用 encode：构造伪输入，每行相同 —— 更直接按索引填
    g_packs = [
        np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)
    ]
    b_packs = [
        np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)
    ]
    nb = cfg.n_hidden_col_blocks
    for pack_i in range(cfg.n_input_packs):
        for j in range(cfg.pack):
            diag = pack_i * cfg.pack + j
            if diag >= cfg.seq_len:
                continue
            for t in range(cfg.seq_len):
                for h in range(cfg.num_heads):
                    block = h % nb
                    row = (diag + t) % cfg.seq_len
                    feat = block * cfg.seq_len + row
                    idx = _slot_index(cfg, j, t, h)
                    g_packs[pack_i][idx] = gamma[feat]
                    b_packs[pack_i][idx] = beta[feat]
    return g_packs, b_packs


def slot_layernorm_he(
    x_packs: list[np.ndarray],
    gamma_packs: list[np.ndarray],
    beta_packs: list[np.ndarray],
    cfg: ThorConfig,
    params: LayerNormHeParams,
    *,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """
    HE 同构 LayerNorm，layout 不变（``input_lower`` packs）。

    步骤对齐 ``he_layernorm_from_config``：
      scale → Σx / Σx² → var → he_invsqrt → (n x - Σx)·inv·γ + β
    """
    if not (len(x_packs) == len(gamma_packs) == len(beta_packs) == cfg.n_input_packs):
        raise ValueError("x/gamma/beta packs 数量须等于 n_input_packs")

    n = float(cfg.hidden_dim)
    max_for_denominator = (params.max_var * params.w_buffer + eps) * (n**2)
    scale = 1.0 / math.sqrt(max_for_denominator)
    e_init = params.min_var / params.max_var

    unique_mask = _head_unique_mask(cfg)
    lane0_mask = _head_lane0_mask(cfg)

    x_scaled = [_scale(np.asarray(p, dtype=np.float64), scale) for p in x_packs]

    sum_x_lane0 = sum_over_hidden(
        x_scaled, cfg, unique_mask=unique_mask, lane0_mask=lane0_mask
    )
    sum_x_bcast = broadcast_hidden_stat(sum_x_lane0, cfg, unique_mask=unique_mask)

    # numerator = n * x_scaled - sum_x
    numerator = [
        add_vec(_scale(xs, n), _scale(sum_x_bcast, -1.0)) for xs in x_scaled
    ]

    x2 = [pt_mult(xs, xs) for xs in x_scaled]
    sigma_x2_lane0 = sum_over_hidden(
        x2, cfg, unique_mask=unique_mask, lane0_mask=lane0_mask
    )
    # variance 只在 lane0：n*Σx² - (Σx)² + eps/M
    sq_sum = pt_mult(sum_x_lane0, sum_x_lane0)
    variance = add_vec(
        _scale(sigma_x2_lane0, n),
        _scale(sq_sum, -1.0),
    )
    variance = _add_scalar(variance, eps / max_for_denominator)
    # 仅 lane0 有效；其余槽保持 0 以免污染 invsqrt
    variance = pt_mult(lane0_mask, variance)

    inv_lane0 = he_invsqrt_slots(
        variance, e_init=e_init, max_iters=params.invsqrt_max_iters
    )
    inv_lane0 = pt_mult(lane0_mask, inv_lane0)
    inv_bcast = broadcast_hidden_stat(inv_lane0, cfg, unique_mask=unique_mask)

    out: list[np.ndarray] = []
    for num, g, b in zip(numerator, gamma_packs, beta_packs):
        # num * inv * gamma + beta
        y = pt_mult(pt_mult(num, inv_bcast), np.asarray(g, dtype=np.float64))
        y = add_vec(y, np.asarray(b, dtype=np.float64))
        out.append(y)
    return out
