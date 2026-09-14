#!/usr/bin/env python3
"""
Liberate DualRail LayerNorm — repo ``layernorm_poly`` ranges + fixed invsqrt iters.

I/O matches THOR ``he_layernorm*``: ciphertext array shape ``(8,)``.
Algorithm follows THOR ``he_layernorm`` / ``he_invsqrt`` (rotsum + head fold),
but uses production ``min_var``/``max_var``/``invsqrt_max_iters`` (no α).
"""
from __future__ import annotations

import os
import sys
from functools import partial

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
for p in (_HERE, _HE, _REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from path_setup import ensure_thor_path

ensure_thor_path()
if _HE not in sys.path:
    sys.path.append(_HE)
if _REPO not in sys.path:
    sys.path.append(_REPO)

from layernorm_poly import _he_invsqrt_kn  # noqa: E402
from slot_layernorm_he import LayerNormHeParams  # noqa: E402
from softmax_he_ct import align_bypass_to_main  # noqa: E402


def _lane0_mask(num_slots: int) -> np.ndarray:
    return np.array(([1] * 1 + [0] * 15) * (num_slots // 16), dtype=np.float64)


def _scale_mask(num_slots: int, scale: float) -> np.ndarray:
    # Active heads: first 6 of every 16 slots (THOR DualRail pack).
    return np.array(([float(scale)] * 6 + [0.0] * 10) * (num_slots // 16), dtype=np.float64)


def he_invsqrt_ct(
    engine,
    numerator,
    denominator,
    e_init: float,
    max_iters: int,
    mask: np.ndarray,
    *,
    bts_hook=None,
    bts_prefix: str | None = None,
):
    """Fixed-iter inv√；每轮微事件 ``{bts_prefix}_invsqrt_{i}``（2→2, d=2）。

    段首按 plan DualRail 打包刷 ``(an, bn)``（代价 1、落地 rem=budget−1）。
    入口与每轮刷后要求二者同 ``level_calc``。勿对 an 做 ``auto_level`` 对齐 bn1。
    """
    an = denominator
    bn = numerator

    def _require_same_level(*, where: str) -> None:
        if int(an.level_calc) != int(bn.level_calc):
            raise ValueError(
                "he_invsqrt_ct requires an/bn (denominator/numerator) at the "
                f"same level_calc ({where}), got an={int(an.level_calc)} "
                f"bn={int(bn.level_calc)}"
                + (f" (prefix={bts_prefix})" if bts_prefix else "")
            )

    _require_same_level(where="entry")
    en = float(e_init)
    if en <= 0.0:
        raise ValueError(f"e_init must be > 0, got {e_init}")
    max_iters = int(max_iters)
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")
    for i in range(max_iters):
        event = f"{bts_prefix}_invsqrt_{i}" if bts_prefix else None
        if bts_hook is not None and event and bts_hook.should_refresh(event):
            from dualrail_bts import plan_refresh_an_bn

            an, bn = plan_refresh_an_bn(engine, an, bn, bts_hook, event)
            _require_same_level(where=f"after {event}")

        kn = _he_invsqrt_kn(en)
        # bn: cm_mult(+1) → level_up(bn2) → ct×ct(+1) ⇒ Δ=2
        bn1 = engine.cm_mult(bn, (kn ** (3.0 / 2.0) / 2.0) * mask)
        bn2 = engine.mc_sub((3.0 / kn) * mask, an)
        bn2 = engine.level_up(bn2, int(bn1.level_calc))
        bn = engine.auto_ct_ct_mult(bn1, bn2)

        # an: cm_mult(+1) ∥ square(mc_sub)(+1) → ct×ct(+1) ⇒ Δ=2
        an1 = engine.cm_mult(an, (kn**3 / 4.0) * mask)
        an2 = engine.square(engine.mc_sub((3.0 / kn) * mask, an))
        an = engine.auto_ct_ct_mult(an1, an2)

        en = kn * en * (3.0 - kn * en) ** 2 / 4.0
    return bn


def he_layernorm_poly(
    engine,
    x: np.ndarray,
    gamma,
    beta,
    params: LayerNormHeParams,
    *,
    var_e: float = 1e-5,
    n: int = 768,
    debug: bool = False,
    sk=None,
    bts_hook=None,
    bts_prefix: str | None = None,
):
    """
    HE LayerNorm（DualRail 8 real packs）。

    微事件边界（plan 名 ``{bts_prefix}_*``）：
      - ``_scale``：8 real DualRail 刷（代价4/落地13）→ cm_mult 8→8, d=1
      - ``_var``：8×enc_l DualRail 刷（代价4/落地13）→ variance 1 + num@AUX, d=2
      - ``_enc_one``：直接刷 variance×1（代价1/落地 budget）→ +enc_one, d=0
      - ``_invsqrt_{i}``：DualRail 打包刷 (an,bn)（代价1/落地13）→ 一轮 d=2
      - ``_post_invsqrt``：直接刷 denom×1 → 旁路 num 对齐 → γ/β 出 8, d=2
        （broadcast 不进图）

    ``params.kind`` = 每层 **槽位**（attn 后 ln1 / FFN 后 ln2）。深度差仅来自
    ``invsqrt_max_iters`` 等配置。输入缩放统一用 ``1/sqrt(max_for_denominator)``，
    **不**再套用 THOR 原版按方差档位的 mask/2 分支。
    """
    del debug, sk
    if x.shape != (8,):
        raise ValueError(f"he_layernorm_poly expects shape (8,), got {x.shape}")

    kind = params.kind  # encoder slot: "ln1" | "ln2"
    min_var = float(params.min_var)
    max_var = float(params.max_var)
    w_buffer = float(params.w_buffer)
    epsilon_var1 = min_var / max_var
    max_for_denominator = (max_var * w_buffer + var_e) * (n**2)

    ln_prefix = bts_prefix or ""
    if ln_prefix and not ln_prefix.endswith(params.kind):
        ln_prefix = f"{ln_prefix}.{params.kind}"
    elif not ln_prefix:
        ln_prefix = None

    def _as_obj8(seq):
        out = np.empty(8, dtype=object)
        for i, ct in enumerate(seq):
            out[i] = ct
        return out

    # ``{ln_prefix}_scale`` / ``_var``：8 real 普通打包 → DualRail 刷
    if ln_prefix and bts_hook is not None:
        from dualrail_bts import plan_refresh_dualrail8

        scale_ev = f"{ln_prefix}_scale"
        if bts_hook.should_refresh(scale_ev):
            x = _as_obj8(
                plan_refresh_dualrail8(
                    engine,
                    list(x),
                    bts_hook,
                    scale_ev,
                    scale_after=0.5,
                )
            )

    nslots = int(engine.num_slots)
    scale = 1.0 / (max_for_denominator ** 0.5)
    mask_scale = _scale_mask(nslots, scale)

    enc_l = [engine.cm_mult(ct, mask_scale) for ct in x]

    # ``{ln_prefix}_var``：段首 DualRail 刷 8×enc_l → variance
    if ln_prefix and bts_hook is not None:
        from dualrail_bts import plan_refresh_dualrail8

        var_ev = f"{ln_prefix}_var"
        if bts_hook.should_refresh(var_ev):
            enc_l = list(
                plan_refresh_dualrail8(
                    engine,
                    enc_l,
                    bts_hook,
                    var_ev,
                    scale_after=0.5,
                )
            )

    # Σx → lane-0, then broadcast for numerator.
    sum_x = enc_l[0]
    for i in range(1, len(enc_l)):
        sum_x = engine.add(sum_x, enc_l[i])

    sum_x = engine.rotsum(sum_x, 2**11)
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, 1))
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, 2))
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, 4))
    lane0 = _lane0_mask(nslots)
    sum_x = engine.cm_mult(sum_x, lane0)
    sq_sum_x = engine.square(sum_x)
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, -1))
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, -2))
    sum_x = engine.add(sum_x, engine.rotate_left(sum_x, -4))

    nx = [engine.mult_int_scalar(ct, n) for ct in enc_l]
    # sum_x 刚走过 lane0 的 cm_mult（+1 rem）；nx 仅 int 乘不抬 level。
    if sum_x.level_calc > nx[0].level_calc:
        nx = [engine.level_up(ct, sum_x.level_calc) for ct in nx]
    elif nx[0].level_calc > sum_x.level_calc:
        sum_x = engine.level_up(sum_x, nx[0].level_calc)
    numerator = [engine.sub(ct, sum_x) for ct in nx]

    sigma_x2 = engine.square(enc_l[0])
    for i in range(1, len(enc_l)):
        sigma_x2 = engine.add(sigma_x2, engine.square(enc_l[i]))
    sigma_x2 = engine.rotsum(sigma_x2, 2**11)
    sigma_x2 = engine.add(sigma_x2, engine.rotate_left(sigma_x2, 1))
    sigma_x2 = engine.add(sigma_x2, engine.rotate_left(sigma_x2, 2))
    sigma_x2 = engine.add(sigma_x2, engine.rotate_left(sigma_x2, 4))
    sigma_x2 = engine.cm_mult(sigma_x2, lane0)

    n_sigma_x2 = engine.mult_int_scalar(sigma_x2, n)
    variance = engine.sub(n_sigma_x2, sq_sum_x)
    variance = engine.add_scalar(variance, var_e / max_for_denominator)

    # ``{ln_prefix}_enc_one``：段首直接刷 variance（单 CT）→ +enc_one
    if ln_prefix and bts_hook is not None:
        enc_ev = f"{ln_prefix}_enc_one"
        if bts_hook.should_refresh(enc_ev):
            variance = bts_hook.refresh(enc_ev, variance)

    enc_one = engine.encode_and_encrypt(lane0, level=int(variance.level_calc))
    denominator = he_invsqrt_ct(
        engine,
        enc_one,
        variance,
        epsilon_var1,
        int(params.invsqrt_max_iters),
        lane0,
        bts_hook=bts_hook,
        bts_prefix=ln_prefix,
    )

    # broadcast lane0 → heads：不改 CT 数 / rem，不进微事件图
    denominator = engine.add(denominator, engine.rotate_left(denominator, -1))
    denominator = engine.add(denominator, engine.rotate_left(denominator, -2))
    denominator = engine.add(denominator, engine.rotate_left(denominator, -4))

    # ``{ln_prefix}_post_invsqrt``：段首刷主路 denom；旁路 num 向 denom 对齐后 γ/β
    post_ev = f"{ln_prefix}_post_invsqrt" if ln_prefix else None
    if (
        bts_hook is not None
        and post_ev
        and bts_hook.should_refresh(post_ev)
    ):
        denominator = bts_hook.refresh(post_ev, denominator)

    numerator = [
        align_bypass_to_main(engine, denominator, ct) for ct in numerator
    ]

    out = np.full((8,), None, dtype=object)
    for i in range(4):
        out[i] = engine.auto_ct_ct_mult(
            numerator[i],
            engine.rescale(engine.pt_ct_mult(gamma[i], denominator)),
        )
        out[i + 4] = engine.auto_ct_ct_mult(
            numerator[i + 4],
            engine.rescale(engine.pt_ct_mult(gamma[i + 4], denominator)),
        )
        out[i] = engine.pc_add(beta[i], out[i])
        out[i + 4] = engine.pc_add(beta[i + 4], out[i + 4])
        # THOR：γ/β 明文 bake 为 1/2，输出恒 ×2 还原（与方差档位无关）。
        out[i] = engine.cc_add(out[i], out[i])
        out[i + 4] = engine.cc_add(out[i + 4], out[i + 4])
    return out


def bind_attention_layernorm(
    thor_attention,
    *,
    task: str,
    layer_idx: int,
    level: int = 2,
    bts_hook=None,
) -> LayerNormHeParams:
    """Replace ``thor_attention.layernorm`` with poly HE LN (ln1)."""
    params = LayerNormHeParams.from_layernorm_poly(
        task, layer_idx, "ln1", level=level
    )
    thor_attention.layernorm = partial(
        he_layernorm_poly,
        engine=thor_attention.engine,
        gamma=thor_attention.weights["LayerNorm.weight"],
        beta=thor_attention.weights["LayerNorm.bias"],
        params=params,
        bts_hook=bts_hook,
        bts_prefix=f"L{int(layer_idx)}.ln1",
    )
    thor_attention._ln_poly_params = params
    return params


def bind_ff_layernorm(
    thor_ff,
    *,
    task: str,
    layer_idx: int,
    level: int = 2,
    bts_hook=None,
) -> LayerNormHeParams:
    """Replace ``thor_ff.layernorm`` with poly HE LN (ln2)."""
    params = LayerNormHeParams.from_layernorm_poly(
        task, layer_idx, "ln2", level=level
    )
    thor_ff.layernorm = partial(
        he_layernorm_poly,
        engine=thor_ff.engine,
        gamma=thor_ff.weights["LayerNorm.weight"],
        beta=thor_ff.weights["LayerNorm.bias"],
        params=params,
        bts_hook=bts_hook,
        bts_prefix=f"L{int(layer_idx)}.ln2",
    )
    thor_ff._ln_poly_params = params
    return params


def bind_layer_layernorm(
    thor_attention,
    thor_ff,
    *,
    task: str,
    layer_idx: int,
    level: int = 2,
    bts_hook=None,
) -> tuple[LayerNormHeParams, LayerNormHeParams]:
    """Bind ln1 + ln2 for one encoder layer."""
    p1 = bind_attention_layernorm(
        thor_attention,
        task=task,
        layer_idx=layer_idx,
        level=level,
        bts_hook=bts_hook,
    )
    p2 = bind_ff_layernorm(
        thor_ff,
        task=task,
        layer_idx=layer_idx,
        level=level,
        bts_hook=bts_hook,
    )
    return p1, p2
