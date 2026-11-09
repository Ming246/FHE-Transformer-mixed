"""THOR bootstrap sites, plus ``gelu_f2``. No DP placer.

Opt-in via ``smoke_thor_chain.py --bootstrap thor_fixed``.

THOR 必做的位置照做。THOR 里看 level 才做的，这里每次都做。
另外在 f1 与 f2 之间加 ``gelu_f2``。

- ``softmax_exp_stockmeyer``：QK 分数之后、Softmax 之前
- ``softmax_sumsq_r*_sum``：Σy² 求和之后、rotsum 之前
- ``softmax_asor_sigma_i*`` / ``softmax_norm_r*_asor*``：每轮 aSOR
- ``softmax_exit_exp``：出口 4 条 exp，乘 inv 之前
- ``bridge_rot_attn_dense``：attention context 之后、dense 之前
- ``ln*_invsqrt_*``：inv√ 每一轮
- ``ln*_post_invsqrt``：inv√ 结束、乘 γ 之前
- ``gelu_f1``：dense1 之后、f1 之前
- ``gelu_f2``：f1 之后、f2 之前
- ``ln2_scale``：FF 残差相加之后、LN2 之前

``real_pack`` 由调用方拆包，钩子内不再 postscale。
``direct`` 在 ``BootstrapHook.refresh`` 里 postscale。
``softmax_sumsq_r*_sum`` 与 ``softmax_exit_exp`` 只做 bootstrap，不乘 100 / 0.01。
"""
from __future__ import annotations

from typing import Iterable

# mrpc high L0–L2 上限（σ 6、Σy² 7、ln2 inv√ 7）。没被调用的名字不会刷。
_N_ITER = 12


def thor_fixed_events(layers: Iterable[int]) -> dict[str, str]:
    """Event name → pack mode (``direct`` | ``real_pack``)."""
    out: dict[str, str] = {}

    def add(name: str, mode: str) -> None:
        out[name] = mode

    for li in layers:
        p = f"L{int(li)}"
        add(f"{p}.softmax_exp_stockmeyer", "real_pack")
        add(f"{p}.softmax_exit_exp", "direct")
        add(f"{p}.bridge_rot_attn_dense", "direct")
        add(f"{p}.gelu_f1", "real_pack")
        add(f"{p}.gelu_f2", "real_pack")
        add(f"{p}.ln2_scale", "real_pack")
        add(f"{p}.ln1_post_invsqrt", "direct")
        add(f"{p}.ln2_post_invsqrt", "direct")
        for r in (0, 1):
            add(f"{p}.softmax_sumsq_r{r}_sum", "direct")
        for i in range(_N_ITER):
            add(f"{p}.softmax_asor_sigma_i{i}", "real_pack")
            add(f"{p}.ln1_invsqrt_{i}", "real_pack")
            add(f"{p}.ln2_invsqrt_{i}", "real_pack")
            for r in (0, 1):
                add(f"{p}.softmax_norm_r{r}_asor{i}", "real_pack")
    return out


def thor_fixed_bare_events(events: dict[str, str]) -> set[str]:
    """Σy² 求和与出口 exp：不再 ×100 / ×0.01。"""
    bare: set[str] = set()
    for name in events:
        if name.endswith(".softmax_exit_exp"):
            bare.add(name)
        elif ".softmax_sumsq_r" in name and name.endswith("_sum"):
            bare.add(name)
    return bare


def install_thor_fixed_bootstrap(hook, layers: Iterable[int]) -> dict[str, str]:
    """Replace the hook plan with the fixed site list. Discards stay off."""
    events = thor_fixed_events(layers)
    hook.mode = "plan"
    hook.plan_events = set(events)
    hook.plan_pack_modes = dict(events)
    hook.plan_events_by_name = {}
    hook.apply_discards = False
    hook.level_discards = []
    hook._keep_after = {}
    hook.bare_scale_events = thor_fixed_bare_events(events)
    return events
