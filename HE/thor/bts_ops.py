"""
Bootstrapping 最优放置 — Branch A（``HE/thor``）本土化副本。

源自 ``HE/bts_ops.py``（主线纯实、budget=15）。本副本：

- **密文几何**：对齐 ``smoke_thor_repro._forward_layer`` / ``geometry.ThorComplexGeometry``
  （LN/残差 8 real；QKV spine 4 cplx；V/context 2 cplx；score→α DualRail 8→128；
  context rot 32；FF1 live 16）。**不是**假想的 8→16→32→64→128 复制链。
- **深度预算**：``BOOTSTRAP_DEPTH_BUDGET = 14``
  （Liberate/THOR 原参：bootstrap 后剩余 ≈ compute scale 段）
- **缩短模数链**：不写死入口 level。每次 bts 后若到下一次 bts 只需深度
  ``D < budget``，则 ``level_up`` 丢掉 ``budget-D``（跨度不设硬顶；
  噪声由起点 lc 主导，见 ``audit_level_up_span.py``）。入口仍宜
  ``encodecrypt(..., level=level_calc)``，避免从很浅 lc 无谓 discard。

与 ``HE/thor`` Liberate 路径联动：``BootstrapHook(mode=\"plan\")`` 消费
``optimize_bootstrap`` 的 placements；``compute_level_discards`` 给出每次
bts 后应保留的剩余深度。

**Bts 刷新语义**（``pack_mode``，与 HE ``dualrail_bts`` 对齐）：

- ``direct``：已是复数/单 CT，或 **仅 1 条 real** → 直接 bootstrap，落地 rem=budget。
- ``real_pack``：多条 real → pack→bts→unpack×½，落地 rem=budget−``dualrail_depth``。

计价（关键路径乐观界）：``layout=dualrail`` 用主路径现场 CT 数；
``layout=real`` 用 ``max(1, n//2)``（打包后复数条数）。

DP 状态是 **spine rem only**（``(rem,)``）。旁路对齐在 HE 侧
``align_bypass_to_main``（含 Softmax exit 的 exp→inv）；不进 DP。

现行微事件划分：**全部 atomic**（不可再拆、段内不可中途 bts；
``rem >= depth + LIBERATE_MIN_LEAVE_REM``，不够则段首 before-boot）。

不修改仓库根 ``cost.py`` / ``HE/bts_ops.py``。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field, replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_HE = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HE, ".."))
# cost / poly at repo root; thor_encoder under HE/. Do NOT put HE ahead of
# thirdparty THOR src or ``import thor`` resolves to this adapter package.
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _p in (_HERE, _HE):
    if _p not in sys.path:
        sys.path.append(_p)

from cost import (
    BOOTSTRAP_DEPTH_BUDGET as _MAINLINE_BOOTSTRAP_DEPTH_BUDGET,
    LAYERNORM_DEPTH_OVERHEAD,
    NUM_LAYERS,
    SOFTMAX_ASOR_ITER_DEPTH as _COST_SOFTMAX_ASOR_ITER_DEPTH,
    SOFTMAX_DEPTH_BASE,
    SOFTMAX_EXP_SQUARE_DEPTH,
    SOFTMAX_EXP_STOCKMEYER_DEPTH,
    gelu_poly_depth,
    layernorm_poly_depth,
    log2_ceil,
    scheme_index,
    softmax_poly_depth,
    validate_scheme,
)

# Liberate/THOR (logN=16, scale_bits=41, pre_quantum)：bootstrap 后可用深度
BOOTSTRAP_DEPTH_BUDGET = 14
_ = _MAINLINE_BOOTSTRAP_DEPTH_BUDGET  # 主线默认 15；本副本不用

# L0 DP 入口 remaining（= ``optimize_bootstrap(initial_level=…)``）。
# THOR notebook 用 enc@20 → rem=8 做跨层统一；本仓库默认满预算 rem=14。
DEFAULT_L0_ENTRY_REMAINING = BOOTSTRAP_DEPTH_BUDGET


def plan_initial_rem_from_enc_level(enc_level: int, num_levels: int) -> int:
    """DP ``initial_level`` ← embedding ``encrypt(..., level=enc_level)``（含 pack 税 −1）。"""
    return max(1, int(num_levels) - int(enc_level) - 1)


def enc_level_from_plan_initial_rem(rem: int, num_levels: int) -> int:
    """Embedding encrypt ``level_calc`` ← DP ``initial_level``（与上式互逆）。"""
    return int(num_levels) - int(rem) - 1
from gelu_poly import gelu_config_for_layer
from layernorm_poly import (
    TASK_NAMES as LAYERNORM_TASK_NAMES,
    count_he_invsqrt_iters,
    layernorm_config_for_layer,
)
from nolinear.gelu_chebyshev import cheb_he_depth_ps_tree, gelu_he_depth_breakdown
from softmax_poly import (
    LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK,
    LAYER_EXP_DIV_BY_TASK,
    SOFTMAX_LEVEL_KEYS,
)

# ---------------------------------------------------------------------------
# Slot CT 几何（对齐 thor_encoder_linear_core.ThorConfig）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotCtGeometry:
    """
    Branch A DualRail CT 条数（对齐 ``geometry.ThorComplexGeometry`` +
    ``smoke_thor_repro._forward_layer``）。

    bert_base：cplx spine=4，LN/resid real=8，pc_rot=64，context rot=32，
    V/context=2，score=8，α=128，ffn=(2,8)=16。
    """

    ct_hidden_cplx: int  # DualRail QKV / pack 后 spine
    ct_hidden_real: int  # LN / residual / W_O 输出
    ct_qkv_out: int  # Q/K/V 打包输出（4）
    ct_v_cplx: int  # context 侧 V（2）
    ct_context_cplx: int  # att context 输出（2）
    ct_att_score: int  # score / Softmax 入口（实数 8）
    ct_pc_rot: int  # make_rotated_copies(4) → 64
    ct_pc_rot_ctx: int  # make_rotated_copies(context 2) → 32
    ct_q_copies: int  # make_copies(Q) → 64
    ct_alpha_copies: int  # DualRail Softmax 出口 → 128
    ct_ffn: int  # FF1 两路 live → 16
    stockmeyer_live_factor: int = 8  # Softmax exp Stockmeyer 活副本因子

    @property
    def ct_hidden(self) -> int:
        """LN/残差侧条数（real）。"""
        return self.ct_hidden_real

    @classmethod
    def from_thor_config(cls, cfg: object) -> SlotCtGeometry:
        """纯实 ThorConfig → 几何（兼容）；Branch A 默认请用 ``from_complex``。"""
        ff_out = int(cfg.seq_len) // int(cfg.ff_pack)
        hr = int(cfg.n_input_packs)
        q = int(cfg.n_out_packed_qkv)
        return cls(
            ct_hidden_cplx=max(1, hr // 2),
            ct_hidden_real=hr,
            ct_qkv_out=q,
            ct_v_cplx=max(1, q // 2),
            ct_context_cplx=max(1, q // 2),
            ct_att_score=hr,
            ct_pc_rot=int(cfg.n_in_slot) // 2,
            ct_pc_rot_ctx=int(cfg.n_in_slot) // 4,
            ct_q_copies=int(cfg.head_dim),
            ct_alpha_copies=int(cfg.seq_len),
            ct_ffn=2 * ff_out,
        )

    @classmethod
    def from_complex(cls, g: object | None = None) -> SlotCtGeometry:
        """从 ``ThorComplexGeometry`` 构造（默认 bert_base_complex）。"""
        if g is None:
            try:
                from .geometry import bert_base_complex
            except ImportError:
                from geometry import bert_base_complex  # type: ignore

            g = bert_base_complex()
        return cls(
            ct_hidden_cplx=int(g.ct_hidden_cplx),
            ct_hidden_real=int(g.ct_hidden_real),
            ct_qkv_out=int(g.ct_qkv_out),
            ct_v_cplx=int(g.ct_v_cplx),
            ct_context_cplx=int(g.ct_context_cplx),
            ct_att_score=int(g.ct_att_score),
            ct_pc_rot=int(g.ct_pc_rot_cplx),
            ct_pc_rot_ctx=int(g.ct_pc_rot_ctx),
            ct_q_copies=int(g.ct_q_copies),
            ct_alpha_copies=int(g.ct_alpha_copies),
            ct_ffn=int(g.ct_ffn),
        )


def default_slot_geometry() -> SlotCtGeometry:
    return SlotCtGeometry.from_complex()


# 模块级活动几何（build / audit 入口会 set）；默认 = bert_base
_GEOM: SlotCtGeometry = default_slot_geometry()


def set_slot_geometry(geom: SlotCtGeometry | None = None) -> SlotCtGeometry:
    """设置活动 CT 几何；``None`` → bert_base。返回当前几何。"""
    global _GEOM
    global CT_HIDDEN, CT_HIDDEN_CPLX, CT_HIDDEN_REAL, CT_QKV_OUT, CT_V, CT_CONTEXT
    global CT_ROTATED, CT_ROT_CTX, CT_ATT_SCORE, CT_Q_COPIES, CT_ALPHA
    global CT_FFN, CT_SOFTMAX_STOCKMEYER_PEAK
    _GEOM = default_slot_geometry() if geom is None else geom
    CT_HIDDEN = _GEOM.ct_hidden_real
    CT_HIDDEN_CPLX = _GEOM.ct_hidden_cplx
    CT_HIDDEN_REAL = _GEOM.ct_hidden_real
    CT_QKV_OUT = _GEOM.ct_qkv_out
    CT_V = _GEOM.ct_v_cplx
    CT_CONTEXT = _GEOM.ct_context_cplx
    CT_ROTATED = _GEOM.ct_pc_rot
    CT_ROT_CTX = _GEOM.ct_pc_rot_ctx
    CT_ATT_SCORE = _GEOM.ct_att_score
    CT_Q_COPIES = _GEOM.ct_q_copies
    CT_ALPHA = _GEOM.ct_alpha_copies
    CT_FFN = _GEOM.ct_ffn
    CT_SOFTMAX_STOCKMEYER_PEAK = (
        CT_ATT_SCORE * _GEOM.stockmeyer_live_factor
    )
    return _GEOM


def active_geometry() -> SlotCtGeometry:
    return _GEOM


# 兼容旧名：只读别名（随 set_slot_geometry 更新）
CT_HIDDEN = _GEOM.ct_hidden_real
CT_HIDDEN_CPLX = _GEOM.ct_hidden_cplx
CT_HIDDEN_REAL = _GEOM.ct_hidden_real
CT_QKV_OUT = _GEOM.ct_qkv_out
CT_V = _GEOM.ct_v_cplx
CT_CONTEXT = _GEOM.ct_context_cplx
CT_ROTATED = _GEOM.ct_pc_rot
CT_ROT_CTX = _GEOM.ct_pc_rot_ctx
CT_ATT_SCORE = _GEOM.ct_att_score
CT_Q_COPIES = _GEOM.ct_q_copies
CT_ALPHA = _GEOM.ct_alpha_copies
CT_FFN = _GEOM.ct_ffn
CT_SOFTMAX_STOCKMEYER_PEAK = CT_ATT_SCORE * _GEOM.stockmeyer_live_factor

# ---------------------------------------------------------------------------
# 线性乘法深度（L0 Liberate probe 标定；见 ``smoke_thor_repro --probe-levels``）
#
# 实测 L0 plan_mock probe（默认 enc@14 → entry rem=14；旧 THOR enc@20 → rem=8）：
#   QKV：make_rotated_copies(depth0) + pt_ct_matmul +2 → 合并为 atomic ``linear_qkv``
#   其后 Q∥K 原子岛 ``linear_att_score``（输入 Q+K，输出 sftmx_in；V 旁路）：
#     K: transpose +1 + complexify(rot_internal+rescale) +2 → 3（与 Q 并行，不叠 spine）
#     Q: rescale +1 + make_copies +2 → 3
#     att_score 融合 +1
#     ⇒ spine depth = DEPTH_MAKE_COPIES + DEPTH_ATT_SCORE（=4）
#   Softmax 出口 rem14→context 后 rem4 ⇒ Δ≈10（plan_mock v4）；旧 9 偏低 1
#   attn_dense +3；ln1：dense rem11→inv√ i=1 an rem2 ⇒ prep+inv0≈9
#   ln1 mid@i=1 后出口 rem≈5 ⇒ post≈3；GeLU entry@13→post rem=2（Δ=11）
# FF bridge：LN1→rot64 的 mc_mult +1；dense1/2 各 +2
# ---------------------------------------------------------------------------

DEPTH_QKV = 2
DEPTH_TRANSPOSE_K = 3  # K 支路深度（与 Q 并行；岛内不计叠）
DEPTH_MAKE_COPIES = 3  # Q 支路：q_rescale+1 + make_copies+2
DEPTH_ATT_SCORE = 1  # QKᵀ 融合
DEPTH_QK_SCORE = DEPTH_MAKE_COPIES + DEPTH_ATT_SCORE  # 原子岛 spine depth
DEPTH_ATT_CONTEXT = 2  # HE L0 rem_guard：after_softmax rem12→after_att_context rem10

DEPTH_ATT_DENSE = 3  # pt_ct_matmul +2 + mc_mult +1
DEPTH_FF_BRIDGE = 1  # LN1 DualRail pack 后 mc_mask（原 bridge_rot_ff1 depth=0）
DEPTH_FF_DENSE = 2  # dense1 / dense2 共用（pt_ct_matmul l→l+2）
DEPTH_FF_DENSE2_KEEP = DEPTH_FF_DENSE
# 与 ``linear_eval.MAX_LEVEL_UP_SPAN`` 一致；None = discard 跨度不截断。
DISCARD_MAX_SPAN: int | None = None


# HE DualRail pack→bts→unpack 后 ``mult_scalar(×½)`` 耗 1 rem（见 dualrail_bts）。
# 段首刷计费约定（与 HE hook 对齐）：
#   - **普通打包**（如 8 real DualRail / an·bn 两实数）：打包成复数再刷再 unpack；
#     ``layout=real`` → cost = n//2；``dualrail_depth=1`` → 落地 rem = budget−1。
#   - **已是复数/单 CT**：直接刷；``layout=dualrail``（或 ``cost_ct``）→ cost = n；
#     ``dualrail_depth=0`` → 落地 rem = budget。
DUALRAIL_UNPACK_DEPTH = 1

# Softmax 通路（非多项式；对照 softmax_he_ct / DualRail exit）
# poly 审计深度仍在 kind=softmax 事件；下列 pathway 只调 HE rem / 放置。
DEPTH_SOFTMAX_ATTN_MASK = 1  # exp 后 key-mask：pt×ct
# σ：8 exp 聚合 → sigma_exp(1)；再 + enc_one → an/bn(2)；exp_u→AUX
DEPTH_SOFTMAX_SIGMA_AGGREGATE = 0
DEPTH_SOFTMAX_SIGMA_ENC_ONE = 0
CT_SOFTMAX_ASOR_SPINE = 2  # enc_one (an) + sigma_exp/summation (bn)
# aSOR 末轮 an/bn DualRail 打包代价=1；丢掉 bn 不占关键路径。
# 图上已无 ``softmax_sigma_inv`` / ``softmax_norm_r*_inv``：
#   σ-aSOR 末轮直接接 Σy² apply；norm-aSOR 末轮直接接 ``*_inv_mask``。
# he_asor_ct 返回后 inv 落主路（1 CT）；Σy² 主路 inv/summation 条数
CT_SOFTMAX_INV_SPINE = 1
DEPTH_SOFTMAX_SUMSQ_ENC_ONE = 0  # apply 后 +enc_one → an/bn(2)
DEPTH_SOFTMAX_EXP_HE_EXTRA = 0  # 保留常量以免旧引用炸掉
# HE Σy² apply 主路：inv(1) → summation(1)；8 exp 旁路 AUX
#   rescale(pt×ct mask·inv) + ct×ct(exp,inv) + square → Δrem≈3
DEPTH_SOFTMAX_SUMSQ_APPLY = 3
# he_asor 返回后 ``cm_mult(inv, masking)``（update_inv_D_fixed）
DEPTH_SOFTMAX_INV_MASK = 1
# DualRail exit 主路：inv 已 mask 后，``auto_ct_ct_mult`` 烧 1 rem
# （L0 probe：post-cm rem13 → sftmx_out rem12；mock/真 bts 一致）。
DEPTH_SOFTMAX_EXIT = 1

# Goldschmidt / aSOR 单次迭代 HE 乘法深度
# Softmax aSOR 每轮 HE rem = cost.SOFTMAX_ASOR_ITER_DEPTH（实测 1）
SOFTMAX_ASOR_ITER_DEPTH = int(_COST_SOFTMAX_ASOR_ITER_DEPTH)
# LN invsqrt 一轮：an/bn 同烧 depth=2
#   bn: cm_mult → level_up(bn2→bn1) → ct×ct
#   an: cm_mult ∥ square(mc_sub) → ct×ct
# （勿 auto_level(an,bn1)：会白烧 an，把一轮拉成 3）
LN_INVSQRT_ITER_DEPTH = 2
GOLDSCHMIDT_ITER_DEPTH = LN_INVSQRT_ITER_DEPTH  # 旧名别名
# Liberate：任意乘法（cm_mult / square / ct×ct）在 rem=1 时 MaximumLevelError。
# 因此 depth>0 的事件不能落到 rem=0；atomic 要求 rem >= depth+MIN_LEAVE。
LIBERATE_MIN_LEAVE_REM = 1
# phase B / cost 占位仍用 LAYERNORM_DEPTH_OVERHEAD(=3)。
# phase C：LN prep 拆为 scale / var / enc_one（对照 ``he_layernorm_poly``）。
# DualRail 入口 unpack 税挂在 ``*_scale``；冷启动实测以后再校准。
DEPTH_LN_SCALE = 1  # cm_mult(x, mask_scale)：8→8
DEPTH_LN_VAR = 2  # enc_l→variance（numerator→AUX）；先填，后测
DEPTH_LN_ENC_ONE = 0  # encode enc_one：1→2（an/bn）
CT_LN_VAR_SPINE = 1  # variance 主路
CT_LN_INVSQRT_SPINE = 2  # enc_one + variance（an/bn）
CT_LN_DENOM_SPINE = 1  # inv√ 收口后只留 bn=denominator
# ``*_inv``（an/bn→denom, d=0）已从图上删除：an/bn DualRail 打包为 1 cplx，
# 丢掉 an 不改变关键路径代价；inv√ 末轮直接接 ``*_post_invsqrt``。
# 旧名：融合 prep 深度（导出图 / 冷启动注释仍可能引用）
DEPTH_LN_PREP = DEPTH_LN_SCALE  # 兼容：仅 scale 段
DEPTH_LN_PREP_COLD = DEPTH_LN_SCALE + DEPTH_LN_VAR  # 兼容：scale+var
# he_layernorm_poly 收尾主路（denom）：
#   rescale(pt×ct(γ,denom)) +1 → auto_ct_ct_mult(num, ·) +1 ⇒ Δ=2
# （broadcast rotate/add、pc_add(β)、cc_add×2 不烧 rem）
DEPTH_LN_POST_INVSQRT = 2

# 旁路槽名（仅 export / 历史命名；DP 不再跟踪 slot rem）
SLOT_RESID = 0  # deprecated alias — HE align_bypass_to_main
SLOT_V = 1  # deprecated alias — Softmax 期间 V
SLOT_AUX = 2  # deprecated alias — Softmax exp / LN numerator
N_SLOTS = 0  # DP rem-only；无 slot 维
SLOT_UNSET = -1
SlotVec = tuple[int, ...]  # length N_SLOTS (== 0 → empty)
DpState = tuple[int, ...]  # (rem,) + empty slots

# 默认审计任务：与 softmax/LN 配置表一致
DEFAULT_AUDIT_TASKS: tuple[str, ...] = tuple(
    t for t in LAYERNORM_TASK_NAMES if t in LAYER_EXP_DIV_BY_TASK
)


@dataclass(frozen=True)
class Event:
    """计算图上的一步；``ct_peak`` 为段内副本峰值（中途 plain bts 上界）。"""

    name: str
    depth: int
    ct_in: int
    ct_out: int
    layer_idx: int = -1
    kind: str = ""  # linear | softmax | ln1 | gelu | ln2 | bridge | pathway
    join: bool = False
    ct_peak: int | None = None
    align_depth: int = 0  # 旧固定对齐（已弃用）
    # Deprecated slot fields（N_SLOTS=0；DP/rem_trace 忽略；保留以免旧调用炸掉）
    save_slot: int | None = None
    entry_save_slot: int | None = None
    join_slot: int | None = None
    join_offset: int = 0
    join_ct: int = 0
    refresh_side: bool = False
    clear_slot: int | None = None
    # Deprecated：保留字段以免旧调用炸掉；DP 已忽略（一律可选）。
    force_bootstrap: bool = False
    # 兼容字段（DP 已不选 DualRail pack；保留以免旧调用炸掉）。
    plain_ct: int | None = None  # 现场刷条数；默认 ct_in(+join)
    dualrail_ct: int | None = None  # 已弃用
    dualrail_depth: int = 0  # 已弃用（不扣 unpack 税）
    boot_ct: int | None = None
    no_level_discard: bool = False
    # 现行微事件划分：**一律 atomic**（不可再拆、段内不可中途 bts；
    # rem 不够只能段首 before-boot）。字段保留兼容；消耗逻辑忽略 False。
    # Liberate 乘法不能烧到 rem=0（见 LIBERATE_MIN_LEAVE_REM）。
    atomic: bool = True
    layout: str = "real"  # "real" | "dualrail"
    cost_ct: int | None = None  # 覆盖计价；None → layout 推导
    # True：段首 before-boot 只刷 off-spine 伴侣（如 Σy² summation），spine rem 不变。
    companion_boot: bool = False

    @property
    def bts_ct(self) -> int:
        """中途 bootstrap 按峰值 CT 计费（若无 peak 则 ct_in）。"""
        if self.ct_peak is not None:
            return max(self.ct_in, self.ct_peak)
        return self.ct_in

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"depth 不能为负：{self.name}")
        if self.ct_in <= 0 or self.ct_out <= 0:
            raise ValueError(f"ct_in/ct_out 须为正：{self.name}")
        if self.ct_peak is not None and self.ct_peak <= 0:
            raise ValueError(f"ct_peak 须为正：{self.name}")
        if self.align_depth < 0:
            raise ValueError(f"align_depth 不能为负：{self.name}")
        # Slot fields deprecated when N_SLOTS=0 (DP rem-only).
        if N_SLOTS <= 0:
            for _fname, _val in (
                ("save_slot", self.save_slot),
                ("entry_save_slot", self.entry_save_slot),
                ("join_slot", self.join_slot),
                ("clear_slot", self.clear_slot),
            ):
                if _val is not None:
                    raise ValueError(f"{_fname} 非法（DP rem-only）：{_val}")
        else:
            if self.save_slot is not None and not (0 <= self.save_slot < N_SLOTS):
                raise ValueError(f"save_slot 非法：{self.save_slot}")
            if self.entry_save_slot is not None and not (
                0 <= self.entry_save_slot < N_SLOTS
            ):
                raise ValueError(f"entry_save_slot 非法：{self.entry_save_slot}")
            if self.join_slot is not None and not (0 <= self.join_slot < N_SLOTS):
                raise ValueError(f"join_slot 非法：{self.join_slot}")
            if self.clear_slot is not None and not (0 <= self.clear_slot < N_SLOTS):
                raise ValueError(f"clear_slot 非法：{self.clear_slot}")
        if self.join_offset < 0:
            raise ValueError(f"join_offset 不能为负：{self.name}")
        if self.join_ct < 0:
            raise ValueError(f"join_ct 不能为负：{self.name}")
        if self.boot_ct is not None and self.boot_ct <= 0:
            raise ValueError(f"boot_ct 须为正：{self.name}")
        if self.dualrail_ct is not None and self.dualrail_ct <= 0:
            raise ValueError(f"dualrail_ct 须为正：{self.name}")
        if self.plain_ct is not None and self.plain_ct <= 0:
            raise ValueError(f"plain_ct 须为正：{self.name}")
        if self.dualrail_depth < 0:
            raise ValueError(f"dualrail_depth 不能为负：{self.name}")
        if self.layout not in ("real", "dualrail"):
            raise ValueError(f"layout 须为 real|dualrail：{self.name}")
        if self.cost_ct is not None and self.cost_ct <= 0:
            raise ValueError(f"cost_ct 须为正：{self.name}")
        # boot_ct → dualrail_ct 别名（frozen：object.__setattr__）
        if self.dualrail_ct is None and self.boot_ct is not None:
            object.__setattr__(self, "dualrail_ct", self.boot_ct)


@dataclass
class BootstrapPlacement:
    event_index: int
    event_name: str
    ct_count: int
    remaining_before: int
    reason: str  # "before_event" | "mid_event"
    pack_mode: str = "direct"  # "direct" | "real_pack"


@dataclass
class LevelDiscard:
    """
    Bootstrap 后缩短模数链（原生库无法「只恢复到所需长度」时的退路）。

    理想：bts 只恢复接下来要用的 RNS 链长。
    Liberate/THOR：bts 固定回到预算满链（``remaining = budget``），不支持
    按需短链刷新 → bts 后 ``level_up`` 丢掉用不到的
    level，使 ``remaining_keep = 到下一次 bts 所需深度``。
    **禁止** L0→深 ``level_up``；入口应直接加密到 ``level_calc``。

    Liberate：``level_calc_target = num_levels - remaining_keep``。
    """

    site: str  # "initial" | placement event name
    placement_index: int | None  # None for initial (before first event)
    reason: str  # "initial" | "before_event" | "mid_event"
    remaining_after_bts: int  # usually = budget (full refresh)
    remaining_keep: int  # depth until next bts (what we actually need)
    discard: int  # remaining_after_bts - remaining_keep (>= 0)


@dataclass
class BootstrapResult:
    bts_count: int
    placements: list[BootstrapPlacement] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    budget: int = BOOTSTRAP_DEPTH_BUDGET
    initial_level: int = BOOTSTRAP_DEPTH_BUDGET
    level_discards: list[LevelDiscard] = field(default_factory=list)
    # Spine rem after each boot / event (for HE rem_guard fail-fast).
    rem_trace: list[RemTraceRow] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"bts={self.bts_count} "
            f"(events={len(self.events)}, budget={self.budget}, "
            f"placements={len(self.placements)}, "
            f"discards={len(self.level_discards)}, "
            f"rem_trace={len(self.rem_trace)})"
        )


def remaining_to_level_calc(remaining: int, num_levels: int) -> int:
    """Convert desired remaining depth → Liberate ``level_calc`` for ``level_up``."""
    if remaining < 0:
        raise ValueError("remaining < 0")
    if remaining > num_levels:
        raise ValueError(f"remaining {remaining} > num_levels {num_levels}")
    return num_levels - remaining


def _event_spine_depth(ev: Event) -> int:
    """Depth charged on the spine for one event (align + body)."""
    d = ev.depth
    if ev.join and ev.align_depth > 0:
        d += ev.align_depth
    return d


def compute_level_discards(
    events: list[Event],
    placements: list[BootstrapPlacement],
    *,
    budget: int,
    initial_level: int | None = None,
) -> list[LevelDiscard]:
    """
    根据已求出的 bootstrap 放置，计算每次刷新后应 ``level_up`` 丢掉的深度。

    理想情况：bts 只恢复会用到的链长。原生库做不到，只能满预算恢复后再
    discard：若到下一次 bts 只需 ``D < remaining_after_bts``，则保留 ``D``。
    THOR 原版入口 level_calc≈21（剩~8）是其放置下的特例，而非常数。
    """
    if budget <= 0:
        raise ValueError("budget 须为正")
    init = budget if initial_level is None else initial_level

    by_ev: dict[int, list[BootstrapPlacement]] = {}
    for p in placements:
        by_ev.setdefault(p.event_index, []).append(p)

    # moments: (site, placement_index, reason, remaining_after_refresh)
    moments: list[tuple[str, int | None, str, int]] = [
        ("initial", None, "initial", init)
    ]
    depth_used: list[int] = []
    mi = 0
    r = init
    used_since = 0

    def on_boot(site: str, pidx: int | None, reason: str) -> None:
        nonlocal mi, r, used_since
        depth_used.append(used_since)
        used_since = 0
        rb = r
        if (
            pidx is not None
            and 0 <= pidx < len(placements)
            and events[placements[pidx].event_index].companion_boot
            and reason == "before_event"
        ):
            r = rb
        else:
            r = budget
            if pidx is not None and 0 <= pidx < len(placements):
                pl = placements[pidx]
                ev_b = events[pl.event_index]
                r = _boot_land_rem(
                    ev_b, budget, pack_mode=pl.pack_mode or pack_mode_for_event(ev_b)
                )
        moments.append((site, pidx, reason, r))
        mi += 1

    for ei, ev in enumerate(events):
        pls = by_ev.get(ei, [])
        before = [p for p in pls if p.reason == "before_event"]
        mids = [p for p in pls if p.reason == "mid_event"]

        if before:
            on_boot(ev.name, placements.index(before[0]), "before_event")

        d = _event_spine_depth(ev)
        mid_i = 0
        if d <= 0:
            continue
        if r <= 0:
            p = mids[mid_i] if mid_i < len(mids) else None
            on_boot(ev.name, placements.index(p) if p else None, "mid_event")
            mid_i += 1
        while d > r:
            used_since += r
            d -= r
            p = mids[mid_i] if mid_i < len(mids) else None
            on_boot(ev.name, placements.index(p) if p else None, "mid_event")
            mid_i += 1
        used_since += d
        r -= d

    depth_used.append(used_since)  # final segment after last boot

    out: list[LevelDiscard] = []
    for i, (site, pidx, reason, rem_after) in enumerate(moments):
        keep = depth_used[i] if i < len(depth_used) else rem_after
        keep = max(0, min(keep, rem_after))
        # Optional cap on discard (= level_up span). Default uncapped.
        if DISCARD_MAX_SPAN is not None and rem_after - keep > int(DISCARD_MAX_SPAN):
            keep = rem_after - int(DISCARD_MAX_SPAN)
        # DualRail exp 刷新后立刻有重计算：保留满预算
        if pidx is not None and 0 <= pidx < len(placements):
            ev = events[placements[pidx].event_index]
            if ev.no_level_discard:
                keep = rem_after
        out.append(
            LevelDiscard(
                site=site,
                placement_index=pidx,
                reason=reason,
                remaining_after_bts=rem_after,
                remaining_keep=keep,
                discard=rem_after - keep,
            )
        )
    return out


@dataclass
class RemTraceRow:
    """One step of DP remaining-depth simulation (for HE reconcile)."""

    kind: str  # "start" | "before_boot" | "after_boot" | "after_event"
    event_name: str
    event_index: int | None
    rem_before: int
    rem_after: int
    depth_charged: int  # rem_before - rem_after for event consume; 0 for boots
    boot_reason: str | None = None  # before_event | mid_event | None
    boot_pack_mode: str | None = None


def simulate_remaining_trace(
    events: list[Event],
    placements: list[BootstrapPlacement],
    *,
    budget: int,
    initial_level: int | None = None,
) -> list[RemTraceRow]:
    """
    Replay placements on the event spine; emit rem before/after each boot and
    after each event body. Same consume rules as ``compute_level_discards``.
    """
    if budget <= 0:
        raise ValueError("budget 须为正")
    init = budget if initial_level is None else int(initial_level)
    by_ev: dict[int, list[BootstrapPlacement]] = {}
    for p in placements:
        by_ev.setdefault(p.event_index, []).append(p)

    rows: list[RemTraceRow] = []
    r = min(init, budget)
    rows.append(
        RemTraceRow(
            kind="start",
            event_name="__initial__",
            event_index=None,
            rem_before=r,
            rem_after=r,
            depth_charged=0,
        )
    )
    # DualRail LN 入口 boot 后，融合 prep 只烧 DEPTH_LN_PREP；否则冷启动 COLD。
    dualrail_ln_entry = False

    def do_boot(ev: Event, ei: int, p: BootstrapPlacement | None, reason: str) -> None:
        nonlocal r
        pm = (
            (p.pack_mode if p is not None else None)
            or pack_mode_for_event(ev)
        )
        rb = r
        if ev.companion_boot and reason == "before_event":
            pass  # summation companion；spine rem 不变
        else:
            r = _boot_land_rem(ev, budget, pack_mode=pm)
        rows.append(
            RemTraceRow(
                kind="after_boot",
                event_name=ev.name,
                event_index=ei,
                rem_before=rb,
                rem_after=r,
                depth_charged=0,
                boot_reason=reason,
                boot_pack_mode=pm,
            )
        )

    def _finish_event(ev: Event, ei: int, rb: int, depth_charged: int) -> None:
        """Record rem-only after_event（无 slot save/clear / post_invsqrt min）。"""
        rows.append(
            RemTraceRow(
                kind="after_event",
                event_name=ev.name,
                event_index=ei,
                rem_before=rb,
                rem_after=r,
                depth_charged=depth_charged,
            )
        )

    for ei, ev in enumerate(events):
        pls = by_ev.get(ei, [])
        before = [p for p in pls if p.reason == "before_event"]
        mids = [p for p in pls if p.reason == "mid_event"]
        if before:
            rows.append(
                RemTraceRow(
                    kind="before_boot",
                    event_name=ev.name,
                    event_index=ei,
                    rem_before=r,
                    rem_after=r,
                    depth_charged=0,
                    boot_reason="before_event",
                    boot_pack_mode=(before[0].pack_mode or pack_mode_for_event(ev)),
                )
            )
            do_boot(ev, ei, before[0], "before_event")
            if int(getattr(ev, "dualrail_depth", 0) or 0) > 0 and (
                ev.name.endswith(".ln1_scale")
                or ev.name.endswith(".ln2_scale")
            ):
                dualrail_ln_entry = True

        d = _event_spine_depth(ev)

        mid_i = 0
        rb = r
        # HE inv√：plan mid 在段首 in_plan 即刷（不必等 rem<MIN）；先落 planned mid。
        # （现行一律 atomic 后 DP 不再产生 mid；保留回放兼容旧 placements。）
        while mid_i < len(mids):
            p = mids[mid_i]
            rows.append(
                RemTraceRow(
                    kind="before_boot",
                    event_name=ev.name,
                    event_index=ei,
                    rem_before=r,
                    rem_after=r,
                    depth_charged=0,
                    boot_reason="mid_event",
                    boot_pack_mode=(p.pack_mode or pack_mode_for_event(ev)),
                )
            )
            do_boot(ev, ei, p, "mid_event")
            if int(getattr(ev, "dualrail_depth", 0) or 0) > 0 and (
                ev.name.endswith(".ln1_scale")
                or ev.name.endswith(".ln2_scale")
            ):
                dualrail_ln_entry = True
            mid_i += 1
            rb = r

        # LN scale/var 已声明 depth；DualRail 入口仅作标记（unpack 税在 dualrail_depth）。
        if ev.name.endswith(".ln1_scale") or ev.name.endswith(".ln2_scale"):
            dualrail_ln_entry = False

        if d <= 0:
            _finish_event(ev, ei, rb, 0)
            continue
        leave = int(LIBERATE_MIN_LEAVE_REM)
        # 一律 atomic：禁止把 depth 拆成多段 mid-boot；也不允许烧到 rem=0。
        if r < d + leave:
            raise RuntimeError(
                f"rem_trace: {ev.name} rem={r} < depth+MIN_LEAVE={d}+{leave}; "
                f"missing before_event placement (atomic micro-event)"
            )
        r -= d
        _finish_event(ev, ei, rb, d)
    return rows


def _validate_scheme_poly_only(scheme: list[int], task_name: str = "") -> None:
    """bts 路径不允许 original(3)。"""
    validate_scheme(scheme, task_name)
    for idx, level in enumerate(scheme):
        if level not in (0, 1, 2):
            raise ValueError(
                f"{task_name} bts_ops 仅允许档位 0/1/2，"
                f"下标 {idx} 为 {level}（不支持 original）"
            )


def _slot_level(scheme: list[int], layer_idx: int, kind: str) -> int:
    return scheme[scheme_index(layer_idx, kind)]


def _gelu_bts_depth(layer_idx: int, level: int) -> int:
    """GeLU 段乘法深度 = ``gelu_poly_depth``（Chebyshev depth_he）。

    （否则 after_gelu rem 比 HE 少 2：max(10,12)=12 vs HE Δ=10）。
    """
    return int(gelu_poly_depth(layer_idx, level))


def _nonlinear_depth(
    task_name: str, layer_idx: int, kind: str, level: int
) -> int:
    if kind == "softmax":
        return softmax_poly_depth(task_name, layer_idx, level)
    if kind == "gelu":
        return _gelu_bts_depth(layer_idx, level)
    if kind in ("ln1", "ln2"):
        return layernorm_poly_depth(task_name, layer_idx, kind, level)
    raise ValueError(f"未知非线性 kind：{kind}")


def _ev(
    prefix: str,
    name: str,
    depth: int,
    ct_in: int,
    ct_out: int,
    layer_idx: int,
    kind: str,
    *,
    join: bool = False,
    ct_peak: int | None = None,
    align_depth: int = 0,
    force_bootstrap: bool = False,
    boot_ct: int | None = None,
    plain_ct: int | None = None,
    dualrail_ct: int | None = None,
    dualrail_depth: int = 0,
    no_level_discard: bool = False,
    atomic: bool = True,  # ignored: all micro-events are atomic
    layout: str = "real",
    cost_ct: int | None = None,
    companion_boot: bool = False,
) -> Event:
    del atomic  # 现行划分一律 atomic；保留形参以免旧调用炸掉
    return Event(
        name=f"{prefix}.{name}",
        depth=depth,
        ct_in=ct_in,
        ct_out=ct_out,
        layer_idx=layer_idx,
        kind=kind,
        join=join,
        ct_peak=ct_peak,
        align_depth=align_depth,
        force_bootstrap=force_bootstrap,
        boot_ct=boot_ct,
        plain_ct=plain_ct,
        dualrail_ct=dualrail_ct,
        dualrail_depth=dualrail_depth,
        no_level_discard=no_level_discard,
        atomic=True,
        layout=layout,
        cost_ct=cost_ct,
        companion_boot=companion_boot,
    )


def _softmax_params(
    task_name: str, layer_idx: int, level: int
) -> tuple[int, list[int], int, int]:
    """返回 (iters_σ, iters_Σy²[每轮], rounds, total_depth)。"""
    level_key = SOFTMAX_LEVEL_KEYS[level]
    iters_sigma = LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK[task_name][level_key][
        layer_idx
    ]
    iters_sq1 = LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK[task_name][level_key][
        layer_idx
    ]
    iters_sq2 = LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK[task_name][level_key][
        layer_idx
    ]
    exp_div = LAYER_EXP_DIV_BY_TASK[task_name][layer_idx]
    rounds = log2_ceil(exp_div)
    iters_sum_sq = [iters_sq1]
    if rounds >= 2:
        iters_sum_sq.append(iters_sq2)
    iters_sum_sq = iters_sum_sq[:rounds]
    total = softmax_poly_depth(task_name, layer_idx, level)
    split = (
        SOFTMAX_DEPTH_BASE
        + iters_sigma * SOFTMAX_ASOR_ITER_DEPTH
        + sum(iters_sum_sq) * SOFTMAX_ASOR_ITER_DEPTH
    )
    if split != total:
        raise RuntimeError(
            f"softmax depth 拆分不一致 L{layer_idx} level={level}: "
            f"split={split} total={total}"
        )
    return iters_sigma, iters_sum_sq, rounds, total


def _cheb_nonzero_count(coeffs: list) -> int:
    return sum(1 for c in coeffs if abs(float(c)) > 1e-12)


def _cheb_tree_peak(base_ct: int, step: int, degree: int) -> int:
    """
    Cheb 平衡 T_k 树第 ``step`` 层（0-based）的活密文峰值。

    构造阶段并行节点数约 ``min(2^(step+1), degree+1)``；中途 bts 按 ``base_ct * live`` 计。
    """
    live = min(1 << (step + 1), max(degree, 1) + 1)
    return base_ct * live


def _append_cheb_ps_tree_events(
    events: list[Event],
    prefix: str,
    tag: str,
    degree: int,
    coeffs: list,
    ct_base: int,
    layer_idx: int,
    kind: str,
) -> None:
    """
    Chebyshev PS-tree 微展开：每层乘法 depth=1；末步 combine（pt 系数，depth=0）。

    树构造后 spine rem 已等于最深 T_k；combine 输出取该 rem（浅项 level_up 到 spine，
    不额外降低 spine remaining）。
    """
    if degree <= 0:
        return
    nz = max(_cheb_nonzero_count(coeffs), 1)
    levels = cheb_he_depth_ps_tree(degree)

    for lv in range(levels):
        events.append(
            _ev(
                prefix,
                f"{tag}_ps_L{lv}",
                1,
                ct_base,
                ct_base,
                layer_idx,
                kind,
                ct_peak=_cheb_tree_peak(ct_base, lv, degree),
            )
        )
    combine_peak = ct_base * min(nz, degree + 1)
    events.append(
        _ev(
            prefix,
            f"{tag}_ps_combine",
            0,
            ct_base,
            ct_base,
            layer_idx,
            kind,
            ct_peak=combine_peak,
        )
    )


def _append_softmax_dualrail_exit(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    *,
    use_aux: bool = True,
) -> None:
    """Softmax 出口：主路 inv(1) → 128 α（旁路 exp 对齐在 HE 侧）。"""
    g = active_geometry()
    events.append(
        _ev(
            prefix,
            "softmax_dualrail_exit",
            DEPTH_SOFTMAX_EXIT,
            CT_SOFTMAX_INV_SPINE,
            g.ct_alpha_copies,
            layer_idx,
            "pathway",
            ct_peak=g.ct_alpha_copies,
            plain_ct=g.ct_att_score,
            layout="real",
            no_level_discard=True,
            atomic=True,
            # 主路径 = inv（1）；exp 旁路乐观不计
            cost_ct=1 if use_aux else None,
        )
    )


def _append_softmax_exp_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
) -> None:
    """
    thor exp：Stockmeyer(deg15, depth=4) + δ1 平方(depth=1)；之和 = SOFTMAX_DEPTH_BASE。
    """
    g = active_geometry()
    peak = g.ct_att_score * g.stockmeyer_live_factor
    # Stockmeyer 整段不可中途 bts；不够 rem 只能段首刷（8 real，计价 n//2）。
    # HE：smoke DualRail8 unpack ×½ → dualrail_depth=1。
    events.append(
        _ev(
            prefix,
            "softmax_exp_stockmeyer",
            SOFTMAX_EXP_STOCKMEYER_DEPTH,
            g.ct_att_score,
            g.ct_att_score,
            layer_idx,
            "softmax",
            ct_peak=peak,
            plain_ct=g.ct_att_score,
            layout="real",
            atomic=True,
            dualrail_depth=DUALRAIL_UNPACK_DEPTH,
        )
    )
    events.append(
        _ev(
            prefix,
            "softmax_exp_square_0",
            SOFTMAX_EXP_SQUARE_DEPTH,
            g.ct_att_score,
            g.ct_att_score,
            layer_idx,
            "softmax",
            atomic=True,
        )
    )


def _append_softmax_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    level: int,
) -> None:
    """阶段 B：exp → aSOR_σ → Σy² 轮 → DualRail exit。"""
    g = active_geometry()
    iters_sigma, iters_sum_sq, rounds, _ = _softmax_params(
        task_name, layer_idx, level
    )
    _append_softmax_exp_events(events, prefix, layer_idx)
    events.append(
        _ev(
            prefix,
            "softmax_asor_sigma",
            iters_sigma * SOFTMAX_ASOR_ITER_DEPTH,
            g.ct_att_score,
            g.ct_att_score,
            layer_idx,
            "softmax",
            join=True,
        )
    )
    for r in range(rounds):
        events.append(
            _ev(
                prefix,
                f"softmax_norm_r{r}",
                iters_sum_sq[r] * SOFTMAX_ASOR_ITER_DEPTH,
                g.ct_att_score,
                g.ct_att_score,
                layer_idx,
                "softmax",
                join=True,
            )
        )
    _append_softmax_dualrail_exit(events, prefix, layer_idx, use_aux=False)


def _append_softmax_events_c(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    level: int,
) -> None:
    """
    阶段 C：exp + attn mask + 逐次 aSOR_σ + 每轮 Σy²(apply + summation@asor0 + aSOR)
    + DualRail 8→128 exit（对齐 ``softmax_he_ct``，非 copy 链）。

    ``softmax_sigma_enc_one`` / ``sumsq_r*_enc_one``：段首只刷 1 条 live CT
    （direct，落地 rem=budget）。随后 ``encode_and_encrypt`` 出 enc_one，不占 bts。
    下一站 ``asor_*`` 才是 an,bn ``real_pack``（代价同为 1，落地 budget−1）。
    """
    g = active_geometry()
    iters_sigma, iters_sum_sq, rounds, _ = _softmax_params(
        task_name, layer_idx, level
    )
    _append_softmax_exp_events(events, prefix, layer_idx)
    events.append(
        _ev(
            prefix,
            "softmax_attn_mask",
            DEPTH_SOFTMAX_ATTN_MASK,
            g.ct_att_score,
            g.ct_att_score,
            layer_idx,
            "pathway",
        )
    )
    events.append(
        _ev(
            prefix,
            "softmax_sigma_aggregate",
            DEPTH_SOFTMAX_SIGMA_AGGREGATE,
            g.ct_att_score,
            CT_SOFTMAX_INV_SPINE,
            layer_idx,
            "pathway",
            plain_ct=CT_SOFTMAX_INV_SPINE,
            layout="real",
        )
    )
    events.append(
        _ev(
            prefix,
            "softmax_sigma_enc_one",
            DEPTH_SOFTMAX_SIGMA_ENC_ONE,
            CT_SOFTMAX_INV_SPINE,
            CT_SOFTMAX_ASOR_SPINE,
            layer_idx,
            "pathway",
            # 段首只刷 σ（1 CT、direct、落地 budget）；enc_one 是随后 encode，不占 bts
            plain_ct=1,
            layout="dualrail",
            cost_ct=1,
        )
    )
    for i in range(iters_sigma):
        events.append(
            _ev(
                prefix,
                f"softmax_asor_sigma_i{i}",
                SOFTMAX_ASOR_ITER_DEPTH,
                CT_SOFTMAX_ASOR_SPINE,
                CT_SOFTMAX_ASOR_SPINE,
                layer_idx,
                "softmax",
                # 2 实数 an/bn → DualRail 打包刷：cost=1，落地 rem=budget−1
                plain_ct=2,
                layout="real",
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
                atomic=True,
            )
        )
    for r in range(rounds):
        events.append(
            _ev(
                prefix,
                f"softmax_sumsq_r{r}_apply",
                DEPTH_SOFTMAX_SUMSQ_APPLY,
                CT_SOFTMAX_INV_SPINE,
                CT_SOFTMAX_INV_SPINE,
                layer_idx,
                "pathway",
                plain_ct=CT_SOFTMAX_INV_SPINE,
                layout="real",
                no_level_discard=True,
                # 主路 inv→summation；旁路 exp 对齐在 HE 侧
                cost_ct=CT_SOFTMAX_INV_SPINE,
                atomic=True,
            )
        )
        events.append(
            _ev(
                prefix,
                f"softmax_sumsq_r{r}_enc_one",
                DEPTH_SOFTMAX_SUMSQ_ENC_ONE,
                CT_SOFTMAX_INV_SPINE,
                CT_SOFTMAX_ASOR_SPINE,
                layer_idx,
                "pathway",
                # 段首只刷 summation×1（direct、落地 budget）；enc_one 随后 encode
                plain_ct=1,
                layout="dualrail",
                cost_ct=1,
            )
        )
        for gi in range(iters_sum_sq[r]):
            events.append(
                _ev(
                    prefix,
                    f"softmax_norm_r{r}_asor{gi}",
                    SOFTMAX_ASOR_ITER_DEPTH,
                    CT_SOFTMAX_ASOR_SPINE,
                    CT_SOFTMAX_ASOR_SPINE,
                    layer_idx,
                    "softmax",
                    plain_ct=2,
                    layout="real",
                    dualrail_depth=DUALRAIL_UNPACK_DEPTH,
                    atomic=True,
                )
            )
        # aSOR 后 cm_mult(inv, head-mask)；在 DualRail exit 之前烧 spine rem
        events.append(
            _ev(
                prefix,
                f"softmax_norm_r{r}_inv_mask",
                DEPTH_SOFTMAX_INV_MASK,
                CT_SOFTMAX_INV_SPINE,
                CT_SOFTMAX_INV_SPINE,
                layer_idx,
                "pathway",
                plain_ct=CT_SOFTMAX_INV_SPINE,
                no_level_discard=True,
            )
        )
    _append_softmax_dualrail_exit(events, prefix, layer_idx)


def _append_gelu_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    level: int,
) -> None:
    """阶段 B：Cheb 复合 f1→f2→×x（深度来自 gelu_he_depth_breakdown）。"""
    g = active_geometry()
    cfg = gelu_config_for_layer(layer_idx, level)
    if cfg["kind"] == "composite":
        bd = gelu_he_depth_breakdown(
            cfg["d1"],
            cfg["d2"],
            eval_method="ps_tree",
            f2_domain=cfg["f2_domain"],
        )
        combo = int(bd.get("cheb_combo_rescale", 0)) // 2
        f1_d = int(bd["f1_eval"]) + combo
        f2_d = int(bd.get("f2_affine", 0)) + int(bd["f2_eval"]) + combo
        recon_d = int(bd["gelu_reconstruct"])
    else:
        bd = gelu_he_depth_breakdown(cfg["degree"], eval_method="ps_tree")
        f1_d = int(bd["y_eval"]) + int(bd.get("cheb_combo_rescale", 0))
        f2_d = 0
        recon_d = int(bd["gelu_reconstruct"])

    # 生产路径为 Chebyshev PS-tree；峰值按 FF live packs
    peak = g.ct_ffn
    if f1_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_f1",
                f1_d,
                g.ct_ffn,
                g.ct_ffn,
                layer_idx,
                "gelu",
                join=True,
                ct_peak=peak,
            )
        )
    if f2_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_f2",
                f2_d,
                g.ct_ffn,
                g.ct_ffn,
                layer_idx,
                "gelu",
                join=True,
                ct_peak=peak,
            )
        )
    events.append(
        _ev(
            prefix,
            "gelu_reconstruct",
            recon_d,
            g.ct_ffn,
            g.ct_ffn,
            layer_idx,
            "gelu",
        )
    )
    expected = _gelu_bts_depth(layer_idx, level)
    # Phase B split from poly breakdown; pad slack onto reconstruct.
    actual = sum(
        e.depth for e in events if e.layer_idx == layer_idx and e.kind == "gelu"
    )
    if actual < expected:
        events.append(
            _ev(
                prefix,
                "gelu_slack",
                expected - actual,
                g.ct_ffn,
                g.ct_ffn,
                layer_idx,
                "gelu",
            )
        )
        actual = expected
    if actual != expected:
        raise RuntimeError(
            f"GeLU depth 拆分不一致 L{layer_idx} level={level}: "
            f"split={actual} expected={expected}"
        )


def _append_gelu_events_c(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    level: int,
) -> None:
    """
    阶段 C：对齐 ``gelu_he_ct`` pack 路径。

    - ``gelu_f1``：f1 Chebyshev（旁路 ``t`` 对齐在 HE 侧，不进 DP）
    - ``gelu_f2``：f2 Chebyshev（段首可刷 f1）
    - ``gelu_reconstruct``：主路 ``y``（段首可刷）；旁路 ``x_phys=t·C`` 由 HE
      ``align_bypass_to_main`` 会合后 ct×ct（depth=1，仅 mul；``×C`` 不计主路）
    """
    g = active_geometry()
    cfg = gelu_config_for_layer(layer_idx, level)
    if cfg["kind"] != "composite":
        raise RuntimeError(
            f"phase C GeLU expects composite, got {cfg['kind']!r} "
            f"L{layer_idx} level={level}"
        )
    bd = gelu_he_depth_breakdown(
        cfg["d1"],
        cfg["d2"],
        eval_method="ps_tree",
        f2_domain=cfg["f2_domain"],
    )
    combo = int(bd.get("cheb_combo_rescale", 0)) // 2
    f1_d = int(bd["f1_eval"]) + combo
    f2_d = int(bd.get("f2_affine", 0)) + int(bd["f2_eval"]) + combo
    recon_d = int(bd["gelu_reconstruct"])  # ct×ct only (=1)
    expected = _gelu_bts_depth(layer_idx, level)
    F = g.ct_ffn
    if f1_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_f1",
                f1_d,
                F,
                F,
                layer_idx,
                "gelu",
                ct_peak=F,
                plain_ct=F,
                layout="real",
                atomic=True,
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            )
        )
    if f2_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_f2",
                f2_d,
                F,
                F,
                layer_idx,
                "gelu",
                ct_peak=F,
                plain_ct=F,
                layout="real",
                atomic=True,
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            )
        )
    if recon_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_reconstruct",
                recon_d,
                F,
                F,
                layer_idx,
                "gelu",
                ct_peak=F,
                plain_ct=F,
                layout="real",
                atomic=True,
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            )
        )
    actual = sum(
        e.depth for e in events if e.layer_idx == layer_idx and e.kind == "gelu"
    )
    if actual != expected:
        raise RuntimeError(
            f"GeLU depth 拆分不一致 L{layer_idx} level={level}: "
            f"split={actual} expected={expected} "
            f"(f1={f1_d} f2={f2_d} recon={recon_d})"
        )


def _append_layernorm_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    kind: str,
    level: int,
) -> None:
    """阶段 B：统计 prep（depth=OVERHEAD）+ 逐次 invsqrt（每 iter depth=1）。"""
    g = active_geometry()
    hr = g.ct_hidden_real
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
    slot = kind  # ln1 | ln2
    events.append(
        _ev(
            prefix,
            f"{slot}_prep",
            LAYERNORM_DEPTH_OVERHEAD,
            hr,
            hr,
            layer_idx,
            kind,
            join=True,
        )
    )
    for i in range(iters):
        events.append(
            _ev(
                prefix,
                f"{slot}_invsqrt_{i}",
                1,
                hr,
                hr,
                layer_idx,
                kind,
                join=(i > 0),
            )
        )
    expected = layernorm_poly_depth(task_name, layer_idx, kind, level)
    actual = sum(
        e.depth
        for e in events
        if e.layer_idx == layer_idx and e.kind == kind
    )
    if actual != expected:
        raise RuntimeError(
            f"LN depth 拆分不一致 L{layer_idx} {kind} level={level}: "
            f"split={actual} expected={expected}"
        )


def _append_layernorm_events_c(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    kind: str,
    level: int,
) -> None:
    """阶段 C：scale → var → enc_one → inv√ → post。

    对照 ``he_layernorm_poly`` / ``he_invsqrt_ct``：
      ``*_scale``：8 real 普通打包 → DualRail 刷（代价 4、落地 budget−1）
        → cm_mult(mask_scale) 8→8, d=1
      ``*_var``：段首 DualRail 刷 enc_l×8（代价 4、落地 budget−1）
        → variance 8→1；numerator 旁路由 HE 对齐（不进 DP）, d=2
      ``*_enc_one``：段首直接刷 variance×1（代价 1、落地 budget）→ +enc_one 1→2, d=0
      ``*_invsqrt_{i}``：段首 DualRail 打包刷 (an,bn)（代价 1、落地 budget−1）
        → 一轮 inv√ 2→2, d=2
      ``*_post_invsqrt``：段首直接刷 denom×1（代价 1、落地 budget）；
        旁路 num ``align_bypass_to_main`` → γ/β 1→8, d=2
        （broadcast 不进图；残差会合亦不进图。
        an/bn→denom 丢掉 an 不单独成事件：打包后仍是 1 cplx。）
    """
    g = active_geometry()
    hr = g.ct_hidden_real
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
    slot = kind
    events.append(
        _ev(
            prefix,
            f"{slot}_scale",
            DEPTH_LN_SCALE,
            hr,
            hr,
            layer_idx,
            kind,
            # 8 real 普通打包 → DualRail：cost=4，落地 budget−1
            plain_ct=hr,
            layout="real",
            dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            atomic=True,
        )
    )
    events.append(
        _ev(
            prefix,
            f"{slot}_var",
            DEPTH_LN_VAR,
            hr,
            CT_LN_VAR_SPINE,
            layer_idx,
            kind,
            # 段首刷 8×enc_l（普通打包）；出口 variance 主路 1
            plain_ct=hr,
            layout="real",
            dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            atomic=True,
        )
    )
    events.append(
        _ev(
            prefix,
            f"{slot}_enc_one",
            DEPTH_LN_ENC_ONE,
            CT_LN_VAR_SPINE,
            CT_LN_INVSQRT_SPINE,
            layer_idx,
            kind,
            # 段首只刷 variance（单 CT 直接刷）：cost=1，落地 budget
            plain_ct=1,
            layout="dualrail",
            cost_ct=1,
        )
    )
    for i in range(iters):
        events.append(
            _ev(
                prefix,
                f"{slot}_invsqrt_{i}",
                LN_INVSQRT_ITER_DEPTH,
                CT_LN_INVSQRT_SPINE,
                CT_LN_INVSQRT_SPINE,
                layer_idx,
                kind,
                # 2 实数 an/bn 普通打包 → DualRail：cost=1，落地 budget−1
                plain_ct=CT_LN_INVSQRT_SPINE,
                layout="real",
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
                atomic=True,
            )
        )
    events.append(
        _ev(
            prefix,
            f"{slot}_post_invsqrt",
            DEPTH_LN_POST_INVSQRT,
            CT_LN_DENOM_SPINE,
            hr,
            layer_idx,
            kind,
            # 段首只刷 denom（单 CT 直接刷）；旁路 num 对齐在 HE 侧
            plain_ct=1,
            layout="dualrail",
            cost_ct=1,
        )
    )
    expected = (
        DEPTH_LN_SCALE
        + DEPTH_LN_VAR
        + DEPTH_LN_ENC_ONE
        + iters * LN_INVSQRT_ITER_DEPTH
        + DEPTH_LN_POST_INVSQRT
    )
    actual = sum(
        e.depth
        for e in events
        if e.layer_idx == layer_idx and e.kind == kind
    )
    if actual != expected:
        raise RuntimeError(
            f"LN depth 拆分不一致 L{layer_idx} {kind} level={level}: "
            f"split={actual} expected={expected}"
        )


def build_bootstrap_events(
    task_name: str,
    scheme: list[int],
    *,
    phase: str = "C",
    geom: SlotCtGeometry | None = None,
    depth_overrides: dict[str, int] | None = None,
) -> list[Event]:
    """
    展开事件序列（CT 几何对齐 ``smoke_thor_repro._forward_layer``）。

    phase A：非线性整段粗事件；B：微展开；C（默认）：微事件 + Softmax 通路 depth
    + 固定 DualRail / Σy² summation bootstrap 站点。
    ``geom``：``None`` 时用当前活动几何（默认 bert_base）。
    """
    if geom is not None:
        set_slot_geometry(geom)
    g = active_geometry()

    _validate_scheme_poly_only(scheme, task_name)
    events: list[Event] = []
    ph = phase.upper()
    if ph not in ("A", "B", "C"):
        raise ValueError(f"phase 须为 A/B/C，收到 {phase!r}")

    Hc = g.ct_hidden_cplx
    Hr = g.ct_hidden_real
    Q, S = g.ct_qkv_out, g.ct_att_score
    C = g.ct_context_cplx
    R, Rctx = g.ct_pc_rot, g.ct_pc_rot_ctx
    Qc, A, F = g.ct_q_copies, g.ct_alpha_copies, g.ct_ffn

    for layer_idx in range(NUM_LAYERS):
        p = f"L{layer_idx}"

        # --- Attention：层入口固定 8real→4cplx（不进微事件）；首事件 linear_qkv ---
        # 段首若刷 4 cplx：HE unpack→残差 x（图上不记残差旁路 rem）
        events.append(
            _ev(
                p,
                "linear_qkv",
                DEPTH_QKV,
                Hc,
                2 * Q,
                layer_idx,
                "linear",
                layout="dualrail",
                cost_ct=Hc,  # before：4 cplx
                ct_peak=R,  # 段内 rot 峰值 64
                atomic=True,
            )
        )
        # Q∥K→sftmx_in 原子岛：段前 bts 刷 Q+K（不计 V）；段后由下一站刷 sftmx_in。
        # V 旁路对齐在 HE 侧（不进 DP）。
        events.append(
            _ev(
                p,
                "linear_att_score",
                DEPTH_QK_SCORE,
                2 * Q,
                S,
                layer_idx,
                "linear",
                layout="dualrail",
                cost_ct=2 * Q,  # before：Q+K
                ct_peak=Qc,
                atomic=True,
            )
        )
        sm_level = _slot_level(scheme, layer_idx, "softmax")
        if ph == "A":
            sm_depth = _nonlinear_depth(task_name, layer_idx, "softmax", sm_level)
            events.append(
                _ev(p, "softmax", sm_depth, S, A, layer_idx, "softmax")
            )
        elif ph == "B":
            _append_softmax_events(events, p, layer_idx, task_name, sm_level)
        else:
            _append_softmax_events_c(events, p, layer_idx, task_name, sm_level)

        # Context：α(128) × V(2) → context(2)；V 对齐在 HE 侧
        events.append(
            _ev(
                p,
                "linear_att_context",
                DEPTH_ATT_CONTEXT,
                A,
                C,
                layer_idx,
                "linear",
                join=True,
                layout="real",
                # 主路径 = α；V 旁路不计
                cost_ct=A // 2,
                atomic=True,
            )
        )
        events.append(
            _ev(
                p,
                "bridge_rot_attn_dense",
                0,
                C,
                Rctx,
                layer_idx,
                "bridge",
                layout="dualrail",
            )
        )
        events.append(
            _ev(
                p,
                "linear_attn_dense",
                DEPTH_ATT_DENSE,
                Rctx,
                Hr,
                layer_idx,
                "linear",
                layout="dualrail",
                atomic=True,
            )
        )
        # 残差会合不进微事件：HE ``align_bypass_to_main`` + add → ln1
        ln1_level = _slot_level(scheme, layer_idx, "ln1")
        if ph == "A":
            ln1_depth = _nonlinear_depth(task_name, layer_idx, "ln1", ln1_level)
            events.append(_ev(p, "ln1", ln1_depth, Hr, Hr, layer_idx, "ln1"))
        elif ph == "B":
            _append_layernorm_events(
                events, p, layer_idx, task_name, "ln1", ln1_level
            )
        else:
            _append_layernorm_events_c(
                events, p, layer_idx, task_name, "ln1", ln1_level
            )

        # --- FFN：8 real → pack/rot 64 → dense1 → (2,8)；dense2 直接回 8 ---
        # FF 残差会合不进微事件（HE align+add）；LN1 出口即 bridge 入口
        events.append(
            _ev(
                p,
                "bridge_rot_ff1",
                DEPTH_FF_BRIDGE,
                Hr,
                R,
                layer_idx,
                "bridge",
                # 8 real LN1 普通打包 → DualRail：cost=4，落地 budget−1
                plain_ct=Hr,
                layout="real",
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            )
        )
        events.append(
            _ev(
                p,
                "linear_ff_dense1",
                DEPTH_FF_DENSE,
                R,
                F,
                layer_idx,
                "linear",
                # 64 rot 已是复数侧：直接刷，cost=64，落地 budget
                plain_ct=R,
                layout="dualrail",
                atomic=True,
            )
        )
        gelu_level = _slot_level(scheme, layer_idx, "gelu")
        if ph == "A":
            gelu_depth = _nonlinear_depth(task_name, layer_idx, "gelu", gelu_level)
            events.append(_ev(p, "gelu", gelu_depth, F, F, layer_idx, "gelu"))
        elif ph == "B":
            _append_gelu_events(events, p, layer_idx, gelu_level)
        else:
            _append_gelu_events_c(events, p, layer_idx, gelu_level)

        # ``ThorBertFF.dense2``：(2,8)→(8,)；段前 DualRail 刷 GeLU 出口
        events.append(
            _ev(
                p,
                "linear_ff_dense2",
                DEPTH_FF_DENSE2_KEEP,
                F,
                Hr,
                layer_idx,
                "linear",
                layout="real",
                atomic=True,
                dualrail_depth=DUALRAIL_UNPACK_DEPTH,
            )
        )
        # 残差会合不进微事件：HE ``align_bypass_to_main`` + add → ln2
        ln2_level = _slot_level(scheme, layer_idx, "ln2")
        if ph == "A":
            ln2_depth = _nonlinear_depth(task_name, layer_idx, "ln2", ln2_level)
            events.append(_ev(p, "ln2", ln2_depth, Hr, Hr, layer_idx, "ln2"))
        elif ph == "B":
            _append_layernorm_events(
                events, p, layer_idx, task_name, "ln2", ln2_level
            )
        else:
            _append_layernorm_events_c(
                events, p, layer_idx, task_name, "ln2", ln2_level
            )

    _validate_event_chain(events)
    return _apply_depth_overrides(events, depth_overrides)


def _apply_depth_overrides(
    events: list[Event],
    depth_overrides: dict[str, int] | None,
) -> list[Event]:
    """Route B: replace ``Event.depth`` with HE-measured segment depths."""
    if not depth_overrides:
        return events
    out: list[Event] = []
    for ev in events:
        d = depth_overrides.get(ev.name)
        if d is None:
            suf = ev.name.split(".", 1)[-1]
            d = depth_overrides.get(suf)
        if d is not None:
            out.append(replace(ev, depth=max(0, int(d))))
        else:
            out.append(ev)
    return out


def _validate_event_chain(events: list[Event]) -> None:
    if not events:
        return
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        # 层边界：上一段 LN2 出口 8 real → 下一段入口固定 pack 成 4 cplx（非微事件）
        if a.layer_idx != b.layer_idx:
            continue
        # 衔接看 DP 代价（打包后关键路径 CT），不是表面条数。
        # 例：inv√ 出口 an,bn 两条 real，post 入口 denom=bn；打包后都是 1 cplx。
        if a.ct_out != b.ct_in and _cost_ct(a) != _cost_ct(b):
            raise ValueError(
                f"CT 衔接断裂：{a.name} ct_out={a.ct_out} cost={_cost_ct(a)} → "
                f"{b.name} ct_in={b.ct_in} cost={_cost_ct(b)}"
            )


def _consume_depth(
    remaining: int,
    depth: int,
    ct_count: int,
    budget: int,
) -> tuple[int, int, int]:
    """Atomic 消耗（兼容旧调用）：不够 ``depth+MIN_LEAVE`` 则不可行。"""
    del ct_count, budget
    if depth < 0:
        raise ValueError("depth < 0")
    leave = int(LIBERATE_MIN_LEAVE_REM)
    if depth == 0:
        return 0, remaining, 0
    if remaining < depth + leave:
        return 10**9, remaining, 0
    return 0, remaining - depth, 0


def _consume_event_depth(
    remaining: int,
    ev: Event,
    budget: int,
) -> tuple[int, int, int]:
    """兼容旧路径：固定 align_depth + depth（无 slot）。"""
    total_add = 0
    total_mid = 0
    r = remaining
    if ev.join and ev.align_depth > 0:
        add, r, mid = _consume_depth(r, ev.align_depth, ev.bts_ct, budget)
        total_add += add
        total_mid += mid
    add, r, mid = _consume_depth(r, ev.depth, ev.bts_ct, budget)
    return total_add + add, r, total_mid + mid


def _side_remaining(slots: SlotVec, ev: Event) -> int:
    """Unused with N_SLOTS=0；保留签名以免旧调用炸掉。"""
    assert ev.join_slot is not None
    saved = slots[ev.join_slot]
    if saved == SLOT_UNSET:
        raise RuntimeError(f"{ev.name}: join_slot={ev.join_slot} 尚未 fork save")
    return max(0, saved - ev.join_offset)


def _pad_slots(slots: tuple[int, ...] | list[int]) -> list[int]:
    sl = list(slots)
    if len(sl) < N_SLOTS:
        sl.extend([SLOT_UNSET] * (N_SLOTS - len(sl)))
    return sl[:N_SLOTS]


def _slots_tuple(slot_list: list[int]) -> SlotVec:
    return tuple(slot_list[:N_SLOTS])


def _spine_plain_ct(ev: Event) -> int:
    """主路径现场条数（不含旁路）。"""
    if ev.plain_ct is not None:
        return int(ev.plain_ct)
    if ev.ct_peak is not None:
        return max(ev.ct_in, ev.ct_peak)
    return ev.ct_in


def pack_mode_for_event(ev: Event) -> str:
    """
    Bts 段首刷新模式。

    - ``direct``：复数 spine / 单条 real / 无 unpack 税 → rem=budget。
    - ``real_pack``：多条 real 且 ``dualrail_depth>0`` → pack→bts→unpack×½。
    """
    if ev.layout == "dualrail":
        return "direct"
    if int(ev.dualrail_depth) <= 0:
        return "direct"
    if _spine_plain_ct(ev) <= 1:
        return "direct"
    return "real_pack"


def _plain_ct(ev: Event) -> int:
    """现场密文条数（执行峰值；一律 spine only，不加 join_ct）。"""
    return _spine_plain_ct(ev)


def _cost_ct(ev: Event) -> int:
    """
    DP 计价条数（关键路径乐观界）。

    只计主路径 CT。固有 DualRail：主路径条数；普通态：``max(1, n//2)``。
    ``cost_ct`` 覆盖值须已是主路径条数。
    """
    if ev.cost_ct is not None:
        return int(ev.cost_ct)
    n = _spine_plain_ct(ev)
    if ev.layout == "dualrail":
        return n
    return max(1, n // 2)


def _boot_land_rem(
    ev: Event, budget: int, *, pack_mode: str | None = None
) -> int:
    """Rem after bootstrap; ``real_pack`` pays ``dualrail_depth`` unpack tax."""
    mode = pack_mode if pack_mode is not None else pack_mode_for_event(ev)
    if mode == "real_pack":
        tax = int(ev.dualrail_depth) or DUALRAIL_UNPACK_DEPTH
        return max(0, budget - tax)
    return budget


def _boot_pay(ev: Event, pack_mode: str, budget: int) -> tuple[int, int]:
    """
    一次 bootstrap 的 (ct_count, rem_after)。

    ``ct_count`` 为关键路径乐观计费（复数打包后条数）。
    ``rem_after``：``direct``→budget；``real_pack``→budget−tax。
    """
    return _cost_ct(ev), _boot_land_rem(ev, budget, pack_mode=pack_mode)


def _boot_ct(ev: Event) -> int:
    return _cost_ct(ev)


def _mid_ct(ev: Event) -> int:
    return _cost_ct(ev)


def _consume_depth_with_style(
    remaining: int,
    depth: int,
    ev: Event,
    budget: int,
    slots: list[int],
    *,
    pack_mode: str | None = None,
) -> tuple[int, int, int]:
    """消耗 ``depth``（一律 atomic：不可中途 bts）。

    要求 ``remaining >= depth + LIBERATE_MIN_LEAVE_REM``；不够则返回
    ``add=10**9``（不可行，须由调用方选段首 before-boot）。
    结束后 ``rem_after = remaining - depth >= MIN_LEAVE``（depth>0 时）。
    """
    if depth < 0:
        raise ValueError("depth < 0")
    del pack_mode, slots, budget, ev  # mid-boot / pack 不再用于消耗
    leave = int(LIBERATE_MIN_LEAVE_REM)
    if depth == 0:
        return 0, remaining, 0
    # rem >= depth + MIN_LEAVE；不够则必须 before-boot（不可行）
    if remaining < depth + leave:
        return 10**9, remaining, 0
    return 0, remaining - depth, 0


def _on_boot_slots(slots: list[int], ev: Event, budget: int) -> None:
    """No-op：DP rem-only；旁路对齐在 HE 侧。"""
    del slots, ev, budget


def _consume_depth_pick(
    remaining: int,
    depth: int,
    ev: Event,
    budget: int,
    slots: list[int],
) -> tuple[int, int, int, str]:
    """消耗 depth（atomic）；返回的 pack_mode 仅作占位。"""
    if depth <= 0:
        return 0, remaining, 0, "direct"
    extra, r, mid = _consume_depth_with_style(
        remaining, depth, ev, budget, slots, pack_mode=pack_mode_for_event(ev)
    )
    return extra, r, mid, pack_mode_for_event(ev)


def _consume_depth_with_side_refresh(
    remaining: int,
    depth: int,
    ct_count: int,
    budget: int,
    slots: list[int],
    ev: Event,
) -> tuple[int, int, int]:
    """旧接口：按 ``ct_count`` 计费（无 DualRail 选择；无 slot refresh）。"""
    del slots, ev
    return _consume_depth(remaining, depth, ct_count, budget)


def _transition_event(
    rem: int,
    slots: tuple[int, ...] | list[int],
    ev: Event,
    budget: int,
    *,
    before_style: str | None,
) -> tuple[int, int, int, SlotVec, str]:
    """
    执行事件一步（rem-only；忽略 ``slots`` / join/save/clear）。

    ``before_style``：None 或任意非 None（仅表示段首是否 boot；落地 rem 由 Event 决定）。
    返回 (add, rem, mid_count, new_slots, mid_pack_mode)。
    """
    add = 0
    mid = 0
    mid_pm = pack_mode_for_event(ev)
    r = rem
    del slots  # N_SLOTS=0；签名保留

    if before_style is not None:
        if ev.companion_boot:
            add += 1
        else:
            ct, r = _boot_pay(ev, pack_mode_for_event(ev), budget)
            add += ct

    if ev.align_depth > 0:
        a, r, m, st = _consume_depth_pick(
            r, ev.align_depth, ev, budget, []
        )
        add += a
        mid += m
        mid_pm = st

    a, r, m, st = _consume_depth_pick(r, ev.depth, ev, budget, [])
    add += a
    mid += m
    if m > 0:
        mid_pm = st

    return add, r, mid, _slots_tuple([]), mid_pm


def _before_styles(ev: Event) -> tuple[str | None, ...]:
    """段首可选：不刷 / boot。落地 rem 由 ``pack_mode_for_event`` 决定。"""
    del ev
    return (None, "plain")


def _align_mid_placements_to_rem_trace(
    events: list[Event],
    placements: list[BootstrapPlacement],
    rem_trace: list[RemTraceRow],
    *,
    budget: int,
) -> tuple[list[BootstrapPlacement], int]:
    """
    Replay may need a mid boot at event E while DP placement sits on E' > E
    (e.g. L1 σ-aSOR: rem→0 at ``asor_sigma_i0``, placement on ``_i1``).
    Pull the earliest surplus mid placement forward; otherwise inject one.
    Returns (placements, extra_bts_ct).
    """
    pls = list(placements)
    placed_mid_ei = {
        int(p.event_index) for p in pls if p.reason == "mid_event"
    }
    missing: list[tuple[int, RemTraceRow]] = []
    for row in rem_trace:
        if row.kind != "after_boot" or row.boot_reason != "mid_event":
            continue
        ei = row.event_index
        if ei is None or int(ei) in placed_mid_ei:
            continue
        missing.append((int(ei), row))
    if not missing:
        return pls, 0
    extra = 0
    for ei, row in sorted(missing, key=lambda x: x[0]):
        if ei in placed_mid_ei:
            continue
        ev = events[ei]
        stolen: int | None = None
        ev_prefix = ev.name.rsplit("_i", 1)[0] if "_i" in ev.name else ev.name
        for j, p in enumerate(pls):
            if p.reason != "mid_event" or int(p.event_index) <= ei:
                continue
            if p.event_name.startswith(ev_prefix):
                stolen = j
                break
        if stolen is not None:
            p = pls[stolen]
            pm = pack_mode_for_event(ev)
            pls[stolen] = replace(
                p,
                event_index=ei,
                event_name=ev.name,
                remaining_before=int(row.rem_before),
                pack_mode=pm,
            )
            placed_mid_ei.add(ei)
            continue
        pm = row.boot_pack_mode or pack_mode_for_event(ev)
        ct, _ = _boot_pay(ev, pm, budget)
        pls.append(
            BootstrapPlacement(
                event_index=ei,
                event_name=ev.name,
                ct_count=ct,
                remaining_before=int(row.rem_before),
                reason="mid_event",
                pack_mode=pm,
            )
        )
        placed_mid_ei.add(ei)
        extra += int(ct)
    return pls, extra


def optimize_bootstrap_events(
    events: list[Event],
    *,
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
    initial_level: int | None = None,
) -> BootstrapResult:
    """
    路径 DP：状态 ``(rem,)``（rem-only；无旁路 slot）→ 最少关键路径 bts。

    旁路对齐（残差 / V / AUX）在 HE 侧 ``align_bypass_to_main`` /
    companion DualRail，不进 DP 状态。
    """
    if budget <= 0:
        raise ValueError(f"budget 须为正，当前 {budget}")
    if initial_level is None:
        initial_level = budget
    if initial_level < 0:
        raise ValueError(f"initial_level 不能为负：{initial_level}")

    n = len(events)
    if n == 0:
        return BootstrapResult(
            bts_count=0, events=events, budget=budget, initial_level=initial_level
        )

    inf = 10**18
    start_r = min(initial_level, budget)
    start_state: DpState = (start_r,) + (SLOT_UNSET,) * N_SLOTS
    cur: dict[DpState, int] = {start_state: 0}
    parents: list[
        dict[
            DpState,
            tuple[DpState, int, int, int, str],
        ]
    ] = [{} for _ in range(n + 1)]

    def _tag(style: str | None) -> int:
        if style is None:
            return 0
        if style == "plain":
            return 1
        if style == "dualrail":
            return 2
        raise ValueError(style)

    for i, ev in enumerate(events):
        nxt: dict[DpState, int] = {}
        for state, cost in cur.items():
            rem = int(state[0])
            slots = state[1:]
            for before in _before_styles(ev):
                add, r_after, mid, ns, mid_style = _transition_event(
                    rem, slots, ev, budget, before_style=before
                )
                if add >= 10**9:
                    continue
                nstate: DpState = (r_after,) + tuple(ns)
                ncost = cost + add
                prev = nxt.get(nstate, inf)
                if ncost < prev:
                    nxt[nstate] = ncost
                    parents[i + 1][nstate] = (
                        state,
                        _tag(before),
                        add,
                        mid,
                        mid_style,
                    )
        if not nxt:
            raise RuntimeError(f"optimize_bootstrap: 事件 {ev.name} 后无可达状态")
        cur = nxt

    total = inf
    end_state = None
    for state, cost in cur.items():
        if cost < total:
            total = cost
            end_state = state
    assert end_state is not None

    placements = _reconstruct_placements_forward(
        events, parents, end_state, budget=budget
    )
    placed_sum = sum(p.ct_count for p in placements)
    if placed_sum != total:
        raise RuntimeError(
            f"placement 回溯与 DP 不一致：placements={placed_sum} dp={total}"
        )

    discards = compute_level_discards(
        events, placements, budget=budget, initial_level=initial_level
    )
    rem_trace = simulate_remaining_trace(
        events,
        placements,
        budget=budget,
        initial_level=initial_level,
    )
    for _ in range(6):
        pls2, extra = _align_mid_placements_to_rem_trace(
            events, placements, rem_trace, budget=budget
        )
        if extra == 0 and len(pls2) == len(placements) and all(
            a.event_index == b.event_index
            and a.event_name == b.event_name
            and a.reason == b.reason
            for a, b in zip(pls2, placements)
        ):
            break
        placements = pls2
        total += int(extra)
        discards = compute_level_discards(
            events, placements, budget=budget, initial_level=initial_level
        )
        rem_trace = simulate_remaining_trace(
            events,
            placements,
            budget=budget,
            initial_level=initial_level,
        )
    return BootstrapResult(
        bts_count=int(total),
        placements=placements,
        events=list(events),
        budget=budget,
        initial_level=initial_level,
        level_discards=discards,
        rem_trace=rem_trace,
    )


def _reconstruct_placements_forward(
    events: list[Event],
    parents: list[
        dict[
            DpState,
            tuple[DpState, int, int, int, str],
        ]
    ],
    end_state: DpState,
    *,
    budget: int,
) -> list[BootstrapPlacement]:
    n = len(events)
    chain: list[tuple[int, DpState, int, int, int, str]] = []
    state = end_state
    for i in range(n, 0, -1):
        prev_state, tag, add, mid, mid_style = parents[i][state]
        chain.append((i - 1, prev_state, tag, add, mid, mid_style))
        state = prev_state
    chain.reverse()

    def _untag(tag: int) -> str | None:
        return (None, "plain", "dualrail")[tag]

    placements: list[BootstrapPlacement] = []
    for i, prev_state, tag, add, mid, mid_style in chain:
        ev = events[i]
        rem_before = prev_state[0]
        paid = 0
        before = _untag(tag)
        if before is not None:
            if ev.companion_boot:
                pm = "direct"
                bct = 1
            else:
                pm = pack_mode_for_event(ev)
                bct, _ = _boot_pay(ev, pm, budget)
            placements.append(
                BootstrapPlacement(
                    event_index=i,
                    event_name=ev.name,
                    ct_count=bct,
                    remaining_before=rem_before,
                    reason="before_event",
                    pack_mode=pm,
                )
            )
            paid += bct
        for _ in range(mid):
            pm = pack_mode_for_event(ev)
            mct, _ = _boot_pay(ev, pm, budget)
            placements.append(
                BootstrapPlacement(
                    event_index=i,
                    event_name=ev.name,
                    ct_count=mct,
                    remaining_before=0,
                    reason="mid_event",
                    pack_mode=pm,
                )
            )
            paid += mct
        if paid != add:
            raise RuntimeError(
                f"事件 {ev.name} placement 记账 {paid} != DP add {add}"
            )
    return placements


def optimize_bootstrap(
    task_name: str,
    scheme: list[int],
    *,
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
    initial_level: int | None = None,
    phase: str = "C",
    geom: SlotCtGeometry | None = None,
    depth_overrides: dict[str, int] | None = None,
) -> BootstrapResult:
    events = build_bootstrap_events(
        task_name,
        scheme,
        phase=phase,
        geom=geom,
        depth_overrides=depth_overrides,
    )
    return optimize_bootstrap_events(
        events, budget=budget, initial_level=initial_level
    )


def compute_scheme_bts(
    task_name: str,
    scheme: list[int],
    *,
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
    initial_level: int | None = None,
    phase: str = "C",
    geom: SlotCtGeometry | None = None,
) -> int:
    return optimize_bootstrap(
        task_name,
        scheme,
        budget=budget,
        initial_level=initial_level,
        phase=phase,
        geom=geom,
    ).bts_count


def scheme_all(level: int) -> list[int]:
    """构造合法 48 维常数方案；层 9/10 禁止 GeLU mid，自动改为 high。"""
    if level not in (0, 1, 2):
        raise ValueError(f"scheme_all 仅允许 0/1/2，收到 {level}")
    from gelu_poly import gelu_level_allowed

    out: list[int] = []
    for layer_idx in range(NUM_LAYERS):
        for kind in ("softmax", "ln1", "gelu", "ln2"):
            lv = level
            if kind == "gelu" and not gelu_level_allowed(layer_idx, lv):
                lv = 2  # mid 禁止时改用 high
            out.append(lv)
    return out


def audit_nonlinear_depth_splits(
    task_name: str,
    scheme: list[int],
    *,
    phase: str = "C",
) -> list[str]:
    """校验各非线性 slot 的 ``sum(event.depth)`` 与对应 depth 函数一致。

    phase C 的 LN 用 HE invsqrt 关键路径（每轮 2），不用 ``layernorm_poly_depth``
    占位（每轮 1）。
    """
    events = build_bootstrap_events(task_name, scheme, phase=phase)
    errors: list[str] = []
    ph = phase.upper()
    for layer_idx in range(NUM_LAYERS):
        for kind in ("softmax", "ln1", "gelu", "ln2"):
            level = _slot_level(scheme, layer_idx, kind)
            if ph == "C" and kind in ("ln1", "ln2"):
                cfg = layernorm_config_for_layer(
                    task_name, layer_idx, kind, level
                )
                iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
                expected = (
                    DEPTH_LN_SCALE
                    + DEPTH_LN_VAR
                    + DEPTH_LN_ENC_ONE
                    + iters * LN_INVSQRT_ITER_DEPTH
                    + DEPTH_LN_POST_INVSQRT
                )
            elif ph == "C" and kind == "softmax":
                iters_sigma, iters_sum_sq, _rounds, _ = _softmax_params(
                    task_name, layer_idx, level
                )
                expected = (
                    SOFTMAX_DEPTH_BASE
                    + iters_sigma * SOFTMAX_ASOR_ITER_DEPTH
                    + sum(iters_sum_sq) * SOFTMAX_ASOR_ITER_DEPTH
                )
            else:
                expected = _nonlinear_depth(task_name, layer_idx, kind, level)
            actual = sum(
                e.depth
                for e in events
                if e.layer_idx == layer_idx and e.kind == kind
            )
            if actual != expected:
                errors.append(
                    f"L{layer_idx}.{kind} level={level}: "
                    f"split={actual} expected={expected}"
                )
    return errors


def run_phase_d_audit(
    *,
    tasks: tuple[str, ...] | None = None,
    levels: tuple[int, ...] = (0, 1, 2),
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
    geom: SlotCtGeometry | None = None,
) -> list[str]:
    """
    阶段 D 自洽审计。返回错误列表（空 = 通过）。

    默认对 ``DEFAULT_AUDIT_TASKS``（全部已配置 Softmax/LN 的任务）× 三档检查。
    """
    if tasks is None:
        tasks = DEFAULT_AUDIT_TASKS
    if geom is not None:
        set_slot_geometry(geom)
    g = active_geometry()
    errors: list[str] = []
    empty_slots: SlotVec = (SLOT_UNSET,) * N_SLOTS  # () when N_SLOTS=0

    # Q∥K 原子岛：before 刷 Q+K（cost=8）；depth=DEPTH_QK_SCORE
    ev_qk = Event(
        name="ut.att_score",
        depth=DEPTH_QK_SCORE,
        ct_in=2 * g.ct_qkv_out,
        ct_out=g.ct_att_score,
        layout="dualrail",
        cost_ct=2 * g.ct_qkv_out,
        atomic=True,
    )
    add, r, mid, _ns, _st = _transition_event(
        8, empty_slots, ev_qk, budget, before_style="plain"
    )
    expect_r = budget - DEPTH_QK_SCORE
    if add != 2 * g.ct_qkv_out or mid != 0 or r != expect_r:
        errors.append(
            f"Q∥K 原子岛 before 失败: add={add} mid={mid} r={r} "
            f"(应 add={2 * g.ct_qkv_out} mid=0 r={expect_r})"
        )
    # 不刷也能跑完（入口 rem 够）
    add0, r0, mid0, _ns0, _ = _transition_event(
        8, empty_slots, ev_qk, budget, before_style=None
    )
    if add0 != 0 or mid0 != 0 or r0 != 8 - DEPTH_QK_SCORE:
        errors.append(
            f"Q∥K 原子岛无 before 失败: add={add0} mid={mid0} r={r0} "
            f"(应 add=0 mid=0 r={8 - DEPTH_QK_SCORE})"
        )

    # 普通态 8 → 计价 4（不再比较 DualRail pack 风格）
    ev_dr = Event(
        name="ut.dualrail",
        depth=0,
        ct_in=g.ct_att_score,
        ct_out=g.ct_att_score,
        plain_ct=g.ct_att_score,
        layout="real",
    )
    add_p, r_p, _, _, _ = _transition_event(
        10, empty_slots, ev_dr, budget, before_style="plain"
    )
    if add_p != g.ct_att_score // 2 or r_p != budget:
        errors.append(
            f"real-8 cost 失败: add={add_p} r={r_p} "
            f"(应 add={g.ct_att_score // 2}, r={budget})"
        )

    # sumsq/sigma enc_one：depth=0，段首 1 CT；不刷则 rem 不变
    ev_enc = Event(
        name="ut.sumsq_enc_one",
        depth=0,
        ct_in=1,
        ct_out=2,
        plain_ct=1,
        layout="dualrail",
        cost_ct=1,
    )
    add, r, mid, _ns, _st = _transition_event(
        5,
        empty_slots,
        ev_enc,
        budget,
        before_style=None,
    )
    if add != 0 or mid != 0 or r != 5:
        errors.append(
            f"sumsq enc_one 失败: add={add} mid={mid} r={r} (应 add=0 mid=0 r=5)"
        )

    for task in tasks:
        for level in levels:
            scheme = scheme_all(level)
            try:
                events = build_bootstrap_events(task, scheme, phase="C")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task} L{level} build: {exc}")
                continue

            # CT 链：条数或打包代价相等即可（an,bn → 单条 inv/denom）
            for i in range(len(events) - 1):
                a, b = events[i], events[i + 1]
                if a.layer_idx != b.layer_idx:
                    continue
                if a.ct_out != b.ct_in and _cost_ct(a) != _cost_ct(b):
                    errors.append(
                        f"{task} L{level} CT 断裂 @ {a.name}"
                    )
                    break

            # 非线性 depth
            errs = audit_nonlinear_depth_splits(task, scheme, phase="C")
            errors.extend(f"{task} {e}" for e in errs)

            # pathway：mask + DualRail exit + 每轮 Σy² apply
            for layer_idx in range(NUM_LAYERS):
                rounds = log2_ceil(LAYER_EXP_DIV_BY_TASK[task][layer_idx])
                expect_pw = (
                    DEPTH_SOFTMAX_ATTN_MASK
                    + DEPTH_SOFTMAX_EXIT
                    + DEPTH_SOFTMAX_EXP_HE_EXTRA
                    + rounds
                    * (DEPTH_SOFTMAX_SUMSQ_APPLY + DEPTH_SOFTMAX_INV_MASK)
                )
                pw = sum(
                    e.depth
                    for e in events
                    if e.layer_idx == layer_idx and e.kind == "pathway"
                )
                if pw != expect_pw:
                    errors.append(
                        f"{task} L{level} L{layer_idx} pathway depth={pw} "
                        f"(expected {expect_pw}, rounds={rounds})"
                    )
                # 真微事件仍在；bts 只作为相邻事件之间的 before 放置
                names_l = {
                    e.name.split(".", 1)[1]
                    for e in events
                    if e.layer_idx == layer_idx
                }
                for need in (
                    "softmax_exp_stockmeyer",
                    "softmax_exp_square_0",
                    "softmax_attn_mask",
                    "softmax_sigma_aggregate",
                    "softmax_sigma_enc_one",
                    "softmax_dualrail_exit",
                    "linear_att_context",
                    "bridge_rot_attn_dense",
                    "linear_attn_dense",
                    "ln1_scale",
                    "ln1_var",
                    "ln1_enc_one",
                    "ln2_scale",
                    "ln2_var",
                    "ln2_enc_one",
                    "gelu_f1",
                    "gelu_f2",
                    "gelu_reconstruct",
                ):
                    if need not in names_l:
                        errors.append(
                            f"{task} L{level} L{layer_idx} 缺微事件 {need}"
                        )
                n_sumsq_apply = sum(
                    1 for n in names_l if re.match(r"softmax_sumsq_r\d+_apply$", n)
                )
                if n_sumsq_apply != rounds:
                    errors.append(
                        f"{task} L{level} L{layer_idx} sumsq_apply={n_sumsq_apply} "
                        f"(expected {rounds})"
                    )
                n_enc_one = sum(
                    1 for n in names_l if re.match(r"softmax_sumsq_r\d+_enc_one$", n)
                )
                if n_enc_one != rounds:
                    errors.append(
                        f"{task} L{level} L{layer_idx} sumsq_enc_one={n_enc_one} "
                        f"(expected {rounds})"
                    )
                n_asor0 = sum(
                    1
                    for n in names_l
                    if n.startswith("softmax_norm_r") and n.endswith("_asor0")
                )
                if n_asor0 != rounds:
                    errors.append(
                        f"{task} L{level} L{layer_idx} Σy² asor0={n_asor0} "
                        f"(expected {rounds})"
                    )

            # 每层关键线性岛（残差会合不进图）
            for layer_idx in range(NUM_LAYERS):
                names = {
                    e.name.split(".", 1)[1]
                    for e in events
                    if e.layer_idx == layer_idx
                }
                for need in (
                    "linear_qkv",
                    "linear_att_score",
                    "linear_att_context",
                    "linear_attn_dense",
                    "linear_ff_dense1",
                    "linear_ff_dense2",
                ):
                    if need not in names:
                        errors.append(
                            f"{task} L{level} L{layer_idx} 缺事件 {need}"
                        )

            # DP + placement 一致（已在 optimize 内断言）
            try:
                result = optimize_bootstrap_events(
                    events, budget=budget
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task} L{level} DP: {exc}")
                continue

            if result.bts_count < 0:
                errors.append(f"{task} L{level} 负 bts")

    # 单调性：同任务 all_low ≤ all_mid ≤ all_high（phase C）
    for task in tasks:
        bts = [
            compute_scheme_bts(task, scheme_all(lv), budget=budget)
            for lv in (0, 1, 2)
        ]
        if not (bts[0] <= bts[1] <= bts[2]):
            errors.append(
                f"{task} 单调性失败: low/mid/high = {bts}"
            )

    return errors


def print_bootstrap_result(result: BootstrapResult, *, max_placements: int = 40) -> None:
    print(result.summary())
    depth_sum = sum(e.depth for e in result.events)
    print(f"  event depth sum = {depth_sum}")
    print(f"  {'idx':>4} {'depth':>5} {'ct_in':>5} {'peak':>5} {'ct_out':>6}  name")
    print("  " + "-" * 62)
    for i, ev in enumerate(result.events):
        peak = ev.ct_peak if ev.ct_peak is not None else "-"
        print(
            f"  {i:4d} {ev.depth:5d} {ev.ct_in:5d} {str(peak):>5} {ev.ct_out:6d}  {ev.name}"
        )
    print(f"  placements ({len(result.placements)}, show ≤{max_placements}):")
    for p in result.placements[:max_placements]:
        print(
            f"    [{p.event_index}] {p.event_name}  "
            f"ct={p.ct_count}  rem_before={p.remaining_before}  "
            f"{p.pack_mode}  {p.reason}"
        )
    if len(result.placements) > max_placements:
        print(f"    ... ({len(result.placements) - max_placements} more)")


if __name__ == "__main__":
    print("bts_ops 阶段 D 自洽审计 + C smoke（全任务 × 三档）")
    print("=" * 56)
    g = active_geometry()
    print(
        f"CT 几何 (bert_base DualRail): "
        f"Hc={g.ct_hidden_cplx} Hr={g.ct_hidden_real} "
        f"V/C={g.ct_v_cplx}/{g.ct_context_cplx} "
        f"qkv={g.ct_qkv_out} score={g.ct_att_score} "
        f"q_copies={g.ct_q_copies} alpha={g.ct_alpha_copies} "
        f"pc_rot={g.ct_pc_rot} rot_ctx={g.ct_pc_rot_ctx} ffn={g.ct_ffn}"
    )
    print(f"审计任务: {DEFAULT_AUDIT_TASKS}")
    d_errs = run_phase_d_audit()
    if d_errs:
        print(f"阶段 D FAILED ({len(d_errs)}):")
        for e in d_errs[:40]:
            print(" ", e)
        raise SystemExit(1)
    print("阶段 D: 全部检查通过")

    # 非线性深度拆解样例（任务×档可变）
    print("\n--- Softmax/GeLU/LN depth 拆解（mrpc L0）---")
    for level, name in enumerate(("low", "mid", "high")):
        it_s, it_sq, rounds, tot = _softmax_params("mrpc", 0, level)
        print(
            f"  softmax[{name}]: base={SOFTMAX_DEPTH_BASE} "
            f"+ σ×{SOFTMAX_ASOR_ITER_DEPTH}×{it_s} "
            f"+ Σy²{it_sq} → total={tot}"
        )
        gd = gelu_poly_depth(0, level)
        print(f"  gelu[{name}]: depth_he={gd}")
        for kind in ("ln1", "ln2"):
            ld = layernorm_poly_depth("mrpc", 0, kind, level)
            print(f"  {kind}[{name}]: {ld} (poly iters+{LAYERNORM_DEPTH_OVERHEAD}; "
                  f"HE C prep={DEPTH_LN_PREP}/cold={DEPTH_LN_PREP_COLD})")

    print("\n--- scheme bts (mrpc) ---")
    for label, level in [("all_low", 0), ("all_mid", 1), ("all_high", 2)]:
        r = optimize_bootstrap("mrpc", scheme_all(level), phase="C")
        print(f"[{label}] {r.summary()}")
        join_pl = [
            p
            for p in r.placements
            if "att_context" in p.event_name
            or "att_score" in p.event_name
        ]
        if join_pl:
            print(f"  join 相关放置: {len(join_pl)}")
            for p in join_pl[:4]:
                print(f"    {p.reason} @ {p.event_name} ct={p.ct_count}")

    print(
        f"\n--- L0 placements (mrpc, initial_level={DEFAULT_L0_ENTRY_REMAINING}) ---"
    )
    for label, level in [("all_mid", 1), ("all_high", 2)]:
        r = optimize_bootstrap(
            "mrpc",
            scheme_all(level),
            phase="C",
            initial_level=DEFAULT_L0_ENTRY_REMAINING,
        )
        l0 = [p for p in r.placements if p.event_name.startswith("L0.")]
        print(
            f"[{label}] billed={r.bts_count} sites={len(r.placements)} L0={len(l0)}"
        )
        for p in l0:
            print(
                f"    {p.event_name}  ct={p.ct_count}  "
                f"rem_before={p.remaining_before}  {p.reason}"
            )
