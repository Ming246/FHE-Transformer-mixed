"""
Bootstrapping 最优放置（实验模块，阶段 C 为主；A/B 为粗粒度对照）。

阶段 C（默认）：
  - Softmax / GeLU / LN 微事件；函数内部可 bts
  - join 按两路实际 remaining：rem ← min(spine, side)
  - ct×ct join 深度不足时对两侧 bootstrap（join_ct），可选刷新旁路槽
  - 旁路 fork：残差、V；Softmax 通路 mask/copy 与多项式分离

阶段 D：``run_phase_d_audit`` 自洽性检查（深度守恒、CT 链、join/DP）。

不修改 cost.py；非线性 kind 的 depth 之和仍 = cost.*_poly_depth。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cost import (
    BOOTSTRAP_DEPTH_BUDGET,
    NUM_LAYERS,
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
from gelu_poly import gelu_config_for_layer
from layernorm_poly import count_he_invsqrt_iters, layernorm_config_for_layer
from nolinear.gelu_chebyshev import cheb_he_depth_ps_tree, gelu_he_depth_breakdown
from softmax_poly import (
    LAYER_ASOR_MAX_ITERS_SIGMA_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ_BY_TASK,
    LAYER_ASOR_MAX_ITERS_SUM_SQ2_BY_TASK,
    LAYER_EXP_DIV_BY_TASK,
    SOFTMAX_LEVEL_KEYS,
)

# ---------------------------------------------------------------------------
# 实数路径 CT（占位：固定 BERT-base / 对照 THOR 工程布局）
#
# TODO(linear-ct)：全部线性段的密文数量与深度表可能不准确。
#   - 当前 CT_HIDDEN=8 来自 THOR 对角 matmul 友好打包（槽位利用率低），
#     并非 128×768÷32768 的容量下界（稠密打包约 3～4 CT）。
#   - 后续：按 THOR 已发表论文的思想做「本土化」线性层模型——
#     打包/旋转/副本规则可配置，支持 hidden/ffn/seq/heads 等规模自由变换，
#     而非写死 base；再据此重标定 CT_* 与 DEPTH_*。非线性 bts DP 可复用。
# ---------------------------------------------------------------------------

CT_HIDDEN = 8
CT_ROTATED = 128  # hidden×16 或 Q×16 / softmax 出口复制
CT_ATT_SCORE = 8
CT_FFN = 16
# Softmax Stockmeyer 段活密文峰值启发式：每路 ~8 个中间项 × 8 score CT
CT_SOFTMAX_STOCKMEYER_PEAK = CT_ATT_SCORE * 8

# ---------------------------------------------------------------------------
# 线性深度（THOR 注释占位；与上 TODO 一并重标定）
# ---------------------------------------------------------------------------

DEPTH_QKV = 2
DEPTH_TRANSPOSE_K = 1
DEPTH_MAKE_COPIES = 1
DEPTH_ATT_SCORE = 1
DEPTH_ATT_CONTEXT = 1
DEPTH_ATT_DENSE = 3
DEPTH_FF_DENSE = 2

# Softmax 通路（非多项式；对照 THOR pt_ct_mult / make_copies）
DEPTH_SOFTMAX_ATTN_MASK = 1  # exp 后 attention_mask：pt×ct
DEPTH_SOFTMAX_COPY = 1  # 8→128：mask+rotsum，与 make_copies 同为 1 层

# Goldschmidt 单次迭代 HE 乘法深度（与 cost 中 ×2 一致）
GOLDSCHMIDT_ITER_DEPTH = 2

# Softmax 复制：8→16→32→64→128（CT 分阶；总乘法深度 = DEPTH_SOFTMAX_COPY）
SOFTMAX_COPY_TARGETS = (16, 32, 64, CT_ROTATED)

# LN prep 拆成 3 步（与 overhead=3 一致）
LAYERNORM_PREP_STEPS = 3

# 旁路 remaining 槽（DP 状态；spine 上 bootstrap 不刷新旁路）
SLOT_RESID = 0  # 残差输入（attn / FF 复用同一槽）
SLOT_V = 1  # Softmax 期间停泊的 V
N_SLOTS = 2
SLOT_UNSET = -1


@dataclass(frozen=True)
class Event:
    """计算图上的一步；``ct_peak`` 为段内副本峰值（中途 bts 计费用）。"""

    name: str
    depth: int
    ct_in: int
    ct_out: int
    layer_idx: int = -1
    kind: str = ""  # linear | softmax | ln1 | gelu | ln2 | bridge | pathway
    join: bool = False
    ct_peak: int | None = None
    align_depth: int = 0  # 旧固定对齐（已弃用）；slot join 用实际 remaining
    save_slot: int | None = None  # depth 消耗后把 spine rem 写入旁路槽
    join_slot: int | None = None  # depth 前：rem ← min(rem, side)
    join_offset: int = 0  # side = max(0, slots[join_slot] - join_offset)
    join_ct: int = 0  # 旁路密文数；ct×ct/残差 join 时 bts 计 spine+side
    refresh_side: bool = False  # bootstrap 时是否把旁路槽刷成 budget（V/残差）

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
        if self.save_slot is not None and not (0 <= self.save_slot < N_SLOTS):
            raise ValueError(f"save_slot 非法：{self.save_slot}")
        if self.join_slot is not None and not (0 <= self.join_slot < N_SLOTS):
            raise ValueError(f"join_slot 非法：{self.join_slot}")
        if self.join_offset < 0:
            raise ValueError(f"join_offset 不能为负：{self.name}")
        if self.join_ct < 0:
            raise ValueError(f"join_ct 不能为负：{self.name}")


@dataclass
class BootstrapPlacement:
    event_index: int
    event_name: str
    ct_count: int
    remaining_before: int
    reason: str  # "before_event" | "mid_event"


@dataclass
class BootstrapResult:
    bts_count: int
    placements: list[BootstrapPlacement] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    budget: int = BOOTSTRAP_DEPTH_BUDGET
    initial_level: int = BOOTSTRAP_DEPTH_BUDGET

    def summary(self) -> str:
        return (
            f"bts={self.bts_count} "
            f"(events={len(self.events)}, budget={self.budget}, "
            f"placements={len(self.placements)})"
        )


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


def _nonlinear_depth(
    task_name: str, layer_idx: int, kind: str, level: int
) -> int:
    if kind == "softmax":
        return softmax_poly_depth(task_name, layer_idx, level)
    if kind == "gelu":
        return gelu_poly_depth(layer_idx, level)
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
    save_slot: int | None = None,
    join_slot: int | None = None,
    join_offset: int = 0,
    join_ct: int = 0,
    refresh_side: bool = False,
) -> Event:
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
        save_slot=save_slot,
        join_slot=join_slot,
        join_offset=join_offset,
        join_ct=join_ct,
        refresh_side=refresh_side,
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
        + iters_sigma * 2
        + sum(iters_sum_sq) * 2
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


def _append_softmax_copy_chain(
    events: list[Event],
    prefix: str,
    layer_idx: int,
) -> None:
    """
    8→128 复制通路：先 pt×ct mask（depth=DEPTH_SOFTMAX_COPY），再分阶膨胀（depth=0）。
    对照 THOR ``make_copies`` / softmax 出口 rotsum；多项式深度不含此段。
    """
    events.append(
        _ev(
            prefix,
            "softmax_copy_mask",
            DEPTH_SOFTMAX_COPY,
            CT_ATT_SCORE,
            CT_ATT_SCORE,
            layer_idx,
            "pathway",
        )
    )
    ct = CT_ATT_SCORE
    for target in SOFTMAX_COPY_TARGETS:
        events.append(
            _ev(
                prefix,
                f"softmax_copy_to_{target}",
                0,
                ct,
                target,
                layer_idx,
                "pathway",
                ct_peak=target,
            )
        )
        ct = target


def _append_softmax_exp_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
) -> None:
    """
    thor exp：Stockmeyer(deg15, depth=4) + δ1 平方(depth=1)；之和 = SOFTMAX_DEPTH_BASE。
    """
    events.append(
        _ev(
            prefix,
            "softmax_exp_stockmeyer",
            SOFTMAX_EXP_STOCKMEYER_DEPTH,
            CT_ATT_SCORE,
            CT_ATT_SCORE,
            layer_idx,
            "softmax",
            ct_peak=CT_SOFTMAX_STOCKMEYER_PEAK,
        )
    )
    events.append(
        _ev(
            prefix,
            "softmax_exp_square_0",
            SOFTMAX_EXP_SQUARE_DEPTH,
            CT_ATT_SCORE,
            CT_ATT_SCORE,
            layer_idx,
            "softmax",
        )
    )


def _append_softmax_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    level: int,
) -> None:
    """阶段 B：exp(Stockmeyer+square) → aSOR_σ → δ2 归一化轮 → 复制膨胀（8→128）。"""
    iters_sigma, iters_sum_sq, rounds, _ = _softmax_params(
        task_name, layer_idx, level
    )
    _append_softmax_exp_events(events, prefix, layer_idx)
    events.append(
        _ev(
            prefix,
            "softmax_asor_sigma",
            iters_sigma * 2,
            CT_ATT_SCORE,
            CT_ATT_SCORE,
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
                iters_sum_sq[r] * 2,
                CT_ATT_SCORE,
                CT_ATT_SCORE,
                layer_idx,
                "softmax",
                join=True,
            )
        )
    events.append(
        _ev(
            prefix,
            "softmax_copy_expand",
            0,
            CT_ATT_SCORE,
            CT_ROTATED,
            layer_idx,
            "softmax",
        )
    )


def _append_softmax_events_c(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    level: int,
) -> None:
    """阶段 C：exp(Stockmeyer+square) + attn mask + 逐次 aSOR + 分阶复制链。"""
    iters_sigma, iters_sum_sq, rounds, _ = _softmax_params(
        task_name, layer_idx, level
    )
    _append_softmax_exp_events(events, prefix, layer_idx)
    events.append(
        _ev(
            prefix,
            "softmax_attn_mask",
            DEPTH_SOFTMAX_ATTN_MASK,
            CT_ATT_SCORE,
            CT_ATT_SCORE,
            layer_idx,
            "pathway",
        )
    )
    for i in range(iters_sigma):
        events.append(
            _ev(
                prefix,
                f"softmax_asor_sigma_i{i}",
                GOLDSCHMIDT_ITER_DEPTH,
                CT_ATT_SCORE,
                CT_ATT_SCORE,
                layer_idx,
                "softmax",
            )
        )
    for r in range(rounds):
        for g in range(iters_sum_sq[r]):
            events.append(
                _ev(
                    prefix,
                    f"softmax_norm_r{r}_asor{g}",
                    GOLDSCHMIDT_ITER_DEPTH,
                    CT_ATT_SCORE,
                    CT_ATT_SCORE,
                    layer_idx,
                    "softmax",
                )
            )
    _append_softmax_copy_chain(events, prefix, layer_idx)


def _append_gelu_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    level: int,
) -> None:
    """阶段 B：Cheb 复合 f1→f2→×x（深度来自 gelu_he_depth_breakdown）。"""
    cfg = gelu_config_for_layer(layer_idx, level)
    if cfg["kind"] == "composite":
        bd = gelu_he_depth_breakdown(cfg["d1"], cfg["d2"], eval_method="ps_tree")
        f1_d, f2_d = int(bd["f1_eval"]), int(bd["f2_eval"])
        recon_d = int(bd["gelu_reconstruct"])
    else:
        bd = gelu_he_depth_breakdown(cfg["degree"], eval_method="ps_tree")
        f1_d, f2_d = int(bd["y_eval"]), 0
        recon_d = int(bd["gelu_reconstruct"])

    peak = CT_FFN * 2  # 阶段 B 启发式
    if f1_d > 0:
        events.append(
            _ev(
                prefix,
                "gelu_f1",
                f1_d,
                CT_FFN,
                CT_FFN,
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
                CT_FFN,
                CT_FFN,
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
            CT_FFN,
            CT_FFN,
            layer_idx,
            "gelu",
        )
    )
    expected = gelu_poly_depth(layer_idx, level)
    actual = sum(
        e.depth for e in events if e.layer_idx == layer_idx and e.kind == "gelu"
    )
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
    """阶段 C：Cheb PS-tree 逐层 + combine + 还原。"""
    cfg = gelu_config_for_layer(layer_idx, level)
    if cfg["kind"] == "composite":
        _append_cheb_ps_tree_events(
            events,
            prefix,
            "gelu_f1",
            int(cfg["d1"]),
            cfg["f1_cheb_coeffs"],
            CT_FFN,
            layer_idx,
            "gelu",
        )
        _append_cheb_ps_tree_events(
            events,
            prefix,
            "gelu_f2",
            int(cfg["d2"]),
            cfg["f2_cheb_coeffs"],
            CT_FFN,
            layer_idx,
            "gelu",
        )
    else:
        _append_cheb_ps_tree_events(
            events,
            prefix,
            "gelu_y",
            int(cfg["degree"]),
            cfg["f_cheb_coeffs"],
            CT_FFN,
            layer_idx,
            "gelu",
        )
    events.append(
        _ev(
            prefix,
            "gelu_reconstruct",
            1,
            CT_FFN,
            CT_FFN,
            layer_idx,
            "gelu",
        )
    )
    expected = gelu_poly_depth(layer_idx, level)
    actual = sum(
        e.depth for e in events if e.layer_idx == layer_idx and e.kind == "gelu"
    )
    if actual != expected:
        raise RuntimeError(
            f"GeLU depth 拆分不一致 L{layer_idx} level={level}: "
            f"split={actual} expected={expected}"
        )


def _append_layernorm_events(
    events: list[Event],
    prefix: str,
    layer_idx: int,
    task_name: str,
    kind: str,
    level: int,
) -> None:
    """阶段 B：统计 prep（depth=3）+ 逐次 invsqrt（每 iter depth=1）。"""
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
    slot = kind  # ln1 | ln2
    events.append(
        _ev(
            prefix,
            f"{slot}_prep",
            3,
            CT_HIDDEN,
            CT_HIDDEN,
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
                CT_HIDDEN,
                CT_HIDDEN,
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
    """阶段 C：prep 三步 + invsqrt 逐步（join 对齐接入）。"""
    cfg = layernorm_config_for_layer(task_name, layer_idx, kind, level)
    iters = count_he_invsqrt_iters(cfg["invsqrt_max_iters"])
    slot = kind
    for s in range(LAYERNORM_PREP_STEPS):
        events.append(
            _ev(
                prefix,
                f"{slot}_prep_s{s}",
                1,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                kind,
                ct_peak=CT_HIDDEN * (s + 1),
            )
        )
    for i in range(iters):
        events.append(
            _ev(
                prefix,
                f"{slot}_invsqrt_{i}",
                1,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                kind,
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


def build_bootstrap_events(
    task_name: str,
    scheme: list[int],
    *,
    phase: str = "C",
) -> list[Event]:
    """
    展开事件序列。
    phase A：非线性整段粗事件；B：阶段 B 微展开；C（默认）：微事件 + 通路 depth。
    各 phase 均含残差 / V 旁路 fork–join（按实际 remaining 对齐）。
    """
    _validate_scheme_poly_only(scheme, task_name)
    events: list[Event] = []
    ph = phase.upper()
    if ph not in ("A", "B", "C"):
        raise ValueError(f"phase 须为 A/B/C，收到 {phase!r}")

    for layer_idx in range(NUM_LAYERS):
        p = f"L{layer_idx}"

        # 残差输入：Attention 块入口停泊（THOR: add(x, dense_out)）
        events.append(
            _ev(
                p,
                "fork_resid_attn",
                0,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                "bridge",
                save_slot=SLOT_RESID,
            )
        )
        events.append(
            _ev(p, "bridge_rot_qkv", 0, CT_HIDDEN, CT_ROTATED, layer_idx, "bridge")
        )
        events.append(
            _ev(
                p,
                "linear_qkv",
                DEPTH_QKV,
                CT_ROTATED,
                CT_HIDDEN,
                layer_idx,
                "linear",
                save_slot=SLOT_V,  # V 停泊；K 的 transpose 深度用 join_offset 表示
            )
        )
        # K transpose 不走 spine 深度（与 Q 并行）；仅保留 CT 注释事件
        events.append(
            _ev(
                p,
                "linear_transpose_k",
                0,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                "linear",
            )
        )
        events.append(
            _ev(
                p,
                "linear_make_copies",
                DEPTH_MAKE_COPIES,
                CT_HIDDEN,
                CT_ROTATED,
                layer_idx,
                "linear",
            )
        )
        # Q*K：ct×ct 前对齐；side = QKV_rem - transpose（= K）；不刷 V 槽（K≠V）
        events.append(
            _ev(
                p,
                "linear_att_score",
                DEPTH_ATT_SCORE,
                CT_ROTATED,
                CT_ATT_SCORE,
                layer_idx,
                "linear",
                join=True,
                join_slot=SLOT_V,
                join_offset=DEPTH_TRANSPOSE_K,
                join_ct=CT_HIDDEN,
                refresh_side=False,
            )
        )

        sm_level = _slot_level(scheme, layer_idx, "softmax")
        if ph == "A":
            sm_depth = _nonlinear_depth(task_name, layer_idx, "softmax", sm_level)
            events.append(
                _ev(
                    p,
                    "softmax",
                    sm_depth,
                    CT_ATT_SCORE,
                    CT_ROTATED,
                    layer_idx,
                    "softmax",
                )
            )
        elif ph == "B":
            _append_softmax_events(events, p, layer_idx, task_name, sm_level)
        else:
            _append_softmax_events_c(events, p, layer_idx, task_name, sm_level)

        # Softmax * V：ct×ct；深度不足时两侧 bts 并刷新 V 槽
        events.append(
            _ev(
                p,
                "linear_att_context",
                DEPTH_ATT_CONTEXT,
                CT_ROTATED,
                CT_HIDDEN,
                layer_idx,
                "linear",
                join=True,
                join_slot=SLOT_V,
                join_offset=0,
                join_ct=CT_HIDDEN,
                refresh_side=True,
            )
        )
        events.append(
            _ev(p, "bridge_rot_attn_dense", 0, CT_HIDDEN, CT_ROTATED, layer_idx, "bridge")
        )
        events.append(
            _ev(
                p,
                "linear_attn_dense",
                DEPTH_ATT_DENSE,
                CT_ROTATED,
                CT_HIDDEN,
                layer_idx,
                "linear",
            )
        )
        events.append(
            _ev(
                p,
                "join_resid_attn",
                0,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                "bridge",
                join=True,
                join_slot=SLOT_RESID,
                join_ct=CT_HIDDEN,
                refresh_side=True,
            )
        )

        ln1_level = _slot_level(scheme, layer_idx, "ln1")
        if ph == "A":
            ln1_depth = _nonlinear_depth(task_name, layer_idx, "ln1", ln1_level)
            events.append(
                _ev(p, "ln1", ln1_depth, CT_HIDDEN, CT_HIDDEN, layer_idx, "ln1")
            )
        elif ph == "B":
            _append_layernorm_events(
                events, p, layer_idx, task_name, "ln1", ln1_level
            )
        else:
            _append_layernorm_events_c(
                events, p, layer_idx, task_name, "ln1", ln1_level
            )

        # FF 残差入口
        events.append(
            _ev(
                p,
                "fork_resid_ff",
                0,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                "bridge",
                save_slot=SLOT_RESID,
            )
        )
        events.append(
            _ev(p, "bridge_rot_ff1", 0, CT_HIDDEN, CT_ROTATED, layer_idx, "bridge")
        )
        events.append(
            _ev(
                p, "linear_ff_dense1", DEPTH_FF_DENSE, CT_ROTATED, CT_FFN, layer_idx, "linear"
            )
        )

        gelu_level = _slot_level(scheme, layer_idx, "gelu")
        if ph == "A":
            gelu_depth = _nonlinear_depth(task_name, layer_idx, "gelu", gelu_level)
            events.append(
                _ev(p, "gelu", gelu_depth, CT_FFN, CT_FFN, layer_idx, "gelu")
            )
        elif ph == "B":
            _append_gelu_events(events, p, layer_idx, gelu_level)
        else:
            _append_gelu_events_c(events, p, layer_idx, gelu_level)

        events.append(
            _ev(p, "bridge_rot_ff2", 0, CT_FFN, CT_ROTATED, layer_idx, "bridge")
        )
        events.append(
            _ev(
                p,
                "linear_ff_dense2",
                DEPTH_FF_DENSE,
                CT_ROTATED,
                CT_HIDDEN,
                layer_idx,
                "linear",
            )
        )
        events.append(
            _ev(
                p,
                "join_resid_ff",
                0,
                CT_HIDDEN,
                CT_HIDDEN,
                layer_idx,
                "bridge",
                join=True,
                join_slot=SLOT_RESID,
                join_ct=CT_HIDDEN,
                refresh_side=True,
            )
        )

        ln2_level = _slot_level(scheme, layer_idx, "ln2")
        if ph == "A":
            ln2_depth = _nonlinear_depth(task_name, layer_idx, "ln2", ln2_level)
            events.append(
                _ev(p, "ln2", ln2_depth, CT_HIDDEN, CT_HIDDEN, layer_idx, "ln2")
            )
        elif ph == "B":
            _append_layernorm_events(
                events, p, layer_idx, task_name, "ln2", ln2_level
            )
        else:
            _append_layernorm_events_c(
                events, p, layer_idx, task_name, "ln2", ln2_level
            )

    _validate_event_chain(events)
    return events


def _validate_event_chain(events: list[Event]) -> None:
    if not events:
        return
    for i in range(len(events) - 1):
        if events[i].ct_out != events[i + 1].ct_in:
            raise ValueError(
                f"CT 衔接断裂：{events[i].name} ct_out={events[i].ct_out} → "
                f"{events[i + 1].name} ct_in={events[i + 1].ct_in}"
            )


def _consume_depth(
    remaining: int,
    depth: int,
    ct_count: int,
    budget: int,
) -> tuple[int, int, int]:
    """
    在剩余深度 ``remaining`` 下消耗 ``depth``；若单段装不下则中途 bootstrap。

    例：budget=15, depth=33, ct=8, remaining=15
      1) 用掉 15，还剩 depth 18 → bootstrap×8，remaining=15
      2) 用掉 15，还剩 depth 3  → bootstrap×8，remaining=15
      3) 用掉 3，remaining=12
      合计 mid=2 次 bootstrap，bts 贡献 16（按 CT 个数）。

    阶段 B：中途 bootstrap 按 ``Event.bts_ct``（含 ``ct_peak``）计费。
    段首强制 bootstrap（``before_event``）仍按 ``ct_in``。

    返回 (bts 贡献, 消耗后剩余深度, 中途 bootstrap 次数)。
    """
    if depth < 0:
        raise ValueError("depth < 0")
    if budget <= 0:
        raise ValueError("budget 须为正")

    extra = 0
    mid = 0
    r = remaining
    d = depth

    if d == 0:
        return 0, r, 0

    if r <= 0:
        extra += ct_count
        mid += 1
        r = budget

    while d > r:
        d -= r
        extra += ct_count
        mid += 1
        r = budget

    r -= d
    return extra, r, mid


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


def _side_remaining(slots: tuple[int, int], ev: Event) -> int:
    assert ev.join_slot is not None
    saved = slots[ev.join_slot]
    if saved == SLOT_UNSET:
        raise RuntimeError(f"{ev.name}: join_slot={ev.join_slot} 尚未 fork save")
    return max(0, saved - ev.join_offset)


def _boot_ct(ev: Event) -> int:
    """join 事件 bootstrap 计 spine + 旁路 CT；否则 spine ct_in。"""
    if ev.join_slot is not None and ev.join_ct > 0:
        return ev.ct_in + ev.join_ct
    return ev.ct_in


def _mid_ct(ev: Event) -> int:
    """中途 bts：join 两侧；否则峰值/ct_in。"""
    if ev.join_slot is not None and ev.join_ct > 0:
        return ev.ct_in + ev.join_ct
    return ev.bts_ct


def _consume_depth_with_side_refresh(
    remaining: int,
    depth: int,
    ct_count: int,
    budget: int,
    slots: list[int],
    ev: Event,
) -> tuple[int, int, int]:
    """
    同 ``_consume_depth``；若 join 且 ``refresh_side``，每次中途 bts 后旁路槽=budget。
    """
    if depth < 0:
        raise ValueError("depth < 0")
    extra = 0
    mid = 0
    r = remaining
    d = depth
    if d == 0:
        return 0, r, 0
    if r <= 0:
        extra += ct_count
        mid += 1
        r = budget
        if ev.refresh_side and ev.join_slot is not None:
            slots[ev.join_slot] = budget
    while d > r:
        d -= r
        extra += ct_count
        mid += 1
        r = budget
        if ev.refresh_side and ev.join_slot is not None:
            slots[ev.join_slot] = budget
    r -= d
    return extra, r, mid


def _transition_event(
    rem: int,
    slots: tuple[int, int],
    ev: Event,
    budget: int,
    *,
    bootstrap_first: bool,
) -> tuple[int, int, int, tuple[int, int]]:
    """
    执行事件一步。

    join_slot：rem ← min(rem, side)。
    ct×ct / 残差 join：bootstrap 按 spine+join_ct；``refresh_side`` 时刷新旁路槽。
    普通 spine bootstrap 不刷新旁路。
    """
    add = 0
    mid = 0
    r = rem
    slot_list = list(slots)

    if bootstrap_first:
        add += _boot_ct(ev)
        r = budget
        if ev.refresh_side and ev.join_slot is not None:
            slot_list[ev.join_slot] = budget

    if ev.join_slot is not None:
        r = min(r, _side_remaining((slot_list[0], slot_list[1]), ev))

    if ev.align_depth > 0:
        a, r, m = _consume_depth_with_side_refresh(
            r, ev.align_depth, _mid_ct(ev), budget, slot_list, ev
        )
        add += a
        mid += m

    a, r, m = _consume_depth_with_side_refresh(
        r, ev.depth, _mid_ct(ev), budget, slot_list, ev
    )
    add += a
    mid += m

    if ev.save_slot is not None:
        slot_list[ev.save_slot] = r
    return add, r, mid, (slot_list[0], slot_list[1])


def optimize_bootstrap_events(
    events: list[Event],
    *,
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
    initial_level: int | None = None,
) -> BootstrapResult:
    """
    路径 DP：状态 (event_index, rem, slot_resid, slot_v) → 最少 bts。

    前向只扩展可达状态。每步可选：直接执行 / 先对 ct_in bootstrap。
    旁路 remaining 在 fork 时保存；spine bootstrap 不刷新旁路。
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
    # cur[(rem, s0, s1)] = min bts to reach this event-index with state
    cur: dict[tuple[int, int, int], int] = {(start_r, SLOT_UNSET, SLOT_UNSET): 0}
    # parent[i+1][new_state] = (old_state, action, add, mid)
    parents: list[dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int, int]]] = [
        {} for _ in range(n + 1)
    ]

    for i, ev in enumerate(events):
        nxt: dict[tuple[int, int, int], int] = {}
        for state, cost in cur.items():
            rem, s0, s1 = state
            slots = (s0, s1)
            for action in (0, 1):
                add, r_after, mid, ns = _transition_event(
                    rem, slots, ev, budget, bootstrap_first=(action == 1)
                )
                nstate = (r_after, ns[0], ns[1])
                ncost = cost + add
                prev = nxt.get(nstate, inf)
                if ncost < prev:
                    nxt[nstate] = ncost
                    parents[i + 1][nstate] = (state, action, add, mid)
        if not nxt:
            raise RuntimeError(f"optimize_bootstrap: 事件 {ev.name} 后无可达状态")
        cur = nxt

    # 终点取最少 bts（任意剩余深度 / 旁路）
    total = inf
    end_state = None
    for state, cost in cur.items():
        if cost < total:
            total = cost
            end_state = state
    assert end_state is not None

    placements = _reconstruct_placements_forward(events, parents, end_state)
    placed_sum = sum(p.ct_count for p in placements)
    if placed_sum != total:
        raise RuntimeError(
            f"placement 回溯与 DP 不一致：placements={placed_sum} dp={total}"
        )

    return BootstrapResult(
        bts_count=int(total),
        placements=placements,
        events=list(events),
        budget=budget,
        initial_level=initial_level,
    )


def _reconstruct_placements_forward(
    events: list[Event],
    parents: list[
        dict[tuple[int, int, int], tuple[tuple[int, int, int], int, int, int]]
    ],
    end_state: tuple[int, int, int],
) -> list[BootstrapPlacement]:
    n = len(events)
    # 从终点回溯到起点
    chain: list[tuple[int, tuple[int, int, int], int, int, int]] = []
    state = end_state
    for i in range(n, 0, -1):
        prev_state, action, add, mid = parents[i][state]
        chain.append((i - 1, prev_state, action, add, mid))
        state = prev_state
    chain.reverse()

    placements: list[BootstrapPlacement] = []
    for i, prev_state, action, add, mid in chain:
        ev = events[i]
        rem_before = prev_state[0]
        paid = 0
        if action == 1:
            bct = _boot_ct(ev)
            placements.append(
                BootstrapPlacement(
                    event_index=i,
                    event_name=ev.name,
                    ct_count=bct,
                    remaining_before=rem_before,
                    reason="before_event",
                )
            )
            paid += bct
        for _ in range(mid):
            mct = _mid_ct(ev)
            placements.append(
                BootstrapPlacement(
                    event_index=i,
                    event_name=ev.name,
                    ct_count=mct,
                    remaining_before=0,
                    reason="mid_event",
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
) -> BootstrapResult:
    events = build_bootstrap_events(task_name, scheme, phase=phase)
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
) -> int:
    return optimize_bootstrap(
        task_name,
        scheme,
        budget=budget,
        initial_level=initial_level,
        phase=phase,
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
    """校验各非线性 slot 的 ``sum(event.depth)`` 与 ``cost.*_poly_depth`` 一致。"""
    events = build_bootstrap_events(task_name, scheme, phase=phase)
    errors: list[str] = []
    for layer_idx in range(NUM_LAYERS):
        for kind in ("softmax", "ln1", "gelu", "ln2"):
            level = _slot_level(scheme, layer_idx, kind)
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
    tasks: tuple[str, ...] = ("mrpc", "rte", "sst2"),
    levels: tuple[int, ...] = (0, 1, 2),
    budget: int = BOOTSTRAP_DEPTH_BUDGET,
) -> list[str]:
    """
    阶段 D 自洽审计。返回错误列表（空 = 通过）。

    检查：CT 链、非线性 depth 守恒、pathway 每层 +2、
    fork/join 配对、join 事件 join_ct、单调性（low≤mid≤high）、
    Softmax×V 双侧 bootstrap 单元、placement 与 DP 一致。
    """
    errors: list[str] = []

    # --- 单元：Softmax×V 双侧 bootstrap ---
    ev = Event(
        name="ut.att_context",
        depth=1,
        ct_in=CT_ROTATED,
        ct_out=CT_HIDDEN,
        join_slot=SLOT_V,
        join_ct=CT_HIDDEN,
        refresh_side=True,
    )
    # spine rem=5, V rem=0 → join 后 rem=0 → mid boot 两侧
    add, r, mid, ns = _transition_event(
        5, (SLOT_UNSET, 0), ev, budget, bootstrap_first=False
    )
    expect_mid_ct = CT_ROTATED + CT_HIDDEN
    if add != expect_mid_ct or mid != 1 or r != budget - 1:
        errors.append(
            f"Softmax×V mid-boot 单元失败: add={add} mid={mid} r={r} "
            f"expected add={expect_mid_ct} mid=1 r={budget - 1}"
        )
    if ns[1] != budget:
        errors.append(f"Softmax×V mid-boot 未刷新 V 槽: {ns[1]} != {budget}")

    # Q×K 双侧计费但不刷 V
    ev_qk = Event(
        name="ut.att_score",
        depth=1,
        ct_in=CT_ROTATED,
        ct_out=CT_ATT_SCORE,
        join_slot=SLOT_V,
        join_offset=DEPTH_TRANSPOSE_K,
        join_ct=CT_HIDDEN,
        refresh_side=False,
    )
    add, r, mid, ns = _transition_event(
        0, (SLOT_UNSET, 2), ev_qk, budget, bootstrap_first=False
    )
    # side = max(0, 2-1)=1; rem=min(0,1)=0 → mid boot；不刷 V
    if add != CT_ROTATED + CT_HIDDEN or ns[1] != 2:
        errors.append(
            f"Q×K mid-boot 单元失败: add={add} V_slot={ns[1]} "
            f"(应 add={CT_ROTATED + CT_HIDDEN}, V 保持 2)"
        )

    # 残差 join：bootstrap_first 刷新 resid
    ev_res = Event(
        name="ut.join_resid",
        depth=0,
        ct_in=CT_HIDDEN,
        ct_out=CT_HIDDEN,
        join_slot=SLOT_RESID,
        join_ct=CT_HIDDEN,
        refresh_side=True,
    )
    add, r, mid, ns = _transition_event(
        3, (1, SLOT_UNSET), ev_res, budget, bootstrap_first=True
    )
    if add != 2 * CT_HIDDEN or r != budget or ns[0] != budget:
        errors.append(
            f"残差 bootstrap_first 单元失败: add={add} r={r} resid={ns[0]}"
        )

    for task in tasks:
        for level in levels:
            scheme = scheme_all(level)
            try:
                events = build_bootstrap_events(task, scheme, phase="C")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task} L{level} build: {exc}")
                continue

            # CT 链
            for i in range(len(events) - 1):
                if events[i].ct_out != events[i + 1].ct_in:
                    errors.append(
                        f"{task} L{level} CT 断裂 @ {events[i].name}"
                    )
                    break

            # 非线性 depth
            errs = audit_nonlinear_depth_splits(task, scheme, phase="C")
            errors.extend(f"{task} {e}" for e in errs)

            # pathway 每层 mask+copy = 2
            for layer_idx in range(NUM_LAYERS):
                pw = sum(
                    e.depth
                    for e in events
                    if e.layer_idx == layer_idx and e.kind == "pathway"
                )
                if pw != DEPTH_SOFTMAX_ATTN_MASK + DEPTH_SOFTMAX_COPY:
                    errors.append(
                        f"{task} L{level} L{layer_idx} pathway depth={pw}"
                    )

            # 每层 fork/join 配对
            for layer_idx in range(NUM_LAYERS):
                names = {
                    e.name.split(".", 1)[1]
                    for e in events
                    if e.layer_idx == layer_idx
                }
                for need in (
                    "fork_resid_attn",
                    "join_resid_attn",
                    "fork_resid_ff",
                    "join_resid_ff",
                    "linear_att_score",
                    "linear_att_context",
                ):
                    if need not in names:
                        errors.append(
                            f"{task} L{level} L{layer_idx} 缺事件 {need}"
                        )

            # join 事件必须带 join_ct
            for e in events:
                if e.join_slot is not None and e.join_ct <= 0:
                    errors.append(f"{task} {e.name} join 无 join_ct")

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
            f"ct={p.ct_count}  rem_before={p.remaining_before}  {p.reason}"
        )
    if len(result.placements) > max_placements:
        print(f"    ... ({len(result.placements) - max_placements} more)")


if __name__ == "__main__":
    print("bts_ops 阶段 D 自洽审计 + C smoke")
    print("=" * 56)
    d_errs = run_phase_d_audit()
    if d_errs:
        print(f"阶段 D FAILED ({len(d_errs)}):")
        for e in d_errs[:20]:
            print(" ", e)
    else:
        print("阶段 D: 全部检查通过")

    task = "mrpc"
    print()
    for label, level in [("all_low", 0), ("all_mid", 1), ("all_high", 2)]:
        r = optimize_bootstrap(task, scheme_all(level), phase="C")
        print(f"[{label}] {r.summary()}")
        join_pl = [
            p for p in r.placements
            if "att_context" in p.event_name
            or "att_score" in p.event_name
            or "join_resid" in p.event_name
        ]
        if join_pl:
            print(f"  join 相关放置: {len(join_pl)}")
            for p in join_pl[:4]:
                print(f"    {p.reason} @ {p.event_name} ct={p.ct_count}")
