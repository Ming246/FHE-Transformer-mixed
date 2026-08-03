"""
THOR Encoder 线性段核心（纯实数 slot，独立可移植）。

从早期 ``linear_logic`` 实验抽出默认全流程，去掉验证 / DualRail / index 旁路。
依赖：仅 ``numpy``。可直接拷贝到其他项目。

---------------------------------------------------------------------------
数据流（单层 ``run_encoder_layer``）
---------------------------------------------------------------------------
  X (seq, hidden)
    --encode_input_lower_diagonals-->  input_lower  [n_input_packs 个 slot 向量]
    --PC-MM Q/K/V-->                   q/k/v_merged [n_out_packed_qkv]
    --transpose(K)-->                  k_lower
    --make_copies_real(Q)-->           q_copies     [head_dim]
    --Cor3.8 score scatter-->          score        [n_input_packs]
    --softmax(layout 不变)-->          alpha
    --make_copies(alpha)-->            alpha_copies [seq_len]
    --Cor3.8 context scatter-->        ctx          [n_out_packed_qkv]
    --PC-MM W_O-->                     wo_pc
    --layout bridge + residual-->      attn_res     [= input_lower layout]
    --FC1 / (GELU占位) / FC2-->        ff_out
    --layout bridge + residual-->      layer_out    [= input_lower，可堆叠]

原语（与 HE 同构）：pt*ct、ct*ct（逐 slot 乘）、rotate、mask、add。
优化：对角打包、make_copies、纯实 BSGS（两种后端）：
  - geometric（默认）：几何 rrot 分组，形状自适应，≡ NumPy
  - ttemp：THOR 固定 rrot + 自适应 ttemp 路由（手写表的闭式推广）；
    纯实布局下不保证 ≡ NumPy（原文该结构与 DualRail 打包耦合）
Softmax / GELU / LayerNorm：本文件仅保留 layout 不变接口
（softmax 用明文 exp；GELU/LN 为 pass-through）。接 FHE 时替换数值即可。

用法::

    from thor_encoder_linear_core import (
        bert_base, bert_medium, MaskBank, encode_input_lower_diagonals,
        run_encoder_layer, run_encoder_stack,
    )

    cfg = bert_base()       # 或 bert_medium() / bert_large()
    masks = MaskBank(cfg)          # 编译期 scatter 表，可复用
    x_in = encode_input_lower_diagonals(x, cfg)
    y = run_encoder_layer(x_in, weights, cfg, masks)
    # 或堆叠：run_encoder_stack(x_in, [w0, w1, ...], cfg, masks)

weights 字典键：``w_q, w_k, w_v, w_o, w1, w2``（普通 NumPy 矩阵）。
支持形状：``bert_base`` / ``bert_large``（pack=16）与 ``bert_medium``（512/8/2048, pack=32）等
满足 ``encoder_linear_applies`` 的配置。
---------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ===========================================================================
# 1. 配置与几何
# ===========================================================================


@dataclass
class ThorConfig:
    """
    THOR 线性层维度配置。

    **仅需指定 5 个量**（BERT 形状 + CKKS slot 总数 ``num_slots = 2^logN``）::

        seq_len, hidden_dim, num_heads, ffn_dim, num_slots

    其余全部由 ``derive()`` 计算（见 ``pack`` / ``slot_stride`` 等属性）::

        head_dim           = hidden_dim / num_heads
        head_slot_pack     = 不小于 num_heads 的最小 2 幂（slot 内 head 轴宽度）
        slot_stride        = seq_len × head_slot_pack（相邻 subpack 在 slot 中的间距）
        pack               = num_slots / slot_stride（每个 slot 向量并行打包的对角线条数 C）
        n_input_packs      = seq_len / pack
        n_out_packed_qkv   = head_dim / pack
        ff_n_splits        = ffn_dim / hidden_dim
    """

    seq_len: int = 128
    hidden_dim: int = 768
    num_heads: int = 12
    ffn_dim: int = 3072
    num_slots: int = 32768

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def head_slot_pack(self) -> int:
        """slot 内 head 轴长度：``num_heads`` 向上取 2 的幂（4 头→4，12 头→16）。"""
        if self.num_heads <= 1:
            return 1
        return 1 << (self.num_heads - 1).bit_length()

    @property
    def head_pad_indices(self) -> list[int]:
        return list(range(self.num_heads, self.head_slot_pack))

    @property
    def slot_stride(self) -> int:
        """同一 slot 向量内，相邻 subpack（j 与 j+1）之间的循环移位步长。"""
        return self.seq_len * self.head_slot_pack

    @property
    def pack(self) -> int:
        """
        每个 slot 密文并行打包的下/上对角线条数 C。

        满足 ``num_slots = pack × slot_stride``；也是 BSGS 内层的分组宽度。
        """
        if self.num_slots % self.slot_stride != 0:
            raise ValueError(
                f"num_slots={self.num_slots} 不能被 slot_stride={self.slot_stride} 整除，"
                "请调整 num_heads / seq_len / num_slots"
            )
        return self.num_slots // self.slot_stride

    @property
    def n_hidden_row_blocks(self) -> int:
        return self.hidden_dim // self.head_dim

    @property
    def n_hidden_col_blocks(self) -> int:
        return self.hidden_dim // self.seq_len

    @property
    def n_ffn_row_blocks(self) -> int:
        return self.ffn_dim // self.seq_len

    @property
    def n_in_slot(self) -> int:
        """PC/CC-MM 内维 slot 条数（纯实数 → 等于 seq_len）。"""
        return self.seq_len

    @property
    def n_in_complex(self) -> int:
        return self.n_in_slot

    @property
    def n_out_packed_qkv(self) -> int:
        return self.head_dim // self.pack

    @property
    def n_input_packs(self) -> int:
        return self.seq_len // self.pack

    @property
    def n_q_groups(self) -> int:
        """CC-MM 中 ``q = n // pack`` 的分组数（= ``n_input_packs``）。"""
        return self.n_input_packs

    @property
    def n_diag_blocks_qkv(self) -> int:
        return min(self.n_hidden_row_blocks, self.n_hidden_col_blocks)

    @property
    def ff_n_splits(self) -> int:
        """FF1 垂直切分（THOR ``vsplit = ffn_dim / hidden_dim``）。"""
        return self.ffn_dim // self.hidden_dim

    @property
    def ff_pack(self) -> int:
        """
        FF 对角打包宽度 C_ff。

        THOR ``model_encoder._encode_w_ff`` 写死 ``pack=16``；对 BERT-base 实为
        ``sqrt(num_slots / seq_len)``（32768/128=256→16）。与 QKV 的 ``self.pack`` 独立。
        """
        ratio = self.num_slots // self.seq_len
        if ratio <= 0 or self.num_slots % self.seq_len != 0:
            raise ValueError(
                f"num_slots={self.num_slots} 须能被 seq_len={self.seq_len} 整除以推导 ff_pack"
            )
        fp = int(ratio**0.5)
        if fp * fp != ratio:
            raise ValueError(
                f"num_slots/seq_len={ratio} 须为完全平方数（当前无法推导 ff_pack）"
            )
        return fp

    @property
    def ff_stride(self) -> int:
        """FF subpack 间距 = ``num_slots / ff_pack``（THOR ``rotate_left(..., 2**11)`` 当 ff_pack=16）。"""
        return self.num_slots // self.ff_pack

    @property
    def ff_band_half(self) -> int:
        """FF slot 半区宽度 = ``ff_pack // 2``（THOR mask 模数 8 = 16//2）。"""
        return self.ff_pack // 2

    @property
    def ff_block_ll(self) -> int:
        """
        FF ``(seq×seq)`` 块对角条数 ll。

        THOR ``linear.py`` 写死 ``delta=6-l``；实为 ``hidden_dim/seq_len``（768/128=6）。
        """
        return self.n_hidden_col_blocks

    def validate(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        if self.hidden_dim % self.seq_len != 0:
            raise ValueError("hidden_dim 必须能被 seq_len 整除（列块划分）")
        if self.ffn_dim % self.seq_len != 0:
            raise ValueError("ffn_dim 必须能被 seq_len 整除")
        if self.ffn_dim % self.hidden_dim != 0:
            raise ValueError("ffn_dim 必须能被 hidden_dim 整除")
        _ = self.pack  # 触发 num_slots 可整除检查
        if self.seq_len % self.pack != 0:
            raise ValueError(f"seq_len={self.seq_len} 必须能被 pack={self.pack} 整除")
        if self.head_dim % self.pack != 0:
            raise ValueError(f"head_dim={self.head_dim} 必须能被 pack={self.pack} 整除")
        if self.num_slots != self.pack * self.slot_stride:
            raise ValueError(
                f"num_slots 应等于 pack×slot_stride = {self.pack}×{self.slot_stride}"
            )

    def format_derived(self) -> str:
        """打印派生参数，便于核对。"""
        lines = [
            f"  seq_len={self.seq_len}  hidden_dim={self.hidden_dim}  "
            f"num_heads={self.num_heads}  ffn_dim={self.ffn_dim}  num_slots={self.num_slots}",
            f"  → head_dim={self.head_dim}  head_slot_pack={self.head_slot_pack}  "
            f"head_pad={self.head_pad_indices or '无'}",
            f"  → slot_stride={self.slot_stride}  pack={self.pack}  "
            f"(num_slots = {self.pack}×{self.slot_stride} = {self.pack * self.slot_stride})",
            f"  → n_in_slot={self.n_in_slot}  n_input_packs={self.n_input_packs}  "
            f"n_out_packed_qkv={self.n_out_packed_qkv}",
            f"  → n_hidden_col_blocks={self.n_hidden_col_blocks}  "
            f"n_hidden_row_blocks={self.n_hidden_row_blocks}  "
            f"n_ffn_row_blocks={self.n_ffn_row_blocks}  ff_n_splits={self.ff_n_splits}",
            f"  → ff_pack={self.ff_pack}  ff_stride={self.ff_stride}  "
            f"ff_block_ll={self.ff_block_ll}  ff_band_half={self.ff_band_half}",
            f"  → PC-MM ll={self.n_diag_blocks_qkv}  "
            f"QKV block=({self.head_dim},{self.seq_len})  "
            f"W_O block=({self.seq_len},{self.head_dim})  "
            f"FF block=({self.seq_len},{self.seq_len})",
        ]
        return "\n".join(lines)


def standard_bert() -> ThorConfig:
    """
    标准验证 preset：4 头、整除友好，无 head padding。

    seq=128, hidden=256, heads=4, ffn=1024, num_slots=32768
    → head_dim=64, head_slot_pack=4, slot_stride=512, pack=64
    """
    return ThorConfig(
        seq_len=128,
        hidden_dim=256,
        num_heads=4,
        ffn_dim=1024,
        num_slots=32768,
    )


def bert_base() -> ThorConfig:
    """原始 BERT-base 尺寸（12 头 → head padding 到 16，pack=16）。"""
    return ThorConfig()


def bert_medium() -> ThorConfig:
    """
    中等尺寸：hidden=512, heads=8, ffn=2048（seq=128, slots=32768）。

    几何：``head_dim=64, head_slot_pack=8, pack=32, n_out=2``（与 base/large 不同打包宽度，
    仍走同一套纯实 scatter / PC-MM；不要求 ``path_b_applies``）。
    """
    return ThorConfig(
        seq_len=128,
        hidden_dim=512,
        num_heads=8,
        ffn_dim=2048,
        num_slots=32768,
    )


def bert_large() -> ThorConfig:
    """
    官方 BERT-large：16 头、无 head padding。

    与 bert_base 同 ``pack=16, n_out=4, head_dim=64`` → Path B 几何可复用；
    差异在 ``hidden=1024``、``ll=8``、``num_heads=16``。
    """
    return ThorConfig(seq_len=128, hidden_dim=1024, num_heads=16, ffn_dim=4096)


def mini_bert() -> ThorConfig:
    """快速 smoke preset（12 头有 padding；num_slots 随尺寸推导为 8192）。"""
    return ThorConfig(
        seq_len=32,
        hidden_dim=192,
        num_heads=12,
        ffn_dim=768,
        num_slots=8192,
    )


def path_b_applies(cfg: ThorConfig) -> bool:
    """经典 Path B 几何（bert_base / bert_large：``pack=16, n_out=4, head_dim=64``）。"""
    return cfg.pack == 16 and cfg.n_out_packed_qkv == 4 and cfg.head_dim == 64


def encoder_linear_applies(cfg: ThorConfig) -> bool:
    """
    本核心模块可跑的线性几何：配置自洽 + FF ``block_diag_2``。

    覆盖 ``bert_base`` / ``bert_large``（pack=16）与 ``bert_medium``（pack=32）等。
    """
    try:
        cfg.validate()
    except ValueError:
        return False
    return _use_ff_block_diag2(cfg)



# ===========================================================================
# 对角 / 分块工具
# ===========================================================================

def ud_entry(matrix: np.ndarray, u: int, i: int, rot: int = 0) -> float:
    i = i + rot
    a, b = matrix.shape
    return matrix[i % a, (u + i) % b]


def ld_entry(matrix: np.ndarray, l: int, i: int) -> float:
    b, c = matrix.shape
    return matrix[(l + i) % b, i % c]


def to_blocks(matrix: np.ndarray, block_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    a, b = matrix.shape
    bh, bw = block_shape
    if a % bh or b % bw:
        raise ValueError("矩阵尺寸必须能被 block_shape 整除")
    v, h = a // bh, b // bw
    blocks = np.empty((v, h), dtype=object)
    row_blocks = np.vsplit(matrix, v)
    for i, row_block in enumerate(row_blocks):
        blocks[i] = np.hsplit(row_block, h)
    l, d = min(v, h), max(v, h)
    diag_blocks = np.empty((l, d), dtype=object)
    for t in range(d):
        for i in range(l):
            diag_blocks[i, t] = blocks[(i + t) % v, t % h]
    return diag_blocks, (l, d)



# ===========================================================================
# HE 同构原语（rotate / mask / add / pt*ct）
# ===========================================================================

def rotate_left(vec: np.ndarray, r: int) -> np.ndarray:
    r = r % len(vec)
    if r == 0:
        return vec.copy()
    return np.roll(vec, -r)


def rotate_cc_left(vec: np.ndarray, rrot: int) -> np.ndarray:
    """CC-MM 左因子旋转：对应 THOR ``engine.rotate_left(vec, -rrot)``（与 ``rotate_left(vec, rrot)`` 同义）。"""
    return rotate_left(vec, rrot)


def pt_mult(pt: np.ndarray, ct: np.ndarray) -> np.ndarray:
    return np.asarray(pt, dtype=np.float64) * np.asarray(ct, dtype=np.float64)


def add_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def rotate_internal(vec: np.ndarray, delta: int, mode: Literal["block_diag_1", "block_diag_2", "att"], cfg: ThorConfig) -> np.ndarray:
    """模拟 THOR ``rotate_internal``（mask + 全局 rotate_left，与 linear.py 一致）。"""
    if delta == 0:
        return vec.copy()
    if mode == "att":
        l_delta = cfg.pack * delta
        r_delta = cfg.slot_stride - l_delta
        mask = np.ones(len(vec), dtype=np.float64)
        mask[np.arange(len(vec)) % cfg.slot_stride >= l_delta] = 0.0
    elif mode == "block_diag_1":
        l_delta = delta
        # THOR ``linear.py``：``r_delta = 12 - l_delta``（= ``num_heads``，非 ``head_slot_pack``）
        r_delta = cfg.num_heads - l_delta
        mask = np.ones(len(vec), dtype=np.float64)
        mask[np.arange(len(vec)) % cfg.head_slot_pack >= l_delta] = 0.0
    else:
        l_delta = delta
        r_delta = cfg.ff_block_ll - l_delta
        mask = np.ones(len(vec), dtype=np.float64)
        mask[np.arange(len(vec)) % cfg.ff_band_half >= l_delta] = 0.0
    temp1 = mask * vec
    temp2 = vec - temp1
    return rotate_left(temp1, -r_delta) + rotate_left(temp2, l_delta)


def _slot_index(cfg: ThorConfig, subpack_j: int, token_t: int, head_d: int) -> int:
    return subpack_j * cfg.slot_stride + token_t * cfg.head_slot_pack + head_d


def _ff_stride(cfg: ThorConfig) -> int:
    return cfg.ff_stride


def _ff_slot_index(cfg: ThorConfig, subpack_j: int, token_t: int, band_d: int) -> int:
    """FF slot 布局：``subpack * ff_stride + token * ff_pack + band``（THOR ``temp+t*16+d``）。"""
    return subpack_j * _ff_stride(cfg) + token_t * cfg.ff_pack + band_d



# ===========================================================================
# 输入 / 权重编码
# ===========================================================================

def encode_input_lower_diagonals(x: np.ndarray, cfg: ThorConfig) -> list[np.ndarray]:
    """
    输入 X (seq_len, hidden_dim) -> 转置 X^T (hidden, seq_len) 的下对角线打包。

    对应 ``ThorDataEncryptor.encrypt_embedding``：
      - 4 个 slot 向量（BERT-base），每个打包 16 条下对角线
      - 每条对角线在 slot 内按 [token][head] 交织，head 维 padding 到 16
    """
    cfg.validate()
    if x.shape != (cfg.seq_len, cfg.hidden_dim):
        raise ValueError(f"期望输入形状 ({cfg.seq_len}, {cfg.hidden_dim})")

    x_t = x.T
    x_blocks = np.vsplit(x_t, cfg.n_hidden_col_blocks)
    n_packs = cfg.n_input_packs
    vectors: list[np.ndarray] = []

    for pack_i in range(n_packs):
        vec = np.zeros(cfg.num_slots, dtype=np.float64)
        for j in range(cfg.pack):
            diag_index = pack_i * cfg.pack + j
            if diag_index >= cfg.seq_len:
                continue
            for t in range(cfg.seq_len):
                for h in range(cfg.num_heads):
                    x_b = x_blocks[h % cfg.n_hidden_col_blocks]
                    idx = _slot_index(cfg, j, t, h)
                    vec[idx] = ld_entry(x_b, diag_index, t)
        vectors.append(vec)
    return vectors


def encode_weight_upper_diagonals(
    w: np.ndarray,
    cfg: ThorConfig,
    n_in: int,
    n_out: int,
    block_shape: tuple[int, int],
    scale: float = 1.0,
) -> np.ndarray:
    """
    权重矩阵 W 的上对角线 plaintext 打包（PC-MM 侧）。

    返回 shape ``(n_out/pack, n_diag_blocks, n_in)`` 的 slot 向量数组（纯实数）。
    对应 ``ThorModelEncoder._encode_w_att``，但内维为完整 ``n_in``（不复数打包）。
    """
    cfg.validate()
    ld_blocks, (ll, _) = to_blocks(w, block_shape)
    n_out_p = n_out // cfg.pack
    n_in_s = n_in
    pts = np.empty((n_out_p, ll, n_in_s), dtype=object)

    for l in range(ll):
        diagonal = ld_blocks[l]
        for n in range(n_in_s):
            for out in range(n_out_p):
                msg = np.zeros(cfg.num_slots, dtype=np.float64)
                for j in range(cfg.pack):
                    i = ((n // cfg.pack) * cfg.pack + out * cfg.pack + (n + j) % cfg.pack) % n_in_s
                    r = out * cfg.pack + j
                    i_rot = (i - r) % n_in
                    subpack_j = j
                    for t in range(cfg.seq_len):
                        for d in range(diagonal.shape[0]):
                            block = diagonal[d]
                            msg[_slot_index(cfg, subpack_j, t, d)] = (
                                scale * ud_entry(block, i_rot, t, r)
                            )
                pts[out, l, n] = msg
    return pts


def encode_ff_weight(
    w: np.ndarray,
    cfg: ThorConfig,
    split: Literal["vertical", "horizontal"],
    n_splits: int,
    scale: float = 1.0,
) -> np.ndarray:
    """
    FF 层权重编码（纯实数 ``block_diag_2``）。

    返回 shape ``(2, n_out_p, ll, n_in)``，``n_in=seq_len=128``（THOR 复数 64 路的实数展开）。
    split='vertical'  -> FF1; split='horizontal' -> FF2
    """
    block_shape = (cfg.seq_len, cfg.seq_len)
    if split == "vertical":
        w_list = np.vsplit(w, n_splits)
    else:
        w_list = np.hsplit(w, n_splits)

    ld_blocks_list = [to_blocks(part, block_shape)[0] for part in w_list]
    ll = ld_blocks_list[0].shape[0]
    n_out = cfg.seq_len
    n_in = cfg.seq_len
    n_out_p = n_out // cfg.ff_pack
    band2 = cfg.ff_pack // 2
    pts = np.empty((2, n_out_p, ll, n_in), dtype=object)

    for rep in range(2):
        b_lo = ld_blocks_list[(rep * 2) % len(ld_blocks_list)]
        b_hi = ld_blocks_list[(rep * 2 + 1) % len(ld_blocks_list)]
        n_bands = min(cfg.n_hidden_col_blocks, b_lo.shape[1])
        for l in range(ll):
            for n in range(n_in):
                for out in range(n_out_p):
                    msg = np.zeros(cfg.num_slots, dtype=np.float64)
                    for j in range(cfg.ff_pack):
                        i = (
                            (n // cfg.ff_pack) * cfg.ff_pack
                            + out * cfg.ff_pack
                            + (n + j) % cfg.ff_pack
                        ) % n_in
                        r = out * cfg.ff_pack + j
                        i_rot = (i - r) % n_in
                        for t in range(cfg.seq_len):
                            for d in range(n_bands):
                                msg[_ff_slot_index(cfg, j, t, d)] = (
                                    scale * ud_entry(b_lo[l, d], i_rot, t, r)
                                )
                                msg[_ff_slot_index(cfg, j, t, d + band2)] = (
                                    scale * ud_entry(b_hi[l, d], i_rot, t, r)
                                )
                    pts[rep, out, l, n] = msg
    return pts



# ===========================================================================
# PC-MM（Algorithm 1）
# ===========================================================================

def make_rotated_copies(vectors: list[np.ndarray], cfg: ThorConfig, n_in: int | None = None) -> list[np.ndarray]:
    """``n_in/pack`` 个打包向量 → ``n_in`` 个旋转副本（纯实数）。"""
    n_in = cfg.n_in_slot if n_in is None else n_in
    n_packs = n_in // cfg.pack
    if len(vectors) != n_packs:
        raise ValueError(f"期望 {n_packs} 个打包向量，得到 {len(vectors)}")
    rots: list[np.ndarray] = [None] * n_in  # type: ignore
    for i in range(n_packs):
        rots[i * cfg.pack] = vectors[i].copy()
        for j in range(1, cfg.pack):
            rots[i * cfg.pack + j] = rotate_left(rots[i * cfg.pack + j - 1], cfg.slot_stride)
    return rots


def parallel_diagonal_pt_ct_mult(
    w_pts: np.ndarray,
    x_vecs: list[np.ndarray],
    cfg: ThorConfig,
    in_pack: int | None = None,
) -> np.ndarray:
    """
    并行计算块下对角线：对应 ``parallel_diagonal_pt_ct_mult``。
    返回 shape ``(n_out_p, ll)`` 的 slot 向量数组。
    """
    pack = cfg.pack if in_pack is None else in_pack
    n_in = w_pts.shape[-1]
    n_out_p = w_pts.shape[0]
    ll = w_pts.shape[1]
    ct_diags = np.empty((n_out_p, ll), dtype=object)

    for out in range(n_out_p):
        for l in range(ll):
            acc = pt_mult(w_pts[out, l, 0], x_vecs[(pack * out) % n_in])
            for n in range(1, n_in):
                acc = add_vec(acc, pt_mult(w_pts[out, l, n], x_vecs[(pack * out + n) % n_in]))
            ct_diags[out, l] = acc
    return ct_diags


def pt_ct_matmul(
    w_pts: np.ndarray,
    x_vecs: list[np.ndarray],
    cfg: ThorConfig,
    mode: Literal["block_diag_1", "block_diag_2"] = "block_diag_1",
    in_pack: int | None = None,
) -> list[np.ndarray]:
    """
    对角块 PC-MM 完整流程（Algorithm 1 Step 1~3 的明文版）。
    返回 ``n_out_p`` 个输出 slot 向量。
    """
    pack = cfg.ff_pack if mode == "block_diag_2" else cfg.pack
    if in_pack is not None:
        pack = in_pack
    ct_diags = parallel_diagonal_pt_ct_mult(w_pts, x_vecs, cfg, in_pack=pack)
    n_out_p, ll = ct_diags.shape
    if mode == "block_diag_1":
        rot_base = cfg.num_heads
    else:
        rot_base = cfg.ff_block_ll

    outputs: list[np.ndarray] = []
    for out in range(n_out_p):
        acc = ct_diags[out, 0].copy()
        for l in range(1, ll):
            delta = rot_base - l
            rotated = rotate_internal(ct_diags[out, l], delta, mode, cfg)
            acc = add_vec(acc, rotated)
        outputs.append(acc)
    return outputs



# ===========================================================================
# CC-MM 辅助：make_copies / K transpose
# ===========================================================================

def make_copies(vectors: list[np.ndarray], cfg: ThorConfig) -> list[np.ndarray]:
    """Replication：第 ``l`` 条下对角线复制到 pack 内全部 subpack（对齐 THOR ``make_copies``）。"""
    n_in = cfg.n_in_slot
    copies: list[np.ndarray] = [None] * n_in  # type: ignore
    if len(vectors) * cfg.pack != n_in:
        raise ValueError(f"期望 {n_in // cfg.pack} 个向量，得到 {len(vectors)}")

    for i, base_vec in enumerate(vectors):
        for j in range(cfg.pack):
            idx = i * cfg.pack + j
            vec = np.zeros(cfg.num_slots, dtype=np.float64)
            for t in range(cfg.seq_len):
                for h in range(cfg.head_slot_pack):
                    src = _slot_index(cfg, j, t, h)
                    val = base_vec[src]
                    for j2 in range(cfg.pack):
                        dst = _slot_index(cfg, j2, t, h)
                        vec[dst] = val
            copies[idx] = vec
    return copies


def make_copies_from_packs(
    vectors: list[np.ndarray], cfg: ThorConfig, n_in: int
) -> list[np.ndarray]:
    """
    实数对角复制（THOR ``make_copies`` 语义，无复数打包）。

    **仅** ``mask`` / ``rotate`` / ``add``（``rotsum``），禁止运行时任意 slot 索引写。
    """
    n_packs = len(vectors)
    if n_packs * cfg.pack != n_in:
        raise ValueError(f"期望 {n_in // cfg.pack} 个 pack，得到 {n_packs}（n_in={n_in}）")
    s, stride, p, hsp = cfg.num_slots, cfg.slot_stride, cfg.pack, cfg.head_slot_pack
    copies: list[np.ndarray] = []
    for i, base_vec in enumerate(vectors):
        for j in range(p):
            # 抽出 subpack j：token×head 块内仅保留列 j
            mask_j = np.zeros(s, dtype=np.float64)
            for t in range(cfg.seq_len):
                for h in range(hsp):
                    mask_j[_slot_index(cfg, j, t, h)] = 1.0
            extracted = pt_mult(mask_j, base_vec)
            # 广播到所有 subpack：沿 pack 轴 rotsum（步长 = slot_stride）
            # 先把 j 列移到 0，再 rotsum，再视需要（此处各 dst 同值）
            shifted = rotate_left(extracted, (j * stride) % s)
            # rotsum over pack positions
            acc = shifted.copy()
            for k in range(1, p):
                acc = add_vec(acc, rotate_left(shifted, (k * stride) % s))
            copies.append(acc)
    if len(copies) != n_in:
        raise RuntimeError(f"make_copies 得到 {len(copies)}，期望 {n_in}")
    return copies


def make_copies_real(
    vectors: list[np.ndarray],
    cfg: ThorConfig,
    n_in: int | None = None,
) -> list[np.ndarray]:
    """实数版 ``make_copies``：见 ``make_copies_from_packs``（HE mask/rotate/rotsum）。"""
    return make_copies_from_packs(vectors, cfg, cfg.head_dim if n_in is None else n_in)


def _build_transpose_masks(cfg: ThorConfig) -> dict:
    """THOR ``linear.py`` transpose mask（仅经典 Path B ``pack=16,n_out=4`` 对齐）。"""
    s, p, hd, stride, hsp = cfg.num_slots, cfg.pack, cfg.head_dim, cfg.slot_stride, cfg.head_slot_pack
    n_out = cfg.n_out_packed_qkv
    masks: dict = {"mask0": {}, "mask1": {}, "mask2": {}, "mask3": {}}

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(s, dtype=np.float64)
        n = min(len(arr), s)
        out[:n] = arr[:n]
        return out

    for i in range(n_out):
        n_diag = p * i
        inner = (n_diag - p) % hd + p
        arr0 = np.ones(p * (hd + inner), dtype=np.float64)
        arr1 = np.concatenate(
            [
                np.zeros(s - p * (hd - inner), dtype=np.float64),
                np.ones(p * (hd - inner), dtype=np.float64),
            ]
        )
        masks["mask0"][i] = _pad(arr0)
        masks["mask1"][i] = _pad(arr1)
        for j in range(p * i + 1, p * (i + 1)):
            l = hd - j
            arr2 = np.concatenate(
                [np.zeros(stride * (p - j % p), dtype=np.float64), np.ones((cfg.seq_len - l) * hsp, dtype=np.float64)]
            )
            arr3 = np.concatenate(
                [
                    np.zeros(stride * (p - j % p - 1), dtype=np.float64),
                    np.zeros((cfg.seq_len - l) * hsp, dtype=np.float64),
                    np.ones(p * l, dtype=np.float64),
                ]
            )
            masks["mask2"][j] = _pad(arr2)
            masks["mask3"][j] = _pad(arr3)
    return masks


def _transpose_upper_to_lower_slot(upper_vecs: list[np.ndarray], cfg: ThorConfig) -> list[np.ndarray]:
    """``linear.py::transpose_upper_to_lower`` 明文模拟（Path B 几何）。"""
    n_out = len(upper_vecs)
    if n_out != cfg.n_out_packed_qkv:
        raise ValueError(f"期望 {cfg.n_out_packed_qkv} 个上对角线向量，得到 {n_out}")
    masks = _build_transpose_masks(cfg)
    p, hd, s, stride = cfg.pack, cfg.head_dim, cfg.num_slots, cfg.slot_stride
    l_temp: list[list[np.ndarray | None]] = [[None, None] for _ in range(n_out)]

    for i in range(n_out):
        n_diag = p * i
        delta = (((hd - n_diag) % hd) * p) % s
        rot_i = rotate_left(upper_vecs[i], delta)
        l_temp[(n_out - i) % n_out][0] = pt_mult(masks["mask0"][i], rot_i)
        l_temp[(n_out - i) % n_out][1] = pt_mult(masks["mask1"][i], rot_i)

    for i in range(n_out):
        for n_diag_u in range(p * i + 1, p * (i + 1)):
            l = hd - n_diag_u
            delta = (
                l * cfg.head_slot_pack
                + (((n_diag_u % max(hd - p, 1)) * 2) % p) * stride
            ) % s
            rot_i = rotate_left(upper_vecs[i], delta)
            ti = (n_out - 1 - i) % n_out
            t2 = pt_mult(masks["mask2"][n_diag_u], rot_i)
            t3 = pt_mult(masks["mask3"][n_diag_u], rot_i)
            l_temp[ti][0] = add_vec(l_temp[ti][0], t2) if l_temp[ti][0] is not None else t2
            l_temp[ti][1] = add_vec(l_temp[ti][1], t3) if l_temp[ti][1] is not None else t3

    lower: list[np.ndarray] = []
    for i in range(n_out):
        assert l_temp[i][0] is not None and l_temp[i][1] is not None
        lower.append(add_vec(l_temp[i][0], rotate_left(l_temp[i][1], -stride)))
    return lower


def _build_transpose_scatter_pairs(
    cfg: ThorConfig,
) -> dict[tuple[int, int, int], list[int]]:
    """
    编译期：上对角 → ``L(K)`` 下对角的几何散射表（任意 pack / n_out）。

    键 ``(pack_u, pack_l, rrot)``；值：写入 ``lower[pack_l]`` 的 dest slot 列表。
    ``rotate(upper[pack_u], rrot)[dest] → lower[pack_l][dest]``。
    """
    n_out = cfg.n_out_packed_qkv
    s, pack, hd = cfg.num_slots, cfg.pack, cfg.head_dim
    seq, stride, hsp = cfg.seq_len, cfg.slot_stride, cfg.head_slot_pack
    groups: dict[tuple[int, int, int], list[int]] = {}

    for pack_l in range(n_out):
        for j in range(pack):
            l = pack_l * pack + j
            if l >= hd:
                continue
            for t in range(seq):
                for h in range(cfg.num_heads):
                    t_query = (l + t) % seq
                    col = t % hd
                    diag_u = (col - t_query) % hd
                    pack_u, j_u = diag_u // pack, diag_u % pack
                    if pack_u >= n_out:
                        continue
                    dest = j * stride + t * hsp + h
                    src = j_u * stride + t_query * hsp + h
                    rrot = (src - dest) % s
                    key = (pack_u, pack_l, rrot)
                    groups.setdefault(key, []).append(dest)
    return groups


def _transpose_upper_to_lower_scatter(
    upper_vecs: list[np.ndarray],
    cfg: ThorConfig,
    scatter_groups: dict[tuple[int, int, int], list[int]] | None = None,
) -> list[np.ndarray]:
    """通用上→下对角：``rotate`` + 编译期 place（≡ index，任意几何）。"""
    n_out = cfg.n_out_packed_qkv
    if len(upper_vecs) != n_out:
        raise ValueError(f"期望 {n_out} 个上对角线向量，得到 {len(upper_vecs)}")
    if scatter_groups is None:
        scatter_groups = _build_transpose_scatter_pairs(cfg)
    lower = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(n_out)]
    for (pack_u, pack_l, rrot), dests in scatter_groups.items():
        rot = rotate_left(upper_vecs[pack_u], rrot)
        sp = lower[pack_l]
        for dest in dests:
            sp[dest] += rot[dest]
    return lower


def transpose_upper_to_lower(upper_vecs: list[np.ndarray], cfg: ThorConfig) -> list[np.ndarray]:
    """
    上对角线 → CC 用下对角线。

    - Path B（``pack=16,n_out=4``）：THOR ``linear.py`` mask/rotate。
    - 其它几何（如 ``bert_medium`` pack=32）：通用几何 scatter（rotate + place）。
    """
    if path_b_applies(cfg):
        return _transpose_upper_to_lower_slot(upper_vecs, cfg)
    return _transpose_upper_to_lower_scatter(upper_vecs, cfg)



# ===========================================================================
# CC-MM：纯实 BSGS
#
# 两条后端（同一套 Cor 3.8 数学目标）：
#   1) geometric（默认）：按几何 rrot 分组 rotate+place；形状自适应，≡ index/NumPy。
#   2) ttemp：THOR 固定 rrot(n)=stride*j-pack*n + ct_ct 四分片 + 自适应 ttemp 路由。
#      路由已从 bert.py 手写表闭式推广到任意 n_out；但固定 rrot 与纯实布局耦合于
#      DualRail 打包，纯实下不保证 ≡ NumPy（见 compute_*(..., backend=\"ttemp\")）。
# ===========================================================================

def _k_lower_slot_index(
    cfg: ThorConfig, h: int, row: int, col: int
) -> tuple[int, int] | None:
    """``L(K)`` 中 ``K[h,row,col]`` 的 ``(pack_i, slot_idx)``；**仅 mask 预计算**。"""
    n_band = cfg.seq_len // cfg.head_dim
    for bi in range(n_band):
        t = col + cfg.head_dim * bi
        if t >= cfg.seq_len:
            continue
        d = (row - t) % cfg.seq_len
        if d >= cfg.head_dim:
            continue
        pack_i, j = d // cfg.pack, d % cfg.pack
        if pack_i >= cfg.n_out_packed_qkv:
            return None
        return pack_i, _slot_index(cfg, j, t, h)
    return None


def _build_cc_score_scatter_pairs(cfg: ThorConfig) -> dict[tuple[int, int, int], list[list[tuple[int, int]]]]:
    """
    编译期：Cor 3.8 每项 ``prod=rotate(K,-rrot)*Q_copy[l]`` 的散射表。

    键 ``(l, k_out, rrot)``；值 ``[pack_i -> [(prod_slot, score_idx), ...]]``。
    每项 ``rrot = (idx_k - idx_q) % num_slots``（逐 term，非固定 ``rrot(l)``）。
    """
    n_out = cfg.n_out_packed_qkv
    m, dim = cfg.seq_len, max(cfg.seq_len, cfg.head_dim, cfg.seq_len)
    groups: dict[tuple[int, int, int], list[list[tuple[int, int]]]] = {}

    def _pairs(key: tuple[int, int, int]) -> list[list[tuple[int, int]]]:
        if key not in groups:
            groups[key] = [[] for _ in range(cfg.n_input_packs)]
        return groups[key]

    for l in range(cfg.head_dim):
        for h in range(cfg.num_heads):
            for r in range(min(m, cfg.seq_len)):
                for t in range(dim):
                    tk = (t + l) % dim
                    row_k = ((r - l) % m + tk) % m
                    col_k = tk % cfg.head_dim
                    loc_k = _k_lower_slot_index(cfg, h, row_k, col_k)
                    if loc_k is None:
                        continue
                    k_out, idx_k = loc_k
                    prod_slot = _slot_index(cfg, 0, t, h)
                    rrot = (idx_k - prod_slot) % cfg.num_slots
                    out_t = t % cfg.seq_len
                    if out_t >= cfg.seq_len:
                        continue
                    row = (r + out_t) % m
                    u = (row - out_t) % cfg.seq_len
                    pack_i, j = u // cfg.pack, u % cfg.pack
                    score_idx = _slot_index(cfg, j, out_t, h)
                    plist = _pairs((l, k_out, rrot))
                    plist[pack_i].append((prod_slot, score_idx))
    return groups


def _slot_cc_score_scatter(
    k_lower: list[np.ndarray],
    q_copies: list[np.ndarray],
    cfg: ThorConfig,
    scatter_groups: dict[tuple[int, int, int], list[list[tuple[int, int]]]],
    scale: float,
) -> list[np.ndarray]:
    """K@Q^T 几何 BSGS：按 ``(l, k_out, rrot)`` 分组 ``rotate`` + place。"""
    score = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)]
    scale_pt = np.full(cfg.num_slots, scale)
    for (l, k_out, rrot), pack_lists in scatter_groups.items():
        prod = pt_mult(rotate_cc_left(k_lower[k_out], rrot), q_copies[l])
        for pack_i, plist in enumerate(pack_lists):
            if not plist:
                continue
            sp = score[pack_i]
            for prod_slot, score_idx in plist:
                sp[score_idx] += prod[prod_slot]
    return [pt_mult(scale_pt, s) for s in score]


def _build_cc_context_scatter_pairs(
    cfg: ThorConfig,
) -> dict[tuple[int, int, int], list[list[tuple[int, int]]]]:
    """
    编译期：纯实 Alpha@V 每项 ``prod=rotate(V,-rrot)*alpha_copy[n]`` 的散射表。

    键 ``(n, v_out, rrot)``；值 ``[ctx_pack -> [(prod_slot, ctx_idx), ...]]``。
    ``n = (key-query)%seq``（alpha 对角）；``rrot`` 为逐 term 几何对齐（非固定 bert ``rrot(n)``）。
    4 路 V pack，无 DualRail 4→2 打包。
    """
    n_out = cfg.n_out_packed_qkv
    s, pack, hd = cfg.num_slots, cfg.pack, cfg.head_dim
    seq, stride, hsp = cfg.seq_len, cfg.slot_stride, cfg.head_slot_pack
    groups: dict[tuple[int, int, int], list[list[tuple[int, int]]]] = {}

    for h in range(cfg.num_heads):
        for query in range(seq):
            prod_slot = query * hsp + h
            for col in range(hd):
                diag_c = (col - query) % hd
                c_out, jc = diag_c // pack, diag_c % pack
                idx_c = jc * stride + query * hsp + h
                for key in range(seq):
                    n = (key - query) % seq
                    diag_v = (col - key) % hd
                    v_out, jv = diag_v // pack, diag_v % pack
                    idx_v = jv * stride + key * hsp + h
                    rrot = (idx_v - prod_slot) % s
                    gkey = (n, v_out, rrot)
                    if gkey not in groups:
                        groups[gkey] = [[] for _ in range(n_out)]
                    groups[gkey][c_out].append((prod_slot, idx_c))
    return groups


def _slot_cc_context_scatter(
    v_upper: list[np.ndarray],
    alpha_copies: list[np.ndarray],
    cfg: ThorConfig,
    scatter_groups: dict[tuple[int, int, int], list[list[tuple[int, int]]]],
    scale: float = 1.0,
) -> list[np.ndarray]:
    """Alpha@V 几何 BSGS：按 ``(n, v_out, rrot)`` 分组 ``rotate`` + place。"""
    n_out = len(v_upper)
    ctx = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(n_out)]
    scale_pt = np.full(cfg.num_slots, scale)
    for (n, v_out, rrot), pack_lists in scatter_groups.items():
        prod = pt_mult(rotate_cc_left(v_upper[v_out], rrot), alpha_copies[n])
        for c_out, plist in enumerate(pack_lists):
            if not plist:
                continue
            sp = ctx[c_out]
            for prod_slot, ctx_idx in plist:
                sp[ctx_idx] += prod[prod_slot]
    return [pt_mult(scale_pt, c) for c in ctx]


def _build_ct_ct_masks(cfg: ThorConfig, n_in: int) -> dict:
    """THOR BSGS ``ct_ct`` 四分片 mask（按 ``n_in`` 预计算）。"""
    s, stride, p = cfg.num_slots, cfg.slot_stride, cfg.pack
    ct_ct: dict = {0: {}, 1: {}, 2: {}, 3: {}}
    for n in range(1, n_in):
        j = n % p
        a0 = np.ones(s, dtype=np.float64)
        a0[np.arange(s) % stride >= (stride - p * n)] = 0
        a1 = np.zeros(s, dtype=np.float64)
        a1[np.arange(s) % stride >= (stride - p * n)] = 1
        if j == 0:
            ct_ct[0][n] = a0
            ct_ct[1][n] = a1
        else:
            a0[: stride * j] = 0
            ct_ct[0][n] = a0.copy()
            a1[-stride:] = 0
            if j > 1:
                a1[: stride * (j - 1)] = 0
            ct_ct[1][n] = a1.copy()
            a2 = np.ones(s, dtype=np.float64)
            a2[np.arange(s) % stride >= (stride - p * n)] = 0
            a2[stride * j :] = 0
            ct_ct[2][n] = a2
            ct_ct[3][n] = np.ones(s, dtype=np.float64) - a0 - a1 - a2
    return ct_ct


def _acc_ttemp(ttemp, temp, ti: int, tj: int, si: int, sj: int, n_out: int) -> None:
    if ti >= n_out or si >= n_out:
        return
    src = temp[si, sj]
    if src is None:
        return
    if ttemp[ti, tj] is None:
        ttemp[ti, tj] = src
    else:
        ttemp[ti, tj] = add_vec(ttemp[ti, tj], src)


def _accumulate_ttemp_adaptive(ttemp, temp, q: int, j: int, n_out: int) -> None:
    """
    自适应 ttemp 路由（闭式推广 ``bert.py`` 手写表）。

    对任意 ``n_out``：
      delta = q + sj//2
      ti = (si + delta) % n_out
      tj = sj%2 + 2*((si+delta)//n_out % 2)
    已验证：完全覆盖 score(n_out=4) 与 context(n_out=2) 手写分支。
    """
    sj_range = (0, 1) if j == 0 else range(4)
    for si in range(n_out):
        for sj in sj_range:
            if temp[si, sj] is None:
                continue
            delta = q + sj // 2
            ti = (si + delta) % n_out
            wrap = (si + delta) // n_out
            tj = (sj % 2) + (2 if (wrap % 2) == 1 else 0)
            _acc_ttemp(ttemp, temp, ti, tj, si, sj, n_out)


def _merge_bsgs_real(
    ttemp: np.ndarray, n_out: int, cfg: ThorConfig, *, mode: Literal["score", "context"]
) -> list[np.ndarray]:
    """ttemp 末段 merge：score → 每 out 两 pack；context → 每 out 一 pack。"""
    stride = cfg.slot_stride
    z = np.zeros(cfg.num_slots, dtype=np.float64)
    output: list[np.ndarray] = []
    for out in range(n_out):
        c0 = ttemp[out, 0] if ttemp[out, 0] is not None else z
        c1 = rotate_left(ttemp[out, 1] if ttemp[out, 1] is not None else z, -stride)
        c2 = ttemp[out, 2] if ttemp[out, 2] is not None else z
        c3 = rotate_left(ttemp[out, 3] if ttemp[out, 3] is not None else z, -stride)
        if mode == "score":
            output.append(c0 + c1)
            output.append(c2 + c3)
        else:
            output.append(c0 + c1 + c2 + c3)
    return output


def _slot_cc_bsgs_ttemp(
    left_vecs: list[np.ndarray],
    right_copies: list[np.ndarray],
    cfg: ThorConfig,
    ct_ct: dict,
    scale: float,
    *,
    n_in: int,
    mode: Literal["score", "context"],
) -> list[np.ndarray]:
    """
    THOR 结构纯实 BSGS：固定 ``rrot(n)`` + ``ct_ct`` + 自适应 ttemp 路由。

    形状随 ``n_out=len(left)`` / ``n_in`` 变化；**不**保证纯实布局 ≡ NumPy。
    """
    n_out = len(left_vecs)
    stride, p = cfg.slot_stride, cfg.pack
    scale_pt = np.full(cfg.num_slots, scale)
    ttemp = np.full((n_out, 4), None, dtype=object)
    for out in range(n_out):
        ttemp[out, 0] = pt_mult(scale_pt, pt_mult(left_vecs[out], right_copies[0]))
    for n in range(1, n_in):
        j = n % p
        q = n // p
        rrot = (stride * j - p * n) % cfg.num_slots
        temp = np.full((n_out, 4), None, dtype=object)
        for out in range(n_out):
            prod = pt_mult(rotate_cc_left(left_vecs[out], rrot), right_copies[n])
            if j == 0:
                temp[out, 0] = pt_mult(ct_ct[0][n], prod)
                temp[out, 1] = prod - temp[out, 0]
            else:
                temp[out, 0] = pt_mult(ct_ct[0][n], prod)
                temp[out, 1] = pt_mult(ct_ct[1][n], prod)
                temp[out, 2] = pt_mult(ct_ct[2][n], prod)
                temp[out, 3] = prod - temp[out, 0] - temp[out, 1] - temp[out, 2]
        _accumulate_ttemp_adaptive(ttemp, temp, q, j, n_out)
    out = _merge_bsgs_real(ttemp, n_out, cfg, mode=mode)
    if mode == "score":
        while len(out) < cfg.n_input_packs:
            out.append(np.zeros(cfg.num_slots, dtype=np.float64))
        return out[: cfg.n_input_packs]
    return out



# ===========================================================================
# FFN（block_diag_2）与 layout 桥
# ===========================================================================

def _ff_input_rot_mask(cfg: ThorConfig) -> np.ndarray:
    """
    FF 输入旋转 mask：每个 ``ff_band_half`` 半区内清零 band ``>= ff_block_ll`` 的 padding。

    THOR 复数路径用 ``%16 >= 6``（6=hidden/seq，8=ff_pack/2）；纯实数双半区用
    ``%ff_pack % ff_band_half >= ff_block_ll``。
    """
    mask = np.ones(cfg.num_slots, dtype=np.float64)
    half = cfg.ff_band_half
    nb = cfg.ff_block_ll
    mask[np.arange(cfg.num_slots) % cfg.ff_pack % half >= nb] = 0.0
    return mask


def _build_ff_input_rots_from_roots(
    roots: list[np.ndarray],
    cfg: ThorConfig,
    *,
    dual_band: bool = False,
) -> list[np.ndarray]:
    """
    FF 根向量 → ``n_in_slot`` 路旋转副本。

    ``dual_band=False``（FC1 / 768 列）：THOR 复数等价式
    ``temp = mask*root + rotate(mask*root, -ff_pack/2)``。
    ``dual_band=True``（FC2 / 1536 列）：纯实数双半区已分别在 band ``d`` / ``d+8``，
    不再做 ``rotate(-8)`` 以免两半混叠。
    """
    ff_stride = _ff_stride(cfg)
    mask = _ff_input_rot_mask(cfg)
    rots: list[np.ndarray] = [None] * cfg.n_in_slot  # type: ignore
    for i, root in enumerate(roots):
        masked = root * mask
        if dual_band:
            temp = masked
        else:
            temp = masked + rotate_left(masked, -cfg.ff_pack // 2)
        base = i * cfg.ff_pack
        rots[base] = temp
        for j in range(1, cfg.ff_pack):
            rots[base + j] = rotate_left(rots[base + j - 1], ff_stride)
    return rots


def input_lower_to_ff_roots(
    input_lower: list[np.ndarray],
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """
    ``encode_input_lower_diagonals`` 布局 → FF ``block_diag_2`` 根向量（纯 slot 置换）。

    列块 ``d`` 对应 ``h % nb == d`` 的全部 head 行；与 ``decode_input_from_lower_diagonals``
    一致，取同列块 **最后写入** 的 head（``d + m*nb`` 中最大的 ``h < num_heads``）。
    """
    n_groups = cfg.n_in_slot // cfg.ff_pack
    roots = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(n_groups)]
    nb = cfg.n_hidden_col_blocks
    for gi in range(n_groups):
        for k in range(cfg.ff_pack):
            diag = gi * cfg.ff_pack + k
            if diag >= cfg.seq_len:
                continue
            pack_in = diag // cfg.pack
            j = diag % cfg.pack
            for t in range(cfg.seq_len):
                for d in range(nb):
                    h = d
                    while h + nb < cfg.num_heads:
                        h += nb
                    src = input_lower[pack_in][_slot_index(cfg, j, t, h)]
                    roots[gi][_ff_slot_index(cfg, k, t, d)] = src
    return roots


def ff_pc_output_to_input_lower_slots(
    ff_vecs: list[np.ndarray],
    cfg: ThorConfig,
    *,
    split: Literal["vertical", "horizontal"] = "horizontal",
) -> list[np.ndarray]:
    """
    FF PC 输出 slot（``decode_ff_pc_output`` 布局）→ ``input_lower`` 布局（纯 slot 置换）。

    ``horizontal``（FC2→残差）：与 ``decode_ff_pc_output(..., split='horizontal')`` 一致，
    对每个列块 ``d`` 取 ``band[d] + band[d+ff_pack/2]`` 再写入 ``input_lower``。
    """
    out = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)]
    nb = cfg.n_hidden_col_blocks
    band2 = cfg.ff_pack // 2
    for pack_i, vec in enumerate(ff_vecs):
        for j in range(cfg.ff_pack):
            diag = pack_i * cfg.ff_pack + j
            if diag >= cfg.seq_len:
                continue
            pack_in = diag // cfg.pack
            j_in = diag % cfg.pack
            for t in range(cfg.seq_len):
                for d in range(nb):
                    if split == "horizontal":
                        src = (
                            vec[_ff_slot_index(cfg, j, t, d)]
                            + vec[_ff_slot_index(cfg, j, t, d + band2)]
                        )
                        for h in range(cfg.num_heads):
                            if h % nb != d:
                                continue
                            out[pack_in][_slot_index(cfg, j_in, t, h)] = src
                    else:
                        src = vec[_ff_slot_index(cfg, j, t, d)]
                        for h in range(cfg.num_heads):
                            if h % nb != d:
                                continue
                            out[pack_in][_slot_index(cfg, j_in, t, h)] = src
                        col1 = (d + nb) * cfg.seq_len + (diag + t) % cfg.seq_len
                        if col1 < cfg.hidden_dim:
                            src2 = vec[_ff_slot_index(cfg, j, t, d + band2)]
                            h1 = col1 // cfg.head_dim
                            out[pack_in][_slot_index(cfg, j_in, t, h1)] = src2
    return out


def pc_ctx_vecs_to_input_lower_slots(
    pc_vecs: list[np.ndarray],
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """
    PC-MM context 输出 slot → ``encode_input_lower_diagonals`` 布局（纯 slot 索引置换）。

    PC 输出按 ``n_hidden_row_blocks``（每头 ``head_dim`` 行）打包；W_O 输入按
    ``n_hidden_col_blocks``（每块 ``seq_len`` 行）打包。逐 slot 建立索引映射，不重编码矩阵。
    """
    out = [np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)]
    for pack_in in range(cfg.n_input_packs):
        for j_in in range(cfg.pack):
            diag_in = pack_in * cfg.pack + j_in
            if diag_in >= cfg.seq_len:
                continue
            for t in range(cfg.seq_len):
                for h_in in range(cfg.num_heads):
                    b = h_in % cfg.n_hidden_col_blocks
                    row_in_b = (diag_in + t) % cfg.seq_len
                    d = b * cfg.seq_len + row_in_b
                    if d >= cfg.hidden_dim:
                        continue
                    h_pc = d // cfg.head_dim
                    hd = d % cfg.head_dim
                    diag_pc = (hd - t) % cfg.head_dim
                    pack_pc = diag_pc // cfg.pack
                    j_pc = diag_pc % cfg.pack
                    if pack_pc >= len(pc_vecs):
                        continue
                    src = pc_vecs[pack_pc][_slot_index(cfg, j_pc, t, h_pc)]
                    out[pack_in][_slot_index(cfg, j_in, t, h_in)] = src
    return out


def pc_output_to_input_lower_slots(
    pc_vecs: list[np.ndarray],
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """PC-MM 输出（QKV / context / W_O）→ ``input_lower`` 布局。"""
    return pc_ctx_vecs_to_input_lower_slots(pc_vecs, cfg)


def slot_vec_add(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """同 layout 的两路 slot 向量逐 slot 相加（THOR ``engine.add`` / 残差）。"""
    if len(a) != len(b):
        raise ValueError(f"slot 向量数不一致: {len(a)} vs {len(b)}")
    return [a[i] + b[i] for i in range(len(a))]


def slot_vecs_pass_through(vecs: list[np.ndarray]) -> list[np.ndarray]:
    """非线性占位（LayerNorm / GELU）：layout 不变，值原样传递。"""
    return [v.copy() for v in vecs]


def _use_ff_block_diag2(cfg: ThorConfig) -> bool:
    """
    FF 是否可走 ``block_diag_2``（由派生几何判定，非 BERT-base 特判）。

    需 ``hidden/seq <= ff_pack/2`` 且 ``ff_stride == seq * ff_pack``（slot 布局不重叠）。
    """
    try:
        fp = cfg.ff_pack
        stride = cfg.ff_stride
    except ValueError:
        return False
    half = fp // 2
    return (
        cfg.n_hidden_col_blocks <= half
        and cfg.seq_len % fp == 0
        and stride == cfg.seq_len * fp
        and cfg.n_in_slot == cfg.seq_len
    )


def compute_fc1_from_encoded(
    input_lower: list[np.ndarray],
    w1_pts: np.ndarray,
    cfg: ThorConfig,
) -> list[list[np.ndarray]]:
    """``input_lower`` + 预编码 ``w1`` → FC1 两路 slot。"""
    if not _use_ff_block_diag2(cfg):
        raise ValueError("compute_fc1_from_encoded 需 block_diag_2 几何")
    x_rots = _build_ff_input_rots_from_roots(
        input_lower_to_ff_roots(input_lower, cfg), cfg
    )
    return [
        list(pt_ct_matmul(w1_pts[rep], x_rots, cfg, mode="block_diag_2"))
        for rep in range(2)
    ]


def compute_fc2_from_encoded(
    fc1_vecs: list[list[np.ndarray]],
    w2_pts: np.ndarray,
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """GELU 后 FC1 两路 + 预编码 ``w2`` → 合并 slot。"""
    if not _use_ff_block_diag2(cfg):
        raise ValueError("compute_fc2_from_encoded 需 block_diag_2 几何")
    wx_reps: list[list[np.ndarray]] = []
    for rep in range(2):
        x_rots = _build_ff_input_rots_from_roots(fc1_vecs[rep], cfg, dual_band=True)
        wx_reps.append(list(pt_ct_matmul(w2_pts[rep], x_rots, cfg, mode="block_diag_2")))
    n_out_p = cfg.seq_len // cfg.ff_pack
    return [wx_reps[0][out_i] + wx_reps[1][out_i] for out_i in range(n_out_p)]


def compute_fc1_slots(
    ff_roots: list[np.ndarray],
    w1: np.ndarray,
    cfg: ThorConfig,
) -> list[list[np.ndarray]]:
    """FC1：FF 根向量 → ``block_diag_2`` PC-MM；返回两路输出 slot（各 ``n_out_p`` 向量）。"""
    if not _use_ff_block_diag2(cfg):
        raise ValueError("compute_fc1_slots 需 block_diag_2 几何")
    x_rots = _build_ff_input_rots_from_roots(ff_roots, cfg)
    w_pts = encode_ff_weight(w1, cfg, split="vertical", n_splits=cfg.ff_n_splits, scale=1.0)
    return [
        list(pt_ct_matmul(w_pts[rep], x_rots, cfg, mode="block_diag_2"))
        for rep in range(2)
    ]


def compute_fc1_from_input_lower(
    input_lower: list[np.ndarray],
    w1: np.ndarray,
    cfg: ThorConfig,
) -> list[list[np.ndarray]]:
    """``input_lower`` → slot 桥 → FC1 输出 slot（算法路径，无矩阵 encode）。"""
    return compute_fc1_slots(input_lower_to_ff_roots(input_lower, cfg), w1, cfg)


def compute_fc2_slots(
    fc1_vecs: list[list[np.ndarray]],
    w2: np.ndarray,
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """
    FC1 输出 slot（两路）→ GELU 占位后 → FC2 ``block_diag_2`` → 合并 slot 向量。

    ``fc1_vecs[rep]`` 为 ``(seq, ffn/2)`` 的 FF 输出布局，可直接作 FC2 输入根向量。
    """
    if not _use_ff_block_diag2(cfg):
        raise ValueError("compute_fc2_slots 需 block_diag_2 几何")
    w_pts = encode_ff_weight(w2, cfg, split="horizontal", n_splits=cfg.ff_n_splits, scale=1.0)
    wx_reps: list[list[np.ndarray]] = []
    for rep in range(2):
        x_rots = _build_ff_input_rots_from_roots(fc1_vecs[rep], cfg, dual_band=True)
        wx_reps.append(list(pt_ct_matmul(w_pts[rep], x_rots, cfg, mode="block_diag_2")))
    n_out_p = cfg.seq_len // cfg.ff_pack
    return [wx_reps[0][out_i] + wx_reps[1][out_i] for out_i in range(n_out_p)]



# ===========================================================================
# MaskBank：编译期表（几何 BSGS pairs + ttemp ct_ct）
# ===========================================================================


@dataclass
class MaskBank:
    """预计算几何 scatter 表与 ttemp ``ct_ct`` mask。"""

    cfg: ThorConfig
    cc_score_scatter_pairs: dict | None = field(default=None, repr=False)
    cc_context_scatter_pairs: dict | None = field(default=None, repr=False)
    ct_ct_score: dict | None = field(default=None, repr=False)
    ct_ct_context: dict | None = field(default=None, repr=False)
    build: bool = True

    def __post_init__(self) -> None:
        if not self.build:
            return
        cfg = self.cfg
        self.cc_score_scatter_pairs = _build_cc_score_scatter_pairs(cfg)
        self.cc_context_scatter_pairs = _build_cc_context_scatter_pairs(cfg)
        self.ct_ct_score = _build_ct_ct_masks(cfg, cfg.head_dim)
        self.ct_ct_context = _build_ct_ct_masks(cfg, cfg.n_in_slot)

    @classmethod
    def from_precomputed(
        cls,
        cfg: ThorConfig,
        *,
        cc_score_scatter_pairs: dict,
        cc_context_scatter_pairs: dict,
        ct_ct_score: dict | None = None,
        ct_ct_context: dict | None = None,
    ) -> "MaskBank":
        """从磁盘缓存恢复，跳过编译期建表。"""
        return cls(
            cfg=cfg,
            cc_score_scatter_pairs=cc_score_scatter_pairs,
            cc_context_scatter_pairs=cc_context_scatter_pairs,
            ct_ct_score=ct_ct_score,
            ct_ct_context=ct_ct_context,
            build=False,
        )


def pc_matmul_from_encoded(
    x_in_lower: list[np.ndarray],
    w_pts: np.ndarray,
    cfg: ThorConfig,
    *,
    mode: Literal["block_diag_1", "block_diag_2"] = "block_diag_1",
) -> list[np.ndarray]:
    """已编码权重 plaintext × input_lower 旋转副本 → PC-MM 输出。"""
    x_rots = make_rotated_copies(x_in_lower, cfg)
    return list(pt_ct_matmul(w_pts, x_rots, cfg, mode=mode))


# ===========================================================================
# Attention：Score (K@Q^T) + Context (Alpha@V) + W_O
# ===========================================================================


def _qkv_pc_from_enc(
    x_in_lower: list[np.ndarray],
    w: np.ndarray,
    cfg: ThorConfig,
    scale: float = 1.0,
) -> list[np.ndarray]:
    """单路 Q/K/V：input_lower × W → PC 上对角 pack（纯 slot，不 decode）。"""
    w_pts = encode_weight_upper_diagonals(
        w, cfg, cfg.n_in_slot, cfg.head_dim, (cfg.head_dim, cfg.seq_len), scale=scale
    )
    x_rots = make_rotated_copies(x_in_lower, cfg)
    return list(pt_ct_matmul(w_pts, x_rots, cfg, mode="block_diag_1"))


def compute_attention_score_slots(
    q_merged: list[np.ndarray],
    k_merged: list[np.ndarray],
    cfg: ThorConfig,
    masks: MaskBank,
    scale: float | None = None,
    *,
    backend: Literal["geometric", "ttemp"] = "geometric",
) -> list[np.ndarray]:
    """
    Score：transpose(K) + make_copies(Q) + 纯实 BSGS → L(S^T)。

    - ``backend=\"geometric\"``（默认）：几何 rrot BSGS，≡ NumPy。
    - ``backend=\"ttemp\"``：固定 rrot + 自适应 ttemp（THOR 结构；纯实下不保证 ≡ NumPy）。
    """
    if scale is None:
        scale = 1.0 / np.sqrt(cfg.head_dim)
    k_lower = transpose_upper_to_lower(k_merged, cfg)
    q_copies = make_copies_real(q_merged, cfg)
    if backend == "ttemp":
        if masks.ct_ct_score is None:
            masks.ct_ct_score = _build_ct_ct_masks(cfg, cfg.head_dim)
        return _slot_cc_bsgs_ttemp(
            k_lower,
            q_copies,
            cfg,
            masks.ct_ct_score,
            scale,
            n_in=cfg.head_dim,
            mode="score",
        )
    if masks.cc_score_scatter_pairs is None:
        masks.cc_score_scatter_pairs = _build_cc_score_scatter_pairs(cfg)
    return _slot_cc_score_scatter(
        k_lower, q_copies, cfg, masks.cc_score_scatter_pairs, scale
    )


def slot_softmax(score_vecs: list[np.ndarray], cfg: ThorConfig) -> list[np.ndarray]:
    """
    Softmax（layout 不变）。明文 exp；接 HE 时换成多项式近似，勿改索引。
    """
    alpha_vecs = [v.copy() for v in score_vecs]
    for h in range(cfg.num_heads):
        for query in range(cfg.seq_len):
            logits = np.empty(cfg.seq_len, dtype=np.float64)
            for key in range(cfg.seq_len):
                u = (key - query) % cfg.seq_len
                pack_i, j = u // cfg.pack, u % cfg.pack
                logits[key] = score_vecs[pack_i][_slot_index(cfg, j, query, h)]
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs = probs / probs.sum()
            for key in range(cfg.seq_len):
                u = (key - query) % cfg.seq_len
                pack_i, j = u // cfg.pack, u % cfg.pack
                alpha_vecs[pack_i][_slot_index(cfg, j, query, h)] = probs[key]
    return alpha_vecs


def compute_attention_context_slots(
    score_vecs: list[np.ndarray],
    v_merged: list[np.ndarray],
    cfg: ThorConfig,
    masks: MaskBank,
    *,
    apply_softmax: bool = True,
    backend: Literal["geometric", "ttemp"] = "geometric",
    softmax_impl: Literal["exact", "he"] = "exact",
    softmax_he_params=None,
    softmax_mask_packs: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """
    Alpha@V：softmax(可选) + 纯实 BSGS。

    - ``geometric``（默认）：≡ index；``ttemp``：自适应路由 THOR 结构（纯实不保证 ≡）。
    - ``softmax_impl="exact"``：明文 ``exp`` softmax（线性对拍用）。
    - ``softmax_impl="he"``：``slot_softmax_he``（thor exp + aSOR，参数通用）。
    """
    if not apply_softmax:
        alpha = slot_vecs_pass_through(score_vecs)
    elif softmax_impl == "he":
        from slot_softmax_he import SoftmaxHeParams, slot_softmax_he

        if softmax_he_params is None:
            raise ValueError('softmax_impl="he" 需要 softmax_he_params')
        if softmax_mask_packs is None:
            raise ValueError('softmax_impl="he" 需要 softmax_mask_packs')
        if not isinstance(softmax_he_params, SoftmaxHeParams):
            raise TypeError("softmax_he_params 须为 SoftmaxHeParams")
        alpha = slot_softmax_he(
            score_vecs, softmax_mask_packs, cfg, softmax_he_params
        )
    else:
        alpha = slot_softmax(score_vecs, cfg)
    if backend == "ttemp":
        if masks.ct_ct_context is None:
            masks.ct_ct_context = _build_ct_ct_masks(cfg, cfg.n_in_slot)
        a_copies = make_copies(alpha, cfg)
        return _slot_cc_bsgs_ttemp(
            v_merged,
            a_copies,
            cfg,
            masks.ct_ct_context,
            1.0,
            n_in=cfg.n_in_slot,
            mode="context",
        )
    if masks.cc_context_scatter_pairs is None:
        masks.cc_context_scatter_pairs = _build_cc_context_scatter_pairs(cfg)
    a_copies = make_copies(alpha, cfg)
    return _slot_cc_context_scatter(
        v_merged, a_copies, cfg, masks.cc_context_scatter_pairs, 1.0
    )


def compute_output_projection_slots(
    ctx_pc_vecs: list[np.ndarray],
    w_o: np.ndarray,
    cfg: ThorConfig,
) -> list[np.ndarray]:
    """W_O：context PC → input_lower 置换 → PC-MM。"""
    x_enc = pc_ctx_vecs_to_input_lower_slots(ctx_pc_vecs, cfg)
    w_pts = encode_weight_upper_diagonals(
        w_o, cfg, cfg.n_in_slot, cfg.seq_len, (cfg.head_dim, cfg.seq_len)
    )
    x_rots = make_rotated_copies(x_enc, cfg)
    return list(pt_ct_matmul(w_pts, x_rots, cfg, mode="block_diag_1"))


# ===========================================================================
# 单层 / 多层 Encoder（线性段入口）
# ===========================================================================


def run_encoder_layer(
    x_in_lower: list[np.ndarray],
    weights: dict[str, np.ndarray],
    cfg: ThorConfig,
    masks: MaskBank | None = None,
    *,
    apply_softmax: bool = True,
    cc_backend: Literal["geometric", "ttemp"] = "geometric",
    softmax_impl: Literal["exact", "he"] = "exact",
    softmax_he_params=None,
    softmax_mask_packs: list[np.ndarray] | None = None,
    gelu_impl: Literal["none", "he"] = "none",
    gelu_he_params=None,
    ln_impl: Literal["none", "he"] = "none",
    ln1_he_params=None,
    ln2_he_params=None,
    ln1_gamma_beta: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    ln2_gamma_beta: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    ln_eps: float = 1e-12,
) -> list[np.ndarray]:
    """
    单层 Encoder。进出均为 ``input_lower`` layout。

    weights 键：``w_q, w_k, w_v, w_o, w1, w2``（原始矩阵；内部会编码）。
    非线性：``softmax_impl`` / ``gelu_impl`` / ``ln_impl``；``he`` 需对应 params。
    """
    if not encoder_linear_applies(cfg):
        raise ValueError(
            "当前 cfg 不满足线性核心几何（validate + FF block_diag_2）。"
            "可用 bert_base / bert_medium(512/8/2048) / bert_large，或调整 seq/hidden/heads/slots。"
        )
    if masks is None:
        masks = MaskBank(cfg)
    cfg.validate()

    q_merged = _qkv_pc_from_enc(x_in_lower, weights["w_q"], cfg)
    k_merged = _qkv_pc_from_enc(x_in_lower, weights["w_k"], cfg)
    v_merged = _qkv_pc_from_enc(x_in_lower, weights["w_v"], cfg)

    score = compute_attention_score_slots(
        q_merged, k_merged, cfg, masks, backend=cc_backend
    )
    ctx = compute_attention_context_slots(
        score,
        v_merged,
        cfg,
        masks,
        apply_softmax=apply_softmax,
        backend=cc_backend,
        softmax_impl=softmax_impl,
        softmax_he_params=softmax_he_params,
        softmax_mask_packs=softmax_mask_packs,
    )
    wo_pc = compute_output_projection_slots(ctx, weights["w_o"], cfg)

    wo_in = pc_output_to_input_lower_slots(wo_pc, cfg)
    attn_res = slot_vec_add(x_in_lower, wo_in)
    if ln_impl == "he":
        from slot_layernorm_he import LayerNormHeParams, slot_layernorm_he

        if ln1_he_params is None or ln1_gamma_beta is None:
            raise ValueError('ln_impl="he" 需要 ln1_he_params 与 ln1_gamma_beta')
        if not isinstance(ln1_he_params, LayerNormHeParams):
            raise TypeError("ln1_he_params 须为 LayerNormHeParams")
        g1, b1 = ln1_gamma_beta
        attn_res = slot_layernorm_he(
            attn_res, g1, b1, cfg, ln1_he_params, eps=ln_eps
        )
    else:
        attn_res = slot_vecs_pass_through(attn_res)

    fc1 = compute_fc1_from_input_lower(attn_res, weights["w1"], cfg)
    if gelu_impl == "he":
        from slot_gelu_he import GeluHeParams, slot_gelu_he

        if gelu_he_params is None:
            raise ValueError('gelu_impl="he" 需要 gelu_he_params')
        if not isinstance(gelu_he_params, GeluHeParams):
            raise TypeError("gelu_he_params 须为 GeluHeParams")
        fc1 = slot_gelu_he(fc1, gelu_he_params)  # type: ignore[assignment]
    else:
        fc1 = [slot_vecs_pass_through(rep) for rep in fc1]
    fc2 = compute_fc2_slots(fc1, weights["w2"], cfg)

    ff_in = ff_pc_output_to_input_lower_slots(fc2, cfg)
    out = slot_vec_add(attn_res, ff_in)
    if ln_impl == "he":
        from slot_layernorm_he import LayerNormHeParams, slot_layernorm_he

        if ln2_he_params is None or ln2_gamma_beta is None:
            raise ValueError('ln_impl="he" 需要 ln2_he_params 与 ln2_gamma_beta')
        if not isinstance(ln2_he_params, LayerNormHeParams):
            raise TypeError("ln2_he_params 须为 LayerNormHeParams")
        g2, b2 = ln2_gamma_beta
        return slot_layernorm_he(out, g2, b2, cfg, ln2_he_params, eps=ln_eps)
    return slot_vecs_pass_through(out)


def run_encoder_layer_he_cached(
    x_in_lower: list[np.ndarray],
    encoded: dict,
    cfg: ThorConfig,
    masks: MaskBank,
    *,
    softmax_he_params,
    softmax_mask_packs: list[np.ndarray],
    gelu_he_params,
    ln1_he_params,
    ln2_he_params,
    ln1_gamma_beta: tuple[list[np.ndarray], list[np.ndarray]],
    ln2_gamma_beta: tuple[list[np.ndarray], list[np.ndarray]],
    ln_eps: float = 1e-12,
    cc_backend: Literal["geometric", "ttemp"] = "geometric",
) -> list[np.ndarray]:
    """
    完整 HE 同构单层：预编码权重 + Softmax/GeLU/LN1/LN2 全部 ``he``。

    ``encoded`` 键：``w_q, w_k, w_v, w_o, w1, w2``。
    """
    from slot_gelu_he import slot_gelu_he
    from slot_layernorm_he import slot_layernorm_he

    cfg.validate()
    q = pc_matmul_from_encoded(x_in_lower, encoded["w_q"], cfg)
    k = pc_matmul_from_encoded(x_in_lower, encoded["w_k"], cfg)
    v = pc_matmul_from_encoded(x_in_lower, encoded["w_v"], cfg)
    score = compute_attention_score_slots(q, k, cfg, masks, backend=cc_backend)
    ctx = compute_attention_context_slots(
        score,
        v,
        cfg,
        masks,
        apply_softmax=True,
        backend=cc_backend,
        softmax_impl="he",
        softmax_he_params=softmax_he_params,
        softmax_mask_packs=softmax_mask_packs,
    )
    # W_O：context → input_lower → PC-MM（与 compute_output_projection_slots 一致，用缓存权重）
    x_wo = pc_ctx_vecs_to_input_lower_slots(ctx, cfg)
    wo = pc_matmul_from_encoded(x_wo, encoded["w_o"], cfg)
    wo_in = pc_output_to_input_lower_slots(wo, cfg)
    attn_res = slot_layernorm_he(
        slot_vec_add(x_in_lower, wo_in),
        ln1_gamma_beta[0],
        ln1_gamma_beta[1],
        cfg,
        ln1_he_params,
        eps=ln_eps,
    )
    fc1 = compute_fc1_from_encoded(attn_res, encoded["w1"], cfg)
    fc1 = slot_gelu_he(fc1, gelu_he_params)  # type: ignore[assignment]
    fc2 = compute_fc2_from_encoded(fc1, encoded["w2"], cfg)  # type: ignore[arg-type]
    ff_in = ff_pc_output_to_input_lower_slots(fc2, cfg)
    return slot_layernorm_he(
        slot_vec_add(attn_res, ff_in),
        ln2_gamma_beta[0],
        ln2_gamma_beta[1],
        cfg,
        ln2_he_params,
        eps=ln_eps,
    )


def run_encoder_stack_he_cached(
    x_in_lower: list[np.ndarray],
    layer_payloads: list[dict],
    cfg: ThorConfig,
    masks: MaskBank,
    *,
    softmax_mask_packs: list[np.ndarray],
    ln_eps: float = 1e-12,
    cc_backend: Literal["geometric", "ttemp"] = "geometric",
) -> list[np.ndarray]:
    """
    多层 HE 同构 stack：每层输出 ``input_lower`` 直接作为下一层输入。

    ``layer_payloads[i]`` 键：
      ``encoded``, ``softmax_he_params``, ``gelu_he_params``,
      ``ln1_he_params``, ``ln2_he_params``, ``ln1_gamma_beta``, ``ln2_gamma_beta``
    ``softmax_mask_packs`` 全层共用（key 有效位与序列位置相关）。
    """
    x = x_in_lower
    for i, pay in enumerate(layer_payloads):
        x = run_encoder_layer_he_cached(
            x,
            pay["encoded"],
            cfg,
            masks,
            softmax_he_params=pay["softmax_he_params"],
            softmax_mask_packs=softmax_mask_packs,
            gelu_he_params=pay["gelu_he_params"],
            ln1_he_params=pay["ln1_he_params"],
            ln2_he_params=pay["ln2_he_params"],
            ln1_gamma_beta=pay["ln1_gamma_beta"],
            ln2_gamma_beta=pay["ln2_gamma_beta"],
            ln_eps=ln_eps,
            cc_backend=cc_backend,
        )
    return x


def run_encoder_stack(
    x_in_lower: list[np.ndarray],
    layer_weights: list[dict[str, np.ndarray]],
    cfg: ThorConfig,
    masks: MaskBank | None = None,
    *,
    apply_softmax: bool = True,
    cc_backend: Literal["geometric", "ttemp"] = "geometric",
) -> list[np.ndarray]:
    """堆叠多层；每层输出 layout = 输入 layout。"""
    if masks is None:
        masks = MaskBank(cfg)
    x = x_in_lower
    for w in layer_weights:
        x = run_encoder_layer(
            x,
            w,
            cfg,
            masks,
            apply_softmax=apply_softmax,
            cc_backend=cc_backend,
        )
    return x


    __all__ = [
        "ThorConfig",
        "MaskBank",
        "standard_bert",
        "bert_base",
        "bert_medium",
        "bert_large",
        "mini_bert",
        "path_b_applies",
        "encoder_linear_applies",
        "encode_input_lower_diagonals",
        "run_encoder_layer",
        "run_encoder_stack",
        "compute_attention_score_slots",
        "compute_attention_context_slots",
        "compute_output_projection_slots",
        "slot_softmax",
        "slot_vecs_pass_through",
        "_accumulate_ttemp_adaptive",
        "_slot_cc_bsgs_ttemp",
    ]
