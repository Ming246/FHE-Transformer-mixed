"""
Slot 布局 decode / 简单 NumPy 辅助（验证与 smoke 用）。

权威运行路径在 ``thor_encoder_linear_core``；本模块只还原矩阵并做 head 拆合。
"""
from __future__ import annotations

import numpy as np

from thor_encoder_linear_core import ThorConfig, _slot_index


def decode_input_from_lower_diagonals(
    vectors: list[np.ndarray], cfg: ThorConfig
) -> np.ndarray:
    """从打包 slot 向量还原 X (seq, hidden)。"""
    x_t = np.zeros((cfg.hidden_dim, cfg.seq_len), dtype=np.float64)
    x_blocks = np.vsplit(x_t, cfg.n_hidden_col_blocks)
    n_packs = len(vectors)

    for pack_i in range(n_packs):
        vec = vectors[pack_i]
        for j in range(cfg.pack):
            diag_index = pack_i * cfg.pack + j
            for t in range(cfg.seq_len):
                for h in range(cfg.num_heads):
                    x_b = x_blocks[h % cfg.n_hidden_col_blocks]
                    idx = _slot_index(cfg, j, t, h)
                    x_b[(diag_index + t) % x_b.shape[0], t] = vec[idx]
    return x_t.T


def split_heads(x: np.ndarray, cfg: ThorConfig) -> np.ndarray:
    """(seq, hidden) -> (num_heads, seq, head_dim)"""
    return x.reshape(cfg.seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)


def merge_heads(x: np.ndarray, cfg: ThorConfig) -> np.ndarray:
    """(num_heads, seq, head_dim) -> (seq, hidden)"""
    return x.transpose(1, 0, 2).reshape(cfg.seq_len, cfg.hidden_dim)


def attention_context_ref_from_scores(
    scores: np.ndarray,
    v: np.ndarray,
    cfg: ThorConfig,
    *,
    apply_softmax: bool = True,
) -> np.ndarray:
    """
    NumPy 参考：``scores[h, key, query]`` → context (seq, hidden)。

    ``apply_softmax=False``：直接用 raw score 作权重。
    """
    if apply_softmax:
        alpha = np.zeros_like(scores)
        for h in range(cfg.num_heads):
            for t in range(cfg.seq_len):
                sl = scores[h, :, t]
                sl = sl - sl.max()
                e = np.exp(sl)
                alpha[h, :, t] = e / e.sum()
    else:
        alpha = scores
    v_h = split_heads(v, cfg)
    ctx_heads = np.stack(
        [alpha[h].T @ v_h[h] for h in range(cfg.num_heads)], axis=0
    )
    return merge_heads(ctx_heads, cfg)
