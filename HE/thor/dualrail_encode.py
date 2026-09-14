"""DualRail message packing for THOR complex PC-MM (no Liberate calls)."""
from __future__ import annotations

import numpy as np

try:
    from .path_setup import ensure_thor_path
except ImportError:
    from path_setup import ensure_thor_path

ensure_thor_path()

from thor.utils.matrix import ld_entry, to_blocks, ud_entry


def pack_embedding_dualrail(embedding: np.ndarray) -> list[np.ndarray]:
    """
    Match ``ThorDataEncryptor.encrypt_embedding`` messages (4 complex vectors).

    embedding: (128, 768)
    """
    if embedding.shape != (128, 768):
        raise ValueError(f"expected (128, 768), got {embedding.shape}")
    x_t = embedding.T
    x_blocks = np.vsplit(x_t, 6)
    msgs: list[np.ndarray] = []
    for i in range(4):
        msg = np.zeros((2**15,), dtype=np.complex128)
        for j in range(16):
            temp = j * (2**11)
            l = i * 16 + j
            for t in range(128):
                for b in range(12):
                    x_b = x_blocks[b % 6]
                    msg[temp + t * 16 + b] = complex(
                        ld_entry(x_b, l, t), ld_entry(x_b, l + 64, t)
                    )
        msgs.append(msg)
    return msgs


def pack_weight_att_dualrail(
    w: np.ndarray,
    *,
    n_in: int,
    n_out: int,
    b_shape: tuple[int, int],
    scale: float = 1.0,
    out_packs: int | None = None,
) -> np.ndarray:
    """
    Match ``ThorModelEncoder._encode_w_att`` messages (complex DualRail).

    Returns object array shape ``(n_out_p, ll, n_in//2)`` of complex slot vectors.
    ``/2`` is baked into the message (replaces CT-side mult_scalar unpack).
    """
    if w.shape != (768, 768):
        raise ValueError(f"expected W (768, 768), got {w.shape}")
    pack = 16
    dim = 128
    n_in_c = n_in // 2
    ld_blocks, (ll, _) = to_blocks(w, b_shape, diag=True)
    n_out_p = n_out // pack
    if out_packs is not None:
        if not (1 <= out_packs <= n_out_p):
            raise ValueError(f"out_packs must be in 1..{n_out_p}")
        n_out_p = out_packs

    msgs = np.full((n_out_p, ll, n_in_c), None, dtype=object)
    for l in range(ll):
        diagonal = ld_blocks[l]
        for n in range(n_in_c):
            for out in range(n_out_p):
                msg = np.zeros((2**15,), dtype=np.complex128)
                for j in range(pack):
                    i = ((n // 16) * 16 + out * 16 + (n + j) % 16) % n_in_c
                    r = out * 16 + j
                    i = (i - r) % n_in
                    temp = j * (2**11)
                    for t in range(dim):
                        for d in range(12):
                            block = diagonal[d]
                            re = (scale * ud_entry(block, i, t, r)) / 2
                            im = -(
                                scale
                                * ud_entry(block, (i + n_in_c) % n_in, t, r)
                            ) / 2
                            msg[temp + t * 16 + d] = complex(re, im)
                msgs[out, l, n] = msg
    return msgs


def pack_weight_qkv_dualrail(
    w: np.ndarray,
    *,
    scale: float = 1.0,
    out_packs: int | None = None,
) -> np.ndarray:
    """QKV DualRail: ``n_in=128, n_out=64, b_shape=(64, 128)``."""
    return pack_weight_att_dualrail(
        w,
        n_in=128,
        n_out=64,
        b_shape=(64, 128),
        scale=scale,
        out_packs=out_packs,
    )


def pack_weight_wo_dualrail(
    w: np.ndarray,
    *,
    scale: float = 1.0,
    out_packs: int | None = None,
) -> np.ndarray:
    """
    Attention output (W_O) DualRail: ``n_in=64, n_out=128, b_shape=(128, 64)``.

    Input is 2 complex context CTs (``make_rotated_copies`` → 32); output is 8
    complex packs before fold/+conj (THOR ``ThorBertAttention.dense``).
    """
    return pack_weight_att_dualrail(
        w,
        n_in=64,
        n_out=128,
        b_shape=(128, 64),
        scale=scale,
        out_packs=out_packs,
    )


def pack_weight_ff_dualrail(
    w: np.ndarray,
    *,
    split: str,
    n_splits: int = 4,
    scale: float = 1.0,
    out_packs: int | None = None,
) -> np.ndarray:
    """
    Match ``ThorModelEncoder._encode_w_ff`` messages (DualRail ``block_diag_2``).

    ``split='vertical'`` → FC1 ``(3072,768)`` via ``vsplit``;
    ``split='horizontal'`` → FC2 ``(768,3072)`` via ``hsplit``.

    Returns object array shape ``(2, n_out_p, ll, n_in_c)`` with ``n_in_c=64``.
    ``/2`` baked into messages. Linear gate uses ``scale=1`` (production GeLU
    path bakes ``1/C`` into FC1 only).
    """
    n_in, n_out, b_shape = 128, 128, (128, 128)
    pack = 16
    dim = 128
    n_in_c = n_in // 2
    if split == "vertical":
        if w.shape != (3072, 768):
            raise ValueError(f"FC1 W expected (3072, 768), got {w.shape}")
        w_list = np.vsplit(w, n_splits)
    elif split == "horizontal":
        if w.shape != (768, 3072):
            raise ValueError(f"FC2 W expected (768, 3072), got {w.shape}")
        w_list = np.hsplit(w, n_splits)
    else:
        raise ValueError(f"split must be vertical|horizontal, got {split!r}")

    ld_blocks_list = [to_blocks(part, b_shape, diag=True)[0] for part in w_list]
    ll = ld_blocks_list[0].shape[0]
    n_out_p = n_out // pack
    if out_packs is not None:
        if not (1 <= out_packs <= n_out_p):
            raise ValueError(f"out_packs must be in 1..{n_out_p}")
        n_out_p = out_packs

    msgs = np.full((2, n_out_p, ll, n_in_c), None, dtype=object)
    for rep in range(2):
        for l in range(ll):
            for n in range(n_in_c):
                for out in range(n_out_p):
                    msg = np.zeros((2**15,), dtype=np.complex128)
                    for j in range(pack):
                        i = ((n // 16) * 16 + out * 16 + (n + j) % 16) % n_in_c
                        r = out * 16 + j
                        i = (i - r) % n_in
                        temp = j * (2**11)
                        for t in range(dim):
                            for d in range(6):
                                block1 = ld_blocks_list[rep * 2 + 0][l, d]
                                block2 = ld_blocks_list[rep * 2 + 1][l, d]
                                re1 = (scale * ud_entry(block1, i, t, r)) / 2
                                im1 = -(
                                    scale
                                    * ud_entry(block1, (i + n_in_c) % n_in, t, r)
                                ) / 2
                                re2 = (scale * ud_entry(block2, i, t, r)) / 2
                                im2 = -(
                                    scale
                                    * ud_entry(block2, (i + n_in_c) % n_in, t, r)
                                ) / 2
                                msg[temp + t * 16 + d] = complex(re1, im1)
                                msg[temp + t * 16 + d + 8] = complex(re2, im2)
                    msgs[rep, out, l, n] = msg
    return msgs


def block_diag1_mask_msg(delta: int, num_slots: int = 2**15) -> np.ndarray:
    """Internal-rotation mask for ``mode='block_diag_1'`` (THOR pre_encode_masks)."""
    arr = np.ones((num_slots,), dtype=np.float64)
    arr[np.arange(num_slots) % 16 >= delta] = 0.0
    return arr


def block_diag2_mask_msg(delta: int, num_slots: int = 2**15) -> np.ndarray:
    """Internal-rotation mask for ``mode='block_diag_2'`` (THOR ``%8``)."""
    arr = np.ones((num_slots,), dtype=np.float64)
    arr[np.arange(num_slots) % 8 >= delta] = 0.0
    return arr
