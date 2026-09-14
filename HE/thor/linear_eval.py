"""Liberate DualRail PC-MM helpers for Branch A (no mult_scalar in main path)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    from .dualrail_encode import (
        block_diag1_mask_msg,
        block_diag2_mask_msg,
        pack_embedding_dualrail,
        pack_weight_ff_dualrail,
        pack_weight_qkv_dualrail,
        pack_weight_wo_dualrail,
    )
except ImportError:
    from dualrail_encode import (  # type: ignore
        block_diag1_mask_msg,
        block_diag2_mask_msg,
        pack_embedding_dualrail,
        pack_weight_ff_dualrail,
        pack_weight_qkv_dualrail,
        pack_weight_wo_dualrail,
    )


# Rotations used by make_rotated_copies + block_diag_1 rotate_internal (ll=6).
# Negatives cover rotate_left(-r_delta) inside rotate_internal (stored as num_slots+δ).
# ``6`` is W_O dense fold; ``±8`` is FF input duplicate / dense2 fold.
QKV_ROT_DELTAS: list[int] = (
    list(range(1, 13))
    + list(range(-12, 0))
    + [6, 8, -8, 2048]
)

# Default encode/mask levels are NOT a fixed working entry (that comes from
# ``bts_ops.compute_level_discards`` → ``remaining_to_level_calc``).
# Chunk smokes pass an explicit ``input_level``; production HE uses the plan.
WEIGHT_LEVEL = None  # encode at CT level_calc (caller supplies)
MASK_LEVEL = 15  # THOR pre_encode_masks level; Liberate tiles to CT level

# Liberate ``level_up`` = one rescale + RNS truncate. Audit
# (``audit_level_up_span.py``): error tracks *start* lc (lc≥15 ~1e-9;
# lc=0 ~5e-4), not jump span — no span≤8 cliff. ``None`` = uncapped.
MAX_LEVEL_UP_SPAN: int | None = None


def safe_level_up(
    engine, ct, dst_level: int, *, max_span: int | None = MAX_LEVEL_UP_SPAN
):
    """
    ``level_up`` to ``dst_level``. If ``max_span`` is set, refuse larger jumps
    (prefer encrypt/bootstrap near target). Default uncapped.
    """
    src = int(ct.level_calc)
    if dst_level < src:
        raise ValueError(f"level_up cannot go down: {src} → {dst_level}")
    if dst_level == src:
        return ct
    span = dst_level - src
    if max_span is not None and span > int(max_span):
        raise ValueError(
            f"refusing large-span level_up {src}→{dst_level} (span={span} > "
            f"{max_span}). Encrypt or bootstrap at/near the target level instead."
        )
    return engine.level_up(ct, dst_level)


def prepare_linear_keys(engine, sk) -> None:
    """Ordinary rotate keys + conjugation key (not rotk_dict / bootstrap)."""
    engine.add_rot_keys_from_sk(list(QKV_ROT_DELTAS), sk)
    if getattr(engine, "conj_key", None) is None:
        engine.add_conj_key(engine.create_conjugation_key(sk))


def make_qkv_evaluator(engine):
    """
    ``ThorLinearEvaluator`` with only ``block_diag_1`` masks (skip transpose/CC masks).
    """
    from thor.linear import ThorLinearEvaluator

    ev = object.__new__(ThorLinearEvaluator)
    ev.engine = engine
    ev.masks = {
        "rot_internal": {"att": {}, "block_diag_1": {}, "block_diag_2": {}},
        "make_copies": {},
        "make_copies_merge": {},
        "make_copies_2": {},
        "transpose": {"mask0": {}, "mask1": {}, "mask2": {}, "mask3": {}},
        "ct_ct_matmul": {0: {}, 1: {}, 2: {}, 3: {}},
        # Score-only bank: same geometry as ct_ct_matmul, values ×(1/rest_div).
        # Kept separate so Context (n_in=128, scale=1) is not polluted.
        "ct_ct_matmul_score": {0: {}, 1: {}, 2: {}, 3: {}},
    }
    t0 = time.time()
    for i in range(1, 16):
        ev.masks["rot_internal"]["block_diag_1"][i] = engine.encode(
            block_diag1_mask_msg(i, engine.num_slots), level=MASK_LEVEL
        )
    for i in range(1, 8):
        ev.masks["rot_internal"]["block_diag_2"][i] = engine.encode(
            block_diag2_mask_msg(i, engine.num_slots), level=MASK_LEVEL
        )
    print(
        f"  block_diag_1/2 masks ready ({time.time()-t0:.1f}s)",
        flush=True,
    )
    return ev


def transpose_rot_deltas(num_slots: int = 2**15) -> list[int]:
    """Deltas used by ``ThorLinearEvaluator.transpose_upper_to_lower``."""
    deltas: set[int] = set()
    for i in range(4):
        n_diag = 16 * i
        deltas.add((((64 - n_diag) % 64) * 16) % num_slots)
        for n_diag_u in range(16 * i + 1, 16 * (i + 1)):
            l = 64 - n_diag_u
            deltas.add(
                (l * (2**4) + (((n_diag_u % 48) * 2) % 16) * (2**11)) % num_slots
            )
    # final merge: rotate_left(..., -2**11)
    deltas.add((- (2**11)) % num_slots)
    deltas.add(2**11)
    return sorted(deltas)


def ensure_transpose_masks(evaluator, engine) -> None:
    """Encode transpose masks (THOR ``pre_encode_masks`` transpose block)."""
    if evaluator.masks["transpose"]["mask0"]:
        return
    t0 = time.time()
    level = MASK_LEVEL
    ns = engine.num_slots
    for i in range(4):
        n_diag = 16 * i
        arr0 = np.array([1] * 16 * (64 + (n_diag - 16) % 64 + 16))
        arr1 = np.array(
            [0] * (ns - 16 * (64 - ((n_diag - 16) % 64 + 16)))
            + [1] * 16 * (64 - ((n_diag - 16) % 64 + 16))
        )
        evaluator.masks["transpose"]["mask0"][i] = engine.encode(arr0, level=level)
        evaluator.masks["transpose"]["mask1"][i] = engine.encode(arr1, level=level)
        for j in range(16 * i + 1, 16 * (i + 1)):
            l = 64 - j
            arr2 = np.array(
                [0] * (2**11) * (16 - j % 16) + [1] * (128 - l) * (2**4)
            )
            arr3 = np.array(
                [0] * (2**11) * (16 - j % 16 - 1)
                + [0] * (128 - l) * (2**4)
                + [1] * 16 * l
            )
            evaluator.masks["transpose"]["mask2"][j] = engine.encode(arr2, level=level)
            evaluator.masks["transpose"]["mask3"][j] = engine.encode(arr3, level=level)
    print(f"  transpose masks ready ({time.time()-t0:.1f}s)", flush=True)


def prepare_transpose_keys(engine, sk) -> None:
    engine.add_rot_keys_from_sk(transpose_rot_deltas(engine.num_slots), sk)
    if getattr(engine, "conj_key", None) is None:
        engine.add_conj_key(engine.create_conjugation_key(sk))


def ensure_make_copies_masks(evaluator, engine) -> None:
    """``make_copies_2`` masks with ``1/4`` baked in (no CT mult_scalar)."""
    if evaluator.masks["make_copies_2"]:
        return
    t0 = time.time()
    level = MASK_LEVEL
    for i in range(8):
        arr = np.zeros((2**15,), dtype=np.float64)
        arr[2**12 * i : 2**12 * (i + 1)] = 1.0
        evaluator.masks["make_copies_2"][i] = engine.encode(arr * (1 / 4), level=level)
    print(f"  make_copies_2 masks ready ({time.time()-t0:.1f}s)", flush=True)


def prepare_make_copies_keys(engine, sk) -> None:
    """rotsum(2**12) + ±2048 used inside ``make_copies``."""
    ns = engine.num_slots
    deltas = [2**11, (-(2**11)) % ns]
    # rotsum interval 2**12 → steps 2**12, 2**13, ...
    iv = 2**12
    while iv < ns:
        deltas.append(iv)
        iv *= 2
    engine.add_rot_keys_from_sk(deltas, sk)
    if getattr(engine, "conj_key", None) is None:
        engine.add_conj_key(engine.create_conjugation_key(sk))


def numpy_make_copies_dualrail(msgs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Plaintext twin of THOR ``make_copies`` (masks include 1/4).

    Input: 4 real slot vectors. Output: 64 real slot vectors matching HE after
    ``+conjugate`` / ``imult`` (2·Re / 2·Im).
    """
    n = 4
    assert len(msgs) == n
    out: list[np.ndarray | None] = [None] * (16 * n)
    mask0 = np.ones(2**15, dtype=np.float64)
    mask0[np.arange(2**15) % (2**12) >= 2**11] = 0.0
    mask1 = 1.0 - mask0

    def rotsum(v: np.ndarray, interval: int) -> np.ndarray:
        rep = int(np.log2((2**15) / interval))
        acc = v.copy()
        for k in range(rep):
            acc = acc + np.roll(acc, -interval * (2**k))
        return acc

    for i in range(n // 2):
        merged = msgs[i].astype(np.complex128) + 1j * msgs[i + n // 2].astype(
            np.complex128
        )
        for j in range(8):
            m = np.zeros(2**15, dtype=np.float64)
            m[2**12 * j : 2**12 * (j + 1)] = 0.25
            copied = rotsum(merged * m, 2**12)
            c0 = copied * mask0
            c1 = copied * mask1
            c0 = c0 + np.roll(c0, 2**11)  # rotate_left(-2048)
            c1 = c1 + np.roll(c1, -(2**11))
            out[i * 16 + 2 * j] = (2.0 * np.real(c0)).astype(np.float64)
            out[(i + n // 2) * 16 + 2 * j] = (2.0 * np.imag(c0)).astype(np.float64)
            out[i * 16 + 2 * j + 1] = (2.0 * np.real(c1)).astype(np.float64)
            out[(i + n // 2) * 16 + 2 * j + 1] = (2.0 * np.imag(c1)).astype(
                np.float64
            )
    return out  # type: ignore[return-value]


def encrypt_embedding_real8(engine, embedding: np.ndarray, pk, *, level: int):
    """
    Encrypt layer-entry embedding as **8 real** packs (``encode_input_lower``).

    Layout matches DualRail unpack(Re/Im); pack with ``re+i·im`` (no rot) recovers
    QKV DualRail. This is the only encoder-entry encrypt path.
    """
    try:
        from thor_encoder_linear_core import (  # noqa: WPS433
            bert_base,
            encode_input_lower_diagonals,
        )
    except ImportError:
        from HE.thor_encoder_linear_core import (  # type: ignore
            bert_base,
            encode_input_lower_diagonals,
        )

    cfg = bert_base()
    msgs = encode_input_lower_diagonals(
        np.asarray(embedding, dtype=np.float64), cfg
    )
    if len(msgs) != 8:
        raise ValueError(f"expected 8 lower packs, got {len(msgs)}")
    cts = np.full((8,), None, dtype=object)
    for i, msg in enumerate(msgs):
        cts[i] = engine.encode_and_encrypt(msg, pk, level=level)
    return cts.astype(object)


def pack_real8_to_dualrail4(engine, x8) -> np.ndarray:
    """8 real → 4 complex (``re+i·im``); no ``rotate_left(-6)`` (L0/embedding layout)."""
    x8 = list(x8)
    if len(x8) != 8:
        raise ValueError(f"expected 8 real packs, got {len(x8)}")
    out = np.full((4,), None, dtype=object)
    for i in range(4):
        out[i] = engine.cc_add(x8[i], engine.imult(x8[i + 4]))
    return out


def encrypt_embedding_dualrail(engine, embedding: np.ndarray, pk, *, level: int):
    """
    QKV-island helper: ``encrypt_embedding_real8`` then pack to 4 complex.

    Does **not** use THOR ``encrypt_embedding`` DualRail plaintext. Prefer calling
    ``encrypt_embedding_real8`` at layer entry; use this only for isolated QKV tests.
    """
    x8 = encrypt_embedding_real8(engine, embedding, pk, level=level)
    return pack_real8_to_dualrail4(engine, x8)


def encode_weight_qkv_dualrail(
    engine,
    w: np.ndarray,
    *,
    level: int,
    scale: float = 1.0,
    out_packs: int = 1,
):
    """Encode DualRail QKV weight PTs at ``level`` (= CT ``level_calc``)."""
    msgs = pack_weight_qkv_dualrail(w, scale=scale, out_packs=out_packs)
    n_out_p, ll, n_in_c = msgs.shape
    pts = np.full((n_out_p, ll, n_in_c), None, dtype=object)
    total = n_out_p * ll * n_in_c
    done = 0
    t0 = time.time()
    for out in range(n_out_p):
        for l in range(ll):
            for n in range(n_in_c):
                pts[out, l, n] = engine.encode(msgs[out, l, n], level=level)
                done += 1
                if done == 1 or done == total or done % 32 == 0:
                    print(
                        f"  weight PT {done}/{total} "
                        f"({time.time()-t0:.1f}s)",
                        flush=True,
                    )
    return pts


def qkv_pcmm_he(
    engine,
    evaluator,
    x_cplx,
    w_pts,
    *,
    input_level: int | None = None,
    real_extract: bool = True,
):
    """
    DualRail QKV PC-MM: rotate copies → pt_ct_matmul → optional +conj.

    Prefer CTs already at the working ``level_calc`` (direct encrypt).
    Optional ``input_level`` only triggers a *short* ``safe_level_up``.
    """
    x = np.full((x_cplx.shape[0],), None, dtype=object)
    for i in range(x_cplx.shape[0]):
        if input_level is None or x_cplx[i].level_calc == input_level:
            x[i] = x_cplx[i]
        else:
            x[i] = safe_level_up(engine, x_cplx[i], input_level)

    rots = evaluator.make_rotated_copies(x)
    wx = evaluator.pt_ct_matmul(w_pts, rots, mode="block_diag_1")
    if not real_extract:
        return wx

    out = np.full((wx.shape[0],), None, dtype=object)
    for i in range(wx.shape[0]):
        out[i] = engine.cc_add(wx[i], engine.conjugate(wx[i]))
    return out


def mainline_qkv_ref(
    embedding: np.ndarray,
    w: np.ndarray,
    *,
    scale: float = 1.0,
    out_packs: int = 1,
) -> list[np.ndarray]:
    """Pure-real PC-MM reference from ``thor_encoder_linear_core`` (first out packs)."""
    import sys
    from pathlib import Path

    he_root = str(Path(__file__).resolve().parents[1])
    if he_root not in sys.path:
        sys.path.insert(0, he_root)
    from thor_encoder_linear_core import (  # noqa: WPS433
        _qkv_pc_from_enc,
        bert_base,
        encode_input_lower_diagonals,
    )

    cfg = bert_base()
    x_in = encode_input_lower_diagonals(embedding, cfg)
    q = _qkv_pc_from_enc(x_in, w, cfg, scale=scale)
    return [np.asarray(q[i], dtype=np.float64) for i in range(out_packs)]


def ensure_ct_ct_matmul_masks(
    evaluator,
    engine,
    *,
    n_max: int = 64,
    level: int | None = None,
    bank: str = "ct_ct_matmul",
    value_scale: float = 1.0,
) -> None:
    """
    Encode BSGS ``ct_ct_matmul`` masks; ``level`` should match score CT ``level_calc``.

    ``bank``: ``ct_ct_matmul`` (Context / native) or ``ct_ct_matmul_score`` (Score dense).
    ``value_scale``: multiply mask values before encode (Score dense uses ``1/8``).
    """
    if bank not in evaluator.masks:
        evaluator.masks[bank] = {0: {}, 1: {}, 2: {}, 3: {}}
    masks = evaluator.masks[bank]
    enc_level = MASK_LEVEL if level is None else int(level)
    vs = float(value_scale)
    # Re-encode if empty or level overridden (score needs CT-matched encode).
    if level is None and masks[0] and max(masks[0].keys()) >= n_max - 1:
        return
    t0 = time.time()
    ns = engine.num_slots
    for n in range(1, n_max):
        rot = n
        j = n % 16
        arr0 = np.full((ns,), 1.0, dtype=np.float64)
        arr0[np.arange(ns) % (2**11) >= (2**11 - 16 * rot)] = 0.0
        arr1 = np.full((ns,), 0.0, dtype=np.float64)
        arr1[np.arange(ns) % (2**11) >= (2**11 - 16 * rot)] = 1.0
        if j == 0:
            masks[0][n] = engine.encode(arr0 * vs, level=enc_level)
            masks[1][n] = engine.encode(arr1 * vs, level=enc_level)
        else:
            arr0[: (2**11) * j] = 0.0
            masks[0][n] = engine.encode(arr0 * vs, level=enc_level)
            arr1[-(2**11) :] = 0.0
            if j > 1:
                arr1[: (2**11) * (j - 1)] = 0.0
            masks[1][n] = engine.encode(arr1 * vs, level=enc_level)
            arr2 = np.full((ns,), 1.0, dtype=np.float64)
            arr2[np.arange(ns) % (2**11) >= (2**11 - 16 * rot)] = 0.0
            arr2[(2**11) * j :] = 0.0
            masks[2][n] = engine.encode(arr2 * vs, level=enc_level)
            arr3 = np.ones((ns,), dtype=np.float64) - arr0 - arr1 - arr2
            masks[3][n] = engine.encode(arr3 * vs, level=enc_level)
    print(
        f"  {bank} masks n<={n_max-1} scale={vs:g} @level={enc_level} "
        f"ready ({time.time()-t0:.1f}s)",
        flush=True,
    )


def bind_att_score_mask_complement(attn, *, rest_div: float = 8.0, n0_amp: float | None = None) -> None:
    """
    Patch THOR ``calculate_attention_score`` for Liberate / island HE smokes:

    1. Replace ``2**41`` integer complement with ``masks[1]/[3]`` PT×CT
       (``2**41`` assumes encode(1)≡Δ at the CT's scale state).
    2. Dense-scale contract (default): use ``ct_ct_matmul_score`` masks with
       values already ``×(1/rest_div)`` (default 8) so n≥1 match n=0 / dense
       with **0 extra mul-depth**. Callers must ``ensure_ct_ct_matmul_masks(...,
       bank='ct_ct_matmul_score', value_scale=1/rest_div)`` before Score.

    ``n0_amp`` is deprecated (ignored). Old ``n0_amp=8`` + global ``/8`` is
    replaced by mask-internal ``/rest_div`` on n≥1 only.
    """
    import inspect
    import textwrap
    import types

    from thor.bert import ThorBertAttention

    if n0_amp is not None and float(n0_amp) != 1.0:
        # Compat: treat legacy n0_amp=8 as rest_div=8 dense contract.
        rest_div = float(n0_amp)

    src = textwrap.dedent(
        inspect.getsource(ThorBertAttention.calculate_attention_score)
    )
    old_j0 = (
        "scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)\n"
        "                temp[out, 1] = self.engine.sub(scaled_temp_ct_ct, temp[out,0])"
    )
    new_j0 = (
        "temp[out, 1] = self.engine.pt_ct_mult_extended(masks[1][n], temp_ct_ct)"
    )
    old_j = (
        "scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)\n"
        "                temp[out, 3] = self.engine.sub(scaled_temp_ct_ct, add_temp)"
    )
    new_j = (
        "temp[out, 3] = self.engine.pt_ct_mult_extended(masks[3][n], temp_ct_ct)"
    )
    old_masks = "masks = self.evaluator.masks['ct_ct_matmul']"
    rd = float(rest_div)
    if abs(rd - 1.0) < 1e-15:
        new_masks = old_masks
    else:
        new_masks = "masks = self.evaluator.masks['ct_ct_matmul_score']"
    if old_j0 not in src or old_j not in src or old_masks not in src:
        raise RuntimeError(
            "THOR calculate_attention_score source changed; mask-complement / "
            "rest_div patch failed"
        )
    src = (
        src.replace(old_j0, new_j0)
        .replace(old_j, new_j)
        .replace(old_masks, new_masks, 1)
    )
    ns: dict = {}
    exec(src, ThorBertAttention.calculate_attention_score.__globals__, ns)
    attn.calculate_attention_score = types.MethodType(
        ns["calculate_attention_score"], attn
    )
    attn._score_rest_div = rd  # type: ignore[attr-defined]


def ensure_rot_internal_att_masks(evaluator, engine, *, deltas: list[int]) -> None:
    """``rotate_internal(..., mode='att')`` masks (e.g. delta=64 for K complexify)."""
    bank = evaluator.masks["rot_internal"]["att"]
    need = [d for d in deltas if d not in bank]
    if not need:
        return
    t0 = time.time()
    level = MASK_LEVEL
    ns = engine.num_slots
    for d in need:
        arr = np.ones((ns,), dtype=np.float64)
        arr[np.arange(ns) % (2**11) >= (16 * d)] = 0.0
        bank[d] = engine.encode(arr, level=level)
    print(f"  rot_internal att masks {need} ready ({time.time()-t0:.1f}s)", flush=True)


def att_score_rot_deltas(num_slots: int = 2**15, *, n_in: int = 64) -> list[int]:
    """
    Left-rot deltas for THOR score/context BSGS.

    ``rotate_left(ct, -rrot)`` indexes ``rot_keys[(-rrot) % num_slots]``.
    Do **not** also register ``rrot`` / ``num_slots-rrot`` separately — that
    triples VRAM (context n_in=128 OOMs a 24GB GPU).
    """
    deltas: set[int] = set()
    for n in range(1, n_in):
        j = n % 16
        rrot = ((2**11) * j - 16 * n) % num_slots
        deltas.add((-rrot) % num_slots)
    # complexify rotate_internal(att, 64): ±1024; merge rotate ±2048
    deltas.update({2**10, (-(2**10)) % num_slots, 2**11, (-(2**11)) % num_slots})
    return sorted(deltas)


def prepare_att_score_keys(engine, sk) -> None:
    """Keys for transpose + make_copies + K complexify + score BSGS."""
    prepare_transpose_keys(engine, sk)
    prepare_make_copies_keys(engine, sk)
    engine.add_rot_keys_from_sk(att_score_rot_deltas(engine.num_slots), sk)
    if getattr(engine, "conj_key", None) is None:
        engine.add_conj_key(engine.create_conjugation_key(sk))


def complexify_k_he(engine, evaluator, k_real):
    """
    Match ``bert.forward`` K path after transpose:
    ``cc_add(level_up(k), imult(rotate_internal(k, 64, 'att')))`` then rescale.
    """
    out = np.full((k_real.shape[0],), None, dtype=object)
    for i in range(k_real.shape[0]):
        up = safe_level_up(engine, k_real[i], int(k_real[i].level_calc) + 1)
        rot = evaluator.rotate_internal(k_real[i], 64, mode="att")
        out[i] = engine.rescale(engine.cc_add(up, engine.imult(rot)))
    return out


def make_att_score_module(engine, evaluator):
    """Thin ``ThorBertAttention`` shell for ``calculate_attention_score``."""
    from thor.bert import ThorBertAttention

    attn = object.__new__(ThorBertAttention)
    attn.engine = engine
    attn.evaluator = evaluator
    return attn


def prep_q_for_copies_he(engine, q_real):
    """Match ``forward.ipynb`` before ``make_copies``: single rescale."""
    out = np.full((q_real.shape[0],), None, dtype=object)
    for i in range(q_real.shape[0]):
        out[i] = engine.rescale(q_real[i])
    return out


def _he_root_on_path() -> None:
    import sys
    from pathlib import Path

    he_root = str(Path(__file__).resolve().parents[1])
    if he_root not in sys.path:
        sys.path.insert(0, he_root)


def dense_qk_scores(
    q: np.ndarray,
    k: np.ndarray,
    cfg,
    *,
    scale: float,
) -> np.ndarray:
    """
    Dense attention logits ``scores[h, key, query] = scale * (K_h @ Q_h.T)``.

    ``q,k`` shape ``(seq, hidden)``; head split is ``(seq, heads, head_dim)``.
    """
    q_h = q.reshape(cfg.seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)
    k_h = k.reshape(cfg.seq_len, cfg.num_heads, cfg.head_dim).transpose(1, 0, 2)
    scores = np.zeros(
        (cfg.num_heads, cfg.seq_len, cfg.seq_len), dtype=np.float64
    )
    for h in range(cfg.num_heads):
        scores[h] = scale * (k_h[h] @ q_h[h].T)
    return scores


def encode_qkv_upper_from_dense(mat: np.ndarray, cfg) -> list[np.ndarray]:
    """
    Pack dense ``(seq, hidden)`` into QKV **upper-diagonal** packs (``n_out_packed_qkv``).

    Layout matches attention geometry (``diag=(d-t)%head_dim`` at ``_slot_index(j,t,h)``).
    Prefer this over PC-MM when the oracle is dense ``Q/K`` matmul.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import _slot_index  # noqa: WPS433

    if mat.shape != (cfg.seq_len, cfg.hidden_dim):
        raise ValueError(f"expected {(cfg.seq_len, cfg.hidden_dim)}, got {mat.shape}")
    hd, pack = cfg.head_dim, cfg.pack
    packs = [
        np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_out_packed_qkv)
    ]
    for h in range(cfg.num_heads):
        for t in range(cfg.seq_len):
            base = h * hd
            for d in range(hd):
                diag = (d - t) % hd
                pack_i, j = diag // pack, diag % pack
                packs[pack_i][_slot_index(cfg, j, t, h)] = mat[t, base + d]
    return packs


def decode_score_packs(score_packs: list[np.ndarray], cfg) -> np.ndarray:
    """Slot score packs → ``scores[h, key, query]`` (same indexing as ``slot_softmax``)."""
    _he_root_on_path()
    from thor_encoder_linear_core import _slot_index  # noqa: WPS433

    scores = np.zeros(
        (cfg.num_heads, cfg.seq_len, cfg.seq_len), dtype=np.float64
    )
    for h in range(cfg.num_heads):
        for query in range(cfg.seq_len):
            for key in range(cfg.seq_len):
                u = (key - query) % cfg.seq_len
                pack_i, j = u // cfg.pack, u % cfg.pack
                scores[h, key, query] = score_packs[pack_i][
                    _slot_index(cfg, j, query, h)
                ]
    return scores


def thor_score_to_mainline_packs(
    thor_packs: list[np.ndarray],
    *,
    gain: float = 1.0,
) -> list[np.ndarray]:
    """
    THOR score extract → 8 mainline score packs.

    THOR returns ``[2·Re0..2·Re3, 2·Im0..2·Im3]`` already in mainline pack
    order ``0..7``. With ``numpy_cc_mm_score_dualrail(..., rest_div=8)``
    (n≥1 masks ×1/8), ``gain=1`` matches geometric / dense (~0).
    """
    assert len(thor_packs) == 8
    inv = 1.0 / float(gain)
    return [inv * np.asarray(p, dtype=np.float64) for p in thor_packs]


def complexify_k_plain(k_lower: list[np.ndarray], cfg) -> list[np.ndarray]:
    """
    Plain twin of ``complexify_k_he`` / ``bert.forward`` K complexify:

    ``K_cplx = K + 1j * rotate_internal(K, 64, mode='att')``.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_internal  # noqa: WPS433

    out: list[np.ndarray] = []
    for k in k_lower:
        kr = np.asarray(k, dtype=np.float64)
        imag = rotate_internal(kr, 64, mode="att", cfg=cfg)
        out.append(kr.astype(np.complex128) + 1j * imag.astype(np.complex128))
    return out


def _merge_bsgs_dualrail_complex(ttemp: np.ndarray, n_out: int, cfg) -> list[np.ndarray]:
    """
    Complex ttemp merge + DualRail unpack (``bert.calculate_attention_score``).

    Per ``out``: rotate lanes 1/3 by ``-stride``; lane2 = ``i·conj(c2+c3)``;
    ``cplx = c0+c1+lane2``; emit ``2·Re`` at ``out`` and ``2·Im`` at ``out+n_out``.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    stride = int(cfg.slot_stride)
    z = np.zeros(int(cfg.num_slots), dtype=np.complex128)
    output: list[np.ndarray | None] = [None] * (2 * n_out)
    for out in range(n_out):
        c0 = ttemp[out, 0] if ttemp[out, 0] is not None else z
        c1 = ttemp[out, 1] if ttemp[out, 1] is not None else z
        c2 = ttemp[out, 2] if ttemp[out, 2] is not None else z
        c3 = ttemp[out, 3] if ttemp[out, 3] is not None else z
        c0 = np.asarray(c0, dtype=np.complex128)
        c1 = rotate_left(np.asarray(c1, dtype=np.complex128), -stride)
        c2 = np.asarray(c2, dtype=np.complex128)
        c3 = rotate_left(np.asarray(c3, dtype=np.complex128), -stride)
        lane2 = 1j * np.conjugate(c2 + c3)
        cplx = c0 + c1 + lane2
        # z+conj → 2 Re; i·(conj−z) → 2 Im
        output[out] = (2.0 * np.real(cplx)).astype(np.float64)
        output[out + n_out] = (2.0 * np.imag(cplx)).astype(np.float64)
    return output  # type: ignore[return-value]


def numpy_cc_mm_score_dualrail(
    k_cplx: list[np.ndarray],
    q_copies: list[np.ndarray],
    cfg,
    ct_ct: dict,
    scale: float,
    *,
    n_in: int | None = None,
    rest_div: float = 8.0,
    n0_amp: float | None = None,
) -> list[np.ndarray]:
    """
    Plain twin of THOR ``calculate_attention_score`` (complex left × real copies).

    Returns 8 real packs as ``[2·Re0..2·Re3, 2·Im0..2·Im3]``.

    Dense contract (default ``rest_div=8``): scale n≥1 BSGS mask lanes by
    ``1/rest_div`` so n=0 and n≥1 land at the same (dense) scale with **0**
    extra depth. Unpack with ``gain=1``.

    Legacy ``n0_amp=8`` (amplify n=0 then global ``/8``) is accepted as an
    alias for ``rest_div=8`` when ``n0_amp`` is passed.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import (  # noqa: WPS433
        _accumulate_ttemp_adaptive,
        rotate_left,
    )

    if n0_amp is not None and float(n0_amp) != 1.0:
        rest_div = float(n0_amp)

    n_out = len(k_cplx)
    if n_out != 4:
        raise ValueError(f"expected 4 K_cplx packs, got {n_out}")
    if n_in is None:
        n_in = int(cfg.head_dim)
    if len(q_copies) < n_in:
        raise ValueError(f"q_copies len {len(q_copies)} < n_in={n_in}")

    stride, p = int(cfg.slot_stride), int(cfg.pack)
    scale_pt = np.full(int(cfg.num_slots), float(scale), dtype=np.float64)
    ttemp = np.full((n_out, 4), None, dtype=object)
    inv_rest = 1.0 / float(rest_div)

    for out in range(n_out):
        prod0 = np.asarray(k_cplx[out], dtype=np.complex128) * np.asarray(
            q_copies[0], dtype=np.float64
        )
        ttemp[out, 0] = scale_pt * prod0

    for n in range(1, n_in):
        j = n % p
        q = n // p
        rrot = (stride * j - p * n) % int(cfg.num_slots)
        temp = np.full((n_out, 4), None, dtype=object)
        right = np.asarray(q_copies[n], dtype=np.float64)
        for out in range(n_out):
            left = rotate_left(
                np.asarray(k_cplx[out], dtype=np.complex128), (-rrot) % int(cfg.num_slots)
            )
            prod = left * right
            if j == 0:
                temp[out, 0] = (inv_rest * np.asarray(ct_ct[0][n], dtype=np.float64)) * prod
                temp[out, 1] = (inv_rest * np.asarray(ct_ct[1][n], dtype=np.float64)) * prod
            else:
                temp[out, 0] = (inv_rest * np.asarray(ct_ct[0][n], dtype=np.float64)) * prod
                temp[out, 1] = (inv_rest * np.asarray(ct_ct[1][n], dtype=np.float64)) * prod
                temp[out, 2] = (inv_rest * np.asarray(ct_ct[2][n], dtype=np.float64)) * prod
                temp[out, 3] = (inv_rest * np.asarray(ct_ct[3][n], dtype=np.float64)) * prod
        _accumulate_ttemp_adaptive(ttemp, temp, q, j, n_out)

    return _merge_bsgs_dualrail_complex(ttemp, n_out, cfg)


def compute_attention_score_dualrail_island(
    q_merged: list[np.ndarray],
    k_merged: list[np.ndarray],
    cfg,
    *,
    scale: float | None = None,
    gain: float = 1.0,
    rest_div: float = 8.0,
    ct_ct: dict | None = None,
) -> list[np.ndarray]:
    """
    途径 A（明文）：纯实 Q/K upper packs → DualRail score 岛 → 纯实 8 score packs.

    Steps (layouts already match THOR QKV upper-diag)::

        K_lower = transpose(K)
        K_cplx  = complexify_k_plain(K_lower)          # 4 complex
        Q_copies = numpy_make_copies_dualrail(Q)       # 64 real
        raw8 = numpy_cc_mm_score_dualrail(..., rest_div=8)
        return thor_score_to_mainline_packs(raw8, gain=1)

    With ``rest_div=8`` and ``gain=1`` this matches geometric / dense (~machine eps).
    CT×CT count ~256 (64×4) vs pure-real geometric/heiso ~8192.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import (  # noqa: WPS433
        _build_ct_ct_masks,
        transpose_upper_to_lower,
    )

    if scale is None:
        scale = 1.0 / float(np.sqrt(cfg.head_dim))
    if ct_ct is None:
        ct_ct = _build_ct_ct_masks(cfg, cfg.head_dim)

    k_lower = transpose_upper_to_lower(k_merged, cfg)
    k_cplx = complexify_k_plain(k_lower, cfg)
    q_copies = numpy_make_copies_dualrail(q_merged)
    raw = numpy_cc_mm_score_dualrail(
        k_cplx, q_copies, cfg, ct_ct, float(scale), rest_div=float(rest_div)
    )
    return thor_score_to_mainline_packs(raw, gain=gain)


def encode_score_packs_from_dense(scores: np.ndarray, cfg) -> list[np.ndarray]:
    """Dense ``scores[h,key,query]`` → ``n_input_packs`` slot vectors."""
    _he_root_on_path()
    from thor_encoder_linear_core import _slot_index  # noqa: WPS433

    packs = [
        np.zeros(cfg.num_slots, dtype=np.float64) for _ in range(cfg.n_input_packs)
    ]
    for h in range(cfg.num_heads):
        for query in range(cfg.seq_len):
            for key in range(cfg.seq_len):
                u = (key - query) % cfg.seq_len
                pack_i, j = u // cfg.pack, u % cfg.pack
                packs[pack_i][_slot_index(cfg, j, query, h)] = scores[h, key, query]
    return packs


def dense_alpha_v_context(alpha: np.ndarray, v: np.ndarray, cfg) -> np.ndarray:
    """
    Dense context ``(seq, hidden)``: ``ctx = merge(alpha[h].T @ V_h)``.

    ``alpha[h,key,query]``, ``v`` shape ``(seq, hidden)``.
    """
    _he_root_on_path()
    from slot_decode import merge_heads, split_heads  # noqa: WPS433

    v_h = split_heads(v, cfg)
    ctx_h = np.stack(
        [alpha[h].T @ v_h[h] for h in range(cfg.num_heads)], axis=0
    )
    return merge_heads(ctx_h, cfg)


def decode_qkv_upper_to_dense(packs: list[np.ndarray], cfg) -> np.ndarray:
    """Inverse of ``encode_qkv_upper_from_dense`` → ``(seq, hidden)``."""
    _he_root_on_path()
    from thor_encoder_linear_core import _slot_index  # noqa: WPS433

    mat = np.zeros((cfg.seq_len, cfg.hidden_dim), dtype=np.float64)
    hd, pack = cfg.head_dim, cfg.pack
    for h in range(cfg.num_heads):
        base = h * hd
        for t in range(cfg.seq_len):
            for d in range(hd):
                diag = (d - t) % hd
                pack_i, j = diag // pack, diag % pack
                mat[t, base + d] = packs[pack_i][_slot_index(cfg, j, t, h)]
    return mat


def thor_context_to_mainline_packs(
    cplx_msgs: list[np.ndarray],
    *,
    gain: float = 1.0,
) -> list[np.ndarray]:
    """
    THOR context returns 2 complex CTs: ``ct[i] ≈ pack[i] + i·pack[i+2]``.

    Decrypt complex → 4 real mainline upper packs ``[p0,p1,p2,p3]`` / ``gain``.

    With geometric ``make_copies`` alpha, ``gain=1`` matches dense. Softmax
    DualRail alpha layout historically used empirical ``gain≈32``.
    """
    assert len(cplx_msgs) == 2
    inv = 1.0 / float(gain)
    out: list[np.ndarray] = [None] * 4  # type: ignore[list-item]
    for i, msg in enumerate(cplx_msgs):
        z = np.asarray(msg, dtype=np.complex128)
        out[i] = inv * np.real(z)
        out[i + 2] = inv * np.imag(z)
    return out


def bind_att_context_mask_complement(attn) -> None:
    """Same ``2**41`` → PT mask patch for ``calculate_attention_context``."""
    import inspect
    import textwrap
    import types

    from thor.bert import ThorBertAttention

    src = textwrap.dedent(
        inspect.getsource(ThorBertAttention.calculate_attention_context)
    )
    old_j0 = (
        "scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)\n"
        "                temp[out, 1] = self.engine.sub(scaled_temp_ct_ct, temp[out,0])"
    )
    new_j0 = (
        "temp[out, 1] = self.engine.pt_ct_mult_extended(masks[1][n], temp_ct_ct)"
    )
    old_j = (
        "scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)\n"
        "                temp[out, 3] = self.engine.sub(scaled_temp_ct_ct, add_temp)"
    )
    new_j = (
        "temp[out, 3] = self.engine.pt_ct_mult_extended(masks[3][n], temp_ct_ct)"
    )
    if old_j0 not in src or old_j not in src:
        raise RuntimeError(
            "THOR calculate_attention_context source changed; mask-complement patch failed"
        )
    src = src.replace(old_j0, new_j0).replace(old_j, new_j)
    ns: dict = {}
    exec(src, ThorBertAttention.calculate_attention_context.__globals__, ns)
    attn.calculate_attention_context = types.MethodType(
        ns["calculate_attention_context"], attn
    )


def make_att_context_module(engine, evaluator):
    """Thin ``ThorBertAttention`` shell for ``calculate_attention_context``."""
    from thor.bert import ThorBertAttention

    attn = object.__new__(ThorBertAttention)
    attn.engine = engine
    attn.evaluator = evaluator
    return attn


def complexify_v_he(engine, v_real):
    """
    Match ``bert.forward`` V path: ``v_cplx[i] = v[i] + i·v[i+2]`` then rescale.

    ``v_real`` length 4 → length 2 complex CTs.
    """
    out = np.full((2,), None, dtype=object)
    for i in range(2):
        merged = engine.cc_add(v_real[i], engine.imult(v_real[i + 2]))
        out[i] = engine.rescale(merged)
    return out


def att_context_rot_deltas(num_slots: int = 2**15, *, n_in: int = 128) -> list[int]:
    """Rotation deltas inside THOR ``calculate_attention_context`` (n_in=128)."""
    return att_score_rot_deltas(num_slots, n_in=n_in)


def prepare_att_context_keys(engine, sk) -> None:
    """Keys for context BSGS (alpha copies are plaintext-encrypted, no HE make_copies)."""
    engine.add_rot_keys_from_sk(att_context_rot_deltas(engine.num_slots), sk)
    if getattr(engine, "conj_key", None) is None:
        engine.add_conj_key(engine.create_conjugation_key(sk))


def pt_rotsum(v: np.ndarray, interval: int) -> np.ndarray:
    """Plaintext twin of ``ThorLinearEvaluator.rotsum`` / Liberate rotate-sum."""
    n = int(v.shape[0])
    if interval <= 0 or interval >= n:
        return np.asarray(v, dtype=np.complex128 if np.iscomplexobj(v) else np.float64)
    rep = int(np.log2(n / interval))
    temp = np.asarray(v).copy()
    for i in range(rep):
        # Liberate rotate_left(k) ≡ np.roll(..., -k)
        temp = temp + np.roll(temp, -int(interval) * (1 << i))
    return temp


def dualrail_alpha_copies_softmax_layout(
    alpha_packs: list[np.ndarray],
    *,
    extract_gain: float = 2.0,
    mode: str = "final",
    inv_eps: float = 1e-12,
) -> list[np.ndarray]:
    """
    Plaintext Twin of THOR ``he_softmax`` **output** layout (128 real vectors).

    Softmax I/O (THOR DualRail)::

      in:  u[0..7]  — score-pack layout (same as att_score extract /8)
      mid: exp_u[i], then DualRail ``exp_cplx[i]=exp[i]+i·exp[i+4]``
      out: 128 = cplx_softmax1(64) + cplx_softmax2(64)
           for i in 0..3, j in 0..15:
             copies[i*16+j]     = 2·Re(rotsum(exp_cplx[i]*inv_rot[j]))
             copies[64+i*16+j] = 2·Im(...)

    After Softmax, ``inv`` is already ``rotsum``-replicated, so ``inv_rot[j]``
    are equal and the 16 ``j`` copies per DualRail pair are **identical**;
    context BSGS still uses different ``n`` (different V rotations).

    Modes (skip-Softmax linear smoke; alpha already row-normalized)::

      ``final`` (default): Softmax **end stage only** — no exp —
        sigma = rotsum(Σ packs, 2**11); inv = 1/sigma;
        then DualRail merge × inv → rotsum → 2Re/2Im ×16.
        (No HE ``mult_int_scalar(1/(2·D_δ))``; exact plaintext inv.)

      ``identity``: DualRail merge + replicate only (no sum/inv/rotsum).
        Kept for A/B; context BSGS expects Softmax-broadcast shape, so this
        usually fails HE vs dense.
    """
    assert len(alpha_packs) == 8
    assert mode in ("final", "identity")
    out: list[np.ndarray] = [None] * 128  # type: ignore[list-item]
    g = float(extract_gain)

    if mode == "identity":
        for i in range(4):
            cplx = alpha_packs[i].astype(np.complex128) + 1j * alpha_packs[
                i + 4
            ].astype(np.complex128)
            re = g * np.real(cplx)
            im = g * np.imag(cplx)
            for j in range(16):
                out[i * 16 + j] = re.copy()
                out[64 + i * 16 + j] = im.copy()
        return out

    # Softmax end-stage: Σ packs → rotsum → inv → ×inv → rotsum → extract
    sigma = np.zeros_like(alpha_packs[0], dtype=np.float64)
    for p in alpha_packs:
        sigma = sigma + np.asarray(p, dtype=np.float64)
    sigma = np.real(pt_rotsum(sigma, 2**11)).astype(np.float64)
    inv = np.zeros_like(sigma)
    nz = np.abs(sigma) > float(inv_eps)
    inv[nz] = 1.0 / sigma[nz]

    for i in range(4):
        cplx = alpha_packs[i].astype(np.complex128) + 1j * alpha_packs[
            i + 4
        ].astype(np.complex128)
        prod = cplx * inv  # inv already rotsum-replicated → all j identical
        copied = pt_rotsum(prod, 2**11)
        re = g * np.real(copied)
        im = g * np.imag(copied)
        for j in range(16):
            out[i * 16 + j] = re.copy()
            out[64 + i * 16 + j] = im.copy()
    return out


def complexify_v_plain(v_merged: list[np.ndarray]) -> list[np.ndarray]:
    """
    Plain twin of ``complexify_v_he`` / ``bert.forward`` V complexify:

    ``V_cplx[i] = V[i] + 1j * V[i+2]`` (length 4 → 2).
    """
    if len(v_merged) != 4:
        raise ValueError(f"expected 4 V packs, got {len(v_merged)}")
    return [
        np.asarray(v_merged[i], dtype=np.complex128)
        + 1j * np.asarray(v_merged[i + 2], dtype=np.complex128)
        for i in range(2)
    ]


def _merge_bsgs_dualrail_context_cplx(
    ttemp: np.ndarray, n_out: int, cfg
) -> list[np.ndarray]:
    """
    Complex ttemp merge for context (``bert.calculate_attention_context``).

    Returns ``n_out`` complex vectors (no 2·Re/2·Im unpack; that is decrypt-side).
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    stride = int(cfg.slot_stride)
    z = np.zeros(int(cfg.num_slots), dtype=np.complex128)
    out: list[np.ndarray] = []
    for out_i in range(n_out):
        c0 = ttemp[out_i, 0] if ttemp[out_i, 0] is not None else z
        c1 = ttemp[out_i, 1] if ttemp[out_i, 1] is not None else z
        c2 = ttemp[out_i, 2] if ttemp[out_i, 2] is not None else z
        c3 = ttemp[out_i, 3] if ttemp[out_i, 3] is not None else z
        c0 = np.asarray(c0, dtype=np.complex128)
        c1 = rotate_left(np.asarray(c1, dtype=np.complex128), -stride)
        c2 = np.asarray(c2, dtype=np.complex128)
        c3 = rotate_left(np.asarray(c3, dtype=np.complex128), -stride)
        lane2 = 1j * np.conjugate(c2 + c3)
        out.append(c0 + c1 + lane2)
    return out


def numpy_cc_mm_context_dualrail(
    v_cplx: list[np.ndarray],
    alpha_copies: list[np.ndarray],
    cfg,
    ct_ct: dict,
    scale: float = 1.0,
    *,
    n_in: int | None = None,
    n0_amp: float = 1.0,
) -> list[np.ndarray]:
    """
    Plain twin of THOR ``calculate_attention_context`` (complex V × alpha copies).

    Returns 2 complex packs (before ``thor_context_to_mainline_packs`` /gain).

    With geometric ``make_copies`` alpha and ``n0_amp=1``, unpack ``gain=1``
    matches dense / geometric (~0). Softmax-final DualRail alpha layout is a
    different encoding (HE Softmax I/O) and is not bit-exact to ``make_copies``.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import (  # noqa: WPS433
        _accumulate_ttemp_adaptive,
        rotate_left,
    )

    n_out = len(v_cplx)
    if n_out != 2:
        raise ValueError(f"expected 2 V_cplx packs, got {n_out}")
    if n_in is None:
        n_in = int(cfg.seq_len)
    if len(alpha_copies) < n_in:
        raise ValueError(
            f"alpha_copies len {len(alpha_copies)} < n_in={n_in}"
        )

    stride, p = int(cfg.slot_stride), int(cfg.pack)
    # THOR applies scale_pt only on n=0; n≥1 products are unscaled here.
    scale_pt = np.full(
        int(cfg.num_slots), float(scale) * float(n0_amp), dtype=np.float64
    )
    ttemp = np.full((n_out, 4), None, dtype=object)

    for out in range(n_out):
        prod0 = np.asarray(v_cplx[out], dtype=np.complex128) * np.asarray(
            alpha_copies[0], dtype=np.float64
        )
        ttemp[out, 0] = scale_pt * prod0

    for n in range(1, n_in):
        j = n % p
        q = (n // p) % 4  # context: wrap q (n_in=128)
        rrot = (stride * j - p * n) % int(cfg.num_slots)
        temp = np.full((n_out, 4), None, dtype=object)
        right = np.asarray(alpha_copies[n], dtype=np.float64)
        for out in range(n_out):
            left = rotate_left(
                np.asarray(v_cplx[out], dtype=np.complex128),
                (-rrot) % int(cfg.num_slots),
            )
            prod = left * right
            if j == 0:
                temp[out, 0] = np.asarray(ct_ct[0][n], dtype=np.float64) * prod
                temp[out, 1] = prod - temp[out, 0]
            else:
                temp[out, 0] = np.asarray(ct_ct[0][n], dtype=np.float64) * prod
                temp[out, 1] = np.asarray(ct_ct[1][n], dtype=np.float64) * prod
                temp[out, 2] = np.asarray(ct_ct[2][n], dtype=np.float64) * prod
                temp[out, 3] = prod - temp[out, 0] - temp[out, 1] - temp[out, 2]
        _accumulate_ttemp_adaptive(ttemp, temp, q, j, n_out)

    return _merge_bsgs_dualrail_context_cplx(ttemp, n_out, cfg)


def compute_attention_context_dualrail_island(
    alpha_packs: list[np.ndarray],
    v_merged: list[np.ndarray],
    cfg,
    *,
    scale: float = 1.0,
    gain: float = 1.0,
    ct_ct: dict | None = None,
    alpha_copy_mode: str = "make_copies",
) -> list[np.ndarray]:
    """
    途径 A（明文）：纯实 alpha(8) + V(4) → DualRail context 岛 → 纯实 4 V-layout packs.

    Steps::

        V_cplx = complexify_v_plain(V)              # 2 complex
        A_copies = make_copies(alpha) | Softmax-layout twin
        cplx2 = numpy_cc_mm_context_dualrail(...)
        return thor_context_to_mainline_packs(cplx2, gain=gain)

    Default ``alpha_copy_mode="make_copies"`` + ``gain=1`` matches geometric /
    dense (~0). Modes ``final``/``identity`` are Softmax DualRail I/O twins
    (not bit-exact to ``make_copies``; used for HE Softmax experiments).
    """
    _he_root_on_path()
    from thor_encoder_linear_core import (  # noqa: WPS433
        _build_ct_ct_masks,
        make_copies,
    )

    if ct_ct is None:
        ct_ct = _build_ct_ct_masks(cfg, cfg.n_in_slot)

    v_cplx = complexify_v_plain(v_merged)
    if alpha_copy_mode == "make_copies":
        a_copies = make_copies(alpha_packs, cfg)
    else:
        a_copies = dualrail_alpha_copies_softmax_layout(
            alpha_packs, extract_gain=2.0, mode=alpha_copy_mode
        )
    raw = numpy_cc_mm_context_dualrail(
        v_cplx, a_copies, cfg, ct_ct, float(scale), n0_amp=1.0
    )
    return thor_context_to_mainline_packs(raw, gain=gain)


# ---------------------------------------------------------------------------
# DualRail W_O (THOR ``ThorBertAttention.dense``) — plain twin + HE
# ---------------------------------------------------------------------------


def numpy_make_rotated_copies_dualrail(
    cts: list[np.ndarray], *, pack: int = 16, stride: int = 2**11
) -> list[np.ndarray]:
    """Plain twin of THOR ``make_rotated_copies`` (``rotate_left(..., +stride)``)."""
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    out: list[np.ndarray] = []
    for ct in cts:
        cur = np.asarray(ct, dtype=np.complex128).copy()
        out.append(cur)
        for _ in range(pack - 1):
            cur = rotate_left(cur, stride)
            out.append(cur)
    return out


def numpy_pt_ct_matmul_dualrail(
    w_msgs: np.ndarray,
    x_rots: list[np.ndarray],
    cfg,
    *,
    mode: str = "block_diag_1",
) -> list[np.ndarray]:
    """
    Plain twin of THOR ``pt_ct_matmul`` (complex DualRail).

    ``w_msgs`` shape ``(n_out_p, ll, n_in_c)``; ``x_rots`` length ``n_in_c``.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_internal  # noqa: WPS433

    n_out_p, ll, n_in_c = w_msgs.shape
    if len(x_rots) != n_in_c:
        raise ValueError(f"expected {n_in_c} rotated copies, got {len(x_rots)}")
    pack = 16
    if mode == "block_diag_1":
        rot_base = 12  # THOR hardcodes delta=12-l for block_diag_1
    else:
        rot_base = 6
    outs: list[np.ndarray] = []
    for out in range(n_out_p):
        diags: list[np.ndarray] = []
        for l in range(ll):
            acc = np.asarray(w_msgs[out, l, 0], dtype=np.complex128) * np.asarray(
                x_rots[(pack * out) % n_in_c], dtype=np.complex128
            )
            for n in range(1, n_in_c):
                acc = acc + np.asarray(
                    w_msgs[out, l, n], dtype=np.complex128
                ) * np.asarray(
                    x_rots[(pack * out + n) % n_in_c], dtype=np.complex128
                )
            diags.append(acc)
        acc = diags[0].copy()
        for l in range(1, ll):
            acc = acc + rotate_internal(
                diags[l], rot_base - l, mode, cfg  # type: ignore[arg-type]
            )
        outs.append(acc)
    return outs


def wo_dense_fold_conj_plain(wx: list[np.ndarray]) -> list[np.ndarray]:
    """
    Plain twin of THOR ``dense`` post-PC-MM (no bias):

      mask (%16 < 6 → 0) → rotate_left(+6) → add → +conjugate → 2·Re
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    if not wx:
        return []
    ns = int(np.asarray(wx[0]).shape[0])
    mask1 = np.ones(ns, dtype=np.float64)
    mask1[np.arange(ns) % 16 < 6] = 0.0
    out: list[np.ndarray] = []
    for z in wx:
        zc = np.asarray(z, dtype=np.complex128)
        folded = zc + rotate_left(zc * mask1, 6)
        out.append(np.real(folded + np.conjugate(folded)).astype(np.float64))
    return out


def wo_emb_duplicate_head_slots(vec: np.ndarray) -> np.ndarray:
    """
    Match ``encode_input_lower`` / embedding DualRail ``b % 6`` layout:

    copy head slots ``d → d+6`` for ``d∈[0,6)``; zero pad ``d≥12``.
    THOR dense fold leaves high-half slots not identical to this duplicate;
    for oracle vs ``encode_input_lower(Y)`` apply this after fold/+conj.
    """
    o = np.asarray(vec, dtype=np.float64).copy()
    idx = np.arange(o.shape[0])
    for d in range(6):
        o[idx % 16 == d + 6] = o[idx % 16 == d]
    o[idx % 16 >= 12] = 0.0
    return o


def max_abs_wo_head6(a: np.ndarray, b: np.ndarray) -> float:
    """max|a-b| on slots with ``%16 < 6`` (meaningful head bands after fold)."""
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    m = (np.arange(aa.shape[0]) % 16) < 6
    return float(np.max(np.abs(aa[m] - bb[m])))


def compute_wo_dualrail_plain(
    ctx_cplx: list[np.ndarray],
    w_o: np.ndarray,
    cfg,
    *,
    scale: float = 1.0,
    out_packs: int | None = None,
    to_input_lower: bool = True,
) -> list[np.ndarray]:
    """
    Plain DualRail W_O twin (THOR path, no bias)::

      ctx 2-complex → make_rotated (32) → PC-MM (n_in=64) → fold/+conj
      → optional emb-duplicate → 8 real packs ≡ ``encode_input_lower(C@Wo.T)``
    """
    if len(ctx_cplx) != 2:
        raise ValueError(f"expected 2 context complex packs, got {len(ctx_cplx)}")
    w_msgs = pack_weight_wo_dualrail(w_o, scale=scale, out_packs=out_packs)
    x_rots = numpy_make_rotated_copies_dualrail(ctx_cplx)
    wx = numpy_pt_ct_matmul_dualrail(w_msgs, x_rots, cfg)
    outs = wo_dense_fold_conj_plain(wx)
    if to_input_lower:
        outs = [wo_emb_duplicate_head_slots(p) for p in outs]
    return outs


def encode_weight_wo_dualrail(
    engine,
    w: np.ndarray,
    *,
    level: int,
    scale: float = 1.0,
    out_packs: int = 8,
):
    """Encode DualRail W_O weight PTs at ``level`` (= CT ``level_calc``)."""
    msgs = pack_weight_wo_dualrail(w, scale=scale, out_packs=out_packs)
    n_out_p, ll, n_in_c = msgs.shape
    pts = np.full((n_out_p, ll, n_in_c), None, dtype=object)
    total = n_out_p * ll * n_in_c
    done = 0
    t0 = time.time()
    for out in range(n_out_p):
        for l in range(ll):
            for n in range(n_in_c):
                pts[out, l, n] = engine.encode(msgs[out, l, n], level=level)
                done += 1
                if done == 1 or done == total or done % 32 == 0:
                    print(
                        f"  Wo weight PT {done}/{total} "
                        f"({time.time()-t0:.1f}s)",
                        flush=True,
                    )
    return pts


def wo_pcmm_he(
    engine,
    evaluator,
    ctx_cplx,
    w_pts,
    *,
    input_level: int | None = None,
    fold_conj: bool = True,
):
    """
    DualRail W_O HE: rotate → pt_ct_matmul → optional fold/+conj (no bias).

    ``ctx_cplx``: length-2 complex CTs. ``w_pts``: ``(n_out_p, ll, 32)``.
    """
    x = np.full((ctx_cplx.shape[0],), None, dtype=object)
    for i in range(ctx_cplx.shape[0]):
        if input_level is None or ctx_cplx[i].level_calc == input_level:
            x[i] = ctx_cplx[i]
        else:
            x[i] = safe_level_up(engine, ctx_cplx[i], input_level)

    rots = evaluator.make_rotated_copies(x)
    wx = evaluator.pt_ct_matmul(w_pts, rots, mode="block_diag_1")
    if not fold_conj:
        return wx

    mask1 = np.ones((engine.num_slots,), dtype=int)
    mask1[np.arange(engine.num_slots) % 16 < 6] = 0
    lev = int(rots[0].level_calc)
    out = np.full((wx.shape[0],), None, dtype=object)
    for i in range(wx.shape[0]):
        # THOR dense: mc_mult + rotate(+6) + level_up(wx) + add + conj
        temp = engine.rotate_left(engine.mc_mult(mask1, wx[i]), 6)
        folded = engine.cc_add(engine.level_up(wx[i], lev + 3), temp)
        out[i] = engine.cc_add(folded, engine.conjugate(folded))
    return out


# ---------------------------------------------------------------------------
# DualRail FF (THOR ``ThorBertFF.dense1`` / ``dense2``) — plain twin + HE
# ---------------------------------------------------------------------------


def ff_input_rots_from_lower8_plain(
    x8: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Plain twin of THOR ``ThorBertFF.forward`` input prep (before dense1)::

      temp = x[i] + i·x[i+4]
      mask (%16 >= 6 → 0)
      root = temp + rotate_left(temp, -8)
      → 16 rotated copies per root  → 64 complex
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    if len(x8) != 8:
        raise ValueError(f"expected 8 input_lower packs, got {len(x8)}")
    ns = int(np.asarray(x8[0]).shape[0])
    mask = np.ones(ns, dtype=np.float64)
    mask[np.arange(ns) % 16 >= 6] = 0.0
    roots: list[np.ndarray] = []
    for i in range(4):
        temp = np.asarray(x8[i], dtype=np.complex128) + 1j * np.asarray(
            x8[i + 4], dtype=np.complex128
        )
        temp = temp * mask
        roots.append(temp + rotate_left(temp, -8))
    return numpy_make_rotated_copies_dualrail(roots)


def ff_dense2_input_rots_plain(
    fc1_28: list[list[np.ndarray]],
) -> list[list[np.ndarray]]:
    """
    Plain twin of dense2 input complexify + rotate (no mask/-8)::

      l[n, 16*i] = fc1[n][i] + i·fc1[n][i+4]  → 64 copies per rep
    """
    if len(fc1_28) != 2 or any(len(r) != 8 for r in fc1_28):
        raise ValueError("expected fc1 shape (2,8)")
    out: list[list[np.ndarray]] = []
    for n in range(2):
        roots = [
            np.asarray(fc1_28[n][i], dtype=np.complex128)
            + 1j * np.asarray(fc1_28[n][i + 4], dtype=np.complex128)
            for i in range(4)
        ]
        out.append(numpy_make_rotated_copies_dualrail(roots))
    return out


def _encode_bias_msg_plain(
    b: np.ndarray,
    *,
    n_blocks: int,
    n_out: int = 128,
    pack: int = 16,
    n_slot: int = 16,
    pad_index: tuple[int, ...] | None = None,
    scale: float = 1.0,
    num_slots: int = 2**15,
) -> list[np.ndarray]:
    """Plain twin of ``ThorModelEncoder._encode_b`` (slot message only, no CKKS)."""
    if pad_index is None:
        pad_index = tuple(i for i in range(n_slot) if i >= n_blocks)
    pad_set = set(int(i) for i in pad_index)
    if b.shape[0] % n_blocks != 0:
        raise ValueError(f"bias len {b.shape[0]} not divisible by n_blocks={n_blocks}")
    blocks = np.split(np.asarray(b, dtype=np.float64), n_blocks)
    n_out_packed = n_out // pack
    dim = 128
    out: list[np.ndarray] = []
    for out_i in range(n_out_packed):
        msg = np.zeros(int(num_slots), dtype=np.float64)
        for j in range(pack):
            temp = j * (2**11)
            r = out_i * pack + j
            for t in range(dim):
                c = 0
                for d in range(n_slot):
                    if d in pad_set:
                        continue
                    block = blocks[c]
                    msg[temp + t * n_slot + d] = (
                        float(scale) * block[(r + t) % block.shape[0]]
                    ) / 2.0
                    c += 1
        out.append(msg)
    return out


def encode_ff_dense1_bias_plain(
    b1: np.ndarray,
    cfg,
    *,
    scale: float = 1.0 / 64.0,
) -> list[list[np.ndarray]]:
    """
    FC1 bias → ``(2, 8)`` slot packs (``1/C`` baked, THOR ``pad_index['ff']``).
    """
    halves = np.split(np.asarray(b1, dtype=np.float64), 2)
    pad = (6, 7, 14, 15)
    return [
        _encode_bias_msg_plain(
            halves[rep],
            n_blocks=12,
            pad_index=pad,
            scale=float(scale),
            num_slots=int(cfg.num_slots),
        )
        for rep in range(2)
    ]


def encode_ff_dense2_bias_plain(b2: np.ndarray, cfg) -> list[np.ndarray]:
    """FC2 bias → 8 ``input_lower`` slot packs (``n_blocks=6``)."""
    return _encode_bias_msg_plain(
        np.asarray(b2, dtype=np.float64),
        n_blocks=6,
        scale=1.0,
        num_slots=int(cfg.num_slots),
    )


def ff_dense1_plain(
    x_rots: list[np.ndarray],
    w1: np.ndarray,
    cfg,
    *,
    scale: float = 1.0,
    out_packs: int | None = None,
    bias_28: list[list[np.ndarray]] | None = None,
) -> list[list[np.ndarray]]:
    """
    Plain dense1: two ``block_diag_2`` PC-MM + optional bias + conjugate → ``(2, n_out_p)`` real.

    Production THOR uses ``scale=1/C`` on W and bias (per-layer GeLU domain C).
    """
    w_msgs = pack_weight_ff_dualrail(
        w1, split="vertical", scale=scale, out_packs=out_packs
    )
    reps: list[list[np.ndarray]] = []
    for rep in range(2):
        wx = numpy_pt_ct_matmul_dualrail(
            w_msgs[rep], x_rots, cfg, mode="block_diag_2"
        )
        row: list[np.ndarray] = []
        for j, z in enumerate(wx):
            zc = np.asarray(z, dtype=np.complex128)
            if bias_28 is not None:
                zc = zc + np.asarray(bias_28[rep][j], dtype=np.float64)
            row.append(np.real(zc + np.conjugate(zc)).astype(np.float64))
        reps.append(row)
    return reps


def ff_dense2_plain(
    fc1_28: list[list[np.ndarray]],
    w2: np.ndarray,
    cfg,
    *,
    scale: float = 1.0,
    out_packs: int | None = None,
    to_input_lower: bool = True,
    bias8: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """
    Plain dense2: per-rep rotate + PC-MM → sum → ``+rotate(+8)`` → +bias → +conj
    → optional emb-duplicate → 8 real input_lower packs.
    """
    _he_root_on_path()
    from thor_encoder_linear_core import rotate_left  # noqa: WPS433

    w_msgs = pack_weight_ff_dualrail(
        w2, split="horizontal", scale=scale, out_packs=out_packs
    )
    n_out_p = w_msgs.shape[1]
    rots_reps = ff_dense2_input_rots_plain(fc1_28)
    wx_reps = [
        numpy_pt_ct_matmul_dualrail(
            w_msgs[rep], rots_reps[rep], cfg, mode="block_diag_2"
        )
        for rep in range(2)
    ]
    outs: list[np.ndarray] = []
    for i in range(n_out_p):
        x_out = wx_reps[0][i] + wx_reps[1][i]
        x_out = x_out + rotate_left(x_out, 8)
        zc = np.asarray(x_out, dtype=np.complex128)
        if bias8 is not None:
            zc = zc + np.asarray(bias8[i], dtype=np.float64)
        real = np.real(zc + np.conjugate(zc)).astype(np.float64)
        if to_input_lower:
            real = wo_emb_duplicate_head_slots(real)
        outs.append(real)
    return outs


def compute_ff_dualrail_plain(
    x8: list[np.ndarray],
    w1: np.ndarray,
    w2: np.ndarray,
    cfg,
    *,
    scale1: float = 1.0,
    scale2: float = 1.0,
    out_packs: int | None = None,
    to_input_lower: bool = True,
) -> list[np.ndarray]:
    """
    Plain DualRail FF chain (no GeLU / no bias)::

      input_lower(8) → FF roots/rots(64) → dense1 (2,8)
        → dense2 → 8 real ≡ encode_input_lower((X@W1.T)@W2.T)
    """
    x_rots = ff_input_rots_from_lower8_plain(x8)
    fc1 = ff_dense1_plain(x_rots, w1, cfg, scale=scale1, out_packs=out_packs)
    return ff_dense2_plain(
        fc1,
        w2,
        cfg,
        scale=scale2,
        out_packs=out_packs,
        to_input_lower=to_input_lower,
    )


def encode_weight_ff_dualrail(
    engine,
    w: np.ndarray,
    *,
    split: str,
    level: int,
    scale: float = 1.0,
    out_packs: int = 8,
):
    """Encode DualRail FF weight PTs at ``level``."""
    msgs = pack_weight_ff_dualrail(
        w, split=split, scale=scale, out_packs=out_packs
    )
    n_rep, n_out_p, ll, n_in_c = msgs.shape
    pts = np.full((n_rep, n_out_p, ll, n_in_c), None, dtype=object)
    total = n_rep * n_out_p * ll * n_in_c
    done = 0
    t0 = time.time()
    for rep in range(n_rep):
        for out in range(n_out_p):
            for l in range(ll):
                for n in range(n_in_c):
                    pts[rep, out, l, n] = engine.encode(
                        msgs[rep, out, l, n], level=level
                    )
                    done += 1
                    if done == 1 or done == total or done % 64 == 0:
                        print(
                            f"  FF-{split} PT {done}/{total} "
                            f"({time.time()-t0:.1f}s)",
                            flush=True,
                        )
    return pts


def ff_dense1_he(engine, evaluator, x_rots, w1_pts):
    """HE dense1: two block_diag_2 PC-MM + conj → (2, n_out_p)."""
    n_out_p = w1_pts.shape[1]
    out = np.full((2, n_out_p), None, dtype=object)
    for rep in range(2):
        wx = evaluator.pt_ct_matmul(w1_pts[rep], x_rots, mode="block_diag_2")
        for j in range(n_out_p):
            out[rep, j] = engine.cc_add(wx[j], engine.conjugate(wx[j]))
    return out


def ff_dense2_he(engine, evaluator, fc1_cts, w2_pts):
    """
    HE dense2: complexify+rotate per rep → PC-MM → sum → +rot8 → +conj.
    ``fc1_cts`` shape (2, n_out_p); ``w2_pts`` shape (2, n_out_p, ll, 64).
    """
    n_out_p = w2_pts.shape[1]
    wx_reps = []
    for n in range(2):
        roots = np.full((4,), None, dtype=object)
        for i in range(4):
            roots[i] = engine.cc_add(
                fc1_cts[n, i], engine.imult(fc1_cts[n, i + 4])
            )
        rots = evaluator.make_rotated_copies(roots)
        wx_reps.append(
            evaluator.pt_ct_matmul(w2_pts[n], rots, mode="block_diag_2")
        )
    result = np.full((n_out_p,), None, dtype=object)
    for i in range(n_out_p):
        x_out = engine.cc_add(wx_reps[0][i], wx_reps[1][i])
        x_out = engine.cc_add(x_out, engine.rotate_left(x_out, 8))
        result[i] = engine.cc_add(x_out, engine.conjugate(x_out))
    return result


def ff_input_rots_he(engine, x8_cts):
    """HE twin of ``ff_input_rots_from_lower8_plain`` (8 real CTs → 64 complex)."""
    mask = np.ones((engine.num_slots,), dtype=int)
    mask[np.arange(engine.num_slots) % 16 >= 6] = 0
    roots = np.full((4,), None, dtype=object)
    for i in range(4):
        temp = engine.cc_add(x8_cts[i], engine.imult(x8_cts[i + 4]))
        temp = engine.mc_mult(mask, temp)
        roots[i] = engine.cc_add(temp, engine.rotate_left(temp, -8))
    rots = np.full((64,), None, dtype=object)
    for i in range(4):
        rots[16 * i] = roots[i]
        for j in range(1, 16):
            rots[16 * i + j] = engine.rotate_left(rots[16 * i + j - 1], 2**11)
    return rots
