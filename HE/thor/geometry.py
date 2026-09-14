"""THOR complex DualRail CT counts vs mainline pure-real SlotCtGeometry (bert_base)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThorComplexGeometry:
    """
    Ciphertext array lengths on the upstream THOR complex path (BERT-base).

    Pure-real counterparts live in ``HE/bts_ops.SlotCtGeometry``; ratios are
    documented in ``HE/thor/README.md``. Branch A ``HE/thor/bts_ops`` consumes
    this via ``SlotCtGeometry.from_complex()`` (default).
    """

    num_slots: int = 32768  # logN=16 → N/2
    seq_len: int = 128
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    pack: int = 16

    # DualRail-packed linear spine
    ct_hidden_cplx: int = 4  # layer I/O complex packs (real would be 8)
    ct_pc_rot_cplx: int = 64  # make_rotated_copies(4) → 16*4
    ct_qkv_out: int = 4  # Q/K/V packed outs (same count as real, different packing)
    ct_q_copies: int = 64  # make_copies(Q)
    ct_v_cplx: int = 2  # V packed for context
    ct_context_cplx: int = 2

    # Real segments (nonlinear / residual)
    ct_att_score: int = 8
    ct_alpha_copies: int = 128
    ct_hidden_real: int = 8  # after unpack for LN / residual
    ct_ffn: int = 16  # (2, 8) FF1 live

    @property
    def ct_pc_rot_ctx(self) -> int:
        """``make_rotated_copies`` on context/V packs: ``16 * ct_context_cplx``."""
        return 16 * int(self.ct_context_cplx)

    def ct_count_for_event(self, event_name: str) -> int:
        """Best-effort map from bts_ops-style event names → THOR CT arity."""
        name = event_name.lower()
        table = {
            "qkv": self.ct_hidden_cplx,
            "input": self.ct_hidden_cplx,
            "pc_rot_ctx": self.ct_pc_rot_ctx,
            "pc_rot": self.ct_pc_rot_cplx,
            "rotated": self.ct_pc_rot_cplx,
            "q": self.ct_qkv_out,
            "k": self.ct_qkv_out,
            "v": self.ct_qkv_out,
            "q_copies": self.ct_q_copies,
            "make_copies": self.ct_q_copies,
            "score": self.ct_att_score,
            "att_score": self.ct_att_score,
            "softmax": self.ct_att_score,
            "dualrail_exit": self.ct_alpha_copies,
            "alpha": self.ct_alpha_copies,
            "att_prob": self.ct_alpha_copies,
            "v_cplx": self.ct_v_cplx,
            "context": self.ct_context_cplx,
            "att_context": self.ct_context_cplx,
            "dense": self.ct_hidden_real,
            "wo": self.ct_hidden_real,
            "ln1": self.ct_hidden_real,
            "ln2": self.ct_hidden_real,
            "resid": self.ct_hidden_real,
            "ff": self.ct_ffn,
            "gelu": self.ct_ffn,
            "ffn": self.ct_ffn,
        }
        for key, n in table.items():
            if key in name:
                return n
        raise KeyError(f"unknown event_name for complex geometry: {event_name!r}")


def bert_base_complex() -> ThorComplexGeometry:
    return ThorComplexGeometry()


def real_vs_complex_table() -> list[tuple[str, int, int, str]]:
    """Rows: (stage, real_ct, complex_ct, note)."""
    g = bert_base_complex()
    return [
        ("hidden / resid", 8, g.ct_hidden_cplx, "DualRail 8→4"),
        ("PC-MM rotated", 128, g.ct_pc_rot_cplx, "n_in_slot/2"),
        ("Q/K/V out", 4, g.ct_qkv_out, "same count, different packing"),
        ("Q copies", 64, g.ct_q_copies, "1"),
        ("att score / softmax in", 8, g.ct_att_score, "real on both"),
        ("alpha copies", 128, g.ct_alpha_copies, "real on both"),
        ("V for context", 4, g.ct_v_cplx, "4→2 complex"),
        ("context out", 4, g.ct_context_cplx, "2 complex"),
        ("LN / W_O out real", 8, g.ct_hidden_real, "1"),
        ("FF / GeLU live", 16, g.ct_ffn, "(2,8)"),
    ]
