#!/usr/bin/env python3
"""
RESEARCH ONLY — not for production inference.

Unified layer-entry slot transform F and inverse for decode-oracle work.
Matches smoke ``engine.rotate_left(v, -6)`` on each real pack message:

    F(v) = v + roll(v, -6)

Old L0 layout: ``encode_input_lower_diagonals`` (production encrypt path).
Research new pack messages: ``inv_f_slot(old_pack)`` so ``F(new) == old`` on valid slots.

Production entry also uses 8 real → 4 complex then ``F`` on each complex CT
(``cc_add(z, rotate_left(z,-6))``). HE validation mirrors that on ciphertext.

**Layer output decode (research):** ``(128,768) → encode_research_new_packs → 8 real``;
restore via ``decode_research_real8_to_matrix`` (= ``F`` on each pack, then
``decode_input_from_lower_diagonals``). Independent of the ``ln2_out`` probe name —
any 8-real tensor in **new** pack layout uses the same inverse.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HE = Path(__file__).resolve().parents[1]
if str(_HE) not in sys.path:
    sys.path.insert(0, str(_HE))

from thor_encoder_linear_core import ThorConfig, bert_base, encode_input_lower_diagonals

# CKKS emb-duplicate step in forward (per pack / per cplx message).
F_ROT_DELTA = 6


def slot_rotate_left(v: np.ndarray, delta: int) -> np.ndarray:
    """Plain slot vector rotation (``engine.rotate_left(v, delta)`` on messages)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    d = int(delta) % n
    if d == 0:
        return v.copy()
    return np.roll(v, -d)


def f_slot(v: np.ndarray, *, rot: int = F_ROT_DELTA) -> np.ndarray:
    """8-real F: v + rotate_left(v, -rot). Same as one step of L1+ entry on messages."""
    v = np.asarray(v, dtype=np.float64).ravel()
    return v + slot_rotate_left(v, -rot)


def inv_f_slot(y: np.ndarray, *, rot: int = F_ROT_DELTA) -> np.ndarray:
    """Solve w + roll(w,-rot) = y (circular). Research inverse for new pack encoding."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    k = np.arange(n, dtype=np.float64)
    # roll(w,-rot)[i]=w[i+rot]  =>  (I + P^{-rot}) w = y  in frequency domain
    denom = 1.0 + np.exp(-2j * np.pi * k * rot / n)
    w = np.fft.ifft(np.fft.fft(y) / denom)
    return np.real(w).astype(np.float64)


def valid_head_slot_mask(n: int) -> np.ndarray:
    """Meaningful head bands (``%16 < 6``), same as ``max_abs_wo_head6``."""
    return (np.arange(n) % 16) < 6


def max_abs_on_valid(a: np.ndarray, b: np.ndarray) -> float:
    m = valid_head_slot_mask(a.size)
    return float(np.max(np.abs(a[m] - b[m])))


def mean_abs_on_valid(a: np.ndarray, b: np.ndarray) -> float:
    m = valid_head_slot_mask(a.size)
    return float(np.mean(np.abs(a[m] - b[m])))


def encode_old_input_lower(h: np.ndarray, cfg: ThorConfig | None = None) -> list[np.ndarray]:
    cfg = cfg or bert_base()
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (cfg.seq_len, cfg.hidden_dim):
        raise ValueError(f"expected ({cfg.seq_len},{cfg.hidden_dim}), got {h.shape}")
    return encode_input_lower_diagonals(h, cfg)


def encode_research_new_packs(h: np.ndarray, cfg: ThorConfig | None = None) -> list[np.ndarray]:
    """Research pack messages: inv_f(old encode). NOT for production encrypt."""
    old = encode_old_input_lower(h, cfg)
    return [inv_f_slot(v) for v in old]


def pack8_real_to_cplx4_plain(real8: list[np.ndarray]) -> list[np.ndarray]:
    """DualRail ``re + i·im`` from 8 decrypted real slot vectors."""
    if len(real8) != 8:
        raise ValueError(f"expected 8 packs, got {len(real8)}")
    out: list[np.ndarray] = []
    for i in range(4):
        re = np.asarray(real8[i], dtype=np.float64).ravel()
        im = np.asarray(real8[i + 4], dtype=np.float64).ravel()
        out.append(re + 1j * im)
    return out


def f_cplx4_plain(cplx4: list[np.ndarray], *, rot: int = F_ROT_DELTA) -> list[np.ndarray]:
    """Forward F on 4 complex slot messages (matches L1+ ``x_cplx`` step)."""
    out: list[np.ndarray] = []
    for z in cplx4:
        z = np.asarray(z, dtype=np.complex128).ravel()
        out.append(z + np.roll(z, -rot))
    return out


def x_cplx_at_qkv_from_real8_plain(
    real8: list[np.ndarray],
    *,
    apply_f: bool,
) -> list[np.ndarray]:
    """
    Plain twin of ``_forward_layer`` entry through ``make_rotated_copies``:

    428–431 ``8→4 cplx``; optional 464–467 ``F`` on each cplx (L1+ / unified L0).
    """
    cplx = pack8_real_to_cplx4_plain(real8)
    if apply_f:
        cplx = f_cplx4_plain(cplx)
    return cplx


def validate_unified_entry_at_qkv(
    h: np.ndarray,
    cfg: ThorConfig | None = None,
    *,
    atol: float = 1e-9,
) -> tuple[bool, float, float]:
    """
    New: ``NewEncode(h)`` → 8→4 → F  vs  old: ``encode_old(h)`` → 8→4 only.

    Target ``x_cplx`` at QKV (before ``make_rotated_copies``).
    """
    cfg = cfg or bert_base()
    old = encode_old_input_lower(h, cfg)
    new = encode_research_new_packs(h, cfg)
    ref = x_cplx_at_qkv_from_real8_plain(old, apply_f=False)
    test = x_cplx_at_qkv_from_real8_plain(new, apply_f=True)
    worst = max(max_abs_valid_cplx(test[i], ref[i]) for i in range(4))
    mean = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        test[i][valid_head_slot_mask(test[i].size)]
                        - ref[i][valid_head_slot_mask(ref[i].size)]
                    )
                )
                for i in range(4)
            ]
        )
    )
    return worst <= atol, worst, mean


def max_abs_valid_cplx(a: np.ndarray, b: np.ndarray) -> float:
    m = valid_head_slot_mask(a.size)
    return float(np.max(np.abs(a[m] - b[m])))


def validate_f_cplx_after_decrypt(
    old_real8: list[np.ndarray],
    new_real8: list[np.ndarray],
    *,
    atol: float = 1e-9,
) -> tuple[bool, float, float]:
    """
    Reference: cplx from **old** decrypt (L0, no F on CT).
    Test: F(cplx(**new** decrypt)) == ref on valid slots.
    """
    ref = pack8_real_to_cplx4_plain(old_real8)
    test = f_cplx4_plain(pack8_real_to_cplx4_plain(new_real8))
    worst = max(max_abs_valid_cplx(test[i], ref[i]) for i in range(4))
    mean = float(
        np.mean(
            [
                np.mean(np.abs(test[i][valid_head_slot_mask(test[i].size)] - ref[i][valid_head_slot_mask(ref[i].size)]))
                for i in range(4)
            ]
        )
    )
    return worst <= atol, worst, mean


def apply_cplx4_from_real8_cts(engine, x8_ct) -> np.ndarray:
    """8 real CTs → 4 cplx (``cc_add(re, imult(im))``), **no** emb-duplicate F."""
    x8 = list(x8_ct)
    x_cplx = np.full((4,), None, dtype=object)
    for i in range(4):
        x_cplx[i] = engine.cc_add(x8[i], engine.imult(x8[i + 4]))
    return x_cplx


def apply_f_on_real8_cts(engine, x8_ct) -> np.ndarray:
    """8 real CTs → 4 cplx → F (rotate_left -6) on ciphertext (L1+ entry)."""
    x_cplx = apply_cplx4_from_real8_cts(engine, x8_ct)
    for i in range(4):
        x_cplx[i] = engine.cc_add(
            x_cplx[i], engine.rotate_left(x_cplx[i], -F_ROT_DELTA)
        )
    return x_cplx


def decrypt_real8(engine, sk, x8_ct) -> list[np.ndarray]:
    return [
        np.asarray(engine.decrode(x8_ct[i], sk, is_real=True), dtype=np.float64)
        for i in range(8)
    ]


def decrypt_cplx4(engine, sk, cplx4_ct) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in range(4):
        d = engine.decrode(cplx4_ct[i], sk, is_real=False)
        out.append(np.asarray(d, dtype=np.complex128).ravel())
    return out


def validate_f_roundtrip_on_packs(
    old_packs: list[np.ndarray],
    new_packs: list[np.ndarray],
    *,
    atol: float = 1e-9,
) -> tuple[bool, float, float]:
    if len(old_packs) != len(new_packs):
        raise ValueError("pack count mismatch")
    worst = 0.0
    mean = 0.0
    n = 0
    for o, nw in zip(old_packs, new_packs):
        f_nw = f_slot(nw)
        worst = max(worst, max_abs_on_valid(f_nw, o))
        mean += mean_abs_on_valid(f_nw, o)
        n += 1
    mean /= max(n, 1)
    ok = worst <= atol
    return ok, worst, mean


def decode_matrix_from_packs(
    packs: list[np.ndarray], cfg: ThorConfig | None = None
) -> np.ndarray:
    """Inverse of ``encode_input_lower_diagonals`` (production / old pack layout)."""
    from slot_decode import decode_input_from_lower_diagonals

    cfg = cfg or bert_base()
    return decode_input_from_lower_diagonals(packs, cfg)


def decode_research_real8_to_matrix(
    real8: list[np.ndarray],
    cfg: ThorConfig | None = None,
) -> np.ndarray:
    """
    Research layer-output decode: **new-encoded** 8 real slot packs → ``(seq, hidden)``.

    ``encode_research_new_packs(h)`` stores ``inv_f(old_encode(h))``; applying ``F``
    per pack recovers the old diagonal layout before standard matrix decode.
    """
    if len(real8) != 8:
        raise ValueError(f"expected 8 real packs, got {len(real8)}")
    cfg = cfg or bert_base()
    old_layout = [f_slot(np.asarray(p, dtype=np.float64).ravel()) for p in real8]
    return decode_matrix_from_packs(old_layout, cfg)


def production_real8_to_research_new(
    real8_production: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Bridge: mock / production layer exit still emits **old** slot packs today.

    Research new packs ``N`` satisfy ``F(N) ≈ O`` for production packs ``O`` decrypt.
    """
    return [inv_f_slot(np.asarray(p, dtype=np.float64).ravel()) for p in real8_production]


def hidden_matrix_error_stats(
    he_mat: np.ndarray,
    ref_mat: np.ndarray,
    *,
    rel_floor: float = 1e-6,
) -> dict[str, float]:
    """Element-wise (128×768) HE decode vs poly; includes masked relative error."""
    he = np.asarray(he_mat, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_mat, dtype=np.float64).reshape(-1)
    n = min(he.size, ref.size)
    he, ref = he[:n], ref[:n]
    err = he - ref
    abs_err = np.abs(err)
    denom = float(np.dot(he, he))
    ls_gain = float(np.dot(he, ref) / denom) if denom > 1e-30 else float("nan")
    he_ls = he * ls_gain if np.isfinite(ls_gain) else he
    abs_ls = float(np.max(np.abs(he_ls - ref)))
    mask = np.abs(ref) >= rel_floor
    rel = abs_err[mask] / np.abs(ref[mask]) if np.any(mask) else np.array([float("nan")])
    corr = (
        float(np.corrcoef(he, ref)[0, 1])
        if np.std(he) > 1e-15 and np.std(ref) > 1e-15
        else float("nan")
    )
    rmse_g1 = float(np.sqrt(np.mean(abs_err**2)))
    max_ref = float(np.max(np.abs(ref)))
    mean_ref = float(np.mean(np.abs(ref)))
    return {
        "max_abs_g1": float(np.max(abs_err)),
        "mean_abs_g1": float(np.mean(abs_err)),
        "rmse_g1": rmse_g1,
        "max_rel_g1": float(np.max(rel)) if rel.size else float("nan"),
        "mean_rel_g1": float(np.mean(rel)) if rel.size else float("nan"),
        "max_rel_pct": float(np.max(rel) * 100.0) if rel.size else float("nan"),
        "mean_rel_pct": float(np.mean(rel) * 100.0) if rel.size else float("nan"),
        "max_abs_ls": abs_ls,
        "ls_gain": ls_gain,
        "corr": corr,
        "n_elem": int(n),
        "n_rel_mask": int(np.sum(mask)),
        "max_abs_ref": max_ref,
        "mean_abs_ref": mean_ref,
    }


def decode_layer_output_to_matrix(real8: list[np.ndarray]) -> np.ndarray:
    """Layer exit 8-real decrypt → (128,768) via ``decode_research_real8_to_matrix``."""
    return decode_research_real8_to_matrix(real8)


def run_he_validation(
    h: np.ndarray,
    *,
    enc_level: int = 15,
    atol_plain: float = 1e-9,
    atol_he: float = 0.05,
) -> int:
    """Encrypt old/new 8-real packs; F on cplx CT for new; compare decrypts."""
    _thor = Path(__file__).resolve().parent
    if str(_thor) not in sys.path:
        sys.path.insert(0, str(_thor))
    from bts_ops import DEFAULT_L0_ENTRY_REMAINING, remaining_to_level_calc
    from engine import THOR_DEFAULT_PARAMS, create_engine, load_thor_keys

    cfg = bert_base()
    old_msgs = encode_old_input_lower(h, cfg)
    new_msgs = encode_research_new_packs(h, cfg)

    print("create engine + keys0 (rot only, no bootstrap) ...", flush=True)
    engine = create_engine(dict(THOR_DEFAULT_PARAMS))
    keys = load_thor_keys(engine, with_bootstrap=False)
    sk, pk = keys["sk"], keys["pk"]
    if enc_level is None:
        enc_level = remaining_to_level_calc(
            DEFAULT_L0_ENTRY_REMAINING, int(engine.num_levels)
        )
    print(f"encrypt/decrypt level_calc={enc_level}", flush=True)

    old_ct = np.empty(8, dtype=object)
    new_ct = np.empty(8, dtype=object)
    for i in range(8):
        old_ct[i] = engine.encode_and_encrypt(old_msgs[i], pk, level=enc_level)
        new_ct[i] = engine.encode_and_encrypt(new_msgs[i], pk, level=enc_level)

    old_dec = decrypt_real8(engine, sk, old_ct)
    new_dec = decrypt_real8(engine, sk, new_ct)

    ok8, w8, m8 = validate_f_roundtrip_on_packs(old_msgs, new_msgs, atol=atol_plain)
    print(
        f"[plain pack] F_real(new_msg) vs old_msg: max={w8:.3e} mean={m8:.3e} PASS={ok8}",
        flush=True,
    )

    ok_c_plain, wc, mc = validate_f_cplx_after_decrypt(
        old_dec, new_dec, atol=atol_plain
    )
    print(
        f"[decrypt→plain cplx] F(cplx(new_dec)) vs cplx(old_dec): "
        f"max={wc:.3e} mean={mc:.3e} PASS={ok_c_plain}",
        flush=True,
    )

    # L0 reference: decrypt 8 real old_ct → DualRail 4 cplx (production skips F).
    ref_cplx = pack8_real_to_cplx4_plain(old_dec)

    old_cplx_ct = apply_cplx4_from_real8_cts(engine, old_ct)
    he_old_cplx = decrypt_cplx4(engine, sk, old_cplx_ct)
    worst_old = max(
        max_abs_valid_cplx(he_old_cplx[i], ref_cplx[i]) for i in range(4)
    )
    mean_old = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        he_old_cplx[i][valid_head_slot_mask(he_old_cplx[i].size)]
                        - ref_cplx[i][valid_head_slot_mask(ref_cplx[i].size)]
                    )
                )
                for i in range(4)
            ]
        )
    )
    ok_old = worst_old <= atol_he
    print(
        f"[HE L0 baseline] decrypt(8→4 cplx on old_ct, no F) vs cplx(old_dec): "
        f"max={worst_old:.3e} mean={mean_old:.3e} atol={atol_he:g} PASS={ok_old}",
        flush=True,
    )

    f_new_ct = apply_f_on_real8_cts(engine, new_ct)
    he_cplx = decrypt_cplx4(engine, sk, f_new_ct)
    worst_he = max(max_abs_valid_cplx(he_cplx[i], ref_cplx[i]) for i in range(4))
    mean_he = float(
        np.mean(
            [
                np.mean(
                    np.abs(
                        he_cplx[i][valid_head_slot_mask(he_cplx[i].size)]
                        - ref_cplx[i][valid_head_slot_mask(ref_cplx[i].size)]
                    )
                )
                for i in range(4)
            ]
        )
    )
    ok_he = worst_he <= atol_he
    print(
        f"[HE cplx F on CT] decrypt(F(new_ct)) vs cplx(old_dec): "
        f"max={worst_he:.3e} mean={mean_he:.3e} atol={atol_he:g} PASS={ok_he}",
        flush=True,
    )

    # Negative control: F on old_ct must *not* match L0 ref (L0 never applies F).
    f_old_ct = apply_f_on_real8_cts(engine, old_ct)
    he_old_f = decrypt_cplx4(engine, sk, f_old_ct)
    drift_f = max(
        max_abs_valid_cplx(he_old_f[i], ref_cplx[i]) for i in range(4)
    )
    print(
        f"[neg ctrl] decrypt(F(old_ct)) vs cplx(old_dec) max={drift_f:.3e} "
        f"(should be large; not an equality check)",
        flush=True,
    )

    return 0 if (ok8 and ok_c_plain and ok_he and ok_old) else 1


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Research: F / invF slot encode validation")
    ap.add_argument("--he", action="store_true", help="run CKKS encrypt/decrypt + cplx F on CT")
    ap.add_argument("--he-atol", type=float, default=0.05, help="max abs on valid cplx slots (HE)")
    ap.add_argument("--enc-level", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--hidden-npy",
        type=str,
        default=None,
        help="optional (128,768) or (1,128,768) plain hidden .npy",
    )
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument(
        "--unified-oracle",
        action="store_true",
        help="only check NewEncode→8→4→F vs old→8→4 at QKV (plain)",
    )
    args = ap.parse_args()
    cfg = bert_base()

    if args.hidden_npy:
        h = np.load(args.hidden_npy)
        h = np.asarray(h, dtype=np.float64).squeeze()
        if h.shape != (cfg.seq_len, cfg.hidden_dim):
            h = h.reshape(cfg.seq_len, cfg.hidden_dim)
    else:
        rng = np.random.default_rng(args.seed)
        h = rng.standard_normal((cfg.seq_len, cfg.hidden_dim))

    if args.he:
        return run_he_validation(
            h, enc_level=args.enc_level, atol_plain=args.atol, atol_he=args.he_atol
        )

    ok_q, w_q, m_q = validate_unified_entry_at_qkv(h, cfg, atol=args.atol)
    print(
        f"[unified QKV oracle] new(8→4→F) vs old(8→4) cplx: "
        f"max={w_q:.3e} mean={m_q:.3e} PASS={ok_q}",
        flush=True,
    )
    if args.unified_oracle:
        return 0 if ok_q else 1

    old = encode_old_input_lower(h, cfg)
    new = encode_research_new_packs(h, cfg)
    ok, worst, mean = validate_f_roundtrip_on_packs(old, new, atol=args.atol)

    print(f"packs={len(old)}  valid_mask=%16<6  atol={args.atol:g}")
    print(f"F(new) vs old (valid slots): max_abs={worst:.3e}  mean_abs={mean:.3e}  PASS={ok}")

    # Matrix round-trip sanity on old encode only (plaintext, no HE)
    mat_old = decode_matrix_from_packs(old, cfg)
    err_enc = float(np.max(np.abs(mat_old - h)))
    print(f"decode(encode_old(h)) vs h: max_abs={err_enc:.3e}")

    mat_f_new = decode_research_real8_to_matrix(new, cfg)
    err_f = float(np.max(np.abs(mat_f_new - h)))
    print(f"decode_research(new 8real) vs h: max_abs={err_f:.3e}")

    ok_c, wc, mc = validate_f_cplx_after_decrypt(
        old, new, atol=args.atol
    )
    print(
        f"F(cplx(new)) vs cplx(old) on messages: max={wc:.3e} mean={mc:.3e} PASS={ok_c}"
    )

    return 0 if (ok_q and ok and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
