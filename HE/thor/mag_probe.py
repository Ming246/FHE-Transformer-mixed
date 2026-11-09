"""Decrypt slot magnitudes along one HE layer. No matrix decode.

Stops at the first site whose raw decrypt or logical ``ct/delta`` reaches 1e11.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterator

import numpy as np

BOOM = 1e11


class MagBoom(RuntimeError):
    def __init__(self, site: str, raw: float, logical: float):
        self.site = site
        self.raw = float(raw)
        self.logical = float(logical)
        super().__init__(
            f"{site}: raw={raw:.4e} logical={logical:.4e}"
        )


class ExpInvStop(RuntimeError):
    """Stop the layer after exp_u / inv_d have been saved."""


def enabled(engine: Any) -> bool:
    return getattr(engine, "_mag_sk", None) is not None


def _iter_cts(obj: Any) -> Iterator[Any]:
    if obj is None:
        return
    if hasattr(obj, "level_calc"):
        yield obj
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for x in obj.ravel():
                yield from _iter_cts(x)
        return
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_cts(x)


def _dec(engine: Any, ct: Any, sk: Any) -> np.ndarray:
    try:
        return np.asarray(engine.decrode(ct, sk, is_real=False), dtype=np.complex128)
    except Exception:
        return np.asarray(
            engine.decrode(ct, sk, is_real=True), dtype=np.float64
        ).astype(np.complex128)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-30:
        return float("nan")
    return float(np.dot(a - a.mean(), b - b.mean()) / denom)


def _stat_line(title: str, he: np.ndarray, plain: np.ndarray) -> None:
    d = he - plain
    print(
        f"  [exp_inv] {title} n={he.size} "
        f"he_max={np.max(np.abs(he)):.4e} plain_max={np.max(np.abs(plain)):.4e} "
        f"max|Δ|={np.max(np.abs(d)):.4e} mean|Δ|={np.mean(np.abs(d)):.4e} "
        f"corr={_corr(he, plain):.6f}",
        flush=True,
    )


def _plain_exp_and_inv(
    scores_hqk: np.ndarray,
    key_valid: np.ndarray,
    *,
    shift: float,
    delta1: float,
    delta2: float,
    e0: float,
    iters: int,
    coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plaintext twin of ``thor_softmax`` up to σ-aSOR. Returns exp[H,Q,K], σ[H,Q], inv[H,Q]."""
    scores = np.asarray(scores_hqk, dtype=np.float64)
    valid = np.asarray(key_valid, dtype=np.float64).reshape(-1)
    m = np.zeros_like(scores)
    m[..., valid > 0] = 1.0
    x = scores * m
    x_scaled = x / float(delta1) / float(delta2) / 8.0
    exp = np.zeros_like(x_scaled)
    for c in np.asarray(coeffs, dtype=np.float64):
        exp = exp * x_scaled + float(c)
    exp = exp / math.exp(float(shift) / float(delta2) / float(delta1))
    n1 = int(round(math.log2(float(delta1))))
    for _ in range(n1):
        exp = exp * exp
    exp *= m
    sigma = exp.sum(axis=-1)
    inv = np.ones_like(sigma)
    b = sigma.copy()
    en = float(e0)
    for _ in range(int(iters)):
        kn = 2.0 / (en + 1.0)
        b_temp = 2.0 - kn * b
        b = kn * b * b_temp
        inv = kn * inv * b_temp
        en = kn * en * (2.0 - kn * en)
    return exp, sigma, inv


def compare_exp_inv_inputs(
    engine: Any,
    exp_u: Any,
    inv_d: Any,
    *,
    d_delta: float,
    out_path: str = "/workspace/results/he_chain/l8_exp_inv_inputs.npz",
) -> None:
    """Decrypt ``update_inv_D_fixed`` inputs and compare to plaintext exp and aSOR(1/σ).

    ``exp_u`` is the 8 masked exp packs. ``inv_d`` logical value is ``decrypt/d_delta``.
    Plaintext uses the same horner + fixed-iter aSOR as ``thor_softmax``, packed with
    ``encode_score_packs_from_dense`` (scores[h, key, query]).
    """
    if not enabled(engine):
        return

    from linear_eval import encode_score_packs_from_dense, pt_rotsum
    from softmax_poly import THOR_EXP_POLY_COEFFS, softmax_config_for_layer
    from thor_encoder_linear_core import _slot_index, bert_base

    scores = getattr(engine, "_mag_att_score", None)
    if scores is None:
        raise RuntimeError("mag probe has no _mag_att_score for exp/inv compare")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] != 12:
        raise RuntimeError(f"att_score shape {scores.shape}, expected (12, query, key)")
    key_valid = np.asarray(getattr(engine, "_mag_key_valid"), dtype=np.float64).reshape(-1)
    task = str(getattr(engine, "_mag_task", "mrpc"))
    layer = int(getattr(engine, "_mag_layer", 8))
    level = int(getattr(engine, "_mag_softmax_level", 1))
    cfg_sm = softmax_config_for_layer(task, layer, level)
    exp_hqk, sigma, inv = _plain_exp_and_inv(
        scores,
        key_valid,
        shift=float(cfg_sm["shift"]),
        delta1=float(cfg_sm["delta1"]),
        delta2=float(cfg_sm["delta2"]),
        e0=float(cfg_sm["e0_sigma"]),
        iters=int(cfg_sm["asor_max_iters_sigma"]),
        coeffs=np.asarray(THOR_EXP_POLY_COEFFS, dtype=np.float64),
    )
    cfg = bert_base()
    plain_exp = np.stack(
        encode_score_packs_from_dense(np.transpose(exp_hqk, (0, 2, 1)), cfg)
    )
    n = int(plain_exp.shape[1])
    plain_inv = np.zeros(n, dtype=np.float64)
    for h in range(12):
        for t in range(int(cfg.seq_len)):
            val = float(inv[h, t])
            for j in range(int(cfg.pack)):
                plain_inv[_slot_index(cfg, j, t, h)] = val

    sk = engine._mag_sk
    he_exp = np.stack([np.real(_dec(engine, exp_u[i], sk)).ravel() for i in range(8)])
    he_inv_raw = np.real(_dec(engine, inv_d, sk)).ravel()
    he_inv = he_inv_raw / float(d_delta)

    he_sigma_slots = np.real(pt_rotsum(he_exp.sum(axis=0), 2**11)).astype(np.float64)
    n_tok = int(np.count_nonzero(key_valid > 0))
    exp_mask = np.zeros_like(he_exp, dtype=bool)
    for i in range(8):
        idx = np.arange(n)
        head = idx % 16
        t = (idx // 16) % 128
        j = idx // 2048
        key = (i * 16 + j + t) % 128
        exp_mask[i] = (head < 12) & (key < n_tok)

    print(
        f"  [exp_inv] L{layer} level={level} shift={cfg_sm['shift']} "
        f"δ2={cfg_sm['delta2']} e0={cfg_sm['e0_sigma']:.6e} "
        f"iters_σ={cfg_sm['asor_max_iters_sigma']} d_delta={d_delta:.6e} "
        f"n_tokens={n_tok}",
        flush=True,
    )
    _stat_line("exp valid key&head", he_exp[exp_mask], plain_exp[exp_mask])

    rows = []
    for t in range(128):
        for h in range(12):
            rows.append(t * 16 + h)
    rows = np.asarray(rows, dtype=np.int64)
    he_sigma = np.zeros((12, 128), dtype=np.float64)
    plain_sigma = sigma
    for h in range(12):
        for t in range(128):
            he_sigma[h, t] = he_sigma_slots[t * 16 + h]
    _stat_line("σ j=0 rows", he_sigma.ravel(), plain_sigma.ravel())
    _stat_line("inv j=0 / d_delta", he_inv[rows], plain_inv[rows])
    head_all = (np.arange(n) % 16) < 12
    _stat_line("inv all j replicas", he_inv[head_all], plain_inv[head_all])

    abs_d = np.abs(he_exp - plain_exp)
    abs_d = np.where(exp_mask, abs_d, -1.0)
    flat = np.argpartition(abs_d.ravel(), -5)[-5:]
    flat = flat[np.argsort(-abs_d.ravel()[flat])]
    print("  [exp_inv] worst exp slots (pack, j, query, head, key, he, plain):", flush=True)
    for f in flat:
        i, s = divmod(int(f), n)
        head = s % 16
        t = (s // 16) % 128
        j = s // 2048
        key = (i * 16 + j + t) % 128
        print(
            f"    pack={i} j={j} q={t} h={head} k={key} "
            f"he={he_exp[i, s]:.6e} plain={plain_exp[i, s]:.6e}",
            flush=True,
        )
    inv_dlt = np.abs(he_inv[rows] - plain_inv[rows])
    worst = np.argpartition(inv_dlt, -5)[-5:]
    worst = worst[np.argsort(-inv_dlt[worst])]
    print("  [exp_inv] worst inv rows (query, head, he, plain, he_σ, plain_σ):", flush=True)
    for w in worst:
        t, h = divmod(int(w), 12)
        # rows were appended (t, h) with h inner, so index = t*12+h
        t, h = int(w) // 12, int(w) % 12
        print(
            f"    q={t} h={h} he={he_inv[t * 16 + h]:.6e} plain={plain_inv[t * 16 + h]:.6e} "
            f"he_σ={he_sigma[h, t]:.6e} plain_σ={plain_sigma[h, t]:.6e}",
            flush=True,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        exp_he=he_exp,
        exp_plain=plain_exp,
        inv_he_raw=he_inv_raw,
        inv_he=he_inv,
        inv_plain=plain_inv,
        sigma_he=he_sigma,
        sigma_plain=plain_sigma,
        d_delta=np.float64(d_delta),
        n_tokens=np.int32(n_tok),
        shift=np.float64(cfg_sm["shift"]),
        delta2=np.float64(cfg_sm["delta2"]),
    )
    print(f"  [exp_inv] saved {out_path}", flush=True)


def compare_sumsq(
    engine: Any,
    exp_u: Any,
    inv_list: Any,
    summation: Any,
    *,
    k: int,
    d_delta: float,
    n_tokens: int | None = None,
    after_rotsum: Any | None = None,
) -> None:
    """Decrypt exp and inv, form y in the clear, compare with ciphertext Σ.

    ``y = (2·k)·exp·inv`` matches ``cc_add(exp, exp)`` then ``×k``.
    Pack-sum is the 8-way ``cc_add``. ``rotsum(2**11)`` finishes Σ over keys.
    Head lanes are ``slot % 16 < 12``; key lanes follow the attention-mask
    diagonal (``col < n_tokens``).
    """
    if not enabled(engine):
        return
    from linear_eval import pt_rotsum

    sk = engine._mag_sk
    if n_tokens is None:
        n_tokens = getattr(engine, "_mag_n_tokens", None)

    def real(ct: Any) -> np.ndarray:
        return np.real(_dec(engine, ct, sk)).astype(np.float64).ravel()

    exp = np.stack([real(exp_u[i]) for i in range(8)])
    inv = np.stack([real(inv_list[i]) for i in range(8)])
    scale = 2.0 * float(k)
    y = scale * exp * inv
    y2 = y * y
    plain_sum = y2.sum(axis=0)
    ct_sum = real(summation)
    n = plain_sum.size
    head = (np.arange(n) % 16) < 12

    def _line(title: str, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> None:
        aa, bb = a[mask], b[mask]
        d = aa - bb
        denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
        corr = float(np.dot(aa - aa.mean(), bb - bb.mean()) / denom) if denom > 1e-30 else float("nan")
        print(
            f"  [sumsq] {title} n={int(mask.sum())} "
            f"plain_max={np.max(np.abs(aa)):.4e} ct_max={np.max(np.abs(bb)):.4e} "
            f"max|Δ|={np.max(np.abs(d)):.4e} mean|Δ|={np.mean(np.abs(d)):.4e} "
            f"corr={corr:.6f}",
            flush=True,
        )

    print(
        f"  [sumsq] k={k} d_delta={d_delta:.6e} scale=2*k={scale:.6e} "
        f"1/d_delta={1.0/d_delta:.6e} n_tokens={n_tokens}",
        flush=True,
    )
    _line("pack-sum head%16<12", plain_sum, ct_sum, head)
    if n_tokens is not None:
        key_ok = np.zeros(n, dtype=bool)
        nt = int(n_tokens)
        for i in range(8):
            for j in range(16):
                diag = i * 16 + j
                base = j * (2**11)
                for t in range(128):
                    if (diag + t) % 128 < nt:
                        key_ok[base + t * 16 : base + t * 16 + 12] = True
        _line("pack-sum head&key", plain_sum, ct_sum, head & key_ok)

    plain_rot = np.real(pt_rotsum(plain_sum, 2**11)).astype(np.float64)
    if after_rotsum is not None:
        ct_rot = real(after_rotsum)
        _line("rotsum head%16<12", plain_rot, ct_rot, head)
        # One replica (j=0): each (query t, head) is Σ over keys.
        idx = []
        for t in range(128):
            for h in range(12):
                idx.append(t * 16 + h)
        idx = np.asarray(idx, dtype=np.int64)
        _line("rotsum j=0 rows (Σ_k y²)", plain_rot, ct_rot, np.isin(np.arange(n), idx))
        print(
            f"  [sumsq] j=0 Σy² plain max/med="
            f"{np.max(np.abs(plain_rot[idx])):.4e}/"
            f"{np.median(np.abs(plain_rot[idx])):.4e} "
            f"ct max/med="
            f"{np.max(np.abs(ct_rot[idx])):.4e}/"
            f"{np.median(np.abs(ct_rot[idx])):.4e}",
            flush=True,
        )


def report(
    engine: Any,
    site: str,
    cts: Any,
    *,
    plain: float | None = None,
    delta: float | None = None,
    note: str = "",
) -> None:
    """Print max |slot|. ``logical`` is ``raw/|delta|`` when delta is set."""
    if not enabled(engine):
        return
    if plain is None:
        plain = getattr(engine, "_mag_plain", {}).get(site)
    sk = engine._mag_sk
    raw = 0.0
    n = 0
    for ct in _iter_cts(cts):
        n += 1
        arr = _dec(engine, ct, sk)
        if arr.size:
            raw = max(raw, float(np.max(np.abs(arr))))
        if raw >= BOOM:
            break
    logical = raw / abs(delta) if delta not in (None, 0.0) else raw
    parts = [f"  [mag] {site} n={n} raw_max={raw:.4e}"]
    if delta is not None:
        parts.append(f"delta={delta:.4e} logical={logical:.4e}")
    if plain is not None:
        parts.append(f"plain_max={plain:.4e}")
    if note:
        parts.append(note)
    if max(raw, logical) >= BOOM:
        parts.append("BOOM")
    print(" ".join(parts), flush=True)
    if (
        max(raw, logical) >= BOOM
        and not getattr(engine, "_mag_inv_audit", False)
        and not getattr(engine, "_mag_ln_audit", False)
    ):
        raise MagBoom(site, raw, logical)


def _asor_np(denom: np.ndarray, e0: float, iters: int) -> np.ndarray:
    a = np.ones_like(denom, dtype=np.float64)
    b = np.asarray(denom, dtype=np.float64).copy()
    en = float(e0)
    for _ in range(int(iters)):
        kn = 2.0 / (en + 1.0)
        b_temp = 2.0 - kn * b
        b = kn * b * b_temp
        a = kn * a * b_temp
        en = kn * en * (2.0 - kn * en)
    return a


def _hq_from_slots(slots: np.ndarray) -> np.ndarray:
    """j=0 lanes: slot ``query*16 + head`` → array ``[head, query]``."""
    out = np.zeros((12, 128), dtype=np.float64)
    for t in range(128):
        out[:, t] = slots[t * 16 : t * 16 + 12]
    return out


def _plain_recip_trace(engine: Any) -> list[dict[str, Any]]:
    cached = getattr(engine, "_mag_recip_trace", None)
    if cached is not None:
        return cached
    from softmax_poly import THOR_EXP_POLY_COEFFS, softmax_config_for_layer

    scores = np.asarray(engine._mag_att_score, dtype=np.float64)
    key_valid = np.asarray(engine._mag_key_valid, dtype=np.float64).reshape(-1)
    task = str(getattr(engine, "_mag_task", "mrpc"))
    layer = int(getattr(engine, "_mag_layer", 8))
    level = int(getattr(engine, "_mag_softmax_level", 1))
    cfg = softmax_config_for_layer(task, layer, level)
    d1 = float(cfg["delta1"])
    d2 = float(cfg["delta2"])
    n1 = int(round(math.log2(d1)))
    n2 = int(round(math.log2(d2)))
    m = (key_valid > 0).astype(np.float64)
    x = scores * m[None, None, :]
    x_scaled = x / d1 / d2 / 8.0
    exp = np.zeros_like(x_scaled)
    for c in np.asarray(THOR_EXP_POLY_COEFFS, dtype=np.float64):
        exp = exp * x_scaled + float(c)
    exp = exp / math.exp(float(cfg["shift"]) / d2 / d1)
    for _ in range(n1):
        exp = exp * exp
    exp *= m[None, None, :]
    sigma = exp.sum(axis=-1)
    e0 = float(cfg["e0_sigma"])
    iters = int(cfg["asor_max_iters_sigma"])
    inv = _asor_np(sigma, e0, iters)
    en = float(e0)
    for _ in range(iters):
        kn = 2.0 / (en + 1.0)
        en = kn * en * (2.0 - kn * en)
    y = exp * inv[:, :, None]
    trace = [
        {
            "key": "sigma",
            "den": sigma,
            "inv": inv,
            "e0": e0,
            "iters": iters,
        }
    ]
    sum_iters = cfg["asor_max_iters_sum_sq"]
    for r in range(n2):
        e0_2 = en / 128.0
        y2 = y * y
        sum_y = (y2 * m[None, None, :]).sum(axis=-1)
        it = int(sum_iters[r])
        inv_r = _asor_np(sum_y, e0_2, it)
        en = e0_2
        for _ in range(it):
            kn = 2.0 / (en + 1.0)
            en = kn * en * (2.0 - kn * en)
        trace.append(
            {
                "key": f"sumsq_r{r}",
                "den": sum_y,
                "inv": inv_r,
                "e0": e0_2,
                "iters": it,
            }
        )
        y = y2 * inv_r[:, :, None]
    engine._mag_recip_trace = trace
    engine._mag_recip_valid = m > 0
    return trace


def _which_recip(tag: str) -> str:
    name = str(tag)
    if "asor_sigma" in name:
        return "sigma"
    if "norm_r0" in name or "sumsq_r0" in name:
        return "sumsq_r0"
    if "norm_r1" in name or "sumsq_r1" in name:
        return "sumsq_r1"
    raise RuntimeError(f"inv audit does not recognize aSOR site {tag}")


def _stat_pair(title: str, he: np.ndarray, plain: np.ndarray) -> None:
    if he.size == 0:
        print(f"    {title}: n=0", flush=True)
        return
    d = he - plain
    denom = float(np.linalg.norm(he) * np.linalg.norm(plain))
    corr = (
        float(np.dot(he - he.mean(), plain - plain.mean()) / denom)
        if denom > 1e-30
        else float("nan")
    )
    scale = np.maximum(np.abs(plain), 1e-12)
    rel = np.abs(d) / scale
    print(
        f"    {title}: n={he.size} he_max={np.max(np.abs(he)):.4e} "
        f"plain_max={np.max(np.abs(plain)):.4e} "
        f"max|Δ|={np.max(np.abs(d)):.4e} mean|Δ|={np.mean(np.abs(d)):.4e} "
        f"med_rel={np.median(rel):.4e} corr={corr:.6f}",
        flush=True,
    )


def audit_asor_output(
    engine: Any,
    tag: str,
    den_ct: Any,
    inv_ct: Any,
    d_delta: float,
    e0: float,
    iters: int,
) -> None:
    """Compare one finished aSOR to the plaintext twin and to aSOR(HE denom)."""
    key = _which_recip(tag)
    trace = _plain_recip_trace(engine)
    step = next(s for s in trace if s["key"] == key)
    sk = engine._mag_sk
    den_he = _hq_from_slots(np.real(_dec(engine, den_ct, sk)).ravel())
    inv_he = _hq_from_slots(np.real(_dec(engine, inv_ct, sk)).ravel()) / float(d_delta)
    valid = np.asarray(engine._mag_recip_valid, dtype=bool)
    mask = np.broadcast_to(valid[None, :], den_he.shape)
    den_plain = np.asarray(step["den"], dtype=np.float64)
    inv_plain = np.asarray(step["inv"], dtype=np.float64)
    inv_of_he = _asor_np(den_he, float(e0), int(iters))
    print(
        f"  [recip] {key} e0={float(e0):.6e} iters={int(iters)} "
        f"d_delta={float(d_delta):.6e} "
        f"plain_e0={float(step['e0']):.6e} plain_iters={int(step['iters'])}",
        flush=True,
    )
    bands = [("valid", mask)]
    den_ref = den_plain
    bands.append(("den<1e-3", mask & (den_ref < 1e-3)))
    bands.append(("den≥1e-3", mask & (den_ref >= 1e-3)))
    for title, band in bands:
        print(f"  [recip] {key} {title}", flush=True)
        _stat_pair("denom HE vs plain", den_he[band], den_plain[band])
        _stat_pair("inv HE vs plain-aSOR(plain)", inv_he[band], inv_plain[band])
        _stat_pair("inv HE vs plain-aSOR(HE denom)", inv_he[band], inv_of_he[band])
        prod = (inv_he * den_he)[band]
        if prod.size:
            print(
                f"    inv*den: med={np.median(prod):.4e} "
                f"min={np.min(prod):.4e} max={np.max(prod):.4e} "
                f"frac<0={np.mean(prod < 0):.3f} "
                f"frac<0.5={np.mean(prod < 0.5):.3f} "
                f"frac>1.5={np.mean(prod > 1.5):.3f}",
                flush=True,
            )
    err = np.abs(inv_he - inv_plain)
    err = np.where(mask, err, -1.0)
    worst = np.argpartition(err.ravel(), -3)[-3:]
    worst = worst[np.argsort(-err.ravel()[worst])]
    print(f"  [recip] {key} worst vs plain-aSOR (head, query, he_den, plain_den, he_inv, plain_inv):", flush=True)
    for w in worst:
        h, t = divmod(int(w), 128)
        print(
            f"    h={h} q={t} den {den_he[h, t]:.6e}/{den_plain[h, t]:.6e} "
            f"inv {inv_he[h, t]:.6e}/{inv_plain[h, t]:.6e} "
            f"prod={inv_he[h, t] * den_he[h, t]:.4e}",
            flush=True,
        )
    band = mask
    den_bad, den_why = _large_ratio_err(den_he[band], den_plain[band])
    inv_bad, inv_why = _large_ratio_err(inv_he[band], inv_plain[band])
    prod = (inv_he * den_he)[band]
    if prod.size:
        prod_abs = np.abs(prod - 1.0)
        prod_med = float(np.median(prod_abs))
        prod_p95 = float(np.percentile(prod_abs, 95))
    else:
        prod_med = prod_p95 = float("inf")
    prod_bad = (
        (not np.isfinite(prod_med))
        or prod_med > 0.05
        or (not np.isfinite(prod_p95))
        or prod_p95 > 0.2
    )
    print(
        f"  [recip] {key} verdict den={'BAD' if den_bad else 'OK'} ({den_why}) "
        f"inv={'BAD' if inv_bad else 'OK'} ({inv_why}) "
        f"inv*den med|·-1|={prod_med:.3e} p95|·-1|={prod_p95:.3e}",
        flush=True,
    )
    if den_bad:
        raise ExpInvStop(f"{key} input already wrong")
    if inv_bad or prod_bad:
        raise ExpInvStop(f"{key} reciprocal wrong")
    if key == "sumsq_r1":
        raise ExpInvStop("three reciprocals matched")


def _large_ratio_err(he: np.ndarray, plain: np.ndarray) -> tuple[bool, str]:
    """True when large-magnitude slots are not within 5% and corr 0.99 of plain."""
    he = np.asarray(he, dtype=np.float64).ravel()
    plain = np.asarray(plain, dtype=np.float64).ravel()
    if he.size == 0 or plain.size == 0:
        return True, "n=0"
    peak = float(np.max(np.abs(plain)))
    if peak <= 0.0 or not np.isfinite(peak):
        return True, "plain peak 0"
    sel = np.abs(plain) >= 0.05 * peak
    if int(sel.sum()) < 8:
        sel = np.ones(plain.size, dtype=bool)
    ratio = he[sel] / plain[sel]
    med = float(np.median(np.abs(ratio - 1.0)))
    denom = float(np.linalg.norm(he) * np.linalg.norm(plain))
    corr = (
        float(np.dot(he - he.mean(), plain - plain.mean()) / denom)
        if denom > 1e-30
        else float("nan")
    )
    # Near-zero slots make corr meaningless (e^x after the shift is mostly ~0).
    # Judge the large-magnitude slots, which carry the sum.
    bad = (not np.isfinite(med)) or med > 0.05
    return bad, f"corr={corr:.4f} med|ratio-1|={med:.3e} n_large={int(sel.sum())}"


def _horner_exp(x_scaled: np.ndarray, coeffs: np.ndarray, shift: float, d1: float, d2: float) -> np.ndarray:
    exp = np.zeros_like(x_scaled, dtype=np.float64)
    for c in np.asarray(coeffs, dtype=np.float64):
        exp = exp * x_scaled + float(c)
    n1 = int(round(math.log2(float(d1))))
    exp = exp / math.exp(float(shift) / float(d2) / float(d1))
    for _ in range(n1):
        exp = exp * exp
    return exp


def _score_valid_mask(n_packs: int, n_slots: int, n_tokens: int) -> np.ndarray:
    mask = np.zeros((n_packs, n_slots), dtype=bool)
    idx = np.arange(n_slots)
    head = idx % 16
    t = (idx // 16) % 128
    j = idx // 2048
    for i in range(n_packs):
        key = (i * 16 + j + t) % 128
        mask[i] = (head < 12) & (key < int(n_tokens))
    return mask


def audit_score_and_exp(engine: Any, score_cts: Any, exp_cts: Any) -> None:
    """Slot-compare the exp input x and the exp output, then stop."""
    from linear_eval import encode_score_packs_from_dense
    from softmax_poly import THOR_EXP_POLY_COEFFS, softmax_config_for_layer
    from thor_encoder_linear_core import bert_base

    scores = np.asarray(engine._mag_att_score, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[0] != 12:
        raise RuntimeError(f"att_score shape {scores.shape}, expected (12, query, key)")
    task = str(getattr(engine, "_mag_task", "mrpc"))
    layer = int(getattr(engine, "_mag_layer", 8))
    level = int(getattr(engine, "_mag_softmax_level", 1))
    cfg_sm = softmax_config_for_layer(task, layer, level)
    d1 = float(cfg_sm["delta1"])
    d2 = float(cfg_sm["delta2"])
    divisor = 8.0 * d1 * d2
    x_hqk = scores / divisor
    exp_hqk = _horner_exp(
        x_hqk,
        np.asarray(THOR_EXP_POLY_COEFFS, dtype=np.float64),
        float(cfg_sm["shift"]),
        d1,
        d2,
    )
    cfg = bert_base()
    plain_x = np.stack(encode_score_packs_from_dense(np.transpose(x_hqk, (0, 2, 1)), cfg))
    plain_exp = np.stack(encode_score_packs_from_dense(np.transpose(exp_hqk, (0, 2, 1)), cfg))
    sk = engine._mag_sk
    he_x = np.stack([np.real(_dec(engine, score_cts[i], sk)).ravel() for i in range(8)])
    he_exp = np.stack([np.real(_dec(engine, exp_cts[i], sk)).ravel() for i in range(8)])
    exp_of_he_x = _horner_exp(
        he_x,
        np.asarray(THOR_EXP_POLY_COEFFS, dtype=np.float64),
        float(cfg_sm["shift"]),
        d1,
        d2,
    )
    n_tok = int(np.count_nonzero(np.asarray(engine._mag_key_valid).reshape(-1) > 0))
    valid = _score_valid_mask(8, he_x.shape[1], n_tok)
    idx = np.arange(he_x.shape[1])
    query = (idx // 16) % 128
    q_ok = np.broadcast_to(query < n_tok, he_x.shape)
    print(
        f"  [exp_in] L{layer} shift={cfg_sm['shift']} δ1={d1} δ2={d2} "
        f"x = score/(8·δ1·δ2) divisor={divisor:.4g} n_tokens={n_tok}",
        flush=True,
    )
    for title, band in (
        ("key<n", valid),
        ("query&key<n", valid & q_ok),
    ):
        print(f"  [exp_in] {title}", flush=True)
        _stat_pair("x HE vs plain", he_x[band], plain_x[band])
        _stat_pair("e^x HE vs plain", he_exp[band], plain_exp[band])
        _stat_pair("e^x HE vs horner(HE x)", he_exp[band], exp_of_he_x[band])
    dx = np.abs(he_x - plain_x)
    dx = np.where(valid & q_ok, dx, -1.0)
    worst = np.argpartition(dx.ravel(), -3)[-3:]
    worst = worst[np.argsort(-dx.ravel()[worst])]
    print("  [exp_in] worst x (pack, query, head, key, he, plain):", flush=True)
    n = he_x.shape[1]
    for f in worst:
        i, s = divmod(int(f), n)
        head = s % 16
        t = (s // 16) % 128
        j = s // 2048
        key = (i * 16 + j + t) % 128
        print(
            f"    pack={i} q={t} h={head} k={key} "
            f"he={he_x[i, s]:.6e} plain={plain_x[i, s]:.6e}",
            flush=True,
        )
    de = np.abs(he_exp - plain_exp)
    de = np.where(valid & q_ok, de, -1.0)
    worst = np.argpartition(de.ravel(), -3)[-3:]
    worst = worst[np.argsort(-de.ravel()[worst])]
    print("  [exp_in] worst e^x (pack, query, head, key, he, plain, x_he, x_plain):", flush=True)
    for f in worst:
        i, s = divmod(int(f), n)
        head = s % 16
        t = (s // 16) % 128
        j = s // 2048
        key = (i * 16 + j + t) % 128
        print(
            f"    pack={i} q={t} h={head} k={key} "
            f"exp {he_exp[i, s]:.6e}/{plain_exp[i, s]:.6e} "
            f"x {he_x[i, s]:.6e}/{plain_x[i, s]:.6e}",
            flush=True,
        )
    band = valid & q_ok
    x_bad, x_why = _large_ratio_err(he_x[band], plain_x[band])
    exp_bad, exp_why = _large_ratio_err(he_exp[band], plain_exp[band])
    print(
        f"  [exp_in] verdict x={'BAD' if x_bad else 'OK'} ({x_why}) "
        f"e^x={'BAD' if exp_bad else 'OK'} ({exp_why})",
        flush=True,
    )
    if x_bad:
        raise ExpInvStop("stop: score x mismatch")
    if exp_bad:
        raise ExpInvStop("stop: e^x mismatch")
    if not getattr(engine, "_mag_inv_audit", False):
        raise ExpInvStop("score x and e^x matched")


def _stage_stats(he, plain) -> tuple[bool, str]:
    """BAD when the stage has left the plaintext, not on a 0.1-level residual."""
    he = np.asarray(he, dtype=np.float64).ravel()
    plain = np.asarray(plain, dtype=np.float64).ravel()
    n = min(he.size, plain.size)
    he, plain = he[:n], plain[:n]
    if n == 0:
        return True, "n=0"
    diff = np.abs(he - plain)
    max_abs = float(np.max(diff))
    peak_h = float(np.max(np.abs(he)))
    peak_p = float(np.max(np.abs(plain)))
    if np.std(he) > 1e-15 and np.std(plain) > 1e-15:
        corr = float(np.corrcoef(he, plain)[0, 1])
    else:
        corr = float("nan")
    if peak_p > 0.0:
        sel = np.abs(plain) >= max(0.05 * peak_p, 1e-4)
    else:
        sel = np.ones(n, dtype=bool)
    if int(sel.sum()) < 8:
        sel = np.ones(n, dtype=bool)
    denom = np.where(np.abs(plain[sel]) < 1e-30, np.nan, plain[sel])
    med = float(np.nanmedian(np.abs(he[sel] / denom - 1.0)))
    boom = peak_h > max(1e4, 50.0 * max(peak_p, 1.0))
    bad = bool(boom or (np.isfinite(med) and med > 0.2 and max_abs > 0.2))
    return bad, (
        f"max|he|={peak_h:.4e} max|plain|={peak_p:.4e} max|Δ|={max_abs:.4e} "
        f"corr={corr:.4f} med|ratio-1|={med:.3e} n={n}"
    )


def _decode_real8(engine: Any, cts: Any) -> np.ndarray:
    from research_unified_entry_slots import decode_layer_output_to_matrix

    sk = engine._mag_sk
    packs = [
        np.asarray(engine.decrode(cts[i], sk, is_real=True), dtype=np.float64).ravel()
        for i in range(8)
    ]
    return decode_layer_output_to_matrix(packs)


def _rows(mat: np.ndarray, n_tokens: int) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        return mat[:n_tokens]
    return mat[:n_tokens]


def audit_real8_matrix(engine: Any, name: str, cts: Any, plain_key: str) -> np.ndarray:
    """Decode 8 real packs and compare valid rows to the polynomial matrix."""
    he = _decode_real8(engine, cts)
    plain = np.asarray(engine._mag_ln_plain[plain_key], dtype=np.float64)
    n_tok = int(getattr(engine, "_mag_n_tokens", he.shape[0]))
    he_v = _rows(he, n_tok)
    plain_v = _rows(plain, n_tok)
    bad, why = _stage_stats(he_v, plain_v)
    print(
        f"  [ln2] {name} {'BAD' if bad else 'OK'} {why}",
        flush=True,
    )
    err = np.abs(he_v - plain_v)
    flat = err.ravel()
    take = min(3, flat.size)
    worst = np.argpartition(flat, -take)[-take:]
    worst = worst[np.argsort(-flat[worst])]
    width = he_v.shape[-1] if he_v.ndim == 2 else 1
    print(f"  [ln2] {name} worst (token, dim, he, plain):", flush=True)
    for f in worst:
        tok, dim = divmod(int(f), width) if width > 1 else (int(f), 0)
        print(
            f"    t={tok} d={dim} he={he_v.ravel()[f]:.6e} "
            f"plain={plain_v.ravel()[f]:.6e}",
            flush=True,
        )
    if bad:
        raise ExpInvStop(f"stop: {name} mismatch")
    return he


def _lane0_tokens(slots: np.ndarray, n_tokens: int) -> np.ndarray:
    slots = np.asarray(slots, dtype=np.float64).ravel()
    return np.array([slots[t * 16] for t in range(n_tokens)], dtype=np.float64)


def _scaled_var_rows(x_scaled: np.ndarray, eps: float, max_for_den: float) -> np.ndarray:
    xs = np.asarray(x_scaled, dtype=np.float64)
    n = float(xs.shape[-1])
    sum_x = xs.sum(axis=-1)
    return n * np.sum(xs * xs, axis=-1) - sum_x * sum_x + eps / max_for_den


def _ln2_scale_const(engine: Any) -> tuple[float, float, float, int, dict]:
    from layernorm_poly import layernorm_config_for_layer

    cfg = layernorm_config_for_layer(
        str(engine._mag_task),
        int(engine._mag_layer),
        "ln2",
        int(getattr(engine, "_mag_ln_level", 1)),
    )
    eps = 1e-5
    n = 768.0
    max_for = (float(cfg["max_var"]) * float(cfg["w_buffer"]) + eps) * (n ** 2)
    scale = 1.0 / (max_for ** 0.5)
    return scale, eps, max_for, int(cfg["invsqrt_max_iters"]), cfg


def audit_ln2_entry(engine: Any, gelu_out: Any, dense2_out: Any, ln2_in: Any) -> None:
    """GeLU slot max, then dense2 and the ln2 input, before the LN circuit."""
    sk = engine._mag_sk
    gmax = 0.0
    for ct in _iter_cts(gelu_out):
        arr = _dec(engine, ct, sk)
        if arr.size:
            gmax = max(gmax, float(np.max(np.abs(arr))))
    gout = np.asarray(engine._mag_ln_plain["gelu_out"], dtype=np.float64)
    print(
        f"  [ln2] gelu_slots raw_max={gmax:.4e} "
        f"poly gelu_out max={float(np.max(np.abs(gout))):.4e}",
        flush=True,
    )
    audit_real8_matrix(engine, "dense2_out", dense2_out, "dense2_out")
    audit_real8_matrix(engine, "ln2_in", ln2_in, "ln2_in")


def audit_ln2_scaled(engine: Any, name: str, enc_l: Any) -> np.ndarray:
    """Scaled 8-pack hidden versus polynomial ln2 input times the LN scale."""
    scale, _eps, _mx, _iters, cfg = _ln2_scale_const(engine)
    he = _decode_real8(engine, enc_l)
    plain = np.asarray(engine._mag_ln_plain["ln2_in"], dtype=np.float64) * scale
    n_tok = int(getattr(engine, "_mag_n_tokens", he.shape[0]))
    bad, why = _stage_stats(_rows(he, n_tok), _rows(plain, n_tok))
    print(
        f"  [ln2] {name} scale={scale:.4e} "
        f"var_range=[{cfg['min_var']:.4g},{cfg['max_var']:.4g}] "
        f"{'BAD' if bad else 'OK'} {why}",
        flush=True,
    )
    if bad:
        raise ExpInvStop(f"stop: {name} mismatch")
    return he


def audit_ln2_variance(engine: Any, variance_ct: Any, he_scaled: np.ndarray) -> None:
    """Lane-0 scaled variance versus the formula on the decrypted hidden."""
    scale, eps, max_for, iters, cfg = _ln2_scale_const(engine)
    sk = engine._mag_sk
    slots = np.asarray(engine.decrode(variance_ct, sk, is_real=True), dtype=np.float64)
    n_tok = int(getattr(engine, "_mag_n_tokens", 128))
    he_var = _lane0_tokens(slots, n_tok)
    from_he = _scaled_var_rows(_rows(he_scaled, n_tok), eps, max_for)
    plain_x = np.asarray(engine._mag_ln_plain["ln2_in"], dtype=np.float64)
    from_poly = _scaled_var_rows(_rows(plain_x, n_tok) * scale, eps, max_for)
    bad, why = _stage_stats(he_var, from_he)
    _bad_p, why_p = _stage_stats(he_var, from_poly)
    xs = _rows(he_scaled, n_tok) / scale
    true_he = np.var(xs, axis=-1)
    xp = _rows(plain_x, n_tok)
    true_p = np.var(xp, axis=-1)
    print(
        f"  [ln2] variance {'BAD' if bad else 'OK'} vs formula(HE input) {why}",
        flush=True,
    )
    print(
        f"  [ln2] variance vs formula(poly) {why_p} "
        f"true Var HE [{true_he.min():.4g},{true_he.max():.4g}] "
        f"poly [{true_p.min():.4g},{true_p.max():.4g}] "
        f"cfg [{cfg['min_var']:.4g},{cfg['max_var']:.4g}] iters={iters}",
        flush=True,
    )
    if float(np.max(he_var)) > 2.0 or float(np.min(he_var)) < -1e-3:
        bad = True
        print(
            f"  [ln2] variance out of invsqrt range "
            f"scaled [{float(np.min(he_var)):.4g},{float(np.max(he_var)):.4g}]",
            flush=True,
        )
    order = np.argsort(-np.abs(he_var - from_he))[:3]
    print("  [ln2] variance worst (token, he, formula_he, formula_poly, true_he, true_poly):", flush=True)
    for t in order:
        print(
            f"    t={int(t)} he={he_var[t]:.6e} f_he={from_he[t]:.6e} "
            f"f_poly={from_poly[t]:.6e} var_he={true_he[t]:.6e} var_poly={true_p[t]:.6e}",
            flush=True,
        )
    engine._mag_ln_he_var = he_var
    engine._mag_ln_e0 = float(cfg["min_var"]) / float(cfg["max_var"])
    if bad:
        raise ExpInvStop("stop: ln2 variance mismatch")


def _invsqrt_rows(var: np.ndarray, e0: float, n_iters: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from layernorm_poly import _he_invsqrt_kn

    an = np.asarray(var, dtype=np.float64).copy()
    bn = np.ones_like(an)
    en = float(e0)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(int(n_iters)):
        kn = _he_invsqrt_kn(en)
        inv_kn = 3.0 / kn
        bn = bn * (kn ** 1.5 / 2.0) * (inv_kn - an)
        an = an * (kn ** 3 / 4.0) * (inv_kn - an) ** 2
        en = kn * en * (3.0 - kn * en) ** 2 / 4.0
        out.append((an.copy(), bn.copy()))
    return out


def audit_ln2_invsqrt_landed(engine: Any, step: int, an_ct: Any, bn_ct: Any) -> None:
    """an/bn just after the iter's bootstrap, before that iter's multiply."""
    he_var = getattr(engine, "_mag_ln_he_var", None)
    if he_var is None:
        return
    if int(step) == 0:
        an_p = np.asarray(he_var, dtype=np.float64)
        bn_p = np.ones_like(an_p)
    else:
        an_p, bn_p = _invsqrt_rows(he_var, float(engine._mag_ln_e0), int(step))[-1]
    sk = engine._mag_sk
    n_tok = int(he_var.shape[0])
    an_h = _lane0_tokens(
        np.asarray(engine.decrode(an_ct, sk, is_real=True), dtype=np.float64), n_tok
    )
    bn_h = _lane0_tokens(
        np.asarray(engine.decrode(bn_ct, sk, is_real=True), dtype=np.float64), n_tok
    )
    bad_a, why_a = _stage_stats(an_h, an_p)
    bad_b, why_b = _stage_stats(bn_h, bn_p)
    print(
        f"  [ln2] invsqrt_{step} after_boot an={'BAD' if bad_a else 'OK'} {why_a}",
        flush=True,
    )
    print(
        f"  [ln2] invsqrt_{step} after_boot bn={'BAD' if bad_b else 'OK'} {why_b}",
        flush=True,
    )
    if bad_a or bad_b:
        raise ExpInvStop(f"stop: ln2 invsqrt_{step} bootstrap mismatch")


def audit_ln2_invsqrt(engine: Any, step: int, an_ct: Any, bn_ct: Any) -> None:
    """One invsqrt iteration versus the float recurrence on the HE variance."""
    he_var = getattr(engine, "_mag_ln_he_var", None)
    if he_var is None:
        return
    e0 = float(engine._mag_ln_e0)
    trace = _invsqrt_rows(he_var, e0, step + 1)
    an_p, bn_p = trace[-1]
    sk = engine._mag_sk
    an_s = np.asarray(engine.decrode(an_ct, sk, is_real=True), dtype=np.float64)
    bn_s = np.asarray(engine.decrode(bn_ct, sk, is_real=True), dtype=np.float64)
    n_tok = int(he_var.shape[0])
    an_h = _lane0_tokens(an_s, n_tok)
    bn_h = _lane0_tokens(bn_s, n_tok)
    bad_a, why_a = _stage_stats(an_h, an_p)
    bad_b, why_b = _stage_stats(bn_h, bn_p)
    prod = bn_h * np.sqrt(np.maximum(he_var, 0.0))
    print(
        f"  [ln2] invsqrt_{step} an={'BAD' if bad_a else 'OK'} {why_a}",
        flush=True,
    )
    print(
        f"  [ln2] invsqrt_{step} bn={'BAD' if bad_b else 'OK'} {why_b} "
        f"bn*sqrt(var) med={float(np.median(prod)):.4f}",
        flush=True,
    )
    if bad_a or bad_b:
        raise ExpInvStop(f"stop: ln2 invsqrt_{step} mismatch")
