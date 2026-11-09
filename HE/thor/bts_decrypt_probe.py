"""Temporary diagnostics: decrypt before/after every ``engine.bootstrap``.

Install with ``install_bts_decrypt_probe(engine, sk)``.  Callers may set
``engine._bts_probe_label`` (or use ``bts_probe_label``) so log lines name the site.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


@contextmanager
def bts_probe_label(engine: Any, name: str) -> Iterator[None]:
    prev = getattr(engine, "_bts_probe_label", None)
    engine._bts_probe_label = name
    try:
        yield
    finally:
        engine._bts_probe_label = prev


def _dec(engine: Any, ct: Any, sk: Any) -> np.ndarray:
    """Prefer complex decrypt (DualRail packed CTs); fall back to real."""
    try:
        return np.asarray(engine.decrode(ct, sk, is_real=False), dtype=np.complex128)
    except Exception:
        return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64).astype(
            np.complex128
        )


def _stats(pre: np.ndarray, post: np.ndarray) -> dict[str, float]:
    err = post - pre
    abs_err = np.abs(err)
    abs_pre = np.abs(pre)
    abs_post = np.abs(post)
    max_pre = float(np.max(abs_pre)) if abs_pre.size else 0.0
    max_post = float(np.max(abs_post)) if abs_post.size else 0.0
    max_err = float(np.max(abs_err)) if abs_err.size else 0.0
    mean_err = float(np.mean(abs_err)) if abs_err.size else 0.0
    rel = max_err / max(max_pre, 1e-30)
    # Real / imag split (useful when DualRail packs look like complex).
    re_err = float(np.max(np.abs(np.real(err)))) if err.size else 0.0
    im_err = float(np.max(np.abs(np.imag(err)))) if err.size else 0.0
    return {
        "max_err": max_err,
        "mean_err": mean_err,
        "rel": rel,
        "max_pre": max_pre,
        "max_post": max_post,
        "re_err": re_err,
        "im_err": im_err,
        "max_re_pre": float(np.max(np.abs(np.real(pre)))) if pre.size else 0.0,
        "max_im_pre": float(np.max(np.abs(np.imag(pre)))) if pre.size else 0.0,
    }


def install_bts_decrypt_probe(engine: Any, sk: Any) -> dict[str, Any]:
    """
    Wrap ``engine.bootstrap``: decrypt → real bootstrap → decrypt → print error.

    Returns info dict with ``real_bootstrap`` for uninstall.
    """
    if getattr(engine, "_bts_decrypt_probe_installed", False):
        return {
            "real_bootstrap": getattr(engine, "_bts_probe_real_bootstrap", engine.bootstrap),
            "already": True,
        }

    real_bs = engine.bootstrap
    counter = [0]
    rows: list[dict[str, Any]] = []

    def _wrapped(ct: Any) -> Any:
        n = counter[0]
        counter[0] = n + 1
        label = getattr(engine, "_bts_probe_label", None) or f"engine.bootstrap#{n}"
        rem_b = None
        lc_b = None
        try:
            lc_b = int(ct.level_calc)
            rem_b = int(engine.num_levels) - lc_b
        except Exception:
            pass

        pre = _dec(engine, ct, sk)
        out = real_bs(ct)

        rem_a = None
        lc_a = None
        try:
            lc_a = int(out.level_calc)
            rem_a = int(engine.num_levels) - lc_a
        except Exception:
            pass
        post = _dec(engine, out, sk)
        st = _stats(pre, post)
        row = {
            "i": n,
            "label": label,
            "rem_before": rem_b,
            "rem_after": rem_a,
            "level_before": lc_b,
            "level_after": lc_a,
            **st,
        }
        rows.append(row)
        print(
            f"  [bts err] #{n} {label}: "
            f"rem {rem_b}→{rem_a} (lc {lc_b}→{lc_a})  "
            f"max|Δ|={st['max_err']:.3e} mean={st['mean_err']:.3e} "
            f"rel={st['rel']:.3e}  "
            f"|pre|max={st['max_pre']:.3e} |post|max={st['max_post']:.3e}  "
            f"ReΔ={st['re_err']:.3e} ImΔ={st['im_err']:.3e} "
            f"(Im|pre|max={st['max_im_pre']:.3e})",
            flush=True,
        )
        return out

    engine.bootstrap = _wrapped  # type: ignore[method-assign]
    engine._bts_decrypt_probe_installed = True
    engine._bts_probe_real_bootstrap = real_bs
    engine._bts_probe_rows = rows
    engine._bts_probe_label = None
    return {"real_bootstrap": real_bs, "rows": rows}


def uninstall_bts_decrypt_probe(engine: Any, info: dict[str, Any] | None = None) -> None:
    real = None
    if info is not None:
        real = info.get("real_bootstrap")
    if real is None:
        real = getattr(engine, "_bts_probe_real_bootstrap", None)
    if real is not None:
        engine.bootstrap = real  # type: ignore[method-assign]
    engine._bts_decrypt_probe_installed = False
