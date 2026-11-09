"""Capture / compare raw decrypted slot vectors (mock ref vs real bts).

Each ciphertext is ``decrode``'d and concatenated. No matrix scatter, no F,
no bake. ``att_context`` is split into real then imag per ciphertext, and
those real vectors are compared elementwise.

Relative error: per-slot |Δ|/|ref| with |ref| ≥ ``CHAIN_MIN_ABS_PRE``; max and
mean reported as percentages (×100).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from bts_site_pack_probe import _msg_stats
except ImportError:
    from .bts_site_pack_probe import _msg_stats  # type: ignore

# |ref| below this is excluded from rel_max / rel_mean only.
CHAIN_MIN_ABS_PRE: float = 0.01


def _is_ct_handle(x: Any) -> bool:
    """Liberate ``DataStruct`` / ciphertext (NamedTuple with ``level_calc``)."""
    return hasattr(x, "level_calc") and hasattr(x, "origin")


def _unwrap_ct(x: Any) -> Any:
    """``_DeltaCt`` wrapper or ``(ct, …)`` accidental tuple."""
    if x is None:
        return x
    if _is_ct_handle(x):
        return x
    if hasattr(x, "ciphertext") and hasattr(x, "delta"):
        return x.ciphertext
    if isinstance(x, tuple) and not _is_ct_handle(x):
        for part in x:
            if _is_ct_handle(part):
                return part
        if len(x) == 1:
            return _unwrap_ct(x[0])
    return x


def _as_ct_list(cts: Any) -> list:
    """Flatten pack lists / ``(N,)`` object arrays — never split one CT."""
    if cts is None:
        return []
    cts = _unwrap_ct(cts)
    if _is_ct_handle(cts):
        return [cts]
    if isinstance(cts, np.ndarray):
        if cts.dtype == object:
            if cts.ndim == 0:
                return _as_ct_list(cts.item())
            return [_unwrap_ct(cts.reshape(-1)[i]) for i in range(cts.size)]
        if cts.ndim == 0:
            return _as_ct_list(cts.item())
        return [_unwrap_ct(cts[i]) for i in range(cts.shape[0])]
    if isinstance(cts, (list, tuple)):
        out: list = []
        for x in cts:
            x = _unwrap_ct(x)
            if _is_ct_handle(x):
                out.append(x)
            elif isinstance(x, (list, tuple)):
                out.extend(_as_ct_list(x))
            elif isinstance(x, np.ndarray) and x.dtype == object and x.size > 1:
                out.extend(_as_ct_list(x))
            else:
                raise TypeError(
                    f"_as_ct_list: expected CT or CT pack, got {type(x)!r}"
                )
        return out
    return [_unwrap_ct(cts)]


def _decrypt_real_flat(engine: Any, sk: Any, ct: Any) -> np.ndarray:
    ct = _unwrap_ct(ct)
    return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64).ravel()


def _decrypt_cplx_flat(engine: Any, sk: Any, ct: Any) -> np.ndarray:
    ct = _unwrap_ct(ct)
    return np.asarray(
        engine.decrode(ct, sk, is_real=False), dtype=np.complex128
    ).ravel()


def decrypt_site_vector(
    engine: Any,
    sk: Any,
    cts: Any,
    site: str,
    *,
    task: str,
    layer_idx: int,
) -> np.ndarray:
    """Raw ``decrode`` slots, one vector per ciphertext, concatenated in order.

    ``task`` / ``layer_idx`` are unused: no per-layer bake or layout restore.
    ``att_context``: each complex ciphertext becomes ``[real | imag]``
    (32768 + 32768). Other sites stay one real vector per ciphertext.
    """
    del task, layer_idx
    seq = _as_ct_list(cts)
    if not seq:
        raise ValueError(f"empty CT sequence for site {site!r}")
    chunks: list[np.ndarray] = []
    for ct in seq:
        ct = _unwrap_ct(ct)
        if not _is_ct_handle(ct):
            raise TypeError(
                f"{site}: slot is not a CT ({type(ct)!r}), len(seq)={len(seq)}"
            )
        if site == "att_context":
            z = _decrypt_cplx_flat(engine, sk, ct)
            chunks.append(np.real(z))
            chunks.append(np.imag(z))
        else:
            chunks.append(_decrypt_real_flat(engine, sk, ct))
    return np.concatenate(chunks)


def _stats_pct(ref: np.ndarray, cur: np.ndarray) -> dict[str, float]:
    ref = np.asarray(ref).ravel()
    cur = np.asarray(cur).ravel()
    n = min(ref.size, cur.size)
    st = _msg_stats(ref[:n], cur[:n], min_abs_pre=CHAIN_MIN_ABS_PRE)
    return {
        "max_abs_err": st["max_abs_err"],
        "mean_abs_err": st["mean_abs_err"],
        "rel_max_pct": 100.0 * st["rel_max"],
        "rel_mean_pct": 100.0 * st["rel_mean"],
        "n_rel_slots": st["n_rel_slots"],
        "max_abs_ref": st["max_abs_pre"],
        "max_abs_cur": st["max_abs_post"],
        "n_elem": float(min(len(ref.ravel()), len(cur.ravel()))),
    }


@dataclass
class ChainBtsMockProbe:
    """``capture`` writes mock ref vectors; ``compare`` diffs real bts vs ref."""

    engine: Any
    sk: Any
    layer_idx: int
    task: str
    mode: str  # capture | compare
    snapshot_dir: str
    compare_csv: str | None = None
    run_bootstrap: str = ""
    sample: int = 0
    _vecs: dict[str, np.ndarray] = field(default_factory=dict)

    def _snapshot_path(self) -> str:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        return os.path.join(self.snapshot_dir, f"mock_L{int(self.layer_idx)}.npz")

    def load_ref(self) -> None:
        path = self._snapshot_path()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing mock snapshot: {path}")
        data = np.load(path, allow_pickle=True)
        self._vecs = {str(k): np.asarray(data[k]) for k in data.files}

    def save_capture(self) -> None:
        path = self._snapshot_path()
        np.savez_compressed(path, **self._vecs)
        print(f"  [chain probe] saved mock ref → {path} ({len(self._vecs)} sites)", flush=True)

    def touch(self, site: str, cts: Any) -> None:
        vec = decrypt_site_vector(
            self.engine,
            self.sk,
            cts,
            site,
            task=self.task,
            layer_idx=int(self.layer_idx),
        )
        key = str(site)
        if self.mode == "capture":
            self._vecs[key] = vec
            return
        ref = self._vecs.get(key)
        if ref is None:
            print(f"  [chain probe] WARN no mock ref for {key}", flush=True)
            return
        st = _stats_pct(ref, vec)
        print(
            f"  [chain probe] L{self.layer_idx}.{key}: "
            f"max|Δ|={st['max_abs_err']:.4e} mean|Δ|={st['mean_abs_err']:.4e} "
            f"rel_max={st['rel_max_pct']:.4f}% rel_mean={st['rel_mean_pct']:.4f}% "
            f"(n_rel={int(st['n_rel_slots'])})",
            flush=True,
        )
        if self.compare_csv:
            row = {
                "run_bootstrap": self.run_bootstrap,
                "dataset": self.task,
                "sample": int(self.sample),
                "layer": int(self.layer_idx),
                "site": key,
                **st,
                "min_abs_pre": CHAIN_MIN_ABS_PRE,
            }
            path = os.path.abspath(self.compare_csv)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            write_header = not os.path.isfile(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)


def active_chain_probe(engine: Any) -> ChainBtsMockProbe | None:
    return getattr(engine, "_chain_bts_mock_probe", None)


def install_chain_probe(engine: Any, probe: ChainBtsMockProbe | None) -> None:
    engine._chain_bts_mock_probe = probe  # type: ignore[attr-defined]
