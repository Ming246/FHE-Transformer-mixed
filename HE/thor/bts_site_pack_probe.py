"""Bootstrap before/after decrypt compare (same outward packing per site).

**rel_max / rel_mean**: per-slot ``|Δ|/|pre|`` (skip ``|pre| < min_abs_pre``), then max/mean.

Plan sites + bypass (``align_bypass_to_main``, companion DualRail, force refresh, …).
"""
from __future__ import annotations

import csv
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

# Slots with |pre| below this are excluded from rel_max / rel_mean only.
MIN_ABS_PRE_FOR_REL: float = 1e-3


def _decrypt_ct(engine: Any, sk: Any, ct: Any) -> np.ndarray:
    try:
        return np.asarray(
            engine.decrode(ct, sk, is_real=False), dtype=np.complex128
        )
    except Exception:
        return np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64).astype(
            np.complex128
        )


def compare_real_vectors(
    cur: np.ndarray,
    ref: np.ndarray,
    *,
    min_abs_pre: float | None = None,
) -> dict[str, float]:
    """HE ``cur`` vs poly ``ref``; rel = |Δ|/|ref| with ``|ref| >= min_abs_pre`` mask."""
    cur = np.asarray(cur, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(len(cur), len(ref))
    pre = ref[:n].astype(np.complex128)
    post = cur[:n].astype(np.complex128)
    return _msg_stats(
        pre,
        post,
        min_abs_pre=MIN_ABS_PRE_FOR_REL if min_abs_pre is None else min_abs_pre,
    )


def _msg_stats(
    pre: np.ndarray,
    post: np.ndarray,
    *,
    min_abs_pre: float = MIN_ABS_PRE_FOR_REL,
) -> dict[str, float]:
    err = post - pre
    abs_err = np.abs(err)
    abs_pre = np.abs(pre)
    abs_post = np.abs(post)
    max_pre = float(np.max(abs_pre)) if abs_pre.size else 0.0
    max_post = float(np.max(abs_post)) if abs_post.size else 0.0
    max_abs_err = float(np.max(abs_err)) if abs_err.size else 0.0
    mean_abs_err = float(np.mean(abs_err)) if abs_err.size else 0.0
    rmse = float(np.sqrt(np.mean(np.abs(err) ** 2))) if err.size else 0.0
    mask = abs_pre >= float(min_abs_pre)
    n_rel = int(np.count_nonzero(mask))
    if n_rel > 0:
        rel = abs_err[mask] / abs_pre[mask]
        rel_max = float(np.max(rel))
        rel_mean = float(np.mean(rel))
    else:
        rel_max = 0.0
        rel_mean = 0.0
    return {
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rmse": rmse,
        "rel_max": rel_max,
        "rel_mean": rel_mean,
        "n_rel_slots": float(n_rel),
        "max_abs_pre": max_pre,
        "max_abs_post": max_post,
    }


def _layer_from_event(event_name: str) -> str:
    m = re.match(r"^(L\d+)\.", event_name)
    return m.group(1) if m else ""


def _site_class(event_name: str) -> str:
    if event_name.startswith("bypass.") or event_name.startswith("companion:"):
        return "bypass"
    if event_name.startswith("bypass:"):
        return "bypass"
    if ".companion:" in event_name or event_name.startswith("align_"):
        return "bypass"
    if event_name.startswith("L") and "." in event_name:
        return "plan"
    return "other"


@contextmanager
def suppress_hook_site_probe(engine: Any) -> Iterator[None]:
    """Skip ``BootstrapHook.refresh`` direct probe (real_pack inner cplx bts)."""
    prev = bool(getattr(engine, "_bts_site_probe_suppress_hook", False))
    engine._bts_site_probe_suppress_hook = True  # type: ignore[attr-defined]
    try:
        yield
    finally:
        engine._bts_site_probe_suppress_hook = prev  # type: ignore[attr-defined]


@dataclass
class BtsSiteProbeCollector:
    engine: Any
    sk: Any
    csv_path: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def _append_row(
        self,
        event_name: str,
        pack_mode: str,
        site_layout: str,
        *,
        n_cts: int,
        pack_index: int,
        n_slots: int,
        stats: dict[str, float],
    ) -> None:
        row = {
            "layer": _layer_from_event(event_name),
            "site_class": _site_class(event_name),
            "event_name": event_name,
            "pack_mode": pack_mode,
            "site_layout": site_layout,
            "n_cts": n_cts,
            "pack_index": pack_index,
            "n_slots": n_slots,
            **stats,
        }
        self.rows.append(row)
        print(
            f"  [bts site] {event_name} {pack_mode}/{site_layout} "
            f"ct[{pack_index}] max|Δ|={stats['max_abs_err']:.3e} "
            f"mean|Δ|={stats['mean_abs_err']:.3e} "
            f"rel_max={stats['rel_max']:.3e} rel_mean={stats['rel_mean']:.3e}",
            flush=True,
        )
        if self.csv_path:
            self._write_csv()

    def record_ct_sequence(
        self,
        event_name: str,
        pack_mode: str,
        site_layout: str,
        pre_cts: Sequence[Any],
        post_cts: Sequence[Any],
    ) -> None:
        pre_cts = list(pre_cts)
        post_cts = list(post_cts)
        if len(pre_cts) != len(post_cts):
            raise ValueError(
                f"ct count mismatch at {event_name}: {len(pre_cts)} vs {len(post_cts)}"
            )
        n = len(pre_cts)
        all_pre: list[np.ndarray] = []
        all_post: list[np.ndarray] = []
        for i, (a, b) in enumerate(zip(pre_cts, post_cts)):
            pre = _decrypt_ct(self.engine, self.sk, a)
            post = _decrypt_ct(self.engine, self.sk, b)
            all_pre.append(pre)
            all_post.append(post)
            self._append_row(
                event_name,
                pack_mode,
                site_layout,
                n_cts=n,
                pack_index=i,
                n_slots=int(pre.size),
                stats=_msg_stats(pre, post),
            )
        if n == 0:
            return
        pre_cat = np.concatenate(all_pre)
        post_cat = np.concatenate(all_post)
        self._append_row(
            event_name,
            pack_mode,
            site_layout,
            n_cts=n,
            pack_index=-1,
            n_slots=int(pre_cat.size),
            stats=_msg_stats(pre_cat, post_cat),
        )

    def record_direct_hook_refresh(
        self,
        event_name: str,
        pack_mode: str,
        pre_cts: Sequence[Any],
        post_cts: Sequence[Any],
    ) -> None:
        n = len(pre_cts)
        layout = "direct_1" if n == 1 else f"direct_{n}"
        self.record_ct_sequence(event_name, pack_mode, layout, pre_cts, post_cts)

    def record_real_pair(
        self,
        event_name: str,
        pack_mode: str,
        an_pre: Any,
        bn_pre: Any,
        an_post: Any,
        bn_post: Any,
    ) -> None:
        self.record_ct_sequence(
            event_name,
            pack_mode,
            "real_pair_2",
            [an_pre, bn_pre],
            [an_post, bn_post],
        )

    def record_dualrail_ff(
        self, event_name: str, pack_mode: str, pre: np.ndarray, post: np.ndarray
    ) -> None:
        if pre.shape != post.shape:
            raise ValueError(f"dualrail_ff shape mismatch at {event_name}")
        flat_pre: list[Any] = []
        flat_post: list[Any] = []
        for r in range(pre.shape[0]):
            for c in range(pre.shape[1]):
                flat_pre.append(pre[r, c])
                flat_post.append(post[r, c])
        layout = f"dualrail_ff_{pre.shape[0]}x{pre.shape[1]}"
        self.record_ct_sequence(
            event_name, pack_mode, layout, flat_pre, flat_post
        )

    def _write_csv(self) -> None:
        if not self.csv_path:
            return
        path = os.path.abspath(self.csv_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fieldnames = [
            "layer",
            "site_class",
            "event_name",
            "pack_mode",
            "site_layout",
            "n_cts",
            "pack_index",
            "n_slots",
            "n_rel_slots",
            "max_abs_err",
            "mean_abs_err",
            "rel_max",
            "rel_mean",
            "rmse",
            "max_abs_pre",
            "max_abs_post",
        ]
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.rows:
                w.writerow({k: row.get(k) for k in fieldnames})
        os.replace(tmp, path)


def install_bts_site_probe(
    engine: Any, sk: Any, *, csv_path: str | None = None
) -> BtsSiteProbeCollector:
    coll = BtsSiteProbeCollector(engine=engine, sk=sk, csv_path=csv_path)
    engine._bts_site_probe = coll  # type: ignore[attr-defined]
    return coll


def active_collector(engine: Any) -> BtsSiteProbeCollector | None:
    return getattr(engine, "_bts_site_probe", None)


def probe_collector(
    engine: Any, hook: Any | None, event_name: str
) -> BtsSiteProbeCollector | None:
    coll = active_collector(engine)
    if coll is None or hook is None:
        return None
    if not hook.should_refresh(event_name):
        return None
    return coll


def pack_mode_for_probe(hook: Any, event_name: str) -> str:
    if hasattr(hook, "pack_mode"):
        return str(hook.pack_mode(event_name))
    return "direct"
