"""Layer exit (8-real ``ln2_out``): decrypt → ``encodecrypt`` @ same ``level_calc``."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    from bts_site_pack_probe import (
        MIN_ABS_PRE_FOR_REL,
        _decrypt_ct,
        _msg_stats,
    )
except ImportError:
    from .bts_site_pack_probe import (  # type: ignore
        MIN_ABS_PRE_FOR_REL,
        _decrypt_ct,
        _msg_stats,
    )


def mock_refresh_packs_same_rem(
    engine: Any, sk: Any, pk: Any, packs: Sequence[Any]
) -> tuple[list[Any], list[tuple[int, int]]]:
    """
    Decrypt each real pack and re-``encodecrypt`` at the same ``level_calc``.

    Returns refreshed CT list and per-pack ``(rem_before, rem_after)``.
    """
    packs = list(packs)
    out: list[Any] = []
    rems: list[tuple[int, int]] = []
    nlev = int(engine.num_levels)
    for ct in packs:
        lc = int(ct.level_calc)
        rem_b = nlev - lc
        msg = np.asarray(engine.decrode(ct, sk, is_real=True), dtype=np.float64)
        new_ct = engine.encodecrypt(msg, pk, level=lc)
        rem_a = nlev - int(new_ct.level_calc)
        rems.append((rem_b, rem_a))
        out.append(new_ct)
    return out, rems


@dataclass
class LayerOutMockProbe:
    engine: Any
    sk: Any
    pk: Any
    csv_path: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)

    def apply_and_record(self, packs: Sequence[Any], layer_idx: int) -> list[Any]:
        packs = list(packs)
        if len(packs) != 8:
            raise ValueError(f"layer {layer_idx}: expected 8 real packs, got {len(packs)}")
        event = f"L{layer_idx}.layer_out_decrypt_reencrypt"
        pre_msgs = [_decrypt_ct(self.engine, self.sk, c) for c in packs]
        out, rems = mock_refresh_packs_same_rem(
            self.engine, self.sk, self.pk, packs
        )
        post_msgs = [_decrypt_ct(self.engine, self.sk, c) for c in out]
        for i in range(8):
            st = _msg_stats(pre_msgs[i], post_msgs[i])
            st["n_slots"] = float(pre_msgs[i].size)
            rem_b, rem_a = rems[i]
            self._append_row(
                event,
                pack_index=i,
                rem_before=rem_b,
                rem_after=rem_a,
                stats=st,
            )
        pre_cat = np.concatenate(pre_msgs)
        post_cat = np.concatenate(post_msgs)
        st_all = _msg_stats(pre_cat, post_cat)
        st_all["n_slots"] = float(pre_cat.size)
        self._append_row(
            event,
            pack_index=-1,
            rem_before=rems[0][0],
            rem_after=rem_a,
            stats=st_all,
        )
        for i, (rb, ra) in enumerate(rems):
            if rb != ra:
                print(
                    f"  [layer out mock] L{layer_idx} pack[{i}] rem {rb}→{ra} "
                    f"(level_calc mismatch)",
                    flush=True,
                )
        return out

    def _append_row(
        self,
        event_name: str,
        *,
        pack_index: int,
        rem_before: int,
        rem_after: int,
        stats: dict[str, float],
    ) -> None:
        row = {
            "layer": event_name.split(".")[0],
            "event_name": event_name,
            "pack_index": pack_index,
            "n_cts": 8,
            "rem_before": rem_before,
            "rem_after": rem_after,
            "min_abs_pre_for_rel": MIN_ABS_PRE_FOR_REL,
            "max_abs_err": stats["max_abs_err"],
            "mean_abs_err": stats["mean_abs_err"],
            "rel_max": stats["rel_max"],
            "rel_mean": stats["rel_mean"],
            "rmse": stats["rmse"],
            "max_abs_pre": stats["max_abs_pre"],
            "max_abs_post": stats["max_abs_post"],
            "n_rel_slots": int(stats.get("n_rel_slots", 0)),
            "n_slots": int(stats.get("n_slots", stats.get("n_rel_slots", 0))),
        }
        self.rows.append(row)
        if pack_index == -1:
            print(
                f"  [layer out mock] {event_name} ALL "
                f"max|Δ|={stats['max_abs_err']:.3e} mean|Δ|={stats['mean_abs_err']:.3e} "
                f"rel_max={stats['rel_max']:.3e} rel_mean={stats['rel_mean']:.3e}",
                flush=True,
            )
        if self.csv_path:
            self._flush_csv()

    def _flush_csv(self) -> None:
        path = os.path.abspath(self.csv_path or "")
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "layer",
            "event_name",
            "pack_index",
            "n_cts",
            "n_slots",
            "n_rel_slots",
            "rem_before",
            "rem_after",
            "min_abs_pre_for_rel",
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


def install_layer_out_mock_probe(
    engine: Any, sk: Any, pk: Any, *, csv_path: str | None
) -> LayerOutMockProbe:
    probe = LayerOutMockProbe(engine=engine, sk=sk, pk=pk, csv_path=csv_path)
    engine._layer_out_mock_probe = probe  # type: ignore[attr-defined]
    return probe
