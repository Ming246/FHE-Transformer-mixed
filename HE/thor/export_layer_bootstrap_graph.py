#!/usr/bin/env python3
"""Export one encoder layer's **static** bts_ops event graph (pre-plan).

This dumps what the optimizer *sees* before DP: declared depth, CT counts,
layout, atomic/join/boot-site flags. No placements, no rem replay.

Usage::

  python3 HE/thor/export_layer_bootstrap_graph.py --layer 0 --level 2 \\
    --out-dir results/bts_graphs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cost import gelu_poly_depth  # noqa: E402
from bts_ops import (  # noqa: E402
    DEPTH_LN_PREP,
    DEPTH_LN_PREP_COLD,
    DEPTH_LN_POST_INVSQRT,
    DUALRAIL_UNPACK_DEPTH,
    LN_INVSQRT_ITER_DEPTH,
    SLOT_AUX,
    SLOT_RESID,
    SLOT_V,
    _cost_ct,
    active_geometry,
    build_bootstrap_events,
    scheme_all,
)
from layernorm_poly import count_he_invsqrt_iters, layernorm_config_for_layer  # noqa: E402

_LEVEL_NAMES = ("low", "mid", "high")
_SLOT_NAMES = {
    SLOT_RESID: "RESID",
    SLOT_V: "V",
    SLOT_AUX: "AUX",
}


def _slot_tag(idx: int | None) -> str:
    if idx is None:
        return ""
    return _SLOT_NAMES.get(int(idx), str(idx))


def _flags(ev) -> str:
    bits: list[str] = []
    if ev.atomic:
        bits.append("atomic")
    if ev.save_slot is not None:
        bits.append(f"save={_slot_tag(ev.save_slot)}")
    if ev.join_slot is not None:
        j = f"join={_slot_tag(ev.join_slot)}"
        if ev.join_offset:
            j += f"-{ev.join_offset}"
        if ev.join_ct:
            j += f",side={ev.join_ct}"
        if ev.refresh_side:
            j += ",refresh"
        bits.append(j)
    if ev.clear_slot is not None:
        bits.append(f"clear={_slot_tag(ev.clear_slot)}")
    if ev.dualrail_depth:
        bits.append(f"dr_tax={ev.dualrail_depth}")
    return ",".join(bits)


def export_layer_structure(
    *,
    task: str,
    layer: int,
    level: int,
    out_dir: str,
) -> tuple[str, str]:
    level_name = _LEVEL_NAMES[level]
    scheme = scheme_all(level)
    events_all = build_bootstrap_events(task, scheme, phase="C")
    g = active_geometry()
    layer_events = [
        (i, e) for i, e in enumerate(events_all) if e.layer_idx == layer
    ]

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{task}_L{layer}_{level_name}_event_structure"
    md_path = os.path.join(out_dir, f"{stem}.md")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    ln_lv = scheme[layer * 4 + 1]
    ge_lv = scheme[layer * 4 + 2]
    ln2_lv = scheme[layer * 4 + 3]
    cfg1 = layernorm_config_for_layer(task, layer, "ln1", ln_lv)
    cfg2 = layernorm_config_for_layer(task, layer, "ln2", ln2_lv)
    it1 = count_he_invsqrt_iters(cfg1["invsqrt_max_iters"])
    it2 = count_he_invsqrt_iters(cfg2["invsqrt_max_iters"])
    gd = gelu_poly_depth(layer, ge_lv)

    lines: list[str] = [
        f"# BERT-base L{layer} 静态事件结构图（{task}, scheme=all-{level_name}）",
        "",
        "> 规划**前**视图：`build_bootstrap_events(phase=C)`。"
        " 只有事件声明的 depth / CT / 站点属性；**不含** DP 放置与 rem 回放。",
        "",
        "## 参数",
        "",
        "| 项 | 值 |",
        "|----|----|",
        f"| 任务 | `{task}` |",
        f"| 层 | L{layer} |",
        f"| Softmax / LN / GeLU 档 | all **{level_name}** (level={level}) |",
        f"| DualRail unpack 税 | {DUALRAIL_UNPACK_DEPTH} "
        f"（boot 后 land = budget − tax） |",
        f"| 全模型事件数 | {len(events_all)} |",
        f"| 本层事件数 | {len(layer_events)} |",
        "",
        "## DualRail CT 几何（条数）",
        "",
        "| 符号 | 含义 | 条数 |",
        "|------|------|------|",
        f"| Hc | 复数 spine packs | {g.ct_hidden_cplx} |",
        f"| Hr | 实数残差 / LN packs | {g.ct_hidden_real} |",
        f"| Q | QKV 输出 | {g.ct_qkv_out} |",
        f"| Qc | Q copies | {g.ct_q_copies} |",
        f"| S | attention score | {g.ct_att_score} |",
        f"| A | Softmax α copies | {g.ct_alpha_copies} |",
        f"| V / C | V / context | {g.ct_v_cplx} / {g.ct_context_cplx} |",
        f"| R / Rctx | rotated copies | {g.ct_pc_rot} / {g.ct_pc_rot_ctx} |",
        f"| F | FFN live | {g.ct_ffn} |",
        "",
        "## 读表说明",
        "",
        "- **depth**：事件图标定的乘法深度（规划输入）。",
        "- **Σdepth**：本层从头到该步（含）的 depth 累计。",
        "- **ct_in → ct_out**：该步活密文条数变化。",
        "- **ct_peak**：段内峰值（若有；中途 plain bts 上界）。",
        "- **cost_ct**：若在此站点 bootstrap，DP 主路径计价条数"
        "（`layout=dualrail` 用全量；`real` 用 `n//2`；可被 `cost_ct` 覆盖）。",
        "- **flags**：`atomic`=段内不可 mid；"
        "`save/join/clear`=旁路槽；`dr_tax`=DualRail unpack 税。"
        " bts 只插在相邻微事件之间（无独立 `fixed_bts_*` 占位）。",
        "",
        "## 本层非线性深度（由档位推导）",
        "",
        "| 算子 | 公式 | 深度 |",
        "|------|------|------|",
        f"| LN1 | prep({DEPTH_LN_PREP}) + {it1}×inv√({LN_INVSQRT_ITER_DEPTH}) "
        f"+ inv(0) + post({DEPTH_LN_POST_INVSQRT}) | "
        f"**{DEPTH_LN_PREP + it1 * LN_INVSQRT_ITER_DEPTH + DEPTH_LN_POST_INVSQRT}** |",
        f"| GeLU | Chebyshev `depth_he` | **{gd}** |",
        f"| LN2 | prep({DEPTH_LN_PREP}) + {it2}×inv√({LN_INVSQRT_ITER_DEPTH}) "
        f"+ inv(0) + post({DEPTH_LN_POST_INVSQRT}) | "
        f"**{DEPTH_LN_PREP + it2 * LN_INVSQRT_ITER_DEPTH + DEPTH_LN_POST_INVSQRT}** |",
        "",
        f"> 冷启动 LN prep 按 **{DEPTH_LN_PREP_COLD}**；DualRail 入口后按 "
        f"**{DEPTH_LN_PREP}**（回放时才区分；本表 depth 列仍写声明值 "
        f"{DEPTH_LN_PREP}）。",
        "",
        f"## L{layer} 事件序列（静态）",
        "",
        "| # | 事件 | kind | depth | Σdepth | ct_in→out | ct_peak | "
        "cost_ct | layout | flags |",
        "|--:|------|------|------:|-------:|----------:|--------:|-------:|"
        "--------|-------|",
    ]

    csv_rows: list[dict] = []
    cum = 0
    for local_i, (gi, ev) in enumerate(layer_events):
        cum += int(ev.depth)
        suf = ev.name.split(".", 1)[-1]
        peak = ev.ct_peak if ev.ct_peak is not None else ""
        cost = _cost_ct(ev)
        fl = _flags(ev)
        lines.append(
            f"| {local_i} | `{suf}` | {ev.kind} | {ev.depth} | {cum} | "
            f"{ev.ct_in}→{ev.ct_out} | {peak if peak != '' else '—'} | "
            f"{cost} | {ev.layout} | {fl if fl else '—'} |"
        )
        csv_rows.append(
            {
                "local_index": local_i,
                "global_index": gi,
                "event": ev.name,
                "suffix": suf,
                "kind": ev.kind,
                "depth": ev.depth,
                "cum_depth": cum,
                "ct_in": ev.ct_in,
                "ct_out": ev.ct_out,
                "ct_peak": peak,
                "cost_ct": cost,
                "layout": ev.layout,
                "atomic": int(ev.atomic),
                "dualrail_depth": ev.dualrail_depth,
                "save_slot": _slot_tag(ev.save_slot),
                "join_slot": _slot_tag(ev.join_slot),
                "join_offset": ev.join_offset,
                "join_ct": ev.join_ct,
                "refresh_side": int(ev.refresh_side),
                "clear_slot": _slot_tag(ev.clear_slot),
                "flags": fl,
            }
        )

    # mermaid: one node per event (no edges; label uses spaces not \\n)
    mm: list[str] = [
        "```mermaid",
        "flowchart TD",
    ]
    for local_i, (_gi, ev) in enumerate(layer_events):
        suf = ev.name.split(".", 1)[-1]
        label = f"{suf}  d={ev.depth}  ct {ev.ct_in}→{ev.ct_out}"
        mm.append(f'  E{local_i}["{label}"]')
    mm.append("```")

    lines += [
        "",
        f"- 本层声明 depth 之和: **{cum}**",
        "",
        "## 逐步结构流（Mermaid）",
        "",
        *mm,
        "",
        "---",
        "",
        "*Generated by `bts_ops.build_bootstrap_events` phase=C（无 optimize）。*",
        "",
    ]

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    return md_path, csv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="mrpc")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--level", type=int, default=2, choices=(0, 1, 2))
    ap.add_argument("--out-dir", default="results/bts_graphs")
    args = ap.parse_args()
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO, out_dir)
    md, csv_p = export_layer_structure(
        task=args.task,
        layer=int(args.layer),
        level=int(args.level),
        out_dir=out_dir,
    )
    print(f"wrote {md}")
    print(f"wrote {csv_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
