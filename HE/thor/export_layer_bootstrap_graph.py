#!/usr/bin/env python3
"""Export one encoder layer's **static** bts_ops event graph (pre-plan).

This dumps what the optimizer *sees* before DP: declared depth, CT counts,
layout, atomic/join/boot-site flags. No placements, no rem replay.

Usage::

  python3 HE/thor/export_layer_bootstrap_graph.py --layer 0 --level 1 \\
    --out-dir results/bts_graphs
  python3 HE/thor/export_layer_bootstrap_graph.py --layer 0 --level 1 --plan
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

from cost import gelu_he_depth  # noqa: E402
from bts_ops import (  # noqa: E402
    BOOTSTRAP_DEPTH_BUDGET,
    DEPTH_LN_PREP,
    DEPTH_LN_PREP_COLD,
    DEPTH_LN_POST_INVSQRT,
    DEFAULT_L0_ENTRY_REMAINING,
    DUALRAIL_UNPACK_DEPTH,
    LN_INVSQRT_ITER_DEPTH,
    SLOT_AUX,
    SLOT_RESID,
    SLOT_V,
    _cost_ct,
    active_geometry,
    build_bootstrap_events,
    optimize_bootstrap,
    scheme_all,
)
from layernorm_poly import count_he_invsqrt_iters, layernorm_config_for_layer  # noqa: E402

_LEVEL_NAMES = ("low", "high")
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
    gd = gelu_he_depth(layer, ge_lv)

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


def _pack_mode_label(pack_mode: str) -> str:
    if pack_mode == "real_pack":
        return "dualrail"
    if pack_mode == "direct":
        return "plain"
    return pack_mode


def _format_event_rem(rows: list) -> str:
    """Build rem cell from ``rem_trace`` rows for one event (chronological)."""
    if not rows:
        return "—"
    enter: int | None = None
    for row in rows:
        if row.kind == "before_boot" and row.boot_reason == "before_event":
            enter = int(row.rem_before)
            break
    if enter is None:
        for row in rows:
            if row.kind == "after_event":
                enter = int(row.rem_before)
                break
    if enter is None:
        enter = int(rows[0].rem_before)
    parts = [str(enter)]
    for row in rows:
        if row.kind == "after_boot":
            parts.append(str(int(row.rem_after)))
        elif row.kind == "after_event":
            parts.append(str(int(row.rem_after)))
    out: list[str] = []
    for p in parts:
        if not out or out[-1] != p:
            out.append(p)
    if len(out) == 1:
        return f"{out[0]}→{out[0]}"
    return "→".join(out)


def _bts_placement_cell(placements: list) -> str:
    if not placements:
        return "—"
    bits: list[str] = []
    for p in placements:
        when = "before" if p.reason == "before_event" else "mid"
        style = _pack_mode_label(str(p.pack_mode or "direct"))
        bits.append(f"**{when}** ×{int(p.ct_count)} ({style})")
    return " ".join(bits)


def export_layer_bootstrap_plan(
    *,
    task: str,
    layer: int,
    level: int,
    out_dir: str,
    initial_level: int | None = None,
) -> tuple[str, str]:
    """DP plan + ``rem_trace`` replay for one encoder layer (phase=C)."""
    level_name = _LEVEL_NAMES[level]
    scheme = scheme_all(level)
    init = (
        DEFAULT_L0_ENTRY_REMAINING
        if initial_level is None
        else int(initial_level)
    )
    plan = optimize_bootstrap(
        task,
        scheme,
        budget=BOOTSTRAP_DEPTH_BUDGET,
        initial_level=init,
        phase="C",
    )
    prefix = f"L{layer}."
    layer_events = [
        (i, e) for i, e in enumerate(plan.events) if e.layer_idx == layer
    ]
    trace_by_ei: dict[int, list] = {}
    for row in plan.rem_trace:
        if row.event_index is None:
            continue
        ev = plan.events[row.event_index]
        if ev.layer_idx != layer:
            continue
        trace_by_ei.setdefault(int(row.event_index), []).append(row)

    pl_by_ei: dict[int, list] = {}
    layer_placements: list = []
    for p in plan.placements:
        if not p.event_name.startswith(prefix):
            continue
        layer_placements.append(p)
        pl_by_ei.setdefault(int(p.event_index), []).append(p)

    g = active_geometry()
    ln_lv = scheme[layer * 4 + 1]
    ge_lv = scheme[layer * 4 + 2]
    ln2_lv = scheme[layer * 4 + 3]
    cfg1 = layernorm_config_for_layer(task, layer, "ln1", ln_lv)
    cfg2 = layernorm_config_for_layer(task, layer, "ln2", ln2_lv)
    it1 = count_he_invsqrt_iters(cfg1["invsqrt_max_iters"])
    it2 = count_he_invsqrt_iters(cfg2["invsqrt_max_iters"])
    gd = gelu_he_depth(layer, ge_lv)

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{task}_L{layer}_{level_name}_bootstrap_graph"
    md_path = os.path.join(out_dir, f"{stem}.md")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    lines: list[str] = [
        f"# BERT-base L{layer} Bootstrap 计算图（{task}, scheme=all-{level_name}）",
        "",
        "## 参数",
        "",
        "| 项 | 值 |",
        "|----|----|",
        f"| 任务 | `{task}` |",
        f"| 层 | L{layer} |",
        f"| Softmax / LN / GeLU 档 | all **{level_name}** (level={level}) |",
        f"| Bootstrap 预算 `budget` | {BOOTSTRAP_DEPTH_BUDGET} |",
        f"| 入口 remaining | {init} |",
        f"| DualRail unpack 税 | {DUALRAIL_UNPACK_DEPTH} "
        f"（boot land = budget − tax） |",
        f"| 全模型 plan | {plan.summary()} |",
        f"| 本层事件数 | {len(layer_events)} |",
        f"| 本层放置数 | {len(layer_placements)} |",
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
        "- **depth**：事件图标定的乘法深度。",
        "- **Δrem**：DP 回放实际扣减（`after_event.depth_charged`）。",
        "- **ct_in → ct_out**：该步活密文条数。",
        "- **cost_ct**：若在此站点 bootstrap，DP 主路径计价条数。",
        "- **BTS**：优化器放置；`before`=段首，`mid`=段内；"
        "`real_pack` 表内写作 dualrail，`direct` 写作 plain。",
        "- **rem**：`进入→[boot后…]→事件后`（land rem = budget − unpack 税）。",
        "",
        "## 本层非线性深度（由参数推导）",
        "",
        "| 算子 | 公式 | 深度 |",
        "|------|------|------|",
        f"| LN1 | scale+var+enc + {it1}×inv√({LN_INVSQRT_ITER_DEPTH}) "
        f"+ post({DEPTH_LN_POST_INVSQRT}) | "
        f"**{1 + 2 + it1 * LN_INVSQRT_ITER_DEPTH + DEPTH_LN_POST_INVSQRT}** |",
        f"| GeLU | Chebyshev `depth_he`（段内拆为 f1+f2+reconstruct） | **{gd}** |",
        f"| LN2 | scale+var+enc + {it2}×inv√({LN_INVSQRT_ITER_DEPTH}) "
        f"+ post({DEPTH_LN_POST_INVSQRT}) | "
        f"**{1 + 2 + it2 * LN_INVSQRT_ITER_DEPTH + DEPTH_LN_POST_INVSQRT}** |",
        "",
        f"> 冷启动 LN prep 按 **{DEPTH_LN_PREP_COLD}**；DualRail 入口后 scale 段按 "
        f"**{DEPTH_LN_PREP}**（微事件 `ln*_scale` 标记 unpack 税，回放 land rem 见上）。",
        "",
        f"## L{layer} 事件序列（完整 + plan 回放）",
        "",
        "| # | 事件 | kind | depth | Δrem | rem | ct_in→out | cost_ct | layout | BTS |",
        "|--:|------|------|------:|-----:|-----|----------:|--------:|--------|-----|",
    ]

    csv_rows: list[dict] = []
    depth_sum = 0
    delta_sum = 0
    for local_i, (gi, ev) in enumerate(layer_events):
        rows = trace_by_ei.get(gi, [])
        delta = 0
        for row in rows:
            if row.kind == "after_event":
                delta = int(row.depth_charged)
        depth_sum += int(ev.depth)
        delta_sum += delta
        suf = ev.name.split(".", 1)[-1]
        rem_s = _format_event_rem(rows)
        bts_s = _bts_placement_cell(pl_by_ei.get(gi, []))
        cost = _cost_ct(ev)
        lines.append(
            f"| {local_i} | `{suf}` | {ev.kind} | {ev.depth} | {delta} | "
            f"{rem_s} | {ev.ct_in}→{ev.ct_out} | {cost} | {ev.layout} | {bts_s} |"
        )
        csv_rows.append(
            {
                "local_index": local_i,
                "global_index": gi,
                "event": ev.name,
                "suffix": suf,
                "kind": ev.kind,
                "depth": ev.depth,
                "delta_rem": delta,
                "rem_replay": rem_s,
                "ct_in": ev.ct_in,
                "ct_out": ev.ct_out,
                "cost_ct": cost,
                "layout": ev.layout,
                "bts": bts_s,
            }
        )

    bts_ct_sum = sum(int(p.ct_count) for p in layer_placements)
    lines += [
        "",
        f"- 事件图 depth 之和（本层）: **{depth_sum}**",
        f"- DP 实际 Δrem 之和（本层）: **{delta_sum}**",
        "",
        "## 本层 Bootstrap 放置一览",
        "",
        "| 事件 | 时机 | style | 密文条数 | rem_before |",
        "|------|------|-------|--------:|-----------:|",
    ]
    for p in sorted(layer_placements, key=lambda x: (x.event_index, x.reason)):
        when = "段首 before" if p.reason == "before_event" else "段内 mid"
        style = _pack_mode_label(str(p.pack_mode or "direct"))
        lines.append(
            f"| `{p.event_name}` | {when} | {style} | **{int(p.ct_count)}** | "
            f"{int(p.remaining_before)} |"
        )
    lines += [
        "",
        f"本层 bts 密文合计（Σ ct_count）: **{bts_ct_sum}**",
        "",
        "## 结构示意（粗粒度）",
        "",
        "```mermaid",
        "flowchart TD",
        "  IN[L0 entry @ rem=14] --> QKV[linear QKV + att_score]",
        "  QKV --> SM[Softmax: exp / aSOR / Σy² / dualrail exit]",
        "  SM --> CTX[att_context + attn_dense]",
        "  CTX --> LN1[LN1: scale var invsqrt x5 post]",
        "  LN1 --> FF[FFN: dense1 gelu_f1/f2 reconstruct dense2]",
        "  FF --> LN2[LN2: scale var invsqrt x5 post]",
        f"  LN2 --> OUT[ln2_out → L{layer + 1}]",
        "```",
        "",
        "---",
        "",
        "*Generated by `export_layer_bootstrap_graph.py --plan` "
        "(`optimize_bootstrap` phase=C + `rem_trace` replay).*",
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
    ap.add_argument("--level", type=int, default=1, choices=(0, 1))
    ap.add_argument("--out-dir", default="results/bts_graphs")
    ap.add_argument(
        "--plan",
        action="store_true",
        help="导出 DP 放置 + rem 回放（bootstrap_graph.md）",
    )
    ap.add_argument(
        "--initial-rem",
        type=int,
        default=None,
        help=f"L0 入口 remaining（默认 {DEFAULT_L0_ENTRY_REMAINING}）",
    )
    args = ap.parse_args()
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO, out_dir)
    if args.plan:
        md, csv_p = export_layer_bootstrap_plan(
            task=args.task,
            layer=int(args.layer),
            level=int(args.level),
            out_dir=out_dir,
            initial_level=args.initial_rem,
        )
    else:
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
