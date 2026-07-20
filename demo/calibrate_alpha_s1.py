"""
Phase 1–2：单点 KL 标定 + α = median_k(K/S)。

对每个 (layer, kind) × 档位 k∈{0,1,2}（跳过非法 GeLU 档）：
  方案 = 全 original，仅该位置设为 k
  测 Output KL → 写入 {task}_kl_single.csv

再与 sensitive_scores_1 的 S 对齐：
  α_i = median_k K_{i,k} / (S_{i,k} + ε)
  再按 median(α) 归一化（median→1）

输出（默认 demo/results/alpha_calib/<task>/）：
  {task}_kl_single.csv
  {task}_alpha.csv
  {task}_alpha_summary.txt

用法（在仓库根或任意处）：
  python3 demo/calibrate_alpha_s1.py --task mrpc --samples 64
  python3 demo/calibrate_alpha_s1.py --task mrpc --samples 64 --resume
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import sys as _sys

_DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
if _DEMO_DIR not in _sys.path:
    _sys.path.insert(0, _DEMO_DIR)

from _repo import DEMO_RESULTS_DIR, REPO_ROOT, SENSITIVE_S1_DIR, chdir_repo

chdir_repo()

from cost import NUM_LAYERS, SCHEME_LEN, SCHEME_ORIGINAL, scheme_index  # noqa: E402
from evolution_infer import SchemeEvaluator, format_elapsed  # noqa: E402
from gelu_poly import gelu_level_allowed  # noqa: E402

KINDS = ("softmax", "ln1", "gelu", "ln2")
LEVELS = (0, 1, 2)
LEVEL_NAMES = {0: "low", 1: "mid", 2: "high"}
EPS = 1e-30
# 单点 KL 数值噪声：低于此阈值的档不参与 α 的 median（仍写入 CSV）
KL_RATIO_FLOOR = 1e-8


@dataclass
class SlotRow:
    layer_idx: int
    kind: str
    s_low: float
    s_mid: float
    s_high: float

    def s_at(self, level: int) -> float:
        return (self.s_low, self.s_mid, self.s_high)[level]


def iter_positions() -> list[tuple[int, str]]:
    return [
        (layer_idx, kind)
        for layer_idx in range(NUM_LAYERS)
        for kind in KINDS
    ]


def level_allowed(layer_idx: int, kind: str, level: int) -> bool:
    if kind == "gelu":
        return gelu_level_allowed(layer_idx, level)
    return level in LEVELS


def load_s1_matrix(task_name: str, sensitive_dir: str) -> dict[tuple[int, str], SlotRow]:
    path = os.path.join(sensitive_dir, f"{task_name}_sensitivity.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out: dict[tuple[int, str], SlotRow] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer_idx = int(row["layer_idx"])
            kind = row["kind"]
            out[(layer_idx, kind)] = SlotRow(
                layer_idx=layer_idx,
                kind=kind,
                s_low=float(row["S_low"]),
                s_mid=float(row["S_mid"]),
                s_high=float(row["S_high"]),
            )
    if len(out) != SCHEME_LEN:
        raise ValueError(f"{task_name} S1 行数应为 {SCHEME_LEN}，当前 {len(out)}")
    return out


def original_scheme() -> list[int]:
    return [SCHEME_ORIGINAL] * SCHEME_LEN


def single_site_scheme(layer_idx: int, kind: str, level: int) -> list[int]:
    scheme = original_scheme()
    scheme[scheme_index(layer_idx, kind)] = level
    return scheme


def load_existing_kl(
    path: str,
) -> dict[tuple[int, str, int], float]:
    """resume：已有 (layer, kind, level) → KL。"""
    if not os.path.isfile(path):
        return {}
    out: dict[tuple[int, str, int], float] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("skipped", "").strip().lower() in ("1", "true", "yes"):
                continue
            try:
                layer_idx = int(row["layer_idx"])
                kind = row["kind"]
                level = int(row["level"])
                kl = float(row["output_kl"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(kl):
                out[(layer_idx, kind, level)] = max(0.0, kl)
    return out


def write_kl_single_csv(
    path: str,
    rows: list[dict[str, object]],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "layer_idx",
        "kind",
        "level",
        "level_name",
        "S",
        "output_kl",
        "ratio_kl_over_S",
        "skipped",
        "skip_reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_alphas(
    kl_by_slot: dict[tuple[int, str, int], float],
    s_mat: dict[tuple[int, str], SlotRow],
) -> list[dict[str, object]]:
    """每个位置一行：alpha + 各档 K/S。"""
    rows: list[dict[str, object]] = []
    raw_alphas: list[float] = []
    for layer_idx, kind in iter_positions():
        ratios: list[float] = []
        slot_s = s_mat[(layer_idx, kind)]
        per_level: dict[str, float] = {}
        for level in LEVELS:
            key = f"ratio_{LEVEL_NAMES[level]}"
            if not level_allowed(layer_idx, kind, level):
                per_level[key] = float("nan")
                continue
            kl = kl_by_slot.get((layer_idx, kind, level))
            if kl is None or not math.isfinite(kl):
                per_level[key] = float("nan")
                continue
            kl_c = max(0.0, float(kl))  # 数值噪声可能略负
            s_val = slot_s.s_at(level)
            ratio = kl_c / (s_val + EPS)
            per_level[key] = ratio
            # 近零 KL 的比值不稳定，不进 median
            if math.isfinite(ratio) and ratio >= 0.0 and kl_c >= KL_RATIO_FLOOR:
                ratios.append(ratio)
        if ratios:
            alpha_raw = float(sorted(ratios)[len(ratios) // 2])  # median
        else:
            alpha_raw = float("nan")
        raw_alphas.append(alpha_raw)
        rows.append(
            {
                "pos_idx": scheme_index(layer_idx, kind),
                "layer_idx": layer_idx,
                "kind": kind,
                "alpha_raw": alpha_raw,
                "n_levels_used": len(ratios),
                **per_level,
                "S_low": slot_s.s_low,
                "S_mid": slot_s.s_mid,
                "S_high": slot_s.s_high,
                "K_low": kl_by_slot.get((layer_idx, kind, 0), float("nan")),
                "K_mid": kl_by_slot.get((layer_idx, kind, 1), float("nan")),
                "K_high": kl_by_slot.get((layer_idx, kind, 2), float("nan")),
            }
        )

    finite = [a for a in raw_alphas if math.isfinite(a) and a > 0.0]
    scale = float(sorted(finite)[len(finite) // 2]) if finite else 1.0
    if scale <= 0.0 or not math.isfinite(scale):
        scale = 1.0

    for row, raw in zip(rows, raw_alphas):
        if math.isfinite(raw):
            row["alpha"] = float(raw / scale)
        else:
            row["alpha"] = 1.0  # 缺测时退回不加权
        row["alpha_scale"] = scale
    return rows


def write_alpha_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "pos_idx",
        "layer_idx",
        "kind",
        "alpha",
        "alpha_raw",
        "alpha_scale",
        "n_levels_used",
        "ratio_low",
        "ratio_mid",
        "ratio_high",
        "S_low",
        "S_mid",
        "S_high",
        "K_low",
        "K_mid",
        "K_high",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row[k]
                if isinstance(v, float):
                    out[k] = f"{v:.10e}" if math.isfinite(v) else ""
                else:
                    out[k] = v
            writer.writerow(out)


def write_summary(path: str, task: str, alpha_rows: list[dict[str, object]]) -> None:
    by_kind: dict[str, list[float]] = {k: [] for k in KINDS}
    for row in alpha_rows:
        a = float(row["alpha"])
        if math.isfinite(a):
            by_kind[str(row["kind"])].append(a)

    lines = [
        f"task={task}",
        f"n_positions={len(alpha_rows)}",
        f"alpha_scale={alpha_rows[0]['alpha_scale'] if alpha_rows else float('nan')}",
        "",
        "alpha by kind (median / min / max after norm):",
    ]
    for kind in KINDS:
        vals = by_kind[kind]
        if not vals:
            lines.append(f"  {kind}: (empty)")
            continue
        vals_s = sorted(vals)
        med = vals_s[len(vals_s) // 2]
        lines.append(
            f"  {kind}: median={med:.4g}  min={min(vals):.4g}  max={max(vals):.4g}"
        )
    lines.append("")
    lines.append("top-8 |alpha| positions:")
    ranked = sorted(
        alpha_rows,
        key=lambda r: abs(float(r["alpha"])),
        reverse=True,
    )
    for row in ranked[:8]:
        lines.append(
            f"  L{row['layer_idx']}_{row['kind']}: alpha={float(row['alpha']):.4g}  "
            f"raw={float(row['alpha_raw']):.4g}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_task(
    task_name: str,
    *,
    sensitive_dir: str,
    out_dir: str,
    eval_samples: int | None,
    seed: int,
    resume: bool,
    max_slots: int | None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    s_mat = load_s1_matrix(task_name, sensitive_dir)
    kl_path = os.path.join(out_dir, f"{task_name}_kl_single.csv")
    alpha_path = os.path.join(out_dir, f"{task_name}_alpha.csv")
    summary_path = os.path.join(out_dir, f"{task_name}_alpha_summary.txt")

    cached = load_existing_kl(kl_path) if resume else {}
    if cached:
        print(f"  resume：已有 {len(cached)} 条单点 KL")

    jobs: list[tuple[int, str, int]] = []
    for layer_idx, kind in iter_positions():
        for level in LEVELS:
            if not level_allowed(layer_idx, kind, level):
                continue
            jobs.append((layer_idx, kind, level))
    if max_slots is not None:
        jobs = jobs[: max(0, max_slots)]

    print(
        f"\n=== {task_name.upper()} 单点 KL 标定 ===\n"
        f"  S1 dir: {sensitive_dir}\n"
        f"  out:    {out_dir}\n"
        f"  jobs:   {len(jobs)}（全 original 仅改一处）\n"
        f"  samples:{eval_samples if eval_samples else 'full'}\n"
    )

    evaluator = SchemeEvaluator(
        task_name, eval_samples=eval_samples, seed=seed
    )
    print(f"  eval: {evaluator.eval_desc}")

    csv_rows: list[dict[str, object]] = []
    # 先写入跳过的非法档（完整记录）
    for layer_idx, kind in iter_positions():
        for level in LEVELS:
            if level_allowed(layer_idx, kind, level):
                continue
            slot = s_mat[(layer_idx, kind)]
            csv_rows.append(
                {
                    "layer_idx": layer_idx,
                    "kind": kind,
                    "level": level,
                    "level_name": LEVEL_NAMES[level],
                    "S": f"{slot.s_at(level):.10e}",
                    "output_kl": "",
                    "ratio_kl_over_S": "",
                    "skipped": 1,
                    "skip_reason": "gelu_forbidden",
                }
            )

    t0 = time.time()
    kl_by_slot: dict[tuple[int, str, int], float] = dict(cached)
    done = 0
    for layer_idx, kind, level in jobs:
        slot = s_mat[(layer_idx, kind)]
        s_val = slot.s_at(level)
        key = (layer_idx, kind, level)
        if key in cached:
            kl = cached[key]
        else:
            scheme = single_site_scheme(layer_idx, kind, level)
            kl = float(evaluator.evaluate_for_scheme(scheme).output_kl)
            if math.isfinite(kl):
                kl = max(0.0, kl)
            kl_by_slot[key] = kl
        ratio = (
            max(0.0, kl) / (s_val + EPS)
            if math.isfinite(kl)
            else float("nan")
        )
        csv_rows.append(
            {
                "layer_idx": layer_idx,
                "kind": kind,
                "level": level,
                "level_name": LEVEL_NAMES[level],
                "S": f"{s_val:.10e}",
                "output_kl": f"{kl:.10e}" if math.isfinite(kl) else "",
                "ratio_kl_over_S": f"{ratio:.10e}" if math.isfinite(ratio) else "",
                "skipped": 0,
                "skip_reason": "",
            }
        )
        done += 1
        if done % 8 == 0 or done == len(jobs):
            elapsed = time.time() - t0
            print(
                f"  [{done}/{len(jobs)}] "
                f"L{layer_idx}_{kind}/{LEVEL_NAMES[level]}  "
                f"KL={kl:.4e}  S={s_val:.4e}  K/S={ratio:.4e}  "
                f"elapsed={format_elapsed(elapsed)}"
            )
        # 增量落盘，便于中断 resume
        write_kl_single_csv(kl_path, _sorted_kl_rows(csv_rows))

    write_kl_single_csv(kl_path, _sorted_kl_rows(csv_rows))
    alpha_rows = compute_alphas(kl_by_slot, s_mat)
    write_alpha_csv(alpha_path, alpha_rows)
    write_summary(summary_path, task_name, alpha_rows)
    print(f"  saved: {kl_path}")
    print(f"  saved: {alpha_path}")
    print(f"  saved: {summary_path}")
    print(f"  total elapsed: {format_elapsed(time.time() - t0)}")


def _sorted_kl_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(r: dict[str, object]) -> tuple:
        return (int(r["layer_idx"]), str(r["kind"]), int(r["level"]))

    return sorted(rows, key=sort_key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单点 KL 标定 → α=median(K/S)（demo）"
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=["mrpc"],
        help="任务名，可多个",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="校准子集大小；<=0 表示全验证集",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sensitive-dir",
        default=SENSITIVE_S1_DIR,
        help="一阶敏感度 CSV 目录",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(DEMO_RESULTS_DIR, "alpha_calib"),
        help="输出根目录（其下按 task 分子目录）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已有 kl_single 行",
    )
    parser.add_argument(
        "--max-slots",
        type=int,
        default=None,
        help="仅跑前 N 个合法单点（smoke test）",
    )
    args = parser.parse_args()
    eval_samples = None if args.samples is not None and args.samples <= 0 else args.samples

    print(f"REPO_ROOT={REPO_ROOT}")
    for task in args.task:
        run_task(
            task,
            sensitive_dir=args.sensitive_dir,
            out_dir=os.path.join(args.out_dir, task),
            eval_samples=eval_samples,
            seed=args.seed,
            resume=args.resume,
            max_slots=args.max_slots,
        )


if __name__ == "__main__":
    main()
