"""
全 low / 全 high 方案在各任务验证集（默认）上的对照评估。

指标：Output KL、准确率 / Δacc、翻转率、F1（MRPC）、bts（f_cost）、乘法深度和。
方案由 ``HE.thor.bts_ops.scheme_all(0|1)`` 生成（GeLU 非法档自动回退 high）。

输出（默认 ``picture/evol_low_high.json``），供后续 ``evolution_compare.py`` 作基线点：

```json
{
  "meta": {...},
  "tasks": {
    "mrpc": {
      "baseline_acc": 0.85,
      "all_low":  { "f_cost": ..., "output_kl": ..., ... },
      "all_high": { ... }
    }
  }
}
```

用法：
  python3 picture/evol_low_high.py
  python3 picture/evol_low_high.py --tasks mrpc rte --eval-samples 64
  python3 picture/evol_low_high.py --split calib
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any

_PIC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PIC_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cost import (  # noqa: E402
    compute_f_cost,
    compute_scheme_cost,
    f_cost_label,
)
from evolution_infer import SchemeEvaluator, format_elapsed  # noqa: E402
from HE.thor.bts_ops import scheme_all  # noqa: E402
from poly_model_inference import CALIB_INDICES_DIR, CALIB_SEED  # noqa: E402

# ===================== 配置区 =====================
TASK_NAMES = ("mrpc", "rte", "sst2", "cola", "qnli", "mnli")
DEFAULT_SPLIT = "validation"
DEFAULT_OUT = os.path.join(_PIC_DIR, "evol_low_high.json")
SCHEME_SPECS: tuple[tuple[str, int], ...] = (
    ("all_low", 0),
    ("all_high", 1),
)
# ==================================================


def _finite_or_none(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _pct_points(frac: float | None) -> float | None:
    """准确率差（小数）→ 百分点，与 Pareto CSV 的 accuracy_delta 一致。"""
    v = _finite_or_none(frac)
    return None if v is None else v * 100.0


def evaluate_scheme(
    task: str,
    scheme_key: str,
    level: int,
    evaluator: SchemeEvaluator,
    *,
    cost_mode: str = "bts",
) -> dict[str, Any]:
    scheme = scheme_all(level)
    t0 = time.perf_counter()
    metrics = evaluator.evaluate_for_scheme(scheme)
    infer_s = time.perf_counter() - t0

    f_cost = int(compute_f_cost(task, scheme, cost_mode=cost_mode))
    total_depth = int(compute_scheme_cost(task, scheme))

    acc_delta = _finite_or_none(metrics.accuracy_delta)
    f1_delta = _finite_or_none(metrics.f1_delta)
    return {
        "label": scheme_key.replace("_", " "),
        "level": level,
        "f_cost": f_cost,
        "total_depth": total_depth,
        "cost_mode": cost_mode,
        "output_kl": _finite_or_none(metrics.output_kl),
        "accuracy": _finite_or_none(metrics.accuracy),
        "accuracy_delta": acc_delta,
        "accuracy_delta_pct": _pct_points(acc_delta),
        "flips_pct": _finite_or_none(metrics.flips_pct),
        "loss": _finite_or_none(metrics.loss),
        "loss_delta": _finite_or_none(metrics.loss_delta),
        "f1": _finite_or_none(metrics.f1),
        "f1_delta": f1_delta,
        "f1_delta_pct": _pct_points(f1_delta),
        "oor_count": int(metrics.oor_count),
        "oor_pct": _finite_or_none(metrics.oor_pct),
        "n_eval_used": int(metrics.n_eval_used),
        "infer_seconds": round(infer_s, 3),
        "scheme": list(scheme),
    }


def run_task(
    task: str,
    *,
    eval_split: str,
    eval_samples: int | None,
    seed: int,
    cost_mode: str,
    calib_seed: int,
    calib_indices_dir: str,
) -> dict[str, Any]:
    print(f"\n=== {task.upper()} ({eval_split}) ===")
    evaluator = SchemeEvaluator(
        task,
        eval_samples=eval_samples,
        seed=seed,
        eval_split=eval_split,
        calib_seed=calib_seed,
        calib_indices_dir=calib_indices_dir,
    )
    print(f"  评估集：{evaluator.eval_desc}")
    print(f"  baseline_acc={evaluator.baseline_acc:.6f}")

    out: dict[str, Any] = {
        "baseline_acc": float(evaluator.baseline_acc),
        "baseline_f1": _finite_or_none(evaluator.baseline_f1),
        "eval_desc": evaluator.eval_desc,
        "eval_split": eval_split,
    }
    for scheme_key, level in SCHEME_SPECS:
        print(f"  评估 {scheme_key} (level={level}) …")
        row = evaluate_scheme(
            task, scheme_key, level, evaluator, cost_mode=cost_mode
        )
        out[scheme_key] = row
        print(
            f"    f_cost={row['f_cost']}  depth={row['total_depth']}  "
            f"KL={row['output_kl']:.6g}  "
            f"acc={row['accuracy']:.4f} "
            f"(Δ={row['accuracy_delta_pct']:+.2f}%)  "
            f"flips={row['flips_pct']:.2f}%  "
            f"({format_elapsed(row['infer_seconds'])})"
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="全 low / 全 high 验证集对照（KL / acc / flips / bts）"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASK_NAMES))
    parser.add_argument(
        "--split",
        choices=("validation", "calib"),
        default=DEFAULT_SPLIT,
        help="评估划分（默认 validation）",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="子集大小；默认全量",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cost-mode",
        choices=("bts", "depth"),
        default="bts",
        help=f"f_cost 定义（默认 bts；{f_cost_label('bts')}）",
    )
    parser.add_argument("--calib-seed", type=int, default=CALIB_SEED)
    parser.add_argument("--calib-indices-dir", default=CALIB_INDICES_DIR)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"输出 JSON（默认 {DEFAULT_OUT}）",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="与已有 JSON 按 task 合并（覆盖同名 task）",
    )
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "meta": {
            "eval_split": args.split,
            "eval_samples": args.eval_samples,
            "seed": args.seed,
            "cost_mode": args.cost_mode,
            "cost_label": f_cost_label(args.cost_mode),
            "scheme_keys": [k for k, _ in SCHEME_SPECS],
            "levels": {k: lv for k, lv in SCHEME_SPECS},
        },
        "tasks": {},
    }
    if args.merge and os.path.isfile(args.out):
        with open(args.out, encoding="utf-8") as f:
            old = json.load(f)
        if isinstance(old.get("tasks"), dict):
            payload["tasks"] = dict(old["tasks"])
            print(f"合并已有任务：{sorted(payload['tasks'])}")

    t_all = time.perf_counter()
    for task in args.tasks:
        payload["tasks"][task] = run_task(
            task,
            eval_split=args.split,
            eval_samples=args.eval_samples,
            seed=args.seed,
            cost_mode=args.cost_mode,
            calib_seed=args.calib_seed,
            calib_indices_dir=args.calib_indices_dir,
        )

    payload["meta"]["elapsed_seconds"] = round(time.perf_counter() - t_all, 2)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n已写入：{args.out}")
    print(f"合计用时：{format_elapsed(payload['meta']['elapsed_seconds'])}")


if __name__ == "__main__":
    main()
