"""
单层直接 GELU：在模型里扫 / 指定最高多项式次数。

ĝ(x)=Chebyshev(x)≈GELU(x)，定义域 LAYER_FIT_X（可非对称，仿射到 [-1,1]）。
  LAYER_FIT_X[i]        该层拟合 x 区间 [lo, hi]
  LAYER_DEGREES[i]      该层最高次数（None 则扫 DEGREES）

只看「仅该层 GeLU 换成该配置、其余 original」的验证集 acc / F1 / flips / KL。

    python3 gelu_single.py
    python3 gelu_single.py --layers 2 3 4 --degrees 31 47 63
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from functools import lru_cache

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from numpy.polynomial.chebyshev import chebvander

from gelu_chebyshev import (  # noqa: E402
    DEFAULT_TRUNCATION_THRESHOLD,
    cheb_eval_on_domain,
    gelu_exact,
    gelu_he_depth_breakdown,
    remez_minimax_cheb,
    truncate_cheb_coeffs,
    x_to_cheb_z,
)
from gelu_poly import (  # noqa: E402
    _coeffs_to_f64,
    _direct_gelu_cheb,
)

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
TASK = "mrpc"
EVAL_SPLIT = "validation"  # "validation" | "calib"
NUM_LAYERS = 12
LAYERS = tuple(range(NUM_LAYERS))
# 每层拟合 x 区间 [lo, hi]（Chebyshev 仿射到 [-1,1]；不再另设 C）
LAYER_FIT_X: list[list[float]] = [
    [-30, 25], [-40, 20], [-35, 75], [-80, 20],
    [-80, 20], [-60, 20], [-40, 20], [-30, 20],
    [-35, 40], [-55, 110], [-50, 160], [-40, 30],
]
# 每层最高次数 L0..L11。非 None 时只测该表，不再扫 DEGREES。
LAYER_DEGREES: list[int] | None = [
    31, 47, 63, 63, 47, 47,
    47, 31, 47, 63, 63, 31,
]
# LAYER_DEGREES = None 时的网格
DEGREES = (15, 23, 31, 39, 47, 55, 63)
# None=只替换对应一层；k≥1=任选 k 层同时近似（其余 original）
COMBO_K: int | None = None
SAVE_COEFFS = True
TRUNCATION_THRESHOLD = DEFAULT_TRUNCATION_THRESHOLD
# 拟合：次数不变。weighted = 加权 Chebyshev LS + Lawson，压负半轴（尤其 |GELU|≈0）。
# remez = 原均匀 minimax（负半轴容易振荡到 >0.1）。
FIT_METHOD = "weighted"
NEG_WEIGHT = 30.0
FLAT_WEIGHT = 12.0  # 1/(|GELU|+ε)，尾部更贴 0
NEAR0_X = 8.0  # 负半轴靠近 0 的窗口
NEAR0_WEIGHT = 20.0
POS_WEIGHT = 0.15  # 正半轴远离 0 的底权
POS_NEAR0_X = 8.0  # 正半轴靠近 0 时权重大、仍低于负半轴
POS_NEAR0_WEIGHT = 12.0
OVERSHOOT_CAP = 0.05  # x<0 时 pred 超过此值加重罚
OVERSHOOT_WEIGHT = 120.0
LAWSON_ITERS = 12
FIT_GRID_N = 12000
RESULTS_DIR = os.path.join(_HERE, "gelu")
LAYER_COEFF_JSON = os.path.join(RESULTS_DIR, "gelu_direct_layer_coeffs.json")


def _he_direct(degree: int) -> dict:
    he = gelu_he_depth_breakdown(degree, eval_method="ps_tree")
    return {
        "y_eval": he["y_eval"],
        "cheb_combo_rescale": he["cheb_combo_rescale"],
        "gelu_reconstruct": 0,
        "total": int(he["y_eval"]) + int(he["cheb_combo_rescale"]),
        "eval_method": he["eval_method"],
    }


def _cheb_nodes(a: float, b: float, n: int) -> np.ndarray:
    if n <= 0 or b <= a:
        return np.zeros(0, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    z = np.cos(np.pi * (2.0 * k + 1.0) / (2.0 * n))
    return (0.5 * (b - a) * z + 0.5 * (a + b)).astype(np.float64)


def _neg_emphasis_weight(x: np.ndarray) -> np.ndarray:
    """负半轴权最大；正半轴越靠近 0 越大，但仍低于负半轴。"""
    x = np.asarray(x, dtype=np.float64)
    g = gelu_exact(x)
    w = np.ones_like(x)
    neg = x < 0.0
    w[neg] *= float(NEG_WEIGHT)
    w[neg] *= 1.0 + float(FLAT_WEIGHT) / (np.abs(g[neg]) + 1e-3)
    near = neg & (x > -float(NEAR0_X))
    w[near] *= float(NEAR0_WEIGHT)
    pos = ~neg
    # 0+ 处 ≈ POS_WEIGHT*(1+POS_NEAR0_WEIGHT)，随 x 增大降回 POS_WEIGHT
    w[pos] *= float(POS_WEIGHT) * (
        1.0
        + float(POS_NEAR0_WEIGHT)
        / (x[pos] / float(POS_NEAR0_X) + 1.0)
    )
    w[pos] = np.minimum(w[pos], float(NEG_WEIGHT) * 0.5)
    return w


def _weighted_cheb_fit(
    x_lo: float,
    x_hi: float,
    degree: int,
) -> tuple[np.ndarray, float]:
    """
    次数固定的加权 Chebyshev 最小二乘 + Lawson IRLS。
    负半轴（尤其真值≈0）误差优先；x<0 且 pred>OVERSHOOT_CAP 再加罚。
    """
    n = int(FIT_GRID_N)
    n_neg = max(degree + 4, int(0.72 * n))
    n_pos = max(degree + 4, n - n_neg)
    x0 = float(x_lo)
    x1 = float(x_hi)
    chunks = [_cheb_nodes(x0, min(0.0, x1), n_neg)]
    if x1 > 0.0:
        chunks.append(_cheb_nodes(max(0.0, x0), x1, n_pos))
    x = np.unique(np.concatenate(chunks))
    y = gelu_exact(x)
    z = x_to_cheb_z(x, x0, x1)
    vander = chebvander(z, int(degree))
    w0 = _neg_emphasis_weight(x)
    w = w0.copy()
    eps = 1e-8
    cheb_c = np.zeros(int(degree) + 1, dtype=np.float64)
    for _ in range(int(LAWSON_ITERS)):
        sw = np.sqrt(np.clip(w, 1e-12, None))
        cheb_c, *_ = np.linalg.lstsq(vander * sw[:, None], y * sw, rcond=None)
        pred = vander @ cheb_c
        err = pred - y
        w = w0 * (np.abs(err) + eps)
        over = (x < 0.0) & (pred > float(OVERSHOOT_CAP))
        if np.any(over):
            w[over] *= float(OVERSHOOT_WEIGHT) * (
                1.0 + (pred[over] - float(OVERSHOOT_CAP)) / float(OVERSHOOT_CAP)
            )
    pred = vander @ cheb_c
    ripple = float(np.max(np.abs(pred - y)))
    return cheb_c.astype(np.float64), ripple


def _region_fit_stats(
    coeffs: tuple[float, ...] | np.ndarray,
    f_domain: tuple[float, float],
    x_lo: float,
    x_hi: float,
    n: int = 8001,
) -> dict[str, float]:
    x = np.linspace(x_lo, x_hi, n, dtype=np.float64)
    pred = cheb_eval_on_domain(
        np.asarray(coeffs, dtype=np.float64), x, f_domain, method="clenshaw"
    )
    g = gelu_exact(x)
    err = np.abs(pred - g)
    neg = x < 0.0
    near0 = neg & (x > -float(NEAR0_X))
    flat = neg & (np.abs(g) < 0.02)
    return {
        "max_abs": float(np.max(err)),
        "neg_max_abs": float(np.max(err[neg])) if np.any(neg) else 0.0,
        "near0_max_abs": float(np.max(err[near0])) if np.any(near0) else 0.0,
        "neg_max_pred": float(np.max(pred[neg])) if np.any(neg) else 0.0,
        "flat_max_pred": float(np.max(pred[flat])) if np.any(flat) else 0.0,
        "flat_max_abs": float(np.max(err[flat])) if np.any(flat) else 0.0,
    }


@lru_cache(maxsize=None)
def _fit_direct(
    x_lo: float, x_hi: float, degree: int, trunc: float
) -> tuple[tuple[float, ...], tuple[float, float], float]:
    a, b = float(x_lo), float(x_hi)

    def target(x: np.ndarray) -> np.ndarray:
        return gelu_exact(np.asarray(x, dtype=np.float64))

    if FIT_METHOD == "remez":
        cheb_c, ripple = remez_minimax_cheb(target, degree, a, b)
    else:
        cheb_c, ripple = _weighted_cheb_fit(a, b, degree)
    cheb_c = truncate_cheb_coeffs(cheb_c, trunc)
    return (
        tuple(float(v) for v in cheb_c.tolist()),
        (a, b),
        float(ripple),
    )


class _DirectGelu:
    """内存系数 → 与 gelu_poly kind=direct 相同的求值。"""

    def __init__(
        self,
        coeffs: tuple[float, ...],
        f_domain: tuple[float, float],
    ):
        self._cfg = {
            "kind": "direct",
            "f_domain": (float(f_domain[0]), float(f_domain[1])),
            "f_cheb_coeffs": list(coeffs),
        }
        self._f: torch.Tensor | None = None
        self._key: tuple[torch.device, torch.dtype] | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if self._key != key:
            self._f = _coeffs_to_f64(self._cfg["f_cheb_coeffs"], x.device)
            self._key = key
        return _direct_gelu_cheb(x, self._cfg, self._f)


def _pick_best(rows: list[dict]) -> dict:
    """KL 更小 → flips 更小 → 次数更低 → acc 最高。"""

    def key(r: dict) -> tuple:
        acc = r["acc"]
        kl = r["kl"]
        flips = r["flips_pct"]
        acc_k = acc if acc == acc else -1.0
        kl_k = kl if kl == kl else 1e9
        fl_k = flips if flips == flips else 1e9
        return (kl_k, fl_k, int(r["degree"]), -acc_k)

    return min(rows, key=key)


def _fit_max_abs(
    coeffs: tuple[float, ...],
    f_domain: tuple[float, float],
    x_lo: float,
    x_hi: float,
    n: int = 4001,
) -> float:
    x = np.linspace(x_lo, x_hi, n, dtype=np.float64)
    pred = cheb_eval_on_domain(np.asarray(coeffs), x, f_domain, method="clenshaw")
    return float(np.max(np.abs(pred - gelu_exact(x))))


def _layer_scheme_item(
    layer_idx: int,
    x_lo: float,
    x_hi: float,
    degree: int,
    coeffs: tuple[float, ...],
    f_domain: tuple[float, float],
    gelu_max_err: float,
) -> dict:
    he = _he_direct(degree)
    return {
        "rank": layer_idx,
        "name": f"direct_L{layer_idx}_d{degree}",
        "layer": int(layer_idx),
        "kind": "direct",
        "fit_x": [float(x_lo), float(x_hi)],
        "depth": int(he["total"]),
        "depth_he_breakdown": he,
        "gelu_max_err": float(gelu_max_err),
        "params": {
            "degree": int(degree),
            "f_cheb_coeffs": [float(v) for v in coeffs],
            "f_domain": [float(f_domain[0]), float(f_domain[1])],
        },
    }


def _save_layer_coeffs(schemes: list[dict], path: str) -> str:
    payload = {
        "kind": "direct",
        "per_layer": True,
        "note": (
            "Per-layer direct GELU: ĝ(x)=Cheb(x) on LAYER_FIT_X "
            f"(affine to [-1,1]); fit={FIT_METHOD}."
        ),
        "schemes": schemes,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单层 GeLU 次数搜索（进模型）")
    p.add_argument("--task", default=TASK)
    p.add_argument(
        "--eval-split",
        choices=("validation", "calib"),
        default=EVAL_SPLIT,
    )
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--degrees", type=int, nargs="+", default=None)
    p.add_argument(
        "--layer-degrees",
        type=int,
        nargs=NUM_LAYERS,
        default=None,
        metavar="D",
        help="覆盖 LAYER_DEGREES：12 个整数，每层一个最高次数",
    )
    p.add_argument("--truncation-threshold", type=float, default=None)
    p.add_argument(
        "--combo-k",
        type=int,
        default=None,
        help="任选 k 层同时近似；默认用配置 COMBO_K",
    )
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--fit-only",
        action="store_true",
        help="只拟合并写系数，不加载模型、不测 acc",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    task = str(args.task)
    eval_split = str(args.eval_split)
    layers = tuple(LAYERS if args.layers is None else args.layers)
    if args.layer_degrees is not None:
        layer_degrees: list[int] | None = [int(x) for x in args.layer_degrees]
    else:
        layer_degrees = (
            None if LAYER_DEGREES is None else [int(x) for x in LAYER_DEGREES]
        )
    if layer_degrees is not None:
        if len(layer_degrees) != NUM_LAYERS:
            raise ValueError(
                f"LAYER_DEGREES 长度须为 {NUM_LAYERS}，当前 {len(layer_degrees)}"
            )
        if any(d < 1 for d in layer_degrees):
            raise ValueError(f"LAYER_DEGREES 须 ≥ 1，收到 {layer_degrees}")
    degrees = tuple(DEGREES if args.degrees is None else args.degrees)
    trunc = float(
        TRUNCATION_THRESHOLD
        if args.truncation_threshold is None
        else args.truncation_threshold
    )
    if layer_degrees is None and any(d < 1 for d in degrees):
        raise ValueError(f"degree 须 ≥ 1，收到 {degrees}")
    if any(L < 0 or L >= NUM_LAYERS for L in layers):
        raise ValueError(f"layer 须在 0..{NUM_LAYERS - 1}，收到 {layers}")
    layer_fit = [[float(a), float(b)] for a, b in LAYER_FIT_X]
    if len(layer_fit) != NUM_LAYERS:
        raise ValueError(
            f"LAYER_FIT_X 长度须为 {NUM_LAYERS}，当前 {len(layer_fit)}"
        )
    for i, pair in enumerate(layer_fit):
        if len(pair) != 2:
            raise ValueError(f"LAYER_FIT_X[{i}] 须为 [lo, hi]，收到 {pair}")
        if pair[0] >= pair[1]:
            raise ValueError(f"LAYER_FIT_X[{i}] 左端须小于右端：{pair}")

    if args.fit_only:
        if layer_degrees is None:
            raise ValueError("--fit-only 需要 LAYER_DEGREES")
        print(f"仅拟合 FIT_METHOD={FIT_METHOD}", flush=True)
        schemes: list[dict] = []
        dummy = torch.zeros(1, dtype=torch.float64)
        for i in range(NUM_LAYERS):
            x_lo, x_hi = layer_fit[i]
            deg = int(layer_degrees[i])
            coeffs, f_domain, ripple = _fit_direct(x_lo, x_hi, deg, trunc)
            st = _region_fit_stats(coeffs, f_domain, x_lo, x_hi)
            schemes.append(
                _layer_scheme_item(
                    i, x_lo, x_hi, deg, coeffs, f_domain, st["max_abs"]
                )
            )
            print(
                f"  L{i:02d} x∈[{x_lo:g},{x_hi:g}] deg={deg} "
                f"max|e|={st['max_abs']:.3e} neg|e|={st['neg_max_abs']:.3e} "
                f"neg_max_pred={st['neg_max_pred']:.4f} "
                f"flat_max_pred={st['flat_max_pred']:.4f} "
                f"({FIT_METHOD} {ripple:.3e})",
                flush=True,
            )
            _DirectGelu(coeffs, f_domain)(dummy)
        if SAVE_COEFFS and not args.no_save:
            path = _save_layer_coeffs(schemes, LAYER_COEFF_JSON)
            print(f"系数已写入 {path}", flush=True)
        return

    from poly_model_inference import (
        abnormal_output_mask,
        collect_output_logits,
        evaluate_logits_metrics,
        fmt_accuracy,
        fmt_accuracy_delta,
        fmt_metric_delta,
        install_eager_attention_poly_patch,
        mean_output_kl,
        _load_tokenized_eval_split,
        _load_tokenizer_and_model,
        _patch_gelu_module,
        _restore_gelu_module,
    )

    install_eager_attention_poly_patch()
    tokenizer, model, _ = _load_tokenizer_and_model(task)
    tokenized, split_desc = _load_tokenized_eval_split(
        task, tokenizer, eval_split=eval_split
    )
    print(f"{task.upper()} {split_desc}", flush=True)
    print("拟合 ĝ(x)=Cheb(x) on LAYER_FIT_X（仿射到 [-1,1]）", flush=True)
    print(f"  LAYER_FIT_X={layer_fit}", flush=True)
    if layer_degrees is not None:
        print(f"  LAYER_DEGREES={layer_degrees}", flush=True)
    else:
        print(f"  degrees={list(degrees)}", flush=True)

    with torch.no_grad():
        logits_base, labels = collect_output_logits(model, tokenized, tokenizer)
    base = evaluate_logits_metrics(logits_base, labels, task)
    if task == "mrpc":
        print(
            f"baseline  acc={fmt_accuracy(base['val_accuracy'])}  "
            f"F1={base['val_f1'] * 100:.2f}%",
            flush=True,
        )
    else:
        print(
            f"baseline  acc={fmt_accuracy(base['val_accuracy'])}",
            flush=True,
        )

    hdr = (
        f"{'L':<4} {'fit_lo':>8} {'fit_hi':>8} {'deg':>5} {'HE':>4} "
        f"{'acc':>8} {'Δacc':>9} {'F1':>8} {'ΔF1':>9} "
        f"{'flips%':>8} {'KL':>12}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    all_rows: list[dict] = []
    best_by_layer: dict[int, dict] = {}
    gelu_mods = [
        model.bert.encoder.layer[i].intermediate.intermediate_act_fn
        for i in range(NUM_LAYERS)
    ]

    dummy = torch.zeros(
        1,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )

    combo_k = COMBO_K if args.combo_k is None else int(args.combo_k)
    do_save = SAVE_COEFFS and not args.no_save

    packed: dict[int, dict] = {}
    if do_save or (combo_k is not None and combo_k >= 1):
        if layer_degrees is None:
            raise ValueError("存系数 / 组合测试需要 LAYER_DEGREES（每层一个次数）")
        print("按当前 LAYER_FIT_X / LAYER_DEGREES 拟合 ...", flush=True)
        schemes: list[dict] = []
        for i in range(NUM_LAYERS):
            x_lo, x_hi = layer_fit[i]
            deg = int(layer_degrees[i])
            coeffs, f_domain, ripple = _fit_direct(x_lo, x_hi, deg, trunc)
            ev = _DirectGelu(coeffs, f_domain)
            with torch.no_grad():
                ev(dummy)
            packed[i] = {
                "fit": (x_lo, x_hi),
                "degree": deg,
                "he": _he_direct(deg),
                "coeffs": coeffs,
                "f_domain": f_domain,
                "ev": ev,
            }
            err = _fit_max_abs(coeffs, f_domain, x_lo, x_hi)
            st = _region_fit_stats(coeffs, f_domain, x_lo, x_hi)
            schemes.append(
                _layer_scheme_item(
                    i, x_lo, x_hi, deg, coeffs, f_domain, err
                )
            )
            print(
                f"  L{i:02d} x∈[{x_lo:g},{x_hi:g}] deg={deg} "
                f"HE={packed[i]['he']['total']} "
                f"max|e|={st['max_abs']:.3e} neg|e|={st['neg_max_abs']:.3e} "
                f"neg_max_pred={st['neg_max_pred']:.4f} "
                f"flat_max_pred={st['flat_max_pred']:.4f} "
                f"({FIT_METHOD} {ripple:.3e})",
                flush=True,
            )
        if do_save:
            path = _save_layer_coeffs(schemes, LAYER_COEFF_JSON)
            print(f"系数已写入 {path}", flush=True)

    if combo_k is not None and combo_k >= 1:
        if combo_k > NUM_LAYERS:
            raise ValueError(f"COMBO_K={combo_k} 超过层数 {NUM_LAYERS}")
        combos = list(itertools.combinations(range(NUM_LAYERS), combo_k))
        print(
            f"任选 {combo_k} 层近似（C({NUM_LAYERS},{combo_k})={len(combos)}），"
            f"其余 original",
            flush=True,
        )
        hdr_c = (
            f"{'layers':<16} {'acc':>8} {'Δacc':>9} {'F1':>8} "
            f"{'flips%':>8} {'KL':>12}"
        )
        print(hdr_c, flush=True)
        print("-" * len(hdr_c), flush=True)
        combo_rows: list[dict] = []
        active: set[int] = set()
        for combo in combos:
            want = set(combo)
            for i in active - want:
                _restore_gelu_module(gelu_mods[i])
            for i in want - active:
                _patch_gelu_module(gelu_mods[i], packed[i]["ev"])
            active = want
            with torch.no_grad():
                logits, _ = collect_output_logits(model, tokenized, tokenizer)
            mtr = evaluate_logits_metrics(
                logits, labels, task, baseline_logits=logits_base
            )
            kl = mean_output_kl(
                logits_base, logits, valid_mask=~abnormal_output_mask(logits)
            )
            f1 = float(mtr.get("val_f1", float("nan")))
            row = {
                "layers": ",".join(str(i) for i in combo),
                "acc": float(mtr["val_accuracy"]),
                "acc_delta": float(mtr.get("accuracy_delta", float("nan"))),
                "f1": f1,
                "f1_delta": float(mtr.get("f1_delta", float("nan"))),
                "flips_pct": float(mtr.get("flips_pct", float("nan"))),
                "kl": float(kl),
                "oor_pct": float(mtr["oor_pct"]),
            }
            combo_rows.append(row)
            f1_s = f"{f1 * 100:7.2f}%" if f1 == f1 else "     nan"
            kl_s = f"{kl:.6f}" if kl == kl else "nan"
            print(
                f"{row['layers']:<16} {fmt_accuracy(row['acc']):>8} "
                f"{fmt_accuracy_delta(row['acc_delta']):>9} {f1_s:>8} "
                f"{row['flips_pct']:7.2f}% {kl_s:>12}",
                flush=True,
            )
        for i in active:
            _restore_gelu_module(gelu_mods[i])
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_csv = os.path.join(
            RESULTS_DIR, f"gelu_combo{combo_k}_{task}_{eval_split}.csv"
        )
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "layers",
                    "acc",
                    "acc_delta",
                    "f1",
                    "f1_delta",
                    "flips_pct",
                    "kl",
                    "oor_pct",
                ],
            )
            w.writeheader()
            w.writerows(combo_rows)

        def _finite_kl(r: dict) -> float:
            k = r["kl"]
            return k if k == k else 1e9

        by_kl = sorted(combo_rows, key=_finite_kl)
        by_acc = sorted(combo_rows, key=lambda r: -r["acc"])
        print()
        print(f"KL 最好 10 / {len(combo_rows)}")
        for r in by_kl[:10]:
            print(
                f"  {r['layers']:<16} acc={fmt_accuracy(r['acc'])} "
                f"Δacc={fmt_accuracy_delta(r['acc_delta'])} "
                f"flips={r['flips_pct']:.2f}% KL={r['kl']:.6f}"
            )
        print(f"acc 最好 10 / {len(combo_rows)}")
        for r in by_acc[:10]:
            print(
                f"  {r['layers']:<16} acc={fmt_accuracy(r['acc'])} "
                f"Δacc={fmt_accuracy_delta(r['acc_delta'])} "
                f"flips={r['flips_pct']:.2f}% KL={r['kl']:.6f}"
            )
        print(f"KL 最差 5")
        for r in by_kl[-5:][::-1]:
            print(
                f"  {r['layers']:<16} acc={fmt_accuracy(r['acc'])} "
                f"Δacc={fmt_accuracy_delta(r['acc_delta'])} "
                f"flips={r['flips_pct']:.2f}% KL={r['kl']:.6f}"
            )
        print(f"明细：{out_csv}")
        return

    for layer_idx in layers:
        x_lo, x_hi = layer_fit[layer_idx]
        layer_rows: list[dict] = []
        deg_list = (
            (int(layer_degrees[layer_idx]),)
            if layer_degrees is not None
            else degrees
        )
        for degree in deg_list:
            coeffs, f_domain, _ripple = _fit_direct(
                x_lo, x_hi, int(degree), trunc
            )
            he = _he_direct(int(degree))
            for m in gelu_mods:
                _restore_gelu_module(m)
            ev = _DirectGelu(coeffs, f_domain)
            _patch_gelu_module(gelu_mods[layer_idx], ev)
            with torch.no_grad():
                ev(dummy)
                logits, _ = collect_output_logits(model, tokenized, tokenizer)
            mtr = evaluate_logits_metrics(
                logits, labels, task, baseline_logits=logits_base
            )
            kl = mean_output_kl(
                logits_base, logits, valid_mask=~abnormal_output_mask(logits)
            )
            f1 = float(mtr.get("val_f1", float("nan")))
            f1_d = float(mtr.get("f1_delta", float("nan")))
            row = {
                "layer": int(layer_idx),
                "fit_lo": x_lo,
                "fit_hi": x_hi,
                "degree": int(degree),
                "he_depth": int(he["total"]),
                "acc": float(mtr["val_accuracy"]),
                "acc_delta": float(mtr.get("accuracy_delta", float("nan"))),
                "f1": f1,
                "f1_delta": f1_d,
                "flips_pct": float(mtr.get("flips_pct", float("nan"))),
                "kl": float(kl),
                "oor_pct": float(mtr["oor_pct"]),
            }
            layer_rows.append(row)
            all_rows.append(row)
            f1_s = f"{f1 * 100:7.2f}%" if f1 == f1 else "     nan"
            f1d_s = fmt_metric_delta(f1_d) if f1_d == f1_d else "nan"
            kl_s = f"{kl:.6f}" if kl == kl else "nan"
            print(
                f"{layer_idx:<4} {x_lo:8.1f} {x_hi:8.1f} "
                f"{degree:5d} {he['total']:4d} "
                f"{fmt_accuracy(row['acc']):>8} "
                f"{fmt_accuracy_delta(row['acc_delta']):>9} "
                f"{f1_s:>8} {f1d_s:>9} "
                f"{row['flips_pct']:7.2f}% {kl_s:>12}",
                flush=True,
            )
        best_by_layer[layer_idx] = (
            layer_rows[0] if layer_degrees is not None else _pick_best(layer_rows)
        )
        if layer_degrees is None:
            b = best_by_layer[layer_idx]
            print(
                f"  → L{layer_idx} 最优 deg={b['degree']}  "
                f"acc={fmt_accuracy(b['acc'])}  "
                f"Δacc={fmt_accuracy_delta(b['acc_delta'])}  "
                f"KL={b['kl']:.6f}  flips={b['flips_pct']:.2f}%",
                flush=True,
            )

    for m in gelu_mods:
        _restore_gelu_module(m)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(
        RESULTS_DIR, f"gelu_degree_search_{task}_{eval_split}.csv"
    )
    fields = [
        "layer",
        "fit_lo",
        "fit_hi",
        "degree",
        "he_depth",
        "acc",
        "acc_delta",
        "f1",
        "f1_delta",
        "flips_pct",
        "kl",
        "oor_pct",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    print()
    if layer_degrees is not None:
        print("各层指定次数")
    else:
        print("各层最优次数（KL ↓flips ↓deg ↓acc）")
    print(
        f"{'L':<4} {'fit_lo':>8} {'fit_hi':>8} {'deg':>5} {'HE':>4} "
        f"{'acc':>8} {'Δacc':>9} {'flips%':>8} {'KL':>12}"
    )
    print("-" * 80)
    for layer_idx in layers:
        b = best_by_layer[layer_idx]
        print(
            f"{layer_idx:<4} {b['fit_lo']:8.1f} {b['fit_hi']:8.1f} "
            f"{b['degree']:5d} {b['he_depth']:4d} "
            f"{fmt_accuracy(b['acc']):>8} "
            f"{fmt_accuracy_delta(b['acc_delta']):>9} "
            f"{b['flips_pct']:7.2f}% {b['kl']:12.6f}"
        )
    print(f"明细：{out_csv}")


if __name__ == "__main__":
    main()
