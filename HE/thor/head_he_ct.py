"""
HE BertPooler + classifier（THOR DualRail 布局）。

入口：第 12 层 encoder 的 8 条 real DualRail（ln2_out）。
出口：n_cls 条密文，logit 在 slot 0（与官方 ThorBertClassifier 相同）。

Bts 不进 bts_plan：
  1. 入口 DualRail 8 强制刷一次（encoder 出口 rem 不固定）
  2. tanh：p1 前、p2 后各 ``engine.bootstrap`` 一次（同 THOR he_tanh）

Tanh：ŷ = p2(p1(z/C))，默认 C=5、nolinear/tanh_out 自拟合幂基。
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import torch
from liberate.fhe.data_struct import DataStruct

from tanh_poly import DEFAULT_COEFF_JSON, load_tanh_fit

try:
    from dualrail_bts import refresh_dualrail8
except ImportError:
    from .dualrail_bts import refresh_dualrail8  # type: ignore

from thor.nonlinear.polynomial import evaluate_polynomial_stockmeyer


def load_weights(path: str) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_THOR_ENC = os.path.join(_REPO, "thirdparty", "THOR-main", "encoded_models_new")

HEAD_ROT_DELTAS: list[int] = [
    32,
    64,
    128,
    1024,
    16384,
    -16,
    -32,
    -64,
    -128,
    -256,
    -512,
    -1024,
]


def encoded_head_pkl(dataset: str, kind: str, *, encoded_dir: str | None = None) -> str:
    if kind not in ("pooler", "cls"):
        raise ValueError(f"kind must be pooler|cls, got {kind!r}")
    root = encoded_dir or os.path.join(_THOR_ENC, dataset)
    return os.path.join(root, f"{kind}.pkl")


def ensure_head_rot_keys(engine, sk) -> None:
    """Classifier / CLS-broadcast 旋转；encoder 加载时未必带齐。"""
    engine.add_rot_keys_from_sk(list(HEAD_ROT_DELTAS), sk)


def _pad16(coeffs: Sequence[float]) -> np.ndarray:
    c = np.asarray(list(coeffs), dtype=np.float64).ravel()
    if c.size < 16:
        c = np.pad(c, (0, 16 - int(c.size)))
    if c.size != 16:
        raise ValueError(f"Stockmeyer 需要 16 个升幂系数，得到 {c.size}")
    return c


def _move_weight_arr_to_device(arr: np.ndarray, device: int) -> None:
    for index in np.ndindex(arr.shape):
        cell = arr[index]
        if isinstance(cell, torch.Tensor):
            arr[index] = [cell.to(f"cuda:{device}")]
        elif isinstance(cell, list) and cell and torch.is_tensor(cell[0]):
            arr[index] = [cell[0].to(f"cuda:{device}")]


def _tanh_coeffs(coeff_path: str | None, C: float | None) -> tuple[float, np.ndarray, np.ndarray]:
    path = coeff_path or DEFAULT_COEFF_JSON
    payload = load_tanh_fit(path)
    scale = float(C) if C is not None else float(payload["C"])
    p1 = _pad16(payload["p1_increasing"])
    p2 = _pad16(payload["p2_increasing"])
    return scale, p1, p2


def he_tanh_ct(engine, ct, *, p1: np.ndarray, p2: np.ndarray):
    """THOR he_tanh：bts → Stockmeyer(p1) → Stockmeyer(p2) → bts。"""
    ct = engine.bootstrap(ct)
    ct = evaluate_polynomial_stockmeyer(engine, p1, ct)
    ct = evaluate_polynomial_stockmeyer(engine, p2, ct)
    ct = engine.bootstrap(ct)
    return ct


def _pooler_dense(evaluator, engine, x4: np.ndarray, w: np.ndarray, b: np.ndarray):
    """官方 ThorBertPooler.dense：CLS mask + 广播 + 6 块对角 GEMM + bias。"""
    if x4.shape != (4,):
        raise ValueError(f"pooler dense 输入应为 (4,) 复密文，得到 {x4.shape}")
    lev = int(x4[0].level_calc)
    nslot = int(engine.num_slots)
    mask = np.zeros((nslot,), dtype=np.float64)
    mask[np.arange(nslot) % (2**11) < 6] = 1.0

    x = list(x4)
    for index, ct in enumerate(x):
        ct_masked = engine.mc_mult(mask, ct)
        for i in range(7):
            ct_rot = engine.rotate_left(ct_masked, -16 * (2**i))
            ct_masked = engine.cc_add(ct_masked, ct_rot)
        x[index] = ct_masked

    wx_l = np.full((6,), None, dtype=object)
    for l in range(6):
        temp = engine.pt_ct_mult(w[l, 0], x[0])
        for i in range(1, 4):
            temp = engine.cc_add(temp, engine.pt_ct_mult(w[l, i], x[i]))
        wx_l[l] = engine.rescale(temp)

    temp2 = engine.level_up(wx_l[0], lev + 3)
    for l in range(1, 6):
        rotated = evaluator.rotate_internal(wx_l[l], delta=6 - l, mode="block_diag_2")
        temp2 = engine.cc_add(temp2, rotated)
    temp2 = evaluator.rotsum(temp2, 2**11)
    out = engine.pc_add(b[0], temp2)
    out = engine.cc_add(out, engine.conjugate(out))
    wx = np.empty((1,), dtype=object)
    wx[0] = out
    return wx


def _classifier_dense(engine, x1: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """官方 ThorBertClassifier.dense：逐类内积 + rot-sum → slot 0。"""
    if x1.shape != (1,):
        raise ValueError(f"classifier 输入应为 (1,)，得到 {x1.shape}")
    n_cls = int(w.shape[0])
    wx = np.full((n_cls,), None, dtype=object)
    for i in range(n_cls):
        wx[i] = engine.pt_ct_mult(w[i], x1[0])
        temp = engine.cc_add(wx[i], engine.rotate_left(wx[i], 1))
        temp2 = engine.cc_add(temp, engine.rotate_left(temp, 2))
        wx[i] = engine.cc_add(temp2, engine.rotate_left(temp, 4))
        for j in range(4, 11):
            wx[i] = engine.cc_add(wx[i], engine.rotate_left(wx[i], 2**j))
        wx[i] = engine.rescale(wx[i])
        wx[i] = engine.pc_add(b[i], wx[i])
    return wx


class HeBertHead:
    """8 DualRail ln2 → pooler tanh → classifier logits CTs。"""

    def __init__(
        self,
        evaluator,
        *,
        dataset: str,
        encoded_dir: str | None = None,
        device: int = 0,
        tanh_C: float | None = None,
        tanh_coeff_path: str | None = None,
    ):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.device = int(device)
        enc = encoded_dir or os.path.join(_THOR_ENC, dataset)
        pooler_path = encoded_head_pkl(dataset, "pooler", encoded_dir=enc)
        cls_path = encoded_head_pkl(dataset, "cls", encoded_dir=enc)
        if not os.path.isfile(pooler_path):
            raise FileNotFoundError(f"missing pooler.pkl: {pooler_path}")
        if not os.path.isfile(cls_path):
            raise FileNotFoundError(f"missing cls.pkl: {cls_path}")
        pw = load_weights(pooler_path)
        cw = load_weights(cls_path)
        self.w_pool = pw["bert.pooler.dense.weight"]
        self.b_pool = pw["bert.pooler.dense.bias"]
        cls_name = (
            "cls.seq_relationship"
            if "cls.seq_relationship.weight" in cw
            else "classifier"
        )
        self.w_cls = cw[f"{cls_name}.weight"]
        self.b_cls = cw[f"{cls_name}.bias"]
        _move_weight_arr_to_device(self.w_pool, self.device)
        _move_weight_arr_to_device(self.b_pool, self.device)
        _move_weight_arr_to_device(self.w_cls, self.device)
        _move_weight_arr_to_device(self.b_cls, self.device)
        self.tanh_C, self.p1, self.p2 = _tanh_coeffs(tanh_coeff_path, tanh_C)

    def forward(self, packs8: Sequence[Any] | np.ndarray) -> np.ndarray:
        packs = list(packs8)
        if len(packs) != 8:
            raise ValueError(f"head 入口应为 8 条 DualRail，得到 {len(packs)}")
        rem0 = int(self.engine.num_levels) - int(packs[0].level_calc)
        print(
            f"  [he-head] entry rem={rem0} → DualRail8 bootstrap (not in plan)",
            flush=True,
        )
        packs = refresh_dualrail8(
            self.engine, packs, None, "head.entry_dualrail8"
        )
        rem1 = int(self.engine.num_levels) - int(packs[0].level_calc)
        print(f"  [he-head] after entry bts rem={rem1}", flush=True)

        x_cplx = np.empty((4,), dtype=object)
        for i in range(4):
            x_cplx[i] = self.engine.cc_add(packs[i], self.engine.imult(packs[i + 4]))
        pooled = _pooler_dense(
            self.evaluator, self.engine, x_cplx, self.w_pool, self.b_pool
        )
        pooled[0] = self.engine.mult_scalar(pooled[0], 1.0 / self.tanh_C)
        print(
            f"  [he-head] tanh C={self.tanh_C:g} Stockmeyer p2(p1(z/C)) + 2×bts",
            flush=True,
        )
        pooled[0] = he_tanh_ct(self.engine, pooled[0], p1=self.p1, p2=self.p2)
        logits_ct = _classifier_dense(self.engine, pooled, self.w_cls, self.b_cls)
        return logits_ct


def decrypt_logits_slot0(engine, sk, logits_ct: np.ndarray) -> np.ndarray:
    out = []
    for i in range(len(logits_ct)):
        vec = np.asarray(engine.decrode(logits_ct[i], sk, is_real=False)).ravel()
        out.append(float(np.real(vec[0])))
    return np.asarray(out, dtype=np.float64)
