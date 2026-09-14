# Softmax HE 定标 / 倍数差距（待挖）

记录 plaintext Δ 仿真 vs 真实 HE 解密在各切点的 **全局 scale 差**；形状/相对分布多数一致，靠 LS gain 可对齐，但 raw 解密不等于 `thor_softmax` 概率（行和≠1）。

**后续假设：** 倍数可能在 DualRail exit 之后某步（context `rescale`、`2·Re` unpack、`make_copies` gain 等）被补回；mid 与 exit 的因子可能不同，需沿 pipeline 逐站查。

---

## 切点与脚本

| 切点 | HE 位置 | 脚本 | plain 对照 |
|------|---------|------|------------|
| **8pack 捷径** | `update_inv_D` 末轮后，**exit 前**；明文用 `exp×inv/Δ` | — | `plain_softmax_delta_sim`（`full_he_exit=False`） |
| **mid-probe** | 同上，HE 解密 `exp_u`+`inv_d` | `smoke_softmax_he_mid_probe.py` | 同上 normalize |
| **full exit** | `dualrail_softmax_exit` **之后**，128 CT 解密 | `smoke_softmax_he_vs_plain_exit.py` | `full_he_exit=True` |

`he_softmax_poly(..., return_before_exit=True)` → 返回 `(exp_u, inv_d, d_delta)`，不跑 exit。

---

## 已观测（mrpc L0 high, seed=0, mock bts）

### 1. 8pack 捷径（plain only）

- vs `thor_softmax`：dense max\|err\| ~ **1e-4**，行和 Σ≈ **1** ✓

### 2. Mid-probe（exit 前解密）

- HE `decrypt(exp_u)×(decrypt(inv_d)/Δ)` vs plain8：max\|err\| ~ **7.8e-3**，LS gain≈ **1**
- HE vs thor：max\|err\| ~ **7.9e-3**；**行和 mean Σ≈ 0.0625（≈1/16）**，thor/plain8≈1
- `max|α|` HE 与 plain8/thor 量级相近（~7.8e-3），但 **归一化（行和）不对**

### 3. Full exit 后（128 解密 → 8pack decode）

- HE vs plain full exit raw：max\|err\| ~ **0.12**，LS gain≈ **16**
- @gain：max\|err\| ~ **1.8e-4**（形状一致）
- 行和：HE raw **Σ≈0.5**，plain raw **Σ≈8**（比 **16×**）；×gain 后 plain 侧可回到 ~1

### 4. 用户提议的成对还原（未完整跑完，理论关系）

- HE×**2** vs plain÷**8** 应接近 mid 的 16× 关系（full exit 上 plain/HE≈16）；待复测写入本表。

---

## 暂结论

- **Softmax 主干（Stockmeyer + aSOR + Σy² + plain `inv/Δ`）与 `thor_softmax` 一致**；问题在 **HE 密文 ↔ logical 的 decrypt/定标**，不是公式本身。
- **Mid 与 exit 倍数不同**（~1/16 行和 vs ~16 pack ratio），说明倍数 **不是单一常数**，可能分阶段累积或在下游抵消。
- Plain full exit 镜像相对真实 HE exit 解密可能有 **~16×** 过冲（待查 `rotsum` / `2·Re` / `scale_k`）。

---

## TODO（后续挖掘）

1. Mid：`exp_u`/`inv_d` 解密是否缺 CKKS scale、`align_bypass` 语义、或 `exp_2u` slot 布局与 decode 不匹配（行和 1/16）。
2. Exit：plain `plain_dualrail_softmax_exit` 哪一步引入 **16×**（vs HE 128 解密）。
3. 沿 `smoke_thor_repro` / context 路径：exit 后 `rescale(α)`、`gain`、`thor_context_to_mainline_packs` 是否补上 mid 缺的因子。
4. 同一输入上补跑 HE×2 vs plain÷8 与逐 micro-event 解密探针。
