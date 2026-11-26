# HE — THOR/Liberate DualRail HE 主路径

本目录是 **THOR + Liberate、DualRail 打包** 的 HE 工作路径。  
多项式档位参数写在 `nonlinear/` 里，层入口打包写在 `data.py`。

| 路径 | 角色 |
|------|------|
| **共享** | 根目录 `*_poly.py` 提供档位表；HE 侧只保留推理用到的参数类 |
| **本目录** | Liberate 密文 + DualRail 线性 + 仓库 `*_poly` 非线性 + `bts_ops` |
| **资产** | `HE/liberate`、`HE/keys`、`HE/encoded_models`；GLUE 在仓库根 `glue_datasets/`，嵌入用 `finetuned_weight/` |

约束：优先改 `HE/**`。档位表在根目录 `*_poly.py`。

进化搜索的 bts / 深度统计在**仓库根** `cost.py`（默认 `--cost-mode bts`），不再使用本目录 shim。

---

## 目录（净化后）

### 核心

| 文件 | 作用 |
|------|------|
| `ckks.py` | CKKS engine 与密钥加载 |
| `linear.py` | DualRail 线性与 level 对齐 |
| `data.py` | 数据集入口与层入口 8-real 打包 |
| `bert.py` | Attention / FFN、单层前向、密文分类头 |
| `model_encoder.py` | 权重编码 |
| `nonlinear/` | Softmax、GeLU、LayerNorm、多项式、tanh |
| `utils/matrix.py` | 对角打包 |
| `bootstrap.py` | 按计划刷新密文（prescale、DualRail、hook） |
| `path_setup.py` | Liberate 运行时路径 |
| `geometry.py` | DualRail CT 条数 |
| `bts_ops.py` | 微事件图 + bootstrap 放置 DP（budget=14） |
| `rem_guard.py` | HE rem 与 DP 对照 |
| `encode_weights.py` | 编码 HE 权重（输出 `encoded_models/`，勿提交） |
| `HE_model_inference.py` | 12 层密文 encoder + 分类头，rem_guard |

---

## 策略

1. **端到端推理**：`HE_model_inference.py`（真实 bootstrap，按 `bts_ops` 计划）。  
2. **非线性**：固定多项式 Softmax / LayerNorm / Chebyshev GeLU，档位由 `--scheme` 或 `--softmax-level` / `--ln-level` / `--gelu-level` 决定。  
3. **bts / 深度**：根目录 `python3 cost.py`；进化用 `evolution_score.py` / `evolution_kl.py`。

---

## 官方资产

Drive：https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO

| 文件 | 放到 |
|------|------|
| `resources.tar` | `HE/liberate/src/liberate/fhe/cache/resources/` |
| `keys.tar` | `HE/keys/` |

引擎参数：`logN=16, scale_bits=41, num_special_primes=4` → `num_slots=32768`。资源与密钥已放在上表路径。

---

## 常用命令

```bash
cd /workspace

python3 HE/encode_weights.py --dataset mrpc --layers 0
python3 HE/HE_model_inference.py --dataset mrpc --sample 0

# 深度 / bts（明文模块在 Plain/）
python3 -c "import sys; sys.path.insert(0,'Plain'); from cost import print_scheme_cost; print_scheme_cost('mrpc', [1]*48)"
```

键约定：`rotk_dict/*` → bootstrap；普通 rotate → `add_rot_keys_from_sk`。

---

## CT 几何（BERT-base）

| 位置 | DualRail 条数 |
|------|----------------|
| hidden / resid | 4 cplx（↔ 8 real） |
| PC-MM rotated | 64 |
| score / Softmax | 8 real |
| Softmax 出口 α | 128 |
| V / context | 2 cplx |
| FF / GeLU | 16 |

---

## BootstrapHook

| mode | 行为 |
|------|------|
| `noop` | 不刷新 |
| `always` | 立刻 `engine.bootstrap` |
| `record` | 只记账 |
| `plan` | 按 `bts_ops` placements 做真实 bootstrap |
