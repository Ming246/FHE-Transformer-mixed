# HE/thor — THOR/Liberate DualRail HE 主路径

本目录是 **THOR + Liberate、DualRail 打包** 的 HE 工作路径。  
共享明文同构：`HE/slot_*_he.py`、`HE/slot_decode.py`、`HE/thor_encoder_linear_core.py`。

| 路径 | 角色 |
|------|------|
| **共享** | `HE/slot_*`、`HE/thor_encoder_linear_core.py` |
| **本目录** | Liberate 密文 + DualRail 线性 + 仓库 `*_poly` 非线性 + `bts_ops` |
| **上游** | `thirdparty/THOR-main/`（勿改业务逻辑；资产放其下） |

约束：优先改 `HE/thor/**`；共享 `slot_*` / core 仅在 DualRail 合约需要时改。

进化搜索的 bts / 深度统计在**仓库根** `cost.py`（默认 `--cost-mode bts`），不再使用本目录 shim。

---

## 目录（净化后）

### 核心

| 文件 | 作用 |
|------|------|
| `path_setup.py` | Liberate/THOR 路径（避免 `HE/thor` 遮蔽上游 `thor`） |
| `engine.py` | Liberate engine / keys |
| `geometry.py` | DualRail CT 条数 |
| `bts_ops.py` | 微事件图 + bootstrap 放置 DP（budget=14） |
| `bootstrap_hook.py` | 按 placements 在 HE 前向里刷新 |
| `rem_guard.py` | HE rem 与 DP 对照 |
| `dualrail_bts.py` / `dualrail_encode.py` | pack→bts→unpack；权重/激活编码 |
| `linear_eval.py` | DualRail 线性（QKV / score / FF / …） |
| `softmax_he_ct.py` / `layernorm_he_ct.py` / `gelu_he_ct.py` | HE 非线性 |
| `encode_thor_weights.py` | 编码 HE 权重（输出 `encoded_models_new/`，勿提交） |
| `he_depth_profile.py` / `level_probe.py` | 实测 rem → depth overrides |
| `export_layer_bootstrap_graph.py` | 导出单层事件图（论文/调试） |

### Smoke / 资产

| 文件 | 作用 |
|------|------|
| `smoke_ckks.py` | 密文原语 |
| `smoke_linear.py` | DualRail 线性 |
| `smoke_bootstrap.py` | 真 bootstrap round-trip |
| `smoke_thor_repro.py` | **主入口**：一层 HE vs 明文 BERT |
| `smoke_thor_chain.py` | 多层链 + rem_guard |
| `fetch_official_assets.py` / `setup_official_assets.sh` | 官方 resources/keys |
| `gen_scale_primes_minimal.py` | 无官方资源时的最小 primes |
| `run_remguard_l0.sh` | 后台跑 L0 plan_mock+rem_guard |

---

## 策略

1. **先复现 THOR**：`smoke_thor_repro.py`（官方 Liberate + DualRail 权重）。  
2. **再换仓库非线性**：`--softmax poly --ln poly --gelu cheb`（默认）。  
3. **bts / 深度**：根目录 `python3 cost.py`；进化用 `evolution_score.py` / `evolution_kl.py`。

---

## 官方资产

Drive：https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO

| 文件 | 放到 |
|------|------|
| `resources.tar` | `thirdparty/THOR-main/liberate/src/liberate/fhe/cache/resources/` |
| `keys.tar` | `thirdparty/THOR-main/keys/` |

```bash
python3 HE/thor/fetch_official_assets.py --only resources.tar keys.tar --resume
bash HE/thor/setup_official_assets.sh
```

引擎参数：`logN=16, scale_bits=41, num_special_primes=4` → `num_slots=32768`。

---

## 常用命令

```bash
cd /workspace

# Liberate 健康检查
python3 HE/thor/smoke_ckks.py
python3 HE/thor/smoke_linear.py
python3 HE/thor/smoke_bootstrap.py --burn-levels 6

# 主复现 / 链
python3 HE/thor/encode_thor_weights.py --dataset mrpc --layers 0
python3 HE/thor/smoke_thor_repro.py --layer 0 --bootstrap mock --rest-div 1
python3 HE/thor/smoke_thor_chain.py --layers 0,1 --bootstrap plan_mock

# 深度 / bts（根目录）
python3 -c "from cost import print_scheme_cost; print_scheme_cost('mrpc', [2]*48)"
python3 HE/thor/export_layer_bootstrap_graph.py --layer 0 --level 2
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
| `plan` / `plan_mock` | 按 `bts_ops` placements |

```bash
python3 HE/thor/smoke_thor_repro.py --bootstrap plan_mock --probe-levels
```
