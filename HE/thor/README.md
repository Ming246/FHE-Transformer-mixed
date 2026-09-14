# HE/thor — THOR/Liberate DualRail HE 主路径

本目录是 **thirdparty/THOR + Liberate、DualRail/复数打包** 的 HE 工作路径。  
共享明文同构算子：`HE/slot_*_he.py`、`HE/thor_encoder_linear_core.py`（配置/几何辅助；旧纯实 e2e / `HE/liberate` 已移除）。

| 路径 | 角色 |
|------|------|
| **共享** | `HE/slot_*_he.py`、`HE/slot_decode.py`、`HE/thor_encoder_linear_core.py` |
| **本目录** | Liberate 密文 + THOR DualRail 线性 + 仓库 `*_poly` 非线性 + `bts_ops` |
| **上游** | `thirdparty/THOR-main/src/thor/*`、`thirdparty/THOR-main/liberate/` |

约束：优先改 `HE/thor/**`；共享 `slot_*` / core 仅在 DualRail 合约需要时改。

---

## 策略（复现 → 再替换）

1. **先原样复现 THOR**（`forward.ipynb` 路径）：官方 Liberate + keys0 + DualRail 编码权重 + THOR 自带 Softmax/GeLU/LN。  
   入口：`smoke_thor_repro.py`（一层 HE vs 明文 BERT 中间量）。  
   权重编码：`encode_thor_weights.py`（单卡 remap `cuda:1→0`，逻辑同上游 `encode.py`）。
2. **复现 PASS 后**再换：真实 `finetuned_weight/` / GLUE 数据 + 仓库多项式非线性（Softmax aSOR、LN、**Chebyshev GeLU** 等）。  
   线性 DualRail 不动；非线性参数/基函数替换应是局部改动。

此前分块 smoke（score/context + Softmax twin / 经验 gain）是探路记录，**不再作为主路线**。

---

## 当前状态（库可用性已验证）

| 项 | 状态 |
|----|------|
| Liberate CUDA 扩展 | **已编译**（`csprng`/`ntt`/`create_switcher` `.so` 在源码树内） |
| `import liberate` / `thor.ckks` | **OK**（见 `path_setup.ensure_thor_path`，避免 `HE/thor` 遮蔽上游 `thor`） |
| 官方 `resources/` + `keys/keys0` | **已就位**（`scale_primes.pkl` ~142KB / 240 keys；`gk` 4.5GB + `rotk_dict`） |
| `smoke_ckks.py` / `smoke_linear.py` | **PASS**（实部噪声 ~2.5e-5；DualRail×2，**不用** `mult_scalar`） |
| `smoke_bootstrap.py` | **PASS**（真 `engine.bootstrap`；~6min 载 key + ~2.5min bs；real max\|err\|~6e-3） |
| `smoke_qkv_pcmm.py` | **PASS**（满 4-pack；直接 enc@level_calc，禁大跨度 level_up；max\|err\|~1.5e-5） |
| `bts_ops.py`（本土） | **budget=14** + DualRail；`compute_level_discards` 推入口（不写死 21） |
| `smoke_transpose_k.py` | **PASS**（K transpose；噪声底 ~1e-2，tol 2e-2） |
| `smoke_make_copies.py` | **PASS**（DualRail make_copies；~5e-3） |
| `smoke_att_score.py` | **PASS**（vs dense ``K@Q.T``；decrypt/8；max\|err\|~1.2e-2，tol 5e-2） |
| `smoke_softmax_real.py` | **PASS**（真实 MRPC 权重+样本；``slot_softmax_he`` vs exact；max\|err\|~2.7e-4） |
| `smoke_thor_repro.py` | layer0 HE vs plain OK（`--rest-div 1`）。**默认非线性=仓库方案**（`--softmax poly --ln poly --gelu cheb`）；`--bootstrap mock` 排 bts 噪声。按层 slim 编码权重防 OOM |
| `gelu_he_ct.py` / `smoke_gelu_ct.py` | Liberate Chebyshev→monomial + THOR Stockmeyer；f2 仿射 bake 进系数（≈11 层、无 GeLU 内/后 bts） |
| `layernorm_he_ct.py` | DualRail LN：THOR 图 + `layernorm_poly` 方差/固定 iters；`--ln poly` |
| `softmax_he_ct.py` | DualRail Softmax：仓库 Stockmeyer exp + **固定步 aSOR**（`SoftmaxHeParams` e0/iters）+ **8→128 出口**；`--softmax poly`（L0 att_context PASS） |
| `mult_scalar` | **正文绕过**；标量 bake 进 weight encode `/2` + `+conjugate` |

---

## 官方资产 1:1（与上游 THOR README 对齐）

工程配置按 THOR 原样还原；**只改我们在 `HE/thor/` 的逻辑代码**。资产来自同一 Drive：

https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO

| Drive 文件 | 放到（解压后） |
|------------|----------------|
| `resources.tar`（~14.8GB） | `thirdparty/THOR-main/liberate/src/liberate/fhe/cache/resources/` |
| `keys.tar` | `thirdparty/THOR-main/keys/` |
| `datasets.tar` | `thirdparty/THOR-main/datasets/`（可选） |
| `encoded_models_new.tar` | `thirdparty/THOR-main/encoded_models_new/`（可选） |
| `finetuned_models.tar` | `thirdparty/THOR-main/finetuned_models/`（可选） |

暂存目录（下载未解压）：`thirdparty/THOR-main/_official_assets/`

```bash
# A) 容器内续传（Drive 未限流时）
python3 HE/thor/fetch_official_assets.py --only resources.tar keys.tar --resume

# B) 主机浏览器下载更稳（推荐，限流时常只能这样）
# 把 resources.tar / keys.tar 拷到：
#   F:\WSM\thirdparty\THOR-main\_official_assets\
# 然后在容器内安装：
bash HE/thor/setup_official_assets.sh

# C) 安装后重跑 smoke，对比噪声是否降到 ~1e-6～1e-9
python3 HE/thor/smoke_ckks.py
python3 HE/thor/smoke_linear.py
```

直链（浏览器打开）：

- resources: https://drive.google.com/uc?id=1LPQex129MuFclJp5F4sJN9fIrMVYLgXH  
- keys: https://drive.google.com/uc?id=1Sgfu0jt6HIyrou6XndZQRiK75L-7R5zJ  

引擎参数与上游一致（`encode.py` / `forward.ipynb`）：

```python
{"logN": 16, "scale_bits": 41, "num_special_primes": 4, "devices": [0], "quantum": "pre_quantum"}
```

→ `num_slots=32768`。

---

## 怎么跑（库 smoke）

```bash
cd /workspace

python3 HE/thor/smoke_ckks.py
python3 HE/thor/smoke_linear.py
# 真 bootstrap（需 GPU 显存充裕；首次载 key ~6min）
python3 HE/thor/smoke_bootstrap.py --burn-levels 6
# DualRail QKV PC-MM（默认 1 个 out-pack；~2–3min）
python3 HE/thor/smoke_qkv_pcmm.py --out-packs 1
python3 HE/thor/smoke_bts_plan.py
python3 HE/thor/smoke_transpose_k.py
python3 HE/thor/smoke_make_copies.py
python3 HE/thor/smoke_att_score.py
python3 HE/thor/smoke_softmax_real.py --sample 0 --layer 0 --level 2
python3 HE/thor/smoke_att_context.py
python3 HE/thor/path_setup.py
```

键加载约定（与 `forward.ipynb` 一致，见 `engine.load_thor_keys`）：

- `rotk_dict/*` → `engine.add_bs_key`（**仅** bootstrap；不是普通 rotate）
- 普通 rotate → `engine.add_rot_keys_from_sk(deltas, sk)`

`PYTHONPATH` 由 `path_setup.ensure_thor_path()` 自动注入：

- `thirdparty/THOR-main/liberate/src`
- `thirdparty/THOR-main/src`（**必须排在 `HE/` 之前**，否则 `import thor` 会进本适配包）

---

## 安装 Liberate（本机已完成的步骤）

```bash
cd /workspace/thirdparty/THOR-main/liberate/src/liberate/csprng
python3 setup.py build_ext --inplace
cd ../ntt && python3 setup.py build_ext --inplace
cd /workspace/thirdparty/THOR-main/liberate && python3 setup.py
```

可选：`pip install -e thirdparty/THOR-main/liberate`（非必须；当前用源码树 + PYTHONPATH）。

---

## 目录

| 文件 | 作用 |
|------|------|
| `path_setup.py` | Liberate/THOR 路径、诊断 |
| `engine.py` | `create_engine` / `create_keys` / `load_thor_keys`（bs_key=rotk_dict） |
| `geometry.py` | 复数 CT 条数表 + event 名粗映射 |
| `bootstrap_hook.py` | `noop` / `always` / `record` / **`plan`**（消费 `bts_ops` placements） |
| `dualrail_bts.py` | DualRail pack→bts→unpack（命名站点，供 smoke 前向） |
| `gen_scale_primes_minimal.py` | 自制最小 `scale_primes.pkl`（官方已到位后备用） |
| `fetch_official_assets.py` | 从 Drive 拉 `resources.tar` / `keys.tar`（支持 resume） |
| `setup_official_assets.sh` | 解压到上游 THOR 目录布局 |
| `smoke_ckks.py` | 密文原语 smoke |
| `smoke_linear.py` | DualRail + `make_rotated_copies`（无 mult_scalar） |
| `smoke_bootstrap.py` | 真 `engine.bootstrap` round-trip |
| `dualrail_encode.py` | DualRail 输入/权重 message 打包（`/2` bake） |
| `linear_eval.py` | QKV PC-MM；`safe_level_up`（span≤8）；入口直接 enc@level |
| `smoke_qkv_pcmm.py` | DualRail QKV PC-MM vs 主线纯实（禁 L0→深 level_up） |
| `bts_ops.py` | 本土化 bootstrap 放置（budget=14, DualRail geom） |
| `smoke_bts_plan.py` | bts_ops audit + plan hook 接线 |
| `smoke_transpose_k.py` | K transpose_upper_to_lower chunk |

---

## 复数 vs 纯实 CT（BERT-base）

| 流水线位置 | 主线纯实 | THOR 复数 | 比 |
|------------|----------|-----------|----|
| hidden / resid | 8 | **4** complex | ~1/2 |
| PC-MM rotated | 128 | **64** | 1/2 |
| Q/K/V out | 4 | 4 | 1 |
| Q copies | 64 | 64 | 1 |
| score / softmax | 8 | 8 real | 1 |
| alpha | 128 | 128 | 1 |
| V / context | 4 | **2** complex | 1/2 |
| FF / GeLU | 16 | (2,8)=16 | 1 |

**禁止**把纯实 `bts_ops` 的 `ct_count` 直接当 THOR 密文条数。

---

## Bootstrap 钩子（Phase 1 已有 / Phase 3 再接方案）

上游 `bert.py` 里到处是硬编码 `engine.bootstrap(...)`。本目录用 `BootstrapHook.refresh(event_name, cts)` 集中调用：

| mode | 行为 |
|------|------|
| `noop` | 不刷新（测线性） |
| `always` | 立刻 `engine.bootstrap`（需 `load_thor_keys(with_bootstrap=True)`） |
| `record` | 只记账 |
| `plan` | 仅 `plan_events` 内站点刷新；短距 discard（span≤8） |

Phase 3（已接线）：`smoke_thor_repro.py --bootstrap plan` 在固定 DualRail / Σy² summation 站点走 hook；aSOR/LN 内仍保留 reactive mid-bts。

### `cost_mode=bts`（thor 内，不改根目录）

根目录 `cost.py` / `evolution_*.py` 的 bts TODO **保持不动**。本目录提供：

| 文件 | 作用 |
|------|------|
| `cost_bts.py` | `compute_f_cost(..., cost_mode="bts")` → `bts_ops.compute_scheme_bts`；`depth` 只读导入根 `cost` |
| `smoke_cost_bts.py` | 无 GPU：all_low/mid/high + 单调性 smoke |
| `evolution_score_bts.py` | patch 根 `evolution_score.compute_f_cost` 后跑 NSGA-II |
| `evolution_kl_bts.py` | 同上，对 `evolution_kl` |
| `smoke_plan_he_reconcile.py` | plan placements vs HE `BootstrapHook` 日志对账 |

```bash
python3 HE/thor/smoke_cost_bts.py --task mrpc
python3 HE/thor/evolution_score_bts.py --tasks mrpc --cost-mode bts --pop 8 --gens 3 --skip-inference
python3 HE/thor/smoke_plan_he_reconcile.py --layers 0 1 2 --he-log /tmp/chain_l0_l1_l2.log
```

---

### Depth reconcile（bts_ops ↔ HE rem，mock bts）

目标：用真实 Liberate rem 消耗校准 `HE/thor/bts_ops.py` 的 `DEPTH_*`，使 DP 放置能指导 HE（暂用 decrypt→reencrypt 排除真 bts 噪声）。

```bash
python3 HE/thor/smoke_thor_repro.py --layer 0 --bootstrap plan_mock --probe-levels --rest-div 1
python3 HE/thor/audit_depth_reconcile.py --he-log /tmp/l0_plan_mock.log --layers 0
python3 HE/thor/smoke_thor_chain.py --layers 0 1 2 --bootstrap plan_mock --probe-levels
```

L0 plan_mock（poly Softmax/LN/Cheb GeLU，2026-08-17）**PASS**（err ~1e-5～2e-2）。已校准：

| 项 | 值 |
|----|----|
| plan `initial_rem` | **14**（enc@14 → entry rem=14；旧 THOR enc@20 → rem=8） |
| `DEPTH_MAKE_COPIES` | **3**；`DEPTH_ATT_CONTEXT` | **5**；`GELU_DEPTH_SLACK` | **1** |
| Softmax pathway | `HE_EXTRA=12`（逼出 Σy² DualRail 站点）+ inv hardcode fallback |
| 已知语义差 | DP mid → HE 入口 DualRail（REMAP）；LN 内 reactive bts（HE_REACTIVE） |

真 bts / ×s 缩放：等 DP 链式跑通后再做。


参数集（当前默认，对齐 THOR 官方 keys）：`logN=16`，`scale_bits=41`（Δ=2⁴¹），`num_levels=29`。

| 项 | 实测 |
|----|------|
| 真 `engine.bootstrap` 落点 | 恒为 **`level_calc=15`，rem=14**（与刷前 rem 无关） |
| 单次 bts 绝对误差（post−pre decrypt） | **≈5e-3～9e-3**；`ls≈1` |
| 与消息幅度 | abs **基本无关**；rel 随幅度增大而变小 |
| 与刷前 level | abs **基本无关** |
| 仅编码噪声 | ~1e-5～4e-5（低 lc）/ ~1e-10（高 lc） |
| 仅 `level_up`（链前段 lc&lt;15） | 一次 **~5e-4**；一次跳 span=1…8 差不多；连续 +1 会累加 |
| 仅 `level_up`（lc≥15，含 bts 落点后） | **~1e-9**，可忽略 |
| DualRail pack→bts→unpack scale | 与 THOR 一致（score 路径 +conj 无 ×½）；**不是 scale 写错** |
| L0 你的 poly + 真 bts | PASS（tol=5e-2），但 sftmx_in/ln1 贴边；mock 下误差小 2～3 个数量级 |

**换参试跑（自建 `create_bs_key` + sparse sk）→ 已放弃加大 Δ**

| scale_bits | bts 数值 |
|------------|----------|
| **41** | 可用，err≈7e-3，ls≈1 |
| **42+**（含 43/45/50/53 预设） | **崩坏**（~1e20） |

排查结论（2026-08-17）：
- 自建密钥 API 本身没问题（41 自建 ≡ 官方量级）。
- `bs` 模块里 **`K=28` / `R_SPARSE=3` 为固定常量**；算法主体 PyArmor，无法按 Δ 重标定。
- Liberate.FHE **已停止维护**（官方导向新 DESILO FHE）；本仓库继续用 THOR 绑定的 **scale_bits=41 + 官方/自建 keys**。
- **放弃**靠加大 `scale_bits` 降 bts 噪声；降噪改走 **DP 少刷**（及接受 ~1e-2 单次底噪）。

**Scale trick（amp=±0.3，scale_bits=41 自建，`audit_bootstrap_scale_tricks.py`）**
1. bts 后乘恢复系数 α：无效（`α_LS≈1`；网格最多 ~1.06×）。
2. bts 前 ×s、后再 /s：**有效**，abs 误差近似 ÷s（s=2→4→8→16→32 约 2×～30×+ 改善）。与「abs 噪声近似与幅度无关」一致。生产可用但须保证 ×s 后不溢出编码范围，且 bts 后有 rem 做 /s。

脚本：`audit_bootstrap_noise.py`、`audit_level_up_noise.py`、`audit_dualrail_bts_scale.py`、`audit_bootstrap_scale53.py`（仅作失败记录）。

### 自建密钥 / 换参（Liberate 能力）

库**支持**自生成密钥（不必死绑官方 `keys0`）：

| API | 作用 |
|-----|------|
| `engine.create_secret_key` / `create_public_key` / `create_evk` | sk / pk / evk |
| `create_galois_key` / `create_conjugation_key` / `create_rotation_key` | 旋转 / 共轭 |
| `bs.create_bs_key(engine, sk, n1=16, n2=4)` | **bootstrap 用 rotk 字典**（可替代加载 `rotk_dict/`） |
| `bs.create_cts_stc_const(engine)` | bts 前常数表（仍须调用） |

建引擎可改 `logN` / `scale_bits` / `num_special_primes` / `num_scales`；模数来自 `resources/` 的 scale primes 缓存（缺参会报 `NotFoundScalePrimes`）。  
Liberate 自带 `presets.params["bootstrapping"]`：`logN=16, scale_bits=53`（比 THOR 的 41 更大 Δ，可能降相对噪声——需实测）。  
注意：`ckks_bootstrapping.py` 本体 PyArmor，但上述 `create_bs_key` / `bootstrap` **可调用**；换参后 DualRail/THOR 路径与官方 keys 的数值对齐要重验，且 gen bs key 很重（显存/时间）。

---

## 下一步（库 OK → 再验逻辑）

| 顺序 | 内容 |
|------|------|
| ✅ | Liberate 原语 / DualRail / bootstrap / QKV PC-MM / **bts_ops@14** |
| ✅ | plan hook + L0→L2 chain；`cost_bts` / evolution_*_bts shim |
| ✅ | plan mid/exit 站点接线 + DualRail `boot_ct` 校正（reconcile: plan sites all wired） |
| ✅ | plan 强制 Softmax DualRail exit + Σy²-apply；discard span≤8；漏站改断言 |
| ✅ | L0 你的 poly + 真 bts PASS；bts/level_up 噪声专项 |
| 可选 | ~~换参 scale_bits=53 降 bts 噪~~ **已放弃**（≥42 自建 bts 崩；Liberate 停维） |
| 进行中 | L0 HE `--bootstrap plan` 数值回验；**DP 少 bts** |

---

## 与主线边界

- 可读：主线 core / slot_* / poly / cost / `bts_ops` / `OPENFHE_HANDOVER.md`
- 可写：仅 `HE/thor/**`；可选 `bts_ops` 复数几何扩展
- OpenFHE 主线计划 **保留、暂缓**
