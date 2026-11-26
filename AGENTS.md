# AGENTS.md — WSM BERT-on-GLUE 多项式近似工具链

面向 AI coding agent 的项目说明。本仓库研究 **BERT-base 在 GLUE 上**，用 **HE 友好** 的多项式近似替换 **Softmax、GeLU、LayerNorm**（分类头 tanh 另有多项式），用敏感度或 Output KL 对 **bts** 做 NSGA-II，再用明文推理或端到端密文推理核对。

明文代码在 `Plain/`，密文代码在 `HE/`。`glue_datasets/`、`finetuned_weight/`、`results/`、`nolinear/` 留在仓库根。运行时不读 `thirdparty/`。

---

## 环境

开发在 Docker 容器内进行（见 `.cursor/rules/docker-wsm.mdc`）：

```bash
docker exec wsm_experiment bash -lc "cd /workspace && python3 <script>"
```

- 工作区挂载：`/workspace`（宿主机 `F:\WSM`）
- 使用 **`python3`**（容器内无 `python`）
- 需要 GPU + 已下载的 `glue_datasets/`、`finetuned_weight/`
- 密文推理另需 `HE/liberate/`、`HE/keys/keys0/`、`HE/encoded_models/{task}/`
- 不要在 Windows 宿主机上假设 Python / CUDA 可用

`Plain/` 下的脚本用 `__file__` 的上一级作为仓库根，因此 `cd /workspace && python3 Plain/...` 与在 `Plain/` 内运行都能找到数据和 `HE/bts_ops.py`。`nolinear/` 里的评估脚本仍按在该目录下运行来写 `../glue_datasets/`。

`HE/`、`nolinear/`、`picture/` 要 `import cost` 或 `*_poly` 时，须把 `Plain/` 放进 `sys.path`（`HE/path_setup.ensure_thor_path` 已做这件事）。`Plain/cost.py` 用文件路径加载 `HE/bts_ops.py`，不要改成 `import HE.bts_ops`。

---

## 目录结构（主路径）

| 路径 | 用途 |
|------|------|
| `Plain/0.Download_weight.py` / `Plain/1.Download_datasets.py` | 下载预训练权重与 GLUE 到仓库根 |
| `Plain/2.FineTuning_bert.py` | BERT 微调，写入 `finetuned_weight/{task}/` |
| `Plain/3.Infer_Fintuned.py` | 原始微调模型推理 |
| `Plain/coverage_metric.py` | 从 train 抽校验集索引，落到 `results/coverage_metrics/` |
| `Plain/sensitive_score.py` | Taylor 型敏感度；句对任务须传 `token_type_ids` |
| `Plain/cost.py` | 方案 cost：乘法深度和，以及 DualRail DP **bts** |
| `Plain/evolution_score.py` | NSGA-II：\(\sum S\) vs **bts**（默认；`--cost-mode depth` 仍可用） |
| `Plain/evolution_kl.py` | NSGA-II：Output KL vs **bts**（默认） |
| `Plain/evolution_infer.py` | 进化搜索共用的明文评估 |
| `Plain/poly_model_inference.py` | 按 48 维方案替换 Softmax/LN/GeLU，pooler tanh 走 `tanh_poly`，评估验证集/校验集 |
| `Plain/gelu_poly.py` / `Plain/softmax_poly.py` / `Plain/layernorm_poly.py` / `Plain/tanh_poly.py` | 多项式实现与 per-layer 配置 |
| `Plain/score_spearman.py` | \(\sum S\) 与 Output KL 的相关性 |
| `HE/HE_model_inference.py` | 12 层 + 分类头的端到端密文推理（真实 bootstrap，rem 检查常开） |
| `HE/bts_ops.py` | 微事件图 + bootstrap 放置 DP（budget **14**） |
| `HE/encode_weights.py` | 把 `finetuned_weight/{task}/model.safetensors` 编成 `HE/encoded_models/` |
| `docs/HE_BOOTSTRAP_COST.md` | Bootstrap / bts cost 交接简报 |
| `nolinear/` | 非线性近似离线评估与 Chebyshev 拟合 |
| `picture/` | Pareto / low-high 对照图 |
| `results/` | 敏感度 CSV、进化 Pareto 等 |

`HE/` 内：`ckks.py`（引擎与 `HE/keys/keys0`）、`linear.py`、`data.py`（从 `glue_datasets/` 取验证集并做 8-real 打包）、`bert.py`（Attention / FFN / 分类头）、`nonlinear/`（固定用本仓库多项式，不挂官方非线性）、`bootstrap.py`、`rem_guard.py`、`model_encoder.py`。Liberate 源码在 `HE/liberate/src`。没有 `HE/thor/`。

---

## 核心概念

### 方案编码 `POLY_SCHEMES`

长度 **48** = 12 层 × 4（每层 **softmax / ln1 / gelu / ln2**）：

```
[layer0_softmax, layer0_ln1, layer0_gelu, layer0_ln2, layer1_softmax, ...]
```

| 值 | 含义 |
|----|------|
| 0 | low 多项式档 |
| 1 | high 多项式档 |
| 3 | original（精确算子，cost/深度=0） |

索引：`scheme_index(layer_idx, "softmax"|"ln1"|"gelu"|"ln2")` → `layer*4 + slot`（见 `Plain/cost.py`）。

GeLU high 按层组 A/B/C 共用复合方案（`gelu_poly._GELU_LAYER_SCHEME`）；low 为每层独立的 direct Chebyshev。Softmax / LayerNorm 只有 low/high。密文计划路径不接受 `3`。未传 `--scheme` 时，level 开关按层广播；传了 `--scheme` 则以 48 维数组为准。

### Cost

默认 \(f_\mathrm{cost}\) 是 DualRail DP 的总 bts（`HE/bts_ops.optimize_bootstrap`，budget **14**，phase C）：主路 placements 再加旁路 `align_bypass` / exp AUX 的实际 bootstrap 次数。放置只由主路 DP 决定。DP 会合取两侧 rem 的 min。

- **GeLU**：`gelu_he_depth` ← `depth_he`（Chebyshev PS-tree + 还原 +1）。high 为复合，low 为 direct。
- **Softmax**（`cost.softmax_he_depth` / aSOR；不含 pathway）：
  - `SOFTMAX_DEPTH_BASE(=5) + asor_σ×SOFTMAX_ASOR_ITER_DEPTH + Σ_r asor_Σy²_r×SOFTMAX_ASOR_ITER_DEPTH`
  - `SOFTMAX_ASOR_ITER_DEPTH=1`（HE `he_asor_ct` 每轮 rem−1）
  - BASE = Stockmeyer(deg15 密路径 4) + δ1 平方 1
  - Σy² 轮数 \(n_2=\lceil\log_2(\delta_2)\rceil\)；未用轮次（\(\delta_2=2\) 时第 2 轮）为 0，不计入深度
- **LayerNorm**：`layernorm_he_depth` = scale+var+enc_one + 2×iters + post（每轮 inv√ 深度 2）
- σ-aSOR：整段禁止中途刷 an/bn；只可在 `softmax_sigma_enc_one` 刷 σ，或在 `softmax_asor_sigma_inv` 刷算出的 1/σ。Σy² aSOR / LN inv√ 仍可在轮首 bts。
- `--cost-mode depth`：\(\lceil\mathrm{depth}/\texttt{COST\_DEPTH\_DIVISOR}\rceil\)（`COST_DEPTH_DIVISOR=1`）
- 刷后落地 rem = budget−1。`direct` 与 `real_pack` 都是这样；`rem_guard` 接受 HE rem 为计划落地值或满 budget。
- 简报见 [`docs/HE_BOOTSTRAP_COST.md`](docs/HE_BOOTSTRAP_COST.md)

### 非线性近似要点

**Softmax（多项式 exp + aSOR）**

- `Plain/softmax_poly.py`（生产）/ `nolinear/eval_softmax_cuda.py`（调参）：多项式近似 exp，**aSOR** 求倒数（固定 kn，不是 Newton，也不是 Goldschmidt）
- 生产：**固定** `max_iters`（无 α）；`e0_σ=min(σ)/2`，minσ 来自 `nolinear/sigma_out/{task}_sigma.json`
- Eval：α + max_iters 同时限制；深度打印用实际 iters
- Σy²：最多两轮（`δ2=4`）；第二轮配置在 `*_SUM_SQ2_*`，未用层填 0

**GeLU**

- 生产配置与密文乘法深度在 `Plain/gelu_poly.py`，系数 JSON 在 `nolinear/gelu/`
- 拟合脚本 `nolinear/gelu_chebyshev.py` 不进入生产 import

**LayerNorm**

- 生产：`Plain/layernorm_poly.py` — M 缩放 + `he_invsqrt` **固定 max_iters**（无 α；无 exp/log）
- 有效 token 方差统计与 hook 在 `Plain/layernorm_stats.py`。离线调参脚本 `nolinear/layernorm.py` 不进入生产 import；方差区间 `nolinear/variance_out/{task}_variance.json`
- `min_var` 只影响 invsqrt 的 \(e_0\)；`max_var` 影响输入缩放 \(M\)

**Pooler tanh**

- 明文与密文分类头都用 `Plain/tanh_poly.py`：\(p_2(p_1(z/C))\)，系数默认 `nolinear/tanh_out/tanh_fit_C5_d15_15.json`

---

## 典型工作流

```
1. Plain/coverage_metric.py     → results/coverage_metrics/ 校验集索引
2. Plain/sensitive_score.py   → results/sensitive_scores/{task}_sensitivity.csv
3. Plain/evolution_score.py 或 evolution_kl.py → 48 维方案（默认 f_cost = bts）
4. Plain/poly_model_inference.py → 明文验证集 / 校验集指标
5. HE/encode_weights.py    → HE/encoded_models/{task}/
6. HE/HE_model_inference.py       → 12 层 + 分类头密文 logits
7. nolinear/eval_softmax_cuda.py / layernorm.py → 逐层近似误差
```

**进化搜索**

| 脚本 | \(f_1\) | \(f_2\) | 输出 |
|------|---------|---------|------|
| `Plain/evolution_score.py` | \(\sum S\)（校验集 CSV） | bts（默认）或 depth | `results/evolution_results/*_pareto_calib.csv` 与 `*_pareto_validation.csv`（同 f_cost 仅留最小 KL） |
| `Plain/evolution_kl.py` | Output KL（校验集搜索） | bts（默认）或 depth | `results/evolution_kl_results/.../*_pareto_kl_calib.csv` 与 `*_pareto_kl_validation.csv` |

句对任务（MRPC/RTE）推理和敏感度须保留并传入 **`token_type_ids`**。

密文链固定 12 层加分类头。验证集样本来自 `glue_datasets/{task}` 的 `validation`，词嵌入来自 `finetuned_weight/{task}`。每层读 `HE/encoded_models/{task}/att_L{n}.pkl`，以及由该层 GeLU 档位决定的 `ff_L{n}_low.pkl` 或 `ff_L{n}_high.pkl`；分类头读 `pooler.pkl` 与 `cls.pkl`。

---

## 常用命令

```bash
# 敏感度
docker exec wsm_experiment bash -lc "cd /workspace && python3 Plain/sensitive_score.py"

# NSGA-II（ΣS）
docker exec wsm_experiment bash -lc "cd /workspace && python3 Plain/evolution_score.py --tasks mrpc"

# NSGA-II（Output KL）
docker exec wsm_experiment bash -lc "cd /workspace && python3 Plain/evolution_kl.py --tasks mrpc"

# 明文多项式推理
docker exec wsm_experiment bash -lc "cd /workspace && python3 Plain/poly_model_inference.py"

# 端到端密文推理（12 层 + 分类头）
docker exec wsm_experiment bash -lc "cd /workspace && python3 HE/HE_model_inference.py --dataset mrpc --sample 0"

# 非线性误差评估（在 nolinear/ 下）
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 eval_softmax_cuda.py"
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 layernorm.py"
```

---

## 编码约定（agent 修改时注意）

1. **最小 diff**：只改任务相关文件；eval 脚本顶部有独立配置块，优先改配置而非重构。
2. **路径**：`Plain/` 用仓库根（`Plain/` 的父目录）。`nolinear/` 下脚本用 `../glue_datasets/`。密文资产在 `HE/liberate`、`HE/keys`、`HE/encoded_models`。
3. **不要**在 HE 近似路径使用 `exp`/`log` 作 InvertSqrt 初值（LayerNorm）；用 `he_invsqrt` + 配置区间。
4. Softmax 倒数是 **aSOR**（固定 kn），不要写成 Newton，也不要再按 Goldschmidt 描述生产路径。
5. **LayerNorm**：`min_var` 只影响 invsqrt 的 \(e_0\)；`max_var` 影响输入缩放 \(M\)。
6. Softmax `SUM_SQ2`：仅 \(\delta_2=4\) 的层非零；未用层保持 0，且 cost / 推理只按 \(n_2\) 取前几轮。
7. **`bts_ops.py` 的计划数学不要随手改**（budget、会合取 min、σ-aSOR 禁止 an/bn 中途刷）。`cost.py` 只通过文件路径加载它。
8. 密文非线性固定走 `HE/nonlinear/` 里的本仓库多项式。不要加回官方非线性切换、mock、探针或断点续跑。
9. **commit**：仅用户明确要求时提交；不要改 git config。
10. **不要**未经请求生成大量 markdown 文档或过度抽象。

---

## `nolinear/` 子目录

| 文件 | 作用 |
|------|------|
| `eval_softmax_cuda.py` | GPU 批量多项式 softmax + aSOR（α+max_iters）；逐层误差 / σ 统计 |
| `layernorm.py` | HE LayerNorm vs `F.layer_norm`（调参可含 α） |
| `softmax.py` | Softmax / aSOR 的 CPU 参考 |
| `gelu_chebyshev.py` / `gelu_single.py` | GeLU Chebyshev 拟合与单点评估 |
| `pooler_tanh.py` / `eval_tanh.py` | pooler tanh 输入区间与拟合评估 |
| `sigma_out/{task}_sigma.json` | Softmax 各层 min/max σ（供生产 e0） |
| `variance_out/` | 各层 ln1/ln2 输入 \(\mathrm{Var}(x)\) 的 min/max |
| `gelu/` | GeLU 系数 JSON |
| `tanh_out/` | tanh 拟合系数与各任务输入区间 |

---

## 已知限制

- 进化默认 cost 已是 budget=14 的 DualRail DP bts。`cost_mode=depth` 只是深度和，不是 bootstrap 次数。
- LayerNorm 深层误差依赖 `[min_var, max_var]` 是否与方差 JSON 一致。
- 密文时间主要跟实际 ciphertext bootstrap 次数走；只把 Softmax 保持 high 时，相对全 high 的墙钟差距有限。

---

## 结果文件

- `results/coverage_metrics/` — 校验集索引
- `results/sensitive_scores/` — 校验集敏感度宽表
- `results/evolution_results/` — score-based Pareto（calib / validation）
- `results/evolution_kl_results/` — KL 搜索 Pareto（calib / validation）
- `results/score_spearman/` — \(\sum S\) vs Output KL 相关性
- `nolinear/variance_out/` — LayerNorm 方差 JSON
- `nolinear/sigma_out/` — Softmax σ JSON
- `nolinear/tanh_out/` — tanh 系数与输入区间

修改搜索或评估逻辑后，用对应任务的小样本（如 `--samples 32`）先 smoke test，再跑全量验证集。密文链不要为了核对路径去跑完整 12 层。
