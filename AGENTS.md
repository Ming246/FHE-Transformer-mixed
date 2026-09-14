# AGENTS.md — WSM BERT-on-GLUE 多项式近似工具链

面向 AI coding agent 的项目说明。本仓库研究 **BERT-base 在 GLUE（mrpc / rte / sst2）上**，用 **HE 友好** 的多项式近似替换 **Softmax、GeLU、LayerNorm**，并通过敏感度、cost、ILP / NSGA-II 搜索档位方案，最后用推理脚本验证准确率。

---

## 环境

开发在 Docker 容器内进行（见 `.cursor/rules/docker-wsm.mdc`）：

```bash
docker exec wsm_experiment bash -lc "cd /workspace && python3 <script>"
```

- 工作区挂载：`/workspace`（宿主机 `F:\WSM`）
- 使用 **`python3`**（容器内无 `python`）
- 需要 GPU + 已下载的 `glue_datasets/`、`finetuned_weight/`
- 不要在 Windows 宿主机上假设 Python / CUDA 可用

---

## 目录结构（主路径）

| 路径 | 用途 |
|------|------|
| `0.Download_weight.py` / `1.Download_datasets.py` | 下载预训练/微调权重与 GLUE 数据 |
| `2.FineTuning_bert.py` | BERT 微调 |
| `3.Infer_Fintuned.py` | 原始微调模型推理 |
| `sensitive_score.py` / `_1` / `_2` | Taylor 型敏感度（阶数/形式不同）；句对任务须传 `token_type_ids` |
| `cost.py` | 方案 cost：深度和 + DualRail DP **bts**（`compute_f_cost` / `compute_scheme_bts`） |
| `docs/HE_BOOTSTRAP_COST.md` | **Bootstrap / bts cost** 交接简报（新 agent 优先读） |
| `ILP_loss_depth.py` | 固定 cost 预算下的 ILP 档位分配 |
| `evolution_score.py` | NSGA-II：\(\sum S\) vs **bts**（默认；`--cost-mode depth` 仍可用） |
| `evolution_kl.py` | NSGA-II：Output KL vs **bts**（默认） |
| `poly_model_inference.py` | 按 48 维方案替换 Softmax/LN/GeLU 并评估验证集/校验集 |
| `gelu_poly.py` / `softmax_poly.py` / `layernorm_poly.py` | 推理时多项式实现与 per-layer 配置 |
| `nolinear/` | 非线性近似 **离线评估** 与 Remez/Chebyshev 拟合 |
| `results/` | 敏感度 CSV、进化 Pareto 等 |

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
| 1 | mid |
| 2 | high |
| 3 | original（精确算子，cost/深度=0） |

索引：`scheme_index(layer_idx, "softmax"|"ln1"|"gelu"|"ln2")` → `layer*4 + slot`（见 `cost.py`）。

GeLU 档位有 **层组约束**（A/B/C，`gelu_poly.gelu_level_allowed`）；Softmax / LayerNorm 三档规则见各自 `*_poly.py`。

### Cost（当前为乘法深度占位）

- **GeLU**：来自 `gelu_poly` 的 `depth_he`（Chebyshev PS-tree + 还原 +1）
- **Softmax**（`cost.softmax_poly_depth` / aSOR）：
  - `SOFTMAX_DEPTH_BASE(=5) + asor_σ×SOFTMAX_ASOR_ITER_DEPTH + Σ_r asor_Σy²_r×SOFTMAX_ASOR_ITER_DEPTH`
  - `SOFTMAX_ASOR_ITER_DEPTH=1`（HE `he_asor_ct` DeltaCt 路径实测每轮 rem−1；明文 `kn*b*b_temp` 概念深度仍为 2）
  - BASE = Stockmeyer(deg15 密路径 4) + δ1 平方 1
  - Σy² 轮数 \(n_2=\lceil\log_2(\delta_2)\rceil\)（`LAYER_EXP_DIV`）；第 1/2 轮 iters 来自 `LAYER_ASOR_MAX_ITERS_SUM_SQ(_2)_BY_TASK`
  - **未用轮次**（\(\delta_2=2\) 时第 2 轮）表项为 **0**，且不计入深度
- **LayerNorm**：`invsqrt_max_iters + 3`（生产路径固定步数，无 α；`layernorm_poly` / `cost.py`）
- 进化默认 \(f_\mathrm{cost}=\) DualRail DP 总 bts（`HE/thor/bts_ops.optimize_bootstrap`，budget **14**，与 `plan_mock` 相同）
- `--cost-mode depth` 仍可用：\(\lceil\mathrm{depth}/\texttt{COST\_DEPTH\_DIVISOR}\rceil\)
- 简报见 [`docs/HE_BOOTSTRAP_COST.md`](docs/HE_BOOTSTRAP_COST.md)

### 非线性近似要点

**Softmax（thor + aSOR）**

- `softmax_poly.py`（生产）/ `nolinear/eval_softmax_cuda.py`（调参）：thor 多项式近似 exp，**aSOR** 求倒数（与 THOR `he_inv` 同形；固定 kn 更新，**不是** Newton，也已不再用 Goldschmidt）
- 生产：**固定** `max_iters`（无 α）；high/mid/low 由表区分；`e0_σ=min(σ)/2`，minσ 来自 `nolinear/sigma_out/{task}_sigma.json`
- Eval：α + max_iters 同时限制（便于调参）；深度打印用实际 iters
- Σy²：最多两轮（`δ2=4`）；第二轮配置在 `*_SUM_SQ2_*`，未用层填 0

**GeLU**

- 生产配置在 `gelu_poly.py`（映射 `gelu_chebyshev.py` 的 Chebyshev/复合方案）
- 拟合/搜索在 `nolinear/gelu_chebyshev.py`、`gelu_minmax.py`

**LayerNorm（HE 路径）**

- 生产：`layernorm_poly.py` — M 缩放 + `he_invsqrt` **固定 max_iters**（无 α；**无 exp/log**）
- 调参：`nolinear/layernorm.py` — 可保留 α；方差区间 `nolinear/variance_out/{task}_variance.json`
- 配置要点：`min_var` → invsqrt 的 \(e_0\)；`max_var` → 输入缩放 \(M\)（二者作用不同）

---

## 典型工作流

```
1. sensitive_score*.py    → results/sensitive_scores_*/{task}_sensitivity.csv
2. cost.py + ILP / evolution_*  → 搜索 48 维档位方案（cost 现为深度占位）
3. poly_model_inference.py → 验证集准确率 + 深度和
4. nolinear/eval_softmax_cuda.py / layernorm.py → 各非线性 HE 近似 vs 参考的逐层误差
5. nolinear 方差/σ 统计 → variance_out / sigma_out JSON
```

**进化搜索**

| 脚本 | \(f_1\) | \(f_2\) | 输出 |
|------|---------|---------|------|
| `evolution_score.py` | \(\sum S\)（校验集 CSV） | bts（默认）或 depth | `*_pareto_calib.csv` / `*_pareto_validation.csv`（同 f_cost 仅留最小 KL） |
| `evolution_kl.py` | Output KL（校验集搜索） | bts（默认）或 depth | `*_pareto_kl_calib.csv` / `*_pareto_kl_validation.csv` |

句对任务（MRPC/RTE）推理/敏感度须保留并传入 **`token_type_ids`**（勿在 `set_format` 时丢掉）。

---

## 常用命令

```bash
# 敏感度
docker exec wsm_experiment bash -lc "cd /workspace && python3 sensitive_score_1.py"

# ILP
docker exec wsm_experiment bash -lc "cd /workspace && python3 ILP_loss_depth.py"

# NSGA-II（ΣS）
docker exec wsm_experiment bash -lc "cd /workspace && python3 evolution_score.py --tasks mrpc"

# 多项式推理
docker exec wsm_experiment bash -lc "cd /workspace && python3 poly_model_inference.py"

# 非线性误差评估（在 nolinear/ 下，路径相对 ../glue_datasets）
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 eval_softmax_cuda.py"
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 layernorm.py"
```

---

## 编码约定（agent 修改时注意）

1. **最小 diff**：只改任务相关文件；eval 脚本顶部有独立配置块，优先改配置而非重构。
2. **路径**：根目录脚本用 `./glue_datasets/`；`nolinear/` 下脚本用 `../glue_datasets/`。
3. **不要**在 HE 近似路径使用 `exp`/`log` 作 InvertSqrt 初值（LayerNorm）；用 `he_invsqrt` + 配置区间。
4. Softmax 倒数是 **aSOR**（固定 kn），**不要**写成 Newton（无 `D*r` 耦合），也不要再按旧 Goldschmidt 循环描述生产路径。
5. **LayerNorm**：`min_var` 只影响 invsqrt 的 \(e_0\)；`max_var` 影响输入缩放 \(M\)。
6. Softmax `SUM_SQ2`：仅 \(\delta_2=4\) 的层非零；未用层保持 0，且 `cost` / 推理只按 \(n_2\) 取前几轮。
7. **commit**：仅用户明确要求时提交；不要改 git config。
8. **不要**未经请求生成大量 markdown 文档或过度抽象。

---

## `nolinear/` 子目录

| 文件 | 作用 |
|------|------|
| `eval_softmax_cuda.py` | GPU 批量 thor_softmax + aSOR（α+max_iters）；逐层误差 / σ 统计 |
| `layernorm.py` | HE LayerNorm vs `F.layer_norm`（调参可含 α） |
| `sigma_out/{task}_sigma.json` | Softmax 各层 min/max σ（供生产 e0） |
| `variance_out/` | 各层 ln1/ln2 输入 \(\mathrm{Var}(x)\) 的 min/max |
| `gelu_chebyshev.py` | GeLU Chebyshev/Remez 拟合与方案导出 |
| `gelu_minmax.py` | GeLU minimax（Remez，幂基） |
| `softmax.py` | Softmax / aSOR 相关 CPU 参考与工具 |

---

## 已知限制 / TODO

- `cost.py` `cost_mode=bts` 已接 DualRail DP（budget=14）；主线 `BOOTSTRAP_DEPTH_BUDGET=15` 仅深度占位遗留
- LayerNorm eval：深层误差依赖 `[min_var, max_var]` 是否与方差 JSON 一致
- ILP 假设 cost 可 per-position 累加；与真实 HE bootstrap cost 不一致时需换建模

---

## 结果文件

- `results/sensitive_scores_1/` — 校验集敏感度宽表
- `results/evolution_results/` — score-based Pareto（calib / validation）
- `results/evolution_kl_results/` — KL 搜索 Pareto（calib / validation）
- `results/score_spearman/` — ΣS vs Output KL 相关性
- `nolinear/variance_out/` — LayerNorm 方差 JSON
- `nolinear/sigma_out/` — Softmax σ JSON

修改搜索/评估逻辑后，用对应任务的 **小样本**（如 `--samples 32`）先 smoke test，再跑全量验证集。
