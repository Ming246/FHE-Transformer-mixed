# AGENTS.md — WSM BERT-on-GLUE 多项式近似工具链

面向 AI coding agent 的项目说明。本仓库研究 **BERT-base 在 GLUE（mrpc / rte / sst2）上**，用 **HE 友好** 的多项式近似替换 **Softmax、GeLU**，并通过敏感度、cost、ILP / NSGA-II 搜索档位方案，最后用推理脚本验证准确率。

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
| `sensitive_score.py` | Taylor 型敏感度 \(S_l^{(k)}\)（三档误差 × 梯度） |
| `cost.py` | 方案 **乘法深度** 估算（占位）；目标为最优 bootstrap 下的 bts（见 `docs/HE_BOOTSTRAP_COST.md`） |
| `docs/HE_BOOTSTRAP_COST.md` | **Bootstrap / bts cost** 交接简报（新 agent 优先读） |
| `ILP_loss_depth.py` | 固定 cost 预算下的 ILP 档位分配 |
| `evolution_score.py` | NSGA-II：\(\sum S\) vs depth（有 cost 容差） |
| `evolution_acc.py` | NSGA-II：\(\|\Delta acc\|\) vs \(\lceil depth/10\rceil\)（严格） |
| `poly_model_inference.py` | 按 24 维方案替换 Softmax/GeLU 并评估验证集 |
| `gelu_poly.py` / `softmax_poly.py` | 推理时 GeLU / Softmax 多项式实现与 per-layer 配置 |
| `nolinear/` | 非线性近似 **离线评估** 与 Remez/Chebyshev 拟合 |
| `results/` | 敏感度 CSV、进化 Pareto、进化 acc 结果等 |

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
- **Softmax**：`5 + gs_sigma*2 + ceil(log2(exp_div)) * gs_sum_sq*2`（Stockmeyer 4 + δ1 square 1；`cost.py`）
- **LayerNorm**：`he_invsqrt` 迭代次数 + 3（`layernorm_poly` / `cost.py`）
- **TODO / 进行中**：完整 cost = 给定方案下 **最优 bootstrapping 放置** 的 bts 次数；`BOOTSTRAP_DEPTH_BUDGET` 默认 15；须建模密文数量膨胀与相加前深度对齐（提前 vs 中途 bts 择优）。简报见 [`docs/HE_BOOTSTRAP_COST.md`](docs/HE_BOOTSTRAP_COST.md)。明文非线性与外部 **THOR** HE 库相近但不完全相同，bts 对齐时两边对照。

### 非线性近似要点

**Softmax（thor + Goldschmidt）**

- `eval_softmax_cuda.py` / `softmax_poly.py`：thor 多项式近似 exp，**Goldschmidt** 求倒数（`y=1-x; result=2-x; loop: y=y²; result*=1+y`）
- **不是** Newton 倒数；每步迭代 ≠ 一次 Newton step

**GeLU**

- 生产配置在 `gelu_poly.py`（映射 `gelu_chebyshev.py` 的 Chebyshev/复合方案）
- 拟合/搜索在 `nolinear/gelu_chebyshev.py`、`gelu_minmax.py`

**LayerNorm（HE 路径，eval 用）**

- `nolinear/eval_layernorm_cuda.py`：M 缩放 + `he_invsqrt`（自适应，**无 exp/log**）
- **取向 B**：不含 CKKS 编码 bookend（ln2 的 mask/2、out×2 已移除）
- 配置 per 任务 × 层 × ln1/ln2：
  - `LAYER_VAR_RANGE_BY_TASK`：`[min_var, max_var]`（`min_var` 仅用于 `e₀=min/max`；`max_var` 定缩放 \(M\)）
  - `LAYER_INV_SQRT_ALPHA_BY_TASK`：`he_invsqrt` 停止阈值 \(\alpha\)（越小越准、迭代越多）
- 方差统计：`nolinear/variance.py` → `variance_out/{task}_variance.json`（仅 **var_min / var_max**）

---

## 典型工作流

```
1. sensitive_score.py     → results/sensitive_scores_*/{task}_sensitivity.csv
2. cost.py + ILP / evolution_*  → 搜索 48 维档位方案（cost 现为深度占位）
3. poly_model_inference.py → 验证集准确率 + 深度和
4. nolinear/eval_*_cuda.py → 各非线性 HE 近似 vs 参考实现的逐层误差
5. nolinear/variance.py     → 为 LayerNorm 填 [min_var, max_var]
```

**进化搜索**

| 脚本 | \(f_1\) | \(f_2\) | 输出 |
|------|---------|---------|------|
| `evolution_score.py` | \(\sum S\) | depth 和（±10 容差支配） | `results/evolution_results/*_pareto.csv` |
| `evolution_acc.py` | \(\|\text{acc}-\text{baseline}\|\) | \(\lceil depth/10\rceil\)（严格） | `results/evolution_acc_results/*_pareto_acc.csv` |

`evolution_acc.py` 需 `token_type_ids`（MRPC/RTE）；与 `poly_model_inference` 共用 `_load_tokenizer_and_model`。

---

## 常用命令

```bash
# 敏感度
docker exec wsm_experiment bash -lc "cd /workspace && python3 sensitive_score.py"

# ILP
docker exec wsm_experiment bash -lc "cd /workspace && python3 ILP_loss_depth.py"

# NSGA-II（acc）
docker exec wsm_experiment bash -lc "cd /workspace && python3 evolution_acc.py --task mrpc"

# 多项式推理
docker exec wsm_experiment bash -lc "cd /workspace && python3 poly_model_inference.py"

# 非线性误差评估（在 nolinear/ 下，路径相对 ../glue_datasets）
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 eval_softmax_cuda.py"
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 eval_layernorm_cuda.py"
docker exec wsm_experiment bash -lc "cd /workspace/nolinear && python3 variance.py"
```

---

## 编码约定（agent 修改时注意）

1. **最小 diff**：只改任务相关文件；eval 脚本顶部有独立配置块，优先改配置而非重构。
2. **路径**：根目录脚本用 `./glue_datasets/`；`nolinear/` 下脚本用 `../glue_datasets/`。
3. **不要**在 HE 近似路径使用 `exp`/`log` 作 InvertSqrt 初值（LayerNorm）；用 `he_invsqrt` + 配置区间。
4. **不要**把 Goldschmidt 倒数写成 Newton 形式（无 `D*r` 耦合项）。
5. **LayerNorm**：`min_var` 只影响 invsqrt 的 `e₀`；`max_var` 影响输入缩放 \(M\)；两者作用不同。
6. **commit**：仅用户明确要求时提交；不要改 git config。
7. **不要**未经请求生成大量 markdown 文档或过度抽象。

---

## `nolinear/` 子目录

| 文件 | 作用 |
|------|------|
| `eval_softmax_cuda.py` | GPU 批量 thor_softmax + Goldschmidt；逐层 max/mean 误差 |
| `eval_layernorm_cuda.py` | HE LayerNorm vs `F.layer_norm` |
| `variance.py` | 各层 ln1/ln2 输入 \(\mathrm{Var}(x)\) 的 min/max |
| `gelu_chebyshev.py` | GeLU Chebyshev/Remez 拟合与方案导出 |
| `gelu_minmax.py` | GeLU minimax（Remez，幂基） |
| `softmax.py` | CPU 参考实现（Goldschmidt 标量版） |

---

## 已知限制 / TODO

- `cost.py` / `evolution_*.py`：`cost_mode=bts` 仍为 TODO → 见 [`docs/HE_BOOTSTRAP_COST.md`](docs/HE_BOOTSTRAP_COST.md)
- LayerNorm eval：深层误差依赖 `[min_var, max_var]` 是否与 `variance.py` 统计一致
- ILP 假设 cost 可 per-position 累加；与真实 HE bootstrap cost 不一致时需换建模

---

## 结果文件

- `results/sensitive_scores_1/` — 敏感度宽表
- `results/evolution_results/` — score-based Pareto
- `results/evolution_acc_results/` — acc-delta Pareto
- `nolinear/variance_out/` — LayerNorm 方差 JSON

修改搜索/评估逻辑后，用对应任务的 **小样本**（如 `--samples 32`）先 smoke test，再跑全量验证集。
