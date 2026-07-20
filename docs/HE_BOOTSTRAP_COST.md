# HE Bootstrapping Cost 工作简报（给新 agent）

本文是 **bootstrap 最优放置 / `cost_mode=bts`** 专用交接文档。项目总览见根目录 [`AGENTS.md`](../AGENTS.md)；Docker 环境见 [`.cursor/rules/docker-wsm.mdc`](../.cursor/rules/docker-wsm.mdc)。

---

## 本任务目标

在给定 **POLY_SCHEMES**（档位方案）后，用算法求出 **bootstrapping 放置位置**，使在当前噪声/模数/深度约束下 **bootstrapping 次数（bts）最少**。

现状：

| 组件 | 现状 |
|------|------|
| `cost.py` | 仅累加各 slot **乘法深度**（占位） |
| `evolution_*.py` 的 `cost_mode="bts"` | TODO，仍回退 depth |
| 完整 cost | 应为「最优 bootstrap 分配下的 bts 次数」（见 `cost.py` 文末 TODO） |

接入点（预留注释已存在）：

- `evolution_score.compute_f_cost(..., cost_mode="bts")`
- `evolution_kl.py` / `evolution_acc.py` 中同类 TODO：`optimize_bootstrap(task_name, scheme)`

**不要**在未实现求解器前静默把 bts 当成 depth；接口应显式区分。

---

## 本仓库在做什么（背景）

研究 **BERT-base @ GLUE（mrpc / rte / sst2）**，用 HE 友好多项式近似替换非线性：

- Softmax（attention）
- GeLU（FFN）
- LayerNorm（ln1 / ln2）

明文侧已有完整近似 + 验证集推理（`poly_model_inference.py`、`evolution_infer.py`）。敏感度 / NSGA-II / Spearman 等是**另一条**「搜档位」线；本任务专注 **HE 深度与 bootstrap 资源**，与敏感度搜索解耦，但最终会替换进化目标里的 `f_cost`。

环境：容器 `wsm_experiment`，工作区 `/workspace`，用 **`python3`**。

```bash
docker exec wsm_experiment bash -lc "cd /workspace && python3 <script>"
```

---

## 方案编码（读 cost 前必懂）

长度 **48** = 12 层 × 4 slot：

```
[layer0_softmax, layer0_ln1, layer0_gelu, layer0_ln2, layer1_softmax, ...]
```

| 值 | 含义 |
|----|------|
| 0 / 1 / 2 | low / mid / high 多项式档 |
| 3 | original（精确算子，**深度记 0**） |

索引：`cost.scheme_index(layer_idx, "softmax"|"ln1"|"gelu"|"ln2")`。

GeLU 有层组档位约束：`gelu_poly.gelu_level_allowed`。

> 注意：部分旧文档仍写「24 维 = softmax+gelu」；**以代码 `SCHEME_LEN=48` 为准**。`cost.py` 的 `__main__` 示例数组若仍为长度 24，属历史残留，实现 bts 时勿照抄。

---

## 当前深度模型（bts 的输入积木）

实现集中在 **`cost.py`**：

| 算子 | 深度来源 |
|------|----------|
| GeLU | `gelu_poly.gelu_config_for_layer` → `depth_he`（Chebyshev / PS-tree + 还原 +1） |
| Softmax | `5 + gs_sigma*2 + ceil(log2(exp_div))*gs_sum_sq*2`（Stockmeyer 4 + δ1 square 1 + Goldschmidt） |
| LayerNorm | `count_he_invsqrt_iters(...) + 3` |

API：`compute_scheme_cost(task, scheme)` → 深度和；`detailed=True` 得 per-slot 明细。

进化占位：`f_cost = ceil(total_depth / COST_DEPTH_DIVISOR)`，`COST_DEPTH_DIVISOR=5`。

**单次 bootstrap 深度预算**：`cost.BOOTSTRAP_DEPTH_BUDGET`，**默认 15**（变量，可调；求解器一律读该常量，勿写死）。

**关键建模点**：真实 CKKS 路径上，深度不是「各 slot 深度简单相加再除常数」；线性层、旋转、rescale、以及 **何时 bootstrap** 决定总 bts。本任务要在「每段乘法深度预算」约束下选放置点，使 bts 最少。

### CKKS 密文数量与深度对齐（bts 算法硬要求）

求解最优放置时必须显式建模，不能只盯 per-slot 深度和：

1. **密文数量变化**：除模型本身的数据维度外，工程上为计算引入的 **副本** 会造成密文数量膨胀（典型：多项式求值产生大量副本，再对齐相加）。
2. **密文相加 ⇒ 深度必须对齐**：参与加法的密文若剩余深度不一致，必须先对齐（额外深度消耗或额外 bootstrap）。这是放置决策的核心约束之一。
3. **Bootstrap 放置不限位置**：不拘泥于函数外 / slot 边界；任意计算点均可，只要总 bts 最优。须比较：
   - **提前 bootstrap**：预留足够深度，撑过后续膨胀与对齐相加；
   - **不提前**：在膨胀/对齐过程中再 bootstrap。

### THOR 路径与密文数量（`thirdparty/THOR-main`）

一层内大致顺序（`bert.py`）：

```
Attention: (实8↔复4) → QKV matmul → K transpose → Q make_copies(×16)
  → att_score → [打包 bootstrap×4] → Softmax → 128 CT × V(复2)
  → [bootstrap×2] → dense → residual → LayerNorm → 实8
FF: 实8→复打包+旋转(64) → dense1 → (2,8) → [打包 bootstrap×8]
  → GeLU → dense2 → residual → [打包 bootstrap×4] → LayerNorm → 实8
```

典型数量变化（THOR → 本仓库实数路径；详见 `bts_ops.py` 模块头注释）：

| 阶段 | THOR | 实数路径（`bts_ops`） |
|------|------|----------------------|
| 层间激活 | 8 实 / 4 复 | **8** |
| 旋转副本 | 4→64 | **8→128** |
| Softmax 主体 / 出口 | 8 → 128 | **8 → 128**（出口复制，非减半） |
| Att×V / context | 128×2复 → 2复 | **128 → 8** |
| FF dense1 / GeLU | **16** | **16** |
| 线性深度 | 见 THOR 注释 l→l+2 等 | `bts_ops` 常量表（QKV=2, dense=3,…） |

实验实现：`bts_ops.py`（**阶段 C 默认**）。档位仅 0/1/2。段深 > `BOOTSTRAP_DEPTH_BUDGET` 时由 `_consume_depth` 中途 bts。

**对齐（join）**：按两路实际 remaining，`rem ← min(spine, side)`（深侧 level_up）。旁路 fork 保存残差与 V；spine bootstrap 不刷新旁路。ct×ct（Q·K、Softmax·V）深度不足时按 **spine+side CT** 计 bts；Softmax·V / 残差可 `refresh_side` 刷旁路槽，Q·K 不刷 V（K≠V）。

**Softmax 深度拆分**：`cost.softmax_poly_depth` 只管多项式；通路另计 `attn_mask`（pt×ct=1）+ `copy`（mask+rotsum=1，对照 THOR `make_copies`）。

**阶段 D**：`run_phase_d_audit()` — 深度守恒、CT 链、fork/join、双侧 bootstrap 单元、low≤mid≤high 单调性。

| 阶段 | 内容 |
|------|------|
| A | 非线性整段粗事件 + fork/join（无通路 depth） |
| B | 粗微段 + fork/join（无通路 depth） |
| C | 微事件 + 通路 depth + fork/join（正式 bts） |

**线性复数打包（8→4）**：THOR 用虚部塞另一半实数据以减半 CT。本仓库 **后续只用实数线性**，不做该减半；因所有方案共用同一线性，相对 bts 比较本质不变，但绝对 CT 数按「实数路径」计。

### TODO：线性层本土化与规模可配置（暂缓）

当前 `bts_ops` 线性段（`CT_*`、`DEPTH_*`、旋转 8→128 等）为 **BERT-base + THOR 工程布局占位**，密文数量可能有问题：

- `CT_HIDDEN=8` ≠ 稠密打包下界（\(128\times768/32768\approx3\)，pad 头维约 4）；8 来自 THOR 对角 matmul 稀疏槽布局。
- `CT_FFN=16` 与 \(128\times3072\) pad 较接近，但仍绑定 THOR `(2,8)` 形态。

**后续工作（勿与当前非线性 DP 混做）**：

1. 精读 THOR 已发表论文 + `thirdparty/THOR-main` 线性编码（对角/旋转/副本）；
2. 生成本土化线性 cost/CT 模型：打包与副本规则参数化；
3. 支持 hidden / intermediate / seq_len / num_heads / slots 等自由变换，不写死 base；
4. 用新表替换 `bts_ops` 常量表并回归阶段 D。

**非线性是否也用复数？**

- GeLU 多项式主循环：**否**，对每个实 CT 独立 Stockmeyer。
- Softmax exp / σ / inv、LayerNorm 主循环：**否**（实 CT）；但 **出口复制、inv 中途 bootstrap、FF 在 GeLU 前后** 会短暂 `imult` 打包。
- 建模时：线性减半可忽略（按实数）；非线性里的打包/拆包与 128 膨胀仍要建模。

### GeLU：THOR 幂基 vs 本仓库 Chebyshev（必须区分）

| | THOR `nonlinear/gelu.py` | 本仓库 `gelu_poly` / `gelu_chebyshev` |
|--|--------------------------|--------------------------------------|
| 近似结构 | `he_tanh`：幂基复合 p1→p2（Stockmeyer），再与 x 相乘 | Chebyshev 复合 f1→f2（或单段），再 `x*(0.5+y)` |
| 求值算法 | baby `x,x²,x³` + giant `x⁴,x⁸,x¹⁶`，子式对齐相加 | HE：**Cheb PS-tree**（`depth_he≈⌈log₂ d⌉` 段累加 + 还原 +1）；明文 Clenshaw 仅验证，**深度 O(d) 不能当 HE cost** |
| 中间密文 | 幂次表 + 多个 baby 结果再 giant 组合 | `T_k` 树与系数组合，**副本集合 ≠ 幂基 baby-giant** |
| 对 bts 的影响 | 深度、对齐点、中途膨胀不同 | **以本仓库 `depth_he` 与 Cheb 事件图为准**；勿把 THOR 度数直接代入 |

同阶直觉：二者都是「对数深度的 baby/giant 类求值 + 复合两段 + 最后 ×x」，但基不同 ⇒ 乘法依赖与需对齐的副本集合不同，最优中途 bts 位置可能不同。

---

## 非线性细节去哪看（明文近似 ≈ THOR 参照）

THOR 在 `thirdparty/THOR-main`。本仓库明文近似与 THOR 非线性 **十分相近但不完全相同**——实现 bts / 对齐深度时：

1. 先读本仓库明文实现与配置；
2. 再对照 THOR 源码提取工程细节（迭代次数、缩放、多项式度数、乘法树形状、CT 膨胀）；
3. 差异处在文档/注释中显式记录，勿假设 1:1（尤其 GeLU 基、线性是否复打包）。

### Softmax（thor + Goldschmidt）

| 角色 | 路径 |
|------|------|
| 生产推理 | `softmax_poly.py`（`ThorSoftmaxEvaluator`、per-task `exp_div` / Goldschmidt 迭代表） |
| 离线误差 | `nolinear/eval_softmax_cuda.py`（GPU 批量 thor + Goldschmidt） |
| CPU 参考 | `nolinear/softmax.py` |

要点（AGENTS 约定）：

- 倒数是 **Goldschmidt**：`y=1-x; result=2-x; loop: y=y²; result*=(1+y)`  
- **不是** Newton（无 `D*r` 耦合项）
- 深度公式与 `cost.softmax_poly_depth` / eval 脚本中的 depth 打印应对齐

### GeLU（Chebyshev）

| 角色 | 路径 |
|------|------|
| 生产配置 + 推理 | `gelu_poly.py`（含 `depth_he`） |
| 拟合 / 方案导出 | `nolinear/gelu_chebyshev.py`、`nolinear/gelu_minmax.py` |

### LayerNorm（he_invsqrt，无 exp/log）

| 角色 | 路径 |
|------|------|
| 生产配置 + 推理 | `layernorm_poly.py` |
| 离线误差 | `nolinear/eval_layernorm_cuda.py` |
| 方差区间 | `nolinear/variance.py` → `nolinear/variance_out/{task}_variance.json` |

要点：

- **不要**用 `exp`/`log` 做 InvertSqrt 初值
- `min_var` → invsqrt 的 \(e_0\)；`max_var` → 缩放 \(M\)；二者勿混
- 取向 B：无 CKKS 编码 bookend（mask/2、out×2 已移除）

### 端到端明文替换推理

- `poly_model_inference.py`：按方案替换 BERT 非线性并跑验证集  
- `evolution_infer.py`：进化 / Pareto 汇报共用的 `SchemeEvaluator`（batch 评估）

---

## 建议实现顺序（bts）

1. **从 THOR 理清**：一层内线性↔非线性衔接、CT 数量轨迹、现有 bootstrap 点；预算用 `BOOTSTRAP_DEPTH_BUDGET`（默认 15）。线性按 **实数路径**（不采用 THOR 8→4 打包）。  
2. **序列化「深度消耗 + 密文数量事件」**：非线性用本仓库档位深度；GeLU 用 **Cheb PS-tree** 事件（非 THOR 幂基 Stockmeyer）；含副本膨胀与对齐相加。  
3. **实现** `optimize_bootstrap(...)`：放置点 **任意**；比较提前 vs 中途 bts，最小化总次数。  
4. **挂到** `cost.py` / `compute_f_cost(..., cost_mode="bts")`，保持 `depth` 模式行为不变。  
5. **小方案 smoke test**（全 high / 全 low / 一条 Pareto 解）对比「纯深度和」与「bts」数量级是否合理。

---

## 编码约定（本任务）

1. 最小 diff：优先扩展 `cost.py` + 薄封装，避免大改进化脚本。  
2. `cost_mode="bts"` 未就绪时保持显式 TODO / 清晰报错或回退说明，勿 silently 等同 depth。  
3. 非线性语义与 THOR 对齐时，以 THOR + 本仓库 eval 脚本为准；改深度公式须同步 `cost.py` 与对应 poly 配置注释。  
4. 仅在用户要求时 commit。

---

## 相关结果与搜索脚本（只读参考）

- `results/evolution_results/sensitive_scores_*/` — ΣS 搜索 Pareto（含 `total_depth`）  
- `results/evolution_kl_results/` — KL 搜索 Pareto  
- `evolution_score.py` / `evolution_kl.py` — `f_cost` 消费方  

新窗口接手后：先读本文 + `AGENTS.md`，再打开用户提供的 **THOR** 树，最后改 `cost.py`。
