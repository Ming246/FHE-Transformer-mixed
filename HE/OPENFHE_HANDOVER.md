# THOR 纯实线性核心 → OpenFHE 交接包

> 把本文整份（或「§0 给下一任 AI 的系统提示」+「§1–§8」）交给另一台机器上的 AI。  
> 目标：平稳接手已完成工作，并在 **OpenFHE** 上落地同态线性 Encoder。

---

## §0 给下一任 AI 的系统提示（可直接粘贴）

```text
你是接手 THOR 纯实数 slot Encoder（线性 + HE 友好非线性）工作的工程师。
前序工作已在明文 NumPy 上完成「与 HE 同构」的全模型流水线。

【已完成、不要重做】
- 纯实数 slot 路径（默认不用 DualRail / 复数打包）
- PC-MM（Q/K/V/W_O/FC1/FC2）+ CC-MM（Score/Context）+ residual + 可堆叠 input_lower
- 形状自适应：bert_base / bert_large / bert_medium 等；默认 CC 后端 fig5 ≡ NumPy
- Softmax HE：thor 多项式 exp（Stockmeyer）+ Goldschmidt 倒数 + aSOR；slot_softmax_he.py
- GeLU HE：Chebyshev PS-tree（对齐 depth_he）；slot_gelu_he.py
- LayerNorm HE：M 缩放 + he_invsqrt（无 exp/log）；slot_layernorm_he.py
- MRPC 12 层全模型 e2e（流式加载 encoded 权重）vs 多项式 NumPy：hidden/logits ~1e-14 PASS
- 明确不要求：论文 Fig.5 原版 ttemp 在纯实上 ≡ NumPy

【你的任务】
1. 阅读 thor_encoder_linear_core.py + slot_{softmax,gelu,layernorm}_he.py
2. 用 OpenFHE（CKKS）同构替换：rotate / EvalMult / EvalAdd / plaintext mask
3. 先 tiny/smoke，再 bert_base 单算子 → 单层 → 多层；每步与明文 core decode 对拍
4. 不要引入 DualRail；不要把 backend=ttemp 当正确性路径
5. 非线性已在明文同构路径接好；密文阶段直接替换原语，勿改回 pass-through

【必读约束】
- 所有矩阵乘必须在 length=num_slots 的一维向量（密文）上完成
- 层间 layout 必须保持 input_lower，才能堆叠
- Score decode：scores[h, key, query]；权重缓存极大（见本文「编码体积」）
- fig5 几何 rrot 集合很大——先统计再 EvalAtIndexKeyGen
- 依赖：明文对照仅需 numpy；非线性配置来自仓库根目录 softmax_poly/gelu_poly/layernorm_poly

【成功标准】
- 明文：verify_e2e_full_mrpc.py 仍 PASS
- 密文：decode(密文) ≡ core 明文（约定容差）
- 文档：N / depth / scale / 旋转键 / 误差 / 耗时
```

---

## §1 迁移文件清单

### 必须带走（最小包）

| 文件 | 作用 |
|------|------|
| `thor_encoder_linear_core.py` | 纯实线性 Encoder + `run_encoder_layer_he_cached` 接线 |
| `slot_softmax_he.py` / `slot_gelu_he.py` / `slot_layernorm_he.py` | HE 友好非线性（明文同构） |
| `slot_decode.py` | decode / split_heads（验证用，非运行热路径） |
| `bts_ops.py` | Bootstrap 放置 DP；CT 几何来自 ThorConfig；非线性 depth 按任务/层/档拆解 |
| `OPENFHE_HANDOVER.md` | 本文 |

### 强烈建议带走

| 文件 | 作用 |
|------|------|
| `prepare_mrpc_cache.py` | MRPC embeddings + 权重 encode 缓存 |
| `verify_e2e_layer_mrpc.py` / `verify_e2e_full_mrpc.py` | 单层 / 12 层全模型对拍 |
| 仓库根 `softmax_poly.py` / `gelu_poly.py` / `layernorm_poly.py` | 多项式档位与 depth 配置 |

### 可选

| 路径 | 作用 |
|------|------|
| `cache/mrpc/` | 已编码权重（每层 `encoded.pkl` ~10.5GB；12 层约 126GB） |

### 不必迁

- 原 THOR CUDA / DualRail / `backend="ttemp"` 探索脚本

### 环境

- 明文对照：Python 3.10+，`numpy`（全模型对拍另需 `transformers` + 微调权重）
- OpenFHE：目标机安装；弱机可只跑明文

---

## §2 用户目标（已收敛）

1. **Slot 向量级**线性矩阵乘：encode → HE 原语运算 → decode → ≡ NumPy  
2. **纯实数**默认路径；不要 DualRail / 复数打包作为默认  
3. 原语仅：`rotate`、`mask(pt*ct)`、`mult`、`add`  
4. Encoder **可堆叠**：进出均为 `input_lower` layout  
5. Softmax / GELU / LayerNorm：明文同构 HE 近似 **已完成**；下一阶段换真实密文原语  
6. 形状：至少 `bert_base`、`bert_large`、`bert_medium`（**seq 仍为 128**）  
7. **自适应** BSGS；Fig.5 思想要有，**不要**追求论文原版 ttemp ≡ NumPy  
8. 下一阶段：**OpenFHE** 同态落地（线性 + 非线性一并替换原语）  

---

## §3 已完成工作（现状）

### 3.1 交付形态

| 模块 | 内容 |
|------|------|
| `thor_encoder_linear_core.py` | 线性 Encoder；`run_encoder_layer(..., softmax/gelu/ln_impl=)`；`run_encoder_layer_he_cached` |
| `slot_*_he.py` | Softmax / GeLU / LayerNorm 的 HE 友好明文实现 |
| `slot_decode.py` | decode / head 拆合（对拍用） |
| `prepare_mrpc_cache.py` | 标定 embeddings + 每层 `raw.npz` / `encoded.pkl` |
| `verify_e2e_layer_mrpc.py` / `verify_e2e_full_mrpc.py` | 单层 / **12 层全模型** 对拍 |

入口（全 HE 非线性）：

```python
x_enc = encode_input_lower_diagonals(x, cfg)
y = run_encoder_layer_he_cached(
    x_enc, encoded, cfg, masks,
    softmax_he_params=..., softmax_mask_packs=...,
    gelu_he_params=...,
    ln1_he_params=..., ln2_he_params=...,
    ln1_gamma_beta=..., ln2_gamma_beta=...,
    ln_eps=...,
)
```

权重：`raw` 为 PyTorch `(out,in)`；`encoded` 为 THOR plaintext 打包（见 §3.8）。

### 3.2 单层数据流（权威，HE 非线性）

```
X (seq, hidden)
  → encode_input_lower_diagonals → input_lower [n_input_packs]
  → PC-MM Q/K/V → q/k/v_merged [n_out_packed_qkv]
  → transpose(K) → k_lower
  → make_copies_real(Q) → q_copies [head_dim]
  → fig5 CC Score → score [n_input_packs]
  → slot_softmax_he（thor exp + Goldschmidt + aSOR）→ alpha
  → make_copies(alpha) → alpha_copies [seq_len]
  → fig5 CC Context → ctx [n_out_packed_qkv]
  → PC-MM W_O → wo_pc
  → residual → LN1 HE → attn_res [= input_lower]
  → FC1 → GeLU HE → FC2 → residual → LN2 HE → layer_out [= input_lower]
```

全模型：12× 上式 → decode → **明文** `tanh` pooler + classifier（尚未 HE 化）。
MRPC sample0 / level=high：`verify_e2e_full_mrpc.py` → hidden max\|Δ\|~3e-14，logits~1e-15，**PASS**。

### 3.3 配置几何（只固定 5 个数）

用户只设：`seq_len, hidden_dim, num_heads, ffn_dim, num_slots`。  
其余由 `ThorConfig` 属性推导：

| 量 | 公式 / 含义 |
|----|-------------|
| `head_dim` | `hidden / num_heads` |
| `head_slot_pack` | `num_heads` 向上取 2 幂（12→16） |
| `slot_stride` | `seq_len * head_slot_pack` |
| `pack` | `num_slots / slot_stride`（并行对角线条数） |
| `n_input_packs` | `seq_len / pack` |
| `n_out_packed_qkv` | `head_dim / pack` |
| `n_in_slot` | `seq_len`（纯实，不再 /2） |

Preset：

| 函数 | 要点 |
|------|------|
| `bert_base()` | 128/768/12/3072/32768 → pack=16, n_out=4 |
| `bert_large()` | 128/1024/16/4096 → pack=16, n_out=4 |
| `bert_medium()` | 128/512/8/2048 → pack=32, n_out=2 |
| `mini_bert()` | 小规模 smoke |
| `standard_bert()` | 4 头无 padding |

门控：`encoder_linear_applies(cfg)` = validate + FF `block_diag_2` 可用。  
`path_b_applies` 仅表示经典 pack=16/n_out=4；**不再**作为能否跑的硬条件。

### 3.4 Slot 索引

```text
_slot_index(cfg, subpack_j, token_t, head_d)
  = subpack_j * slot_stride + token_t * head_slot_pack + head_d
```

### 3.5 CC-MM 后端（重要！）

| `backend` / `cc_backend` | 含义 | 纯实 ≡ NumPy？ |
|--------------------------|------|----------------|
| **`fig5`（默认）** | 按对角 `n` 外环；每项用**几何** `(left_out, rrot)` rotate×right[n]，place 进输出 pack（`ttemp`） | **是** |
| `geometric` | 与 fig5 同一实现路径（别名） | **是** |
| `ttemp` | 固定 `rrot(n)=stride*j-pack*n` + `ct_ct` 四分片 + 自适应 ttemp 路由 | **否**（DualRail 耦合；仅对照） |

实现要点（`_slot_cc_bsgs_fig5`）：

- 编译期表：`_build_cc_score_scatter_pairs` / `_build_cc_context_scatter_pairs`
- 键：`(diag_n, left_pack, rrot)` → 各输出 pack 的 `(prod_slot, dest_idx)` 列表
- `MaskBank` 预计算并索引为 `cc_score_by_diag` / `cc_context_by_diag`
- **不要**用固定 `rrot(n)`：实测与几何 rrot 几乎全不一致（base ~8/8192，medium ~2/8192）
- 每个 `(n, left_out)` 常有**多个** rrot；必须按表循环，不能假设单 rrot

Transpose：

- `path_b_applies` → THOR 风格 mask/rotate（`_transpose_upper_to_lower_slot`）
- 否则（如 medium pack=32）→ 通用几何 scatter（`_transpose_upper_to_lower_scatter`）  
  （曾发现：pack=32 若误用 Path B 固定 mask，K_lower 会错）

### 3.6 已验证结果（明文，抽样）

在开发机上对 **fig5** 做过对拍（decode 用 `slot_decode`）：

| 配置 | Score vs `K@Q^T/√d` | Context（`apply_softmax=False`）vs 参考 |
|------|---------------------|----------------------------------------|
| bert_base | ~1e-15 | ~1e-14 |
| bert_medium | ~1e-15 | ~1e-14 |

注意 Score 布局历史上为 `scores[h, key, query]`；当前回归以整层 decode（`slot_decode.decode_input_from_lower_diagonals`）+ poly NumPy 为准。

`fig5` 与旧 flat scatter wrapper 数值差为 **0**（同运算不同外环顺序）。

回归脚本（均在 `HE/`）：

| 脚本 | 覆盖 |
|------|------|
| `verify_e2e_layer_mrpc.py` | 单层 HE 全通路 vs poly NumPy |
| `verify_e2e_full_mrpc.py` | **12 层流式** + 明文 head |

### 3.7 明确未做 / 不做

- 论文原版 ttemp 在纯实 ≡ NumPy（**不是目标**）
- **OpenFHE / 真实 CKKS**（下一阶段）
- 密文噪声、level、bootstrap、真实 HE cost
- Pooler / Classifier 的 HE 化（当前明文）
- 非 MRPC 任务的 encoded 缓存（逻辑通用，cache 未批量生成）

### 3.8 编码体积：为什么每层 `encoded.pkl` ~10.5GB？

**结论先说**：不是「只膨胀 128 倍就该几十 MB→几 GB」这么简单——确实 **每个权重标量在 Q/K/V 编码里重复写入 `seq_len=128` 次**，再乘以 **BSGS 所需的海量 plaintext 向量** 与 **`num_slots=32768` 的整槽存储（含 head padding）**，单层就会到约 **10.5GB**。

#### 原始权重 vs 编码后（MRPC / bert_base，一层）

| | 原始 `raw.npz` | 编码 `encoded.pkl` |
|--|----------------|---------------------|
| 内容 | 6 个 dense 矩阵 float64 | 同上，THOR 上对角 plaintext 打包 |
| 体积 | **~56.6 MB** | **~10.47 GB**（≈ **185×**） |

按矩阵拆开（每个 leaf = 一条 `float64[32768]` 的 slot 向量）：

| 键 | container shape | 向量条数 | 体积 | 相对 raw 膨胀 |
|----|-----------------|----------|------|----------------|
| `w_q/k/v` 各 | `(4, 6, 128)` | 3072 | 0.805 GB | **≈170.7×** = 128×(16/12) |
| `w_o` | `(8, 6, 128)` | 6144 | 1.61 GB | **≈341×**（向量数 ×2） |
| `w1` / `w2` 各 | `(2, 8, 6, 128)` | 12288 | 3.22 GB | **≈170.7×** |
| **合计** | | **~39936 条** | **10.47 GB** | **~185×** |

`32768 × 8 B = 256 KiB` / 向量；约 4 万条 → 10GB 量级，与文件大小一致。

#### 「重复 128 次」对不对？

对 **Q/K/V 类** `encode_weight_upper_diagonals`：用唯一整数填满 `W` 后统计，

- 几乎每个非零权重值在编码结果里出现 **恰好 128 次**（=`seq_len`）；
- 来源：编码内层对 `t in range(seq_len)` 把对角元写进 token 轴各位，以便与 `input_lower` 的 `[subpack][token][head]` 布局对齐，做 pt×ct 的 PC-MM；
- **不是**「整张矩阵只复制 128 份」——而是 **每个标量沿 token 轴铺 128 份**，同时还要为 BSGS 的每个 baby-step 准备独立 plaintext 向量（shape 最后一维 `n_in=128`）。

单条 slot 向量内 nnz≈75%：`head_slot_pack=16`、真实 `num_heads=12`，**1/4 槽位是 head padding 零**。  
故存储膨胀相对「逻辑 ×128」再乘 `16/12`：

\[
128 \times \frac{16}{12} \approx 170.67
\]

与上表 Q/K/V/FF 一致。`w_o` 的 `n_out_p` 为 `seq_len/pack=8`（QKV 为 `head_dim/pack=4`），向量数多一倍 → ≈341×。

#### 直觉核对（回答「128 倍也不该这么大？」）

- 若 **只** ×128：56.6MB × 128 ≈ **7.2GB**——已经接近 10GB；
- 再加上 head padding（×16/12）与 `w_o`/FF 的额外向量维，到 **~10.5GB/层** 完全合理；
- 12 层同时加载 ≈ **126GB**，故全模型验证必须 **按层流式** `load → run → del`（`verify_e2e_full_mrpc.py`）。

#### 与真实 HE 的关系

当前 cache 是 **明文 float64 的同构打包**，便于对拍。上 OpenFHE 后每条向量变为 CKKS ciphertext/plaintext，体积通常更大（多项式系数、多 level）；旋转密钥另计。工程上应：按层加载、考虑压缩/mmap、或延迟 encode。

---

## §4 关键设计决策与坑（必读）

1. **验证口径**：必须是 slot encode → slot 运算 → decode → NumPy，不是「矩阵级对角线公式 vs matmul」单独当过关。  
2. **纯实 vs 原 THOR**：原 THOR 用复数 / DualRail 打包；很多 bert.py 常数（固定 rrot、手写 ttemp）在纯实下**不能**当正确性实现。  
3. **QKV decode**：按 head 行块（`n_hidden_row_blocks`），不要按 `hidden/seq` 列块拆（早期大 bug）。  
4. **W_O**：`block_shape=(head_dim, seq_len)`，`n_in=n_in_slot`；context PC → `pc_ctx_vecs_to_input_lower_slots` 再 PC-MM。  
5. **make_copies_real(Q)**：CC Score 用；`make_copies(alpha)`：CC Context 用；不要混。  
6. **Score 输出向量数** = `n_input_packs`（不是 `n_out_packed_qkv`）。  
7. **FF**：`block_diag_2` + 双 half-band；`input_lower_to_ff_roots` / `ff_pc_output_to_input_lower_slots` 做 layout 桥。  
8. **`__all__` 曾误缩进进函数**；现已修到模块级。  
9. 几何 scatter 表构建对 base/medium **较慢**（分钟级）；`MaskBank` 应复用，不要每层重建。  

---

## §5 明文 API 速查

```python
from thor_encoder_linear_core import (
    bert_base, bert_medium, MaskBank,
    encode_input_lower_diagonals,
    run_encoder_layer, run_encoder_stack,
)

cfg = bert_base()          # 或 bert_medium() / bert_large()
masks = MaskBank(cfg)      # 编译期表，多层复用
x_in = encode_input_lower_diagonals(x, cfg)  # x: (seq, hidden)
y = run_encoder_layer(
    x_in, weights, cfg, masks,
    apply_softmax=True,
    cc_backend="fig5",     # 默认；正确性路径
)
# weights = dict(w_q=..., w_k=..., w_v=..., w_o=..., w1=..., w2=...)
```

原语（HE 映射目标）：

| NumPy core | 语义 |
|------------|------|
| `rotate_left` / `rotate_cc_left` | 循环左旋 |
| `pt_mult` | plaintext × ciphertext（或 mask） |
| `*` 于两 slot 向量 | ct × ct（逐 slot） |
| `add_vec` | 逐 slot 加 |
| `rotate_internal` | mask + 全局 rotate（PC-MM 块内） |

---

## §6 OpenFHE 落地建议（下一任任务分解）

### Phase 0 — 环境与对照

1. 安装 OpenFHE + Python 绑定；确认 CKKS Encrypt/EvalRotate/EvalMult/EvalAdd  
2. 拷贝最小包；跑通明文：`MaskBank` + `verify_e2e_layer_mrpc.py`  
3. decode 已在 `slot_decode.py`（勿再依赖已删除的 `linear_logic.py`）

### Phase 1 — 原语适配层

实现薄封装，例如：

```text
class HeContext:  # OpenFHE CKKS
    encrypt(vec) / decrypt(ct)
    rotate(ct, k)
    mul_pt(ct, pt) / mul_ct(ct, ct)
    add(ct, ct)
```

要求：同一 `num_slots`；先与 NumPy 原语对拍（随机向量）。

### Phase 2 — 自底向上替换

建议顺序（每步 decrypt 对拍 core）：

1. encode 权重为 CKKS plaintext；输入 encrypt（注意 §3.8 体积，按层加载）  
2. PC-MM → transpose + Score fig5 → Context fig5  
3. Softmax HE（已有 `slot_softmax_he` 算法；换 ct 原语）  
4. W_O + residual + LN1 HE  
5. FC1 → GeLU HE → FC2 → LN2 HE  
6. 整层 → 多层（depth / bootstrap）→ 再考虑 pooler/classifier  

### Phase 3 — 参数与工程

- 一层 multiplicative depth：线性 + Softmax/GeLU/LN 迭代（与 `cost.py` / `depth_he` 对齐）  
- 旋转键：fig5 几何 `rrot` 全集 + Softmax `rotsum` 等  
- 权重 cache：勿一次加载 12 层；评估 CKKS plaintext 压缩  
- 记录：N、scale、level、误差、墙钟时间  

### 非目标（暂缓）

- 复现 Liberate / 原 THOR CUDA 栈  
- 强制纯实 ttemp ≡ NumPy  
- 一上来就全 GLUE 精度  

---

## §7 对拍口径（防踩坑）

```text
# 整层 / 全模型（推荐）
y_he  = decode_input_from_lower_diagonals(layer_out_packs, cfg)
y_ref = _numpy_poly_layer_ref(...)   # verify_e2e_layer_mrpc
# 全模型另加明文 pooler+classifier，见 verify_e2e_full_mrpc

# 线性 smoke（prepare_mrpc_cache --smoke）
attn_res vs  x + context_ref @ W_o.T
```

容差：明文应 ~1e-12–1e-15；密文按 scale/噪声另定。

---

## §8 交接检查清单（执行用）

**迁移前**

- [ ] 复制 core + `slot_*_he.py` + `slot_decode.py` + 本文  
- [ ] 复制 `prepare_mrpc_cache.py` / `verify_e2e_*.py`（或至少读懂对拍口径）  
- [ ] 确认目标机：Python + numpy；OpenFHE 可装  
- [ ] 把 §0 提示词发给新 AI  

**接手理解**

- [ ] 读完 core + `slot_*_he` + `run_encoder_layer_he_cached`  
- [ ] 能口述 data flow 与 `input_lower` 不变式  
- [ ] 能区分 `fig5` vs `ttemp`，知道默认用谁  
- [ ] 知道 Score 输出是 `n_input_packs` 个向量  
- [ ] 理解 §3.8：权重 ×128 token 复制 + 32768 槽 → ~10GB/层  
- [ ] 跑通 `verify_e2e_layer_mrpc.py`（或读已有 PASS 日志）  

**OpenFHE 开工前**

- [ ] 明文回归：base 或 medium 的 Score/Context 仍 ≡ NumPy  
- [ ] 列出 fig5 所需 rotate 集合大小  
- [ ] 选定第一支 smoke 配置（建议 `mini_bert`）  
- [ ] 写原语对拍测试（rotate/mult/add）  

**阶段完成定义**

- [ ] Decrypt(PC-MM) ≡ core  
- [ ] Decrypt(Score) ≡ core  
- [ ] Decrypt(整层线性) ≡ core  
- [ ] 文档：参数表 + 误差表 + 已知限制  

---

## §9 给人类用户的短说明

- **线性核心**：`thor_encoder_linear_core.py`；**非线性**：`slot_{softmax,gelu,layernorm}_he.py`  
- **正确性后端**：`cc_backend="fig5"`；全模型回归：`verify_e2e_full_mrpc.py`  
- **编码很大**：每权重沿 token 轴 ×128，再整槽 32768 存储 → ~10.5GB/层（见 §3.8）  
- **下一阶段**：OpenFHE 换原语，不是再改 Fig.5 纯实 ttemp  

若与本文冲突：以源码与用户最新口头目标为准。
