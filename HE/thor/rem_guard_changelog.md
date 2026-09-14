# rem_guard L0 迭代记录（plan_mock）

原则：优先改 `bts_ops.py` 主路 DEPTH / 事件图；HE 仅在刷新语义与 DP 不一致时改。

## 历史（摘要）

| # | 症状 | 根因 | 处理 |
|---|------|------|------|
| 1 | Softmax aSOR DP×2 vs HE rem−1 | 并行 an∥bn 按串行计 | `SOFTMAX_ASOR_ITER_DEPTH=1` |
| 2 | `after_softmax` rem 差大 | APPLY/EXIT 过大 | `SUMSQ_APPLY=3`, `EXIT=2`, `INV_MASK=1` |
| 3 | apply rem→0 路径分叉 | DP 选 equal-cost fixed_bts | `atomic_min_leave=1` |
| 4 | exit 只刷 AUX | 主路 rem 对不上 | HE exit 刷 inv+exp；`EXIT_MIN_REM=5` |
| 5 | `after_att_context` HE10 vs DP2 | CONTEXT depth=10 | `DEPTH_ATT_CONTEXT=2` |
| 6 | LN1 inv√ 入口 rem 抬高 | smoke rem&lt;8 off-plan DualRail | plan_mock 禁止偷偷刷 |
| 7 | LN prep/inv√ Δ 不对 | prep=7、iter=2 | prep=3, `LN_INVSQRT_ITER=3`, POST=1 |
| 8 | 进程/会话被杀、OOM | Await/内存 | double-fork runner；线性权重分段释放 |

## 本轮（v20+）

| # | 版本 | 症状 | 根因 | 处理 |
|---|------|------|------|------|
| 9 | v20 | `after_bts_bridge_rot_ff1` HE=4 vs DP=3 | probe 在 mc_mult 前，却对 `after_event`（已扣 `DEPTH_FF_BRIDGE=1`） | rem_guard：去掉该 probe 映射；深度对齐改看 `after_ff_bridge_mask` |
| 10 | v21 | `after_bts_gelu_in` HE=14 vs DP=13 | placement `style=plain` → HE 明文刷落 budget；DP `dualrail_depth=1` 期望 land=13 | `bts_ops`：`dualrail_depth>0` 的 placement 标 `style=dualrail`，HE DualRail unpack×½ 对齐 land rem |
| 11 | v22 | `after_gelu` HE=3 vs DP=1 | `_gelu_bts_depth=max(depth_he, ENTRY_MIN=12)` 把段深抬成 12；HE Chebyshev Δ=10 | `bts_ops`：段深= `gelu_poly_depth`；`GELU_ENTRY_MIN_REM` 仅作 atomic 入口门槛 |
| 12 | v23 | `boot_before ln2_invsqrt_3` HE=4 vs DP=2 | `prep_s0` before-boot 后 DP 仍扣 3 步 prep；HE 只再烧 ~1 | `simulate_remaining_trace`：prep_s0 boot 后跳过 2 步 prep；planned mid 段首执行 |
| 13 | v24 | `after_ln2` HE=12 vs DP=13 | post before-boot 后 HE 烧 2；DP `DEPTH_LN_POST=1` | `simulate`：post before-boot 后 depth=2 |

## 结果

**v25：L0 `plan_mock` rem_guard + 明文对拍 PASS**（`after_ln2` rem_ok=12；sftmx/context/ln1/ln2 OK）。

## L0 all-mid（sm/ln/gelu=1）

| # | 版本 | 症状 | 说明 |
|---|------|------|------|
| 14 | v26_mid | 对拍 `ref_scheme` 崩 | 全 mid 触 GeLU 层组禁 mid（L9）；已改为 `ref_scheme=plan scheme` |
| 15 | v27_mid | **STOP** `after_ln1` HE rem=**4** vs DP **7** | Softmax→Attn dense 全 `rem_ok`；mid LN `iters=3`，DP post 后期望 7，HE 多烧 ~3。**按用户要求不符即停，未继续修。** |

## L0 all-mid

| 16 | v28_mid | (fix) after_ln1 HE4 vs DP7 | **根因**：prep 末 numerator 冻在 rem=4；mid denom 事后更高，HE `auto_ct_ct_mult` 输出 `min(num,denom)=4`；DP 只跟 denom 期望 7 | 见 #17 |
| 17 | v28_mid | 误把 Softmax `join_slot` 也 `min(spine,AUX)` → exit 14→9、LN1 入口 4、AUX 冻成 1、post→1（高档也回归） | Softmax join 只对齐旁路，**不夹主路**（`_transition_event`）；仅 LN `*_post_invsqrt` 的 HE ct×ct 输出才是 `min(num,denom)` | `simulate_remaining_trace`：只在 `*_post_invsqrt` 对 AUX 做 min；exit 恢复 14→12、dense→7、mid/high `ln1_post`→4 |

| 18 | v29_mid | rem 全 ok 但 ln2_out err=0.10>tol | mid LN2 3 轮 inv√ 近似精度 | （已撤销）proactive / prep skip 全 3 |
| 19 | v30_high | STOP ln1_invsqrt_3 HE10 vs DP8 | proactive 让 high 多出 plan 外 inv√_3 mid | **已删除** proactive；HE plan 模式禁止 pull-forward |

## L0→L1 chain all-high

| # | 版本 | 症状 | 根因 | 处理 |
|---|------|------|------|------|
| 20 | v33_chain | L0 **PASS**（34 rem_ok）；L1 STOP `after_bts_score` HE=6 vs DP=2 | L1 plan **无** `softmax_exp_stockmeyer` boot（rem=6 够跑 stockmeyer→2）；probe 在 stockmeyer **前**，却对照 post-stockmeyer `after_event` | `rem_guard`：`after_bts_score` 无 pre-softmax placement 时改对 `linear_att_score` |
| 21 | v33_chain | L1 `MaximumLevelError` @ attn_mask（rem 耗尽） | DP placement 在 `asor_sigma_i1`，replay 在 `i0` 需 mid boot；HE 只认 placement | `bts_ops._align_mid_placements_to_rem_trace`：把 σ-aSOR mid 从 `i1` 前拉到 `i0`（全层 L1–L11 同理） |
| 22 | v34_chain | L1 仍在 mask 饿死（σ-i0 boot 在 aSOR 入口，晚于 mask） | HE `thor_exp` 合 stockmeyer+δ1，mask 前 rem=1；DP boot 记在 mask 后 | ~~pre-mask σ-i0~~ → **改** `att_score rem≤6` 层补 `softmax_exp_stockmeyer` before boot（同 L0）； prune 冗余 σ-i0 mid |

## Route B（HE-driven depth profile）

| # | 版本 | 症状 | 根因 | 处理 |
|---|------|------|------|------|
| 23 | routeB_p2 | L0 plan+profile STOP `after_make_copies` HE=3 vs DP=4 | 连续 probe Δrem 把 K 侧 `q_rescale→make_copies` 算进段深（2≠3） | `he_depth_profile.SPINE_SEGMENT_PAIRS`：按 DP 事件边界 `(after_qkv→after_make_copies)` 量深；`build_depth_overrides` 从 probes 重算 spine 段深。**L0 plan_mock+profile safe PASS**（34 rem_ok） |
| 26 | ln_slot | 误用 THOR 方差模式分支 | `kind!=ln1` → mask/2、`rem_keep` 7 vs 10 按槽位分叉 | 删 mask/2 模式，输入缩放统一；post `rem_keep` 统一 |
