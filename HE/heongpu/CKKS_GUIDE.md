# HEonGPU CKKS 方案指南

面向 WSM 项目 HE 路径的参考文档。基于 `/workspace/thirdparty/HEonGPU-main` 头文件、实现与官方示例整理。

---

## 1. 整体架构

```cpp
#include <heongpu/heongpu.hpp>
using Scheme = heongpu::Scheme::CKKS;

// 上下文是 shared_ptr
heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>(
    heongpu::sec_level_type::sec128);  // 默认 sec128
```

| 组件 | 类型 | 职责 |
|------|------|------|
| `HEContext<CKKS>` | `shared_ptr<HEContextImpl>` | 参数、NTT 表、GPU 上下文 |
| `HEEncoder` | 值类型 | 编解码 |
| `HEEncryptor` / `HEDecryptor` | 值类型 | 加解密 |
| `HEKeyGenerator` | 值类型 | 生成各类密钥 |
| `HEArithmeticOperator` | 值类型，**公有继承** `HEOperator` | 算术 + Regular/Slim bootstrap |
| `HELogicOperator` | 值类型，**私有继承** `HEOperator` | 逻辑门 + Bit/Gate bootstrap |
| `HEMultiPartyManager` | 值类型 | MPC / 分布式 bootstrap |

所有密文/明文数据默认在 **GPU** 上；可通过 `ExecutionOptions` 和 `store_in_host()` 控制 HOST↔DEVICE。

---

## 2. 参数体系

### 2.1 配置顺序（必须遵守）

```cpp
context->set_poly_modulus_degree(N);           // 1. 先设 n
context->set_coeff_modulus_bit_sizes(Q_bits, P_bits);  // 2. 再设模数链
context->generate();                           // 3. 生成后不可再改
```

`generate()` 之后参数冻结；`set_coeff_modulus` 后不能再改 `poly_modulus_degree`。

### 2.2 核心参数

| 参数 | API | 约束 / 说明 |
|------|-----|------------|
| 环维度 `n` | `set_poly_modulus_degree(n)` | 2 的幂，`[4096, 65536]`（`defines.h`） |
| Slot 数 | — | `n/2`（SIMD 打包容量） |
| 密文模数链 Q | `set_coeff_modulus_bit_sizes({...}, P)` 或 `set_coeff_modulus_values` | 每个素数 30–60 bit |
| 辅助模数 P | 同上 | 决定 keyswitch 算法 |
| 安全级别 | `GenHEContext(sec_level_type)` | `none` / `sec128` / `sec192` / `sec256` |
| Scale Δ | `encoder.encode(plain, msg, scale)` | `scale > 0` 且 `log2(scale) < total_coeff_bit_count` |

### 2.3 Keyswitch 算法（由 P 个数自动决定）

| P 素数个数 | `keyswitching_type` | 说明 |
|-----------|---------------------|------|
| 1 | `KEYSWITCHING_METHOD_I` | SEAL 风格 |
| ≥2 | `KEYSWITCHING_METHOD_II` | External product |

Relinkey / Galoiskey / Switchkey 的生成与运算路径与此一致，**不能混用**。

### 2.4 模数链与乘法深度

- `Q_size` = Q 中素数个数 = `context->get_ciphertext_modulus_count()`
- 每次 **rescale** 消耗一个 Q 素数 → `depth_++`
- 初始可用乘法深度 ≈ **`Q_size - 1`**
- 参数设计建议（来自 `2_basic_ckks.cpp`）：
  - 初始 scale 与中间 Q 素数 bit 数对齐（如 scale=2³⁰，Q=`{60,30,30,30}`）
  - 深度 D 的电路需要 D 次 rescale
  - 剩余最后一个 Q 素数应略大于 scale

### 2.5 内存池（可选）

```cpp
heongpu::MemoryPoolConfig pool;
pool.initial_device_fraction = 80.0f;
pool.max_device_fraction = 90.0f;
context->generate(pool_config);
```

Bootstrapping 需要大量 Galois 密钥，显存紧张时可调 `defines.h` 中的内存池比例，或将 Galois key 存 HOST。

---

## 3. 编解码 / 加解密

### 3.1 编码

```cpp
heongpu::HEEncoder<Scheme> encoder(context);

// SLOT 编码（默认，SIMD）
encoder.encode(plain, vector<double>, scale);
encoder.encode(plain, vector<Complex64>, scale);  // 复数 slot

// COEFFICIENT 编码（系数域，不能 decode 到 complex）
encoder.encode(plain, vector<double>, scale, {}, heongpu::encoding::COEFFICIENT);

// 标量
encoder.encode(plain, double_value, scale);
```

约束：

- SLOT：消息长度 ≤ `n/2`
- COEFFICIENT：消息长度 ≤ `n`
- 编码后 plaintext：`depth_=0`，`in_ntt_domain_=true`

### 3.2 加密 / 解密

```cpp
encryptor.encrypt(ciphertext, plaintext);   // plaintext.depth() 必须为 0
decryptor.decrypt(plaintext, ciphertext);   // 输出继承 ct 的 depth/scale/encoding
```

---

## 4. 密钥体系

```cpp
heongpu::HEKeyGenerator<Scheme> keygen(context);

keygen.generate_secret_key(secret_key);
keygen.generate_secret_key_v2(secret_key);   // v2 bootstrap 用

keygen.generate_public_key(public_key, secret_key);
keygen.generate_relin_key(relin_key, secret_key);

// Galois key 三种构造方式
Galoiskey gk(context);                          // 默认旋转范围
Galoiskey gk(context, vector<int> shifts);      // 指定 shift
Galoiskey gk(context, vector<int> key_index);   // bootstrap 用
keygen.generate_galois_key(gk, secret_key);

// 密钥切换（换 secret key 解密）
Switchkey swk(context);
keygen.generate_switch_key(swk, new_sk, old_sk);
```

| 密钥 | 用于 |
|------|------|
| Publickey | 加密 |
| Secretkey | 解密 |
| Relinkey | ct×ct 乘法后 3→2 多项式 |
| Galoiskey | 旋转 `rotate_rows`、共轭 `conjugate`、bootstrap |
| Switchkey | `keyswitch` 换密钥 |

稀疏密钥：`Secretkey(context, hamming_weight)`（bootstrap 示例常用 hw=16）。

---

## 5. 算术算子（`HEArithmeticOperator`）

通过 `HEArithmeticOperator` 调用，它公有继承 `HEOperator` 的全部算术 API。

### 5.1 密文-密文

| 操作 | API | 备注 |
|------|-----|------|
| 加 | `add` / `add_inplace` | 须同 depth、encoding、size |
| 减 | `sub` / `sub_inplace` | 同上 |
| 取负 | `negate` / `negate_inplace` | |
| 乘 | `multiply` / `multiply_inplace` | 输出 size=3，设 `rescale_required` + `relinearization_required` |

### 5.2 密文-明文

| 操作 | API | 备注 |
|------|-----|------|
| 加明文 | `add_plain` / `add_plain_inplace` | plain 须同 depth |
| 加常数 | `add_plain(ct, double, out)` | 常数自动乘 ct.scale() |
| 减明文 | `sub_plain` 系列 | |
| 乘明文 | `multiply_plain` / `multiply_plain_inplace` | 设 `rescale_required` |
| 乘常数 | `multiply_plain(ct, double, scale, out)` | |
| v2 复数常数 | `add_plain_v2` / `multiply_plain_v2` | 不预乘 scale |

### 5.3 后处理 / 层管理

| 操作 | API | 密钥 | 效果 |
|------|-----|------|------|
| 重线性化 | `relinearize_inplace(ct, relin_key)` | Relinkey | 3→2 多项式 |
| Rescale | `rescale_inplace(ct)` | 无 | `depth_++`，`scale /= q_last` |
| Mod drop | `mod_drop` / `mod_drop_inplace` | 无 | `depth_++`，**scale 不变** |
| 明文 mod drop | `mod_drop_inplace(plain)` | 无 | 对齐 ct 层 |

### 5.4 旋转 / 共轭 / 密钥切换

| 操作 | API | 密钥 |
|------|-----|------|
| 行旋转 | `rotate_rows` / `rotate_rows_inplace(ct, gk, shift)` | Galoiskey |
| Galois 变换 | `apply_galois(ct, out, gk, galois_elt)` | Galoiskey |
| 共轭 | `conjugate(ct, out, gk)` | Galoiskey（`c_data()`） |
| 密钥切换 | `keyswitch(ct, out, switch_key)` | Switchkey |

### 5.5 复数辅助

| 操作 | 说明 |
|------|------|
| `scale_up(ct, scale, out)` | 放大逻辑 scale，不 drop 模数 |
| `mult_i` / `div_i` | 乘/除虚数单位 i |

### 5.6 标准乘法流水线

```cpp
ops.multiply_inplace(C, C);          // 或 multiply(C1, C2, C)
ops.relinearize_inplace(C, relin_key); // 必须
ops.rescale_inplace(C);              // 必须
// 之后才能继续乘、旋转等
```

---

## 6. 深度与 Scale 状态机

### 6.1 密文状态字段

```cpp
ct.depth()                    // 已 rescale/mod_drop 次数
ct.level()                    // = Q_size - depth - 1，剩余可用层
ct.scale()                    // 当前逻辑 scale
ct.size()                     // 2（正常）或 3（乘法后未 relin）
ct.rescale_required()         // 待 rescale
ct.relinearization_required() // 待 relinearize
ct.encoding_type()            // SLOT 或 COEFFICIENT
```

### 6.2 rescale vs mod_drop

| | rescale | mod_drop |
|--|---------|----------|
| depth | +1 | +1 |
| scale | ÷ q_last | **不变** |
| 用途 | 乘法后标准降噪 | bootstrap 前降层（不改 scale） |

Bootstrap 准备典型做法：循环 `mod_drop_inplace` 直到 `depth() == Q_size - 2`（仅剩 1 个有效 Q 层）。

### 6.3 状态标志阻塞关系

| 标志为 true 时阻塞 | multiply, rotate, conjugate, keyswitch 等 |
|-------------------|---------------------------------------------|
| `rescale_required_` | 须先 rescale |
| `relinearization_required_` | 须先 relinearize |

### 6.4 注意

- **add/sub 不检查 scale 是否相等**，不同 scale 相加可能静默出错
- CKKS **没有** `noise_budget()` API，须手动跟踪 `depth()` / `level()`
- ct-plain 运算须 **depth 和 encoding 对齐**；跨层 plain-cipher 乘前须 `mod_drop_inplace(plain)`

---

## 7. Bootstrapping（四类）

### 7.1 类型对照

| 类型 | 枚举 | 算子类 | 消息 | 特殊约束 |
|------|------|--------|------|----------|
| Regular | `REGULAR_BOOTSTRAPPING` | `HEArithmeticOperator` | 复数 `Complex64` | 长 Q 链；ModRaise→CtoS→EvalMod→StoC |
| Slim | `SLIM_BOOTSTRAPPING` | `HEArithmeticOperator` | **实数** `double` | 先 StoC；消息须为实数 |
| Bit | `BIT_BOOTSTRAPPING` | `HELogicOperator` | 二进制 | 末模 **q_L = 2×scale** |
| Gate | `GATE_BOOTSTRAPPING` | `HELogicOperator` | 二进制 | 末模 **q_L = 3×scale** |

### 7.2 `BootstrappingConfig`

```cpp
BootstrappingConfig(CtoS_piece=3, StoC_piece=3, taylor_number=11, less_key_mode=false);
```

| 参数 | 范围 | 说明 |
|------|------|------|
| `CtoS_piece_` | [2, 5] | CoeffToSlot 分段 |
| `StoC_piece_` | [2, 5] | SlotToCoeff 分段 |
| `taylor_number_` | [6, 15] | EvalMod Taylor 项数 |
| `less_key_mode_` | bool | Galois 密钥少 30%，慢 15–20% |

### 7.3 Regular Bootstrap 标准流程

```cpp
heongpu::HEArithmeticOperator ops(context, encoder);

BootstrappingConfig boot_config(3, 3, 11, true);
ops.generate_bootstrapping_params(scale, boot_config,
    arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

auto key_index = ops.bootstrapping_key_indexs();
Galoiskey galois_key(context, key_index);
keygen.generate_galois_key(galois_key, secret_key);

// 降至 depth = Q_size - 2
for (int i = 0; i < Q_size - 2; ++i)
    ops.mod_drop_inplace(C);

auto booted = ops.regular_bootstrapping(C, galois_key, relin_key);
// 输出 depth_=0, scale 刷新
```

Regular bootstrap 示例参数：`n=4096`，Q 链 31 个素数（60 + 29×50），`scale=2^50`，`sec_level::none`。

### 7.4 v2 路径（非稀疏密钥优化）

- `generate_bootstrapping_params_v2(scale, BootstrappingConfigV2)`
- `regular_bootstrapping_v2(ct, gk, relin_key, swk_dense_to_sparse?, swk_sparse_to_dense?)`
- 输入须在**最大 depth**（`current_decomp_count == 1`）
- 可用 `set_coeff_modulus_values` 精细布置各段素数
- 参考：`thirdparty/HEonGPU-main/example/bootstrapping/5_ckks_regular_bootstrapping_v2.cpp`

### 7.5 暴露的 CtoS / StoC

`HEArithmeticOperator` 通过 `using` 暴露了 `coeff_to_slot` 和 `slot_to_coeff`，可用于调试（见 `7_ckks_coeff_to_slot_roundtrip.cpp`）。

---

## 8. 逻辑运算（`HELogicOperator`）

**私有继承** `HEOperator`：不能直接调用 `multiply` / `rescale` 等，须用逻辑门 API。

```cpp
HELogicOperator logic_ops(context, encoder, scale);

// 非 bootstrap 逻辑门（ct×ct 需 relin_key）
logic_ops.AND(ct1, ct2, out, relin_key);
logic_ops.OR / XOR / NAND / NOR / XNOR / NOT(...);

// 逻辑 bootstrap
logic_ops.generate_bootstrapping_params(scale, boot_config,
    logic_bootstrapping_type::BIT_BOOTSTRAPPING);
auto booted = logic_ops.bit_bootstrapping(ct, galois_key, relin_key);
auto gated  = logic_ops.AND_bootstrapping(ct1, ct2, galois_key, relin_key);
```

---

## 9. MPC（多方计算）

```cpp
HEMultiPartyManager<CKKS> mpc(context, encoder, scale);

// 联合公钥 / Relinkey / Galoiskey
mpc.generate_public_key_share(...);
mpc.assemble_public_key_share(...);

// 阈值解密
mpc.decrypt_partial(...);
mpc.decrypt(partial_cts, plain);

// 分布式 bootstrap
mpc.distributed_bootstrapping_participant(...);
mpc.distributed_bootstrapping_coordinator(...);
```

参考：

- `thirdparty/HEonGPU-main/example/mpc/2_multiparty_computation_ckks.cpp`
- `thirdparty/HEonGPU-main/example/mpc/4_mpc_collective_bootstrapping_ckks.cpp`

---

## 10. ExecutionOptions 与存储

```cpp
ExecutionOptions opts;
opts.set_stream(cudaStream)
    .set_storage_type(storage_type::DEVICE)  // 输出放哪
    .set_initial_location(true);             // 输入是否回原位置
```

对象方法：`store_in_device()` / `store_in_host()` / `is_on_device()`。

Galois 密钥可生成到 HOST 以省显存：

```cpp
keygen.generate_galois_key(gk, sk,
    ExecutionOptions().set_storage_type(storage_type::HOST));
```

---

## 11. 密钥需求速查

| 操作 | Relinkey | Galoiskey | Switchkey |
|------|----------|-----------|-----------|
| add/sub/negate | — | — | — |
| multiply (ct×ct) | ✓ | — | — |
| rescale / mod_drop | — | — | — |
| rotate / conjugate | — | ✓ | — |
| keyswitch | — | — | ✓ |
| regular/slim bootstrap | ✓ | ✓（大量） | v2 可选 |
| bit/gate bootstrap | ✓ | ✓ | — |
| 逻辑 AND/XOR (ct×ct) | ✓ | — | — |

---

## 12. 与 WSM 项目的关联

结合仓库里的 `HE/thor/`、`bts_ops.py`、`docs/HE_BOOTSTRAP_COST.md`：

| HEonGPU 概念 | WSM 侧对应 |
|-------------|-----------|
| `depth()` / `level()` | `cost.py` 乘法深度占位 → 未来 `cost_mode=bts` |
| `mod_drop` vs `rescale` | bootstrap 放置建模中「降层但不改 scale」 |
| Regular bootstrap | BERT 深电路刷新噪声的主路径 |
| `generate_bootstrapping_params` + Galois 密钥体积 | bootstrap cost / 显存规划 |
| Chebyshev 多项式求值（EvalMod 内部） | 与 `gelu_poly` Chebyshev PS-tree 深度建模可对照 |
| `encoding::SLOT` vs `COEFFICIENT` | THOR 实数密文打包策略 |
| `rotate_rows` / `keyswitch` | 线性层 matmul、attention 旋转 |

本目录下的 `src/smoke_test.cpp` 已覆盖最基础的 encode → encrypt → add/mul → relin → rescale 流水线。

---

## 13. 常见陷阱

1. **`HEContext` 是 `shared_ptr`**，用 `context->method()` 而非 `context.method()`
2. **乘法后必须 relin → rescale**，顺序不能跳
3. **mod_drop ≠ rescale**：bootstrap 准备用 mod_drop，乘法后用 rescale
4. **add 不校验 scale**：相加前须自行对齐
5. **Bit/Gate bootstrap 对末模有硬约束**（2Δ / 3Δ）
6. **Slim bootstrap 只支持实数**，不能用 `Complex64`
7. **`HELogicOperator` 私有继承**：算术运算用 `HEArithmeticOperator`
8. **Galois 密钥极占显存**：bootstrapping 前用 `bootstrapping_key_indexs()` 查看数量；可用 `less_key_mode` 或 HOST 存储
9. **加密要求 plaintext depth=0**
10. **多项式求值**（`evaluate_poly` 等）在 `HEOperator` 内部，非公有 API；自定义多项式电路需走 bootstrap 内部路径或扩展

---

## 14. 推荐示例索引

| 文件 | 内容 |
|------|------|
| `thirdparty/HEonGPU-main/example/basic/2_basic_ckks.cpp` | 基础算术：平方、ct×pt 乘、跨层 mod_drop |
| `thirdparty/HEonGPU-main/example/basic/5_switchkey_methods_ckks.cpp` | 旋转、keyswitch |
| `thirdparty/HEonGPU-main/example/basic/6_ckks_coefficient_encoding.cpp` | 系数域编码 |
| `thirdparty/HEonGPU-main/example/basic/7_ckks_coeff_to_slot_roundtrip.cpp` | CtoS/StoC 往返 |
| `thirdparty/HEonGPU-main/example/basic/12_basic_ckks_logic.cpp` | CKKS 逻辑门 |
| `thirdparty/HEonGPU-main/example/bootstrapping/1_ckks_regular_bootstrapping.cpp` | Regular bootstrap |
| `thirdparty/HEonGPU-main/example/bootstrapping/2_ckks_slim_bootstrapping.cpp` | Slim bootstrap |
| `thirdparty/HEonGPU-main/example/bootstrapping/3_ckks_bit_bootstrapping.cpp` | Bit bootstrap |
| `thirdparty/HEonGPU-main/example/bootstrapping/4_ckks_gate_bootstrapping.cpp` | Gate bootstrap |
| `thirdparty/HEonGPU-main/example/bootstrapping/5_ckks_regular_bootstrapping_v2.cpp` | v2 非稀疏密钥 |
| `thirdparty/HEonGPU-main/docs/bootstrapping.rst` | Bootstrap 原理文档 |

---

## 15. 最小工作模板

```cpp
constexpr auto Scheme = heongpu::Scheme::CKKS;
auto context = heongpu::GenHEContext<Scheme>();
context->set_poly_modulus_degree(8192);
context->set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
context->generate();

const double scale = std::pow(2.0, 30);

HEKeyGenerator<Scheme> keygen(context);
Secretkey<Scheme> sk(context);
Publickey<Scheme> pk(context);
Relinkey<Scheme> rk(context);
keygen.generate_secret_key(sk);
keygen.generate_public_key(pk, sk);
keygen.generate_relin_key(rk, sk);

HEEncoder<Scheme> encoder(context);
HEEncryptor<Scheme> encryptor(context, pk);
HEDecryptor<Scheme> decryptor(context, sk);
HEArithmeticOperator<Scheme> ops(context, encoder);

// encode → encrypt → compute → decrypt → decode
```

---

## 16. 关键源文件索引

| 路径 | 内容 |
|------|------|
| `thirdparty/HEonGPU-main/src/include/heongpu/heongpu.hpp` | 总头文件 |
| `thirdparty/HEonGPU-main/src/include/heongpu/util/schemes.h` | Scheme 枚举、keyswitch/boot 类型 |
| `thirdparty/HEonGPU-main/src/include/heongpu/util/util.cuh` | BootstrappingConfig, EvalModConfig |
| `thirdparty/HEonGPU-main/src/include/heongpu/util/storagemanager.cuh` | ExecutionOptions |
| `thirdparty/HEonGPU-main/src/include/heongpu/host/ckks/context.cuh` | HEContextImpl |
| `thirdparty/HEonGPU-main/src/include/heongpu/host/ckks/operator.cuh` | 全部算子 |
| `thirdparty/HEonGPU-main/src/include/heongpu/host/ckks/encoder.cuh` | 编解码 |
| `thirdparty/HEonGPU-main/src/include/heongpu/host/ckks/mpcmanager.cuh` | MPC |
| `thirdparty/HEonGPU-main/src/include/heongpu/host/ckks/chebyshev_interpolation.cuh` | Chebyshev 拟合 |
| `thirdparty/HEonGPU-main/src/include/heongpu/kernel/defines.h` | MAX_POLY_DEGREE、MAX_SHIFT 等 |

---

## 17. 本目录工程说明

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 下游 CMake 项目，链接已安装的 `HEonGPU::heongpu` |
| `src/smoke_test.cpp` | CKKS 编码/解码、同态加/乘 smoke test |
| `build/smoke_test` | 编译产物（`cmake --build build` 后生成） |

构建命令：

```bash
cd /workspace/HE/heongpu
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
./build/smoke_test
```
