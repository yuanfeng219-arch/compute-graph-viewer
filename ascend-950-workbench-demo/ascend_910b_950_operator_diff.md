# 910B/A3 与 950/A5 算子迁移差异说明

> 说明：本文按仓库代码和文档里的架构口径描述。`dav-2201` 覆盖 Atlas A2/A3 系列相关样例，文档和 CMake 中常映射到 `Ascend910B1` 或 `__NPU_ARCH__=2201`；`dav-3510` 对应 Ascend 950PR/950DT，映射到 `__NPU_ARCH__=3510`。实际板卡营销名以 `npu-smi info` 和项目 README 为准，判断算子兼容时优先看架构号、SoC 字符串和编译配置。

## 一句话结论

`CMAKE_ASC_ARCHITECTURES=dav-2201` 改成 `dav-3510` 只说明“用 3510 目标架构重新编译”。它不等于所有 2201 算子都能在 3510 上无改动正确运行，也不等于性能自动更好。

官方迁移指南明确说：每代芯片有独特微架构特性，不能保证所有接口只靠重编译就完全跨代兼容。迁移时要看三件事：

- 是否使用兼容 API。
- 是否依赖 2201 特定数据通路、buffer、矩阵分形、稀疏能力、Subnormal 语义。
- 是否使用 3510 新能力，例如 RegBase、SIMT、Histograms、MXFP4/8。

## 架构号和编译变量

在 `asc-devkit` 样例中：

| 目标 | 编译参数 | Device 宏 | 仓库映射 |
|---|---|---:|---|
| Atlas A2/A3 相关样例 | `--npu-arch=dav-2201` | `__NPU_ARCH__=2201` | `Ascend910B1` |
| Ascend 950PR/950DT | `--npu-arch=dav-3510` | `__NPU_ARCH__=3510` | `Ascend950PR_9599` |

所以 `CMAKE_ASC_ARCHITECTURES` 的实际作用是把 CMake 变量传给 ASC 编译器：

```cmake
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
```

结果是针对不同 NPU 架构生成不同 kernel。不是运行时自动识别后动态适配。

## API 兼容性的分层

官方文档把 Ascend C API 兼容性分成几层：

| API 层级 | 跨 2201/3510 的实际含义 |
|---|---|
| 高阶 API | 同领域内基本兼容，但特定扩展特性不保证兼容。3510 下也有“不支持卷积类高阶 API”等限制。 |
| 基础 API | 分兼容 API 和 ISASI/体系结构相关 API。后者不保证跨架构兼容，例如 `LoadData`、`Mmad` 等 Cube 侧接口。 |
| 语言扩展层 C API | SIMD/SIMT C API 也分兼容 API 和体系结构相关 API；当前 SIMD C API 文档中写明仍是体系结构相关。 |
| 编译器 BuiltIn API | 不保证兼容。 |

这解释了为什么很多简单样例能通过 `dav-2201`/`dav-3510` 重新编译，但一旦进入矩阵、搬运、稀疏、Reg、SIMT、低比特数据类型，就不能只看 CMake 变量。

## 950/3510 的关键新增能力

### 1. SIMD 跨代兼容能力

950 仍支持 SIMD 编程，并提供一批跨代兼容 API，用于帮助 A2/A3 算子平滑迁移。影响是：

- 逐元素加减乘、简单数据搬运、简单规约等没有特殊硬件假设的样例，往往可以通过重编译迁移。
- 但如果代码写死 UB 大小、bank 结构、核数、L0/L1 路径、矩阵分形，仍然需要改。
- Device 侧建议用 `__NPU_ARCH__` 隔离 2201/3510 分支，Host 侧用 `SocVersion` 隔离。

典型分支形式：

```cpp
#if __NPU_ARCH__ == 3510
// 950/3510 implementation
#elif __NPU_ARCH__ == 2201
// A2/A3/2201 implementation
#else
#error "Unsupported NPU architecture"
#endif
```

### 2. Reg/RegBase 编程

3510 的 AIV 核采用 RegBase 架构，而 2201 是 Membase 架构。核心区别是：

| 维度 | 2201/Membase | 3510/RegBase |
|---|---|---|
| 向量计算对象 | 更多围绕 LocalTensor/UB 上的数据组织计算 | 可以直接操作 Vector 寄存器/Register |
| 编程粒度 | 更偏 tensor/memory based | 更接近寄存器级数据搬运、计算、mask、同步 |
| 适用场景 | 常规 SIMD 算子 | 更细粒度控制、低延迟、小块数据、复杂向量流程 |
| 迁移影响 | 旧代码可能能编译，但不一定利用新架构 | 要获得 950 性能，常需要写 RegBase 专用 kernel 或 tiling |

`asc-devkit` 里的 RegBase Add 样例只声明支持 Ascend 950PR/950DT，并展示流程：

1. GM 数据搬到 Local Memory。
2. 从 UB/Local Memory 加载到 Reg 矢量计算寄存器。
3. 用 `asc_add` 在寄存器上计算。
4. 结果搬回 Local Memory，再搬出 GM。

所以看到 `RegTensor`、`MaskReg`、`AddrReg`、`LoadAlign`、`StoreAlign`、`LocalMemBar`、`RegBase` 这类关键词时，应默认它是 3510/950 方向的新实现，不要假设可直接回跑 2201。

### 3. SIMD + SIMT 混合编程

950/3510 支持 SIMT 和 SIMD/SIMT 混合编程。编译 SIMT 代码时不仅要指定 `dav-3510`，还要加：

```cmake
--enable-simt
```

这说明 SIMT 不是普通 SIMD kernel 的开关，而是新的编程/编译模式。适合场景通常是：

- 离散访存，例如 Gather、Scatter、索引选择。
- 线程粒度控制更自然的逻辑。
- Warp/ThreadBlock 风格的归约、直方图、原子操作。
- SIMT 处理不规则部分，SIMD/Cube 处理规则批量计算。

例子：

- `basic_gather` 用 SIMT 实现固定 shape 的 Gather，展示离散内存访问类算子。
- `histogram` 用 SIMT 的 `asc_atomic_add` 在 UB 中统计直方图，每个 Warp 维护局部直方图以降低冲突。

迁移含义：

- 2201 SIMD Gather 样例不等于 3510 SIMT Gather，只是功能类似。
- 如果 950 版本改用 SIMT，通常需要重写 kernel 切分、线程组织、同步和局部缓存布局。
- 如果源码或 CMake 写死 `--enable-simt`，基本可以视作 3510/950 专用路径。

### 4. MXFP4/8 与低比特 Matmul

950/3510 的 Matmul 高阶 API 增加了 MXFP4/8 等低比特矩阵能力。实际影响：

- DeepSeek-V4 文档中，950PR/DT 使用 Hybrid FP8-MXFP4、Hybrid MXFP8-MXFP4，属于新代际硬件能力。
- 950 上部分 FP8/MXFP 量化路径不是 2201 的普通 int8/int4 路径能直接替代的。
- 低比特 Matmul 可能涉及 `LoadData` 的 MicroScaling 扩展、数据格式变化、scale 布局和专用 tiling。

要特别注意：3510 的 Cube 计算单元反而不支持 2201 上的 `int4b_t` Cube 计算。官方迁移方案是先在 Vector Core 做 `int4b_t -> int8_t` Cast，再经 UB/L1 走 Mmad。因此“950 支持新低比特”不等价于“旧 int4 kernel 原样兼容”。

## 2201 到 3510 的硬件差异及影响

### 数据通路变化

3510 数据搬运变化很多，最容易影响旧 kernel 的正确性：

| 变化 | 对旧 2201 算子的影响 |
|---|---|
| 删除 L1 Buffer -> GM 数据通路 | 不能直接从 L1 搬到 GM，要改为通过 L0C/Fixpipe 等路径。 |
| 删除 GM -> L0A/L0B 数据通路 | 原来 `LoadData` 直接 GM 到 L0A/L0B 的写法要拆成 GM -> L1 -> L0。 |
| 新增 UB -> L1 | 950 上可减少 UB -> GM -> L1 的绕路。 |
| 新增 L0C -> UB | 某些矩阵后处理可避免 L0C -> GM -> UB 的绕路。 |
| `LoadData` 支持 MicroScaling | MXFP/MX 低比特路径可能使用新搬运语义。 |
| 新增 ND-DMA / loop mode / DN 分型 | 高维搬运和矩阵格式可以更灵活，但旧 stride/format 假设要重查。 |

判断经验：

- 只用 GM <-> UB 的简单向量算子，迁移风险较低。
- 用 GM/L1/L0A/L0B/L0C/Fixpipe 的矩阵或融合算子，迁移风险较高。

### 计算单元变化

| 变化 | 影响 |
|---|---|
| AIV 从 Membase 到 RegBase | 旧基础 API 的部分高维切分场景可能性能下降；新实现应考虑 RegBase。 |
| 3510 默认不支持 Subnormal | `Exp/Ln/Reciprocal/Sqrt/Rsqrt/Div` 等结果可能有精度差异，需要配置算法模式。 |
| Cube 不支持 `int4b_t` | 2201 的 int4 Matmul 不能原样搬到 3510；要 Cast 到 int8 或改低比特方案。 |
| 不支持 4:2 结构化稀疏 | `LoadDataWithSparse/MmadWithSparse` 路径不能在 3510 原样使用。 |
| L0A 分形变化 | L0A 切分场景需要重新计算地址和 tiling。 |

其中 L0A 分形变化很关键：

- 2201 矩阵乘中，A/B/C 分别是 `ZZ / ZN / NZ`。
- 3510 矩阵乘中，A/B/C 分别是 `NZ / ZN / NZ`。

非 L0A 切分场景可能兼容；L0A 切分场景需要按新分形重新适配。

### 存储和 bank 变化

3510 删除 L1 Buffer 边界值设定，不支持 `SetLoadDataBoundary`。如果旧 kernel 依赖 L1 边界绕回读取，需要手工拆成多条 Load 指令并自己处理绕回。

UB bank 结构也变了：

- 2201：16 个 bank group，每组 3 个 bank，每个 bank 4KB。
- 3510：8 个 bank group，每组 2 个 bank，每个 bank 16KB。

这意味着旧的“错开地址避免 bank 冲突”的经验不一定还能直接套用。能编译不代表访问模式高效。

### 同步和调试变化

3510 新增 Mutex 和新的核间同步控制能力。与此同时，部分调试接口也被移除，例如 `CheckLocalMemoryIA`。如果旧代码依赖这些调测能力或旧同步习惯，需要重新审查。

## 三类样例的迁移判断

### A. 简单 SIMD 样例

典型特征：

- CMake 中 `CMAKE_ASC_ARCHITECTURES` 写成 `dav-2201, dav-3510` 可选。
- kernel 主要是 GM <-> UB 搬运、逐元素计算、简单 reduce。
- 不写死特殊矩阵分形、L0A/L0B/L1 复杂路径。
- 不使用 BuiltIn API 或内部 `impl` 接口。

处理方式：

1. 用 `dav-2201` 编一版，在 2201 设备验证。
2. 用 `dav-3510` 编一版，在 3510 设备验证。
3. 对照精度和性能，尤其关注 Subnormal 和 UB bank 冲突。

这类样例“往往改 CMake 就能先跑起来”，但仍要测精度和性能。

### B. 用到 SIMT、RegBase、950 专用指令、MXFP4/8

典型特征：

- CMake 固定 `--npu-arch=dav-3510`。
- CMake 包含 `--enable-simt`。
- README 只写支持 Ascend 950PR/950DT。
- 代码出现 `RegBase`、`RegTensor`、`MaskReg`、`AddrReg`。
- 算子注册里有 `AddConfig("ascend950", regbaseCfg)`。
- 文档提到 Histograms、MXFP8、MXFP4、FP8-MXFP4。

处理方式：

- 默认按 3510/950 专用实现看待。
- 如果要兼容 2201，需要另写 2201 分支，而不是简单改编译参数。
- Host 侧算子注册要同时声明不同 SoC 的实现，例如 `ascend910b` 和 `ascend950` 分别配置。
- Tiling、workspace、scale 布局和量化格式要按代际拆开。

### C. 用到 A2/A3 特定 tiling、buffer、指令假设

典型特征：

- 矩阵计算使用 L0A/L0B/L0C、`LoadData`、`Mmad`、`Fixpipe`。
- 使用 `int4b_t` Cube Matmul。
- 使用 4:2 结构化稀疏。
- 使用 `SetLoadDataBoundary`。
- 代码假设 GM -> L0 或 L1 -> GM 通路存在。
- 写死 UB 大小、bank group、核数、分形布局。

处理方式：

- 不能只改 `CMAKE_ASC_ARCHITECTURES`。
- 先按迁移指南逐项检查数据通路和矩阵分形。
- 重写或拆分搬运路径，例如 GM -> L1 -> L0。
- int4 路径改为 Vector Cast 到 int8，或改用 950 的 MXFP/FP8 低比特方案。
- 重新调 tiling 和 bank 规避策略。

## 在仓库里如何快速判断一个算子属于哪类

### 看 CMake

```bash
rg -n "CMAKE_ASC_ARCHITECTURES|--npu-arch|--enable-simt" path/to/op
```

判断：

- 有 `dav-2201, dav-3510`：可能是跨架构样例。
- 固定 `dav-3510`：偏 950。
- 有 `--enable-simt`：SIMT，基本是 950/3510。

### 看 Device 代码

```bash
rg -n "__NPU_ARCH__|RegBase|RegTensor|MaskReg|LoadData|Mmad|Fixpipe|SetLoadDataBoundary|MmadWithSparse|LoadDataWithSparse" path/to/op
```

判断：

- 有 `__NPU_ARCH__`：已有代际隔离，继续看每个分支。
- 有 `LoadData/Mmad/Fixpipe`：重点查矩阵分形和数据通路。
- 有 `SetLoadDataBoundary`：3510 需要改。
- 有 `MmadWithSparse/LoadDataWithSparse`：3510 不支持 4:2 稀疏路径。
- 有 `RegTensor/MaskReg/RegBase`：偏 3510 RegBase。

### 看算子注册

在 `cann-recipes-infer-master/ops/ascendc/src/*/op_host/*_def.cpp` 中看：

```cpp
this->AICore().AddConfig("ascend910b");
this->AICore().AddConfig("ascend950", regbaseCfg);
```

判断：

- 同时注册 `ascend910b` 和 `ascend950`：同一个算子名有多代实现或多代编译配置。
- 只注册 `ascend950`：950 专用概率高。
- 只注册 `ascend910b`/`ascend910_93`：2201/A2/A3 路径，迁移到 950 要补注册和实现。

## 迁移 checklist

把 2201 算子迁到 3510 前，至少检查这些项：

- [ ] 是否使用公开 Ascend C API，而非内部 `impl` 或 BuiltIn API。
- [ ] CMake 是否能切到 `dav-3510`，SIMT 是否需要 `--enable-simt`。
- [ ] Device 侧是否需要 `__NPU_ARCH__ == 3510` 分支。
- [ ] Host/tiling 侧是否需要按 `SocVersion` 分支。
- [ ] 是否硬编码 UB/L1/L0 容量、bank 结构、核数。
- [ ] 是否使用 GM -> L0A/L0B 或 L1 -> GM 通路。
- [ ] 是否使用 L0A ZZ 分形或依赖 2201 的矩阵布局。
- [ ] 是否使用 `SetLoadDataBoundary`。
- [ ] 是否使用 `int4b_t` Cube Matmul。
- [ ] 是否使用 4:2 结构化稀疏。
- [ ] 是否涉及 Subnormal 敏感计算。
- [ ] 是否能用 3510 的 RegBase/SIMT/MXFP 能力重写热点。
- [ ] 是否重新跑精度、性能、边界 shape、长序列/大 batch 场景。

## 参考依据

- `asc-devkit-master/README.md:20-24`：950 支持 SIMD 跨代兼容、RegBase、SIMT/MIX、MXFP4/8 Matmul。
- `asc-devkit-master/examples/README.md:16-17`：`dav-3510` 对应 950PR/DT，`dav-2201` 对应 Atlas A2/A3。
- `asc-devkit-master/cmake/asc/asc_modules/CMakeASCInformation.cmake:111-115`：CPU/仿真模式下 `dav-2201 -> Ascend910B1`，`dav-3510 -> Ascend950PR_9599`。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/概述.md:3-24`：不能保证仅重编译完全兼容，建议用 `__NPU_ARCH__` 和 `SocVersion` 隔离。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/Ascend-C-API兼容策略.md:3-37`：API 兼容策略分层。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201到3510架构变更.md:3-70`：3510 架构新增能力和硬件变化。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/基础API迁移指导.md:141-148`：`SetLoadDataBoundary` 不支持。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/基础API迁移指导.md:244-248`：3510 Cube 不支持 `int4b_t`。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/基础API迁移指导.md:308-334`：L0A 分形变化。
- `asc-devkit-master/docs/guide/跨代迁移兼容性指南/3510架构迁移指导/2201迁移3510指导/基础API迁移指导.md:370-380`：3510 不支持 4:2 稀疏和 GM -> L0 通路。
- `asc-devkit-master/docs/guide/编程指南/编译与运行/算子编译/AI-Core算子编译基本用法.md:248-255`：SIMT 编译需要 `--enable-simt`。
- `cann-recipes-infer-master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md:4-6`：DeepSeek-V4 的 SAS、Compressor、LightningIndexer 同步支持 Atlas-A3 和 950PR/DT。
- `cann-recipes-infer-master/docs/models/deepseek-v4/deepseek_v4_ascendc_operator_guide.md:263-269`：A3 与 950 的 LightningIndexer Top-k 方案不同，950 使用 Histograms 指令。
