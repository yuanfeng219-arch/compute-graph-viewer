https://mp.weixin.qq.com/s/xsirSciqrQ6F6LcdLEW3CQ
下面按“后续画硬件结构图”的口径整理。主材料是你上传的粘贴正文；我同时用公开 CANN/昇腾文档补了 AI Core、AIC/AIV、MTE、Buffer 归属。

## 0. 先给结论：这篇文章真正有用的结构信息

这篇文章的有效内容不是完整硬件手册，而是 **PTO 通信 ISA 如何把同一套 tile 级搬运语义映射到不同硬件通路**。它提到的核心对象可以分成四类：

1. **存储单元**：GM、UB/UBuffer、L1、L0A/L0B/L0C、BT Buffer、FP Buffer、CCU Memory Slice 等。

2. **计算核/计算单元**：AIC/Cube Core、AIV/Vector Core、Cube、Vector、Scalar。

3. **搬运/通信引擎**：MTE1/MTE2/MTE3、FixPipe、SDMA、URMA、CCU。

4. **PTO 指令/API 语义**：TPUT/TGET、TPUT_ASYNC/TGET_ASYNC、TNOTIFY/TWAIT/TTEST、TGATHER/TSCATTER/TBROADCAST/TREDUCE。它们不是硬件模块，是编译后映射到上述搬运引擎的指令抽象。


后续画图时，最容易混的是 **GM、UB、L1/L0、SDMA、URMA、CCU 的归属**。简化口径如下：

|对象|应画成什么|是否在 AI Core 内|Vector Core/AIV 归属|Cube Core/AIC 归属|
|---|---|--:|--:|--:|
|GM / Global Memory|全局设备内存，通常画在 AI Core 外，靠近 HBM/DDR/L2/内存控制器|否|AIV 可通过 MTE/SDMA/URMA 访问|AIC 可通过 MTE/FixPipe 访问|
|L2 Cache|全局/核外缓存层，介于 AI Core 与 GM 之间|否|共享|共享|
|UB / UBuffer / Unified Buffer|AIV 本地 SRAM / scratchpad|是|是，Vector 计算输入输出所在|一般否；特定新架构有 UB↔L1 / L0C→UB 通路|
|L1 Buffer|AIC/Cube 侧较大的本地中转 Buffer|是|分离架构下通常不属于 AIV|是|
|L0A/L0B|Cube 输入 Buffer|是|否|是|
|L0C|Cube 输出/累加 Buffer|是|否；新架构可有 L0C→UB 通路|是|
|BT Buffer|BiasTable Buffer|是|否|是|
|FP Buffer|FixPipe Buffer|是|否|是|
|Register / RegTensor / MaskReg|AIV 侧 SIMD/SIMT 寄存器层|是|是|否|
|Memory Slice / MS|CCU 片上缓存|否，属于 CCU|否|否|
|scratch tile|异步会话控制用的 UB 小块，不是数据货物|是|是|否|
|MTE1/2/3|AI Core 内搬运单元/通路|是|AIV 主要有 MTE2/MTE3|AIC 有 MTE1/MTE2/MTE3|
|FixPipe|AIC/Cube 输出后处理与搬运单元|是|否|是|
|SDMA|独立 DMA 引擎，画在 AI Core 外|否|AIV 只提交描述符/轮询状态|否|
|URMA|950 侧远程内存访问/RDMA 类引擎，画在 AI Core 外、靠近 UnifiedBus/互连|否|AIV 只发起/等待|否|
|CCU|集合通信协处理器，文章称位于 IO-Die|否|AIV 只握手|否|

官方 Ascend C 文档把 Core 定义为拥有独立 Scalar 的计算核，并区分 Cube Core 与 Vector Core：Cube Core 由 Scalar、矩阵计算单元、搬运单元等组成，不包含矢量计算单元；Vector Core 由 Scalar、矢量计算单元、搬运单元等组成，不包含矩阵计算单元。AIC 是分离模式下一组 Cube/Vector 组合中的 Cube Core，AIV 是其中的 Vector Core。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件实现-编程指南-Ascend C算子开发-算子开发-CANN商用版8.5.0开发文档-昇腾社区"))

---

## 1. 存储单元归属：哪些是“芯片上的存储单元”，哪些只是“通路”

### 1.1 GM：不要画进 AI Core

文章把 GM 说成“所有 AICore 共享、跨卡通信的起点和终点”，这对通信路径是对的；但画硬件结构时，建议把 GM 画在 **AI Core 外**。在 CANN 的分离架构说明中，AIC 架构和 AIV 架构都把 GM 标成“核外”，AIC 内部列 L1/L0A/L0B/L0C/BT/FP，AIV 内部列 UB。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC2.alpha002开发文档-昇腾社区"))

所以图上建议：

```text
NPU / Device
 ├─ Global Memory / HBM / DDR / GM
 ├─ L2 Cache / Memory Fabric
 ├─ AI Core Cluster
 │   ├─ AIC / Cube Core
 │   └─ AIV / Vector Core
 └─ Communication / IO subsystem: SDMA / URMA / CCU / interconnect
```

GM 是 **跨 AI Core 共享的全局地址空间**，不是 AICore 内部 SRAM。公开文档还说明，通过搬运单元读写 GM 的数据默认会缓存在 L2 Cache 中，所以 L2 也应该画在 AI Core 外、GM 与各 Core/搬运引擎之间。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0010.html "存储单元-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC3.alpha003开发文档-昇腾社区"))

### 1.2 UB / UBuffer：属于 Vector Core / AIV

UB 是向量计算和标量计算的输入输出空间。CANN 文档明确说 Vector 所有计算的源数据和目标数据都要求在 Unified Buffer 中，通常有 32B 对齐要求。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件实现-编程指南-Ascend C算子开发-算子开发-CANN商用版8.5.0开发文档-昇腾社区"))

文章里 TPUT 同步路径用 “UBuffer tile” 做中转，这个中转仓库应画在 **AIV/Vector Core 内**。文章也强调，异步路径里的 `scratch tile` 不是装数据的中转箱，而是 AIV 与 SDMA 交接队列尾指针、轮询标志等控制信息的小块 UB。

### 1.3 L1、L0A、L0B、L0C：属于 Cube Core / AIC

CANN 文档对 Cube 数据访问说得很清楚：L0A 存左矩阵，L0B 存右矩阵，L0C 存矩阵乘结果和中间结果。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件实现-编程指南-Ascend C算子开发-算子开发-CANN商用版8.5.0开发文档-昇腾社区"))

分离架构下，AIC 包含 GM（核外）、L1、L0A、L0B、L0C、BiasTable Buffer、Fixpipe Buffer；AIV 包含 GM（核外）和 UB。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC2.alpha002开发文档-昇腾社区"))

因此画图时：

```text
AIC / Cube Core
 ├─ Scalar
 ├─ Cube
 ├─ MTE1 / MTE2 / MTE3
 ├─ L1
 ├─ L0A
 ├─ L0B
 ├─ L0C
 ├─ BT Buffer
 ├─ FP Buffer
 └─ FixPipe
```

```text
AIV / Vector Core
 ├─ Scalar
 ├─ Vector
 ├─ MTE2 / MTE3
 ├─ UB / UBuffer
 └─ SIMD/SIMT Registers, if drawing newer architecture
```

### 1.4 BT Buffer、FP Buffer、FixPipe：Cube/AIC 侧

BT Buffer 是 BiasTable Buffer，FP Buffer 是 Fixpipe Buffer，官方存储单元表将它们列为 AI Core 内部存储；分离架构图把它们放在 AIC 侧。FixPipe 负责 L0C 到 GM/L1 的输出搬运，并可做随路格式/类型转换。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0010.html "存储单元-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC3.alpha003开发文档-昇腾社区"))

### 1.5 351x / 新架构的特殊点

CANN 9.0 的 NPU_ARCH 351x 文档说明，AI Core 分为 AIC 和 AIV 两个独立核，AIC:AIV 配比为 1:2；该架构新增 L0C→UB、UB↔L1 数据通路，删除 GM→L0A/L0B 直达通路，删除 L1→GM 通路，并增加 SSBuffer 用于 AIC/AIV 核间通信。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00065.html "NPU架构版本351x-架构规格-硬件实现-编程指南-Ascend C算子开发-编程指南-CANN商用版9.0.0开发文档-昇腾社区"))

这意味着后续如果画的是 **351x 或更近的新架构**，不要直接沿用老图里的 GM→L0A/L0B 箭头；应该改成：

```text
GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM/L1
```

并补上：

```text
L0C → UB
UB ↔ L1
SSBuffer: AIC/AIV 核间通信/同步辅助
```

---

## 2. 通路和搬运引擎：哪些是“路”，哪些是“车”

### 2.1 MTE 是 AI Core 内部搬运单元，不是跨卡专用 DMA

官方 Ascend C 存储单元表给出 MTE 的典型职责：

|搬运单元|典型通路|
|---|---|
|MTE1|L1→L0A/L0B；耦合架构下 L1→UB；分离架构下 L1→BT Buffer|
|MTE2|GM→L1 / L0A/B / UB，具体支持随架构版本变化|
|MTE3|UB→GM|
|FixPipe|L0C→GM/L1，且可做随路格式/类型转换|

这些是 AI Core 内部的数据搬运硬通道，和 SDMA/URMA 这种 AI Core 外的专用通信/DMA 引擎不同。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0010.html "存储单元-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC3.alpha003开发文档-昇腾社区"))

文章中 TPUT/TGET 同步路径实际是把跨卡 GM→GM 语义拆成两段：

```text
本地 GM → 本地 AIV 的 UB staging tile → 远端 GM
```

文章说明这样做的原因是 MTE 读写通路分离：MTE2 负责把 GM 搬进 UB，MTE3 负责从 UB 搬回 GM；TPUT 把切块、双缓冲、AtomicAdd 等细节隐藏在 tile 语义下。

### 2.2 SDMA 是 AI Core 外的异步 GM↔GM 直传引擎

文章把 SDMA 定义为独立于 AICore 的专用 DMA 引擎。TPUT_ASYNC/TGET_ASYNC 由 AIV 发起，但数据本体不经过 UB，而是走：

```text
本地 GM → SDMA → 远端 GM
```

或反向：

```text
远端 GM → SDMA → 本地 GM
```

AIV 使用小块 UB scratch 填写传输描述符、更新 Queue Tail、轮询完成标志；scratch tile 只是调度剪贴板，不是数据 payload。

### 2.3 URMA 是 950 侧的异步远程内存访问路径

文章说 Ascend 950 新增 URMA，即 UB Remote Memory Access，基于 UnifiedBus 的硬件用户态 RDMA；对 PTO 使用者仍然是 build session、submit、wait/test 的异步接口，底层可在 SDMA 和 URMA 之间编译期或平台参数分派。

公开 Huawei roadmap 只明确说 Ascend 950 系列会有更高互连带宽，950 芯片总互连带宽达到 2 TB/s，并支持 SIMD+SIMT 等变化；URMA 的具体微结构、端口和与 GM/L2 的精确关系，我没有在公开官方低层手册里确认到。([huawei](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech "Groundbreaking SuperPoD Interconnect: Leading a New Paradigm for AI Infrastructure - Huawei"))

绘图建议：把 URMA 画在 **通信/互连子系统**，不要画进 AICore：

```text
GM / L2 / Memory Fabric ↔ URMA / UnifiedBus RDMA Engine ↔ Remote NPU GM
```

### 2.4 CCU 是 950 侧集合通信协处理器，不在 AICore 内

文章和 CANN 社区技术文都把 CCU 描述为 Ascend 950 引入的集合通信处理器，用于卸载 AllReduce 等集合通信，避免 AI CPU/AI Vector 等计算核资源和访存带宽被通信抢占。技术文进一步说 CCU 可接收调度器下发的集合通信任务，完成跨 NPU 同步、地址交换、数据传输和 Reduce 运算。([cann.csdn.net](https://cann.csdn.net/69d4b15854b52172bc67789d.html "集合通信处理器（CCU）技术解读文档_昇腾_昇腾CANN-CANN开发者社区"))

CCU 应画在 AI Core 外，靠近 IO-Die / interconnect：

```text
CCU
 ├─ Microcode / Control
 ├─ Trans / DMA-like transfer
 ├─ Reduce Engine: Sum / Max / Min
 ├─ Memory Slice / on-chip cache
 └─ Links to local GM and remote GM / interconnect
```

文章说 PTO 与 CCU 对接时，Host 侧编译/注册/Launch 产生 CcuDeviceSession；Device 侧由 AIV 拿 session 与 CCU 握手，AIV 不参与真实搬运和归约，重活由 CCU 完成。

---

## 3. AICore、Vector Core、Cube Core 的归属关系

### 3.1 耦合模式 vs 分离模式

CANN 文档把 AI Core 工作模式分成两类：

|模式|结构|产品口径|
|---|---|---|
|耦合模式|Cube、Vector 共用一个 Scalar，部署在一个 AI Core 上|旧 Atlas 推理/训练系列、部分 A2 推理产品|
|分离模式|Cube Core 和 Vector Core 各有独立 Scalar，按一定比例组成一个 AI Core|Atlas A2/A3 训练/推理系列|

官方文档说明分离模式下，AI Core 核数以 Cube Core 为准，Cube Core 和 Vector Core 按 1:N 组合；Atlas A2/A3 训练/推理系列属于分离模式。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件实现-编程指南-Ascend C算子开发-算子开发-CANN商用版8.5.0开发文档-昇腾社区"))

### 3.2 画分离模式时的推荐结构

```text
AI Core group / split-mode logical AI Core
 ├─ AIC / Cube Core
 │   ├─ Scalar
 │   ├─ Cube
 │   ├─ MTE1 / MTE2 / MTE3
 │   ├─ L1
 │   ├─ L0A / L0B / L0C
 │   ├─ BT Buffer / FP Buffer
 │   └─ FixPipe
 ├─ AIV0 / Vector Core
 │   ├─ Scalar
 │   ├─ Vector
 │   ├─ MTE2 / MTE3
 │   ├─ UB
 │   └─ SIMD/SIMT Register layer
 └─ AIV1 / Vector Core, if this product/arch is 1:2
```

注意：文中 GEMM_AR 示例写“910B 的架构天然支持这种并行——24 个 AIC 和 24 个 AIV 物理独立”，但这不一定等价于全芯片硬件总量。CANN 9.0 对 351x 架构明确写 AIC:AIV=1:2；一篇 2025 年关于 Ascend 910B 的论文也描述为一个 AI Core 包含一个 AIC 和通常两个 AIV，并指出 910B 中 AIC/AIV 数据交换通常要经 GM/L2。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00065.html "NPU架构版本351x-架构规格-硬件实现-编程指南-Ascend C算子开发-编程指南-CANN商用版9.0.0开发文档-昇腾社区")) ([arXiv](https://arxiv.org/html/2505.15112v1 "Parallel Scan on Ascend AI Accelerators"))

所以后续画图建议不要把“24 AIC + 24 AIV”画成通用 910B 物理定论。更稳妥的标注是：

```text
AIC/AIV 数量与配比依产品和 NPU_ARCH 而定。
GEMM_AR 示例中使用 24 个 AIC 与 24 个 AIV 做计算-通信流水线调度。
若按 351x 文档画，则 AIC:AIV = 1:2。
```

---

## 4. 不同芯片/场景下的数据搬移路径

### 4.1 单核 Vector 计算路径

适用：AIV / Vector Core 内做 elementwise、reduce、gather 等向量操作。

```text
GM / L2 → MTE2 → UB → Vector → UB → MTE3 → GM / L2
```

这个路径属于 **AIV 内部计算路径**。CANN 文档也把分离架构的 Vector 典型计算数据流写成 `GM-UB-[Vector]-UB-GM`。([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC2.alpha002开发文档-昇腾社区"))

### 4.2 单核 Cube/GEMM 计算路径

适用：AIC / Cube Core 内做矩阵乘、卷积类主计算。

旧/通用分离架构可按：

```text
GM / L2 → MTE2 → L1 → MTE1 → L0A/L0B → Cube → L0C → FixPipe → GM / L1
```

CANN 文档把 Cube 典型计算数据流写成：

```text
GM-L1-L0A/L0B-Cube-L0C-FixPipe-GM
GM-L1-L0A/L0B-Cube-L0C-FixPipe-L1
```

([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC2.alpha002开发文档-昇腾社区"))

若画 351x，需要补充：

```text
L0C → UB
UB ↔ L1
```

并删除旧图里的：

```text
GM → L0A
GM → L0B
L1 → GM
```

因为 351x 文档明确列出这些新增/删除通路。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00065.html "NPU架构版本351x-架构规格-硬件实现-编程指南-Ascend C算子开发-编程指南-CANN商用版9.0.0开发文档-昇腾社区"))

### 4.3 910/950 全系列：PTO 同步点对点搬运，TPUT/TGET → MTE

适用条件：tile 粒度、小块、2D strided、不规则形状、低启动延迟、需要精细控制；代价是 AIV 同步等待。

**TPUT：本地写远端**

```text
Local GM → Local AIV UB staging tile → Remote GM
```

**TGET：从远端读本地**

```text
Remote GM → Local AIV UB staging tile → Local GM
```

文章明确把 TPUT/TGET 的同步路径归为 MTE，并说它是 910 到 950 全系列的保底路径；TPUT 还支持自动分批、Ping-Pong 双缓冲、远端 AtomicAdd。

画图时可写：

```text
AIV issues TPUT/TGET
MTE2: GM → UB
MTE3 / remote write path: UB → GM
AIV waits until finished
```

这里“UB→远端 GM”不是普通本地 `UB→GM` 的简单箭头，建议标成：

```text
UB → MTE/BIU/interconnect remote write → Remote GM
```

因为跨卡地址、远端原子等细节文章没有展开。

### 4.4 910/950 全系列：PTO 异步点对点搬运，TPUT_ASYNC/TGET_ASYNC → SDMA

适用条件：大块连续数据、希望通信与 AIV 后续工作重叠；代价是 session/descriptor 启动开销较高，不适合极小数据。

**TPUT_ASYNC：**

```text
AIV writes descriptor in UB scratch
Local GM → SDMA → Remote GM
AIV continues
AIV wait/test AsyncEvent later
```

**TGET_ASYNC：**

```text
AIV writes descriptor in UB scratch
Remote GM → SDMA → Local GM
AIV continues
AIV wait/test AsyncEvent later
```

文章强调数据不经过 UB 中转，scratch tile 只是 AIV 与 SDMA 之间的控制剪贴板。

### 4.5 Ascend 950：PTO 异步点对点搬运，TPUT_ASYNC/TGET_ASYNC → URMA

适用条件：950 平台、目标路径选择 URMA、高带宽远程内存访问、仍希望保持同一套异步接口。

逻辑路径：

```text
AIV writes descriptor/session control
Local GM → URMA / UnifiedBus remote memory path → Remote GM
AIV wait/test, or uses Quiet semantics
```

反向 TGET_ASYNC：

```text
Remote GM → URMA / UnifiedBus remote memory path → Local GM
```

PTO 层面仍是 build session、submit、wait/test。文章说 TGET_ASYNC 在 910/950 可走 SDMA，950 上可选择 URMA；不走的分支由 `if constexpr` 在编译期消掉。

### 4.6 910 系列：集合通信由 AIV 拼点对点搬运

适用：AllReduce、ReduceScatter、Broadcast 等集合通信，但没有 CCU 硬件卸载。

文章给出的跨代兼容表是：

```text
Ascend 910 系列：
  同步搬运：MTE
  异步搬运：SDMA
  集合通信：AIV 自己搬
```

典型 ReduceScatter 可画成：

```text
AIC computes GEMM tile
AIC writes result to GM
AIV sees tile ready
AIV uses TPUT AtomicAdd
Local GM tile → UB staging → owner rank GM with remote AtomicAdd
TNOTIFY/TWAIT form barrier
```

文章的 GEMM_AR 示例说，TPUT 在 ReduceScatter 阶段把本地 GEMM 结果原子累加到 owner rank 地址；TTEST 非阻塞检查 Ready Queue；TNOTIFY/TWAIT 实现 DeviceBarrier。

### 4.7 Ascend 950：集合通信可走 CCU 硬件卸载

适用：AllReduce、ReduceScatter、Broadcast、Reduce 等集合通信，尤其是大模型训练/推理中通信与计算抢资源严重的场景。

可画成：

```text
Host:
  compile/register/launch CCU kernel → CcuDeviceSession

Device:
  AIC computes tile
  AIV notifies / launches CCU
  CCU reads local/remote data
  CCU Memory Slice buffers data
  CCU Reduce Engine does Sum/Max/Min
  CCU writes result to local/remote GM
  CCU_DONE signal returns to AIV
```

CCU 技术文说，CCU 可利用片上缓存降低内存访问，并可在本地内存、远端内存、片上缓存之间搬运数据，部分搬运可支持随路规约。([cann.csdn.net](https://cann.csdn.net/69d4b15854b52172bc67789d.html "集合通信处理器（CCU）技术解读文档_昇腾_昇腾CANN-CANN开发者社区"))

文章还提到，在 PTO API 层，传统 AIV 自己搬和 CCU 卸载可以复用同名接口，只是多传一个 session 参数；但该文当时也说 CCU 路径已完成 CPU-SIM 功能仿真验证，真实硬件平台对接仍在进行。

---

## 5. GEMM + AllReduce 示例的准确路径

文章中的示例是 8 卡 910B，BF16，`M=5416, K=6144, N=1408`。有效结论是：PTO 把通信调度下放到 kernel 内，以 tile 粒度做到“算完一块，搬走一块”，而不是 Host 侧先 GEMM 后通信。文中给出的结果是纯计算 365 us，先算后通 743 us，流水线 631 us，约 31% 通信被计算掩盖。

### 5.1 传统串行

```text
AIC/Cube:
  GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM

Host:
  waits GEMM kernel finish
  launches communication kernel

AIV/communication:
  GM → communication path → remote GM
```

问题：计算和通信分属两个粗粒度 kernel，Host 调度无法感知 tile ready。

### 5.2 PTO 910B 流水线

```text
AIC / Cube Core:
  tile0 GEMM → L0C → FixPipe → GM
  tile1 GEMM → L0C → FixPipe → GM
  tile2 GEMM → L0C → FixPipe → GM
  ...

AIV / Vector Core:
  TTEST Ready Queue
  if tile ready:
      TPUT AtomicAdd to owner rank GM
  else:
      TWAIT or hardware wait

Barrier:
  TNOTIFY + TWAIT
```

这里 AIC 和 AIV 的关系不是“一个核内两条线程”，而是在分离架构中 Cube Core 与 Vector Core 独立执行、通过 GM/队列/同步机制协作。文章说 AIC 负责 GEMM tile，AIV 负责搬 tile；官方分离架构也说明 AIC/AIV 是独立核，各自有 Scalar，能独立加载代码段。 ([昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0008.html "基本架构-硬件架构-Ascend C算子开发-算子开发-CANN社区版8.0.RC2.alpha002开发文档-昇腾社区"))

### 5.3 PTO 950 + CCU 的推测/目标路径

如果走 CCU，示意路径应变为：

```text
AIC:
  tile GEMM done → result in GM / accessible buffer

AIV:
  signal tile ready / AIV_LAUNCH

CCU:
  fetch tile data
  reduce in Memory Slice / Reduce Engine
  write result
  signal CCU_DONE

AIV:
  continue next tile / next stage
```

AIV 不再逐块执行搬运和归约，只负责与 CCU 握手。这个路径应画成 **AIC → AIV control → CCU data+reduce**，不要画成 AIV 仍在执行实际 Reduce。

---

## 6. 最适合后续画图的“分层硬件草图”

建议先画一个抽象总图：

```text
Ascend NPU
│
├── Global Memory / GM / HBM or DDR
│   └── L2 Cache / memory fabric
│
├── AI Core cluster
│   │
│   ├── AIC / Cube Core
│   │   ├── Scalar
│   │   ├── Cube
│   │   ├── MTE1 / MTE2 / MTE3
│   │   ├── L1
│   │   ├── L0A / L0B / L0C
│   │   ├── BT Buffer / FP Buffer
│   │   └── FixPipe
│   │
│   └── AIV / Vector Core
│       ├── Scalar
│       ├── Vector
│       ├── MTE2 / MTE3
│       ├── UB / UBuffer
│       └── SIMD/SIMT Registers
│
├── AI Core inter-core / intra-group sync
│   ├── GM / Ready Queue
│   ├── CrossCore flags / signal grid
│   └── SSBuffer, if drawing 351x
│
└── Communication subsystem
    ├── SDMA: GM ↔ remote GM async
    ├── URMA: 950, UnifiedBus/RDMA remote memory path
    ├── CCU: 950, collective communication offload
    │   ├── Memory Slice
    │   ├── Reduce Engine
    │   └── Trans / Ctrl / Load microcode paths
    └── interconnect ports / UnifiedBus / RDMA plane
```

再在图上叠三类箭头：

```text
红色：AIC/Cube 计算路径
GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM/L1

蓝色：AIV/Vector 计算路径
GM → UB → Vector → UB → GM

绿色：跨卡通信路径
MTE sync:   GM → UB → remote GM
SDMA async: GM → SDMA → remote GM
URMA async: GM → URMA/UnifiedBus → remote GM
CCU coll:   GM ↔ CCU Memory Slice/Reduce ↔ remote GM
```

---

## 7. 需要标注“不确定/产品相关”的点

1. **910B 的 AIC/AIV 数量**：文章示例写 24 AIC + 24 AIV，但公开 CANN 351x 文档写 AIC:AIV=1:2，910B 相关论文也说通常 1 AIC + 2 AIV。画图时不要把 24/24 当通用硬件定论，除非你明确画的是该 GEMM_AR 示例的调度资源，而不是芯片全量资源。([昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00065.html "NPU架构版本351x-架构规格-硬件实现-编程指南-Ascend C算子开发-编程指南-CANN商用版9.0.0开发文档-昇腾社区")) ([arXiv](https://arxiv.org/html/2505.15112v1 "Parallel Scan on Ascend AI Accelerators"))

2. **URMA 的具体物理位置**：文章说它基于 UnifiedBus，是 950 的用户态 RDMA 路径；公开官方 roadmap 只确认 950 的高互连带宽和新架构方向，未确认 URMA 微结构。图中建议标成 “URMA / UnifiedBus RDMA engine, outside AI Core”。 ([huawei](https://www.huawei.com/en/news/2025/9/hc-xu-keynote-speech "Groundbreaking SuperPoD Interconnect: Leading a New Paradigm for AI Infrastructure - Huawei"))

3. **CCU 的落地状态**：文章称 PTO CCU 路径当时 CPU-SIM 验证完成，硬件平台对接进行中；CCU 技术文确认 CCU 的设计目标和能力，但公开资料未给出完整 RTL/端口/Memory Slice 容量。画图可画 CCU 模块，但 Memory Slice 大小、端口数、与 GM/L2 的精确拓扑先留空。 ([cann.csdn.net](https://cann.csdn.net/69d4b15854b52172bc67789d.html "集合通信处理器（CCU）技术解读文档_昇腾_昇腾CANN-CANN开发者社区"))

4. **TPUT 的远端 AtomicAdd 实现点**：文章说硬件原子加在目标侧完成，但没有说明原子是在远端内存控制器、互连协议层，还是专门 atomic 单元完成。图上建议标成 “remote-side atomic at target GM path”，不要画死到某个未确认模块。