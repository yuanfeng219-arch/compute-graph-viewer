# msinsight-view-selector — 为诊断结果推荐 MindStudio Insight 可视化视图

> 本 skill 给 AI 阅读。性能/算子分析报告中**每一个**问题点（瓶颈、异常、优化建议）后必须追加"问题举证视图"字段，告诉用户去 MindStudio Insight 哪个视图、载入哪个文件，能直观复现这个问题。

## 触发条件

适用于所有 **Hermes / Icarus** 工作流（性能 Profiling 分析、算子调优）。**Accuracy 工作流不适用**（精度问题不用可视化复现）。

只要报告里列出"问题 / 瓶颈 / 异常 / 优化建议"，每条都要带视图字段，不需要用户显式触发。

---

## 视图总览（MindStudio Insight）

| 视图 | 适用场景 | 主要能力 | 关键数据源 |
|------|---------|---------|----------|
| **Timeline 视图（系统调优）** | 训练/推理整体分析 | host API 与 device task 时间轴关联呈现，host vs device 瓶颈定性 | `trace_view.json` / `msprof_*.json` / `operator_details.csv` |
| **Timeline 视图（算子调优）** | 单算子内部分析 | AI Core 每个 Pipe 指令调用顺序与耗时 | `*trace.json` / `*.bin` |
| **Timeline 视图（服务化调优）** | 推理服务化性能 | 请求调度、显存管理、批处理端到端各阶段耗时 | `chrome_tracing.json` / `profiler.db` |
| **算子视图** | 算子热点定位 | 计算算子 / 通信算子耗时饼图 | `kernel_details.csv` |
| **通信视图** | 集群通信链路 | 全网链路带宽、各节点通信性能 | `cluster_communication_matrix.json` / `cluster_communication.json` |
| **内存视图** | 内存占用与通路 | 内存折线图、算子内存表格 | `memory_record.csv` / `npu_module_mem.csv` / `static_op_mem.csv` / `operator_memory.csv` |
| **源码视图** | 指令热点 | 算子源码与指令集映射、指令耗时热点 | `*.bin` |
| **详情视图** | 算子瓶颈定性 | 算子基础信息、计算单元负载、Roofline 瓶颈分析 | `*.bin` |

---

## DB-only 导出（`export_type=db`）的视图映射

> **背景**：下方「问题点 → 视图映射表」的「所需文件」默认是 **Text 导出件**（`trace_view.json` / `kernel_details.csv` / `cluster_communication_matrix.json` 等）。当采集只开了 `export_type=["db"]` 时，落盘目录里**这些文件一个都不存在**，只有 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db`、`analysis.db` 和集群侧 `cluster_analysis_output/cluster_analysis.db`。

此时**不要标"无视图"，也不要硬造一个渲染不出来的文件名**，按下表替换：

| 映射表里的 Text 件 | DB-only 下的替代（MindStudio Insight 可直接载入 `.db`） |
|---|---|
| `trace_view.json`（Timeline 系统调优） | 该 rank 的 `ascend_pytorch_profiler_{rank}.db` |
| `kernel_details.csv`（算子视图） | 同上 `ascend_pytorch_profiler_{rank}.db`（`COMPUTE_TASK_INFO`） |
| `cluster_communication_matrix.json` / `cluster_communication.json`（通信视图） | `cluster_analysis_output/cluster_analysis.db`（`ClusterCommunication*` 表） |
| `memory_record.csv` / `operator_memory.csv`（内存视图） | 该 rank `*.db` 的 `MEMORY_RECORD` / `OP_MEMORY` 表 |

两类**天生没有时序几何、无法用泳道图渲染**的发现，按"证据本质"挂载，而非套 Timeline：

- **advisor 文本类建议**（亲和 API 替换、"修改代码避免 AICPU"、动态 shape、JIT 在线编译等）：主证据 = **advisor 的 `mstt_advisor_*.html`**（定位到对应段落），辅以该 rank `.db` 的 `PYTORCH_API` / `CANN_API` 调用点。
- **recipe 派生统计**（`HcclTopOpStats` 的 min/max-rank 双峰、`SlowRank` 的 slowAffectCount、`FreeAnalysis` 的 launch 间隙等）：证据 = **对应 `-m` 模式产出的 `cluster_analysis.db`**（注明查哪张表、怎么排序），这类统计**只存在于 recipe 输出里**，原始落盘和原始 `cluster_analysis.db` 都没有。

> 采集端建议（写进报告「数据与方法/采集建议」）：要用标准 Insight 视图，**采集时同时开 `export_type=["text","db"]`**——这样既有 db 供集群 recipe 分析，又有 Text 件供视图直接渲染。

---

## 问题点 → 视图映射表

### 集群 / 多卡问题（Hermes）

| 诊断结果 | 推荐视图 | 关注点 | 所需文件 |
|---------|---------|-------|---------|
| Host 下发型慢卡（伪快卡） | Timeline 视图（系统调优） | 找慢卡 rank 的 `launch`、`aclrtSynchronizeDevice` 等 host API，对比 Free Time 占比 | 慢卡 rank 的 `trace_view.json` |
| 计算型慢卡 | 算子视图 + Timeline 视图（系统调优） | 算子视图看算子耗时占比；Timeline 看具体哪几个算子拖后腿 | 慢卡 rank 的 `kernel_details.csv` + `trace_view.json` |
| 通信/慢链路瓶颈 | 通信视图 | 链路带宽矩阵热力图，定位低带宽链路 | `cluster_communication_matrix.json` + `cluster_communication.json` |
| 集合通信（HCCL/AllReduce/AllGather）耗时高 | 通信视图 + Timeline 视图（系统调优） | 通信视图看链路；Timeline 看通信算子是否与计算重叠 | 同上 + `trace_view.json` |

### 单卡 Profiling 问题（Hermes）

| 诊断结果 | 推荐视图 | 关注点 | 所需文件 |
|---------|---------|-------|---------|
| 计算热点（TopK 算子耗时） | 算子视图 | 计算算子耗时饼图，按占比排序 | `kernel_details.csv` |
| 下发调度拥塞（PyTorch / CANN / Device 三层耗时不匹配） | Timeline 视图（系统调优） | 对比 host API 时间轴与 device task 时间轴的间隔 | `trace_view.json` + `operator_details.csv` |
| 内存峰值过高 / 内存碎片 | 内存视图 | 折线图看内存随时间变化；算子内存表定位大对象 | `memory_record.csv` + `operator_memory.csv` |

### 算子调优问题（Icarus）

| 诊断结果 | 推荐视图 | 关注点 | 所需文件 |
|---------|---------|-------|---------|
| 算子瓶颈类型未知（Compute / Memory / Latency Bound） | 详情视图 | Roofline 图直接定性 | `*.bin`（含 Roofline metrics） |
| 算子 MFU 低（< 20%） | 详情视图 + 算子视图 | Roofline 看落点；算子视图看耗时占比 | `*.bin` + `kernel_details.csv` |
| Scalar 指令占比过高（`aiv_scalar_ratio > 50%`） | Timeline 视图（算子调优）+ 源码视图 | Timeline 看 Scalar Pipe 占比；源码视图定位标量循环 | `*trace.json` + `*.bin` |
| 冗余同步（PipeBarrier / SetFlag/WaitFlag 过多） | Timeline 视图（算子调优） | 看各 Pipe 流水线停顿点 | `*trace.json` |
| 单缓冲无法流水 / Double Buffer 未开启 | Timeline 视图（算子调优） | CopyIn / Compute / CopyOut 三阶段是否完全串行 | `*trace.json` |
| L2 Cache 命中率低（`ai*_total_hit_rate < 80%`） | 详情视图 | 查看计算单元 L2 命中率指标 | `*.bin` |
| L2Cache 切分缺失 | 详情视图 + 内存视图 | 详情视图看命中率；内存视图看 HBM 带宽 | `*.bin` + `memory_record.csv` |
| 多核切分不足（blockDim 未饱和） | Timeline 视图（算子调优） | 看核占用，是否有空闲核 | `*trace.json` |
| 流水线负载不均（MTE1/MTE2/Cube/Vector 占比异常） | Timeline 视图（算子调优）+ 详情视图 | Timeline 看各 Pipe 占比；详情视图看计算单元负载 | `*trace.json` + `*.bin` |
| UB Buffer 未融合（中间结果 GM 往返） | Timeline 视图（算子调优）+ 内存视图 | Timeline 看冗余搬运；内存视图看 UB↔GM 流量 | `*trace.json` + `memory_record.csv` |
| L0C / L1 / BT / FP Buffer 使用不当 | 详情视图 + Timeline 视图（算子调优） | 详情视图看各 Buffer 利用率；Timeline 看搬运指令 | `*.bin` + `*trace.json` |
| 指令热点 / 算子源码映射 | 源码视图 | 直接看哪几行 AscendC 代码对应最耗时的指令 | `*.bin` |
| PMSampling 内存通路波形（simulator） | 内存视图 | 内存通路波形图 | simulator `*.bin`（启用 PMSampling） |

### 推理服务化问题

| 诊断结果 | 推荐视图 | 关注点 | 所需文件 |
|---------|---------|-------|---------|
| 请求调度 / 批处理 / 显存管理瓶颈 | Timeline 视图（服务化调优） | 端到端各关键阶段耗时 | `chrome_tracing.json` + `profiler.db` |

---

## 在报告中的呈现方式

### 方式 A：表格中独立一列（推荐用于多个问题并列时）

```markdown
| 问题点 | 现象 | 定性 | 建议 | 问题举证视图 |
|-------|------|------|------|----------|
| Host 下发型慢卡 | rank3 Free Time 占比 18% | CPU 下发慢导致 NPU 饿死 | 排查绑核与同步 API | **Timeline 视图（系统调优）** — 载入 `evidence/rank3/trace_view.json`（源：`/data/prof/.../rank3/trace_view.json`），过滤 `aclrtSynchronizeDevice` |
```

### 方式 B：每条问题后单独一行（适合分点列举）

```markdown
**问题 2：算子 MFU 仅 14%**
- 现象：MatMul 算子 MFU = 14%，远低于硬件能力
- 定性：Memory Bound（带宽受限）
- 建议：增大 M/N/K 维度或采用算子融合
- **问题举证视图**：详情视图 — 载入算子 `evidence/op_MatMul/kernel.bin`（源：`/data/prof/.../op_MatMul/kernel.bin`），查看 Roofline 图，确认落点位于 Memory Roof 下方
```

### 字段格式约束

每条 `问题举证视图` 字段必须包含三要素：

1. **视图名**（含 Timeline 子场景标注，如"Timeline 视图（算子调优）"）
2. **所需文件**——该文件必须已复制进报告目录的 `evidence/` 子目录（见 profiling-workflow 规则 5），书写时**副本相对路径与原始落盘路径两者都写**：`evidence/<...>/文件名`（源：`原始落盘绝对路径`）
3. **打开后看什么**（一句话，告诉用户进入视图后过滤/聚焦哪里）

反例（缺要素，不要这样写）：
- ❌ "问题举证视图：Timeline" — 缺子场景、缺文件、缺关注点
- ❌ "问题举证视图：用 MindStudio Insight 看" — 没说哪个视图
- ❌ "Timeline 视图 — 载入 `trace_view.json`" — 未复制进 `evidence/`、未写副本路径与原始来源

正例：
- ✅ "Timeline 视图（系统调优）— 载入 `evidence/rank3/trace_view.json`（源：`/data/prof/.../rank3/trace_view.json`），过滤 host 侧 `launch` API，观察与 device task 之间的 Free Time 间隙"

---

## 多视图组合

若一个问题需要多个视图交叉印证，按"主视图 + 辅助视图"列出，主视图在前：

```markdown
**问题举证视图**：
- 主：算子视图 — `evidence/rank0/kernel_details.csv`（源：`/data/prof/.../rank0/kernel_details.csv`），确认 TopK 算子集中度
- 辅：Timeline 视图（系统调优）— `evidence/rank0/trace_view.json`（源：`/data/prof/.../rank0/trace_view.json`），确认这些算子是否被 host 下发拖慢
```

> 上述 `evidence/...` 副本由 profiling-workflow 规则 5 在出报告时统一复制；多视图引用同一文件时只复制一份。

---

## 找不到映射的情况

若诊断出的问题不在上方映射表中（如新发现的边角问题），按以下原则就近选视图：

| 问题特征 | 默认视图 |
|---------|---------|
| 涉及"耗时占比 / TopK / 热点" | 算子视图 |
| 涉及"时间轴 / 并发 / 流水 / 等待" | Timeline 视图（按场景选系统/算子/服务化） |
| 涉及"带宽 / 通信 / 链路" | 通信视图 |
| 涉及"内存 / 显存 / Buffer / 搬运" | 内存视图 |
| 涉及"Roofline / 算力 / 计算单元负载" | 详情视图 |
| 涉及"代码定位 / 指令热点" | 源码视图 |

仍不确定时，在报告中标注 `**问题举证视图**：暂无直接映射的 Insight 视图（问题需结合代码与日志分析）`，**不要硬凑**。
