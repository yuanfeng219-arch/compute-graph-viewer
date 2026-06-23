# AscendProfKit — Ascend NPU 性能与精度分析能力包

本文件夹是针对 Ascend NPU 性能分析、算子调优和精度分析的**专家知识资产包**，包含 15 个 Skill（专家知识库）、3 个 Agent 角色定义及对应提示词。**将本文件夹复制到任意项目**后，在 VSCode 等 IDE 中通过 Claude Code 对话，即可针对项目内的性能数据文件展开分析——Claude 会自动读取相应 Skill 知识库，按专家流程指导或直接执行分析。

---

## 目录结构

```
AscendProfKit/
├── CLAUDE.md                                      # Claude Code 专家角色加载指引（自动生效）
├── skills/                                        # 全部 Skill 知识库（共 15 个）
│   ├── profiling-workflow/                        # 通用报告输出规范（所有分析均适用）
│   ├── performance-health-score/                  # 性能健康度评分 PHS（报告结论速览首行，含场景权重规则 + 显存容量利用率诊断指标）
│   ├── ascend-profiler-db-explorer/               # Profiling DB SQL 查询（含 CTE 宏）
│   ├── ascendc-operator-performance-optim/        # AscendC 端到端算子调优（5 阶段闭环）
│   ├── cluster-fast-slow-rank-detector/           # 集群快慢卡诊断（含分析脚本）
│   ├── timeline-swimlane-analyzer/                 # Timeline 泳道时序结构分析（关键路径/重叠/空挡/bubble/抖动）
│   ├── github-raw-fetch/                          # GitHub 文档抓取辅助
│   ├── mindstudio_profiler_data_check/            # Profiler 数据完整性校验
│   ├── dataset-source-identifier/                 # 落盘数据来源识别（来源/是否 LLM/模型/用途，含识别依据，无依据留空不猜）
│   ├── msprof-analyze-cli/                        # 集群慢卡/慢链路综合分析与专家建议
│   ├── msinsight-view-selector/                   # 为每个诊断结果推荐 MindStudio Insight 可视化视图
│   ├── msot-msopprof-operator-profiler/           # msOpProf 算子深度性能分析
│   ├── nan-overflow-detection/                    # 多卡分布式训练 NaN/Inf 溢出检测与根因追溯
│   ├── op-mfu-calculator/                         # 算子 MFU 计算（GEMM / FlashAttention，支持从 profiler 落盘批量提取聚合）
│   └── rl-consistency-analysis/                   # 训推一致性根因分析（含分析脚本）
└── resources/
    └── configs/
        └── default/
            ├── agents/                            # Agent 角色定义（Hermes / Icarus / Accuracy）
            ├── prompts/
            │   ├── agents/                        # Agent 系统提示词
            │   ├── shared/                        # 压缩共享提示词
            │   ├── subagents/                     # 子 Agent 提示词
            │   └── suffixes/                      # 环境后缀提示词
            └── subagents/                         # 子 Agent 配置
```

---

## 如何在其他项目中使用

本能力包**不依赖任何特定软件**，只需两步。

### 第一步：将本文件夹复制到目标项目根目录

```
your-project/
├── AscendProfKit/            ← 复制到这里（含 CLAUDE.md、skills/、resources/）
├── profiling_output/         ← 你的性能数据
└── ...
```

`CLAUDE.md` 已随文件夹附带，Claude Code 会自动加载其中的专家角色与知识库路径，无需额外配置。

### 第二步：在 VSCode 中直接对话

打开 VSCode，在 Claude Code 聊天框中描述问题即可：

- `"帮我分析 profiling_output/rank0_profiler_20240101.db，找出计算瓶颈"`
- `"这次集群训练有没有快慢卡问题？"`
- `"我的算子 MFU 只有 18%，帮我诊断一下"`
- `"对比 rollout 和 train 的 dump，找训推不一致的根因"`

Claude 会按 `CLAUDE.md` 的指引读取对应 `SKILL.md`（专家知识）和 `scripts/`（分析脚本），按专家流程展开分析。

### 可选：安装 msprof-analyze，获取集群自动诊断增强

如需使用 `msprof-analyze-cli` skill（集群综合分析与 advisor 建议），需在运行环境中安装 `msprof-analyze` 命令行工具并将其加入 PATH。

**安装后可用的功能**：
- 集群耗时拆解与慢卡矩阵（`msprof-analyze -m cluster_time_summary`）
- 自动专家建议（`msprof-analyze advisor all`）
- 多维集群分析（compute_op_sum、hccl_sum、slow_rank、slow_link 等）

**未安装也没关系**：AscendProfKit 的其他分析能力（DB SQL 查询、MFU 计算、算子调优、NaN 溢出检测等）不受影响，报告头部会标注"Advisor 未调用"。

---

## Agent 一览

| 名称 | 定位与用途 | 依赖 Skills |
|------|-----------|------------|
| **Hermes** | **性能调优**：聚焦 Ascend Profiling 分析，覆盖单卡、多卡、集群等场景，擅长快慢卡、慢节点、MFU、通信瓶颈、算子热点、下发调度等性能问题定位与优化建议。默认启动 Agent。 | `ascend-profiler-db-explorer`（DB SQL 查询）<br>`cluster-fast-slow-rank-detector`（快慢卡诊断）<br>`timeline-swimlane-analyzer`（Timeline 泳道时序结构分析）<br>`mindstudio_profiler_data_check`（数据校验）<br>`dataset-source-identifier`（落盘数据来源/模型识别，无依据留空不猜）<br>`msprof-analyze-cli`（集群综合分析与专家建议）<br>`op-mfu-calculator`（MFU 计算）<br>`msinsight-view-selector`（Insight 视图推荐）<br>`github-raw-fetch`（文档抓取） |
| **Icarus** | **算子调优**：聚焦 Ascend NPU 算子性能调优，包括算子性能深度分析、端到端算子性能优化（编译→采集→诊断→改码→验证），辅助提升算子性能优化效率，降低开发难度。 | `ascendc-operator-performance-optim`（端到端调优闭环）<br>`msot-msopprof-operator-profiler`（msOpProf 深度分析）<br>`msinsight-view-selector`（Insight 视图推荐） |
| **Accuracy** | **精度调优**：聚焦 Ascend 精度分析与优化，覆盖单卡、多卡、集群等场景，可处理 RL 训推一致性分析、loss/gnorm NaN 分析等常见精度问题。 | `rl-consistency-analysis`（训推 dump 根因分析）<br>`nan-overflow-detection`（多卡 NaN/Inf 溢出检测与根因追溯） |

---

## 指标速查与权重规则

### Timeline 泳道时序结构指标（`timeline-swimlane-analyzer`）

聚合统计型 skill（DB SQL、msprof-analyze recipe）只给"耗时占比 / TopK / 慢卡"等聚合量，**无法**给出从 timeline 泳道几何结构派生的指标。下列指标由 `timeline-swimlane-analyzer` 补齐，其中维度 1/2/3/6 的几何部分由本 skill 自带脚本 `scripts/timeline_geometry.py` 直接算出（输入 `trace_view.json`，可选叠加 `step_trace_time.csv`），维度 4/5/7/8 由 AI 结合数据按算法判断。

| 指标 | 含义 | 数据来源 | 越线阈值（⚠） | 产出方式 |
|------|------|---------|--------------|---------|
| 关键路径占比 | 最长链路 ÷ step 总耗时，决定优化天花板 | `trace_view.json`（device∪暴露通信忙时并集近似） | < 80% 存在可掩盖空泡 | 脚本 |
| 计算-通信重叠率 / 暴露通信 | 通信被计算掩盖的比例 | trace 内 `Overlap Analysis` 轨道；无则区间交集回退 | < 70% 掩盖不足 | 脚本 |
| 泳道空挡比例 / 最大间隔 | 各泳道空闲占比与最大空隙，按空闲排名 | `trace_view.json` 各 `pid/tid` 泳道 | device 泳道空挡 > 10% 多为 host 下发跟不上 | 脚本 |
| step 抖动 / 长尾 | step 耗时 CV、P95、长尾 step 序号 | `step_trace_time.csv` | CV > 10% 训练不稳定 | 脚本 |
| 跨泳道对齐间隙 | 依赖等待空隙（上游结束→下游开始） | `trace_view.json` 依赖边 | TopN 间隙集中即依赖过紧 | 脚本+AI |
| Host-bound 下发对齐 | device 周期性小间隙是否对齐 host `launch` | Python/CANN vs Ascend Hardware 泳道 | launch gap 占比 > 5% | AI |
| PP 流水线 bubble 率 | 实测 bubble vs `(p-1)/(p-1+m)` 理论 | P2P send/recv 泳道 + 计算空隙 | 实测远大于理论 | AI |
| 关键路径热点 | TopK 裸耗时算子是否落在关键路径再加权 | compute_view 结果 × 关键路径 | 榜首不在关键路径则优化无效 | AI |
| 推理 prefill/decode / TTFT / TPOT | 按请求边界拆两阶段（算力 vs 访存 bound） | 服务化 `chrome_tracing.json` + `profiler.db` | 两阶段瓶颈不同需分开归因 | AI |

> 脚本只算几何近似（关键路径无依赖边、重叠率依赖 `Overlap Analysis` 桶语义），所有桶值均打印，须结合 MindStudio Insight Timeline 视图人工校正。

### 计算与显存利用率指标（`op-mfu-calculator` / `performance-health-score`）

从 profiler 落盘（**目录名 / 采集等级不限**，识别 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/`）批量提取的利用率指标。MFU 需采集时开 `record_shapes=true`，显存需 `profile_memory=true`。

| 指标 | 含义 | 数据来源 | 归属 skill |
|------|------|---------|-----------|
| MFU（算子达成率） | Σ算子 FLOPs ÷ Σ算子耗时 ÷ 芯片峰值；matmul 用输出 shape 锚定 M/N、A 非 M 维定 K（**转置安全**） | `kernel_details.csv`（shape+耗时）/ DB `COMPUTE_TASK_INFO` | `op-mfu-calculator` |
| MFU（端到端 step） | Σ整网 FLOPs ÷ (step 总跨度 × 峰值)，永远 ≤ 算子达成率 | 上 + `step_trace_time.csv` | `op-mfu-calculator` |
| cube_utilization | 硬件实测 AI Core cube 流水线利用率（≠ MFU，互补同报） | `kernel_details.csv` 自带列 / `aic_mac_ratio` | `op-mfu-calculator` |
| 显存容量利用率 | 显存占用峰值(GB) ÷ 单卡 HBM 总容量；与"带宽利用率"是两个维度 | `memory_record.csv`/`npu_module_mem.csv`/`operator_memory.csv` / DB `NPU_MEM` | `performance-health-score`（诊断项，不计入 PHS） |
| 芯片型号/峰值 | 定 MFU 峰值与 HBM 容量；落盘 metadata 常缺，优先查 DB | `ascend_pytorch_profiler_*.db` 的 `NPU_INFO` 表 | 两者共用 |

> ⚠️ 见到 MFU > 100% 多半是 matmul 的 M/N/K 用输入 shape 直接推导、未处理 NN/NT/TN 转置布局所致；务必用**输出 shape** 锚定 M/N。
>
> ⚠️ **MFU 双重失效兜底**（`op-mfu-calculator`）：当主力 matmul 为 TP 融合算子（`AllGatherMatmul`/`MatmulReduceScatter` 及 `*Aicpu` 变体，`InputShapes=N/A`、且耗时内嵌通信）**且** `aic_metrics` 非 `PipeUtilization`（如选了 `ACL_AICORE_MEMORY_UB`，没采 `cube_utilization`）时，FLOPs 与 cube_util 两档都不可用——退到第三档**时间口径 device-busy 代理**（Σ计算算子耗时 ÷ step 跨度，按 AICPU / Block Dim 占比下修），并显式标注"代理值，非真实 MFU"，不要硬算假 MFU。
>
> ⚠️ **纯 DB 导出（`export_type=db`）与时间单位**：落盘无 `trace_view.json`/`kernel_details.csv`/`communication_matrix.json` 时，Insight 直接载入 `.db`（见 `msinsight-view-selector` 的「DB-only 视图映射」）；对耗时换算前先核对单位——`ClusterStepTraceTime` 为 **μs**、`*Ns`/`STEP_TIME` 为 **ns**、通信 `transit_time` 为 **ms**，混用会差 1000×（详见 `ascend-profiler-db-explorer` 的「字段时间单位速查」）。

### 优先级权重规则（贯穿所有性能报告）

**1. 关键路径加权（`timeline-swimlane-analyzer` 核心原则）**

```
优化收益 ≈ 是否在关键路径 × 可压缩程度 × 出现频次
```

"最高单任务 / 最大泳道间隔 / 耗时占比"等"看最大值"的指标，都先叠一层"是否落在关键路径"——不在关键路径上的耗时，能被并行掩盖，压了也不省 step 时间。

**2. 性能健康度 PHS 场景权重表（`performance-health-score`）**

每份性能/算子报告"结论速览"首行的 0-100 评分，由四个子项加权求和，权重按工作负载场景切换（完整公式与 N/A 归一化以 `performance-health-score/SKILL.md` 为准）：

| 场景 | 计算 | 通信 | 调度 | 内存 | 集群均衡度 |
|------|-----|-----|-----|-----|----------|
| 大模型多卡训练（默认） | 0.40 | 0.30 | 0.20 | 0.10 | — |
| 单卡训练 / 推理 | 0.50 | N/A | 0.30 | 0.20 | — |
| 单算子调优 | 0.50 | N/A | 0.20 | 0.30 | — |
| 集群慢卡场景 | 0.20 | 0.30 | 0.30 | 0.10 | 0.10 |

> 子项缺数据记 N/A，剩余权重按 `归一化权重 = 原权重 ÷ (1 - N/A 子项权重之和)` 放大，避免把"未采集"误判为"不健康"。`timeline-swimlane-analyzer` 的关键路径占比 / 重叠率 / 空挡比例可作为 PHS 通信效率、调度效率子项的佐证，但不改 PHS 公式。

---

## 诊断知识库

以下表格汇总各 Skill 中内嵌的**如何诊断问题**的专家知识，供快速查阅。

| 所在文件 | 问题 | 现象 | 定性 | 动作或建议 |
|---------|------|------|------|-----------|
| `cluster-fast-slow-rank-detector/SKILL.md` | Host 下发型慢卡（伪快卡） | 某卡 Free Time 极长（占比 >10% 或远超均值），且 Compute / Communication 时间异常偏短 | 该卡是真正的慢卡：CPU 下发慢导致 NPU 饿死，其他卡已等待多时，故其通信瞬间完成，外观似"快卡" | 调用 `compare_api_stats.py`，重点观察 `launch`、`aclrtSynchronizeDevice` 等下发/同步 API 的耗时与间隙差异；优化方向：绑核、减少 CPU 侧同步 |
| `cluster-fast-slow-rank-detector/SKILL.md` | 计算型慢卡 | 各卡 Free Time 普遍较短且均匀，但某卡 Compute Time 显著大于均值 | 计算型慢卡：若算子调用次数（count）不同则负载切分不均；若次数相同但平均耗时（avg_time）激增，则为算子硬件劣化或动态 Shape | 调用 `compare_op_stats.py` 对比算子执行差异；优化方向：检查切分策略或排查 Shape 异常 |
| `cluster-fast-slow-rank-detector/SKILL.md` | 通信/慢链路瓶颈 | 各卡通信带宽远低于理论值（如 SDMA < 2 GB/s） | 小包通信（ZeRO3 切分过细）、SDMA 地址未对齐或硬件链路故障 | 检查通信粒度与切分策略；确认 SDMA 地址对齐；排查硬件链路 |
| `timeline-swimlane-analyzer/SKILL.md` | 空泡多但单算子不慢 | 关键路径占比 < 80%，device 泳道空挡比例高 | 结构性空泡：算力被等待饿死，而非算子本身慢 | 先做重叠（计算-通信、host-device），不要急着优化单算子 |
| `timeline-swimlane-analyzer/SKILL.md` | 通信未被掩盖 | `Overlap Analysis` 中 `Communication(Not Overlapped)` 占比高，或重叠率 < 70% | 暴露通信进入关键路径 | 开启通信-计算并发、放宽依赖、增大 micro-batch；检查通信是否独立 stream |
| `timeline-swimlane-analyzer/SKILL.md` | device 周期性小间隙 | device gap 与 host `launch`/同步 API 周期对齐，launch gap 占比 > 5% | Host-bound（下发跟不上），单看 device 看不出 | 绑核、算子融合/图模式减少下发条数、消除 host 同步点 |
| `timeline-swimlane-analyzer/SKILL.md` | PP bubble 偏大 | 实测 bubble 率远大于 `(p-1)/(p-1+m)` 理论值 | micro-batch 少 / stage 切分不均 / 重计算放大 | 增大 num_microbatch、均衡 stage、interleaved 1F1B |
| `timeline-swimlane-analyzer/SKILL.md` | step 抖动大 | step 耗时 CV > 10% 或存在 >1.5×median 的长尾 step | 训练不稳定，全局最大值被异常 step 污染 | 先 per-step 归一取典型 step 再分析；周期长尾排查 ckpt/eval/GC |
| `timeline-swimlane-analyzer/SKILL.md` | 裸热点优化无效 | TopK 耗时榜首不在关键路径 | 该任务可被并行掩盖，压它不省 step | 改打"关键路径热点榜"榜首 |
| `timeline-swimlane-analyzer/SKILL.md` | 推理 prefill/decode 混判 | 用整体 timeline 看推理，瓶颈结论摇摆 | 两阶段瓶颈不同（算力 vs 访存） | 按请求边界拆 prefill/decode，分别用 TTFT / TPOT 归因 |
| `ascend-profiler-db-explorer/SKILL.md` | 定位算子计算热点 | 用户需查询"哪些算子最耗时"/"TopK 算子"/"计算瓶颈" | 需从 Profiling DB 的 `COMPUTE_TASK_INFO` 聚合算子耗时 | 使用 Compute CTE 宏查询 `compute_view`，按 `SUM(duration_ns) DESC LIMIT 20` 排序 |
| `ascend-profiler-db-explorer/SKILL.md` | 集合通信耗时分析 | 用户需分析 HCCL / AllReduce / AllGather 耗时 | 需从 `COMMUNICATION_OP` 表聚合通信算子时长 | 使用 Communication CTE 宏查询 `comm_view`，按耗时排序 |
| `ascend-profiler-db-explorer/SKILL.md` | 下发调度拥塞 | 用户需分析 PyTorch 框架下发 vs CANN 下发 vs 设备执行的耗时差异 | 框架层 / CANN 层 / 设备层耗时不匹配，存在调度拥塞 | 使用 Dispatch CTE 宏查询 `dispatch_view`，对比 `pytorch_duration_ns`、`cann_duration_ns`、`task_duration_ns` 三列差值 |
| `ascend-profiler-db-explorer/SKILL.md` | 跨表耗时单位不一致（差 1000×） | `ClusterStepTraceTime` 为 **μs**，而 `*Ns`/`STEP_TIME.startNs/endNs` 为 **ns**、通信 `transit_time` 为 **ms**（`transit_size`=MB、`bandwidth`=GB/s）；混用曾把 6.33 **秒** 的 step 误标成 6.33 ms | 单位陷阱，非性能问题 | 对耗时做拆解/换算前查「字段时间单位速查」表；用 `STEP_TIME`(ns) 交叉验算 `ClusterStepTraceTime`(μs)，统一换算 ms（秒级 step 标 s）并显式标单位 |
| `mindstudio_profiler_data_check/SKILL.md` | Profiler 采集未正常 Stop | 框架 profiler：`profiler_info.json` 缺失；msprof：`PROF_{}/device_{}/end_info.*` 缺失 | 采集过程中进程异常退出或 `profiler.stop()` 未被调用，数据不完整 | 检查代码中 `profiler.stop()` 调用路径；修复后重新采集；不得对未正常 Stop 的数据继续分析 |
| `mindstudio_profiler_data_check/SKILL.md` | Profiler 数据未解析 | `ASCEND_PROFILER_OUTPUT` 目录（框架 profiler）或 `mindstudio_profiler_output` 目录（msprof）缺失 | 原始 profiling 数据尚未导出为可分析格式，后续工具无法运行 | 框架 profiler 执行 `offline_parse_pytorch.py` 或 `offline_parse_mindspore.py`；msprof 执行 `msprof --export=on --output=<path>` |
| `mindstudio_profiler_data_check/SKILL.md` | 关键交付件缺失 | Text 模式缺 `trace_view.json` / `kernel_details.csv`；DB 模式缺 `*_profiler_*.db`；msprof 缺 `msprof_*.db` 或 `op_summary.csv` | 对应分析功能（Timeline / 算子分析）将无法执行 | 检查采集时的 `export_type` 配置；必要时重新采集并确认导出完整 |
| `msinsight-view-selector/SKILL.md` | 纯 DB 导出（`export_type=db`）无标准视图文件 | 落盘只有 `*.db`，无 `trace_view.json`/`kernel_details.csv`/`communication_matrix.json` | Insight 可直接载入 `.db`——映射表的 Text 件不存在 ≠「无视图」，不要硬造文件名 | 按「DB-only 视图映射」用对应 `.db` 替代（rank `ascend_pytorch_profiler_*.db` / 集群 `cluster_analysis.db`）；advisor 文本类建议挂 `mstt_advisor_*.html`、recipe 派生统计挂对应 `-m` 产出 db；采集端建议同时开 `export_type=["text","db"]` |
| `profiling-workflow/SKILL.md`（规则 5） | 举证文件遗漏派生证据 | 结论引用 `SlowRank`/`HcclTopOpStats`/`FreeAnalysis` 等表或 advisor 建议，但 `evidence/` 只复制了原始落盘 / 原始 `cluster_analysis.db` | 这些派生统计**只在** msprof-analyze 各 `-m` 产出与 `mstt_advisor_*.html` 里，原始 db 不含 | 把对应 recipe 的 `cluster_analysis.db` 与 advisor html 一并复制进 `evidence/` 并登记进举证清单，注明查哪张表 |
| `dataset-source-identifier/SKILL.md` | 落盘数据来源/模型识别 | 需判定这批数据是什么模型/用途、是否 LLM 训练 | 按落盘内证据取证：tokenizer vocab（`151936`→Qwen 系列）、算子签名（RMSNorm/SwiGlu/RoPE/FlashAttention→Transformer LLM；含 `*Grad`/`ApplyAdamWV2`→训练）、多 Embedding 表+小 hidden→推荐/CTR、单算子 simulator→算子调优 | **有依据才写，无确证依据的字段（如被 PP/TP 切分无法反推的模型规模）留空、禁止臆测**；结论 + 识别依据写入 report.md §5「数据来源与落盘信息」块（规则 8），前端自动填「落盘文件信息」卡片 |
| `op-mfu-calculator/SKILL.md` | 算子 MFU 极低 | MFU < 20% | 算子远未吃满算力，存在明显性能损失 | 排查内存带宽瓶颈、launch overhead、Shape 不规则（过小或非对齐）等因素 |
| `op-mfu-calculator/SKILL.md` | 算子 MFU 中等 | MFU 30%～60% | 通用工作负载典型水平，仍有提升空间 | 结合 Roofline 分析确认是 Compute Bound 还是 Memory Bound，针对性优化 Shape 或并行度 |
| `op-mfu-calculator/SKILL.md` | 算子 MFU 高 | MFU > 70% | 算子形状、并行度和实现已接近设备上限 | 继续保持；若仍需提升可尝试混合精度或算子融合 |
| `op-mfu-calculator/SKILL.md` | MFU 算出 > 100%（不合理） | 落盘聚合 MFU 冲到 100%~180% | matmul 的 M/N/K 直接用输入 shape 推导、未处理 NN/NT/TN 转置布局，N 被错当成 K 维 | 用**输出 shape** 锚定 M/N，K 取输入 A 中非 M 的那一维；FA 不要套 matmul 公式 |
| `op-mfu-calculator/SKILL.md` | MFU 无法计算（融合/AICPU 算子 + 非 cube PMU） | 主力 matmul 为 `AllGatherMatmul`/`MatmulReduceScatter`（含 `*Aicpu`），`InputShapes=N/A`；且 `aic_metrics=ACL_AICORE_MEMORY_UB` 未采 `cube_utilization` | FLOPs 与 cube_util 双重缺失；融合算子耗时内嵌通信，本就不该套 matmul MFU | 走兜底链 ①FLOPs→②cube_util→③时间口径 device-busy 代理（按 AICPU/Block Dim 占比下修），显式标注"代理值，非真实 MFU"，不硬算 |
| `performance-health-score/SKILL.md` | 显存容量利用率高 / OOM 风险 | `memory_record.csv` 的 `Total Reserved` 峰值 ÷ 单卡 HBM 容量 > 90% | 接近显存上限，易触发 OOM | recompute / 调小 micro-batch / 调整并行切分；若 Reserved≫Allocated 则查分配器碎片（`expandable_segments`） |
| `msot-msopprof-operator-profiler/SKILL.md` | 算子瓶颈类型未知 | 算子性能不达预期，不确定是计算限制还是内存限制 | 需通过 Roofline 定性瓶颈类型（Compute Bound / Memory Bound / Latency Bound） | 使用 `--aic-metrics=Roofline,Default` 采集，用 MindStudio Insight 查看 `visualize_data.bin` 中的 Roofline 图 |
| `msot-msopprof-operator-profiler/SKILL.md` | `--kernel-name` 参数不生效 | 指定算子名过滤但无效，采集了全部算子 | `--kernel-name` 仅在 `application` 输入形态下有效，对 `config` / `export` 无效 | 切换为 `application` 输入形态；若使用 `config`，改用其他过滤方式（如 `--launch-count`） |
| `msot-msopprof-operator-profiler/SKILL.md` | simulator `--soc-version` 不生效 | `msprof op simulator --config` 场景下 `--soc-version` 指定无效 | `config` 场景下仿真器通过 `LD_LIBRARY_PATH` 指定，`--soc-version` 不生效 | 改用 `export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/<SocVersion>/lib:$LD_LIBRARY_PATH` |
| `msot-msopprof-operator-profiler/SKILL.md` | `TimelineDetail` 在仿真模式无效 | simulator 模式下采集 `TimelineDetail` 但未生成对应数据 | `TimelineDetail` 是 device 模式专有能力，simulator 不支持 | 切换到 device 模式使用 `TimelineDetail`；simulator 中改用 `trace.json` 分析指令流水 |
| `msot-msopprof-operator-profiler/SKILL.md` | `PMSampling` 数据为空 | simulator 模式下内存通路波形图无数据 | `PMSampling` 在 simulator 默认关闭，需显式启用 | 在 `--aic-metrics` 中显式加入 `PMSampling`；注意 `--core-id` 对 PMSampling 不生效，它解析全部核 |
| `msot-msopprof-operator-profiler/SKILL.md` | `replay-mode=range` 与其他指标冲突 | 同时使用 `--replay-mode=range` 与 `TimelineDetail` / `Source` / `MemoryDetail` 时报错或数据异常 | 这三个指标与 `range` 模式不兼容 | 二选一：要用 `range` 则去掉上述三个指标；要用这三个指标则改用其他 replay 模式 |
| `msot-msopprof-operator-profiler/experiences/simulator-needs-sim-build.md` | 仿真拉起 signal 6 / Bad address | `msprof op simulator` 运行时抛出 `basic_filebuf::xsgetn error reading the file: Bad address` 或 `Child process killed by signal 6` | 当前可执行文件可能不是仿真兼容产物，或缺少与仿真器匹配的构建选项/依赖 | 检查工程构建脚本是否有 `--simulator` / `sim` 开关；生成仿真兼容产物后重试；同时确认 `LD_LIBRARY_PATH` 指向正确的仿真器目录 |
| `ascendc-operator-performance-optim/SKILL.md` | Scalar 指令占比过高 | `aiv_scalar_ratio > 50%` | 大量标量操作（`GetValue` / `SetValue` 循环 / `for` 循环逐元素）占用流水 | 用向量化 API（`Duplicate`、`Exp`、`Log`、`Mul` 等）替代标量循环；将 `TPipe` 移至 kernel 入口函数外（类外创建）以消除约 17% scalar_time |
| `ascendc-operator-performance-optim/SKILL.md` | 冗余同步导致流水停顿 | `PipeBarrier` 和 `SetFlag/WaitFlag` 数量过多，流水线频繁停顿 | 不必要的同步屏障使 Compute 与搬运无法重叠 | 合并连续向量操作，减少中间同步；检查是否可以使用 `TQueBind` 消除冗余的 LocalTensor 间 DataCopy |
| `ascendc-operator-performance-optim/SKILL.md` | 单缓冲无法流水 | 搬运和计算串行执行，流水线利用率低（Vector 利用率约 33%） | `InitBuffer` 的 buffer 个数为 1，CopyIn / Compute / CopyOut 完全串行 | 将 `InitBuffer` 的 buffer 个数改为 2，开启 Double Buffer；前提：循环次数 ≥ 2 且搬运时间不可忽略 |
| `ascendc-operator-performance-optim/SKILL.md` | L2 Cache 命中率低 | `ai*_total_hit_rate(%) < 80%` | 数据复用性差，访问模式不连续，频繁访问 HBM（~1.6 TB/s）而非 L2Cache（~7 TB/s） | 优化数据排布和访问模式；检查是否缺少 L2Cache 切分（见下条） |
| `ascendc-operator-performance-optim/SKILL.md` | 流水线负载不均 | 某条流水线（MTE1 / MTE2 / Cube / Vector）占比异常高或异常低 | 计算与搬运时间不匹配，或计算单元间负载不均 | 调整 Double Buffer 策略；优化计算分工；检查 Tiling 切分是否均衡 |
| `ascendc-operator-performance-optim/references/tiling-prof.md` | 多核切分不足 | `blockDim` 未设为硬件核数，部分核空闲 | 未充分利用硬件并行，整体吞吐受限 | 耦合架构用 `GetCoreNumAiv()` / `GetCoreNumAic()`；分离架构 Vector 算子用 AIV 核数（如 40），Cube 算子用 AIC 核数（如 20），MIX 算子用物理核组数（如 20） |
| `ascendc-operator-performance-optim/references/tiling-prof.md` | L2Cache 切分缺失 | 输入+输出数据量 > L2Cache 容量（约 192 MB），但未按 L2Cache 大小分块 | 每次循环都从 HBM 加载数据，L2Cache 频繁淘汰，带宽利用率低 | 将数据按 L2Cache 大小切块，外层循环按块迭代，所有核协同处理同一块后再切换，使第二次读命中 L2Cache |
| `ascendc-operator-performance-optim/references/tiling-prof.md` | 核间负载不均（尾块偏斜） | L2Cache 切分后部分核始终分到尾块，整体完成时间被拖慢 | 尾块固定分配给相同核，导致某些核每次 pass 多算一个块 | 在不同 pass 间交替分配尾块，实现全局负载均衡（例如 pass 1 尾块给核 1–5，pass 2 尾块给核 6–10） |
| `ascendc-operator-performance-optim/references/pipeline-prof.md` | Double Buffer 未开启 | CopyIn / Compute / CopyOut 三阶段完全串行，Vector 利用率仅约 33% | `InitBuffer` buffer 数为 1，不同切片间无法重叠执行 | 将 `InitBuffer` buffer 数改为 2；确认循环次数 ≥ 2 且搬运耗时相对计算不可忽略 |
| `ascendc-operator-performance-optim/references/pipeline-prof.md` | MIX 模式同步 Iterate 开销大 | Matmul MIX 场景下每次迭代均有 AIC/AIV 同步消息，流水效率低 | 使用了 `Iterate<true>()`（同步模式），每次迭代发一条同步消息，开销大 | 改用 `Iterate<false>()`（异步模式），仅首次发消息，后续迭代无需 AIC/AIV 同步 |
| `ascendc-operator-performance-optim/references/memory-prof.md` | UB Buffer 未融合 | 连续 Vector 运算（如 Exp → Abs）中间结果经 GM 往返，搬运次数为 2n | 每次运算都 CopyOut 到 GM 再 CopyIn，造成冗余搬运 | 将中间结果保留在 UB 内链式计算，搬运次数从 2n 降为 2（一次 CopyIn + 一次 CopyOut） |
| `ascendc-operator-performance-optim/references/memory-prof.md` | L0C 未原地累加矩阵乘 | `A1*B1 + A2*B2 + ...` 场景中每次 Mmad 结果都先搬出 GM 再在 UB 求和 | 逐次搬出导致 CO1→GM→UB→Add→GM 的冗长路径 | 在 Mmad 中设 `cmatrixInitVal=false` 使结果在 CO1（L0C）中原地累加，最后一次再搬出 |
| `ascendc-operator-performance-optim/references/memory-prof.md` | 小矩阵未长驻 L1 | 矩阵乘中每次迭代都重新加载左右矩阵，L1 利用率低 | L1 能容纳较小矩阵但每次仍重新搬入，增加无效 HBM 访问 | 将较小矩阵一次加载后常驻 L1，仅循环搬运较大矩阵；2 个切片时 HBM 搬运次数从 8 降为 3 |
| `ascendc-operator-performance-optim/references/memory-prof.md` | bias 未用 BT Buffer 融合（分离架构） | Matmul 后 bias 在 UB 中单独做 Add，路径为 CO1→GM→UB→Add→GM | 分离架构未利用 BT Buffer，bias 加法需额外搬运和计算 | 将 bias 存入 BT Buffer（C2），在 `Mmad` 中一步融合 bias 加法，消除 UB 中的单独 Add |
| `ascendc-operator-performance-optim/references/memory-prof.md` | 量化参数未用 FP Buffer 融合（分离架构） | 量化参数在 UB 中单独计算，路径为 CO1→GM→UB→量化→GM | 分离架构未利用 FP Buffer，量化计算需额外 GM 往返 | 将量化参数存入 FP Buffer（C2PIPE2GM），通过 `Fixpipe` 在搬出路径上随路量化 |
| `rl-consistency-analysis/SKILL.md` | 训推一致性差异（结构性假阳性） | 模块映射比对存在大量 mismatch，但怀疑部分为结构差异而非真正数值不一致 | fused / unfused 实现导致的结构性假阳性（如 fused QKV 边界），并非真正的数值发散 | 运行 `run_root_cause_analysis.py`，过滤 `structural_false_positive`；优先处理 `module_priority_rank` 1、2 的候选项 |
| `rl-consistency-analysis/SKILL.md` | 训推一致性差异（多余激活函数） | 对齐边界与不匹配模块之间，rollout 侧存在未匹配的中间激活模块（`act_fn` / `silu` / `swiglu` / `gelu` 等） | rollout 侧有多余的激活函数或融合算子，train 侧缺失，导致该层输出发散 | 检查 train 与 rollout 在该模块的激活函数实现/配置是否一致；典型案例：rollout 使用 `swiglu`，train 使用 `megatrongelu`，bridge 参数未正确激活 |
| `rl-consistency-analysis/SKILL.md` | 训推一致性差异（参数/权重不一致） | 输入对齐但输出不一致，且两侧参数（权重）不同 | `parameter_or_checkpoint_issue`：checkpoint 加载不一致或权重同步问题 | 对比 train 与 rollout 加载的 checkpoint 路径和权重值，确认权重来源一致 |
| `rl-consistency-analysis/SKILL.md` | 训推一致性差异（模块内实现差异） | 输入和参数均对齐，但模块输出不一致 | `in_module_impl_difference`：模块内部算法实现或数值精度处理不同 | 对比 train 与 rollout 在该模块的具体实现代码（算子选择、数值类型、计算顺序等） |
| `rl-consistency-analysis/SKILL.md` | 训推一致性差异（上游传播） | 当前模块参数对齐，但输入已发散 | `upstream_propagation`：真正根因在更上游模块，差异逐层传播至此 | 回溯上游模块，找到第一个发生 divergence 的位置作为真正根因；使用 `output_5_root_cause_report.json` 中的边界信息定位 |
