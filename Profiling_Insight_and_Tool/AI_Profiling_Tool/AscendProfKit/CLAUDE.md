# Ascend NPU 性能与精度分析专家能力

本目录包含 Ascend NPU 性能分析、算子调优、精度分析的专家知识库。当用户提出以下类型问题时，读取对应的知识库再展开分析。

---

## 通用报告规范（所有分析均适用）

**每次输出分析报告前，必须先读取以下文件，并严格遵守其中的报告输出规则：**

- `AscendProfKit/skills/profiling-workflow/SKILL.md`

## 性能 Profiling 分析

**适用问题**：Profiling DB 文件分析、快慢卡诊断、MFU 计算、通信瓶颈、算子热点、下发调度等

读取以下文件获取专家指导：

- `AscendProfKit/resources/configs/default/prompts/agents/Hermes.md`（角色定义与工作流）
- `AscendProfKit/skills/mindstudio_profiler_data_check/SKILL.md`（先校验数据完整性）
- `AscendProfKit/skills/dataset-source-identifier/SKILL.md`（识别并记录落盘数据来源/模型/用途，含识别依据，无依据留空不猜）
- `AscendProfKit/skills/ascend-profiler-db-explorer/SKILL.md`（Profiling DB SQL 查询）
- `AscendProfKit/skills/cluster-fast-slow-rank-detector/SKILL.md`（快慢卡诊断）
- `AscendProfKit/skills/timeline-swimlane-analyzer/SKILL.md`（Timeline 泳道时序结构分析：关键路径、计算-通信重叠、空挡/间隔、PP bubble、step 抖动、prefill/decode）
- `AscendProfKit/skills/op-mfu-calculator/SKILL.md`（MFU 计算）
- `AscendProfKit/skills/msprof-analyze-cli/SKILL.md`（集群慢卡/慢链路综合分析与专家建议，含工具选择决策流程）
- `AscendProfKit/skills/msinsight-view-selector/SKILL.md`（为每个诊断结果推荐 MindStudio Insight 可视化视图，必须为每个问题点附上）

---

## 算子性能调优

**适用问题**：算子性能不达预期、AscendC 算子优化、msOpProf 分析报告

读取以下文件获取专家指导：

- `AscendProfKit/resources/configs/default/prompts/agents/Icarus.md`（角色定义与工作流）
- `AscendProfKit/skills/msot-msopprof-operator-profiler/SKILL.md`（msOpProf 深度分析）
- `AscendProfKit/skills/ascendc-operator-performance-optim/SKILL.md`（端到端调优闭环）
- `AscendProfKit/skills/msinsight-view-selector/SKILL.md`（为每个诊断结果推荐 MindStudio Insight 可视化视图，必须为每个问题点附上）

---

## 精度分析与训推一致性

**适用问题**：训推不一致、loss/gnorm NaN、精度发散根因分析

读取以下文件获取专家指导：

- `AscendProfKit/resources/configs/default/prompts/agents/Accuracy.md`（角色定义与工作流）
- `AscendProfKit/skills/rl-consistency-analysis/SKILL.md`（训推一致性根因分析）
- `AscendProfKit/skills/nan-overflow-detection/SKILL.md`（多卡分布式训练 NaN/Inf 溢出检测与根因追溯）
