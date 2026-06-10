# AscendProfKit 变更记录

## 2026-06-10 规则 7 指标看板：HTML 注释块 → 可见 Markdown 表格

### 背景

时序结构指标原以 `<!-- METRICS {json} -->` HTML 注释写在报告末尾，对人不可见。这些指标本身有价值，没必要藏起来。

本次将规则 7 改为：在报告末尾以**可见的 `###` 表格**「时序结构指标看板」呈现（指标 / 值 / 状态 / 说明），首列用标准中文指标名（前端按名映射看板 key）。仍保持"只列测得的指标、未采集不列"。MindStudioNext `parseReport` 优先解析该可见表格，旧报告的注释块仍兼容回退。

### 修改文件

- `skills/profiling-workflow/SKILL.md`：重写规则 7（注释块 → 可见表格 + 中文指标名↔key 映射表）。
- 配套：`MindStudioNext` 新增 `parseMetricsTable`（按中文指标名映射 key），优先于注释块解析；`level2_profiling_analysis_20260610/report.md` 的 METRICS 注释改为可见表格。

## 2026-06-10 新增"落盘数据来源识别"能力（dataset-source-identifier）

### 背景

报告未系统记录"被分析的这批落盘数据是什么"——来源类型、是否 LLM 训练、模型/用途、大小、下载链接。此前模型/用途多靠经验臆测（如仅凭并行配置猜 "Qwen2-7B"），既不可证伪，也无法被前端稳定消费。

本次新增能力：分析落盘数据时，**必须基于落盘内的事实证据**（tokenizer vocab、算子签名、InputShapes、framework 标记等）识别数据来源，参考 `data/analysis_report.html` 顶部总结表的形态，把结论 + **识别依据**写入 `report.md` 第 5 章固定块。**硬性原则：没有依据识别的字段一律留空，严禁猜测。**

### 修改文件

- 新增 `skills/dataset-source-identifier/SKILL.md`：识别信号、留空原则、report.md 输出块格式。
- `skills/profiling-workflow/SKILL.md`：新增"规则 8 — 数据来源与落盘信息块"，约定固定字段/格式，前端按此解析。
- `CLAUDE.md`：性能 Profiling 分析下引用新 skill。
- 配套：`MindStudioNext` 的 `parseReport` 解析该块，自动填充"任务信息"页"落盘文件信息"卡片（无依据字段渲染为"—（无识别依据，留空）"，新增"识别依据"行）。

## 2026-06-05 报告生成后追加输出"分析过程记录"（Analysis Process.md）

### 背景

原工作流只产出结论性的 `report.md`，缺少对"分析是怎么一步步做出来的、花了多少代价"的记录，无法复盘分析路径、估算成本，也缺少 Demo 演示素材。

本次新增规则：完整分析工作流跑完、`report.md` 定稿后，必须在同级报告目录再产出一份 `Analysis Process.md`，记录本次分析的步骤、耗时（真实墙钟或相对占比）、Token 消耗、用户确认节点，以及对最终报告的浓缩摘要。格式参考 `MindStudioNext/Analysis Process.md` 样例，但阶段划分须贴合本次真实分析路径。

### 修改文件

- `skills/profiling-workflow/SKILL.md`：
  - 新增**规则 6**，定义 `Analysis Process.md` 的触发 / 豁免 / 时机（report.md 写完后）、落盘位置（与 report.md 同级）、四章骨架（Token 消耗 / TodoList 含耗时占比 / 各阶段在做什么 / 诊断摘要）及"如实记录不得虚构"原则。
  - 规则 3 推荐子结构示例补入 `Analysis Process.md` 一行。

---

## 2026-06-05 报告目录命名加被分析对象标识词前缀

### 背景

分析报告目录原命名为 `ascend_analysis_YYYYMMDD`，所有分析共用同一前缀，多对象分析并存时难以一眼区分各报告对应哪份输入数据。

本次将命名格式改为 **`<被分析对象标识词>_profiling_analysis_YYYYMMDD/`**：取本次分析的输入文件夹 / 文件名作前缀。例如分析文件夹 `a/` → 报告目录 `a_profiling_analysis_20260605/`。

### 命名规则

- **前缀 = 被分析对象的标识词**：输入文件夹 / 文件名。
- **名字过长只取标识词**：截取最具区分度的一段作前缀，去掉时间戳、`rankN`、`Ncard`、`ascend_pt` 等冗余尾巴；标识词 ~20 字符内，空格 / 特殊字符替换为 `_`。例：`MultiProfLevel2MemoryUB_node1_20260602_rank0_8card/` → `MultiProfLevel2MemoryUB_profiling_analysis_20260605/`。
- **多次分析**：同对象靠日期区分，同日追加 `_2` / `_3` 序号。
- 单文件输出同样带前缀：`a_profiling_analysis_20260605.md`。

### 修改文件

- `skills/profiling-workflow/SKILL.md`：规则 3 重写"新建一级目录"命名段，给出格式、标识词截取、多次分析区分规则；同步更新规则 1 第 5 章"输出位置"示例与推荐子结构示例。
- `skills/cluster-fast-slow-rank-detector/SKILL.md`、`skills/msprof-analyze-cli/SKILL.md`：命令示例 `-o` 输出目录、推荐目录文案改用新格式（按 `-d` 输入名派生前缀，如 `./cluster_data_profiling_analysis_YYYYMMDD/`）。
- `skills/ascendc-operator-performance-optim/SKILL.md`、`skills/msot-msopprof-operator-profiler/SKILL.md`：算子调优产物目录改为以 `<op_name>` 为标识词前缀。
- `skills/nan-overflow-detection/SKILL.md`：脚本输出路径示例改用 `<标识词>_profiling_analysis_YYYYMMDD/nan_overflow/`。

---

## 2026-06-05 问题举证文件随报告自包含落盘

### 背景

性能 / 算子分析报告第 3 章每个问题点都带"问题举证视图"字段，告诉用户去 MindStudio Insight 哪个视图、载入落盘数据里的哪个文件来复现问题现象（如 `trace_view.json`、`kernel_details.csv`、`communication_matrix.json`、`*.bin` 等）。但这些文件仍散落在庞大的只读原始落盘目录里，用户拿到报告后还要回原目录逐个翻找，报告不自包含。

本次新增能力：出报告时，把每个问题点举证所需的落盘文件**复制一份到报告输出目录的 `evidence/` 子目录**，并在视图字段中同时标注"副本相对路径 + 原始落盘来源"，使报告目录可整体打包带走、直接复现视图。

---

### 修改文件

#### `skills/profiling-workflow/SKILL.md`

- **新增规则 5：问题举证视图所需文件必须复制到报告目录**
  - 触发 / 豁免与规则 4 一致（Hermes / Icarus 触发；Accuracy、数据校验失败豁免）
  - 收集第 3 章所有"所需文件"，按源路径**去重**后复制到 `evidence/`
  - **保留可区分来源的相对结构**（保留 rank / 节点这一层），避免跨 rank 同名文件互相覆盖
  - **一律复制，不设大小阈值**：`trace_view.json` / `profiler.db` / `*.bin` 等大文件也照常复制
  - 复制方向永远是 落盘目录 → 报告目录，承规则 3 的只读保护
  - 第 5 章"数据与方法"附录追加"举证文件清单"表（副本路径 / 原始来源 / 引用问题点 / 大小）
  - 失败处理：源文件缺失不阻塞报告，字段末尾标注 `⚠ 源文件未找到，未复制`
- **规则 3**：推荐子目录结构新增 `evidence/`
- **规则 4**：三要素中"所需文件"改为"副本相对路径（源：原始落盘路径）"两者都写，并指向规则 5
- **规则 1（示例 3.1）**：问题举证视图示例改用 `evidence/...`（源：...）双路径写法

#### `skills/msinsight-view-selector/SKILL.md`

- **字段格式约束**：三要素中"所需文件"要求复制进 `evidence/` 并写副本 + 原始来源两路径，补充反例 / 正例
- **方式 A / 方式 B / 多视图组合**示例统一改为 `evidence/...`（源：...）写法，并说明多视图引用同一文件只复制一份

---

### 行为变更对照

| | 改造前 | 改造后 |
|---|---|---|
| 举证文件位置 | 仅在只读原始落盘目录 | 复制一份至报告目录 `evidence/` |
| 视图字段"所需文件" | 仅写文件名 / 原始路径 | 副本相对路径 + 原始落盘来源（两者都写） |
| 大文件（DB / bin / trace） | — | 一律复制，不设阈值 |
| 报告可移植性 | 依赖原始落盘目录 | 报告目录自包含，可整体打包 |
| 第 5 章附录 | 无 | 新增"举证文件清单"表 |

---

<!-- 后续变更在此处向上追加 -->
