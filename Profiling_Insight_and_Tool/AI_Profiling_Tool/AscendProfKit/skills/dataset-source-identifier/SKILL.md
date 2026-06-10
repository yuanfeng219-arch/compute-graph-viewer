# 落盘数据来源识别（dataset-source-identifier）

> **职责**：在分析任何 Profiling 落盘数据时，**识别并记录**这批数据"是什么"——数据目录、来源类型、是否 LLM 训练、模型/用途、落盘大小、来源链接，以及**识别依据**。结论必须有证据支撑；**没有依据识别的项一律留空，严禁臆测/猜测**。识别结果写入 `report.md` 第 5 章的"数据来源与落盘信息"块，供人阅读、也供 MindStudioNext 自动填充"落盘文件信息"卡片。

参考样例：`data/analysis_report.html` 顶部"总结"表格 + 每个数据目录的"关键证据"卡片，就是本 skill 期望产出的内容形态。

---

## 触发与豁免

- **触发**：所有 Hermes（性能 Profiling）、Icarus（算子调优）报告——只要分析的是一份落盘数据目录/文件，就必须识别并记录其来源。
- **豁免**：纯 SQL 速查、未形成正式 report.md 的场景。

---

## 一、要记录的字段（对齐 analysis_report.html 总结表）

| 字段 | 含义 | 取值约定 |
|---|---|---|
| 数据目录 | 被分析的落盘目录/文件名 | 如 `level2/`、`verl/` |
| 来源 | 数据的采集场景类型 | 如"分布式训练 Profiling"/"推理 Profiling"/"算子 Simulator"/"RL 训练 Profiling" |
| 是否 LLM 训练 | 是 / 否 / 不适用 | 训练且为大语言模型→是；非 LLM（推荐/CTR/CV 等）→否；单算子/非模型→不适用 |
| 模型 / 用途 | 识别出的模型或任务用途 | 有依据才写具体名；**无足够依据时留空**，或只写能确证的粒度（如"Qwen 系列"而非"Qwen2-7B"） |
| 落盘大小 | 数据目录/举证副本大小 | 如 `64.9 MB`、`~47 MB（evidence 副本）` |
| 来源链接 | 公开数据集下载链接（若已知） | Markdown 链接；未知则留空 |

外加一条 **识别依据**：逐条列出每个结论是从哪些证据得出的；某字段无依据则在依据里写明"X 无确证依据，留空"。

---

## 二、识别信号（证据来源，按可信度优先）

**只用从落盘数据里能直接读到的事实**（算子签名、InputShapes、字段、目录结构、framework 标记），不要靠"看起来像"。

1. **词表大小（tokenizer vocab）** — 最强的模型家族指纹。
   - `vocab=151936` → Qwen / Qwen2 系列专属 tokenizer。
   - `vocab=152064` → Qwen2.5。`vocab=128256` → LLaMA-3。`vocab=32000` → LLaMA / LLaMA-2。
   - 出现在 lm_head 投影 / loss 算子的 InputShapes 词表维（如 `MatMulV3 ...;151936,1024`、`Cast/Exp 4096,1,151936`）。
2. **hidden_size / 层数** — 区分同家族不同规模。从 MatMul/Attention 的 InputShapes 反推 hidden；从 PP stage 层数反推总层数。**注意**：lm_head 形如 `tokens,hidden;vocab,hidden`，被切分（TP/PP）时单卡 shape ≠ 全模型 shape，反推规模需谨慎，拿不准就**只报家族、不报规模**。
3. **架构算子签名** — 判定"是不是 Transformer LLM"。
   - `RmsNorm/RmsNormGrad` + `SwiGlu/SwiGluGrad` + `RotaryPositionEmbedding(RoPE)` + `FlashAttentionScore(/Grad)` → 典型现代 LLM（Qwen/LLaMA 系）。
   - 含 `...Grad` 反向算子 / `ApplyAdamWV2` 优化器算子 → **训练**（非纯推理）。
   - 含 `PagedAttention` + 同时有训练算子 → RL（generate+train 双阶段，如 verl）。
4. **非 LLM 的反指纹**。
   - 多张独立 Embedding 表、vocab 大小分散（百万级且各不相同）、hidden_dim 极小（如 16）、缺 RMSNorm/RoPE/SwiGlu/FlashAttention → 推荐系统 / CTR 多特征 Embedding 模型（非 LLM）。
   - 仅 1~少数 cube/vec core 的微观行为、报告标题为单算子（如"MatmulLeakyRelu simulator"）、文件为 `visualize_data.bin` → AscendC 单算子调优（不适用"模型"）。
5. **框架/环境标记** — 佐证用途。`torch.op_mark`/`op_range`/`memory_usage` → PyTorch 训练采集；`ray::WorkerDict` → Ray（RL 常见）；算子名含 `bf16` → 精度；CANN/torch_npu 版本。
6. **目录结构/规模** — `node1_*`+`ubuntu2204_*` 双节点 → 多机；`rank_{0..N}` 数 → 卡数；`.db` vs `.csv` → 采集导出格式。

---

## 三、硬性原则：有依据才写，没依据就留空

- **禁止猜测**：仅凭"常见""可能""一般是"不能下结论。`vocab=151936` 能确证"Qwen 系列"，但若 hidden/层数被并行切分、无法反推全模型规模，就**不要**写"7B/14B"，把规模维度留空或只写家族。
- **留空 ≠ 失败**：某字段无确证依据时，"模型/用途"留空、并在"识别依据"里写"具体规模无依据，留空"。这正是预期行为（对齐前端"没数据就空着"）。
- **可证伪优先**：每个非空字段都要能在"识别依据"里指到具体证据（算子名/shape/字段/文件）。

---

## 四、写入 report.md（第 5 章固定块格式）

在第 5 章"数据与方法"中，**`使用的 Skills` 之后、`举证文件清单` 附近**追加如下块（字段顺序固定，便于前端解析）：

```markdown
- **数据来源与落盘信息**（落盘文件信息卡片；无确切识别依据的项留空，不臆测）：
  - 数据目录：`level2/`
  - 来源：分布式训练 Profiling
  - 是否 LLM 训练：是
  - 模型 / 用途：Qwen 系列 LLM（vocab=151936）
  - 落盘大小：~47 MB（evidence 举证副本）
  - 来源链接：[level2.rar](https://gitcode.com/.../level2.rar)
  - 识别依据：vocab=151936 命中 Qwen 系列 tokenizer；evidence 中含 FlashAttentionScore/Grad、RmsNormGrad、ApplyAdamWV2 → 判为 LLM 训练。具体参数规模（hidden/层数）因被 PP/TP 切分无法从单卡 shape 反推，留空不写。
```

格式约定（**前端按此解析，务必照写**）：

- 块标题固定为 `**数据来源与落盘信息**`，下挂 7 个二级 `  - <字段>：<值>` 子项，字段名一字不差：`数据目录`/`来源`/`是否 LLM 训练`/`模型 / 用途`/`落盘大小`/`来源链接`/`识别依据`。
- `是否 LLM 训练` 取值只用 `是`/`否`/`不适用`。
- `来源链接` 用 Markdown 链接 `[文本](url)`；未知则把值留空（冒号后不写内容）。
- **留空的字段**：保留该行、冒号后为空（或整行省略）；前端渲染为"—（无识别依据，留空）"，不显示任何猜测值。

写入第 5 章"使用的 Skills"列表：

```
- `dataset-source-identifier`：识别并记录落盘数据来源/模型/用途（含识别依据，无依据留空不猜）
```

---

## 五、与其它 skill 的关系

- 先于本 skill 跑 `mindstudio_profiler_data_check`（校验数据完整性、识别数据类型：框架 profiler / msprof / DB / bin）；其判定的数据类型是本 skill"来源"字段的输入。
- 模型规模/利用率类结论若需 hidden/FLOPs，配合 `op-mfu-calculator`；但**规模反推不确定时仍以"留空"为准**，不要为了填满字段而编规模。
