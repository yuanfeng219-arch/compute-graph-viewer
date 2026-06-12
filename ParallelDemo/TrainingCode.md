# 开源真实模型训练代码参考（并行训练 / 算子切分 / MoE）

> 需求对照：覆盖数据并行 / 模型(张量)并行 / 流水线并行中至少 2 种；展示算子/函数如何分到不同模型层、不同层如何在多张训练卡上布置、数据集如何切分；尽量包含 MoE。

---

## Demo 方案：并行训练可视化（三栏交互）

**目标**：左侧控制器（勾选并行模式、卡数、卡型、选用哪些层；滑动条设 batchsize / step / 学习率）→ 中间模型代码随配置改动 → 右侧从上到下可视化每张训练卡里有哪些层、哪些算子。

**前提判断**：这是**可视化/演示型**前端（控制器→代码→卡片图），不必真在多张物理 GPU 上跑训练。选库标准不是"性能强"，而是"**代码够短、层到卡的映射够显式**"——这样中间面板能直接展示代码、右侧面板能从代码结构推出每张卡装了什么。按这个标准 Megatron/DeepSpeed 不合适（代码太重，中间面板放进去看不懂）。

**结论：picotron 作主干，torchtitan 作 MoE 补充。**

### 推荐一：huggingface/picotron（主干）
- **控制器 ↔ 代码**：每种并行是一个独立 <300 行文件（`data_parallel.py` / `tensor_parallel.py` / `pipeline_parallel.py` / `context_parallel.py`）。勾选"开启 TP / 关闭 PP"时中间面板就切换/高亮对应文件——天然的"配置驱动代码"映射。
- **右侧卡片可视化的数据来源**：picotron 里有等价于 Megatron `parallel_state` 的 rank 分组逻辑——给定 `dp/tp/pp` 大小，算出每个 rank（卡）属于哪个组、负责哪段层。这是右侧"每张卡里有什么层"的唯一真相源。
- **层/算子切分够直观**：`tensor_parallel.py` 标了 Linear 怎么按列/行劈到不同卡、在哪做 all-reduce；`pipeline_parallel.py` 标了 transformer 层段怎么分到不同卡。
- **数据集切分**：dataloader 里按 DP rank 的 sharding 一目了然，对应控制器的"卡数 / batchsize"。
- 缺点：主线没有 MoE。

### 推荐二：pytorch/torchtitan（专补 MoE + 现代放置语义）
- 含 Llama4 / DeepSeek-V3 的 **MoE + Expert Parallel**：router、专家 all-to-all 分发、专家怎么分到不同卡——直接喂给右侧"卡N: Expert 3,4"的可视化。
- 用 `DeviceMesh` + `DTensor` 的 **placement（Shard / Replicate）** 描述"这一层在这组卡上怎么摆"。placement 概念和右侧面板几乎一一对应：一个 DTensor 的 placement 就是"这张卡持有张量的哪一片"，右侧的图基本是 placement 的直接渲染。

### 三个面板 ↔ 两个库的对应

| 面板 | 数据/代码来源 |
|---|---|
| 左·控制器（并行模式、卡数、卡型、层选择、batch/step/lr） | 映射成一组配置：`{dp, tp, pp, ep, layers[], batch, lr, ...}` |
| 中·模型代码（随配置变） | 主要展示 **picotron** 对应文件；勾 MoE 时切到 **torchtitan** 的 MoE 模块 |
| 右·卡片可视化（每卡的层/算子） | 由配置经 **picotron 的 rank 分组逻辑** / **torchtitan 的 DTensor placement** 计算出"卡→层→算子分片"再渲染 |

### 落地建议
demo 不必真跑分布式。最实际做法：把 picotron / torchtitan 的关键文件作为**静态素材**放进前端，再写一个"placement 计算器"——输入控制器配置，输出"每张卡持有哪些层、每层是整块还是行/列切片、MoE 专家怎么分"，右侧据此画图。两个库提供的是**正确的切分规则和可展示的真实代码**，前端负责把规则可视化。

---

## 最推荐：先看这两个（代码量小、可读性高）

### 1. huggingface/picotron —— 最适合入门精读
https://github.com/huggingface/picotron

NanoGPT 风格的「4D 并行」教学框架，核心文件每个都 **300 行以内**，是目前对照需求最理想的样本：

- **多种并行（满足≥2种，实际是4种）**：仓库直接把每种并行拆成独立文件 —— `data_parallel.py`、`tensor_parallel.py`、`pipeline_parallel.py`、`context_parallel.py`。可以单独读每一个，不会被工程细节淹没。
- **算子/层怎么切到不同卡**：`tensor_parallel.py` 里能清楚看到对 `nn.Linear`（全连接/注意力投影）按列切（ColumnParallel）和按行切（RowParallel），以及前向/反向时的 all-reduce 通信点 —— 正是「算子如何分到层、层如何在卡上摆放」的最直观例子。
- **层在卡上的布置**：`pipeline_parallel.py` 展示了把 transformer 的不同层段（layer stages）分到不同卡，以及 1F1B 流水线调度。
- **数据集切分**：`data_parallel.py` + 配套的 dataloader 里有按 DP rank 做的 sharding（DistributedSampler 思路）。

缺点：picotron 主线**没有 MoE**。MoE 要看下面两个。

### 2. pytorch/torchtitan —— PyTorch 官方、现代且含 MoE
https://github.com/pytorch/torchtitan

PyTorch 原生的训练平台，强调「对模型代码改动最小地叠加多维并行」：

- **并行**：FSDP/HSDP（数据并行）、Tensor Parallel、Pipeline Parallel、Context Parallel 都有，且是用 PyTorch 最新的 `DeviceMesh` + `DTensor` 实现，是看「现代写法」的最佳参考。
- **MoE**：torchtitan 已经包含 Llama4 / DeepSeek-V3 等带 MoE 的模型，含 **Expert Parallel（专家并行）**，能看到 MoE 层的 router、专家分发（all-to-all）以及专家如何分到不同卡。
- **层放置 / 数据切分**：`parallelize_*.py` 里是「给定一个 model，如何把它的各层 apply 到 DeviceMesh 上」的逻辑，数据侧用 DataLoader + DP sharding。

## 工业级（功能全，但代码庞大，适合查而非通读）

### 3. NVIDIA/Megatron-LM / Megatron-Core
https://github.com/NVIDIA/Megatron-LM

最权威的参考，**五维混合并行**：Tensor / Pipeline / Data / Context / **Expert(MoE) Parallelism** 全都有。对照需求：

- 算子切分：`megatron/core/tensor_parallel/layers.py` —— ColumnParallelLinear / RowParallelLinear 的实现细节（全连接、注意力投影怎么切）。
  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
- 卡的分组/层放置：`megatron/core/parallel_state.py` —— 各种并行通信组（process group）如何建立，决定「哪张卡负责哪段层、哪些专家」。
  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
- MoE：`megatron/core/transformer/moe/` 下有 router、token dispatch、专家并行。

### 4. microsoft/DeepSpeed + DeepSpeed-MoE
数据并行（含 ZeRO）、tensor-slicing、pipeline、expert parallelism 都支持，论文 DeepSpeed-MoE 配合代码读。
- 论文：https://arxiv.org/pdf/2201.05596

## 专门看 MoE 实现的小仓库

- **FastMoE**：研究向，专门演示 MoE 在数据并行 + 专家（模型）并行下怎么训练，代码比 Megatron 小很多。
- **junfanz1/MoE-Mixture-of-Experts-in-PyTorch**：一个单卡版 + 一个多卡分布式版，专门展示 MoE 核心原理（router、专家分发），适合先建立 MoE 直觉。
  https://github.com/junfanz1/MoE-Mixture-of-Experts-in-PyTorch

---

## 阅读路线建议

| 需求 | 首选看哪里 |
|---|---|
| 想完整通读、真正搞懂每种并行 | **picotron**（4种并行各一个文件，≤300行） |
| 想看算子/Linear 怎么按行列切到卡 | picotron `tensor_parallel.py` → Megatron `layers.py` |
| 想看层怎么分段摆到不同卡（流水线） | picotron `pipeline_parallel.py` |
| 想看数据集切分 | picotron/torchtitan 的 dataloader + DP sampler |
| 想看 MoE + 专家并行 | **torchtitan**（现代、易读）或 Megatron `moe/` |

**最高效的组合**：先用 `picotron` 把 DP/TP/PP/数据切分彻底读懂（最短最干净），再去 `torchtitan` 看 MoE/专家并行的现代写法，最后用 `Megatron-LM` 当「字典」查工业级实现细节。

## 参考来源

- huggingface/picotron — https://github.com/huggingface/picotron
- pytorch/torchtitan — https://github.com/pytorch/torchtitan
- NVIDIA/Megatron-LM — https://github.com/nvidia/megatron-lm
  - tensor_parallel/layers.py — https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
  - parallel_state.py — https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
- DeepSpeed-MoE 论文 — https://arxiv.org/pdf/2201.05596
- junfanz1/MoE-Mixture-of-Experts-in-PyTorch — https://github.com/junfanz1/MoE-Mixture-of-Experts-in-PyTorch
- Hybrid Tensor-Expert-Data Parallelism (MoE) 论文 — https://arxiv.org/pdf/2303.06318
