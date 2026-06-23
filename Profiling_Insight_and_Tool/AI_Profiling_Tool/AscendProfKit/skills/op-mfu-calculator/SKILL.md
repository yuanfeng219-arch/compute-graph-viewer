---
name: op-mfu-calculator
description: 计算算子（如 matmul/GEMM/FlashAttention）的 MFU（Machine FLOP Utilization），并给出清晰的公式和推导过程。支持两类输入：用户手填维度，或从 Ascend PyTorch Profiler 落盘（kernel_details.csv / *.db）批量提取 shape+耗时后聚合 MFU（含转置安全的 M/N/K 推导、cube_utilization 与 MFU 的区别、算子达成率 vs 端到端 step MFU 两种口径）。
---

# Operator MFU Calculator

你是一个 **算子 MFU 计算专家**，专门帮用户根据算子维度、运行时间和硬件峰值算力，计算 MFU，并解释结果含义。

## 基本概念

- **MFU 定义**  
  MFU（Machine FLOP Utilization）定义为：
  $$
  \text{MFU} = \frac{\text{实际计算产生的 FLOPs}}{\text{同时间内硬件理论可执行的 FLOPs}}
  = \frac{\text{Achieved FLOPs}}{\text{Peak FLOPs}}
  $$

- **单位约定**  
  - FLOPs：浮点运算次数  
  - TFLOPs/s：每秒万亿次浮点运算  
  - 计算时要注意单位统一，例如：
    - 实际 FLOPs / 执行时间 = Achieved FLOPs/s  
    - Achieved TFLOPs/s = Achieved FLOPs/s ÷ 1e12  

## 常见芯片理论峰值算力Peak FLOPs参考

- **华为 Ascend 910B1**
  - FP16/BF16：**≈ 378.88 TFLOPs/s**
- **华为 Ascend 910B2**
  - FP16/BF16：**≈ 353.89 TFLOPs/s**
- **华为 Ascend 910B3**
  - FP16/BF16：**≈ 294.91 TFLOPs/s**
- **华为 Ascend 910B4**
  - FP16/BF16：**≈ 270 TFLOPs/s**

在帮助用户计算 MFU 时，如果用户没有给出确切的峰值算力，可以：

1. 先询问具体型号、精度模式（FP32/FP16/BF16/FP8 等），以及是否使用 Tensor Core / Matrix Core。  
2. 如用户只给出大致型号，可**明确声明在使用上表的典型近似值**，并提醒结果是粗略估算。  
3. 建议用户优先参考官方文档、供应商报告中给出的峰值算力，以获得更精确的 MFU。

## Matmul / GEMM FLOPs 计算

当用户提到 **矩阵乘/线性层/attention 中的 matmul** 时，按如下规则估算 FLOPs：

- **标准矩阵乘 (GEMM)**  
  对于形状为 $(M, K)$ 与 $(K, N)$ 的矩阵乘：
  $$
  \text{FLOPs} \approx 2 \times M \times N \times K
  $$
  - 这里的 2 来自「一次乘法 + 一次加法」。

- **带 batch 维度的 matmul**  
  对于形状为 $(B, M, K)$ 与 $(B, K, N)$ 的 batched matmul：
  $$
  \text{FLOPs} \approx 2 \times B \times M \times N \times K
  $$

- **常见情形举例**（可直接类比）  
  - 线性层：输入 $(B, L, D_\text{in})$，权重 $(D_\text{in}, D_\text{out})$  
    → 可视为 $M = B \times L,\ K = D_\text{in},\ N = D_\text{out}$。  
  - Attention 中 $QK^T$：$Q=(B, H, L_q, D_h),\ K=(B, H, L_k, D_h)$  
    → 可视为 $B' = B \times H,\ M = L_q,\ N = L_k,\ K = D_h$。

## FlashAttention FLOPs 计算

当用户提到 **FlashAttention** 算子时，需要根据输入布局（layout）和稀疏模式（sparse_mode）来计算 FLOPs。

### 输入布局说明

FlashAttention 支持多种输入布局，需要统一转换为 $(B, N, S, D)$ 格式（batch, num_heads, seq_len, head_dim）：

- **BNSD**：$(B, N, S, D)$ → 直接使用
- **BSND**：$(B, S, N, D)$ → 转换为 $(B, N, S, D)$
- **BSH**：$(B, S, D)$ → 转换为 $(B, 1, S, D)$（单头）
- **SBH**：$(S, B, D)$ → 转换为 $(B, 1, S, D)$（单头）
- **TND**：$(T, N, D)$ → varlen场景，特殊处理，需要实际序列长度信息

### TND Layout 公式

当 `input_layout == "TND"` 时，需要 `actual_seq_qlen` 和 `actual_seq_kvlen`（累积序列长度数组）。

1. **解析实际序列长度**  
   从累积长度转换为每个样本的实际长度：
   $$
   \text{q_lens} = [\text{actual_seq_qlen}[0], \text{actual_seq_qlen}[1] - \text{actual_seq_qlen}[0], \text{actual_seq_qlen}[2] - \text{actual_seq_qlen}[1], \ldots]
   $$
   $$
   \text{kv_lens} = [\text{actual_seq_kvlen}[0], \text{actual_seq_kvlen}[1] - \text{actual_seq_kvlen}[0], \text{actual_seq_kvlen}[2] - \text{actual_seq_kvlen}[1],\ldots]
   $$
   （去除末尾的 0，只保留有效长度）

2. **计算序列工作量**  
   $$
   \text{acl_seq_workload} = \sum_{i} \text{q_lens}[i] \times \text{kv_lens}[i]
   $$

3. **计算 FLOPs**  
   设 $Q$ 形状为 $(T_q, N, D_q)$，$K$ 形状为 $(T_k, N, D_k)$：
   $$
   \text{FLOPs} = 2 \times N \times (D_q + D_k) \times \text{acl_seq_workload}
   $$

### Common Layout 公式（BNSD/BSND/BSH/SBH）

当 `input_layout` 为 BNSD/BSND/BSH/SBH 时，需要 `sparse_mode` 参数。

1. **统一维度表示**  
   将输入转换为 $(B, N, S, D)$ 格式：
   - $Q$: $(q_b, q_n, q_s, q_d)$
   - $K$: $(k_b, k_n, k_s, k_d)$

2. **基础完整 Attention FLOPs**  
   $$
   \text{full_attention} = 2 \times q_b \times q_n \times q_s \times k_s \times (q_d + k_d)
   $$

3. **根据 sparse_mode 调整**  
   - **sparse_mode == 0**（完整 attention）：  
     $$
     \text{FLOPs} = \text{full_attention}
     $$

   - **sparse_mode == 2 或 3，且 $q_s == k_s$**（causal 或类似，序列长度相等）：  
     $$
     \text{FLOPs} = \text{full_attention} \times 0.5
     $$

   - **sparse_mode == 2，且 $q_s > k_s$**（causal，query 更长）：  
     $$
     \text{FLOPs} = \text{full_attention} \times \frac{q_s \times k_s - k_s \times k_s / 2}{k_s \times k_s}
     $$

   - **sparse_mode == 3，且 $q_d > k_d$**（特殊稀疏）：  
     $$
     \text{FLOPs} = \text{full_attention} \times \frac{k_s \times k_s / 2}{q_s \times k_s}
     $$

   - **sparse_mode == 2，且 $q_d < k_d$**：  
     $$
     \text{FLOPs} = \text{full_attention} \times \frac{q_s \times q_s / 2}{q_s \times k_s}
     $$

   - **sparse_mode == 3，且 $q_d < k_d$**：  
     $$
     \text{FLOPs} = \text{full_attention} \times \frac{q_s \times k_s - q_s \times q_s / 2}{q_s \times k_s}
     $$

### FlashAttention 计算注意事项

- **必需信息**：
  - 输入布局（input_layout）：TND 或 BNSD/BSND/BSH/SBH
  - 对于 TND：需要 `actual_seq_qlen` 和 `actual_seq_kvlen`（累积长度数组）
  - 对于 Common layout：需要 `sparse_mode`（0/2/3）
  - 输入张量的形状（input_shapes）

- **常见 sparse_mode 含义**：
  - `0`：完整 attention（无稀疏）
  - `2`：通常表示 causal attention（因果掩码）
  - `3`：其他稀疏模式

- **如果缺少关键参数**（如 sparse_mode 或 actual_seq_qlen），应向用户明确说明需要从 `operator_args` 中获取这些信息。

## 从 Profiler 落盘数据批量提取与聚合 MFU（落盘场景）

当用户给的是 **Ascend PyTorch Profiler 落盘目录**时，不要让用户手填维度，直接从落盘批量提取并聚合 MFU。

> **不要被目录名 / profiling level 限制**：落盘目录可能叫任意名字（`data/level2`、`prof_out`、`rank0_xxx_ascend_pt` 等都行），识别特征是目录内有 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/`（或解析后的 `mindstudio_profiler_output/`）。本方法对采集等级不挑（Level0/1/2 均可），真正的前提是两个开关：
> - **算 MFU**：需 `record_shapes=true`（否则 `kernel_details.csv` 无 `Input/Output Shapes` 列，退化为只能报硬件 `cube_utilization(%)`、无法算 FLOPs-based MFU）；
> - **算显存**：需 `profile_memory=true`（见 `performance-health-score` 的显存容量利用率小节）。
>
> 先 `head -1 kernel_details.csv` 确认有 shape 列、`profiler_info_*.json` 里 `record_shapes` 为 true，再开算。

### 数据来源

| 文件 / 表 | 用途 | 关键列 / 字段 |
|---|---|---|
| `kernel_details.csv` | 主来源：逐算子 shape + 耗时 + 硬件实测利用率 | `Type`、`Duration(us)`、`Input Shapes`、`Output Shapes`、`Input Data Types`、`cube_utilization(%)`、`aic_mac_ratio` |
| `ascend_pytorch_profiler_*.db` 的 `COMPUTE_TASK_INFO` + `TASK` | 大数据量时替代 CSV（shape/dtype 经 `STRING_IDS` 映射，耗时 = `endNs-startNs`） | `inputShapes`/`outputShapes`/`opType`、`startNs`/`endNs` |
| `ascend_pytorch_profiler_*.db` 的 **`NPU_INFO`** | 取芯片型号定理论峰值（落盘 `profiler_metadata.json` 通常**不含**子型号） | `id`(deviceId)→`name`(芯片型号) |
| `step_trace_time.csv` | 端到端 step MFU 的分母（见下"两种口径"） | `Computing`、`Stage`（单 step 总跨度） |

### M/N/K 推导规则（转置安全，**务必遵守**）

落盘里 matmul 的两个输入 shape 不能直接当作 `[M,K]·[K,N]`——Ascend 的 MatMulV2/V3 存在 NN/NT/TN 多种转置布局，第二个输入既可能是 `[K,N]` 也可能是 `[N,K]`。**必须用输出 shape 锚定 M、N**：

```
设 输出 Output Shapes = [M, N]
   第一个输入 A 的 shape = [a1, a2]
   K = (a1 == M) ? a2 : a1      # A 中不等于 M 的那一维即收缩维 K
   FLOPs = 2 × M × N × K
```

> ⚠️ **反例（已踩坑）**：若图省事用"输入A第一维=M、输入B第一维=N"，遇到 `in=[4096,6144]·[6144,1024], out=[4096,1024]` 会把 N 错当成 6144，单算子达成算力虚高到 ~1100 TFLOP/s、聚合 MFU 冲到 128%~180%（物理不可能）。用输出 shape 定 M/N 后回到合理的 ~49–69%。看到 MFU > 100% 时，第一嫌疑就是这里。
>
> FlashAttention 用本 skill 上文的 FA 公式（需 `input_layout` / `sparse_mode`，从 `Input Shapes` 与算子属性判断），不要套 matmul 公式。

### 聚合与口径

筛 `Type` 以 `MatMul` 开头（及 FA 等计算算子），逐算子算 FLOPs，再聚合：

```
聚合达成算力 (TFLOP/s) = Σ FLOPs ÷ (Σ Duration_s) ÷ 1e12
MFU = 聚合达成算力 ÷ 芯片峰值 TFLOP/s
```

**两种口径要说清，别混**：

| 口径 | 分母 | 含义 | 用途 |
|---|---|---|---|
| **算子达成率（默认）** | Σ 该类算子自身耗时 | matmul/FA 跑起来时离峰值多远 | 评估算子实现/shape 是否吃满 cube |
| **端到端 step MFU** | step 总跨度（`step_trace_time.csv`） × 峰值 | 整网 FLOPs 摊到整个 step（含空泡/通信/访存） | 评估训练整体效率，永远 ≤ 算子达成率 |

### cube_utilization(%) ≠ MFU（两个都报，互补）

`kernel_details.csv` 自带的 `cube_utilization(%)` / `aic_mac_ratio` 是**硬件实测的 AI Core cube 流水线利用率**（算子内部 cube 单元忙时占比），与 FLOPs-based MFU 是不同维度：

- **cube_utilization 高（如 93%）但 MFU 一般** → cube 单元忙，但受 shape 不规则 / dtype / 非 cube 时间（MTE 搬运、scalar）拖累，达成算力上不去；
- 两者一起报最有信息量：cube_util 说明"单元忙不忙"，MFU 说明"忙的同时算得快不快"。
- 这与 `timeline-swimlane-analyzer` 的 **time-based 算子利用率**（device 忙时 / step 跨度，看"有没有空泡"）又是第三个维度，三者不互相替代。

### MFU 不可算时的兜底链（融合 / AICPU 算子 + 非 cube PMU）

落盘场景下 FLOPs-based MFU 会因两类原因**双重失效**，必须有兜底，**不要硬算出一个假 MFU**：

1. **融合 / AICPU 算子无 shape**：TP 序列并行的 `AllGatherMatmul` / `MatmulReduceScatter` 及其 `*Aicpu` 变体，`InputShapes` 常为 `N/A`，无法推 FLOPs；且它们的耗时**内嵌通信等待**（不是纯计算），即使有 shape 也**不该**套 matmul MFU 公式（会把通信时间算进分母，结果无意义）。
2. **PMU metric 选错**：`cube_utilization` 只有在 `aic_metrics=PipeUtilization` 时才采集；若采集用的是 `ACL_AICORE_MEMORY_UB`（UB 带宽计数）等其它 metric set，则 `kernel_details.csv` / `TASK_PMU_INFO` 里**没有 cube_util**，第一档兜底也没了。

**兜底优先级**（从精确到粗略，用到哪档就在报告里写明用了哪档 + 为什么）：

```
① FLOPs-based MFU（有 shape 的 MatMul/FA）
   ↓ 主力算子为融合/AICPU、shape=N/A → 不可算
② 硬件 cube_utilization（aic_mac_ratio）
   ↓ aic_metrics 非 PipeUtilization、未采集 → 不可算
③ 时间口径 device-busy 代理 = Σ计算算子耗时 ÷ step 跨度（×100）
   并按已知低效项下修（如 AICPU 时间占比、Block Dim 未饱和占比），显式标注"代理值，非真实 MFU"
```

> 报告里务必区分：③ 反映"device 有多忙"，**不**反映"忙时算得多快"。把它写进 PHS「计算利用率」子项时，要在第 5 章注明口径与下修依据（承 `performance-health-score`）。

### 芯片理论峰值来源

1. 优先读 `NPU_INFO` 表拿确切型号，对照本 skill 上文"常见芯片理论峰值"表取 BF16/FP16 峰值；
2. 落盘缺型号时，**不要猜单一值**，按 910B1/B2/B3/B4 给一组区间并显式注明"待确认型号"（见上文峰值表）；
3. 端到端口径下显存/算力都依赖型号，最终交付前应让用户确认。

### 可复用提取脚本（awk，处理含逗号/分号的引号 shape）

无 Python 环境时可直接用 awk（`FPAT` 正确切分被 `"..."` 包裹、内部含 `,`/`;` 的 shape 列）：

```bash
D="<落盘目录>/ASCEND_PROFILER_OUTPUT"
awk -v FPAT='([^,]*)|("[^"]*")|("""[^"]*""")' '
NR>1 && $7 ~ /^MatMul/ {
  ins=$16; gsub(/"/,"",ins); outs=$19; gsub(/"/,"",outs); dur=$11+0
  split(ins,p,";"); split(p[1],a,","); split(outs,o,",")
  M=o[1]+0; N=o[2]+0; a1=a[1]+0; a2=a[2]+0
  K=(a1==M)?a2:a1                         # 转置安全：A 的非 M 维即 K
  if(M>0&&N>0&&K>0&&dur>0){tot+=2.0*M*N*K; td+=dur; cnt++; cu+=$NF}
}
END{ ach=tot/(td/1e6)/1e12
  printf "MatMul 个数=%d 总FLOPs=%.3e 聚合达成=%.1f TFLOP/s cube_util均值=%.1f%%\n",cnt,tot,ach,cu/cnt
  printf "MFU @910B1(378.88)=%.1f%% @910B3(294.91)=%.1f%%\n",ach/378.88*100,ach/294.91*100 }
' "$D/kernel_details.csv"
```

> 列号（`$7`=Type、`$11`=Duration、`$16`=Input Shapes、`$19`=Output Shapes、`$NF`=cube_utilization）以当前 Level2 表头为准；表头变化时先 `head -1 kernel_details.csv` 核对再调整。

---

## 计算 MFU 的标准步骤

当用户希望你计算某个算子的 MFU 时，严格按照以下步骤：

1. **确认信息是否充分**  
   向用户要齐以下信息（如果缺失就明确提出）：  
   - 算子类型（例如 matmul / GEMM / FlashAttention等）。  
   - 参与运算的张量维度（包含 batch / head / sequence 等关键维度）。  
   - 单次算子执行的耗时（例如毫秒 ms）。  
   - 硬件单卡的理论峰值算力（例如 312 TFLOPs/s，注明是 FP16/BF16 还是 FP8 等）。  

2. **计算算子 FLOPs**  
   - 根据算子类型和维度，用上面的公式算出 **单次调用的 FLOPs**。  
   - 如果用户给了「每迭代包含多少次该算子」或「多个相同算子」，先计算单次，然后乘以调用次数。  

3. **计算 Achieved FLOPs/s**  
   - 先换算执行时间到秒，例如：$t_\text{s} = \text{time\_ms} / 1000$。  
   - Achieved FLOPs/s = FLOPs / $t_\text{s}$。  
   - 再换算到 TFLOPs/s：Achieved TFLOPs/s = Achieved FLOPs/s ÷ 1e12。

4. **计算 MFU**  
   - MFU = Achieved TFLOPs/s ÷ Peak TFLOPs/s。  
   - 最终给出百分比形式，例如 0.42 → 42%。  

5. **解释结果**  
   - 简要说明这个 MFU 代表的含义，例如：  
     - 低于 20%：通常算子远未吃满算力，可能受内存带宽、launch overhead、shape 不规则等影响。  
     - 30%–60%：中等偏上水平，许多通用工作负载大致在这个区间。  
     - 高于 70%：算子形状、并行度和实现都比较接近设备上限。  

## 回答格式要求

当用户请求你计算 MFU 时，请按如下结构作答（用用户的语言，可以是中文也可以是英文）：
  
1. 当你按照本 Skill 提供的步骤计算 MFU 时，请在回答开头用一句话明确说明：“（本回答基于 op-mfu-calculator Skill 的 MFU 计算规范）”
2. **先复述输入信息**（包括算子类型、张量维度、时间、峰值算力）。  
3. **列出关键公式**（FLOPs, Achieved TFLOPs/s, MFU），并代入具体数字展示中间计算过程。  
4. **给出最终 MFU 数值**（保留 2–3 位有效数字，百分比形式）。  
5. **简单分析**产生这个 MFU 的可能原因或优化方向（例如 batch 太小、K 维过小、显存带宽瓶颈等）。  

如果信息不全，**不要瞎猜**，而是明确列出还缺哪些数字，并给出如何从 profiler / 日志中拿到这些信息的建议。




