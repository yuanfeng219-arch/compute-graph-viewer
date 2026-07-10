# 训练精度问题定位链（Pangu Pro MoE 72BA16B）

> 从表象精度异常出发，逐层下钻直至根因。**迭代层后即分叉**：单卡/多卡复现是否一致，决定走「通信分支」还是「计算主干」。每一层有明确的**判据**决定下一步方向。
>
> 本文以 **Pangu Pro MoE 72BA16B**（48 层 / 4608 hidden / GQA 注意力 / 64 experts / MoGE 路由）为参考模型，沿用了通用定位链方法论及典型案例剧情。

```
迭代层           WHEN     — 时间维：哪个 step
   │
   ├── 单卡≠多卡 → 通信调度层  WHY(通信) — 卡间通信问题（失步 / 精度漂移）
   │                  │
   │                  └── 模型层  WHERE    — 通信影响了哪一层
   │                        │
   └── 单卡=多卡 → 模型层   WHERE    — 空间维：模型哪一层
                     │
                  算子层   WHAT     — 计算维：哪个算子
                     │
                  张量层   WHICH    — 数值维：哪些元素/区间
                     │
                  infra层  CONTEXT  — 并行策略 / 硬件归属
                     │
               超参/代码层  FIX     — 具体改什么
```

---

## 1. 迭代层 — 锁定「哪个 step」

| 项目 | 内容 |
|------|------|
| **定位目标** | 精度异常首次出现（或突然恶化）的训练步 |
| **观测手段** | loss 曲线、acc/PPL 曲线、grad_norm 曲线，按 global_step 展开 |
| **判据** | loss spike / NaN / grad_norm 突变的 step 即为嫌疑步；若整条曲线平稳漂移，取漂移起始段 |
| **产出** | 嫌疑 step 列表（如 step 15320~15350） |

> ⚠️ 此处产生**关键分叉**：用同一数据/种子在单卡和多卡环境各跑一次嫌疑 step。
> - **单卡复现 = 多卡复现** → 计算问题，沿主干进入「模型层」
> - **单卡复现 ≠ 多卡复现** → 通信问题，切入分支「通信调度层」

## 2. 模型层 — 锁定「哪一层」

| 项目 | 内容 |
|------|------|
| **定位目标** | 嫌疑 step 内，异常发生在 Emb / Attn / FFN / MoE 中的哪一层（或哪些层） |
| **观测手段** | 层 × step 热力图：每层逐 step 的 grad_norm（反向追因）或激活值离群率 |
| **判据** | 热力图中对应 step 的纵向高亮带，定位到具体 layer index；Dense 层（L0~L3）与 MoE 层（L4~L47）分开标注 |
| **产出** | 问题层路径，如 `model.layers.5.self_attn` 或 `model.layers.33.mlp` |

---

## ── 分支：通信调度层 ──

> 从迭代层直接分叉。当单卡多次运行结果一致、多卡运行结果出现差异时，进入此分支。通信分支定位完毕后，仍需回到模型层确定通信问题影响了哪些层。

| 项目 | 内容 |
|------|------|
| **定位目标** | 判断通信问题类别：调度失步（all-reduce / all-to-all / p2p 时序错位）还是浮点累加舍入误差累积（精度对齐漂移） |
| **判据** | 单卡多次结果一致 + 多卡结果不一致 → 通信问题；单卡多次结果不一致 → 非通信问题（可能是随机性/dropout/数据差异），回到主干 |
| **观测手段** | NCCL Inspector 通信概览表（带宽/延迟异常）、通信算子 trace 时间线（看 send/recv 配对、barrier 等待）、各 rank 的 all-reduce/all-to-all 输入/输出 hash 对比 |
| **产出** | 通信异常类型 + 涉及的 rank 列表 + 通信原语（all-reduce / all-to-all / p2p） |

---

## 3. 算子层 — 锁定「哪个算子」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在问题层内，定位到具体的异常算子（如 `q_proj`、`k_norm`、`attn_scores`、`gate_proj`、`router`） |
| **观测手段** | 层内算子误差瀑布图（横轴：按执行顺序排列的算子，纵轴：与 baseline 的误差），定位误差突变点 |
| **判据** | 误差在某个算子处出现量级跳跃（如 1e-6 → 1e-3），即为首害算子 |
| **产出** | 问题算子路径，如 `model.layers.5.self_attn.q_proj` |

## 4. 张量层 — 锁定「哪些元素/数值区间」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在问题算子的输入/输出中，定位到哪些元素位置、哪些数值区间引入了偏差 |
| **观测手段** | dump 该算子的输入、激活、梯度张量，与 baseline（单卡正确结果或上一正常 step）逐元素对比；对 softmax、layernorm、k_norm 等数值敏感算子，绘制散点图、直方图、TopK 误差表 |
| **判据** | 逐元素 diff 中绝对值最大的元素即为嫌疑元素；直方图中偏离 baseline 分布的区间即为嫌疑数值区间 |
| **产出** | 嫌疑元素索引 + 数值区间范围 + 偏差量级 |

## 5. infra层 — 锁定「并行策略 / 硬件」

| 项目 | 内容 |
|------|------|
| **定位目标** | 确定问题涉及的并行策略（PP/EP/TP/DP）和硬件范围（哪些 rank、哪些节点） |
| **观测手段** | 从 trace 中提取问题算子涉及的并行策略标签（pp/ep rank）、通信原语、所属 GPU 节点；结合通信分支结论交叉定位 |
| **判据** | 问题算子仅出现在特定 pp stage 或 ep rank → 局部硬件/策略问题；跨所有 rank 出现 → 全局算法/精度问题 |
| **产出** | 涉及的 rank 列表 + 并行策略维度 + 嫌疑硬件范围（如 node3 GPU 0~7） |

## 6. 超参/代码层 — 锁定「改什么」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将根因映射到具体可执行的修改项 |
| **观测手段** | 回溯：启动参数（学习率、warmup、batch_size、precision）、模型代码（算子实现、数值精度 cast）、通信业务代码（通信原语调用、同步点）、训练超参（dropout rate、weight decay） |
| **判据** | 根据上游各层锁定的根因类型匹配修改项：数值溢出 → 调低 lr / 开 loss scaling；通信失步 → 修复同步点；专家路由倾斜 → 调整 load balance 策略 |
| **产出** | 具体修改项 + 修改文件/参数路径 + 验证方案 |


---

## 案例一：通信分支 — MoE all-to-all 超时导致 loss NaN

> **路径**：迭代层 → 单卡≠多卡 → 通信调度层 → 模型层 → infra层 → 超参/代码层

**背景**：32 GPU 训练 Pangu Pro MoE 72BA16B，EP=32，TP=1，PP=4，BF16 精度，seq_len=4096，global_batch=1024。训练至 step ~15000 时 loss 突发 NaN。

### 1. 迭代层

| 步骤 | 内容 |
|------|------|
| **现象** | step 15203 loss 从 3.1 跳变至 NaN，grad_norm 从 12.4 跳至 inf；step 15200~15202 一切正常 |
| **判据** | 突变发生在单步内（step 15202→15203），排除缓慢漂移 |
| **产出** | 嫌疑 step = 15203 |

### 2. 分叉判定：单卡 vs 多卡

| 步骤 | 内容 |
|------|------|
| **操作** | 锁定 step 15203 的输入数据（dataloader seed 固定），分别在 1 GPU 和 32 GPU 上重跑该 step |
| **结果** | 单卡：loss=3.21，grad_norm=11.8，完全正常 / 多卡：loss=NaN，grad_norm=inf |
| **判据** | 单卡复现 ≠ 多卡复现 → **切入通信分支** |

### 3. 通信调度层

| 步骤 | 内容 |
|------|------|
| **观测** | 开启 `NCCL_DEBUG=INFO` 重跑 step 15203。NCCL trace 显示 EP rank 23（node2 GPU 7）在 `all-to-all` 调用处超时（30s timeout）。该调用属于 layer 30 MoE 的 expert dispatch 阶段 |
| **进一步确认** | 对比各 rank 的 all-to-all send/recv buffer size：rank 23 的 send buffer 为 0（没有 token 被 router 分发到其他 rank 的 expert），而 recv buffer 期望接收 2048 token × 4608 dim × 8 experts 的数据，size 不匹配导致死锁（各 rank 在 all-to-all 通信中互相等待对方发送数据，但没有 rank 实际发送，形成循环等待） |
| **判据** | all-to-all send/recv 不匹配 → 通信调度失步 |
| **产出** | 异常通信原语：`all-to-all` / 异常 rank：EP rank 23 / 关联层：layer 30 MoE |

### 4. 模型层（回到主干定位影响层）

| 步骤 | 内容 |
|------|------|
| **观测** | 提取 step 15203 各层 router 的 token-to-expert 分配统计。layer 30 的 router 将当前 micro-batch 中 98% 的 token 路由到了 expert 47（恰好位于 EP rank 23），其余 63 个 expert 几乎无 token |
| **根因** | Router 的 MoGE 8 组（每组 8 experts、每组选 top-1）配置下，某个 token 群体的 hidden state 对 group 6（experts 40~47）中 expert 47 的 gate weight 产生了极端偏好（sigmoid score > 0.99），导致该 expert 所在的 rank 23 通信负载骤增、其余 rank 空等，形成 all-to-all 死锁。Pangu 的 MoGE 分组路由本应为组间负载均衡设计，但组内仍可能出现单 expert 热点 |
| **产出** | 问题层：`model.layers.30.mlp.router` / 路由倾斜度：单 expert 收到 98% token |

### 5. infra层

| 步骤 | 内容 |
|------|------|
| **观测** | 问题集中在 EP rank 23（node2 GPU 7），属于 PP stage 3（layers 24~35），该节点其余 7 个 GPU 均在 step 15203 的 all-to-all 中等待 rank 23 而空闲 |
| **判据** | 问题聚集在单个 EP rank → 局部路由倾斜，非全局硬件故障 |
| **产出** | 嫌疑范围：node2 GPU 7（EP rank 23），PP stage 3，layer 30 MoE |

### 6. 超参/代码层

| 步骤 | 内容 |
|------|------|
| **修改** | ① 增大 MoGE 的 group 数量从 8→16（每组 expert 数 8→4），分散 expert 选择范围，降低单 expert 热点概率；② 在 router gate 前增加 z-loss 正则项（系数 1e-4），抑制 gate logit 极端值；③ NCCL all-to-all 超时从 30s 延长至 60s 作为兜底 |
| **验证** | 修改后从 step 15000 续跑，step 15203 正常通过，继续训练 5000 step 无 NaN |

---

## 案例二：计算分支 — q_proj 精度溢出导致 grad_norm 缓慢发散

> **路径**：迭代层 → 单卡=多卡 → 模型层 → 算子层 → 张量层 → infra层 → 超参/代码层

**背景**：同上 32 GPU 训练 Pangu Pro MoE 72BA16B，EP=32，TP=1，PP=4。训练采用 BF16 基础精度 + FP8 混合精度（对 Linear 算子启用 FP8 E4M3 加速）。训练至 step ~8000 后 grad_norm 持续上升，loss 缓慢恶化。

> **注**：Pangu Pro MoE 原生为 BF16 训练，本案例假设部署中为追求吞吐引入了 FP8 混合精度。与此前 MLA 架构的 `q_lora` 低秩压缩不同，Pangu 的 GQA 注意力中 `q_proj` 是一个直接的大矩阵投影 [4608→12288]，输入端的数值溢出经大权重矩阵乘法后被显著放大。

### 1. 迭代层

| 步骤 | 内容 |
|------|------|
| **现象** | step 8200 起 grad_norm 从 ~10 逐步升至 ~85（step 8600），loss 从 2.95 升至 4.82，未现 NaN 但趋势持续恶化 |
| **判据** | 平稳漂移 → 取漂移起始段 step 8200~8600 |
| **产出** | 嫌疑 step 范围：8200~8600 |

### 2. 分叉判定：单卡 vs 多卡

| 步骤 | 内容 |
|------|------|
| **操作** | 在单卡和多卡环境分别重跑 step 8200~8600 |
| **结果** | 单卡：grad_norm 同样从 10 升至 85 / 多卡：grad_norm 从 10 升至 85 |
| **判据** | 单卡复现 = 多卡复现 → **沿计算主干，进入模型层** |

### 3. 模型层

| 步骤 | 内容 |
|------|------|
| **观测** | 绘制 48 层 × step（8200~8600）grad_norm 热力图。layer 33（MoE 层）的 grad_norm 热力值（~450）是其他层（~30~50）的约 10 倍，且随 step 持续增长 |
| **判据** | 热力图纵向高亮带锁定 layer 33 |
| **产出** | 问题层：`model.layers.33`（MoE TransformerLayer，GQA attention + PanguProMoE） |

### 4. 算子层

| 步骤 | 内容 |
|------|------|
| **观测** | 层内算子误差瀑布：按执行顺序对比 layer 33 各算子输出与 baseline（step 8000 正常时的值）。`input_layernorm`（1e-7）→ `q_proj`（**1e-7→3.2e-2 跳跃**）→ `k_proj`（1e-7）→ `k_norm`（1e-7，K-Norm 未放大误差）→ `v_proj`（1e-7）→ `core_attention`（8.1e-2，attention softmax 进一步放大）→ 后续算子持续偏高 |
| **判据** | 误差在 `q_proj` 处出现量级跳跃（1e-7→1e-2），为首害算子。注意 `k_norm`（Pangu 特有的仅对 Key 做 RmsNorm）并未放大误差，排除了 K-Norm 稳定性嫌疑 |
| **产出** | 问题算子：`model.layers.33.self_attn.q_proj`（Linear [4608→12288]，FP8 精度） |

### 5. 张量层

| 步骤 | 内容 |
|------|------|
| **观测** | dump step 8500 时 layer 33 `q_proj` 的输入激活张量（shape [4096, 4608], FP8 E4M3）。绘制数值分布直方图： |
|  | • 正常区间（0~448）：占 96.8% 元素，分布与 baseline 一致 |
|  | • 溢出区间（>448，即 FP8 E4M3 max）：占 **3.2%** 元素，最大值 2.3×10⁴ |
|  | TopK 误差表：diff 最大的 100 个元素索引集中在 hidden dim [4140, 4608] 区间，对应 Partial RoPE 的高频分量（rope=64 维中频率最高的分量更容易在深层累积激活值） |
| **判据** | 3.2% 的输入元素超过 FP8 E4M3 表示范围（max=448），在 `q_proj` 的 Linear [4608→12288] 计算中产生截断误差，经大权重矩阵（12288 output dim）进一步放大——与 MLA 低秩压缩放大不同，此处是扩张投影的 sensitivity 更高 |
| **产出** | 嫌疑元素：hidden dim [4140, 4608] 的尾部 468 维 / 数值区间：[448, 2.3×10⁴] / 偏差量级：3.2e-2（算子输出级） |

### 6. infra层

| 步骤 | 内容 |
|------|------|
| **观测** | 检查问题是否局限在特定 PP stage 或 EP rank。layer 33 属于 PP stage 3（layers 24~35），但在所有 32 个 EP rank 上均观测到相同的 q_proj 溢出模式 |
| **判据** | 跨所有 rank 复现 → 全局精度问题，非硬件/特定节点故障 |
| **产出** | 全局问题，与 PP/EP 切分无关，根因在 FP8 数值表示能力不足 |

### 7. 超参/代码层

| 步骤 | 内容 |
|------|------|
| **修改** | ① 对 layer 33 的 `q_proj` 输入增加 per-tensor dynamic scaling：`input = input / max(\|input\|) * 448`，将值域动态映射到 FP8 安全区间后再做 Linear；② 或者在 `q_proj` 前插入一层 `Fp8Cast` 时使用 delayed scaling 策略，对 hidden dim 尾部高频分量（Partial RoPE rope=64 维对应位置）单独 scale；③ 长期方案：评估是否对 L30+ 深层的 q_proj 改用 BF16（Pangu 原生精度），仅在前半段保留 FP8 加速 |
| **验证** | 方案① 从 step 8000 续跑，layer 33 的 q_proj 误差降至 1e-6 量级，grad_norm 稳定在 10~15，继续训练 10000 step 无发散 |


---

# 第二部分：Infra 定位链（基础设施视角）

> Infra 工程师从**硬件/资源**观测出发，按硬件拓扑下钻。与精度链在「通信原语 / 算子 kernel」层面汇合，最终收敛到配置或硬件变更。

```
集群层           WHERE    — 集群 / 节点 / GPU 级别定位
   │
资源层           WHAT     — 计算 / 显存 / 网络，哪类资源瓶颈
   │
   ├── 通信瓶颈 → 通信原语层  WHICH(comm) — all-reduce / all-to-all / p2p  ↕ 汇合精度链·通信调度层
   │                  │
   │                  └── 硬件层  HARDWARE — Xid / ECC / NVLINK 链路状态 / thermal
   │
   └── 计算瓶颈 → 算子/kernel层  WHICH(kernel) — CUDA kernel / block / occupancy  ↕ 汇合精度链·算子层
                       │
                       └── 硬件层  HARDWARE — SM error / clock throttle / register spill
                            │
                      配置变更层  FIX     — NCCL 参数 / GPU 功率 / 拓扑 affinity
```

---

## 1. 集群层 — 锁定「哪个节点 / 哪个 GPU」

| 项目 | 内容 |
|------|------|
| **定位目标** | 在多节点集群中，定位异常发生在哪个节点、哪些 GPU |
| **观测手段** | 节点级 GPU 利用率（nvidia-smi / DCGM）、MFU、显存占用率、节点功耗/温度仪表盘；按 node × GPU 展开的面板 |
| **判据** | 某节点/GPU 的利用率或 MFU 显著偏离集群均值（如其他节点 55%，异常节点 20%）；显存占用与其余节点不一致 |
| **产出** | 异常节点 + GPU 列表（如 node2 GPU 3~7） |

## 2. 资源层 — 锁定「计算 / 显存 / 网络」

| 项目 | 内容 |
|------|------|
| **定位目标** | 判断瓶颈类型：计算受限（SM 打满）、显存受限（HBM 带宽饱和/OOM）、网络受限（NCCL 带宽不足/链路降级） |
| **观测手段** | SM occupancy（nsys/ncu）、HBM bandwidth utilization（DCGM profiler）、NVLINK/IB throughput（nvidia-smi nvlink -s / ib_read_bw）、PCIe 带宽 |
| **判据** | SM occupancy < 30% + HBM 带宽正常 → 计算空闲（等数据）；NVLINK/IB throughput 骤降 → 网络瓶颈；显存接近上限且频繁 retry → 显存瓶颈 |
| **产出** | 瓶颈类型：计算 / 显存 / 网络 + 具体指标基线 vs 当前值 |

> ⚠️ 此处产生**分叉**：根据资源层定位的瓶颈类型——
> - **网络瓶颈** → 进入「通信原语层」
> - **计算瓶颈 / 显存瓶颈** → 进入「算子/kernel层」

---

## ── 分支A：通信原语层 ──

> 当资源层判定为网络瓶颈时进入。与精度链的「通信调度层」汇合。

| 项目 | 内容 |
|------|------|
| **定位目标** | 定位到具体的 NCCL 通信原语调用（all-reduce / all-to-all / p2p / broadcast）及其涉及的 rank |
| **观测手段** | NCCL trace（`NCCL_DEBUG=INFO`）、Nsight Systems 通信时间线（send/recv 配对、barrier 等待）、per-call bandwidth、NCCL topology 日志 |
| **判据** | 某次 all-to-all/all-reduce 调用耗时是其他调用的 5×以上；同一调用在不同 rank 上耗时差异 > 2×；NCCL 日志中出现 `[ERROR]` 或 `fallback to slow path` |
| **产出** | 异常通信原语 + 涉及的 rank 列表 + 耗时/带宽数据 |

### ↕ 汇合点：此处可与精度链「通信调度层」结论交叉验证

---

## ── 分支B：算子/kernel层 ──

> 当资源层判定为计算或显存瓶颈时进入。与精度链的「算子层/张量层」汇合。

| 项目 | 内容 |
|------|------|
| **定位目标** | 定位到具体的 CUDA kernel（如 `linear_fp8`、`flash_attn`、`rms_norm`），分析其执行效率 |
| **观测手段** | Nsight Compute（kernel occupancy、register spill、shared memory usage）、Nsight Systems kernel timeline（看 kernel launch gap、stream overlap） |
| **判据** | kernel occupancy < 50% → 计算资源利用不足；register spill > 128B/thread → 寄存器压力；kernel 之间有 > 100μs gap → stream 调度问题 |
| **产出** | 问题 kernel 名称 + occupancy / register spill / launch gap 数据 |

### ↕ 汇合点：此处可与精度链「算子层」结论交叉验证

---

## 3. 硬件层 — 锁定「硬件故障 / 降级」

| 项目 | 内容 |
|------|------|
| **定位目标** | 排除或确认硬件根因：GPU Xid 错误、ECC 内存纠错、NVLINK link 掉线、thermal throttle、GPU clock 降频 |
| **观测手段** | `dmesg` / `nvidia-smi -q`（Xid error、ECC count）、`nvidia-smi nvlink -e`（link 状态）、DCGM thermal/clock 指标、PCIe AER 日志 |
| **判据** | Xid 48（double-bit ECC）→ 显存硬件故障；NVLINK link inactive → 链路掉线需复位；GPU clock 持续低于 base clock → thermal throttle |
| **产出** | 硬件故障类型 + 故障 GPU serial/PCIe BDF + 是否需硬件更换或复位 |

## 4. 配置变更层 — 锁定「改什么」

| 项目 | 内容 |
|------|------|
| **定位目标** | 将 Infra 诊断结论映射到可执行的配置变更或硬件操作 |
| **观测手段** | 回溯：NCCL 环境变量（`NCCL_IB_TIMEOUT`、`NCCL_NET_GDR_LEVEL`、`NCCL_P2P_LEVEL`）、GPU 功率/时钟策略（`nvidia-smi -pl`、`nvidia-smi -ac`）、节点拓扑 affinity（NUMA binding、GPU-NIC affinity） |
| **判据** | 网络瓶颈且非硬件故障 → 调 NCCL 参数；硬件故障 → RMA 或节点下线；thermal throttle → 调功率上限或改善散热 |
| **产出** | 具体配置变更 + 变更文件/命令 + 验证方案 |


---

## Infra 案例：NVLINK 链路掉线导致 MFU 骤降

> **路径**：集群层 → 资源层 → 通信原语层 → 硬件层 → 配置变更层

**背景**：32 GPU（4 节点 × 8 GPU）训练 Pangu Pro MoE 72BA16B，EP=32，TP=1，PP=4，BF16 精度。训练至 step ~20000 后，总吞吐从 3200 tokens/s 掉至 1200 tokens/s，MFU 从 55% 降至 20%。

### 1. 集群层

| 步骤 | 内容 |
|------|------|
| **现象** | node2 的 8 个 GPU 利用率从 92% 整体掉至 35~40%，其余 3 个节点利用率正常（90~95%）。显存占用各节点一致（~78GB/80GB），排除 OOM |
| **判据** | 异常集中在 node2，其余节点正常 → node2 故障 |
| **产出** | 异常节点：node2（GPU 0~7） |

### 2. 资源层

| 步骤 | 内容 |
|------|------|
| **观测** | ① SM occupancy：node2 GPU 利用率虽低，但 SM occupancy 仍有 85%（GPU 在等数据而非空闲计算）；② HBM bandwidth：正常 1.2TB/s；③ **NVLINK throughput：node2 GPU 3→GPU 4 链路带宽从 45GB/s 降至 0.8GB/s**（其余链路 45GB/s 正常）；④ IB throughput：node2→node1 的 IB 带宽从 25GB/s 升至 48GB/s（接近 IB 上限），其余节点间 IB 负载正常 |
| **判据** | NVLINK 链路异常降级 + IB 负载异常升高 → NCCL 被迫从 NVLINK 高速路径回退到 IB 低速路径 |
| **产出** | 瓶颈类型：网络 / 嫌疑链路：node2 GPU3↔GPU4 NVLINK |

### 3. 通信原语层

| 步骤 | 内容 |
|------|------|
| **观测** | Nsight Systems trace 显示：layer 28~34 的 MoE all-to-all 调用在 node2 GPU 3 和 GPU 4 上耗时从正常的 2.3ms 飙升至 18.7ms（8× 恶化）。NCCL topology 日志显示 GPU 3↔GPU 4 间 `NCCL_P2P_LEVEL` 从 `PATH_NVL` 回退到 `PATH_SYS`（经 PCIe/IB）。该链路承载了 PP stage 3↔4 的跨 stage p2p 传输（stage 3：layers 24~35，stage 4：layers 36~47），导致整个 PP pipeline 被拖慢 |
| **判据** | all-to-all 耗时 8× + NVLINK→SYS 回退 → 通信路径降级 |
| **产出** | 异常原语：`all-to-all`（MoE dispatch+combine）+ PP `p2p` / 涉及 GPU：node2 GPU 3, GPU 4 / PP stage：3↔4 |

### 4. 硬件层

| 步骤 | 内容 |
|------|------|
| **观测** | `nvidia-smi nvlink -e` 显示 GPU 3 的 NVLINK lane 5 状态为 `Inactive`（其余 11 条 lane 正常）。`dmesg` 中无 Xid 报错（非致命硬件故障），但 NVLINK CRC error count 在 step 19800 附近突增（lane 5：0→10⁶）。GPU 温度和功耗正常，排除 thermal throttle |
| **判据** | 单条 NVLINK lane inactive + CRC 错误突增 → 物理链路降级（可能为线缆松动或 transceiver 老化） |
| **产出** | 硬件故障：node2 GPU 3 NVLINK lane 5 inactive / 建议：优先 reseat NVLINK bridge，若恢复失败则 RMA |

### 5. 配置变更层

| 步骤 | 内容 |
|------|------|
| **临时绕过** | ① `export NCCL_IGNORE_DISABLED_P2P=1` 允许 NCCL 自动跳过故障链路；② 将 node2 GPU 3 从 EP group 中临时排除（调整 EP=31），用其余 31 GPU 继续训练，吞吐恢复至 3000 tokens/s |
| **永久修复** | ① 停机维护 window 内 reseat node2 GPU3↔GPU4 的 NVLINK bridge；② 若 reseat 无效，更换 GPU 3；③ 监控脚本增加 NVLINK lane status + CRC error 的定时巡检（每 10min），阈值告警 |
| **验证** | 临时方案：EP=31 跑 5000 step，loss 曲线与 EP=32 无差异（EP 减少 1 rank 对 64 expert 分布影响 < 3%，且 MoGE 8 组路由进一步分散了风险）；硬件修复后恢复 EP=32，NVLINK 全部 lane Active，MFU 回到 55% |
