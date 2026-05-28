# Memory Viewer Demo 页面功能 PRD

## 1. 背景与定位

Memory Viewer Demo 是面向 Ascend 910B 的计算图执行步骤与片上内存层级可视化页面。它把一段 PyPTO 计算图样例拆成按执行顺序排列的 operation schedule，并联动展示计算图节点状态、AIC/AIV 架构区域、L1/L0A/L0B/L0C/UB buffer 占用、MTE 搬运通路和当前 step 的读写 tensor。

页面当前使用固定样例：

- Header 标识：`IndexerPrologQuant · PATH0_leaf293`。
- Graph 数据：`data/sample-graph.json`。
- 执行数据：`data/ops.js` 中的 `OP_DATA`、`TENSOR_TOBE`、`TENSOR_META`。

业务口径需与 PyPTO/Ascend 910B 保持一致：

- Ascend 910B 支持 AIC 与 AIV 计算单元；AIC 侧重点包含 L1、L0A、L0B、L0C、Cube、FixPipe；AIV 侧重点包含 UB、Vector。
- PyPTO `MemoryType` 包含 `MEM_UB`、`MEM_L1`、`MEM_L0A`、`MEM_L0B`、`MEM_L0C`、`MEM_DEVICE_DDR`。
- 当前 demo 的 tier 映射为 `TENSOR_TOBE`: `1=L1`、`2=L0A`、`3=L0B`、`4=L0C`、`15=DDR`。
- 当前 demo 的调度顺序直接使用 `OP_DATA` 的 JSON 顺序，过滤无输入输出的 alloc marker 后作为播放 step；不是重新执行 PyPTO 编译器或调度器。

## 2. 目标用户与使用场景

目标用户：

- 算子开发者：理解某段计算图中 tensor 在片上内存之间的移动。
- 性能调优人员：观察 COPY、L1 到 L0、Cube MMA 等操作对 buffer 与硬件单元的影响。
- Demo 讲解人员：按 step 演示 Ascend 910B 内存层级和计算图执行关系。

核心场景：

- 顺序播放一段 operation schedule，观察每一步执行的 op 和涉及 tensor。
- 在计算图上查看已完成、执行中、待执行状态。
- 在架构图上查看 L1/L0A/L0B/L0C/UB 的占用率、容量、空闲空间和 read/write 高亮。
- 点击计算图 op 查看当前 op 的 step、状态、输入输出 tensor、shape、dtype、size、tier。
- 拖动时间轴快速跳转到特定 step。
- 调整上下分屏比例，兼顾计算图和内存架构图。

## 3. 页面内容

### 3.1 顶部工具栏

展示内容：

- 返回主页按钮。
- 模块名称：Memory Viewer。
- 硬件标签：Ascend 910B。
- 样例标签：IndexerPrologQuant · PATH0_leaf293。
- Step 指示器：当前 step、总 step、当前 op name、op magic。

功能：

- 提供页面身份和当前播放状态摘要。
- Step 指示器随 `goToStep(step)` 同步更新。

交互：

- 点击返回按钮回到 PTO launch 页面。
- 工具栏自身不提供数据切换；当前 demo 数据固定。

### 3.2 上半区：Compute Graph

展示内容：

- 区域标题：Compute Graph。
- 图控制按钮：Fit、Zoom In、Zoom Out。
- 缩放比例显示。
- 当前 op 概览行：Op、Magic、Inputs、Outputs、Tiers。
- 计算图画布：由 `data/sample-graph.json` 解析并复用 pass-ir 图渲染栈绘制。
- 加载占位：Loading graph。
- 颜色模式面板：Semantic、Subgraph、Latency、Engine / Memory、None。
- 图例区域。

功能：

- 展示样例计算图，并按当前 step 高亮：
  - 已完成 op。
  - 当前执行 op。
  - 待执行 op。
  - 当前 op 的输入 tensor。
  - 当前 op 的输出 tensor。
  - live tensor 的 dim 状态。
- 支持按不同色彩语义观察图：
  - Semantic：按操作语义分类。
  - Subgraph：按 subgraph id。
  - Latency：按 latency 梯度。
  - Engine / Memory：按执行引擎或内存层级。
  - None：中性显示。
- 默认 auto-follow 当前执行 op，使播放时视图自动居中到当前 op 及其输入输出。
- 当前 op 概览行汇总 `op.n`、`op.m`、`op.i`、`op.o` 和涉及 tier。

交互：

- 点击 Fit：适配计算图到视口。
- 点击 `+`/`-`：以视口中心放大/缩小。
- 鼠标拖拽画布：平移图。
- 鼠标滚轮缩放被刻意禁用，缩放只通过按钮控制。
- 点击颜色模式按钮：重绘计算图并更新图例。
- 点击计算图 operation 节点：打开 Selected Op Detail 浮层；再次点击同一 op 取消选择。
- 播放/跳步时，如果图已加载，会自动应用 step 高亮并跟随当前执行 op。

### 3.3 中间分隔条

展示内容：

- 横向 resizer，带 grip。

功能：

- 调整上半区计算图和下半区内存架构图的高度比例。
- 比例限制在 28% 到 78% 之间，避免某一区完全不可用。

交互：

- 按住分隔条拖动，上下区高度实时变化。
- 拖动过程中禁用文本选择。

### 3.4 下半区：运行状态栏

展示内容：

- Done 计数。
- 当前运行 op。
- Pending 计数。
- Runtime legend：COPY、L1→L0A、L1→L0B、MMA、Done、Active。

功能：

- 快速说明当前 schedule 进度。
- 作为内存/架构图颜色含义的辅助说明。

交互：

- 只读展示，随 step 自动刷新。

### 3.5 下半区：Ascend 910B 架构图

展示内容：

- AIC 区域：
  - GM strip。
  - L1 Buffer：1024 KB、约 20cy、row-major。
  - L0A：64 KB、约 5cy、fracZ。
  - L0B：64 KB、约 5cy、fracZ。
  - Cube MMA 单元。
  - L0C：256 KB、约 5cy、nZ。
  - FixPipe。
  - Scalar。
- AIC/AIV 分隔：`1 : N (AIV)`。
- AIV 区域：
  - UB：256 KB、约 10cy、row-major。
  - Scalar。
  - Vector。
- 每个 buffer 内部：
  - 使用率百分比。
  - 已用/总容量。
  - pixel grid。
  - Free 容量。
  - 起止地址。
  - 最多 5 个 live tensor 图例。

功能：

- 将当前 step 的 live tensor 按 tier 投影到对应 buffer。
- 使用 `TENSOR_META.b` 与每格 bytes-per-cell 估算占用格数：
  - L1：8x8，每格 16 KB。
  - L0A：4x4，每格 4 KB。
  - L0B：4x4，每格 4 KB。
  - L0C：8x8，每格 4 KB。
  - UB：8x8，每格 4 KB。
- 当前 op 的输入 tensor 标记 read，输出 tensor 标记 write。
- active tensor 在 buffer grid 中发光高亮。
- Cube/FixPipe 在 `A_MUL_B`、`A_MULACC_B` 时高亮。
- Vector 预留给 UB vector ops；当前样例 `UB_VEC_OPS` 为空，因此不会触发 Vector active。
- AIC/AIV 背景按是否存在对应 live tensor 轻微高亮。

交互：

- 鼠标悬停 buffer cell：显示浮动 tooltip，包含 tier、tensor name、shape、dtype、size、addr、read/write。
- Tooltip 会根据视窗边界自动换位，避免超出屏幕。
- 架构图本身不支持修改数据，只随 step 变化。

### 3.6 MTE 搬运通路覆盖层

展示内容：

- DDR 卡片。
- MTE1：DDR ↔ L1。
- MTE2-A：L1 → L0A。
- MTE2-B：L1 → L0B。
- MTE3：L0C → UB，当前 trace 作为拓扑参考，空闲。

功能：

- 按当前 op name 高亮对应搬运路径：
  - `COPY_IN`、`COPY_OUT` 激活 MTE1；`COPY_OUT` 方向反向。
  - `L1_TO_L0A` 激活 MTE2-A。
  - `L1_TO_L0B` 激活 MTE2-B。
  - MTE3 当前没有绑定 op，仅展示拓扑。
- DDR 卡片在 MTE1 active 时高亮。
- resize 时重新计算各硬件卡片之间的线段位置。

交互：

- 只读联动；随 step 自动刷新。

### 3.7 Selected Op Detail 浮层

展示内容：

- 标题：Selected Op Detail。
- 关闭按钮。
- 当前选择 op：
  - op name。
  - pipe/category badge。
  - Step index。
  - Magic。
  - Status：未调度、已完成、执行中、待执行。
  - Memory Ops 表格：IN/OUT、tensor name、shape、dtype、size、tier。

功能：

- 在播放过程中保持选中 op，并随当前 step 更新状态。
- 用 `scheduleIndexOf` 判断该 op 和当前 step 的关系。
- 用 `TENSOR_META` 与 `getTensorTier()` 展示 tensor 元信息。

交互：

- 点击计算图 op 打开浮层。
- 再次点击同一 op 关闭。
- 点击关闭按钮关闭。
- 点击页面非浮层、非节点区域关闭。
- 窗口 resize 时自动重新定位。

### 3.8 底部悬浮播放工具条

展示内容：

- 折叠/展开按钮。
- 折叠态播放按钮。
- 上一步按钮。
- Play/Pause 按钮。
- 下一步按钮。
- Replay 按钮。
- Timeline 标签。
- 当前 step / 最大 step。
- 当前 scrubber op name。
- range scrubber。
- scrubber hover tooltip。

功能：

- 控制全页面唯一 step 状态。
- 播放间隔固定为 1400ms。
- 到达最后一步自动停止。
- Replay 回到 step 0。
- 折叠态仍可快速展开或暂停播放。

交互：

- 点击 Play：从当前 step 开始播放；如果已在最后一步，则先回到 0。
- 点击 Pause：停止自动播放。
- 点击上一步/下一步：停止播放并跳到相邻 step。
- 拖动 scrubber：停止播放并跳转。
- scrubber hover：显示 `Step <n> · <op> #<magic>`。
- 点击 Replay：停止播放并回到 0。
- 点击折叠：工具条收起；点击折叠态按钮展开。

### 3.9 键盘快捷键

支持快捷键：

- `→` / `l`：下一步。
- `←` / `h`：上一步。
- `Space`：播放/暂停。
- `f`：适配计算图。
- `Home`：跳到第一步。
- `End`：跳到最后一步。

限制：

- 当事件目标是 input 时不处理快捷键，避免干扰 range 控件。

## 4. 数据处理规则

Schedule：

- `OP_DATA` 经过 `EXEC_OP_DATA` 过滤，去掉无输入输出的 `L1_ALLOC`、`L0A_ALLOC`、`L0B_ALLOC`、`L0C_ALLOC` 等 alloc marker。
- `SCHEDULE = EXEC_OP_DATA.map(op => op.m)`。
- `totalSteps = SCHEDULE.length`。

Tensor liveness：

- `tensorBorn`：tensor 第一次作为某 op 输出时的 step。
- `tensorDies`：tensor 最后一次作为某 op 输入时的 step。
- 无 consumer 的 tensor 生命周期持续到最后一步。
- 某一步 live tensor 满足 `born <= step <= dies`。

Tier 推导：

- 优先使用 `TENSOR_TOBE` 中的 tobe code。
- 若缺失，按 producer op 兜底推断：
  - `L1_TO_L0A` → L0A。
  - `L1_TO_L0B` → L0B。
  - Cube op → L0C。
  - `COPY_IN` → L1。
  - 其他 → DDR。

当前 step 的 active tensor：

- 当前 op 的输入和输出 tensor 合并为 active set。
- 输入用于 read 标记，输出用于 write 标记。

## 5. 状态与异常

页面状态：

- 初始加载：显示 graph loading 占位，step 初始化为 0。
- Graph 已加载：计算图可缩放、平移、染色和点击。
- 播放中：step 自动递增，Play 按钮变为 Pause。
- 暂停：保留当前 step。
- Op 已选择：显示 Selected Op Detail。
- 工具条折叠：保留播放入口。

异常与限制：

- 当前 demo 没有上传/切换数据入口；更换图需要替换 `data/sample-graph.json` 与 `data/ops.js`。
- 当前 schedule 使用数据文件顺序，不校验真实依赖拓扑是否完整。
- Buffer grid 是按 tensor magic 顺序顺排的容量近似视图，不表示真实硬件物理地址分配。
- MTE3 仅作为拓扑参考，当前 trace 不激活。
- UB vector ops 预留但当前样例为空。

## 6. 验收标准

- 页面加载后 step 显示为 `1 / <total>`，底部 scrubber 最大值为 `totalSteps - 1`。
- 点击 Play 后，每 1400ms step 前进一次，计算图、运行状态、buffer grid、MTE 通路、op detail 同步刷新。
- 点击上一步/下一步、拖动 scrubber、Home/End 快捷键均能准确跳转。
- Compute Graph 的 Fit、Zoom In、Zoom Out 和拖拽平移可用。
- Color Mode 五个选项可切换，图例和节点颜色同步变化；不可用模式应禁用。
- AIC/AIV 架构图在不同 step 显示对应 tier 的 live tensor、使用率、free 容量和 read/write 高亮。
- 悬停 buffer cell 能展示 tensor tooltip。
- 点击计算图 op 能打开详情浮层；详情中状态随 step 从待执行变为执行中/已完成。
- 分隔条拖动能改变上下分屏比例，且不会让任一区不可用。
- COPY_IN/COPY_OUT、L1_TO_L0A、L1_TO_L0B、A_MUL_B/A_MULACC_B 等 op 出现时，对应 MTE 或 Cube/FixPipe 高亮准确。

## 7. 业务核验依据

- 页面实现：`/Users/yin/pto/mem_viewer/index.html`、`/Users/yin/pto/mem_viewer/js/playback.js`、`/Users/yin/pto/mem_viewer/js/schedule.js`、`/Users/yin/pto/mem_viewer/js/constants.js`、`/Users/yin/pto/mem_viewer/js/buffer-grid.js`、`/Users/yin/pto/mem_viewer/js/memory-panel.js`、`/Users/yin/pto/mem_viewer/js/mte-overlay.js`、`/Users/yin/pto/mem_viewer/js/graph-viewer.js`、`/Users/yin/pto/mem_viewer/js/op-detail.js`。
- 当前数据：`/Users/yin/pto/mem_viewer/data/sample-graph.json`、`/Users/yin/pto/mem_viewer/data/ops.js`。
- PyPTO 业务依据：`/Users/yin/gitcode/pypto-master/docs/tutorials/debug/debug.md`、`/Users/yin/gitcode/pypto-master/docs/tools/computation_graph/查看计算图.md`、`/Users/yin/gitcode/pypto-master/framework/include/tilefwk/data_type.h`、`/Users/yin/gitcode/pypto-master/python/src/bindings/enum.cpp`、`/Users/yin/gitcode/pypto-master/framework/src/cost_model/simulation_ca/mock/mock_types.h`。
