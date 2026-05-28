# Pass-IR Demo 页面功能 PRD

## 1. 背景与定位

Pass-IR Demo 是 PyPTO 编译阶段计算图查看器，用于查看 `compile_debug_mode` 输出的 pass 前后 JSON 图文件，帮助算子开发者理解 Tensor Graph、Tile Graph、Block Graph、Execute Graph 在编译流程中的变化。页面重点不是运行时性能播放，而是围绕“某个 pass、某条 PATH、某个 snapshot”查看计算图结构、节点属性、数据流上下游和控制流映射。

业务口径需与 PyPTO 保持一致：

- 计算图由 Tensor 节点和 Operation 节点构成，以 DAG 表示数据流和计算逻辑。
- PyPTO 编译过程包含 Tensor Graph、Tile Graph、Block Graph、Execute Graph 等阶段。
- Tile Graph 会体现 Tile 展开、内存层级推导及搬运节点插入；Block Graph 是 Tile Graph 切分后的 AI Core 可调度子图；Execute Graph 包含对 Block Graph 的调用节点。
- Tensor/Incast/Outcast 的 `asis`、`tobe` 等内存字段应按 PyPTO `MemoryType` 理解，包括 `MEM_UB`、`MEM_L1`、`MEM_L0A`、`MEM_L0B`、`MEM_L0C`、`MEM_DEVICE_DDR` 等。

## 2. 目标用户与使用场景

目标用户：

- PyPTO 算子开发者：定位 pass 前后图结构是否符合预期。
- 编译/性能调优人员：观察 Tile 展开、子图切分、内存层级、搬运节点和热点路径。
- Demo 讲解人员：用样例图快速展示 PTO 计算图查看能力。

核心场景：

- 打开单个 `.json` 计算图，查看节点、边、属性和上下游。
- 打开一个 PyPTO 输出目录，按 pass 时间线浏览不同 pass 的 `Before/After` 图。
- 在多 PATH、多 unroll、多 ROOT/LEAF snapshot 中切换，观察同一函数在不同编译分支下的图形态。
- 从控制流面板选择编译 PATH，并同步定位到对应图 snapshot。
- 对某个节点锁定上下游链路，只看与该节点相关的计算流。

## 3. 输入数据与加载规则

页面支持以下输入：

- 样例数据：`samples/mini-attn.json`，当前样例为 `TENSOR_LOOP_RESHAPE_Unroll6`，包含 raw tensor、tensor、VIEW、RESHAPE、ASSEMBLE、incast、outcast 等结构。
- 单个 JSON 文件：通过顶部菜单、空态按钮、URL `?file=` 或拖拽文件加载。
- 本地 pass 输出目录：通过目录选择器或 `webkitdirectory` 扫描目录内 JSON 文件，按文件名解析 `Pass_xx_xxx/Before|After_xxx_...json`。
- 外部桥接数据：支持从其它 PTO 页面通过 sessionStorage/message 传递 program graph 与 focus 信息。
- 最近一次打开的图：使用 `localStorage` 缓存最近 JSON 与显示名。

目录解析规则：

- 只纳入 `.json` 文件。
- pass 目录需符合 `Pass_<序号>_<PassName>`。
- snapshot 文件支持 `main`、`ROOT`、`LEAF_program_id_xx`。
- PATH 由文件名中的 `PATH<major>_<minor>` 得出。
- 当前页面侧边 `Before/After` 选择默认隐藏，实际导航固定优先使用 `after`，如果缺失则 fallback 到 `before`。

## 4. 页面内容

### 4.1 顶部工具栏

展示内容：

- 返回主页按钮。
- 模块品牌：Pass-IR。
- 当前图标题 `graphTitle`。
- 当前图统计 `graphStats`：incast、ops、tensors、groups、outcast 数量。
- 打开入口按钮：Open Pass Folder。
- 图加载下拉菜单：样例图、打开本地文件、打开 pass folder。
- 链路锁定退出按钮：仅在 Locked Flow 模式显示。

功能：

- 作为页面全局入口，承载加载、状态显示和锁定模式退出。
- 成功加载图后，标题显示当前 graph name；进入聚合视图或锁定链路时追加状态后缀。
- 统计随当前视图变化：原始图、聚合图、锁定子图统计不同。

交互：

- 点击 `Open Pass Folder` 展开菜单。
- 点击样例项加载 `mini-attn`。
- 点击 `Open Local File...` 触发单文件选择。
- 点击 `Open Pass Folder...` 触发目录选择。
- 点击“退出链路锁定”恢复完整图视图。

### 4.2 空态入口

展示内容：

- 标题：Compute Graph Viewer。
- 副标题：Visualize Ascend NPU computation graphs。
- 主按钮：Open Pass Folder。
- 次按钮：Open single JSON file。
- Recent 区：最近加载文件名。
- Samples 区：Mini Self-Attention。
- 提示：支持拖拽 `.json` 文件。

功能：

- 在未加载图时提供所有主要入口。
- 最近记录用于快速恢复上一次查看的图。

交互：

- 点击主按钮打开目录。
- 点击次按钮打开单个文件。
- 点击 Recent 从 localStorage 读取最近图并加载。
- 点击 Sample 加载样例图。
- 将 `.json` 拖入画布可直接加载。

### 4.3 Pass 导航条

展示内容：

- Loop 选择：`RESHAPE`、`MAIN` 等，根据本地目录索引动态生成。
- Unroll 选择：当前实现识别 `×32/×16/×8/×4/×2/×1`，仅 MAIN loop 下显示。
- Pass 时间线：按阶段分组展示 pass dot，阶段包括 Tensor、Tile、Block、Execute。
- Source 标签：显示 nav index 的来源目录。
- Snapshot 选择：`main`、`root`、`leaf<program_id>`。

功能：

- 将本地 pass 输出目录转化为可点击的编译流程导航。
- 支持从某个 pass dot 切换到同一路径下该 pass 的 snapshot。
- 支持在同一 pass 中切换 loop、unroll 和 ROOT/LEAF snapshot。
- 导航选择后通过 `window.loadFile(fileRef)` 加载对应 JSON。

交互：

- 点击 pass dot 加载该 pass 对应 snapshot，并滚动居中 active dot。
- 鼠标悬停 pass dot 显示 `Pxx · PassName` tooltip。
- 点击阶段 chip 跳转到该阶段第一个 pass。
- 点击 loop/unroll/snapshot pill 展开菜单；选择后重新解析目标 fileRef 并加载。
- 点击控制流面板中的 PATH 会同步调用 `navSelectPath(pathId)`，刷新 loop/unroll 状态。

### 4.4 控制流面板

展示内容：

- 面板标题：Control Flow。
- 左列：Source `.py` 控制流结构。
- 右列：Compiled PATHs。
- 中间 SVG 曲线：源代码结构与编译 PATH 的映射关系。
- 收起按钮与顶部重开按钮。

功能：

- 展示源码控制流与编译展开 PATH 的对应关系。
- 支持从编译 PATH 反向驱动 Pass 导航条选择。
- 选中某条 PATH 后高亮对应源码节点、编译节点和映射线。

交互：

- 点击右侧 compiled PATH 节点：高亮该 PATH，并同步切换 Pass 导航的 path。
- 点击收起：隐藏控制流面板，顶部显示重开按钮。
- 点击重开：恢复控制流面板并重绘映射线。
- 浏览器 resize 时重新计算映射线位置。

### 4.5 主图画布

展示内容：

- 计算图节点卡片：Incast、Tensor、Operation、Outcast、Group。
- 边：按数据流方向连接 Tensor 与 Operation。
- 画布背景与可拖拽区域。
- 大图模式下的虚拟化渲染结果。

功能：

- 将 PyPTO JSON 解析为 graph model，并通过 Sugiyama 分层布局展示。
- 节点数据展示：
  - Incast/Outcast：slot、shape、rawshape、offset、asis、dtype。
  - Tensor：dtype、magic、symbol、shape、rawshape、asis、offset。
  - Operation：subgraph、magic、opcode/semantic label、latency、shape、from/to tensor。
  - Group：聚合后的成员数量、结构摘要、语义/子图/内存提示。
- 支持原始图和聚合图两种视图。聚合图会把相同局部结构、相同 flow signature 的重复节点压缩成 group。
- 对超大图启用性能保护：节点/边数量超过阈值时使用紧凑布局、视口虚拟化；低缩放比例下隐藏边。

交互：

- 鼠标拖拽空白区域平移画布。
- `Ctrl/Meta + 鼠标滚轮` 按鼠标位置缩放。
- 触摸单指平移、双指缩放。
- 点击节点：选中节点，关联边高亮，打开节点详情浮层。
- 点击画布空白：关闭详情并清除选中。
- 键盘：`Esc` 清空选择；`f/F` 适配视图；`+`/`=` 放大；`-` 缩小。

### 4.6 颜色与视图面板

展示内容：

- View：
  - 原始：full node graph。
  - 聚合：current mode。
- Color：
  - Semantic：按算子语义、边界节点和 pipeline label 着色。
  - Latency：按 operation latency 梯度着色。
  - Compact：当前实现对应 subgraph coloring，按子图/边界分组着色。
  - None：中性色。
- Legend：显示当前颜色模式下的图例和数量。
- 折叠/展开按钮。

功能：

- 控制主图节点颜色和视图粒度。
- 对不可用模式自动禁用：
  - 无子图信息时禁用 subgraph/Compact。
  - 无 latency 时禁用 Latency。
  - 无可聚合节点时禁用聚合视图。
- 颜色模式变化时重建 color map、刷新 legend、清空当前详情选择。

交互：

- 点击 View 选项切换原始/聚合视图。
- 点击 Color 选项切换着色模式。
- 点击折叠按钮收起面板，状态写入 localStorage。
- 支持接收外部 message 同步颜色模式。

### 4.7 缩略图与缩放控件

展示内容：

- 缩略图 canvas。
- 当前视口框。
- 缩放控制按钮：放大、缩小、适配。
- 折叠/展开按钮。

功能：

- 提供全图方位感，尤其用于大图浏览。
- 缩略图使用当前节点颜色绘制，随视图模式、颜色模式、缩放和平移更新。
- 折叠状态写入 localStorage。

交互：

- 点击 `+`、`-` 以画布中心缩放。
- 点击 Fit 适配全图。
- 点击折叠按钮隐藏/显示缩略图。
- 画布平移缩放时实时更新视口框。

### 4.8 节点详情面板

展示内容：

- 类型 badge：INCAST、OUTCAST、OP、TENSOR、GROUP。
- 节点名称。
- 锁定计算流按钮。
- 关闭按钮。
- 节点属性分区：
  - Operation：semantic、opcode、magic、kind、latency、subgraph、attributes。
  - Tensor：symbol、magic、shape、dtype、kind、mem_id、raw connections。
  - Group：title、group_type、members、semantic、reason、subgraph、latency_avg、engine_or_mem、shape、mem。
- 输入/输出连接 chips。

功能：

- 给出当前节点的业务属性和上下游关系。
- 连接 chips 支持跳转到相邻节点或打开相邻节点详情。
- 详情面板会根据选中节点在视口中的位置自动贴近节点，并避免溢出视口。

交互：

- 点击节点打开详情。
- 点击详情中的 Inputs/Outputs chip：定位或切换详情到目标节点。
- 点击关闭按钮或画布空白关闭。
- 点击锁定按钮进入 Locked Flow。

### 4.9 Locked Flow 模式

展示内容：

- 顶部显示退出链路锁定按钮。
- 主画布切换为以当前节点为锚点的上下游子图。
- 标题追加 `locked flow`。

功能：

- 从选中节点同时向上游和下游收集可达节点，生成局部子图。
- 局部子图用于在复杂全图中聚焦一条计算链路。
- 若在聚合视图中触发锁定，页面会回到原始视图再锁定。

交互：

- 点击详情面板锁定按钮：进入 Locked Flow。
- 再次点击锁定按钮、顶部退出按钮或 flow unlock：退出 Locked Flow。
- 退出后尝试回到锚点节点并保留详情打开状态。

## 5. 状态与异常

页面状态：

- 空态：无 graph 数据。
- 已加载单图：显示主图、颜色面板、缩略图和详情交互。
- 已加载 pass folder：额外显示 Pass 导航条和控制流面板。
- 聚合视图：显示 group 节点和 group 统计。
- Locked Flow：显示局部上下游子图。
- 大图模式：启用紧凑布局、虚拟化渲染和低缩放隐藏边。

异常处理：

- JSON parse 失败：弹窗提示解析失败。
- 样例加载失败：提示需通过本地 server 运行。
- 目录中没有 JSON：弹窗提示。
- 目录不符合 PyPTO pass 输出命名：弹窗提示。
- URL 文件加载失败：保留空态并在 console 记录错误。

## 6. 验收标准

- 能加载 `samples/mini-attn.json`，并展示 1 个 incast、13 个 op、12 个普通 tensor、1 个 outcast 的基本图形结构。
- 能加载单个 PyPTO 计算图 JSON，并正确展示 graph title、节点数量统计、节点卡片和边。
- 能打开 pass 输出目录，并按 pass 序号生成时间线，支持 pass dot、loop/unroll、ROOT/LEAF snapshot 切换。
- 控制流面板点击 PATH 后，Pass 导航与主图应同步切换。
- 点击任一节点后，应高亮节点、关联边，并展示节点详情。
- 详情面板的 Inputs/Outputs chip 能继续跳转关联节点。
- Fit、放大、缩小、平移、键盘快捷键、minimap 折叠状态可正常工作。
- Color Mode 切换后，节点颜色和 legend 应同步刷新；无数据支撑的模式应禁用。
- Locked Flow 能生成当前节点上下游子图，并可退出回到完整图。

## 7. 业务核验依据

- 页面实现：`/Users/yin/pto/pass-ir/index 18.19.27 18.19.27.html`、`/Users/yin/pto/js/app.js`、`/Users/yin/pto/js/parser.js`、`/Users/yin/pto/js/nav.js`、`/Users/yin/pto/js/controlflow.js`、`/Users/yin/pto/js/renderer.js`。
- 样例数据：`/Users/yin/pto/pass-ir/samples/mini-attn.json`。
- PyPTO 业务依据：`/Users/yin/gitcode/pypto-master/docs/tutorials/debug/debug.md`、`/Users/yin/gitcode/pypto-master/docs/tools/computation_graph/查看计算图.md`、`/Users/yin/gitcode/pypto-master/framework/src/passes/pass_mgr/pass_manager.cpp`、`/Users/yin/gitcode/pypto-master/framework/include/tilefwk/data_type.h`。
