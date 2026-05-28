# A5 PMU Group 2 Pipeline Balance 诊断工作台 PRD

## 项目背景

A5 / dav_3510 的 PMU profiling 一次运行只能采集一个 PMU group。现有 `06-a5-pmu-visualization-group2-loop.html` 是 Group 2 Pipeline Balance 的 L3 demo，用合成 trace 展示了从泳道任务、PMU counter composition、critical wrap、源码/CALL/Block Graph/硬件路径到下一次采集建议的完整诊断闭环。

Group 2 关注的问题是：在 AIC / AIV0 / AIV1 的 wrap 组合中，谁占用了流水、谁拖尾、拖尾是否能被 cycle-like counter 解释。原型中一个 run 包含 24 个 wrap，每个 wrap 关联 1 条 AIC lane 和 2 条 AIV lane，并辅以 AI CPU 调度 lane、MTE in/out lane。关键 PMU counter 包括 `cube_instr_busy`、`pmu_idc_aic_vec_busy_o`、`scalar_instr_busy`、`mte1_instr_busy`、`mte2_instr_busy`、`mte3_instr_busy`、`pmu_fix_instr_busy`；`icache_req`、`icache_miss` 是事件计数，不参与 `clc_cycle` 求和。

本项目目标是把该 L3 demo 收敛成可工程化的 PMU 诊断工作台页面：保留 Group 2 的诊断模型和交互路径，迁移到 PTO 设计系统与可拖拽 IDE 工作台框架，最终支持真实 trace 数据接入和后续 PMU group 的扩展。

## 典型用户

- Kernel 性能工程师：定位 matmul / conv / vector kernel 中的流水占用、拖尾和疑似 wait。
- 编译器与算子优化工程师：把 PMU counter 反向关联到源码行、CALL 节点、Block Graph 节点和硬件路径。
- 模型性能 triage 负责人：快速判断一次 run 的首要瓶颈 wrap、critical lane 和下一次采集 group。
- PMU / 架构维护者：维护 counter 到 pipe、下一采集 group、A5 硬件拓扑的解释映射。

## User Scenarios

1. 用户打开一次已采集 Group 2 的 profiling run，首先在左侧 Immediate Attention 看到 explain ratio、critical lane、top suspect wrap、dominant counter 和下一次采集建议。
2. 用户在默认模式下按 lane 类型浏览全局时间线，确认 AIC、AIV、MTE、AI CPU 调度任务的整体分布。
3. 用户打开 Mix Mode，泳道改为按 wrap 分组显示，每组展示 AIC + 两条 AIV peer lane，并用 Group 2 counter composition 标出 compute、scalar、MTE1/2/3、FixPipe、uncovered 和 overlap。
4. 用户点击 critical marker、泳道 task 或 Top-3 Wrap 列表，右侧 Detail 面板给出 conclusion-first 诊断：`wait`、`overlap` 或 `accounted`，并展示 `clc`、`total`、`gap_ratio`、dominant counter、counter breakdown 和同 wrap peer task。
5. 用户从 Detail 面板一跳打开源码片段、CALL 定位、Block Graph、硬件路径，验证 dominant counter 指向的代码和硬件链路。
6. 用户点击“下一次采集”，复制如 `PROF_PMU_EVENT_TYPE=4` 的命令，用 dominant counter 的 next-group 规则继续采样闭环。
7. 用户切换到未采集的 PMU group 时，页面不展示伪数据，而是给出空状态、当前已采集 group 和重新采集命令。
8. 用户在宽屏、VS Code webview 或独立页面中拖拽调整左摘要、中心泳道、右详情面板宽度，布局稳定且不丢失选中上下文。

## Key Features

### 1. 数据接入与标准模型

- 定义 run / chip / trace / lane / wrap / task / PMU group / trace link 的前端数据 schema。
- 支持 A5 / dav_3510 Group 2 数据接入，至少覆盖 24 wrap、AIC lane、AIV lane、AI CPU lane、MTE in/out lane。
- task 必须包含 `id`、`laneId`、`laneKind`、`wrapId`、`opType`、`opName`、`start`、`end`、`totalCycle`、`isWrapPrimary`、`g2`、`traceLinks`。
- `traceLinks` 至少包含源码路径与热行、`callOpMagic`、`leafHash`，用于后续一跳定位。
- 未采集 group 不允许合成 counter；必须进入明确空状态。

### 2. Group 2 指标推导

- `clc_cycle` 等于 Group 2 cycle-like counters 的求和；事件计数不参与求和。
- `gap = clc_cycle - totalCycle`，`gapRatio = gap / totalCycle`。
- `gapRatio < -0.10` 判定为 `wait`：当前 group 不能解释全部 task 生命周期，疑似 uncovered / wait / idle / sync / 未统计 stall。
- `gapRatio > 0.10` 判定为 `overlap`：cycle-like 求和超过 task 生命周期，疑似 pipeline overlap 或 non-exclusive counter double-count，不能直接解读为“更好”。
- 其余状态判定为 `accounted`：当前 group 对 task 生命周期解释充分。
- dominant counter 取 cycle-like composition 中 value 最大的 counter，并映射到 pipe、解释文案和 next group。
- wrap 级别需要推导 critical task、peer task、tail length、uncovered ratio、status、dominant counter。
- Summary 需要输出 critical lane、Top-3 wraps、overall dominant counter、overall explain ratio。

### 3. Immediate Attention 面板

- 左侧面板始终 conclusion-first，默认展示当前 group 的核心结论。
- 必须展示 explain ratio、critical lane、top suspect wrap、dominant counter、硬件路径摘要、Top-3 wraps。
- Top-3 wraps 按 wrap severity 排序，点击后选中对应 critical task 并滚动泳道定位。
- 如果存在 next-group suggestion，展示原因和可复制命令。
- 对未采集 group 展示空状态、采集命令和回到已采集 Group 2 的入口。

### 4. Swimlane 视图

- 默认模式按 lane 类型排序：AI CPU Ctrl / Sched、AIC、AIV、MTE in/out。
- Mix Mode 按 wrap 排序：每个 wrap 展示 AIC、AIV0、AIV1，关键 wrap 需要有明显 marker。
- 支持 zoom in、zoom out、fit，目标范围为 `0.5x` 到 `4x`。
- 支持 task hover tooltip，内容包含 op、lane、timing、status、dominant counter、wrap metadata。
- 支持 task / marker 点击选中，选中态需同步 Detail 面板。
- 默认着色按 op identity；Mix Mode 在保留 task identity 的基础上展示 Group 2 composition rail。
- `wait` task 需要展示 uncovered striped segment；`overlap` task 需要展示 overlap overlay 和溢出提示。

### 4.1 Swimlane 取色与图例机制

- Swimlane 的基础 task bar 取色必须来自 `swimlane-task` pattern 提供的共享 colormap，不允许在 PMU 页面内硬编码一套独立 task 色板。
- 默认模式用于回答“哪些 op 在什么 lane 上发生”：task 主体颜色按稳定的 op identity 分配，取色 key 优先来自 `opType`，其次为 `opName` / `label`；同义 op 需要归并到同一语义色，例如 `conv` / `matmul_trans` 归并到 `matmul`，`vec_cast` 归并到 `vec_elementwise`，`mte_store` 归并到 `mte_load`，`cpu_ctrl` 归并到 `cpu_sched`。
- 默认模式图例显示 identity legend，至少覆盖 `matmul / conv`、`vec_softmax`、`vec_layernorm`、`vec_elementwise`、`vec_reduce`、`mte_load / store`、`cpu_sched / ctrl`。
- Mix Mode 用于回答“同一个 wrap 内 AIC / AIV peer 的生命周期由哪些 Group 2 counter 解释”：task 主体仍保留 op identity 色，避免用户丢失任务身份；额外在 task 底部显示 composition rail。
- Mix composition rail 的颜色按 Group 2 cycle-like counter family 分配：`compute` 对应 `cube_instr_busy` / `pmu_idc_aic_vec_busy_o`，`scalar` 对应 `scalar_instr_busy`，`mte1` 对应 `mte1_instr_busy`，`mte2` 对应 `mte2_instr_busy`，`mte3` 对应 `mte3_instr_busy`，`fixpipe` 对应 `pmu_fix_instr_busy`。
- `accounted` 状态下，rail segment 宽度按各 counter value 在 `clc_cycle` 中的占比分配，并铺满 task 宽度。
- `wait` 状态下，cycle-like segment 只能占据 `clc_cycle / totalCycle` 对应的前缀宽度，剩余未解释尾段显示为 striped `uncovered`，表示当前 Group 2 不能解释全部 task 生命周期。
- `overlap` 状态下，rail 展示 composition 后叠加 `overlap` striped overlay，并在 task 右侧用 overflow marker 表示 `clc_cycle > totalCycle` 的超出量；该状态表示 pipeline overlap 或 non-exclusive counter double-count 风险，不能解释为“性能更好”。
- `icache_req`、`icache_miss` 等 event counter 不参与 `clc_cycle`，也不进入 composition rail；它们只出现在 Detail 的 event counter 区域。
- Swimlane 图例必须随模式切换：默认模式显示 shared identity colormap；Mix Mode 显示 `Mix · group 2 composition`，包含 compute、scalar、MTE1/2/3、FixPipe、uncovered、overlap，并用 striped swatch 表达 uncovered / overlap 的非普通占比语义。

### 5. Detail 面板

- 右侧 Detail 面板必须按 conclusion、counter breakdown、hardware path、peer tasks、jump actions、next suggestion 排列。
- 未选中 task 时展示可操作空状态，引导用户点击泳道或 Top-3 Wrap。
- 对无 PMU 数据的 lane 展示 No PMU 解释，不进入 Group 2 误判。
- Counter breakdown 区分 cycle-like counters 与 event counters，并突出 dominant counter。
- Peer tasks 展示同 wrap 的 AIC / AIV primary task，点击 peer 后切换选中并滚动定位。
- Jump actions 至少包含源码、CALL、Block Graph、硬件路径、Peer 视图、下一次采集。

### 6. 一跳定位与诊断闭环

- 源码定位在生产版本中必须跳转真实 kernel 源码和 inline 展开点；webview 内可提供只读预览，但真实编辑交给 VS Code editor。
- CALL 视图必须基于 `callOpMagic` 定位真实调用节点，并支持上下游展开。
- Block Graph 必须基于真实 IR / Block IR 节点和边生成，当前 demo 的示意图不能作为生产数据。
- 硬件路径必须使用 A5 / dav_3510 的真实拓扑和 counter 到 pipe 的映射。
- next-group suggestion 必须来自可维护的 counter meta 表，不写死在 UI 文案里。

## 设计系统调用

### tokens / css

- 页面必须加载 PTO 设计系统根 tokens：`tokens/foundation.css`、`tokens/semantic.css`、`tokens/components.css`、`css/style.css`。
- 页面局部 CSS 只能定义 PMU 数据可视化语义，如 lane tint、counter family、状态 overlay；通用 surface、font、border、button、input、chip、panel 样式必须使用设计系统 token 或组件类。
- 不允许复制设计系统文件、创建私有 theme、引入远程 framework runtime 或一次性视觉资产。
- 当前原型中的硬编码颜色需要收敛为语义 token、组件 token或 PMU data-viz 专用变量；PMU 专用变量需要集中定义并记录用途。
- Buttons、segmented control、select、panel、tag、copy action 等常规控件优先复用 PTO 组件类。

### swimlane-task

- 必须加载 `patterns/swimlane-task/pattern.css` 和 `patterns/swimlane-task/pattern.js`。
- task bar 基础绘制必须调用 `window.PtoSwimlaneTaskPattern.drawTaskBar`；不可用 DOM/CSS 在业务页重建任务条。
- tooltip 必须调用 `window.PtoSwimlaneTaskPattern.initHoverTooltip` 和 `formatTaskTooltip`。
- 如需要组合 segment spec，优先调用 `window.PtoSwimlaneTaskPattern.buildTaskSegmentSpec`，不要在页面里 fork 共享 task bar 的 segment math、alpha、font threshold、文本截断规则。
- task identity 取色必须调用 `window.PtoSwimlaneTaskPattern.createTaskColormap` 或共享 pattern 暴露的等价 API；PMU 页面只允许声明 op alias、legend key 和 Group 2 counter family 映射，不允许复制 colormap 算法。
- 业务页允许传入 pattern 允许覆盖的 task 字段，如 `opName`、`laneKind`、`laneId`、`totalCycle`、`clcCycle`、`gap`、`gapRatio`、`status`、`dominantCounter`、`wrapId`。
- PMU composition rail 属于 Group 2 domain overlay；若该 overlay 需要沉淀为通用 swimlane 行为，必须先扩展共享 pattern，再由业务页调用。

### ide-frame / workbench-shell 拖拽迁移

- 当前原型的手写三栏 `.workspace`、`.left-panel`、`.center-panel`、`.right-panel` 需要迁移到 `patterns/ide-frame` 作为上层 IDE shell。
- 页面需要加载 ide-frame 依赖：`patterns/workbench-shell/pattern.css`、`patterns/workbench-shell/pattern.js`、`patterns/floating-playback-control/pattern.css`、`patterns/floating-playback-control/pattern.js`，再调用 `window.PtoIdeFrame.init(root, options)` 或 `initAll()`。
- 三个业务 pane 映射为 workbench split：左侧 Immediate Attention、中心 Swimlane renderer、右侧 Detail inspector。
- 初始 pane 尺寸建议沿用原型意图：左侧约 `280px`、右侧约 `400px`、中心填充；独立页面中 explorer 默认遵循 ide-frame 的 `300px` 约定。
- 拖拽 resize 必须由 `workbench-shell` 提供，不允许在 PMU 页面局部重写 `.pto-workbench-shell__*` 内部样式或自建 resize kernel。
- 独立页面可展示 PTO topbar、window controls、activity rail、pane-local tabs 和局部 status strip；VS Code webview 模式必须隐藏独立 chrome，依赖 VS Code 原生 explorer、tabs、status bar、search、git、terminal、settings、command palette、theme colors 和字体。
- Preview / editor tab 只能放在 pane 内部，不新增独立顶层 chrome band。
- ide-frame 只提供框架 shell，不承载业务样例数据、placeholder kernel、timeline lane、trace 节点或 inspector 文案。

## 约束 / 非目标

- 不在本阶段实现 PMU 数据采集器、profiler runtime 或后端存储。
- 不在本阶段支持所有芯片；首版只要求 A5 / dav_3510，其他 chip 展示 disabled 或 coming soon。
- 不把 L3 demo 的合成数据、伪源码、伪 CALL tree、伪 Block Graph、伪硬件拓扑作为生产事实。
- 不在 webview 内编辑源码；源码编辑必须路由到 VS Code editor。
- 不一次性实现所有 PMU group 的完整工作台；非 Group 2 可以先有 registry、空状态和采集建议。
- 不在 PMU 页面复制或修改 PTO 设计系统共享 pattern 源码。
- 不为了单个 demo 新增一次性 UI 风格；新增视觉需求必须先评估是否应进入 PTO 设计系统或 pattern。
- 不保证 counter 解释文案的架构正确性，除非由 PMU / 架构维护者确认映射表。

## 验收标准

1. 给定一份 Group 2 trace 数据，页面能渲染 Immediate Attention、Swimlane、Detail 三个 pane，首屏无 console error。
2. `clc_cycle`、`gap`、`gapRatio`、`wait`、`overlap`、`accounted`、dominant counter、Top-3 wraps 的推导有单元测试覆盖，事件计数不参与 `clc_cycle`。
3. 默认模式和 Mix Mode 均可在 78 条 lane、24 个 wrap、10ms 时间范围内稳定渲染；zoom `0.5x` 到 `4x` 不导致文本重叠或任务条消失。
4. task hover tooltip 使用 `swimlane-task` pattern；基础 task bar 由 `drawTaskBar` 绘制，不存在页面局部 DOM/CSS 复刻。
5. 点击 marker、task、Top-3 Wrap、peer row 会同步选中态、滚动定位和 Detail 内容。
6. Detail 面板能正确区分有 PMU 数据与无 PMU 数据的 lane，并展示 conclusion、counter breakdown、hardware path 摘要、peer tasks、jump actions、next suggestion。
7. 未采集 PMU group 展示空状态和 `PROF_PMU_EVENT_TYPE=<group>` 采集命令，不显示伪诊断。
8. 源码、CALL、Block Graph、硬件路径入口在真实数据缺失时明确标记为 demo / unavailable，不误导为已验证事实。
9. 页面迁移到 `ide-frame` 后，在 standalone 和 `data-host="vscode-webview"` 两种模式下都符合 host contract；VS Code webview 不出现 fake explorer、fake editor tabs 或 fake global status bar。
10. 左 / 中 / 右 pane 拖拽 resize 由 `workbench-shell` 驱动，刷新或状态重渲染不破坏当前尺寸、选中 task 和 scroll 定位。
11. 所有常规视觉 token 来自 PTO design system；PMU 专用 data-viz 颜色集中声明，无私有主题文件、远程 runtime 或共享 pattern copy。
12. 通过桌面宽屏和最小支持宽度的截图检查，确认 topbar、summary、swimlane、detail、modal、tooltip、copy toast 无遮挡、溢出或不可读文本。
