# MindStudioNext

Ascend 性能分析工作台 — 本地 Demo。预置真实 Profiling 分析报告，4 标签页（总览 / 问题 / 代码 / 文档）联动展示，问题卡片按需渲染 Timeline 泳道、算子视图、通信视图等可视化组件。

---

## 启动方式

需要本地 HTTP 服务（直接双击 `index_v2.html` 因 ES module 限制无法运行）。

**Python（推荐）**

```bash
cd d:\Projects\msagent\MindStudioNext
python -m http.server 8080
# 或 uv：
C:\Users\24565\AppData\Roaming\uv\python\cpython-3.14-windows-x86_64-none\python.exe -m http.server 8080
```

然后浏览器打开 `http://localhost:8080/index_v2.html`

---

## 预置报告

`app.js` 的 `REPORTS` 数组内置五份真实 Profiling 场景：

| 报告 ID | 标题 | 卡数 |
|---|---|---|
| `r20260526` | Level2 4 卡训练诊断（PP=2/DP=2）| 6 |
| `r20260529op` | 自定义算子 matmul_leakyrelu 调优 | 6 |
| `r20260531single` | 单卡推理（eager 模式 CPU 瓶颈）| 8 |
| `r20260602multi` | 集群性能诊断（16 卡，多机多卡）| 7 |
| `r20260602grpo` | GRPO 训练 rollout 生成瓶颈 | 7 |

---

## 问题卡渲染规则

每张问题卡（`action`）有一个 `visualization` 字段，`renderIssueDetail()` 根据它和对应数据按顺序渲染以下视图组件，无数据时显示占位符提示所需文件：

### 1. Timeline 视图
- **触发**：`visualization` 包含 `"Timeline"`
- **数据源**（优先级从高到低）：
  1. `SWIMLANE_DATA[rid][aid]` — 手工构建的精简 trace JSON 片段
  2. `FREE_ANALYSIS_DATA[rid][aid]` → `buildFreeAnalysisSwimlane()` 动态转换
- **渲染**：`SwimlaneRenderer`（`swimlane-skill-pack` 组件）
- **无数据降级**：`⚠ 需要 trace_view.json / ascend_pytorch_profiler_*.db`

### 2. 算子视图
- **触发**：`visualization` 包含 `"算子视图"`
- **数据源**：`OP_VIEW_DATA[rid][aid]`
- **渲染**（按 `chartType` 分发）：
  - `chartType: 'table'` → `renderOpTable()` — 可展开父子行表格，问题行高亮
  - 其他 / 默认 → `renderOpPieCharts()` — 按算子类型 + 加速核两个饼图
- **无数据降级**：`⚠ 需要 kernel_details.csv 或 op_statistic.csv`

### 3. 通信视图
- **触发**：`visualization` 包含 `"通信视图"`
- **数据源**：`COMM_VIEW_DATA[rid][aid]`
- **渲染**（按 `chartType` 分发）：
  - `chartType: 'bw-table'`（默认）→ `renderCommTable()` — 16 卡 HCCS/RDMA 带宽热力表，问题列（AR/RS）标旗帜，AllReduce avg < 12 GB/s 的行加红色左边框
  - 未来可扩展：`'heatmap'` 等
- **无数据降级**：`⚠ 需要 cluster_analysis.db · ClusterCommunicationBandwidth 表`

### 4. 源码视图
- **触发**：`visualization` 包含 `"源码视图"`
- **数据源**：`SOURCE_VIEW_DATA[rid][aid]`（在 `app.js` 内声明，非 `chart-data.js`）
- **渲染**：`renderSourceView()` — 左右双栏布局
  - **左栏**：源码表格，列为行号 / cycle 占比 / 代码文本；热点行红色高亮，CACHEMISS 次数显示为 `MISS ×N` 角标
  - **右栏**：关联 SCALAR/VECTOR 汇编指令表，列为 PC 地址 / 源码行 / 指令类型 / 指令文本 / CACHEMISS 次数
- **典型场景**：
  - SCALAR 栈帧 CACHEMISS —— `per_core_event.csv` PC 热图反汇编定位到 `.cpp` 行
  - 高 cycle 函数体 —— `pipe_instr_top.csv` 逐行 cycle 占比标注
  - 内联失败的函数调用开销 —— `nm`/`objdump` 反汇编验证
- **无数据降级**：`⚠ 需要 visualize_data.bin（simulator 模式）含调试符号`
- **接入示例**：`r20260527` / `actionId=6`（CACHEMISS 累计 326 次，定位到 `matmul_leakyrelu_custom.cpp:206-207`）

### 问题行/列标注约定（通信视图）

| 标注 | 含义 | 触发条件 |
|---|---|---|
| 列头 `⚑` + 红色背景 | 问题列 | `entry.problemCols` 包含该列 key |
| 行左红线 | 问题行 | `d.ar_avg < 12.0` GB/s |
| 格子色 | 带宽效率分层 | < 13 红 / 13–16 橙 / 16–19 绿 / ≥ 19 蓝 |
| 小包标签 | 时延主导 | `bandwidth < 1 GB/s` |

---

## 数据层结构

所有可视化数据在 `chart-data.js` 中声明，挂在 `window.*` 上：

```
window.SWIMLANE_DATA       — Timeline 泳道数据（精简 trace JSON）
window.FREE_ANALYSIS_DATA  — FreeAnalysis 统计数据（自动转 Timeline）
window.OP_VIEW_DATA        — 算子视图数据（rows / byType / byCore + chartType）
window.COMM_VIEW_DATA      — 通信视图数据（rows + chartType + theoryBw + problemCols）
```

`SOURCE_VIEW_DATA` 是例外：因数据量小且与报告 issue 强绑定，**直接在 `app.js` 内声明**，不走 `chart-data.js`。结构：

```
SOURCE_VIEW_DATA[rid][aid] = {
  source:      string,   // 来源描述，显示在视图标题行
  file:        string,   // 源文件名（如 my_kernel.cpp）
  contextNote: string,   // 简短上下文说明
  lines: [               // 展示的源码行（仅含问题上下文，不需全文件）
    { ln, cycles, hot, code, cachemiss? }
  ],
  instrs: [              // 热点汇编指令
    { pc, ln, type, instr, cachemiss }
  ],
}
```

**添加新视图数据的最小步骤**：

**Timeline / 算子视图 / 通信视图**：
1. 在 `chart-data.js` 对应的 `window.*` 对象中加入 `{ [rid]: { [aid]: { source, chartType, ...} } }`
2. 确保对应 `action.visualization` 字段包含触发关键词
3. 刷新页面，该问题卡底部自动渲染

**源码视图**：
1. 在 `REPORTS[n].issues` 补充 `{ id: '3.N', ... }` 条目（含 `visualization: '…源码视图…'`）
2. 在 `app.js` 的 `SOURCE_VIEW_DATA` 中按 `rid → aid` 填入 `lines[]` + `instrs[]`（`code` 字段需 HTML 转义）
3. 刷新页面，`renderIssueDetail` 自动检测并渲染；CSS 无需改动

---

## 目录结构

```
MindStudioNext/
├── index_v2.html   # 主应用页面（当前版本）
├── index.html      # 旧版（保留）
├── app.js          # 报告数据 + 全部渲染逻辑
├── chart-data.js   # 可视化数据（SWIMLANE / OP_VIEW / COMM_VIEW / FREE_ANALYSIS）
├── styles.css      # 暗色主题 + 各视图组件样式
├── build.js        # 打包脚本（合并为单文件 dist/）
├── dist/           # 历史打包快照
├── key-features.html  # 特性说明页
└── comm_view.html  # 通信视图独立调试页（开发用）
```

---

## 依赖（全部 CDN，无需安装）

- [ECharts](https://echarts.apache.org/) — 算子视图饼图
- [marked.js](https://marked.js.org/) — Markdown → HTML
- `SwimlaneRenderer`（swimlane-skill-pack 本地组件）— Timeline 泳道渲染
