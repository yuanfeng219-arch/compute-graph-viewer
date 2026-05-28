# Whitepaper Generator

把资料文件生成一份图文并茂的 HTML 白皮书报告，风格与 `/Users/yin/pto/计算领域竞分/index.html` 一致。

## 用法

```
/whitepaper <source-file> [output-dir]
```

- `source-file`: 原始资料（md / txt）
- `output-dir`: 可选，默认在 source-file 同目录下，文件名 `<basename>-whitepaper.html`

---

## 文件结构

每次生成两个文件：

```
output-dir/
├── <name>-whitepaper.html
└── shared.css   ← 若目录里没有，从 /Users/yin/pto/计算领域竞分/shared.css 复制
```

HTML 头部固定引用：
```html
<link rel="stylesheet" href="./shared.css">
```

---

## 布局与导航规则（必须遵守）

- 楼层导航点击后，section 标题必须出现在窗口顶部附近，不能垂直居中到页面中部。
- `section.floor` 默认使用 `justify-content: flex-start`，不要用 `justify-content: center` 承载长内容。
- 禁用桌面端 `scroll-snap-type`；长报告内容高度不固定，scroll snap 会导致锚点跳转后内容显示不全。
- 固定顶栏场景中，section 自身使用顶部 padding 避开 nav；不要再叠加大的 `scroll-margin-top`，否则标题会被推到视口中部。
- 交互图、长表格、代码块、图片组所在楼层允许超过一屏高度，优先保证内容完整，不强求每个楼层刚好一屏。
- 图片默认完整显示，优先 `object-fit: contain`，除非明确是封面背景或裁剪预览。

---

## Step 1 — 读资料，提炼结构

从 source-file 中提取：

- **主题** 和目标读者
- **关键概念**（5-10 个，成为各 section 的核心）
- **核心洞察**（读完必须记住的一件事）
- **数据 / 公式 / 数字** → `callout` 组件
- **流程 / 步骤** → `arch` 架构层 或 `journey` 组件
- **A vs B 对比** → `grid cols-2/3` + `card`
- **术语表** → `table`

---

## Step 2 — 规划 section

先告诉用户规划，再生成 HTML。

| # | Section | 说明 |
|---|---------|------|
| cover | 封面 | 大标题 + lead + metric 卡片 |
| overview | 目录 | journey-map 全局地图 |
| bg | 背景 | 为什么重要，行业现状 |
| 03…N | 核心概念 | 每个主题一个 floor |
| framework | 框架总结 | 核心模型/体系 |
| findings | 结论 | rec-grid + callout |
| terms | 术语表 | table，必须有 |

---

## Step 3 — 生成 HTML

### 外壳

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TITLE</title>
  <link rel="stylesheet" href="./shared.css">
</head>
<body>
  <nav class="topnav">
    <div class="brand">TOPIC</div>
    <div class="navlinks">
      <a href="#cover">封面</a>
      <!-- 每个 section 一个 a -->
    </div>
  </nav>

  <!-- sections -->
</body>
</html>
```

### Section 模板

```html
<section class="floor" id="SLUG">
  <div class="kicker">CATEGORY · SUBTITLE</div>
  <h2>Section 标题</h2>
  <p class="lead">一到两句介绍。</p>

  <!-- 组件 -->
</section>
```

---

## 组件速查

### 封面（必须用）

```html
<section class="floor" id="cover">
  <div class="kicker">Whitepaper · YEAR</div>
  <h1>主标题</h1>
  <p class="lead">一段 hook。</p>
  <div class="grid cols-4">
    <div class="card metric"><strong>42</strong><span>说明文字</span></div>
    <!-- 3-4 个数字卡 -->
  </div>
</section>
```

### 目录（可选，长报告用）

```html
<div class="journey-map">
  <div class="jm-head"></div>
  <div class="jm-head">01 背景</div>
  <!-- 每列一个 section -->
  <div class="jm-label">回答问题</div>
  <div class="jm-cell">...</div>
  <div class="jm-label">本页内容</div>
  <div class="jm-cell">...</div>
  <div class="jm-label">输出结果</div>
  <div class="jm-cell">...</div>
</div>
```

### 卡片网格（最常用）

```html
<div class="grid cols-3">
  <div class="card">
    <h3>标题</h3>
    <p>说明</p>
    <div class="pill-row">
      <span class="pill">标签</span>
      <span class="pill good">优势</span>
      <span class="pill hot">风险</span>
    </div>
  </div>
</div>
```

### Callout（关键公式 / 洞察）

```html
<div class="callout">
  <b>核心要点：</b>此处写公式、数字或必须记住的结论。
</div>
```

### 架构层图（展示系统结构）

```html
<div class="arch">
  <div class="arch-layer">
    <div class="arch-label">应用层</div>
    <div class="arch-boxes">
      <div class="arch-box">模块 A</div>
      <div class="arch-box">模块 B</div>
    </div>
  </div>
  <!-- 重复 arch-layer -->
  <div class="arch-note">一句说明这个架构的核心判断。</div>
</div>
```

### 终端 / 代码块（展示命令行、伪代码、ASCII 图）

```html
<div class="mt">
  <div class="mtbar">
    <div class="mdt"></div>
    <span class="mtt">标题或文件名</span>
  </div>
  <div class="mb">
// 代码或 ASCII 图，用 pre 风格排版
P = α · C · V² · f
  </div>
</div>
```

颜色辅助类（在 `.mb` 内使用 span）：
- `<span class="green">` 正常 / 成功
- `<span class="err">` 错误 / 警告
- `<span class="purple">` 关键词
- `<span class="yellow">` 注意

### 截图面板（展示界面 / 示意图）

```html
<div class="screenshot">
  <div class="shotbar"><span class="dots"></span>面板标题</div>
  <div class="shotbody">
    <!-- 内容：可以是 arch、mt、metric-bar 等 -->
  </div>
</div>
```

有真实图片时：
```html
<div class="shotbody flush">
  <img class="real-shot" src="./assets/xxx.png" alt="说明">
</div>
<div class="evidence-caption">图注说明。</div>
```

### 复杂架构插图（白皮书重点图）

当某个 section 需要解释复杂层级、系统拓扑、执行链路或协议递归时，不要只堆卡片或表格。优先做一张具有"架构图效果"的主插图，目标是让读者先看懂大结构，再读细节。

设计原则：
- **先画大结构关系**：用漏斗、分层、泳道、树、坐标轴、总线、嵌套框等稳定布局表达主关系。图的第一眼必须能看出"谁包含谁 / 谁流向谁 / 谁逐级缩放到谁"。
- **必须有主线**：用高亮色、粗线、箭头或居中路径标出唯一阅读路径。旁支、sibling、备选实例只能做低对比辅助，不能抢主线。
- **关键边界贴在关系上**：网络边界、协议边界、编译边界、调度边界等信息要放在箭头、分隔线、层间 gap 的正中位置，而不是丢到图外右侧变成注释。
- **icon 放在结构列里**：每层或每类节点配一个小 icon，放在稳定的左侧或顶部标签列。icon 负责建立物理直觉，主画布负责表达关系。
- **避免重复 tag 噪音**：如果每层都有同一套角色或协议，不要在每个节点里重复贴 tag。改用图注、legend 或单独放大图解释。
- **细节要可视化，不要全写成字**：硬件单元、仓库、模块、网络、核心类型等用小图形表达；文字只保留层级名、节点名、边界名和少量必要数字。
- **主图 + 详表分工**：主图负责结构和心智模型；表格负责"是什么 / 规模 / 边界 / 调度看到什么"。不要让一张 SVG 同时承载所有解释。

可复用模式：
```html
<div class="screenshot">
  <div class="shotbar"><span class="dots"></span>主题 · 主架构图</div>
  <div class="shotbody flush" style="padding:18px 12px;background:#fff">
    <svg viewBox="0 0 1200 720" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block">
      <!-- 左侧：层级 + icon -->
      <!-- 中间：高亮主线 / 漏斗 / 分层结构 -->
      <!-- 箭头中间：边界标签 -->
      <!-- 右侧：低对比 sibling / 补充样例 -->
    </svg>
  </div>
  <div class="evidence-caption">一句话解释颜色、主线和图的阅读方式。</div>
</div>
```

好图判断标准：
- 第一眼能看到大结构，而不是先看到一堆文字。
- 读者能沿着一条主线从起点走到终点。
- 每个 icon 都在帮助理解对象的物理或功能形态。
- 细节存在，但不会破坏主线。
- 图下方详表能补全细节，主图不需要解释所有字段。

### 对比表（多产品 / 方案横向比较）

```html
<table>
  <thead>
    <tr>
      <th>维度</th>
      <th><div class="ph-name">方案 A</div><div class="ph-sub">定位描述</div></th>
      <th><div class="ph-name">方案 B</div><div class="ph-sub">定位描述</div></th>
    </tr>
  </thead>
  <tbody>
    <tr><td>核心优势</td><td>...</td><td>...</td></tr>
  </tbody>
</table>
```

### 核心张力（结论三列）

```html
<div class="tension">
  <div><b>核心张力</b><span>描述设计矛盾</span></div>
  <div><b>优化方向</b><span>具体建议</span></div>
  <div><b>局限性</b><span>无法改变的底层约束</span></div>
</div>
```

### 建议面板（结论 / 行动项）

```html
<div class="rec-grid">
  <div class="rec-panel">
    <h3>短期建议</h3>
    <div class="rec-item">
      <div class="rec-num">01</div>
      <div><b>建议标题</b><span>说明</span></div>
    </div>
  </div>
  <div class="rec-panel">
    <h3>长期方向</h3>
    <!-- rec-item -->
  </div>
</div>
```

### 评分条（定量对比）

```html
<div class="metric-bar">
  <span>维度名称</span>
  <div class="bar-track"><div class="bar-fill" style="width:72%"></div></div>
  <b>72%</b>
</div>
```

### 评级点（s1-s5）

```html
<div class="score s4"><span></span><span></span><span></span><span></span><span></span></div>
```

---

## 写作原则

- 每个 section 至少一个视觉组件（callout / arch / mt / screenshot / table 之一）
- `callout` 放公式、数字、必须记住的结论
- `arch` 放系统结构、层次关系
- `mt` 放代码、ASCII 图、伪代码
- `screenshot` 包装复杂内容面板
- 术语表（`table`）永远是最后一节
- 不允许发明数据，只用资料里有的数字
- 全部中文，专有名词保留英文

---

## Step 4 — 输出

1. 写 HTML 到 output-dir
2. 如果目录里没有 `shared.css`，执行：`cp /Users/yin/pto/计算领域竞分/shared.css <output-dir>/shared.css`
3. 启动预览（如果还没跑）：`python3 -m http.server 8765 --directory <output-dir> &`
4. 告诉用户预览地址
