(() => {
  'use strict';

  const N = {
    l2: '[data-mem950-node="rail:L2"]',
    gm: '[data-mem950-node="rail:GM"]',
    ub: '[data-aiv-node="buffer:UB"]',
    scalar: '[data-aiv-node="scalar:Scalar"]',
    vector: '[data-aiv-node="vector:Vector"]',
    simt: '[data-aiv-node="exec:SIMT"]',
    simd: '[data-aiv-node="exec:SIMD"]',
    cache: '[data-aiv-node="buffer:ND-DMA Cache"]',
    l1: '[data-aic-node="buffer:L1"]',
    l0a: '[data-aic-node="buffer:L0A"]',
    l0b: '[data-aic-node="buffer:L0B"]',
    l0c: '[data-aic-node="buffer:L0C"]',
    cube: '[data-aic-node="cube:CUBE"]',
    fp: '[data-aic-node="buffer:FP"]',
  };

  const CATEGORIES = [
    { id: 1, badge: '开发者侧', title: '开发者声明跑哪一代', sub: '声明目标架构', scenario: 'A', arch: 'ascend910b', essence: '架构号是编译与注册声明，只说明“想跑哪一代”，不能单独证明实现可运行。', signals: ['CMAKE_ASC_ARCHITECTURES', '--npu-arch=dav-3510', '__NPU_ARCH__', 'AddConfig("ascend950")'], actions: ['核对 Device 与 Host 两侧的代际分支', '双架构分别编译、运行并对比精度与性能'], selectors: [], routes: [], related: ['vector', 'cube'] },
    { id: 2, badge: 'AIV 核', title: 'Membase → RegBase', sub: '编程对象变了', scenario: 'B', arch: 'ascend950b', essence: '950 AIV 把寄存器张量、谓词与地址寄存器变成一等编程对象，GM → UB → Reg 路径被显式化。', signals: ['RegTensor', 'MaskReg', 'AddrReg', 'LoadAlign / StoreAlign', 'LocalMemBar'], actions: ['把 UB 布局与寄存器压力一起评估', '为 910B 保留独立 Membase 实现'], selectors: [N.ub, N.vector, N.scalar], routes: ['l2-to-aiv1'], related: ['vector'] },
    { id: 3, badge: 'Vector 核', title: '纯 SIMD → SIMD + SIMT', sub: '执行模型变了', scenario: 'B', arch: 'ascend950b', essence: '950 在 Vector 内新增 SIMT 子系统，不替换 SIMD；离散访存、线程分歧和线程级原子有了原生表达。', signals: ['--enable-simt', 'Warp / ThreadBlock', 'asc_atomic_add', 'Gather / Scatter'], actions: ['仅在离散访存或线程语义明确时启用 SIMT', '分别验证 SIMD 与 SIMT 路径的性能边界'], selectors: [N.simt, N.simd, N.ub], routes: [], related: ['vector'] },
    { id: 4, badge: 'Cube 通路', title: '芯片内互连变了', sub: '物理数据通路重接', scenario: 'C', arch: 'ascend950b', essence: 'Cube 周围的物理连线发生变化：旧直通路径移除，同时增加 UB↔L1、L0C→UB、NDDMA 等新路径。', signals: ['删除 GM → L0A/L0B', '删除 L1 → GM', '新增 UB → L1', '新增 L0C → UB', 'NDDMA / SSBuf'], actions: ['按 950 真实通路重写搬运链', '寻找 C-V 直连带来的融合机会'], selectors: [N.gm, N.l1, N.l0a, N.l0b, N.l0c, N.fp, N.ub], routes: ['l2-to-aic', 'aic-to-aiv1'], related: ['cube', 'gemm-ar'] },
    { id: 5, badge: 'Cube ISA', title: '计算指令移除 / 新增', sub: 'Cube 电路重新规划', scenario: 'C', arch: 'ascend950b', essence: '950 移除部分 int4、结构化稀疏与边界绕回能力，并为 MX 低比特路径重新配置 Cube 电路。', signals: ['MmadWithSparse 移除', 'int4 Cube Matmul 移除', 'SetLoadDataBoundary 移除', 'MX 系列新增'], actions: ['替换已移除指令与隐含硬件假设', '低比特方案转向 FP8 / MX 并重做精度验证'], selectors: [N.cube, N.l0a, N.l0b, N.l0c], routes: [], related: ['cube'] },
    { id: 6, badge: '矩阵分形', title: 'Cube 喂数排布变了', sub: 'L0A 从 ZZ 改成 NZ', scenario: 'C', arch: 'ascend950b', essence: 'L0A 的矩阵 A 分形从 ZZ 改为 NZ；写死切分与地址计算的实现必须迁移。', signals: ['A / L0A: ZZ → NZ', 'B / L0B: ZN 不变', 'C / L0C: NZ 不变'], actions: ['定位写死 L0A 分形的地址公式', '按 950 分形重新生成 tiling 与搬运参数'], selectors: [N.l0a, N.l1, N.cube], routes: [], related: ['cube'] },
    { id: 7, badge: '低比特', title: 'HiF8 / FP8 / MX 系列新增', sub: '低比特成为核心路径', scenario: 'B', arch: 'ascend950b', essence: '新格式不仅改变 dtype，还联动 scale 布局、搬运、舍入、饱和与量化融合。', signals: ['HiF8', 'FP8 E5M2 / E4M3', 'MXFP4 / MXFP8', 'MicroScaling', 'Histograms'], actions: ['把 scale 张量纳入 tiling 主路径', '补齐端到端精度和饱和行为验证'], selectors: [N.cube, N.fp, N.l1], routes: ['l2-to-aic'], related: ['cube'] },
    { id: 8, badge: 'UB / SRAM', title: 'bank 拓扑变了', sub: 'UB SRAM 微架构', scenario: 'C', arch: 'ascend950b', essence: 'UB bank group、每组 bank 数与单 bank 容量变化，旧的错位地址经验不再可靠。', signals: ['bank group 16 → 8', '每组 3 → 2 banks', '单 bank 4KB → 16KB', 'UB 192KB → 256KB'], actions: ['删除写死容量与 bank 错位公式', '用 profiling 验证冲突与带宽'], selectors: [N.ub], routes: [], related: ['vector', 'tput-sync'] },
    { id: 9, badge: '语义', title: '浮点 / 同步 / 调试语义', sub: '数值与控制原语', scenario: 'A', arch: 'ascend950b', essence: '即使结构不变，Subnormal、核间同步和调试接口变化也可能造成结果或诊断方式不同。', signals: ['Subnormal 默认不支持', '核间 Mutex 新增', 'CheckLocalMemoryIA 移除'], actions: ['建立跨代数值回归基线', '将同步与调试接口纳入迁移检查表'], selectors: [N.vector, N.scalar], routes: [], related: ['vector', 'gemm-ar'] },
    { id: 10, badge: '通信', title: 'HCCL 软件通信 → CCU 硬化通信', sub: '集合通信专用引擎', scenario: 'B', arch: 'ascend950b', essence: 'HCCL 仍提供上层语义，但 950 把部分集合通信执行下沉至 CCU，需要联合观察硬件资源与通算重叠。', signals: ['CCU', 'ReduceScatter', 'AllGatherMatMul', 'Dispatch / Combine', 'CCU profiling'], actions: ['区分 HCCL 接口语义与 CCU 执行落点', '用 CCU profiling 验证通信、片上内存和 AI Core 协同'], selectors: [N.l2, N.scalar, N.cube, N.l0c], routes: [], related: ['ccu-collective', 'gemm-ar'] },
  ];

  const FLOWS = [
    { id: 'vector', title: 'AIV Vector 计算', short: 'GM/L2 → UB → Vector → UB → GM', confidence: 'verified', arch: 'ascend910b', summary: '单核 AIV 的标准 Vector 路径。数据经 MTE2 进入 UB，计算结果再由 MTE3 写回。', path: 'GM / L2 → MTE2 → UB → Vector → UB → MTE3 → GM / L2', selectors: [N.l2, N.ub, N.vector], routes: ['l2-to-aiv1', 'aiv1-to-l2'], meta: { 'PTO 语义': '普通 AIV tile 计算', 'payload 经过 UB': '是', '适用': 'elementwise / reduce / gather / cast' }, steps: [
      { label: '读 GM/L2', text: '源 tile 位于 GM，全局访问经 L2 层。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'MTE2 入 UB', text: '把源 tile 搬入 AIV Unified Buffer。', selectors: [N.l2, N.ub], routes: ['l2-to-aiv1'] },
      { label: 'Vector 消费', text: 'Vector/SIMD 从 UB 读取并把结果留在 UB。', selectors: [N.ub, N.vector], routes: [] },
      { label: 'MTE3 写回', text: '结果从 UB 写回 GM/L2。', selectors: [N.ub, N.l2], routes: ['aiv1-to-l2'] },
    ] },
    { id: 'cube', title: 'AIC Cube / GEMM 计算', short: 'GM/L2 → L1 → L0A/B → Cube → L0C', confidence: 'verified', arch: 'ascend910b', summary: 'AIC 的矩阵主路径。数据进入 L1 与 L0，Cube 计算后由 L0C/FixPipe 承接结果。', path: 'GM / L2 → L1 → L0A / L0B → Cube → L0C → FixPipe', selectors: [N.l2, N.l1, N.l0a, N.l0b, N.cube, N.l0c, N.fp], routes: ['l2-to-aic'], meta: { 'PTO 语义': 'GEMM / Cube tile', 'payload 经过 UB': '否', '适用': 'GEMM / convolution / matmul' }, steps: [
      { label: 'MTE2 入 L1', text: 'GM/L2 数据先进入 AIC L1 Buffer。', selectors: [N.l2, N.l1], routes: ['l2-to-aic'] },
      { label: 'MTE1 入 L0A/B', text: 'A、B tile 分别进入 L0A 与 L0B。', selectors: [N.l1, N.l0a, N.l0b], routes: [] },
      { label: 'Cube 计算', text: 'Cube 消费 A/B tile，累加结果进入 L0C。', selectors: [N.l0a, N.l0b, N.cube, N.l0c], routes: [] },
      { label: 'FixPipe 承接', text: 'L0C 结果经 FixPipe 写回或随路处理。', selectors: [N.l0c, N.fp], routes: [] },
    ] },
    { id: 'tput-sync', title: 'TPUT / TGET 同步搬运', short: 'GM → UB staging → remote GM', confidence: 'verified', arch: 'ascend910b', summary: '同步点对点搬运由 AIV 控制，payload 经过本地 UB staging，API 返回前 AIV 等待完成。', path: 'AIV issue → GM/L2 → UB staging → remote GM → wait', selectors: [N.l2, N.ub, N.scalar], routes: ['l2-to-aiv1', 'aiv1-to-l2'], meta: { 'PTO API': 'TPUT / TGET', 'payload 经过 UB': '是', '适用': '小块、strided、低启动时延' }, steps: [
      { label: 'AIV 发起', text: 'AIV 决定 tile、双缓冲与 atomic 细节。', selectors: [N.scalar], routes: [] },
      { label: 'MTE2 staging', text: '本地或远端 tile 进入本地 UB。', selectors: [N.l2, N.ub], routes: ['l2-to-aiv1'] },
      { label: 'MTE3 远端写', text: 'UB staging 数据写向目标 GM。', selectors: [N.ub, N.l2], routes: ['aiv1-to-l2'] },
      { label: 'AIV 等待', text: '同步语义阻塞到搬运完成。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'sdma-async', title: 'TPUT_ASYNC / TGET_ASYNC → SDMA', short: 'descriptor 在 UB；payload GM ↔ GM', confidence: 'verified', arch: 'ascend910b', summary: 'AIV 在 UB scratch 写 descriptor，但 payload 不经过 UB，由 AI Core 外部 SDMA 异步搬运。', path: 'AIV descriptor → SDMA → remote GM → AsyncEvent wait/test', selectors: [N.ub, N.scalar, N.gm], routes: [], meta: { 'PTO API': 'TPUT_ASYNC / TGET_ASYNC', 'payload 经过 UB': '否', '适用': '大块连续数据、通信计算重叠' }, steps: [
      { label: '写 descriptor', text: 'UB scratch 保存描述符、队尾与完成标记。', selectors: [N.ub, N.scalar], routes: [] },
      { label: '提交 SDMA', text: 'AIV 提交后继续执行。', selectors: [N.scalar], routes: [] },
      { label: 'GM↔GM DMA', text: '外部 SDMA 搬运 payload。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'wait/test', text: '稍后通过 AsyncEvent 收敛完成状态。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'urma-async', title: '950 URMA 异步远程内存', short: 'GM → UnifiedBus / URMA → remote GM', confidence: 'claim', arch: 'ascend950b', summary: '950 可把异步接口映射到远程内存路径；公开资料能支持方向判断，但具体微结构仍需硬件手册确认。', path: 'AIV session → local GM → UnifiedBus / URMA → remote GM → wait/test', selectors: [N.scalar, N.l2, N.gm], routes: [], meta: { 'PTO API': 'TPUT_ASYNC / TGET_ASYNC', 'payload 经过 UB': '否', '证据边界': '互连方向可确认；端口细节待确认' }, steps: [
      { label: 'AIV 建会话', text: '构建 session 并提交远程访问。', selectors: [N.scalar], routes: [] },
      { label: '进入互连面', text: 'payload 从 GM/L2 进入 950 IO 互连。', selectors: [N.gm, N.l2], routes: [] },
      { label: 'URMA 访问', text: '外部子系统承担远程内存语义。', selectors: [N.l2], routes: [] },
      { label: '完成语义', text: 'wait/test 或 Quiet 收敛可见性。', selectors: [N.scalar], routes: [] },
    ] },
    { id: 'ccu-collective', title: '950 CCU 集合通信卸载', short: 'AIV launch；CCU 完成搬运归约', confidence: 'claim', arch: 'ascend950b', summary: 'AIV 负责握手与控制，CCU 路径承担集合通信的数据搬运与规约。外部 CCU 以证据说明呈现，不虚构进 AI Core 图。', path: 'AIC tile → AIV launch → CCU fetch/reduce → result GM → CCU_DONE', selectors: [N.cube, N.l0c, N.fp, N.scalar, N.l2], routes: [], meta: { 'PTO 语义': 'TGATHER / TSCATTER / TBROADCAST / TREDUCE', 'payload 经过 UB': '否', '适用': 'AllReduce / ReduceScatter / Broadcast' }, steps: [
      { label: 'AIC 产 tile', text: 'Cube 结果变成 CCU 可访问的数据。', selectors: [N.cube, N.l0c, N.fp], routes: [] },
      { label: 'AIV launch', text: 'AIV 通知外部 CCU 启动任务。', selectors: [N.scalar], routes: [] },
      { label: 'CCU fetch + reduce', text: 'CCU 子系统完成缓冲、搬运与规约。', selectors: [N.l2], routes: [] },
      { label: '写回 + DONE', text: '结果回到 GM，完成信号返回 AIV。', selectors: [N.gm, N.scalar], routes: [] },
    ] },
    { id: 'gemm-ar', title: 'GEMM + AllReduce 流水线', short: 'AIC 产 tile；AIV 通信；device barrier', confidence: 'inferred', arch: 'ascend950b', summary: 'AIC 不等整次 GEMM 结束；AIV 按 ready tile 发起通信，以 tile 粒度形成计算与通信重叠。', path: 'AIC tile → Ready Queue → AIV TTEST → TPUT AtomicAdd → barrier', selectors: [N.cube, N.l0c, N.scalar, N.ub, N.l2], routes: ['l2-to-aiv2', 'aiv2-to-l2'], meta: { 'PTO API': 'TTEST / TPUT AtomicAdd / TNOTIFY / TWAIT', '并行关系': 'AIC 生产，AIV 消费', '证据等级': '由示例流程推导' }, steps: [
      { label: 'AIC 产 tile', text: 'Cube 完成一个可消费的 GEMM tile。', selectors: [N.cube, N.l0c], routes: [] },
      { label: 'Ready Queue', text: 'AIV 用 TTEST 非阻塞检查 tile。', selectors: [N.scalar], routes: [] },
      { label: 'TPUT AtomicAdd', text: 'AIV 把 tile 送向 owner rank。', selectors: [N.ub, N.l2], routes: ['l2-to-aiv2', 'aiv2-to-l2'] },
      { label: 'Device barrier', text: 'TNOTIFY/TWAIT 收敛阶段依赖。', selectors: [N.scalar], routes: [] },
    ] },
  ];

  const SCENARIOS = {
    A: { title: 'A · 跨架构可兼容', body: '实现不依赖特定硬件路径，但仍需双架构编译、精度与性能验证。', actions: ['分别编译 dav-2201 与 dav-3510', '重点检查 Subnormal 与 UB bank 行为'] },
    B: { title: 'B · 950 原生能力', body: '代码直接使用 950 新对象或新执行资源；若兼容 910B，需要独立实现分支。', actions: ['按代际拆分注册、tiling 与 workspace', '验证新能力对应的硬件和 profiling 证据'] },
    C: { title: 'C · 写死旧硬件假设', body: '旧代码依赖 910B/A2/A3 的物理通路、分形或 SRAM 经验，不能只改编译参数。', actions: ['先逐项定位旧代硬件假设', '重写搬运、分形、tiling 与 bank 策略'] },
  };

  const state = { mode: 'migration', arch: 'ascend910b', selectedId: null, activeStep: -1, playing: false, timer: null, overlay: null, viewport: null, playback: null, playbackHover: null };
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

  function renderLists() {
    $('#categoryList').innerHTML = CATEGORIES.map((item, index) => `
      <li><button class="entity-button" type="button" data-category-id="${item.id}">
        <span class="entity-index">${String(index + 1).padStart(2, '0')}</span>
        <span class="entity-main"><span class="entity-title">${escape(item.title)}</span><span class="entity-sub">${escape(item.sub)}</span></span>
        <span class="mini-badge">${escape(item.badge)}</span>
      </button></li>`).join('');
    $('#flowList').innerHTML = FLOWS.map((item, index) => `
      <li><button class="entity-button" type="button" data-flow-id="${item.id}">
        <span class="entity-index">${String(index + 1).padStart(2, '0')}</span>
        <span class="entity-main"><span class="entity-title">${escape(item.title)}</span><span class="entity-sub">${escape(item.short)}</span></span>
        <span class="evidence-badge">${escape(item.confidence)}</span>
      </button></li>`).join('');
  }

  function renderHardware() {
    const host = $('#hardwareGraph');
    window.PtoMemoryArchitecturePattern.renderArchitecture(host, state.arch);
    state.overlay?.destroy?.();
    state.overlay = window.PtoMemoryArchitecturePattern.createRouteOverlay(host, state.arch);
    window.PtoMemoryArchitecturePattern.attachHoverInteractions(host, state.arch);
    requestAnimationFrame(() => {
      state.overlay?.update?.();
      state.viewport?.fit?.();
      applyCurrentFocus();
    });
  }

  function setArch(arch) {
    if (!arch || state.arch === arch) return;
    state.arch = arch;
    $$('[data-arch-id]').forEach((button) => {
      const selected = button.dataset.archId === arch;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
    renderHardware();
  }

  function focusHardware(selectors = [], routes = []) {
    const host = $('#hardwareGraph');
    window.PtoMemoryArchitecturePattern.clearPathFocus(host);
    if (selectors.length || routes.length) {
      window.PtoMemoryArchitecturePattern.setPathFocus(host, state.arch, { selectors, routes });
    }
    requestAnimationFrame(() => state.overlay?.update?.());
  }

  function applyCurrentFocus() {
    if (state.selectedId === null) return focusHardware();
    const item = state.mode === 'migration'
      ? CATEGORIES.find((entry) => entry.id === Number(state.selectedId))
      : FLOWS.find((entry) => entry.id === state.selectedId);
    if (!item) return focusHardware();
    if (state.mode === 'flow' && state.activeStep >= 0) {
      const step = item.steps[state.activeStep];
      return focusHardware(step.selectors || [], step.routes || []);
    }
    focusHardware(item.selectors || [], item.routes || []);
  }

  function section(title, content) {
    return `<section class="inspector-section"><h3>${escape(title)}</h3>${content}</section>`;
  }

  function renderCategoryInspector(item) {
    const related = item.related.map((id) => {
      const flow = FLOWS.find((entry) => entry.id === id);
      return flow ? `<button class="btn btn-ghost related-flow" type="button" data-related-flow="${flow.id}">→ ${escape(flow.title)}</button>` : '';
    }).join('');
    $('#inspectorMeta').textContent = `Category ${String(item.id).padStart(2, '0')}`;
    $('#inspectorBody').innerHTML = `<div class="inspector-content">
      ${section(item.badge, `<span class="inspector-kicker">${escape(item.scenario)} 类迁移判断</span><h2>${escape(item.title)}</h2><p>${escape(item.essence)}</p>`)}
      ${section('判断信号', `<div class="tag-row">${item.signals.map((signal) => `<span class="path-chip">${escape(signal)}</span>`).join('')}</div>`)}
      ${section('建议动作', `<ul class="action-list">${item.actions.map((action) => `<li>${escape(action)}</li>`).join('')}</ul>`)}
      ${section('关联执行流', related)}
      ${section('数据说明', '<div class="inspector-card"><strong>L2 · 解释性数据</strong><small>由两个原 demo 的事实与交互结构整合，用于产品原型；不声明为真实运行结果。</small></div>')}
    </div>`;
  }

  function renderFlowInspector(flow) {
    $('#inspectorMeta').textContent = `${flow.confidence} · ${flow.steps.length} steps`;
    $('#inspectorBody').innerHTML = `<div class="inspector-content">
      ${section('Execution flow', `<span class="inspector-kicker">${escape(flow.confidence)}</span><h2>${escape(flow.title)}</h2><p>${escape(flow.summary)}</p><div class="inspector-card"><strong>路径</strong><small>${escape(flow.path)}</small></div>`)}
      ${section('上下文', `<dl class="meta-grid">${Object.entries(flow.meta).map(([key, value]) => `<div class="meta-row"><dt>${escape(key)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>`)}
      ${section('执行步骤', `<ol class="step-list">${flow.steps.map((step, index) => `<li><button class="step-button${state.activeStep === index ? ' is-selected' : ''}" type="button" data-step="${index}"><span class="step-number">${String(index + 1).padStart(2, '0')}</span><span class="step-copy"><strong>${escape(step.label)}</strong><small>${escape(step.text)}</small></span></button></li>`).join('')}</ol>`)}
      ${section('证据边界', `<div class="inspector-card"><strong>${escape(flow.confidence)}</strong><small>${flow.confidence === 'verified' ? '硬件路径与行为由原 demo 的已校验说明继承。' : flow.confidence === 'claim' ? '支持方向判断，但外部引擎端口与微结构不在本图中虚构。' : '由公开示例的协作时序推导，需以运行证据复核。'}</small></div>`)}
    </div>`;
    syncPlayback();
  }

  function selectCategory(id) {
    stopPlayback();
    state.selectedId = Number(id);
    state.activeStep = -1;
    const item = CATEGORIES.find((entry) => entry.id === state.selectedId);
    if (!item) return;
    $$('[data-category-id]').forEach((button) => button.classList.toggle('is-selected', Number(button.dataset.categoryId) === item.id));
    $$('[data-scenario]').forEach((button) => button.classList.toggle('is-selected', button.dataset.scenario === item.scenario));
    $('#selectionCaption').textContent = item.title;
    $('#emptySelection').hidden = true;
    renderCategoryInspector(item);
    if (state.arch !== item.arch) setArch(item.arch); else applyCurrentFocus();
  }

  function selectFlow(id) {
    stopPlayback();
    state.selectedId = id;
    state.activeStep = -1;
    const item = FLOWS.find((entry) => entry.id === id);
    if (!item) return;
    $$('[data-flow-id]').forEach((button) => button.classList.toggle('is-selected', button.dataset.flowId === item.id));
    $('#selectionCaption').textContent = item.title;
    $('#emptySelection').hidden = true;
    renderFlowInspector(item);
    if (state.arch !== item.arch) setArch(item.arch); else applyCurrentFocus();
  }

  function selectStep(index) {
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    if (!flow) return;
    state.activeStep = Math.max(0, Math.min(flow.steps.length - 1, Number(index)));
    renderFlowInspector(flow);
    applyCurrentFocus();
  }

  function setMode(mode) {
    if (!['migration', 'flow'].includes(mode)) return;
    stopPlayback();
    state.mode = mode;
    state.selectedId = null;
    state.activeStep = -1;
    $$('[data-mode]').forEach((button) => {
      const selected = button.dataset.mode === mode;
      button.classList.toggle('is-selected', selected);
      button.setAttribute('aria-selected', String(selected));
    });
    $('#migrationExplorer').hidden = mode !== 'migration';
    $('#flowExplorer').hidden = mode !== 'flow';
    $('#playbackMount').hidden = mode !== 'flow';
    $('#explorerTitle').textContent = mode === 'migration' ? '迁移分类' : '执行流';
    $('#explorerCount').textContent = mode === 'migration' ? `${CATEGORIES.length} 项` : `${FLOWS.length} 条`;
    $('#selectionCaption').textContent = '尚未选择分析项';
    $('#emptySelection').hidden = false;
    $('#inspectorMeta').textContent = '等待选择';
    $('#inspectorBody').innerHTML = '<div class="inspector-empty"><span>CONTEXTUAL INSPECTOR</span><strong>选择迁移项或执行流</strong><p>这里会显示判断信号、硬件影响、迁移动作、执行步骤和证据等级。</p></div>';
    $$('.entity-button, .scenario-chip').forEach((button) => button.classList.remove('is-selected'));
    focusHardware();
    syncPlayback();
  }

  function showScenario(key) {
    const scenario = SCENARIOS[key];
    if (!scenario) return;
    state.selectedId = null;
    $('#emptySelection').hidden = false;
    $('#selectionCaption').textContent = scenario.title;
    $$('[data-scenario]').forEach((button) => button.classList.toggle('is-selected', button.dataset.scenario === key));
    $$('[data-category-id]').forEach((button) => button.classList.remove('is-selected'));
    $('#inspectorMeta').textContent = `Scenario ${key}`;
    $('#inspectorBody').innerHTML = `<div class="inspector-content">${section('快速判断', `<span class="inspector-kicker">Migration scenario</span><h2>${escape(scenario.title)}</h2><p>${escape(scenario.body)}</p>`)}${section('下一步', `<ul class="action-list">${scenario.actions.map((item) => `<li>${escape(item)}</li>`).join('')}</ul>`)}</div>`;
    focusHardware();
  }

  function syncPlayback() {
    if (!state.playback) return;
    const root = $('#playbackMount');
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    const total = flow?.steps.length || 1;
    const current = state.activeStep >= 0 ? state.activeStep : 0;
    const scrubber = $('.pto-floating-playback__scrubber', root);
    const label = $('.pto-floating-playback__counter', root);
    const opname = $('.pto-floating-playback__opname', root);
    const play = $('.pto-floating-playback__button--primary', root);
    if (scrubber) { scrubber.max = Math.max(0, total - 1); scrubber.value = current; scrubber.disabled = !flow; }
    if (label) label.textContent = flow ? `${current + 1} / ${total}` : '0 / 0';
    if (opname) opname.textContent = flow ? flow.steps[current].label : '选择执行流';
    if (play) play.innerHTML = window.PtoFloatingPlaybackControl.iconLabel(state.playing ? 'pause' : 'play', state.playing ? 'Pause' : 'Play');
    state.playback.sync({ playing: state.playing });
  }

  function stopPlayback() {
    window.clearInterval(state.timer);
    state.timer = null;
    state.playing = false;
    syncPlayback();
  }

  function togglePlayback() {
    const flow = FLOWS.find((entry) => entry.id === state.selectedId);
    if (!flow) return;
    if (state.playing) return stopPlayback();
    if (state.activeStep < 0 || state.activeStep >= flow.steps.length - 1) selectStep(0);
    state.playing = true;
    syncPlayback();
    state.timer = window.setInterval(() => {
      if (state.activeStep >= flow.steps.length - 1) return stopPlayback();
      selectStep(state.activeStep + 1);
      state.playing = true;
      syncPlayback();
    }, 1500);
  }

  function initPlayback() {
    const root = $('#playbackMount');
    const control = window.PtoFloatingPlaybackControl.createControl({ className: 'pto-floating-playback--preview pto-ide-frame__floating-playback', showTimeline: true });
    root.appendChild(control);
    state.playback = window.PtoFloatingPlaybackControl.init({ root, isPlaying: () => state.playing });
    state.playbackHover = window.PtoFloatingPlaybackControl.initScrubberHover({
      root,
      getTotalSteps: () => FLOWS.find((entry) => entry.id === state.selectedId)?.steps.length || 1,
      getLabelForStep: (index) => FLOWS.find((entry) => entry.id === state.selectedId)?.steps[index]?.label || '选择执行流',
    });
    $('.pto-floating-playback__button--primary', root)?.addEventListener('click', togglePlayback);
    $('#step-back-btn', root)?.addEventListener('click', () => selectStep(state.activeStep - 1));
    $('#step-fwd-btn', root)?.addEventListener('click', () => selectStep(state.activeStep + 1));
    $('#replay-btn', root)?.addEventListener('click', () => { stopPlayback(); selectStep(0); });
    $('.pto-floating-playback__scrubber', root)?.addEventListener('input', (event) => { stopPlayback(); selectStep(event.target.value); });
    syncPlayback();
  }

  function initHardware() {
    window.PtoMemoryArchitecturePattern.renderArchitecture($('#hardwareGraph'), state.arch);
    state.overlay = window.PtoMemoryArchitecturePattern.createRouteOverlay($('#hardwareGraph'), state.arch);
    window.PtoMemoryArchitecturePattern.attachHoverInteractions($('#hardwareGraph'), state.arch);
    state.viewport = window.PtoHardwareArchitectureViewport.mount($('#hardwareViewport'), {
      mode: 'inline',
      viewport: '[data-stage]',
      scaleEl: '[data-scale]',
      inlineHost: '#hardwareGraph',
      detailToggle: '[data-detail]',
      zoomOut: '[data-zoom-out]',
      zoomIn: '[data-zoom-in]',
      fit: '[data-fit]',
      readout: '[data-readout]',
      defaultScale: 0.6,
      fitOnMount: true,
      pan: true,
      wheelZoom: true,
      onScaleChange: () => requestAnimationFrame(() => state.overlay?.update?.()),
      onPanChange: () => requestAnimationFrame(() => state.overlay?.update?.()),
    });
    requestAnimationFrame(() => { state.overlay?.update?.(); state.viewport?.fit?.(); });
  }

  function initEvents() {
    document.addEventListener('click', (event) => {
      const mode = event.target.closest('[data-mode]');
      if (mode) return setMode(mode.dataset.mode);
      const category = event.target.closest('[data-category-id]');
      if (category) return selectCategory(category.dataset.categoryId);
      const flow = event.target.closest('[data-flow-id]');
      if (flow) return selectFlow(flow.dataset.flowId);
      const scenario = event.target.closest('[data-scenario]');
      if (scenario) return showScenario(scenario.dataset.scenario);
      const related = event.target.closest('[data-related-flow]');
      if (related) { setMode('flow'); selectFlow(related.dataset.relatedFlow); return; }
      const step = event.target.closest('[data-step]');
      if (step) { stopPlayback(); selectStep(step.dataset.step); }
    });
    $$('[data-arch-id]').forEach((button) => button.addEventListener('click', () => setArch(button.dataset.archId)));
    $('#inspectorToggle').addEventListener('click', () => {
      const hidden = !$('#inspectorPane').hidden;
      $('#inspectorPane').hidden = hidden;
      $('#inspectorToggle').classList.toggle('is-selected', !hidden);
      $('#inspectorToggle').setAttribute('aria-pressed', String(!hidden));
      requestAnimationFrame(() => state.viewport?.fit?.());
    });
    const frame = $('[data-ide-frame]');
    frame.addEventListener('pointermove', (event) => {
      const rect = frame.getBoundingClientRect();
      frame.style.setProperty('--ide-cursor-x', `${event.clientX - rect.left}px`);
      frame.style.setProperty('--ide-cursor-y', `${event.clientY - rect.top}px`);
      frame.style.setProperty('--ide-cursor-alpha', '0.16');
    });
    frame.addEventListener('pointerleave', () => frame.style.setProperty('--ide-cursor-alpha', '0'));
  }

  renderLists();
  initHardware();
  initPlayback();
  initEvents();
  setMode('migration');
})();
