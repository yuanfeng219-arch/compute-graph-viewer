const esc = value => String(value ?? '').replace(/[&<>"']/g, ch => ({
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
}[ch]));

function readCssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function hexToRgb(hex) {
  const clean = hex.replace('#', '').trim();
  const value = parseInt(clean.length === 3 ? clean.split('').map(ch => ch + ch).join('') : clean, 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function mixColor(a, b, t) {
  const ar = hexToRgb(a), br = hexToRgb(b);
  return `rgb(${Math.round(ar[0] + (br[0] - ar[0]) * t)},${Math.round(ar[1] + (br[1] - ar[1]) * t)},${Math.round(ar[2] + (br[2] - ar[2]) * t)})`;
}

function loadColor(value, maxValue = 1.35) {
  const theme = document.documentElement.dataset.theme || 'dark';
  const low = theme === 'light' ? '#e7edf4' : '#222a35';
  const normal = theme === 'light' ? '#7fcbd3' : '#1fb8cc';
  const high = theme === 'light' ? '#e8ce69' : '#d6aa35';
  const hot = theme === 'light' ? '#d775aa' : '#c95ba8';
  if (value <= 0) return low;
  const x = Math.max(0, Math.min(maxValue, value));
  if (x < 0.72) return mixColor(low, normal, x / 0.72);
  if (x < 1) return mixColor(normal, high, (x - 0.72) / 0.28);
  return mixColor(high, hot, (x - 1) / Math.max(0.01, maxValue - 1));
}

function tokenColor(tokenName, fallback) {
  const raw = readCssVar(tokenName, fallback);
  if (raw.startsWith('#')) return raw;
  return fallback;
}

function makeStatGrid(stats) {
  return `<div class="opv-analysis-stats">${stats.map(stat => `
    <div class="opv-analysis-stat">
      <span>${esc(stat.label)}</span>
      <b>${esc(stat.value)}</b>
    </div>
  `).join('')}</div>`;
}

function makeTooltip(parent) {
  const tip = document.createElement('div');
  tip.className = 'opv-analysis-tooltip';
  tip.hidden = true;
  parent.appendChild(tip);
  return {
    show(event, html) {
      tip.innerHTML = html;
      tip.hidden = false;
      const bounds = parent.getBoundingClientRect();
      const x = Math.min(bounds.width - 220, Math.max(8, event.clientX - bounds.left + 12));
      const y = Math.min(bounds.height - 130, Math.max(8, event.clientY - bounds.top + 12));
      tip.style.transform = `translate(${x}px, ${y}px)`;
    },
    hide() {
      tip.hidden = true;
    },
  };
}

function resizeCanvas(canvas, width, height) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function createAnalysisDock({ tabsRoot, titleEl, metaEl, views, initialView = 'timeline', onViewChange }) {
  const viewMap = new Map(views.map(view => [view.id, view]));
  const tabs = new Map();
  let activeView = viewMap.has(initialView) ? initialView : views[0]?.id;

  tabsRoot.innerHTML = '';
  views.forEach(view => {
    const tab = document.createElement('button');
    tab.type = 'button';
    tab.className = 'opv-analysis-tab';
    tab.dataset.analysisTab = view.id;
    tab.textContent = view.label;
    tab.addEventListener('click', () => setActiveView(view.id));
    tabsRoot.appendChild(tab);
    tabs.set(view.id, tab);
    view.panel.hidden = true;
    view.mount?.();
  });

  function setActiveView(id) {
    const next = viewMap.has(id) ? id : views[0]?.id;
    if (!next) return;
    activeView = next;
    views.forEach(view => {
      const active = view.id === activeView;
      view.panel.hidden = !active;
      tabs.get(view.id)?.classList.toggle('is-active', active);
      tabs.get(view.id)?.setAttribute('aria-pressed', String(active));
    });
    const view = viewMap.get(activeView);
    titleEl.textContent = view.title || view.label;
    metaEl.textContent = typeof view.meta === 'function' ? view.meta() : (view.meta || '');
    localStorage.setItem('op-rank-time-analysis-view', activeView);
    requestAnimationFrame(() => view.render?.());
    onViewChange?.(activeView);
  }

  function refresh() {
    viewMap.get(activeView)?.render?.();
  }

  function resize() {
    viewMap.get(activeView)?.resize?.();
  }

  setActiveView(activeView);
  return { setActiveView, refresh, resize, get activeView() { return activeView; } };
}

export function createMoeLoadView({ panel, viewModel, onSelect }) {
  let metricId = viewModel.metricOptions[0]?.id || 'loadRatio';
  let canvas;
  let detail;
  let tooltip;
  let hoverCell = null;
  let selectedCell = null;

  function metric() {
    return viewModel.metricOptions.find(item => item.id === metricId) || viewModel.metricOptions[0];
  }

  function cellAt(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const left = 56;
    const top = 12;
    const rightPad = 10;
    const bottomPad = 22;
    const gridW = Math.max(1, rect.width - left - rightPad);
    const gridH = Math.max(1, rect.height - top - bottomPad);
    if (x < left || x > left + gridW || y < top || y > top + gridH) return null;
    const row = Math.min(viewModel.metrics.layerCount - 1, Math.max(0, Math.floor((y - top) / gridH * viewModel.metrics.layerCount)));
    const expertId = Math.min(viewModel.metrics.expertsPerLayer - 1, Math.max(0, Math.floor((x - left) / gridW * viewModel.metrics.expertsPerLayer)));
    return { row, layer: viewModel.metrics.firstMoeLayer + row, expertId };
  }

  function cellValue(cell) {
    return metric().array[cell.row * viewModel.metrics.expertsPerLayer + cell.expertId] || 0;
  }

  function renderDetail() {
    if (!detail) return;
    const layer = selectedCell ? viewModel.metrics.layers[selectedCell.row] : viewModel.metrics.layers.reduce((best, item) => item.maxLoadRatio > best.maxLoadRatio ? item : best, viewModel.metrics.layers[0]);
    if (!layer) {
      detail.innerHTML = '';
      return;
    }
    const title = selectedCell ? `L${selectedCell.layer} · expert ${selectedCell.expertId}` : `Worst layer · L${layer.layer}`;
    const value = selectedCell ? cellValue(selectedCell) : layer.maxLoadRatio;
    detail.innerHTML = `
      <div class="opv-analysis-detail-title">${esc(title)}</div>
      <div class="opv-analysis-detail-grid">
        <span>metric</span><b>${esc(metric().label)} ${Number(value).toFixed(metricId === 'loadRatio' ? 2 : 0)}${metricId === 'loadRatio' ? 'x' : ''}</b>
        <span>avg load</span><b>${layer.avgLoadRatio.toFixed(2)}x</b>
        <span>p95 load</span><b>${layer.p95LoadRatio.toFixed(2)}x</b>
        <span>overload</span><b>${layer.overloadedExperts} experts</b>
        <span>idle</span><b>${layer.idleExperts} experts</b>
        <span>A2A skew</span><b>${layer.allToAllSkew.toFixed(2)}x</b>
      </div>`;
  }

  function draw() {
    if (!canvas || panel.hidden) return;
    const bounds = panel.getBoundingClientRect();
    const width = Math.max(460, bounds.width - 24);
    const left = 56;
    const top = 12;
    const rightPad = 10;
    const bottomPad = 22;
    const gridW = width - left - rightPad;
    const layerCount = viewModel.metrics.layerCount;
    const experts = viewModel.metrics.expertsPerLayer;
    // 正方形格子：边长由列宽决定，行高跟随列宽，整体高度不随窗口高度拉伸
    const colW = gridW / experts;
    const rowH = colW;
    const gridH = rowH * layerCount;
    const height = top + gridH + bottomPad;
    const ctx = resizeCanvas(canvas, width, height);
    const option = metric();
    const values = option.array;
    const maxValue = metricId === 'loadRatio' ? 1.35 : Math.max(1, ...values);
    ctx.clearRect(0, 0, width, height);
    ctx.font = '600 10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = readCssVar('--foreground-secondary', 'rgba(255,255,255,0.62)');

    for (let row = 0; row < layerCount; row++) {
      const y = top + row * rowH;
      const layer = viewModel.metrics.firstMoeLayer + row;
      if (row % 4 === 0 || row === layerCount - 1) ctx.fillText(`L${layer}`, left - 8, y + rowH * 0.5);
      for (let expertId = 0; expertId < experts; expertId++) {
        const index = row * experts + expertId;
        const raw = values[index] || 0;
        const normalized = metricId === 'loadRatio' ? raw : raw / maxValue * 1.35;
        ctx.fillStyle = loadColor(normalized);
        ctx.fillRect(left + expertId * colW, y, Math.max(1, colW + 0.25), Math.max(1, rowH + 0.25));
      }
    }

    ctx.strokeStyle = readCssVar('--border-default', 'rgba(255,255,255,0.12)');
    ctx.strokeRect(left, top, gridW, gridH);
    ctx.textAlign = 'center';
    ctx.fillStyle = readCssVar('--foreground-muted', 'rgba(255,255,255,0.45)');
    [0, 64, 128, 192, 255].forEach(expertId => {
      const x = left + gridW * expertId / Math.max(1, experts - 1);
      ctx.fillText(`E${expertId}`, x, height - 9);
    });

    const outline = cell => {
      if (!cell) return;
      ctx.strokeStyle = tokenColor('--primary', '#4369ef');
      ctx.lineWidth = cell === selectedCell ? 2 : 1.2;
      ctx.strokeRect(left + cell.expertId * colW, top + cell.row * rowH, Math.max(5, colW), Math.max(5, rowH));
    };
    outline(hoverCell);
    outline(selectedCell);
    renderDetail();
  }

  function mount() {
    panel.innerHTML = `
      <div class="opv-analysis-view">
        ${makeStatGrid(viewModel.stats)}
        <div class="opv-analysis-controls" data-metric-tabs></div>
        <div class="opv-analysis-grid-wrap">
          <canvas class="opv-moe-heatmap" aria-label="MoE expert load heatmap"></canvas>
          <aside class="opv-analysis-detail"></aside>
        </div>
      </div>`;
    canvas = panel.querySelector('canvas');
    detail = panel.querySelector('.opv-analysis-detail');
    tooltip = makeTooltip(panel);
    const metricTabs = panel.querySelector('[data-metric-tabs]');
    viewModel.metricOptions.forEach(item => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'opv-analysis-chip';
      btn.textContent = item.label;
      btn.addEventListener('click', () => {
        metricId = item.id;
        metricTabs.querySelectorAll('button').forEach(tab => tab.classList.toggle('is-active', tab === btn));
        draw();
      });
      if (item.id === metricId) btn.classList.add('is-active');
      metricTabs.appendChild(btn);
    });
    canvas.addEventListener('pointermove', event => {
      hoverCell = cellAt(event);
      if (!hoverCell) {
        tooltip.hide();
        draw();
        return;
      }
      const layer = viewModel.metrics.layers[hoverCell.row];
      const value = cellValue(hoverCell);
      tooltip.show(event, `
        <b>L${hoverCell.layer} · Expert ${hoverCell.expertId}</b>
        <span>${esc(metric().label)}: ${Number(value).toFixed(metricId === 'loadRatio' ? 2 : 0)} ${esc(metric().unit)}</span>
        <span>Layer avg: ${layer.avgLoadRatio.toFixed(2)}x · p95: ${layer.p95LoadRatio.toFixed(2)}x</span>
      `);
      draw();
    });
    canvas.addEventListener('pointerleave', () => {
      hoverCell = null;
      tooltip.hide();
      draw();
    });
    canvas.addEventListener('click', event => {
      const cell = cellAt(event);
      if (!cell) return;
      selectedCell = cell;
      onSelect?.({ type: 'expert', layer: cell.layer, expertId: cell.expertId, metricId, value: cellValue(cell) });
      draw();
    });
  }

  function setViewModel(next) { viewModel = next; selectedCell = null; hoverCell = null; mount(); draw(); }

  return { panel, mount, render: draw, resize: draw, setViewModel };
}

const CARDLOAD_STATE_LABEL = { ok: '正常', warn: '过载', alert: '饥饿' };

export function createCardLoadView({ panel, viewModel, onSelect }) {
  let tooltip;
  let grid;
  const pct = v => Math.round((v || 0) * 100);

  function paintCards() {
    if (!grid) return;
    const localGroupSize = Math.max(1, (viewModel.tpCount || 1) * (viewModel.epCount || 1));
    grid.style.setProperty('--card-cols', String(localGroupSize));
    grid.innerHTML = viewModel.cards.map(card => {
      const state = card.state === 'alert' ? ' is-alert' : card.state === 'warn' ? ' is-warn' : '';
      const heat = Math.round(4 + card.utilRatio * 14);            // 4–18% 低饱和平涂
      const flow = Math.max(4, pct(card.commRatio));
      return `<button class="opv-cardload-cell${state}" type="button" data-card-id="${card.cardId}" style="--heat-soft:${heat}%;--flow:${flow}%">`
        + `<div class="cl-top"><b>R${card.cardId}</b><span>D${card.dp}P${card.stage}T${card.tp}E${card.ep}</span></div>`
        + `<div class="cl-meter"><i></i></div>`
        + `<div class="cl-bot"><span>u${pct(card.utilRatio)}</span><span>c${pct(card.commRatio)}</span></div>`
        + `</button>`;
    }).join('');
    grid.querySelectorAll('[data-card-id]').forEach(el => {
      const card = viewModel.cards.find(item => String(item.cardId) === el.dataset.cardId);
      el.addEventListener('pointermove', event => tooltip.show(event, `
        <b>Card / Rank ${card.cardId}</b>
        <span>D${card.dp} · P${card.stage} · TP${card.tp} · EP${card.ep}</span>
        <span>util ${pct(card.utilRatio)}% · comm ${pct(card.commRatio)}% · ${CARDLOAD_STATE_LABEL[card.state]}</span>
      `));
      el.addEventListener('pointerleave', () => tooltip.hide());
      el.addEventListener('click', () => onSelect?.({ type: 'card', rank: card.cardId, dp: card.dp, stage: card.stage, tp: card.tp, ep: card.ep }));
    });
  }

  return {
    panel,
    mount() {
      panel.innerHTML = `
        <div class="opv-analysis-view opv-cardload">
          <div class="opv-cardload-head">
            ${makeStatGrid(viewModel.stats)}
            <button class="opv-cardload-info" type="button" aria-label="占用率规则" title="占用率规则">
              <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="8" cy="4.6" r="1" fill="currentColor"/><rect x="7.2" y="6.8" width="1.6" height="5.2" rx="0.8" fill="currentColor"/></svg>
            </button>
            <div class="opv-cardload-pop" hidden>
              <h4>卡占用率怎么算</h4>
              <p><code>util = Σ compute_us / iter_wall_us</code> · 每个训练 step 聚合一次</p>
              <p>采样粒度 <b>1 值 / 卡 / step</b>，随播放条走；跨过路由坍缩点占用会重分布。</p>
              <ul><li>底色浓淡 = util 占用</li><li>下方 meter = comm 通信占比</li></ul>
              <div class="opv-cardload-legend">
                <span class="ok">正常</span><span class="warn">过载 util&gt;95% / comm&gt;50%</span><span class="alert">饥饿 util&lt;30%</span>
              </div>
            </div>
          </div>
          <div class="opv-analysis-scroll"><div class="opv-cardload-grid"></div></div>
        </div>`;
      tooltip = makeTooltip(panel);
      grid = panel.querySelector('.opv-cardload-grid');
      const infoBtn = panel.querySelector('.opv-cardload-info');
      const pop = panel.querySelector('.opv-cardload-pop');
      infoBtn.addEventListener('click', event => {
        event.stopPropagation();
        pop.hidden = !pop.hidden;
        if (!pop.hidden) {
          const cleanup = () => { document.removeEventListener('click', onDoc); document.removeEventListener('keydown', onKey); };
          const onDoc = e => { if (!pop.contains(e.target) && e.target !== infoBtn) { pop.hidden = true; cleanup(); } };
          const onKey = e => { if (e.key === 'Escape') { pop.hidden = true; cleanup(); } };
          setTimeout(() => { document.addEventListener('click', onDoc); document.addEventListener('keydown', onKey); }, 0);
        }
      });
      paintCards();
    },
    render: paintCards,
    resize() {},
    setViewModel(next) { viewModel = next; if (!panel.hidden) paintCards(); },
  };
}

// 层内算子小圆点色（按语义）——独立于 3D 场景色，仅用于详情栏图例
const LAYER_SCAN_SEM_DOT = {
  'sem:attention': '#38bdf8',
  'sem:norm': '#94a3b8',
  'sem:gate': '#f59e0b',
  'sem:comm': '#10b981',
  'sem:moe': '#a78bfa',
  'module:decoder': '#7dd3fc',
};

// ===== 第 2 段（层级）+ 第 3 段（算子）：逐层 × step 异常热力图 + 层内算子分解 =====
export function createLayerScanView({ panel, model, onSelect, getStep }) {
  let canvas, detail, tooltip;
  let hover = null;
  let selLayer = model.epicenter.layer;   // 选中层贯穿始终；step 始终跟随全局播放头
  const L = 42, T = 10, R = 8, B = 20;     // 热力图内边距

  const stepNow = () => {
    const s = typeof getStep === 'function' ? getStep() : null;
    return s ?? model.epicenter.firstDivergeStep ?? model.steps[0];
  };

  function scanColor(v) {
    const theme = document.documentElement.dataset.theme || 'dark';
    const cold = theme === 'light' ? '#e7edf4' : '#1c232e';
    const calm = theme === 'light' ? '#8fb7a6' : '#2f7d63';
    const warn = theme === 'light' ? '#e8ce69' : '#d6aa35';
    const hot = theme === 'light' ? '#e2564e' : '#e5484d';
    if (v <= 0.06) return cold;
    if (v < 0.35) return mixColor(cold, calm, (v - 0.06) / 0.29);
    if (v < 0.7) return mixColor(calm, warn, (v - 0.35) / 0.35);
    return mixColor(warn, hot, Math.min(1, (v - 0.7) / 0.3));
  }

  function gridBox() {
    const wrap = panel.querySelector('[data-scan-wrap]');
    const wb = wrap.getBoundingClientRect();
    const width = Math.max(360, wb.width - 250 - 8);   // 减去 detail(250) + gap(8)
    const height = Math.max(220, wb.height - 2);
    return { width, height, gridW: width - L - R, gridH: height - T - B };
  }

  function cellAt(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const gridW = rect.width - L - R, gridH = rect.height - T - B;
    if (x < L || x > L + gridW || y < T || y > T + gridH) return null;
    const layer = Math.min(model.totalLayers - 1, Math.max(0, Math.floor((y - T) / gridH * model.totalLayers)));
    const si = Math.min(model.stepCount - 1, Math.max(0, Math.floor((x - L) / gridW * model.stepCount)));
    return { layer, step: model.steps[si], si };
  }

  function draw() {
    if (!canvas || panel.hidden) return;
    const { width, height, gridW, gridH } = gridBox();
    const ctx = resizeCanvas(canvas, width, height);
    const rows = model.totalLayers, cols = model.stepCount;
    const rowH = gridH / rows, colW = gridW / cols;
    ctx.clearRect(0, 0, width, height);

    for (let layer = 0; layer < rows; layer++) {
      const y = T + layer * rowH;
      for (let si = 0; si < cols; si++) {
        ctx.fillStyle = scanColor(model.scores[layer * cols + si]);
        ctx.fillRect(L + si * colW, y, Math.max(1, colW + 0.4), Math.max(1, rowH + 0.4));
      }
    }
    ctx.strokeStyle = readCssVar('--border-default', 'rgba(255,255,255,0.12)');
    ctx.strokeRect(L, T, gridW, gridH);

    // 层轴标签（每 8 层）
    ctx.fillStyle = readCssVar('--foreground-muted', 'rgba(255,255,255,0.45)');
    ctx.font = '600 9px JetBrains Mono, monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let layer = 0; layer < rows; layer += 8) ctx.fillText(`L${layer}`, L - 6, T + (layer + 0.5) * rowH);

    // 分区分隔线：Dense↔MoE 边界
    const moeY = T + model.firstMoeLayer * rowH;
    ctx.strokeStyle = readCssVar('--border-strong', 'rgba(255,255,255,0.28)');
    ctx.setLineDash([2, 2]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, moeY); ctx.lineTo(L + gridW, moeY); ctx.stroke(); ctx.setLineDash([]);

    // step 轴标签（首 / fault / collapse / 末）
    const mapX = step => L + (model.steps.indexOf(step) + 0.5) * colW;
    ctx.fillStyle = readCssVar('--foreground-muted', 'rgba(255,255,255,0.45)');
    ctx.font = '9px JetBrains Mono, monospace'; ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText(String(model.steps[0]), L, T + gridH + 5);
    ctx.textAlign = 'right'; ctx.fillText(String(model.steps[cols - 1]), L + gridW, T + gridH + 5);

    // 故障 / 坍缩竖线
    const danger = readCssVar('--danger', '#e5484d');
    [[model.faultStep, '故障', readCssVar('--warning', '#d6aa35')], [model.collapseStep, '坍缩', danger]].forEach(([st, name, col]) => {
      const cx = mapX(st);
      ctx.strokeStyle = col; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, T); ctx.lineTo(cx, T + gridH); ctx.stroke(); ctx.setLineDash([]);
    });

    // 首问题层 (epicenter)：行高亮 + ◆ 标记在首超标 step
    const epi = model.epicenter;
    const ey = T + (epi.layer + 0.5) * rowH;
    ctx.strokeStyle = danger; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, ey); ctx.lineTo(L + gridW, ey); ctx.stroke();
    if (epi.firstDivergeStep != null) {
      const dx = mapX(epi.firstDivergeStep);
      ctx.fillStyle = danger;
      ctx.beginPath();
      ctx.moveTo(dx, ey - 4); ctx.lineTo(dx + 4, ey); ctx.lineTo(dx, ey + 4); ctx.lineTo(dx - 4, ey); ctx.closePath();
      ctx.fill();
    }

    // 全局 step 游标（竖线）
    const cursorX = mapX(stepNow());
    ctx.strokeStyle = readCssVar('--foreground-secondary', 'rgba(255,255,255,0.62)'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cursorX, T); ctx.lineTo(cursorX, T + gridH); ctx.stroke();

    // 选中层 + hover 描边
    const primary = tokenColor('--primary', '#4369ef');
    const outlineRow = (layer, lw) => {
      ctx.strokeStyle = primary; ctx.lineWidth = lw;
      ctx.strokeRect(L, T + layer * rowH, gridW, Math.max(2, rowH));
    };
    if (hover && hover.layer !== selLayer) outlineRow(hover.layer, 1);
    outlineRow(selLayer, 1.8);
  }

  function renderSpark(el, layer, step) {
    if (!el) return;
    const w = 226, h = 42, pad = 3, n = model.stepCount;
    const xs = i => pad + (n > 1 ? i / (n - 1) : 0) * (w - 2 * pad);
    const ys = v => (h - pad) - v * (h - 2 * pad);
    let d = '';
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + xs(i).toFixed(1) + ' ' + ys(model.scores[layer * n + i]).toFixed(1) + ' ';
    const thrY = ys(model.warnThreshold);
    const ci = Math.max(0, model.steps.indexOf(step));
    const cx = xs(ci), cv = model.scores[layer * n + ci];
    el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" aria-label="L${layer} 异常分随 step">
      <line x1="${pad}" x2="${w - pad}" y1="${thrY.toFixed(1)}" y2="${thrY.toFixed(1)}" stroke="var(--warning)" stroke-dasharray="3 3" stroke-width="1" opacity="0.7"/>
      <path d="${d}" fill="none" stroke="var(--danger)" stroke-width="1.4"/>
      <line x1="${cx.toFixed(1)}" x2="${cx.toFixed(1)}" y1="${pad}" y2="${h - pad}" stroke="var(--foreground-secondary)" stroke-width="1"/>
      <circle cx="${cx.toFixed(1)}" cy="${ys(cv).toFixed(1)}" r="2.4" fill="var(--danger)"/>
    </svg>`;
  }

  function renderDetail() {
    if (!detail) return;
    const step = stepNow();
    const layer = selLayer;
    const info = model.layers[layer];
    const pl = model.perLayer[layer];
    const score = model.scoreAt(layer, step);
    const ops = [...model.opBreakdown(layer, step)].sort((a, b) => b.share - a.share);
    const bars = ops.map(o => `
      <div class="opv-scan-op">
        <span class="opv-scan-op__name"><i style="background:${LAYER_SCAN_SEM_DOT[o.sem] || '#888'}"></i>${esc(o.label)}</span>
        <div class="opv-scan-op__bar"><i style="width:${(o.share * 100).toFixed(0)}%;background:${scanColor(o.score)}"></i></div>
        <span class="opv-scan-op__val">${(o.share * 100).toFixed(0)}%</span>
      </div>`).join('');
    detail.innerHTML = `
      <div class="opv-analysis-detail-title">L${layer} · step ${step}</div>
      <div class="opv-scan-spark" data-scan-spark></div>
      <div class="opv-analysis-detail-grid">
        <span>异常分</span><b>${score.toFixed(2)}</b>
        <span>层类型</span><b>${info.isMoe ? 'MoE' : 'Dense'}</b>
        <span>首超标</span><b>${pl.firstDivergeStep != null ? 'step ' + pl.firstDivergeStep : '—'}</b>
        <span>峰值</span><b>${pl.peak.toFixed(2)} @${pl.peakStep}</b>
      </div>
      <div class="opv-scan-oplabel">算子分解 · 占该层异常</div>
      <div class="opv-scan-ops">${bars}</div>`;
    renderSpark(detail.querySelector('[data-scan-spark]'), layer, step);
  }

  function mount() {
    panel.innerHTML = `
      <div class="opv-analysis-view">
        ${makeStatGrid(model.stats)}
        <div class="opv-scan-legend">
          <span>行 = 层 L0–L${model.lastMoeLayer}</span>
          <span>列 = step（时间）</span>
          <span class="opv-scan-key">低<i style="background:${scanColor(0.2)}"></i><i style="background:${scanColor(0.5)}"></i><i style="background:${scanColor(0.9)}"></i>高</span>
          <span class="opv-scan-epi">◆ 首问题层 L${model.epicenter.layer}${model.epicenter.firstDivergeStep != null ? ' @ step ' + model.epicenter.firstDivergeStep : ''}</span>
        </div>
        <div class="opv-analysis-grid-wrap" data-scan-wrap>
          <canvas class="opv-scan-heatmap" aria-label="逐层 × step 异常热力图"></canvas>
          <aside class="opv-analysis-detail" data-scan-detail></aside>
        </div>
      </div>`;
    canvas = panel.querySelector('.opv-scan-heatmap');
    detail = panel.querySelector('[data-scan-detail]');
    tooltip = makeTooltip(panel.querySelector('[data-scan-wrap]'));
    canvas.addEventListener('pointermove', event => {
      hover = cellAt(event);
      if (!hover) { tooltip.hide(); draw(); return; }
      const score = model.scoreAt(hover.layer, hover.step);
      const info = model.layers[hover.layer];
      tooltip.show(event, `
        <b>L${hover.layer} · step ${hover.step}</b>
        <span>异常分 ${score.toFixed(2)} · ${info.isMoe ? 'MoE' : 'Dense'}</span>
        <span>点击下钻该层算子分解</span>`);
      draw();
    });
    canvas.addEventListener('pointerleave', () => { hover = null; tooltip.hide(); draw(); });
    canvas.addEventListener('click', event => {
      const cell = cellAt(event);
      if (!cell) return;
      selLayer = cell.layer;
      onSelect?.({ type: 'layer', layer: cell.layer, step: cell.step, score: model.scoreAt(cell.layer, cell.step) });
      draw(); renderDetail();
    });
  }

  return {
    panel, mount,
    render() { draw(); renderDetail(); },
    resize() { draw(); },
  };
}
