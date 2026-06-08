/* GEW.swimlane — bottom evidence swimlane on an HTML5 canvas.
   Task bars are drawn by PtoSwimlaneTaskPattern.drawTaskBar (pattern owns segment
   geometry); step lane bars are proportional compute/comm/free/overlap segments;
   coverage lane contrasts runtime-evidenced graph nodes vs text-only nodes.
   Hover tooltip + click selection via canvas hit-testing. See CONTRACT.md §5/§6/§7/§8. */
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};
  var GEW = w.GEW;
  var util = GEW.util;
  var P = w.PtoSwimlaneTaskPattern;

  var FONT = 'var(--font-sans)';
  // resolved concrete font-family for canvas (canvas can't use CSS vars)
  var CANVAS_FONT = '12px -apple-system, "Segoe UI", Roboto, system-ui, sans-serif';
  var CANVAS_FONT_FAMILY = '-apple-system, "Segoe UI", Roboto, system-ui, sans-serif';

  // layout constants
  var AXIS_H = 26;            // top time-axis height
  var ROW_H = 38;             // lane row height
  var ROW_GAP = 8;
  var BAR_H = 22;             // task bar height inside a row
  var LABEL_W = 96;           // left gutter for row labels
  var PAD_R = 16;
  var MIN_BAR_W = 6;

  // semantic colors resolved once from CSS vars (canvas needs concrete colors)
  var COLORS = {};
  function resolveColors() {
    var cs = getComputedStyle(document.getElementById('gew-root') || document.documentElement);
    function v(name, fallback) {
      var x = cs.getPropertyValue(name);
      return (x && x.trim()) || fallback;
    }
    COLORS = {
      danger: v('--danger', '#d9544e'),
      warning: v('--warning', '#d99a2b'),
      success: v('--success', '#4a9568'),
      primary: v('--primary', '#4d70ba'),
      muted: v('--foreground-secondary', '#6b7280'),
      axis: v('--border-subtle', '#e3e5e9'),
      text: v('--foreground', '#1b1d22'),
      surface: v('--gew-panel-bg', v('--surface-2', '#ffffff')),
      playhead: v('--primary', '#4d70ba'),
      chipOk: v('--success', '#4a9568'),
      chipMiss: v('--warning', '#d99a2b'),
    };
  }

  // module state
  var S = {
    container: null,
    laneModel: null,
    canvas: null,
    ctx: null,
    tooltip: null,
    colormap: null,
    activeTab: 'step',
    selectedTaskId: null,
    selectedStepId: null,
    highlightNodeId: null,
    relatedTaskIds: {},     // taskId -> true
    playheadUs: null,
    hitRects: [],           // [{x,y,w,h, task, nodeId}]
    ro: null,
    busWired: false,
  };

  // ---- time scale -------------------------------------------------------
  function timeBounds() {
    var tr = (S.laneModel && S.laneModel.timeRange) || { startUs: 0, endUs: 0 };
    var start = tr.startUs || 0;
    var end = tr.endUs || (start + 1);
    if (end <= start) end = start + 1;
    return { start: start, end: end };
  }
  function plotWidth() {
    var cw = S.container ? S.container.clientWidth : 800;
    return Math.max(120, cw - LABEL_W - PAD_R);
  }
  function pxPerUs() {
    var tb = timeBounds();
    return plotWidth() / (tb.end - tb.start);
  }
  function xForUs(us) {
    var tb = timeBounds();
    return LABEL_W + (us - tb.start) * pxPerUs();
  }

  // ---- lane selection ---------------------------------------------------
  function lanesOfKind(kind) {
    if (!S.laneModel || !S.laneModel.lanes) return [];
    return S.laneModel.lanes.filter(function (l) { return l.kind === kind; });
  }
  function gapFor(kind) {
    if (!S.laneModel || !S.laneModel.gaps) return null;
    // a gap entry whose kind matches; only treat as empty if the kind has NO
    // lane with tasks (avoid hiding a partially-present lane).
    var lanes = lanesOfKind(kind);
    var anyTasks = lanes.some(function (l) { return l.tasks && l.tasks.length; });
    if (kind === 'coverage') return null; // coverage built locally, never gap-blocked
    if (anyTasks) return null;
    var g = S.laneModel.gaps.filter(function (x) { return x.kind === kind; });
    return g.length ? g : null;
  }

  // ---- mapped-task -> nodeId reverse lookup -----------------------------
  function nodeIdForTask(task) {
    if (task && task.nodeId) return task.nodeId;
    var byNode = (S.laneModel && S.laneModel.byNode) || {};
    var ids = Object.keys(byNode);
    for (var i = 0; i < ids.length; i++) {
      if (byNode[ids[i]].indexOf(task.id) >= 0) return ids[i];
    }
    return null;
  }

  // ---- colormap (semantic op coloring via pattern) ----------------------
  function baseColorForTask(task) {
    if (S.colormap) return S.colormap.colorForTask(task, 'semantic');
    return '#5b8def';
  }

  function lightColormapOptions() {
    return {
      palette: ['#315fbe', '#0f766e', '#7c3aed', '#b45309', '#be185d', '#2563eb', '#5f6b7a', '#2f7d32'],
      saturation: 52,
      lightness: 38,
      subgraphSaturation: 48,
      subgraphLightness: 40,
      labelColors: {
        MatMulV3: '#315fbe',
        MatMulV2: '#315fbe',
        Exp: '#be185d',
        Sub: '#b45309',
        RealDiv: '#7c3aed',
        ReduceSum: '#0f766e',
        Cast: '#5f6b7a',
        ArgMaxWithValue: '#9f1239',
        ApplyAdamWV2: '#7c2d12',
        hcom: '#0f766e',
        fake: '#64748b',
        unknown: '#64748b',
      },
      laneKindColors: {
        fake: '#64748b',
        aic: '#315fbe',
        AIC: '#315fbe',
        aiv: '#7c3aed',
        AIV: '#7c3aed',
        aicpu: '#0f766e',
        AICCtrl: '#0f766e',
        AICSched: '#0f766e',
        MTEIn: '#b45309',
        MTEOut: '#b45309',
        other: '#5f6b7a',
      },
    };
  }

  // map our LaneModel task -> pattern task fields
  function mapTask(t) {
    var clc = Math.max(0, (t.durUs || 0) - (t.waitUs || 0));
    return {
      // identity / labels
      id: t.id,
      opName: t.opName || t.label,
      label: t.label || t.opName,
      laneKind: t.laneKind,
      laneId: t.streamId != null ? 'Stream ' + t.streamId : (t.rankId != null ? 'Rank ' + t.rankId : t.laneKind),
      status: t.status,
      // pattern timing aliases
      totalCycle: t.durUs,
      clcCycle: clc,
      gap: t.waitUs,
      gapRatio: t.durUs ? (t.waitUs || 0) / t.durUs : null,
      inputRawMagic: t.inputRawMagic,
      outputRawMagic: t.outputRawMagic,
      // pass-through originals (for our custom tooltip)
      _src: t,
    };
  }

  function isPinnedTask(t) {
    return !!(t && (S.selectedTaskId === t.id || S.relatedTaskIds[t.id]));
  }

  function compactLaneTasks(tasks) {
    if (S.activeTab !== 'stream' || !Array.isArray(tasks) || tasks.length < 8) return tasks || [];
    var ppu = pxPerUs();
    var sorted = tasks.slice().sort(function (a, b) { return (a.startUs || 0) - (b.startUs || 0); });
    var out = [];
    var group = null;

    function flushGroup() {
      if (!group) return;
      if (group.items.length === 1) {
        out.push(group.items[0]);
      } else {
        var first = group.items[0];
        var dur = Math.max(1, group.endUs - group.startUs);
        out.push({
          id: 'burst-' + first.id + '-' + group.items.length,
          label: group.items.length + ' ops',
          opName: group.items.length + ' runtime ops',
          laneKind: first.laneKind,
          streamId: first.streamId,
          rankId: first.rankId,
          startUs: group.startUs,
          durUs: dur,
          waitUs: group.waitUs,
          status: group.waitUs > dur * 0.18 ? 'wait' : 'ok',
          sourceFile: first.sourceFile || 'trace_view.json',
          _grouped: true,
          _items: group.items.slice(),
        });
      }
      group = null;
    }

    sorted.forEach(function (t) {
      var start = t.startUs || 0;
      var end = start + (t.durUs || 0);
      var widthPx = (t.durUs || 0) * ppu;
      if (isPinnedTask(t) || widthPx >= 14) {
        flushGroup();
        out.push(t);
        return;
      }
      var gapPx = group ? (start - group.endUs) * ppu : 0;
      if (!group || gapPx > 18 || group.items.length >= 16) {
        flushGroup();
        group = { startUs: start, endUs: end, waitUs: 0, items: [] };
      }
      group.items.push(t);
      group.endUs = Math.max(group.endUs, end);
      group.waitUs += t.waitUs || 0;
    });
    flushGroup();
    return out;
  }

  // ---- canvas setup -----------------------------------------------------
  function ensureCanvas() {
    if (S.canvas) return;
    S.canvas = util.el('canvas', { style: 'display:block;position:absolute;left:0;top:0;' });
    S.ctx = S.canvas.getContext('2d');
    // insert before the empty-state overlay so overlay stays on top
    var empty = util.qs('gew-swimlane-empty');
    if (empty && empty.parentNode === S.container) S.container.insertBefore(S.canvas, empty);
    else S.container.appendChild(S.canvas);

    // tooltip (reuse pattern's tooltip class + CSS)
    S.tooltip = P ? P.createTooltip() : util.el('div', { class: 'pto-swimlane-task-tooltip' });
    S.container.appendChild(S.tooltip);

    S.canvas.addEventListener('mousemove', onMove);
    S.canvas.addEventListener('mouseleave', onLeave);
    S.canvas.addEventListener('click', onClick);
  }

  function sizeCanvas(cssW, cssH) {
    var dpr = w.devicePixelRatio || 1;
    S.canvas.style.width = cssW + 'px';
    S.canvas.style.height = cssH + 'px';
    S.canvas.width = Math.round(cssW * dpr);
    S.canvas.height = Math.round(cssH * dpr);
    S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ---- drawing primitives ----------------------------------------------
  function drawAxis(cssW) {
    var ctx = S.ctx;
    var tb = timeBounds();
    ctx.save();
    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, cssW, AXIS_H);
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, AXIS_H - 0.5);
    ctx.lineTo(cssW, AXIS_H - 0.5);
    ctx.stroke();

    ctx.font = '10px ' + CANVAS_FONT_FAMILY;
    ctx.fillStyle = COLORS.muted;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    var span = tb.end - tb.start;
    var ticks = 6;
    for (var i = 0; i <= ticks; i++) {
      var us = tb.start + (span * i) / ticks;
      var x = xForUs(us);
      ctx.strokeStyle = COLORS.axis;
      ctx.beginPath();
      ctx.moveTo(x, AXIS_H - 6);
      ctx.lineTo(x, AXIS_H);
      ctx.stroke();
      var rel = us - tb.start; // relative time from start, more readable
      var lbl = util.fmtUs(rel);
      if (i === ticks) ctx.textAlign = 'right';
      ctx.fillText(lbl, i === ticks ? x - 1 : x + 3, AXIS_H / 2 - 1);
      ctx.textAlign = 'left';
    }
    ctx.restore();
  }

  function drawPlayhead(cssH) {
    if (S.playheadUs == null) return;
    var x = xForUs(S.playheadUs);
    var ctx = S.ctx;
    ctx.save();
    ctx.strokeStyle = COLORS.playhead;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, cssH);
    ctx.stroke();
    ctx.restore();
  }

  function rowLabel(text, y) {
    var ctx = S.ctx;
    ctx.save();
    ctx.font = '10px ' + CANVAS_FONT_FAMILY;
    ctx.fillStyle = COLORS.muted;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    var t = String(text);
    if (t.length > 14) t = t.slice(0, 13) + '…';
    ctx.fillText(t, 8, y + ROW_H / 2);
    ctx.restore();
  }

  // draw a task lane (stream / communication / overlap) — one row per lane group
  function drawTaskLane(lanes, topY) {
    var y = topY;
    var ppu = pxPerUs();
    lanes.forEach(function (lane) {
      rowLabel(lane.label, y);
      var barY = y + (ROW_H - BAR_H) / 2;
      compactLaneTasks(lane.tasks || []).forEach(function (t) {
        var x = xForUs(t.startUs);
        var width = Math.max(MIN_BAR_W, (t.durUs || 0) * ppu);
        // clamp to plot region
        if (x + width < LABEL_W || x > S.canvas.clientWidth) return;
        var mapped = mapTask(t);
        var isSel = S.selectedTaskId === t.id;
        var isRel = !!S.relatedTaskIds[t.id];
        if (P && P.drawTaskBar) {
          P.drawTaskBar(S.ctx, {
            task: mapped,
            x: x, y: barY, width: width, height: BAR_H,
            baseColor: baseColorForTask(t),
            fontFamily: CANVAS_FONT_FAMILY,
            isSelected: isSel,
            isRelated: isRel,
            isEmphasized: isRel && !isSel,
          });
        }
        S.hitRects.push({ x: x, y: barY, w: width, h: BAR_H, task: t, nodeId: t._grouped ? null : nodeIdForTask(t) });
      });
      y += ROW_H + ROW_GAP;
    });
    return y;
  }

  // step lane — proportional compute/comm/free/overlap segments per Step
  function drawStepLane(topY) {
    var steps = (S.laneModel && S.laneModel.steps) || [];
    var ctx = S.ctx;
    var ppu = pxPerUs();
    var y = topY;
    steps.forEach(function (s) {
      rowLabel('Step ' + s.stepId, y);
      var barY = y + (ROW_H - BAR_H) / 2;
      var x0 = xForUs(s.startUs);
      var totalW = Math.max(MIN_BAR_W, (s.durUs || 0) * ppu);
      var isSel = S.selectedStepId === s.stepId;
      var stepTask = {
        id: 'step-' + s.stepId, label: 'Step ' + s.stepId, opName: 'Step ' + s.stepId,
        startUs: s.startUs, durUs: s.durUs, waitUs: s.free, stepId: s.stepId,
        totalCycle: s.durUs,
        clcCycle: Math.max(0, (s.compute || 0) - (s.overlap || 0)),
        gap: s.free,
        gapRatio: s.durUs ? (s.free || 0) / s.durUs : null,
        status: s.overlap > 0 ? 'overlap' : (s.free > 0.3 * s.durUs ? 'wait' : 'ok'),
        laneKind: 'step', sourceFile: 'trace_view.json',
        _step: s,
      };

      // compute/comm/free are mutually exclusive wall-clock proportions here.
      // overlap is shown by status/playhead, not added as a fourth duration segment.
      var effectiveCompute = Math.max(0, (s.compute || 0) - (s.overlap || 0));
	      var parts = [
	        { v: effectiveCompute, c: COLORS.success, label: 'compute' },
	        { v: s.comm, c: COLORS.primary, label: 'comm' },
	        { v: s.free, c: COLORS.danger, label: 'free' },
	      ];
      var sum = parts.reduce(function (a, p) { return a + Math.max(0, p.v || 0); }, 0) || 1;

      if (P && P.drawTaskBar) {
        P.drawTaskBar(ctx, {
          task: stepTask,
          x: x0,
          y: barY,
          width: totalW,
          height: BAR_H,
          baseColor: s.free > 0.3 * s.durUs ? COLORS.danger : COLORS.success,
          fontFamily: CANVAS_FONT_FAMILY,
          isSelected: isSel,
          isRelated: false,
          isEmphasized: s.overlap > 0,
        });
      } else {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(x0, barY, totalW, BAR_H, 3);
        ctx.fillStyle = COLORS.surface;
        ctx.fill();
        ctx.restore();
      }

      var segX = x0;
      var railY = barY + BAR_H - 5;
	      parts.forEach(function (p) {
	        var pw = (Math.max(0, p.v || 0) / sum) * totalW;
	        if (pw <= 0) return;
	        ctx.fillStyle = p.c;
	        ctx.globalAlpha = isSel ? 0.95 : 0.78;
	        ctx.fillRect(segX, railY, pw, 4);
	        segX += pw;
	      });
	      if (s.overlap > 0 && s.durUs > 0) {
	        ctx.fillStyle = COLORS.warning;
	        ctx.globalAlpha = isSel ? 0.98 : 0.86;
	        ctx.fillRect(x0, railY - 4, Math.max(2, Math.min(totalW, (s.overlap / s.durUs) * totalW)), 3);
	      }
	      ctx.globalAlpha = 1;

      // a synthetic task for hover/click on the step row (maps to step focus)
      S.hitRects.push({ x: x0, y: barY, w: totalW, h: BAR_H, task: stepTask, nodeId: null, isStep: true });
      y += ROW_H + ROW_GAP;
    });

    // compact legend row
    drawStepLegend(y);
    return y + ROW_H;
  }

  function drawStepLegend(y) {
    var ctx = S.ctx;
    var items = [
      { c: COLORS.success, t: 'compute' },
      { c: COLORS.primary, t: 'comm' },
      { c: COLORS.warning, t: 'overlap' },
      { c: COLORS.danger, t: 'free / bubble' },
    ];
    ctx.save();
    ctx.font = '10px ' + CANVAS_FONT_FAMILY;
    ctx.textBaseline = 'middle';
    var x = LABEL_W;
    var cy = y + 10;
    items.forEach(function (it) {
      ctx.fillStyle = it.c;
      ctx.fillRect(x, cy - 5, 10, 10);
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = 'left';
      ctx.fillText(it.t, x + 14, cy);
      x += 14 + ctx.measureText(it.t).width + 18;
    });
    ctx.restore();
  }

  // coverage lane — runtime-evidenced nodes vs text/AI-only nodes
  function drawCoverageLane(topY) {
    var ctx = S.ctx;
    var byNode = (S.laneModel && S.laneModel.byNode) || {};
    var data = GEW.state.data || {};
    var problemMap = data.problemMap || {};
    var problemNodes = problemMap.problemNodes || {};
    var graphNodes = (data.graph && data.graph.nodes) || [];

    // candidate node universe: graph nodes + problem nodes
    var universe = {};
    graphNodes.forEach(function (n) { universe[n.id] = n.label || n.id; });
    Object.keys(problemNodes).forEach(function (id) {
      if (!universe[id]) universe[id] = (problemNodes[id].title || id);
    });

    var covered = [];   // has runtime byNode evidence
    var textOnly = [];  // appears in graph/problem but no runtime tasks
    Object.keys(universe).forEach(function (id) {
      if (byNode[id] && byNode[id].length) covered.push(id);
      else textOnly.push(id);
    });

    var y = topY;
    function group(title, ids, color, withEvidence) {
      ctx.save();
      ctx.font = '10px ' + CANVAS_FONT_FAMILY;
      ctx.fillStyle = COLORS.muted;
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.fillText(title + ' (' + ids.length + ')', 8, y + 9);
      ctx.restore();
      y += 20;

      var x = LABEL_W;
      var chipH = 20;
      ids.forEach(function (id) {
        ctx.save();
        ctx.font = '10px ' + CANVAS_FONT_FAMILY;
        var label = id;
        var tw = ctx.measureText(label).width + 18;
        if (x + tw > (S.canvas.clientWidth - PAD_R)) { x = LABEL_W; y += chipH + 6; }
        ctx.beginPath();
        ctx.roundRect(x, y, tw, chipH, 4);
        ctx.fillStyle = withEvidence ? 'rgba(74,149,104,0.12)' : 'rgba(217,154,43,0.10)';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.stroke();
        // marker dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x + 7, y + chipH / 2, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = COLORS.text;
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        ctx.fillText(label, x + 14, y + chipH / 2);
        ctx.restore();

        // chips clickable → select node
        S.hitRects.push({
          x: x, y: y, w: tw, h: chipH,
          task: { id: 'cov-' + id, nodeId: id, label: id, opName: id, laneKind: 'coverage' },
          nodeId: id, isCoverage: true,
        });
        x += tw + 8;
      });
      y += chipH + 14;
    }

    group('运行时已覆盖 · runtime evidence', covered, COLORS.chipOk, true);
    group('仅文本/AI 推断 · no runtime task', textOnly, COLORS.chipMiss, false);
    return y;
  }

  // ---- master render ----------------------------------------------------
  function render() {
    if (!S.container || !S.ctx) return;
    S.hitRects = [];

    var empty = util.qs('gew-swimlane-empty');
    var gaps = gapFor(S.activeTab);
    if (gaps) {
      // show empty overlay with gap reasons; clear canvas
      if (empty) {
        empty.classList.add('is-visible');
        var reason = gaps.map(function (g) { return g.reason; }).join('；');
        empty.firstChild ? (empty.firstChild.textContent = '缺少 trace 数据：' + reason)
          : (empty.textContent = '缺少 trace 数据：' + reason);
      }
      var cw = Math.max(120, S.container.clientWidth);
      sizeCanvas(cw, S.container.clientHeight || 220);
      S.ctx.clearRect(0, 0, cw, S.container.clientHeight || 220);
      return;
    }
    if (empty) empty.classList.remove('is-visible');

    // compute content height for this tab
    var contentH = AXIS_H + 8;
    if (S.activeTab === 'step') {
      contentH += ((S.laneModel.steps || []).length) * (ROW_H + ROW_GAP) + ROW_H;
    } else if (S.activeTab === 'coverage') {
      contentH += 320; // generous; chips wrap
    } else {
      contentH += lanesOfKind(S.activeTab).length * (ROW_H + ROW_GAP) + 8;
    }
    var cssW = Math.max(120, S.container.clientWidth);
    var cssH = Math.max(S.container.clientHeight || 200, contentH);
    sizeCanvas(cssW, cssH);

    var ctx = S.ctx;
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, cssW, cssH);

    var bodyTop = AXIS_H + 8;
    if (S.activeTab === 'coverage') {
      // coverage has no time axis meaning; draw a header band instead
      ctx.save();
      ctx.fillStyle = COLORS.muted;
      ctx.font = '10px ' + CANVAS_FONT_FAMILY;
      ctx.textBaseline = 'middle';
      ctx.fillText('计算图节点的运行时证据覆盖', 8, AXIS_H / 2);
      ctx.restore();
      drawCoverageLane(bodyTop);
    } else {
      drawAxis(cssW);
      if (S.activeTab === 'step') drawStepLane(bodyTop);
      else drawTaskLane(lanesOfKind(S.activeTab), bodyTop);
      drawPlayhead(cssH);
    }
  }

  // ---- hover / click ----------------------------------------------------
  function hitAt(mx, my) {
    // iterate reverse so topmost wins
    for (var i = S.hitRects.length - 1; i >= 0; i--) {
      var r = S.hitRects[i];
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) return r;
    }
    return null;
  }

  function tooltipHtml(task) {
    // start with the pattern's formatter, then append start/wait/stream/rank/source
    var mapped = task._step ? null : mapTask(task);
    var base = (P && mapped) ? P.formatTaskTooltip(mapped, { durationUnit: 'µs' }) : '';
    var rows = [];
    function row(k, v, cls) {
      if (v == null || v === '') return;
      rows.push('<div class="pto-swimlane-task-tooltip__row"><span class="pto-swimlane-task-tooltip__key">'
        + util.escapeHtml(k) + '</span><span class="pto-swimlane-task-tooltip__value' + (cls ? ' ' + cls : '')
        + '">' + util.escapeHtml(v) + '</span></div>');
    }
    if (task._step) {
      var s = task._step;
      rows.push('<div class="pto-swimlane-task-tooltip__title">Step ' + util.escapeHtml(s.stepId) + '</div>');
      row('span', util.fmtUs(s.durUs));
      row('compute', util.fmtUs(s.compute));
      row('comm', util.fmtUs(s.comm));
      row('overlap', util.fmtUs(s.overlap));
      row('free', util.fmtUs(s.free), 'is-bad');
      return rows.join('');
    }
    if (task._grouped) {
      rows.push('<div class="pto-swimlane-task-tooltip__title">' + util.escapeHtml(task.opName || task.label) + '</div>');
      row('ops', String((task._items && task._items.length) || 0));
      row('span', util.fmtUs(task.durUs));
      row('start', util.fmtUs((task.startUs || 0) - timeBounds().start));
      row('stream', task.streamId != null ? 'Stream ' + task.streamId : '');
      row('source', task.sourceFile || 'trace_view.json');
      return rows.join('');
    }
    row('op', task.opName || task.label);
    row('lane', task.laneKind + (task.streamId != null ? ' · Stream ' + task.streamId : ''));
    row('start', util.fmtUs((task.startUs || 0) - timeBounds().start));
    row('duration', util.fmtUs(task.durUs));
    row('wait', util.fmtUs(task.waitUs), task.waitUs > 0 ? 'is-bad' : '');
    if (task.rankId != null) row('rank', String(task.rankId));
    row('source', task.sourceFile || 'trace_view.json');
    return base + rows.join('');
  }

  function onMove(e) {
    var rect = S.canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var hit = hitAt(mx, my);
    if (!hit) { hideTip(); S.canvas.style.cursor = 'default'; return; }
    S.canvas.style.cursor = 'pointer';
    showTip(hit.task, e);
  }
  function onLeave() { hideTip(); }

  function showTip(task, e) {
    if (!S.tooltip) return;
    S.tooltip.innerHTML = tooltipHtml(task);
    S.tooltip.classList.add('is-visible');
    S.tooltip.setAttribute('aria-hidden', 'false');
    var bRect = S.container.getBoundingClientRect();
    var x = e.clientX - bRect.left + 14;
    var y = e.clientY - bRect.top + 14;
    var tw = S.tooltip.offsetWidth;
    var th = S.tooltip.offsetHeight;
    x = Math.max(8, Math.min(bRect.width - tw - 8, x));
    y = Math.max(8, Math.min(bRect.height - th - 8, y));
    S.tooltip.style.left = x + 'px';
    S.tooltip.style.top = y + 'px';
  }
  function hideTip() {
    if (!S.tooltip) return;
    S.tooltip.classList.remove('is-visible');
    S.tooltip.setAttribute('aria-hidden', 'true');
  }

  function onClick(e) {
    var rect = S.canvas.getBoundingClientRect();
    var hit = hitAt(e.clientX - rect.left, e.clientY - rect.top);
    if (!hit) return;
    if (hit.isStep) {
      S.selectedStepId = hit.task.stepId;
      S.playheadUs = hit.task.startUs;
      render();
      GEW.bus.emit('step:focus', { stepId: hit.task.stepId });
      return;
    }
    var nodeId = hit.nodeId;
    S.selectedTaskId = hit.task.id;
    render();
    if (nodeId) {
      GEW.state.selectedNodeId = nodeId;
      GEW.bus.emit('selection:change', { nodeId: nodeId, source: 'swimlane' });
    }
    // no mapped node → select task visually only, emit nothing (graceful)
  }

  // ---- highlight related tasks for a node -------------------------------
  function computeRelated(nodeId) {
    S.relatedTaskIds = {};
    if (!nodeId) return [];
    var byNode = (S.laneModel && S.laneModel.byNode) || {};
    var ids = byNode[nodeId] || [];
    ids.forEach(function (tid) { S.relatedTaskIds[tid] = true; });
    return ids;
  }

  // find the first lane kind that contains any of the related task ids
  function laneKindForTasks(taskIds) {
    if (!taskIds.length || !S.laneModel) return null;
    var set = {};
    taskIds.forEach(function (id) { set[id] = true; });
    var lanes = S.laneModel.lanes || [];
    for (var i = 0; i < lanes.length; i++) {
      if (lanes[i].kind === 'step' || lanes[i].kind === 'coverage') continue;
      var has = (lanes[i].tasks || []).some(function (t) { return set[t.id]; });
      if (has) return lanes[i].kind;
    }
    return null;
  }

  // ---- tab UI -----------------------------------------------------------
  function setActiveTabButton(kind) {
    var tabs = util.qs('gew-swimlane-tabs');
    if (!tabs) return;
	    Array.prototype.forEach.call(tabs.querySelectorAll('button[data-lane]'), function (b) {
	      var on = b.getAttribute('data-lane') === kind;
	      b.classList.toggle('is-selected', on);
	      b.classList.toggle('is-active', on);
	    });
  }

  function wireTabs() {
    var tabs = util.qs('gew-swimlane-tabs');
    if (!tabs) return;
    tabs.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-lane]');
      if (!btn) return;
      api.setTab(btn.getAttribute('data-lane'));
    });
  }

  function wireBus() {
    if (S.busWired) return;
    S.busWired = true;
    GEW.bus.on('selection:change', function (d) {
      if (!d || d.source === 'swimlane') return;
      api.highlightNode(d.nodeId);
    });
    GEW.bus.on('step:focus', function (d) {
      if (!d) return;
      api.focusStep(d.stepId);
    });
  }

  // ---- public API -------------------------------------------------------
  var api = {
    init: function (opts) {
      opts = opts || {};
      S.container = opts.container || util.qs('gew-swimlane-body');
      S.laneModel = opts.laneModel || null;
      if (!S.container) return;
      resolveColors();
      if (P && P.createTaskColormap) {
        S.colormap = P.createTaskColormap(lightColormapOptions());
      }
      ensureCanvas();
      wireTabs();
      wireBus();

      // ResizeObserver → redraw on container resize
      if (w.ResizeObserver && !S.ro) {
        S.ro = new ResizeObserver(function () { render(); });
        S.ro.observe(S.container);
      } else {
        w.addEventListener('resize', function () { render(); });
      }

      setActiveTabButton(S.activeTab);
      render();
    },

    setTab: function (laneKind) {
      if (!laneKind) return;
      S.activeTab = laneKind;
      setActiveTabButton(laneKind);
      render();
    },

    highlightNode: function (nodeId) {
      S.highlightNodeId = nodeId || null;
      var ids = computeRelated(nodeId);
      // mark selected task = first related
      S.selectedTaskId = ids.length ? ids[0] : S.selectedTaskId;
      // switch to a lane that contains the node's tasks, if current lane lacks them
      var kind = laneKindForTasks(ids);
      if (kind && kind !== S.activeTab) {
        S.activeTab = kind;
        setActiveTabButton(kind);
      }
      // move playhead to the first related task start
      if (ids.length && S.laneModel) {
        var all = [];
        (S.laneModel.lanes || []).forEach(function (l) { (l.tasks || []).forEach(function (t) { all.push(t); }); });
        var first = all.filter(function (t) { return ids.indexOf(t.id) >= 0; })
          .sort(function (a, b) { return a.startUs - b.startUs; })[0];
        if (first) S.playheadUs = first.startUs;
      }
      render();
    },

    focusStep: function (stepId) {
      if (stepId == null) return;
      S.activeTab = 'step';
      setActiveTabButton('step');
      S.selectedStepId = Number(stepId);
      GEW.state.selectedStepId = S.selectedStepId;
      var steps = (S.laneModel && S.laneModel.steps) || [];
      var s = steps.filter(function (x) { return x.stepId === S.selectedStepId; })[0];
      if (s) S.playheadUs = s.startUs;
      render();
    },
  };

  GEW.swimlane = api;
})(window);
