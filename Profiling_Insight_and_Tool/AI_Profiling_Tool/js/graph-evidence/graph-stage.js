/* GEW.graphStage — model-graphviz render + pan/zoom/cluster/selection/filter.
   Renders via window.PtoModelGraphvizPattern (the pattern owns node geometry and
   the P0/P1/P2 priority overlays — we never draw our own borders). See CONTRACT.md §6,§8. */
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};

  var MIN_ZOOM = 0.15;
  var MAX_ZOOM = 3.0;
  var DIM_OPACITY = '0.72';

  var S = {
    container: null,
    svg: null,
    graph: null,
    problemMap: null,
    nodeEls: [],            // DOM node groups, index-aligned to graph.nodes
    nodeById: {},           // nodeId -> { el, node }
    edgeEls: [],            // { el, source, target }
    clusterEls: [],         // { el, cluster, toggle, icon }
    clusterChildren: {},
    collapsed: null,        // Set of collapsed cluster ids
    tx: 0, ty: 0, zoom: 1,
    pan: null,
    suppressClick: false,
    width: 1280,
    height: 1540,
    keyHandler: null,
    hoverPanel: null,
  };

  // --- cluster descendant resolution (port of getAllClusterNodes/getAllSubClusters) ---
  function allClusterNodes(cid) {
    var children = S.clusterChildren[cid] || [];
    var out = [];
    children.forEach(function (c) {
      if (S.clusterChildren[c]) out = out.concat(allClusterNodes(c));
      else out.push(c);
    });
    return out;
  }
  function allSubClusters(cid) {
    var children = S.clusterChildren[cid] || [];
    var out = [];
    children.forEach(function (c) {
      if (S.clusterChildren[c]) { out.push(c); out = out.concat(allSubClusters(c)); }
    });
    return out;
  }

  // --- transform ---
	  function applyTransform() {
	    if (!S.svg) return;
	    S.svg.style.width = S.width + 'px';
	    S.svg.style.height = S.height + 'px';
	    S.svg.style.transform =
	      'translate(' + S.tx + 'px, ' + S.ty + 'px) scale(' + S.zoom + ')';
	  }

	  function cssPxVar(name, fallback) {
	    var root = document.getElementById('gew-root');
	    if (!root) return fallback;
	    var raw = getComputedStyle(root).getPropertyValue(name);
	    var val = parseFloat(raw);
	    return isNaN(val) ? fallback : val;
	  }

	  function visibleViewport() {
	    var inspectorW = cssPxVar('--gew-inspector-w', 400);
	    var swimlaneH = cssPxVar('--gew-swimlane-h', 300);
	    var gap = cssPxVar('--gew-panel-gap', 12);
	    return {
	      width: Math.max(420, (S.container ? S.container.clientWidth : 1280) - inspectorW - gap * 3),
	      height: Math.max(320, (S.container ? S.container.clientHeight : 900) - swimlaneH - gap * 3),
	    };
	  }

  function esc(value) {
    return GEW.util && GEW.util.escapeHtml ? GEW.util.escapeHtml(value) : String(value == null ? '' : value);
  }

  function priorityClass(priority) {
    return String(priority || 'P3').toLowerCase();
  }

  function priorityTagHtml(priority) {
    if (!priority) return '';
    var p = String(priority).toUpperCase();
    return '<span class="gew-priority-tag ' + priorityClass(p) + '">' + esc(p) + '</span>';
  }

  function nodePriority(nodeId) {
    var data = GEW.state.data || {};
    var pm = data.problemMap && data.problemMap.problemNodes;
    if (pm && pm[nodeId] && pm[nodeId].priority) return pm[nodeId].priority;
    var issue = data.report && data.report.issues && data.report.issues[nodeId];
    if (issue && issue.diagnosis && issue.diagnosis.priority) return issue.diagnosis.priority;
    return null;
  }

  function ensureHoverPanel() {
    if (S.hoverPanel || !S.container) return;
    S.hoverPanel = document.createElement('div');
    S.hoverPanel.id = 'gew-node-hover';
    S.hoverPanel.className = 'gew-node-hover-panel';
    S.hoverPanel.setAttribute('aria-hidden', 'true');
    S.container.appendChild(S.hoverPanel);
  }

  function hoverHtml(node) {
    var data = GEW.state.data || {};
    var info = data.nodeInfo && data.nodeInfo[node.id];
    var issue = data.report && data.report.issues && data.report.issues[node.id];
    var priority = nodePriority(node.id);
    var chips = [];
    if (node.typeLabel) chips.push('<span class="gew-hover-meta-chip">' + esc(node.typeLabel) + '</span>');
    ((info && info.clusters) || []).slice(0, 3).forEach(function (c) {
      chips.push('<span class="gew-hover-meta-chip">' + esc(c) + '</span>');
    });

    var ioRows = [];
    function io(label, arr, key) {
      (arr || []).slice(0, 2).forEach(function (item) {
        ioRows.push('<span class="gew-hover-io-label">' + esc(label) + '</span><span>'
          + esc(item[key] || '') + (item.desc ? ' · ' + esc(item.desc) : '') + '</span>');
      });
    }
    if (info) {
      io('输入', info.inputs, 'from');
      io('输出', info.outputs, 'to');
    }

    var diagnosis = issue && issue.diagnosis
      ? '<div class="gew-hover-body"><strong>' + esc(issue.diagnosis.title || '诊断') + '</strong>'
        + (issue.diagnosis.summary ? '<br>' + esc(issue.diagnosis.summary) : '') + '</div>'
      : '';

    return [
      '<div class="gew-hover-title">',
        priorityTagHtml(priority),
        '<div class="gew-hover-title-main">',
          '<div class="gew-hover-kicker">' + esc(node.id) + (info && info.idEn ? ' · ' + esc(info.idEn) : '') + '</div>',
          '<div class="gew-hover-name">' + esc(node.label || node.id) + '</div>',
        '</div>',
      '</div>',
      chips.length ? '<div class="gew-hover-chip-row">' + chips.join('') + '</div>' : '',
      info && info.what ? '<div class="gew-hover-body">' + esc(info.what) + '</div>' : '',
      ioRows.length ? '<div class="gew-hover-io">' + ioRows.join('') + '</div>' : '',
      info && info.params ? '<div class="gew-hover-body"><strong>参数</strong> · ' + esc(info.params) + '</div>' : '',
      diagnosis,
    ].filter(Boolean).join('');
  }

  function placeHoverPanel(e) {
    if (!S.hoverPanel || !S.container) return;
    var rect = S.container.getBoundingClientRect();
    var x = e.clientX - rect.left + 16;
    var y = e.clientY - rect.top + 16;
    var panelW = S.hoverPanel.offsetWidth || 360;
    var panelH = S.hoverPanel.offsetHeight || 160;
    x = Math.max(10, Math.min(rect.width - panelW - 10, x));
    y = Math.max(10, Math.min(rect.height - panelH - 10, y));
    S.hoverPanel.style.left = x + 'px';
    S.hoverPanel.style.top = y + 'px';
  }

  function showNodeHover(node, e) {
    ensureHoverPanel();
    if (!S.hoverPanel) return;
    S.hoverPanel.innerHTML = hoverHtml(node);
    S.hoverPanel.classList.add('is-visible');
    S.hoverPanel.setAttribute('aria-hidden', 'false');
    placeHoverPanel(e);
  }

  function moveNodeHover(e) {
    if (!S.hoverPanel || !S.hoverPanel.classList.contains('is-visible')) return;
    placeHoverPanel(e);
  }

  function hideNodeHover() {
    if (!S.hoverPanel) return;
    S.hoverPanel.classList.remove('is-visible');
    S.hoverPanel.setAttribute('aria-hidden', 'true');
  }

  function parseColor(raw) {
    var value = String(raw || '').trim();
    var m;
    if (!value) return null;
    if (value[0] === '#') {
      var hex = value.slice(1);
      if (hex.length === 3) hex = hex.split('').map(function (c) { return c + c; }).join('');
      if (hex.length !== 6) return null;
      var n = parseInt(hex, 16);
      if (isNaN(n)) return null;
      return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
    }
    m = value.match(/rgba?\(([^)]+)\)/);
    if (m) {
      var parts = m[1].split(',').map(function (p) { return parseFloat(p); });
      if (parts.length >= 3) return { r: parts[0], g: parts[1], b: parts[2] };
    }
    return null;
  }

  function luminance(rgb) {
    if (!rgb) return 1;
    function channel(v) {
      v = v / 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    }
    return 0.2126 * channel(rgb.r) + 0.7152 * channel(rgb.g) + 0.0722 * channel(rgb.b);
  }

  function applyNodeTextContrast() {
    S.nodeEls.forEach(function (el) {
      var rect = el.querySelector('rect');
      var rgb = rect && parseColor(rect.getAttribute('fill') || getComputedStyle(rect).fill);
      var dark = luminance(rgb) < 0.34;
      var label = el.querySelector('.pto-model-graphviz-node-label');
      var type = el.querySelector('.pto-model-graphviz-node-type');
      if (label) {
        label.setAttribute('y', '0');
        label.setAttribute('dominant-baseline', 'middle');
        label.style.fill = dark ? 'rgba(255,255,255,0.96)' : '#111827';
        label.setAttribute('fill', dark ? 'rgba(255,255,255,0.96)' : '#111827');
        label.setAttribute('stroke', dark ? 'rgba(0,0,0,0.18)' : 'rgba(255,255,255,0.34)');
        label.setAttribute('stroke-width', dark ? '0.55' : '0.45');
        label.style.paintOrder = 'stroke';
      }
      if (type) {
        type.style.display = 'none';
        type.setAttribute('aria-hidden', 'true');
      }
    });
  }
	
	  function fitZoom() {
	    if (!S.container) return 1;
	    var vp = visibleViewport();
	    var cw = Math.max(400, vp.width - 96);
	    var ch = Math.max(320, vp.height - 96);
	    return Math.min(0.88, Math.min(cw / S.width, ch / S.height));
	  }
	
	  function centerView() {
	    if (!S.container) return;
	    var vp = visibleViewport();
	    S.tx = Math.max(8, (vp.width - S.width * S.zoom) / 2);
	    S.ty = Math.max(8, (vp.height - S.height * S.zoom) / 2);
	    applyTransform();
	  }

  function fit() {
    S.zoom = fitZoom();
    centerView();
  }

  // --- selection + dim ---
  function isPriorityVisible(priority) {
    var f = (GEW.state && GEW.state.activeFilter) || 'all';
    if (f === 'off') return false;
    if (f === 'all') return true;
    return String(priority || '').toUpperCase() === f;
  }

  // Toggle the pattern-drawn priority overlays per the active filter.
  function applyFilterOverlays() {
    // Per-node badges (rect + text drawn inside each node group).
    S.nodeEls.forEach(function (el, i) {
      var node = S.graph.nodes[i];
      if (!node) return;
      var badge = el.querySelector('.pto-model-graphviz-report-node-badge');
      var badgeText = el.querySelector('.pto-model-graphviz-report-node-badge-text');
      var on = node.reportPriority && isPriorityVisible(node.reportPriority);
      if (badge) badge.style.display = on ? '' : 'none';
      if (badgeText) badgeText.style.display = on ? '' : 'none';
      ['p0', 'p1', 'p2', 'p3'].forEach(function (p) { el.classList.remove('is-problem-' + p); });
      if (on) {
        el.classList.add('is-problem-' + String(node.reportPriority).toLowerCase());
      }
    });
    // Cluster title pills (separate top-level groups).
    var pills = S.svg.querySelectorAll('.pto-model-graphviz-cluster-title-pill');
    pills.forEach(function (g) {
      var pri = g.getAttribute('data-report-priority');
      g.style.display = isPriorityVisible(pri) ? '' : 'none';
    });
  }

  function clearSelectionDim() {
    S.nodeEls.forEach(function (el) {
      el.style.opacity = '';
      el.classList.remove('is-graph-selected');
    });
    S.edgeEls.forEach(function (e) { e.el.style.opacity = ''; });
  }

  // highlight nodeId + dim unrelated nodes/edges (neighbors stay lit)
  function applySelection(nodeId) {
    if (!nodeId || !S.nodeById[nodeId]) { clearSelectionDim(); return; }
    var related = {};
    related[nodeId] = true;
    S.graph.edges.forEach(function (edge) {
      if (edge.source === nodeId) related[edge.target] = true;
      if (edge.target === nodeId) related[edge.source] = true;
    });
    S.nodeEls.forEach(function (el, i) {
      var node = S.graph.nodes[i];
      if (!node) return;
      var on = !!related[node.id];
      el.classList.toggle('is-graph-selected', node.id === nodeId);
      el.style.opacity = on ? '' : DIM_OPACITY;
    });
    S.edgeEls.forEach(function (e) {
      var touches = e.source === nodeId || e.target === nodeId;
      e.el.style.opacity = touches ? '' : DIM_OPACITY;
    });
  }

  // --- cluster collapse/expand (port of updateClusterVisibility) ---
  function updateClusterVisibility() {
    var hiddenNodes = {};
    var hiddenClusters = {};
    S.collapsed.forEach(function (cid) {
      allClusterNodes(cid).forEach(function (nid) { hiddenNodes[nid] = true; });
      allSubClusters(cid).forEach(function (sid) { hiddenClusters[sid] = true; });
    });
    S.nodeEls.forEach(function (el, i) {
      var node = S.graph.nodes[i];
      if (node) el.style.display = hiddenNodes[node.id] ? 'none' : '';
    });
    S.edgeEls.forEach(function (e) {
      e.el.style.display = (hiddenNodes[e.source] || hiddenNodes[e.target]) ? 'none' : '';
    });
    S.clusterEls.forEach(function (c) {
      c.el.style.display = hiddenClusters[c.cluster.id] ? 'none' : '';
    });
  }

  // --- node focus (pan/zoom node to center, smooth) ---
	  function focusNode(nodeId) {
	    var entry = S.nodeById[nodeId];
	    if (!entry || !S.container) return;
	    var node = entry.node;
	    var prev = S.svg.style.transition;
	    S.svg.style.transition = 'transform 0.32s ease';
	    var vp = visibleViewport();
	    S.tx = vp.width / 2 - node.x * S.zoom;
	    S.ty = vp.height / 2 - node.y * S.zoom;
	    applyTransform();
    w.setTimeout(function () { if (S.svg) S.svg.style.transition = prev || ''; }, 360);
  }

  // --- pan/zoom wiring (port of setupGraphTabPanZoom) ---
  function wirePanZoom() {
    var stage = S.container;

    stage.addEventListener('wheel', function (e) {
      if (!e.metaKey && !e.ctrlKey) return;
      e.preventDefault();
      var rect = stage.getBoundingClientRect();
      var px = e.clientX - rect.left;
      var py = e.clientY - rect.top;
      var factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      var z0 = S.zoom;
      var z1 = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z0 * factor));
      S.tx = px - (px - S.tx) * (z1 / z0);
      S.ty = py - (py - S.ty) * (z1 / z0);
      S.zoom = z1;
      applyTransform();
    }, { passive: false });

    stage.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      if (e.target.closest('.pto-model-graphviz-node, .pto-model-graphviz-toggle')) return;
      S.suppressClick = false;
      S.pan = { id: e.pointerId, x: e.clientX, y: e.clientY, tx: S.tx, ty: S.ty, moved: false };
    });

    stage.addEventListener('pointermove', function (e) {
      if (!S.pan || S.pan.id !== e.pointerId) return;
      var dx = e.clientX - S.pan.x;
      var dy = e.clientY - S.pan.y;
      if (!S.pan.moved) {
        if (Math.hypot(dx, dy) < 4) return;
        S.pan.moved = true;
        stage.classList.add('is-panning');
        try { stage.setPointerCapture(e.pointerId); } catch (_) {}
      }
      S.tx = S.pan.tx + dx;
      S.ty = S.pan.ty + dy;
      applyTransform();
      e.preventDefault();
    });

    function endPan(e) {
      if (!S.pan || S.pan.id !== e.pointerId) return;
      if (S.pan.moved) S.suppressClick = true;
      S.pan = null;
      stage.classList.remove('is-panning');
      if (stage.hasPointerCapture && stage.hasPointerCapture(e.pointerId)) {
        stage.releasePointerCapture(e.pointerId);
      }
    }
    stage.addEventListener('pointerup', endPan);
    stage.addEventListener('pointercancel', endPan);
    stage.addEventListener('lostpointercapture', endPan);

    // Ctrl/Cmd+0 reset
    S.keyHandler = function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key === '0') {
        if (!stage.offsetParent && stage.offsetWidth === 0) return;
        e.preventDefault();
        fit();
      }
    };
    w.addEventListener('keydown', S.keyHandler);
  }

  // --- toolbar buttons (#gew-graph-toolbar [data-graph-action]) ---
  function wireToolbar() {
    var bar = document.getElementById('gew-graph-toolbar');
    if (!bar) return;
    bar.addEventListener('click', function (e) {
      var btn = e.target.closest('[data-graph-action]');
      if (!btn) return;
      var action = btn.getAttribute('data-graph-action');
      if (action === 'fit') { fit(); return; }
      if (action === 'zoom-in' || action === 'zoom-out') {
        var rect = S.container.getBoundingClientRect();
        var px = rect.width / 2;
        var py = rect.height / 2;
        var factor = action === 'zoom-in' ? 1.2 : 1 / 1.2;
        var z0 = S.zoom;
        var z1 = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z0 * factor));
        S.tx = px - (px - S.tx) * (z1 / z0);
        S.ty = py - (py - S.ty) * (z1 / z0);
        S.zoom = z1;
        applyTransform();
      }
    });
  }

  // --- bus subscriptions ---
  var busWired = false;
  function wireBus() {
    if (busWired) return;
    busWired = true;
    GEW.bus.on('selection:change', function (d) {
      if (!d || d.source === 'graph') return;
      api.select(d.nodeId);
      if (d.source === 'mapped' || d.source === 'swimlane' || d.source === 'action' || d.source === 'reportIssue') {
        api.focusNode(d.nodeId);
      }
    });
    GEW.bus.on('filter:change', function (d) {
      if (!d) return;
      api.setFilter(d.priority);
    });
  }

  var api = {
    init: function (opts) {
      opts = opts || {};
      var container = opts.container;
      var graph = opts.graph;
      var problemMap = opts.problemMap;
      // Graceful no-op when there's no graph/container (app handles #gew-graph-empty).
      if (!container || !graph || !graph.nodes || !graph.nodes.length) return;
      if (!w.PtoModelGraphvizPattern) {
        console.error('[GEW.graphStage] PtoModelGraphvizPattern not loaded');
        return;
      }

      S.container = container;
      S.graph = graph;
      S.problemMap = problemMap || null;
      S.clusterChildren = graph.clusterChildren || {};
      S.width = graph.width || 1280;
      S.height = graph.height || 1540;

      // Merge problemMap priority → node.reportPriority so the pattern draws overlays.
      var pNodes = (problemMap && problemMap.problemNodes) || {};
      graph.nodes.forEach(function (n) {
        if (pNodes[n.id] && pNodes[n.id].priority) n.reportPriority = pNodes[n.id].priority;
      });

      var svg = w.PtoModelGraphvizPattern.render(container, graph, {
        ariaLabel: 'Qwen2-7B 模型计算图',
        width: S.width,
        height: S.height,
        colormap: {
          saturation: 0.58,
          lightness: 0.72,
        },
      });
      if (!svg) return;
      S.svg = svg;

      // Index node groups by their array order (pattern draws nodes in graph.nodes order).
      S.nodeEls = Array.prototype.slice.call(
        container.querySelectorAll('.pto-model-graphviz-node'));
      S.nodeById = {};
      S.nodeEls.forEach(function (el, i) {
        var node = graph.nodes[i];
        if (!node) return;
        el.dataset.nodeId = node.id;       // pattern does not expose id; set it ourselves
        el.style.cursor = 'pointer';
        if (node.reportPriority) {
          el.classList.add('is-problem-' + String(node.reportPriority).toLowerCase());
        }
        S.nodeById[node.id] = { el: el, node: node };
        el.addEventListener('pointerenter', function (e) { showNodeHover(node, e); });
        el.addEventListener('pointermove', moveNodeHover);
        el.addEventListener('pointerleave', hideNodeHover);
        el.addEventListener('click', function () {
          if (S.suppressClick) { S.suppressClick = false; return; }
          hideNodeHover();
          GEW.state.selectedNodeId = node.id;
          applySelection(node.id);
          GEW.bus.emit('selection:change', { nodeId: node.id, source: 'graph' });
        });
      });
      applyNodeTextContrast();

      // Map edge DOM (pattern dedupes duplicate source->target before drawing).
      var edgeDom = Array.prototype.slice.call(
        container.querySelectorAll('.pto-model-graphviz-edge'));
      var seen = {};
      var visibleEdges = [];
      var nodeIdSet = {};
      graph.nodes.forEach(function (n) { nodeIdSet[n.id] = true; });
      (graph.edges || []).forEach(function (edge) {
        if (!nodeIdSet[edge.source] || !nodeIdSet[edge.target]) return;
        var key = edge.source + '->' + edge.target;
        if (seen[key]) return;
        seen[key] = true;
        visibleEdges.push(edge);
      });
      S.edgeEls = edgeDom.map(function (el, i) {
        var edge = visibleEdges[i] || {};
        return { el: el, source: edge.source, target: edge.target };
      });

      // Cluster groups + collapse toggles.
      S.clusterEls = Array.prototype.slice.call(
        container.querySelectorAll('.pto-model-graphviz-cluster')).map(function (el, i) {
        var cluster = (graph.clusters || [])[i];
        return {
          el: el,
          cluster: cluster || { id: '__c' + i },
          toggle: el.querySelector('.pto-model-graphviz-toggle'),
          icon: el.querySelector('.pto-model-graphviz-toggle-icon'),
        };
      });
      S.collapsed = new Set();
      S.clusterEls.forEach(function (c) {
        if (!c.cluster || !S.clusterChildren[c.cluster.id] || !c.toggle) return;
        c.toggle.addEventListener('click', function (e) {
          e.stopPropagation();
          // preserve viewport anchor + selection (visibility only; no recenter)
          var id = c.cluster.id;
          if (S.collapsed.has(id)) {
            S.collapsed.delete(id);
            if (c.icon) c.icon.textContent = '-';
          } else {
            S.collapsed.add(id);
            if (c.icon) c.icon.textContent = '+';
          }
          updateClusterVisibility();
          if (GEW.state.selectedNodeId) applySelection(GEW.state.selectedNodeId);
        });
      });

      wirePanZoom();
      wireToolbar();
      wireBus();

      fit();
      applyFilterOverlays();
      if (GEW.state.selectedNodeId && S.nodeById[GEW.state.selectedNodeId]) {
        applySelection(GEW.state.selectedNodeId);
      }
    },

    // inbound sync path — highlight only, no event emit
    select: function (nodeId) {
      if (!S.svg) return;
      GEW.state.selectedNodeId = nodeId || null;
      applySelection(nodeId);
    },

    focusNode: function (nodeId) {
      if (!S.svg) return;
      focusNode(nodeId);
    },

    setFilter: function (priority) {
      if (priority) GEW.state.activeFilter = priority;
      if (!S.svg) return;
      applyFilterOverlays();
    },

    fit: function () {
      if (!S.svg) return;
      fit();
    },
  };

  GEW.graphStage = api;
})(window);
