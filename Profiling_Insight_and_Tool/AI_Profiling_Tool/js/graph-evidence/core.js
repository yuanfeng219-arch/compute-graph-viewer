/* Graph Evidence Workbench — core runtime (loaded first).
   Defines window.GEW: namespace, event bus, shared state, utilities.
   See CONTRACT.md. Do not put business data here. */
(function (w) {
  'use strict';

  function SchemaError(message, detail) {
    this.name = 'SchemaError';
    this.message = message;
    this.detail = detail || null;
  }
  SchemaError.prototype = Object.create(Error.prototype);

  // --- event bus ---
  function createBus() {
    const map = new Map();
    return {
      on(type, fn) {
        if (!map.has(type)) map.set(type, new Set());
        map.get(type).add(fn);
        return () => map.get(type) && map.get(type).delete(fn);
      },
      emit(type, detail) {
        const set = map.get(type);
        if (!set) return;
        set.forEach((fn) => {
          try { fn(detail || {}); }
          catch (e) { console.error('[GEW.bus]', type, e); }
        });
      },
    };
  }

  // --- utilities ---
  const util = {
    qs(id) { return document.getElementById(id); },
    el(tag, attrs, children) {
      const node = document.createElement(tag);
      if (attrs) Object.keys(attrs).forEach((k) => {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k.slice(0, 2) === 'on' && typeof attrs[k] === 'function') node.addEventListener(k.slice(2), attrs[k]);
        else if (k === 'dataset') Object.assign(node.dataset, attrs[k]);
        else if (attrs[k] != null) node.setAttribute(k, attrs[k]);
      });
      (Array.isArray(children) ? children : children != null ? [children] : []).forEach((c) => {
        if (c == null) return;
        node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      });
      return node;
    },
    escapeHtml(s) {
      return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
      ));
    },
    fmtUs(n) {
      if (n == null || isNaN(n)) return '—';
      const v = Number(n);
      if (Math.abs(v) >= 1000) return (v / 1000).toFixed(2) + ' ms';
      return v.toFixed(1) + ' µs';
    },
    getParams() {
      const p = new URLSearchParams(w.location.search);
      return {
        reportId: p.get('reportId') || null,
        nodeId: p.get('nodeId') || null,
        actionId: p.get('actionId') || null,
        issueRef: p.get('issueRef') || null,
        priority: p.get('priority') || null,
        stepId: p.get('stepId') != null ? p.get('stepId') : null,
      };
    },
    setParams(obj) {
      const p = new URLSearchParams(w.location.search);
      Object.keys(obj).forEach((k) => {
        if (obj[k] == null || obj[k] === '') p.delete(k);
        else p.set(k, obj[k]);
      });
      const qs = p.toString();
      w.history.replaceState(null, '', qs ? '?' + qs : w.location.pathname);
    },
  };

  w.GEW = {
    SCHEMA_VERSION: '0.1',
    MODEL_ID: 'qwen2-7b',
    SchemaError,
    bus: createBus(),
    util,
    state: {
      reportId: null,
      selectedNodeId: null,
      selectedActionId: null,
      selectedReportIssueRef: null,
      activeFilter: 'all',
      selectedStepId: null,
      data: null,
    },
    // module singletons fill these in:
    loader: null,
    traceParser: null,
    graphStage: null,
    inspector: null,
    swimlane: null,
    app: null,
  };
})(window);
