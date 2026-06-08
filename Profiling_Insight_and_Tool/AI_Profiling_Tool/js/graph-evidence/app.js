/* GEW.app — orchestrator: boot, load, init stages, wire bus, URL params, export/copy.
   See CONTRACT.md §6/§7. Keeps GEW.state + URL in sync; stages react to the bus
   themselves and ignore their own source. */
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};
  var GEW = w.GEW;
  var util = GEW.util;
  var qs = util.qs;

  function veilError(msg) {
    var veil = qs('gew-loading');
    if (veil) { veil.classList.remove('is-done'); veil.textContent = msg; }
  }
  function veilDone() {
    var veil = qs('gew-loading');
    if (veil) veil.classList.add('is-done');
  }

  // header filter wiring
  function wireFilter() {
    var bar = qs('gew-filter');
    if (!bar) return;
    bar.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-priority]');
      if (!btn) return;
      applyFilter(btn.getAttribute('data-priority'));
    });
  }
	  function applyFilter(priority) {
	    var bar = qs('gew-filter');
	    if (bar) {
	      Array.prototype.forEach.call(bar.querySelectorAll('button[data-priority]'), function (b) {
	        var on = b.getAttribute('data-priority') === priority;
	        b.classList.toggle('is-selected', on);
	        b.classList.toggle('is-active', on);
	      });
	    }
	    GEW.state.activeFilter = priority;
	    util.setParams({ priority: priority === 'all' ? null : priority });
	    GEW.bus.emit('filter:change', { priority: priority });
  }

  // current inspector evidence (issue) for the selected node
  function currentIssue() {
    var data = GEW.state.data;
    var nid = GEW.state.selectedNodeId;
    if (!data || !nid) return null;
    var iss = data.report && data.report.issues && data.report.issues[nid];
    return iss || null;
  }

  function actionForId(actionId) {
    var data = GEW.state.data || {};
    var list = (data.report && data.report.actions) || [];
    var id = String(actionId || '');
    return list.filter(function (a) { return String(a.id) === id; })[0] || null;
  }

  function actionIssueRefs(action) {
    var refs = [];
    function add(ref) {
      if (ref == null) return;
      var key = String(ref);
      if (refs.indexOf(key) < 0) refs.push(key);
    }
    add(action && action.reportIssueRef);
    ((action && action.reportIssueRefs) || []).forEach(add);
    add(action && action.issueRef);
    return refs;
  }

  function issueRefForNode(nodeId) {
    if (!nodeId) return null;
    var data = GEW.state.data || {};
    var issue = data.report && data.report.issues && data.report.issues[nodeId];
    return issue && issue.reportIssueRef ? String(issue.reportIssueRef) : null;
  }

  function actionIdForIssueRef(issueRef) {
    if (!issueRef) return null;
    var data = GEW.state.data || {};
    var list = (data.report && data.report.actions) || [];
    var ref = String(issueRef);
    var action = list.filter(function (a) {
      return actionIssueRefs(a).indexOf(ref) >= 0;
    })[0];
    return action ? String(action.id) : null;
  }

  function nodeIdForIssueRef(issueRef) {
    if (!issueRef) return null;
    var data = GEW.state.data || {};
    var ref = String(issueRef);
    var issues = (data.report && data.report.issues) || {};
    var exact = Object.keys(issues).filter(function (nodeId) {
      return String((issues[nodeId] || {}).reportIssueRef || '') === ref;
    })[0];
    if (exact) return exact;
    var action = actionForId(actionIdForIssueRef(ref));
    return action && (action.nodeId || ((action.mappedNodes || [])[0])) || null;
  }

  function actionIdForNode(nodeId) {
    if (!nodeId) return null;
    var data = GEW.state.data || {};
    var list = (data.report && data.report.actions) || [];
    var action = list.filter(function (a) {
      return (a.mappedNodes || []).indexOf(nodeId) >= 0;
    })[0];
    return action ? String(action.id) : null;
  }

  function evidenceSummaryText() {
    var data = GEW.state.data || {};
    var nid = GEW.state.selectedNodeId;
    var iss = currentIssue();
    var lines = [];
    lines.push('# Profiling 证据摘要');
    lines.push('Report: ' + (data.meta ? data.meta.reportId : '—'));
    lines.push('Node: ' + (nid || '—'));
    if (iss && iss.diagnosis) {
      var d = iss.diagnosis;
      lines.push('Priority: ' + (d.priority || '—') + (d.dimension ? ' · ' + d.dimension : ''));
      if (d.title) lines.push('Title: ' + d.title);
      if (d.summary) lines.push('Summary: ' + d.summary);
      if (Array.isArray(d.metrics)) d.metrics.forEach(function (m) { lines.push('  · ' + m[0] + ': ' + m[1]); });
    }
    if (iss && Array.isArray(iss.evidence)) {
      lines.push('Evidence:');
      iss.evidence.forEach(function (e) {
        lines.push('  - [' + (e.confidence || '?') + '] ' + e.text + ' (' + e.sourceFile + (e.sourceField ? '/' + e.sourceField : '') + ')');
      });
    }
    if (iss && Array.isArray(iss.actions)) {
      lines.push('Actions:');
      iss.actions.forEach(function (a) { lines.push('  - ' + a.text + (a.inferred ? ' [inferred]' : '')); });
    }
    if (!iss) lines.push('（该节点无诊断问题。）');
    return lines.join('\n');
  }

  function wireHome() {
    var btn = qs('gew-home');
    if (!btn) return;
    btn.addEventListener('click', function () {
      if (w.history.length > 1) w.history.back();
      else w.location.href = '../../launch.html';
    });
  }

  function wireExport() {
    var btn = qs('gew-export');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var data = GEW.state.data || {};
      var iss = currentIssue();
      var snapshot = {
        schemaVersion: GEW.SCHEMA_VERSION,
        reportId: data.meta ? data.meta.reportId : (GEW.state.reportId || null),
        generatedAt: new Date().toISOString(),
        selectedNodeId: GEW.state.selectedNodeId || null,
        activeFilter: GEW.state.activeFilter,
        selectedStepId: GEW.state.selectedStepId != null ? GEW.state.selectedStepId : null,
        evidence: iss || null,
        sourceFiles: data.meta ? data.meta.sourceFiles : [],
      };
      var blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
      var url = URL.createObjectURL(blob);
      var a = util.el('a', {
        href: url,
        download: 'gew-snapshot-' + (snapshot.reportId || 'report') + '.json',
      });
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 0);
    });
  }

	  function wireCopy() {
    var btn = qs('gew-copy');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var text = evidenceSummaryText();
      var done = function () {
        var old = btn.textContent;
        btn.textContent = '已复制';
        setTimeout(function () { btn.textContent = old; }, 1400);
      };
      if (w.navigator && navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
      } else {
        fallbackCopy(text, done);
      }
    });
  }

	  function wireSwimlaneResize() {
	    var root = qs('gew-root');
	    var grip = qs('gew-swimlane-resize');
	    if (!root || !grip) return;
	    var saved = Number(w.localStorage && localStorage.getItem('gew.swimlaneHeight'));
	    if (saved && saved >= 160 && saved <= 560) {
	      root.style.setProperty('--gew-swimlane-h', saved + 'px');
	    }
	    grip.addEventListener('pointerdown', function (e) {
	      if (e.button !== 0) return;
	      e.preventDefault();
	      var startY = e.clientY;
	      var current = parseFloat(getComputedStyle(root).getPropertyValue('--gew-swimlane-h')) || 300;
	      function move(ev) {
	        var next = Math.max(160, Math.min(560, current + (startY - ev.clientY)));
	        root.style.setProperty('--gew-swimlane-h', next + 'px');
	        w.dispatchEvent(new Event('resize'));
	      }
	      function up() {
	        var finalH = parseFloat(getComputedStyle(root).getPropertyValue('--gew-swimlane-h')) || current;
	        if (w.localStorage) localStorage.setItem('gew.swimlaneHeight', String(Math.round(finalH)));
	        w.removeEventListener('pointermove', move);
	        w.removeEventListener('pointerup', up);
	        w.removeEventListener('pointercancel', up);
	      }
	      w.addEventListener('pointermove', move);
	      w.addEventListener('pointerup', up);
	      w.addEventListener('pointercancel', up);
	    });
	  }

  function fallbackCopy(text, done) {
    var ta = util.el('textarea', { style: 'position:fixed;left:-9999px;top:0;' });
    ta.value = text;
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try { document.execCommand('copy'); done && done(); } catch (e) { /* ignore */ }
    document.body.removeChild(ta);
  }

  // central selection sync (state + URL). Stages handle their own bus reactions.
  function wireSelectionSync() {
    GEW.bus.on('selection:change', function (d) {
      if (!d) return;
      GEW.state.selectedNodeId = d.nodeId || null;
      util.setParams({ nodeId: d.nodeId || null });
      GEW.state.selectedReportIssueRef = d.reportIssueRef || issueRefForNode(d.nodeId);
      util.setParams({ issueRef: GEW.state.selectedReportIssueRef });
      if (d.source !== 'action') {
        GEW.state.selectedActionId = d.actionId != null ? String(d.actionId) : actionIdForNode(d.nodeId);
        util.setParams({ actionId: GEW.state.selectedActionId });
      }
    });
    GEW.bus.on('step:focus', function (d) {
      if (!d) return;
      GEW.state.selectedStepId = d.stepId != null ? Number(d.stepId) : null;
      util.setParams({ stepId: d.stepId != null ? d.stepId : null });
    });
  }

  function wireActionSync() {
    GEW.bus.on('action:select', function (d) {
      if (!d) return;
      GEW.state.selectedActionId = d.actionId != null ? String(d.actionId) : null;
      util.setParams({ actionId: GEW.state.selectedActionId });
      var action = actionForId(GEW.state.selectedActionId);
      var refs = actionIssueRefs(action);
      GEW.state.selectedReportIssueRef = d.reportIssueRef || refs[0] || null;
      util.setParams({ issueRef: GEW.state.selectedReportIssueRef });
      if (d.nodeId) {
        GEW.state.selectedNodeId = d.nodeId;
        GEW.bus.emit('selection:change', {
          nodeId: d.nodeId,
          actionId: GEW.state.selectedActionId,
          reportIssueRef: GEW.state.selectedReportIssueRef,
          mappedNodes: d.mappedNodes || [d.nodeId],
          source: 'action',
        });
      }
    });
  }

  function wireReportIssueSync() {
    GEW.bus.on('report-issue:select', function (d) {
      if (!d || d.reportIssueRef == null) return;
      var ref = String(d.reportIssueRef);
      var actionId = d.actionId != null ? String(d.actionId) : actionIdForIssueRef(ref);
      var nodeId = d.nodeId || nodeIdForIssueRef(ref);
      GEW.state.selectedReportIssueRef = ref;
      GEW.state.selectedActionId = actionId;
      util.setParams({ issueRef: ref, actionId: actionId });
      if (nodeId) {
        GEW.state.selectedNodeId = nodeId;
        GEW.bus.emit('selection:change', {
          nodeId: nodeId,
          actionId: actionId,
          reportIssueRef: ref,
          source: 'reportIssue',
        });
      }
    });
  }

  GEW.app = {
    async start() {
      try {
        var params = util.getParams();

        var data;
        try {
          data = await GEW.loader.load(params.reportId || GEW.MODEL_ID);
        } catch (err) {
          if (err && err.name === 'SchemaError') {
            veilError('数据加载失败：' + err.message);
            // if it's a graph/empty problem, reveal the graph empty-state too
            var ge = qs('gew-graph-empty');
            if (ge) ge.classList.add('is-visible');
            console.error('[GEW.app] load failed', err);
            return;
          }
          throw err;
        }

        GEW.state.data = data;
        GEW.state.reportId = data.meta.reportId;

        // header text
        var nameEl = qs('gew-report-name');
        if (nameEl) nameEl.textContent = 'Qwen2-7B · ' + data.meta.reportId;
	        var hasGraph = data.graph && data.graph.nodes && data.graph.nodes.length;

        // init stages
        GEW.inspector.init({ container: qs('gew-inspector-body') });
        if (hasGraph) {
          GEW.graphStage.init({
            container: qs('gew-graph-stage'),
            graph: data.graph,
            problemMap: data.problemMap,
          });
        } else {
          var ge2 = qs('gew-graph-empty');
          if (ge2) ge2.classList.add('is-visible');
        }
        GEW.swimlane.init({ container: qs('gew-swimlane-body'), laneModel: data.laneModel });

        // wiring
        wireFilter();
	        wireHome();
	        wireExport();
	        wireCopy();
	        wireSwimlaneResize();
	        wireSelectionSync();
	        wireActionSync();
	        wireReportIssueSync();

        // success: drop the veil
        veilDone();

        // emit report:loaded for any late subscribers
        GEW.bus.emit('report:loaded', { reportId: data.meta.reportId, data: data });

        // apply initial URL state
        if (params.priority) {
          applyFilter(params.priority);
        }

        var initialNode = params.nodeId || null;
	        if (initialNode) {
	          GEW.state.selectedNodeId = initialNode;
	          GEW.bus.emit('selection:change', { nodeId: initialNode, source: params.nodeId ? 'url' : 'init' });
	        }

        if (params.actionId) {
          var initialAction = actionForId(params.actionId);
          if (initialAction) {
            GEW.state.selectedActionId = String(initialAction.id);
            GEW.bus.emit('action:select', {
              actionId: String(initialAction.id),
              nodeId: initialAction.nodeId || ((initialAction.mappedNodes || [])[0]),
              mappedNodes: initialAction.mappedNodes || [],
              source: 'url',
            });
          }
        }

        if (params.issueRef) {
          GEW.state.selectedReportIssueRef = String(params.issueRef);
          GEW.bus.emit('report-issue:select', { reportIssueRef: String(params.issueRef), source: 'url' });
        }

        if (!params.nodeId && !params.actionId && !params.issueRef) {
          GEW.inspector.clear();
        }

        if (params.stepId != null) {
          GEW.bus.emit('step:focus', { stepId: Number(params.stepId) });
        }
      } catch (e) {
        console.error('[GEW.app] start failed', e);
        veilError('初始化失败：' + (e && e.message ? e.message : e));
      }
    },
  };
})(window);
