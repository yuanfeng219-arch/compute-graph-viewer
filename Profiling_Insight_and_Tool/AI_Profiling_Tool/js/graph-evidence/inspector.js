/* GEW.inspector — right-panel diagnosis closure (CONTRACT §6).
   Renders the 7 evidence sections from GEW.state.data.report.issues[nodeId],
   or a node-info-only view for non-problem nodes. PTO-classes only; no new
   badge/card styles, no border-left rails, no card-in-card nesting. */
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};
  var GEW = w.GEW;
  var el = GEW.util.el;
  var esc = GEW.util.escapeHtml;

  var container = null;

  // --- shared inline-style tokens (no new CSS classes) ---
  var S = {
    section: 'display:flex;flex-direction:column;gap:var(--space-2);'
      + 'padding:var(--space-3) 0 0;margin-top:var(--space-1);',
    sectionFirst: 'display:flex;flex-direction:column;gap:var(--space-2);padding-bottom:var(--space-3);',
    head: 'font:var(--text-label);text-transform:uppercase;letter-spacing:0.06em;'
      + 'color:var(--foreground-secondary);',
    row: 'display:flex;gap:var(--space-3);align-items:baseline;justify-content:space-between;',
    rowKey: 'font:var(--text-body);color:var(--foreground-secondary);flex:0 1 auto;min-width:0;',
    rowVal: 'font:var(--text-mono);color:var(--foreground);text-align:right;flex:0 0 auto;white-space:nowrap;',
    body: 'font:var(--text-body);color:var(--foreground);',
    muted: 'font:var(--text-body);color:var(--foreground-secondary);',
    listItem: 'display:flex;flex-direction:column;gap:var(--space-1);'
      + 'padding:var(--space-2);border-radius:var(--radius-md);'
      + 'background:var(--gew-panel-fill,var(--surface-1));',
    metaLine: 'display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap;'
      + 'font:var(--text-label);color:var(--foreground-secondary);',
    chipRow: 'display:flex;gap:var(--space-2);align-items:center;flex-wrap:wrap;',
  };

  // priority → text label + accent token (text-labelled, not color-only)
  var PRIORITY = {
    P0: { label: 'P0 · 严重', token: 'var(--danger)' },
    P1: { label: 'P1 · 重要', token: 'var(--warning)' },
    P2: { label: 'P2 · 关注', token: 'var(--primary)' },
    P3: { label: 'P3 · 记录', token: 'var(--foreground-secondary)' },
  };
  // confidence → text label (visually distinct, but always text-labelled)
  var CONFIDENCE = {
    raw: { label: '实测 raw', token: 'var(--success)' },
    derived: { label: '派生 derived', token: 'var(--primary)' },
    inferred: { label: '推断 inferred', token: 'var(--warning)' },
  };

  function priorityBadge(priority) {
    var p = PRIORITY[priority] || { label: esc(priority || '—'), token: 'var(--foreground-secondary)' };
    // reuse stat-chip; carry accent via text + filled tint (text label is authoritative).
    return el('span', {
      class: 'stat-chip',
      style: 'color:' + p.token + ';border-color:transparent;background:color-mix(in srgb, '
        + p.token + ' 10%, var(--gew-panel-fill,var(--surface-1)));font:var(--text-label);',
    }, p.label);
  }

  function confidenceChip(confidence) {
    var c = CONFIDENCE[confidence] || { label: esc(confidence || '—'), token: 'var(--foreground-secondary)' };
    return el('span', {
      class: 'stat-chip',
      style: 'color:' + c.token + ';border-color:transparent;background:color-mix(in srgb, '
        + c.token + ' 10%, var(--gew-panel-fill,var(--surface-1)));font:var(--text-label);',
    }, c.label);
  }

  function dimensionChip(text) {
    if (!text) return null;
    return el('span', {
      class: 'node-tag',
      style: 'text-transform:none;letter-spacing:0;border-color:transparent;background:var(--gew-panel-fill,var(--surface-1));',
    }, text);
  }

  function priorityFill(priority, strength) {
    var p = PRIORITY[priority] || { token: 'var(--foreground-secondary)' };
    var pct = strength || 8;
    return 'background:color-mix(in srgb, ' + p.token + ' ' + pct
      + '%, var(--gew-panel-fill,var(--surface-1)));';
  }

  function priorityClass(priority) {
    return String(priority || 'P3').toLowerCase();
  }

  function priorityTag(priority) {
    var p = String(priority || 'P3').toUpperCase();
    return el('span', { class: 'gew-priority-tag ' + priorityClass(p) }, p);
  }

  function normalizeActionId(id) {
    return id == null ? null : String(id);
  }

  function headBlock(title, children, first) {
    var kids = [el('div', { style: S.head }, title)].concat(
      Array.isArray(children) ? children : [children]
    );
    return el('section', { style: first ? S.sectionFirst : S.section }, kids);
  }

  function kvRow(key, val) {
    return el('div', { style: S.row }, [
      el('span', { style: S.rowKey }, String(key)),
      el('span', { style: S.rowVal }, String(val)),
    ]);
  }

  // === priority lookup for mapped-node filtering ===
  function nodePriority(nodeId) {
    var data = GEW.state.data;
    var pm = data && data.problemMap && data.problemMap.problemNodes;
    if (pm && pm[nodeId]) return pm[nodeId].priority;
    var iss = data && data.report && data.report.issues;
    if (iss && iss[nodeId] && iss[nodeId].diagnosis) return iss[nodeId].diagnosis.priority;
    return null;
  }

  function nodeLabel(nodeId) {
    var data = GEW.state.data;
    var ni = data && data.nodeInfo && data.nodeInfo[nodeId];
    if (ni && ni.idEn) return nodeId + ' · ' + ni.idEn;
    return nodeId;
  }

  function actionMappedNodes(action) {
    var data = GEW.state.data || {};
    var nodeSet = {};
    (((data.graph || {}).nodes) || []).forEach(function (n) {
      if (n && n.id) nodeSet[n.id] = true;
    });
    var out = [];
    function add(id) {
      if (!id || !nodeSet[id] || out.indexOf(id) >= 0) return;
      out.push(id);
    }
    add(action && action.nodeId);
    ((action && action.mappedNodes) || []).forEach(add);
    return out;
  }

  function selectedActionId() {
    return normalizeActionId(GEW.state.selectedActionId);
  }

  function selectedReportIssueRef() {
    return GEW.state.selectedReportIssueRef == null ? null : String(GEW.state.selectedReportIssueRef);
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

  function reportIssueActions(issueRef) {
    var report = GEW.state.data && GEW.state.data.report;
    var ref = issueRef == null ? null : String(issueRef);
    if (!ref) return [];
    return ((report && report.actions) || []).filter(function (action) {
      return actionIssueRefs(action).indexOf(ref) >= 0;
    });
  }

  function reportIssueMappedNodes(issueRef) {
    var data = GEW.state.data || {};
    var ref = issueRef == null ? null : String(issueRef);
    var out = [];
    function add(id) {
      if (!id || out.indexOf(id) >= 0) return;
      out.push(id);
    }
    Object.keys((data.report && data.report.issues) || {}).forEach(function (nodeId) {
      var issue = data.report.issues[nodeId] || {};
      if (String(issue.reportIssueRef || '') === ref) add(nodeId);
    });
    reportIssueActions(ref).forEach(function (action) {
      actionMappedNodes(action).forEach(add);
    });
    return out;
  }

  // ---------------------------------------------------------------------------
  // Sections (issue present)
  // ---------------------------------------------------------------------------

  function sectionDiagnosis(d, first) {
    var kids = [];
    kids.push(el('div', { style: S.chipRow }, [
      priorityBadge(d.priority),
      dimensionChip(d.dimension),
    ].filter(Boolean)));
    if (d.title) kids.push(el('div', { style: 'font:var(--text-body);font-weight:600;color:var(--foreground);' }, d.title));
    if (d.summary) kids.push(el('div', { style: S.muted }, d.summary));
    if (Array.isArray(d.metrics) && d.metrics.length) {
      var grid = el('div', { style: 'display:flex;flex-direction:column;gap:var(--space-1);margin-top:var(--space-1);' },
        d.metrics.map(function (m) { return kvRow(m[0], m[1]); }));
      kids.push(grid);
    }
    return headBlock('诊断 Diagnosis', kids, first);
  }

  function sectionEvidence(list) {
    var items = (list || []).map(function (e) {
      var meta = el('div', { style: S.metaLine }, [
        confidenceChip(e.confidence),
        el('span', { style: 'font:var(--text-mono);' },
          e.sourceFile + (e.sourceField ? ' · ' + e.sourceField : '')),
      ]);
      return el('div', { style: S.listItem }, [
        el('div', { style: S.body }, e.text),
        meta,
      ]);
    });
    return headBlock('证据 Evidence', items);
  }

  function sectionOperators(list) {
    var items = (list || []).map(function (o) {
      var name = o.name + (o.runtimeOpName && o.runtimeOpName !== o.name ? ' (' + o.runtimeOpName + ')' : '');
      var tail = o.value + (o.streamId != null ? ' · Stream ' + o.streamId : '');
      return kvRow(name, tail);
    });
    return headBlock('算子 Operators', items);
  }

  function sectionActions(list) {
    var items = (list || []).map(function (a) {
      var line = [el('span', { style: S.body }, a.text)];
      var tags = [];
      if (a.inferred) tags.push(el('span', {
        class: 'node-tag',
        style: 'text-transform:none;letter-spacing:0;color:var(--warning);border-color:transparent;'
          + 'background:color-mix(in srgb, var(--warning) 10%, var(--gew-panel-fill,var(--surface-1)));',
      }, '推断 inferred'));
      if (a.evidenceRef) tags.push(el('span', { class: 'io-chip' }, a.evidenceRef));
      var kids = [el('div', { style: S.body }, a.text)];
      if (tags.length) kids.push(el('div', { style: S.metaLine }, tags));
      return el('div', { style: S.listItem }, kids);
    });
    return headBlock('建议 Actions', items);
  }

	  function sectionVerification(list) {
    var items = (list || []).map(function (v) {
      return el('div', { style: S.row }, [
        el('span', { style: S.rowKey }, v.metric),
        el('span', { style: S.rowVal }, [
          el('span', { style: 'color:var(--foreground-secondary);' }, String(v.current)),
          document.createTextNode(' → '),
          el('span', { style: 'color:var(--success);' }, String(v.target)),
        ]),
      ]);
    });
	    return headBlock('验证 Verification', items);
	  }

	  function sectionCodeHint(text) {
	    if (!text) return null;
	    return headBlock('代码示例 Code', el('pre', {
	      style: 'margin:0;white-space:pre-wrap;overflow:auto;max-height:260px;'
	        + 'font:var(--text-mono);color:var(--foreground);'
	        + 'padding:var(--space-3);border-radius:var(--radius-md);'
	        + 'background:var(--gew-panel-fill,var(--surface-1));',
	    }, text));
	  }

	  function sectionReportOverview(report) {
	    var ov = report && report.overview;
	    if (!ov) return null;
	    var phs = ov.phs || {};
	    var summary = ov.summary || {};
	    var kids = [];
	    kids.push(el('div', { style: S.chipRow }, [
	      el('span', { class: 'node-tag', style: 'text-transform:none;letter-spacing:0;' }, ov.taskType || 'Profiling'),
	      el('span', { class: 'io-chip' }, ov.reportDate || report.reportId || ''),
	      phs.current != null ? el('span', { class: 'stat-chip' }, 'PHS ' + phs.current + ' / 100 · ' + (phs.grade || '')) : null,
	      phs.estimated != null ? el('span', { class: 'stat-chip' }, '优化后 ' + phs.estimated + ' · ' + (phs.estGrade || '')) : null,
	    ].filter(Boolean)));
	    kids.push(el('div', { style: 'font:var(--text-body);font-weight:600;color:var(--foreground);' }, ov.title || '原始报告'));
	    if (ov.subtitle) kids.push(el('div', { style: S.muted }, ov.subtitle));
	    if (summary.conclusion) kids.push(el('div', { style: S.listItem }, [
	      el('div', { style: S.metaLine }, [el('span', { class: 'io-chip' }, '结论')]),
	      el('div', { style: S.body }, summary.conclusion),
	    ]));
	    if (summary.topBottleneck) kids.push(el('div', { style: S.listItem }, [
	      el('div', { style: S.metaLine }, [el('span', { class: 'io-chip' }, '头号瓶颈')]),
	      el('div', { style: S.body }, summary.topBottleneck),
	    ]));
	    if (summary.maxGain) kids.push(el('div', { style: S.listItem }, [
	      el('div', { style: S.metaLine }, [el('span', { class: 'io-chip' }, '收益上限')]),
	      el('div', { style: S.body }, summary.maxGain),
	    ]));
	    return headBlock('原始报告 Report', kids);
	  }

	  function sectionReportActions(report) {
	    var list = report && report.actions;
	    if (!Array.isArray(list) || !list.length) return null;
	    var items = list.map(function (a) {
	      var mapped = actionMappedNodes(a);
	      var actionId = normalizeActionId(a.id);
	      var issueRefs = actionIssueRefs(a);
	      var isSelected = selectedActionId() === actionId;
	      var isRelated = mapped.indexOf(GEW.state.selectedNodeId) >= 0;
	      var cls = 'gew-action-card'
	        + (isSelected ? ' is-selected' : '')
	        + (isRelated ? ' is-node-related' : '');
	      var issueText = issueRefs.length ? '对应 issue ' + issueRefs.join(' / ') : '报告级';
	      var linkedText = mapped.length
	        ? '映射 ' + mapped.map(function (id) { return nodeLabel(id); }).join(' / ')
	        : '未映射图节点';
	      return el('button', {
	        class: cls,
	        type: 'button',
	        dataset: { actionId: actionId || '' },
	        'aria-pressed': isSelected ? 'true' : 'false',
	        title: issueText + ' · ' + linkedText,
	        onclick: function () {
	          GEW.state.selectedActionId = actionId;
	          if (mapped[0]) GEW.state.selectedNodeId = mapped[0];
	          GEW.bus.emit('action:select', {
	            actionId: actionId,
	            nodeId: mapped[0] || null,
	            mappedNodes: mapped,
	            source: 'inspector',
	          });
	          GEW.inspector.render(mapped[0] || GEW.state.selectedNodeId);
	        },
	      }, [
	        el('div', { class: 'gew-action-card-meta' }, [
	          priorityTag(a.priority),
	          el('span', { class: 'io-chip' }, '#' + a.id),
	          el('span', { class: 'io-chip' }, issueText),
	          el('span', { class: 'io-chip' }, a.benefit || '收益待估'),
	          el('span', { class: 'io-chip' }, '难度 ' + (a.difficulty || '—')),
	        ]),
	        el('div', { class: 'gew-action-card-title' }, a.problem || ''),
	        a.location ? el('div', { style: S.muted }, '位置：' + a.location) : null,
	        a.visualization ? el('div', { style: S.muted }, '视图：' + a.visualization) : null,
	        el('div', { class: 'gew-action-link-state' }, linkedText),
	      ].filter(Boolean));
	    });
	    return headBlock('行动清单 Actions', items);
	  }

	  function sectionReportIssues(report, currentRef) {
	    var list = report && report.reportIssues;
	    if (!Array.isArray(list) || !list.length) return null;
	    var selectedRef = selectedReportIssueRef() || (currentRef == null ? null : String(currentRef));
	    var sorted = list.slice().sort(function (a, b) {
	      return String(a.id).localeCompare(String(b.id));
	    });
	    var items = sorted.map(function (it) {
	      var ref = String(it.id);
	      var isSelected = ref === selectedRef;
	      var actions = reportIssueActions(ref);
	      var mapped = reportIssueMappedNodes(ref);
	      var title = actions.length
	        ? 'Actions ' + actions.map(function (a) { return '#' + a.id; }).join(' / ')
	        : '无对应 Action';
	      if (mapped.length) title += ' · 映射 ' + mapped.map(function (id) { return nodeLabel(id); }).join(' / ');
	      return el('button', {
	        class: 'gew-report-issue-card' + (isSelected ? ' is-selected' : ''),
	        type: 'button',
	        dataset: { issueRef: ref },
	        'aria-pressed': isSelected ? 'true' : 'false',
	        title: title,
	        onclick: function () {
	          var action = actions[0] || null;
	          GEW.state.selectedReportIssueRef = ref;
	          if (action) GEW.state.selectedActionId = normalizeActionId(action.id);
	          if (mapped[0]) GEW.state.selectedNodeId = mapped[0];
	          GEW.bus.emit('report-issue:select', {
	            reportIssueRef: ref,
	            actionId: action ? normalizeActionId(action.id) : null,
	            nodeId: mapped[0] || null,
	            mappedNodes: mapped,
	            source: 'inspector',
	          });
	          GEW.inspector.render(mapped[0] || GEW.state.selectedNodeId);
	        },
	      }, [
	        el('div', { style: S.metaLine }, [
	          priorityBadge(it.priority),
	          el('span', { class: 'io-chip' }, 'issue ' + it.id),
	          isSelected ? el('span', { class: 'io-chip' }, '当前') : null,
	          actions.length ? el('span', { class: 'io-chip' }, actions.map(function (a) { return '#' + a.id; }).join(' / ')) : null,
	        ].filter(Boolean)),
	        el('div', { style: 'font:var(--text-body);font-weight:600;color:var(--foreground);' }, it.title || ''),
	        it.evidence ? el('div', { style: S.muted }, '证据：' + it.evidence) : null,
	        it.impact ? el('div', { style: S.muted }, '影响：' + it.impact) : null,
	        it.verification ? el('div', { style: S.muted }, '验证：' + it.verification) : null,
	      ].filter(Boolean));
	    });
	    return headBlock('报告问题详情 Report Issues', items);
	  }

	  function sectionNoProblems(report) {
	    var list = report && report.noProblems;
	    if (!Array.isArray(list) || !list.length) return null;
	    return headBlock('已确认无问题', list.map(function (text) {
	      return el('div', { style: S.listItem }, text);
	    }));
	  }

	  function sectionReportMeta(report) {
	    var meta = report && report.meta;
	    if (!meta) return null;
	    var rows = [];
	    if (meta.dataPath) rows.push(kvRow('数据路径', meta.dataPath));
	    if (meta.range) rows.push(kvRow('范围', meta.range));
	    if (meta.version) rows.push(kvRow('版本', meta.version));
	    if (meta.output) rows.push(kvRow('输出', meta.output));
	    if (meta.advisorStatus) rows.push(el('div', { style: S.listItem }, 'Advisor：' + meta.advisorStatus));
	    if (Array.isArray(meta.skills)) {
	      rows.push(el('div', { style: S.metaLine }, meta.skills.map(function (s) {
	        return el('span', { class: 'io-chip' }, s);
	      })));
	    }
	    return rows.length ? headBlock('数据与方法 Meta', rows) : null;
	  }

  function sectionMappedNodes(issue, selfId) {
    var ids = [];
    (issue.mappedNodes || []).forEach(function (id) { if (ids.indexOf(id) < 0) ids.push(id); });
    if (ids.indexOf(selfId) < 0) ids.unshift(selfId);

    var filter = GEW.state.activeFilter;
    var items = ids.map(function (id) {
      var prio = nodePriority(id);
      var dim = filter && filter !== 'all' && filter !== 'off' && prio !== filter;
      var btn = el('button', {
        class: 'btn btn-ghost btn-sm',
        style: 'justify-content:flex-start;width:100%;text-align:left;'
          + (dim ? 'opacity:0.42;' : ''),
        type: 'button',
        title: nodeLabel(id),
        onclick: function () { GEW.bus.emit('selection:change', { nodeId: id, source: 'mapped' }); },
      }, [
        el('span', { style: 'flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;' },
          id + (id === selfId ? ' （当前）' : '')),
        prio ? el('span', { style: 'font:var(--text-label);color:var(--foreground-secondary);flex:0 0 auto;' }, prio) : null,
      ].filter(Boolean));
      return btn;
    });
    return headBlock('关联节点 Mapped Nodes',
      el('div', { style: 'display:flex;flex-direction:column;gap:var(--space-1);' }, items));
  }

  function sectionCoverage(cov) {
    var kids = [];
    var pct = cov.ratio != null ? Math.round(cov.ratio * 100) + ' %' : '—';
    kids.push(kvRow('数据覆盖率', pct));
    if (Array.isArray(cov.covered) && cov.covered.length) {
      kids.push(el('div', { style: S.metaLine }, [
        el('span', { style: 'color:var(--success);' }, '已覆盖'),
      ].concat(cov.covered.map(function (c) { return el('span', { class: 'io-chip' }, c); }))));
    }
    if (Array.isArray(cov.missing) && cov.missing.length) {
      kids.push(el('div', { style: S.metaLine }, [
        el('span', { style: 'color:var(--warning);' }, '缺失'),
      ].concat(cov.missing.map(function (m) { return el('span', { class: 'io-chip' }, m); }))));
    }
    return headBlock('数据覆盖 Data Coverage', kids);
  }

  // ---------------------------------------------------------------------------
  // Node-info-only view (non-problem node)
  // ---------------------------------------------------------------------------

  function nodeInfoView(nodeId, info, first) {
    var kids = [];
    kids.push(el('div', { style: 'font:var(--text-body);font-weight:600;color:var(--foreground);' },
      nodeId + (info.idEn ? ' · ' + info.idEn : '')));
    if (info.what) kids.push(el('div', { style: S.muted }, info.what));
    if (Array.isArray(info.clusters) && info.clusters.length) {
      kids.push(el('div', { style: S.chipRow },
        info.clusters.map(function (c) { return el('span', { class: 'node-tag', style: 'text-transform:none;letter-spacing:0;' }, c); })));
    }
    var blocks = [headBlock('节点信息 Node Info', kids, first)];

    function ioBlock(title, arr, dirKey) {
      if (!Array.isArray(arr) || !arr.length) return null;
      var items = arr.map(function (x) {
        return el('div', { style: S.listItem }, [
          el('div', { style: 'font:var(--text-mono);color:var(--foreground);' }, x[dirKey] || ''),
          x.desc ? el('div', { style: S.muted }, x.desc) : null,
        ].filter(Boolean));
      });
      return headBlock(title, items);
    }
    var ib = ioBlock('输入 Inputs', info.inputs, 'from');
    var ob = ioBlock('输出 Outputs', info.outputs, 'to');
    if (ib) blocks.push(ib);
    if (ob) blocks.push(ob);

    if (info.params) blocks.push(headBlock('参数 Params', el('div', { style: S.body }, info.params)));

    if (Array.isArray(info.sources) && info.sources.length) {
      var src = info.sources.map(function (s) {
        return el('a', {
          class: 'io-chip', href: s.url || '#', target: '_blank', rel: 'noopener',
          style: 'text-decoration:none;cursor:pointer;',
        }, s.text || s.url);
      });
      blocks.push(headBlock('来源 Sources', el('div', { style: S.chipRow }, src)));
    }

    // muted note: no diagnosis for this node
    blocks.push(el('section', { style: S.section }, [
      el('div', { style: S.muted + 'padding:var(--space-2);'
        + 'border-radius:var(--radius-md);background:var(--gew-panel-fill,var(--surface-1));' }, '该节点无诊断问题。'),
    ]));
    return blocks;
  }

  // ---------------------------------------------------------------------------

  function selectedNodeNotice(nodeId) {
    if (!nodeId) return null;
    var prio = nodePriority(nodeId);
    return headBlock('当前节点 Selection', el('div', { style: S.listItem }, [
      el('div', { style: S.metaLine }, [
        el('span', { class: 'io-chip' }, nodeId),
        prio ? priorityTag(prio) : null,
      ].filter(Boolean)),
      el('div', { style: S.muted }, '该节点暂无诊断问题。'),
    ]), true);
  }

  function reportBlocks(data, currentRef, nodeId) {
    var blocks = [];
    [
      sectionReportIssues(data && data.report, currentRef),
    ].forEach(function (block) { if (block) blocks.push(block); });
    if (nodeId) issueBlocks(
      data && data.report && data.report.issues && data.report.issues[nodeId],
      nodeId
    ).forEach(function (block) { if (block) blocks.push(block); });
    [
      sectionReportActions(data && data.report),
      sectionNoProblems(data && data.report),
      sectionReportMeta(data && data.report),
    ].forEach(function (block) { if (block) blocks.push(block); });
    return blocks;
  }

  function issueBlocks(issue, nodeId) {
    var blocks = [];
    if (!issue) {
      var notice = selectedNodeNotice(nodeId);
      if (notice) blocks.push(notice);
      return blocks;
    }
    if (issue.diagnosis) blocks.push(sectionDiagnosis(issue.diagnosis, true));
    if (issue.evidence) blocks.push(sectionEvidence(issue.evidence));
    if (issue.operators) blocks.push(sectionOperators(issue.operators));
    if (issue.actions) blocks.push(sectionActions(issue.actions));
    var codeBlock = sectionCodeHint(issue.codeHint);
    if (codeBlock) blocks.push(codeBlock);
    if (issue.verification) blocks.push(sectionVerification(issue.verification));
    if (issue.mappedNodes || true) blocks.push(sectionMappedNodes(issue, nodeId));
    if (issue.coverage) blocks.push(sectionCoverage(issue.coverage));
    return blocks;
  }

  function placeholder(text) {
    return el('div', {
      style: 'display:flex;align-items:center;justify-content:center;height:100%;'
        + 'min-height:160px;text-align:center;padding:var(--space-5);'
        + 'font:var(--text-body);color:var(--foreground-secondary);',
    }, text || '选择计算图节点或泳道任务查看诊断证据');
  }

  function mount(children) {
    if (!container) return;
    container.textContent = '';
    var wrap = el('div', { style: 'display:flex;flex-direction:column;' },
      Array.isArray(children) ? children : [children]);
    container.appendChild(wrap);
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  GEW.inspector = {
    init: function (opts) {
      container = (opts && opts.container) || GEW.util.qs('gew-inspector-body');
      GEW.bus.on('selection:change', function (d) {
        if (d && d.source === 'inspector') return; // avoid self-churn
        GEW.inspector.render(d && d.nodeId);
      });
      GEW.bus.on('action:select', function () {
        GEW.inspector.render(GEW.state.selectedNodeId);
      });
      GEW.bus.on('report-issue:select', function () {
        GEW.inspector.render(GEW.state.selectedNodeId);
      });
      GEW.bus.on('filter:change', function (d) {
        GEW.inspector.render(GEW.state.selectedNodeId);
      });
    },

    render: function (nodeId) {
      if (!container) return;
      var data = GEW.state.data;
      if (!data) { GEW.inspector.clear(); return; }

      var issue = data.report && data.report.issues && data.report.issues[nodeId];
      var currentRef = selectedReportIssueRef() || (issue && issue.reportIssueRef);
      var blocks = reportBlocks(data, currentRef, nodeId);
      mount(blocks.length ? blocks : placeholder());
	    },

    clear: function () {
      var data = GEW.state.data;
      if (data) mount(reportBlocks(data, null, null));
      else mount(placeholder());
    },
  };
})(window);
