/* GEW.loader — async load data/*.json, validate schemaVersion, parse trace -> laneModel.
   Classic IIFE, attaches to window.GEW. See CONTRACT.md §4/§6.
   Throws GEW.SchemaError on 404, version mismatch, or missing required top-level keys. */
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};

	  // file set per reportId. MVP = single qwen2-7b set; parameterizable.
	  function fileSetFor(reportId) {
	    const requested = reportId || (GEW.MODEL_ID || 'qwen2-7b');
	    const id = requested === 'r20260526' ? (GEW.MODEL_ID || 'qwen2-7b') : requested;
	    return {
	      reportId: id,
	      requestedReportId: requested,
	      base: 'data/',
      files: {
        graph: id + '.graph.json',
        nodeInfo: id + '.node-info.json',
        problemMap: id + '.problem-map.json',
        report: id + '.demo-report.json',
        trace: id + '.demo.trace_view.json',
      },
    };
  }

  async function fetchJson(url) {
    let res;
    try {
      res = await fetch(url, { cache: 'no-cache' });
    } catch (e) {
      throw new GEW.SchemaError('Failed to fetch ' + url + ' (' + (e && e.message) + ')', { url: url });
    }
    if (res.status === 404) {
      throw new GEW.SchemaError('Data file not found (404): ' + url, { url: url, status: 404 });
    }
    if (!res.ok) {
      throw new GEW.SchemaError('Failed to load ' + url + ' (HTTP ' + res.status + ')', { url: url, status: res.status });
    }
    try {
      return await res.json();
    } catch (e) {
      throw new GEW.SchemaError('Invalid JSON in ' + url + ' (' + (e && e.message) + ')', { url: url });
    }
  }

  function checkSchema(obj, url) {
    const v = obj && obj.schemaVersion;
    if (v !== GEW.SCHEMA_VERSION) {
      throw new GEW.SchemaError(
        'Schema version mismatch in ' + url + ': got ' + JSON.stringify(v) +
        ', expected ' + JSON.stringify(GEW.SCHEMA_VERSION),
        { url: url, got: v, expected: GEW.SCHEMA_VERSION }
      );
    }
  }

  function requireKey(obj, key, url) {
    if (!obj || obj[key] == null) {
      throw new GEW.SchemaError('Missing required key "' + key + '" in ' + url, { url: url, key: key });
    }
  }

  function addUnique(list, nodeSet, nodeId) {
    if (!nodeId || !nodeSet[nodeId] || list.indexOf(nodeId) >= 0) return;
    list.push(nodeId);
  }

  function linkReportActions(report, graphData, problemMap) {
    var nodeSet = {};
    ((graphData && graphData.nodes) || []).forEach(function (n) {
      if (n && n.id) nodeSet[n.id] = true;
    });
    var reportIssueSet = {};
    ((report && report.reportIssues) || []).forEach(function (issue) {
      if (issue && issue.id != null) reportIssueSet[String(issue.id)] = true;
    });

    var refToNodes = {};
    function addRef(ref, nodeId) {
      if (!ref || !nodeSet[nodeId]) return;
      var key = String(ref);
      refToNodes[key] = refToNodes[key] || [];
      if (refToNodes[key].indexOf(nodeId) < 0) refToNodes[key].push(nodeId);
    }

    Object.keys((report && report.issues) || {}).forEach(function (nodeId) {
      var issue = report.issues[nodeId] || {};
      addRef(issue.reportIssueRef, nodeId);
    });
    Object.keys((problemMap && problemMap.problemNodes) || {}).forEach(function (nodeId) {
      var issue = problemMap.problemNodes[nodeId] || {};
      addRef(issue.issueRef, nodeId);
    });

    ((report && report.actions) || []).forEach(function (action) {
      var mapped = [];
      if (action.nodeId) addUnique(mapped, nodeSet, action.nodeId);
      (action.mappedNodes || []).forEach(function (id) { addUnique(mapped, nodeSet, id); });

      var refs = [];
      if (action.reportIssueRef) refs.push(action.reportIssueRef);
      if (action.issueRef) refs.push(action.issueRef);
      if (action.id != null) refs.push('3.' + action.id);
      var linkedIssueRefs = [];
      refs.forEach(function (ref) {
        var key = String(ref);
        if (reportIssueSet[key] && linkedIssueRefs.indexOf(key) < 0) linkedIssueRefs.push(key);
        (refToNodes[key] || []).forEach(function (id) { addUnique(mapped, nodeSet, id); });
      });

      var hay = [
        action.problem,
        action.location,
        action.visualization,
        action.priority,
      ].filter(Boolean).join(' ').toLowerCase();
      if (/lm[_\s-]?head|vocab|matmulv3|applyadam/.test(hay)) addUnique(mapped, nodeSet, 'lm_head');
      if (/cross[-\s]?entropy|logits|softmax|realdiv|argmax|reducesum|loss/.test(hay)) {
        addUnique(mapped, nodeSet, 'output_logits');
      }

      action.mappedNodes = mapped;
      action.reportIssueRefs = linkedIssueRefs;
      if (!action.reportIssueRef && linkedIssueRefs.length) action.reportIssueRef = linkedIssueRefs[0];
      if (!action.nodeId && mapped.length) action.nodeId = mapped[0];
    });
  }

  async function load(reportId) {
    if (!GEW.traceParser || typeof GEW.traceParser.parse !== 'function') {
      throw new GEW.SchemaError('GEW.traceParser not available — load order error');
    }
    const set = fileSetFor(reportId);
    const url = (name) => set.base + set.files[name];

    // fetch all in parallel
    const [graph, nodeInfo, problemMap, report, trace] = await Promise.all([
      fetchJson(url('graph')),
      fetchJson(url('nodeInfo')),
      fetchJson(url('problemMap')),
      fetchJson(url('report')),
      fetchJson(url('trace')),
    ]);

    // validate schemaVersion on non-trace files
    checkSchema(graph, url('graph'));
    checkSchema(nodeInfo, url('nodeInfo'));
    checkSchema(problemMap, url('problemMap'));
    checkSchema(report, url('report'));

    // validate required top-level keys
    requireKey(graph, 'graph', url('graph'));
    requireKey(nodeInfo, 'nodeInfo', url('nodeInfo'));
    requireKey(problemMap, 'problemNodes', url('problemMap'));
    requireKey(report, 'issues', url('report'));

    // parse trace -> laneModel
    const laneModel = GEW.traceParser.parse(trace, { problemMap: problemMap });
    linkReportActions(report, graph.graph, problemMap);

    const sourceFiles = Object.keys(set.files).map((k) => set.files[k]);
    const generatedAt = graph.generatedAt || report.generatedAt || new Date().toISOString();
    // business reportId lives in the data files; set.reportId is only the file-set key.
    const businessReportId = graph.reportId || report.reportId || problemMap.reportId || set.reportId;

    return {
      graph: graph.graph,
      nodeInfo: nodeInfo.nodeInfo,
      problemMap: problemMap,
      report: report,
      laneModel: laneModel,
      meta: { reportId: businessReportId, fileSetId: set.reportId, sourceFiles: sourceFiles, generatedAt: generatedAt },
    };
  }

  GEW.loader = { load: load, _fileSetFor: fileSetFor };
})(window);
