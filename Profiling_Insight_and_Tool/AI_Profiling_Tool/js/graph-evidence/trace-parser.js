/* GEW.traceParser — parse Chrome Trace Event Format (trace_view.json) into GEW.LaneModel.
   Classic IIFE, attaches to window.GEW. See CONTRACT.md §4/§5.

   Conventions (documented per CONTRACT §5):
   - All time fields (timeRange, startUs, durUs, steps.*) are microseconds (µs), derived
     directly from trace ts/dur. They are NOT ratios.
   - step.compute = sum of hardware-stream task dur within the step.
   - step.comm    = sum of communication (HCCL/hcom) task dur within the step.
   - step.free    = step span (durUs) - busyUs, where busyUs is the union-merged wall-clock
     coverage of all hardware + communication tasks in the step (so overlapping work is not
     double-counted). free is clamped to >= 0.
   - step.overlap = (compute + comm) - busyUs  (µs of wall-clock where compute & comm coincide).
   - task.status  = 'wait' if waitUs > 0.3 * durUs, else 'overlap' if the task wall-clock
     intersects another lane's task, else 'ok'.
*/
(function (w) {
  'use strict';
  w.GEW = w.GEW || {};

  // ---- layer classification by thread / process name --------------------
  function classifyLayer(threadName, procName) {
    const n = String(threadName || '') + ' ' + String(procName || '');
    const s = n.toLowerCase();
    // Communication first (may also contain "stream")
    if (/hccl|hcom|communicat/.test(s)) return 'communication';
    if (/ascend hardware|hardware.*stream|^stream\b|\/stream\s*\d+/.test(s) || /stream\s*\d+/.test(s)) {
      // hardware stream lanes — but only when it's the Ascend Hardware process,
      // not a CANN host thread that merely mentions stream
      if (/ascend hardware|hardware/.test(s) || /stream\s*\d+/.test(s)) return 'hardware';
    }
    if (/ascendcl|acl\b|\bge\b|graph engine|runtime|cann/.test(s)) return 'cann';
    if (/python|pytorch|aten|torch/.test(s)) return 'python';
    return 'other';
  }

  function parseStreamFromName(name) {
    const m = /stream\s*(\d+)/i.exec(String(name || ''));
    return m ? Number(m[1]) : null;
  }

  // pick the first defined value among arg key aliases
  function arg(args, keys) {
    if (!args) return undefined;
    for (let i = 0; i < keys.length; i++) {
      if (args[keys[i]] != null) return args[keys[i]];
    }
    return undefined;
  }

  function numOrNull(v) {
    if (v == null || v === '') return null;
    const n = Number(v);
    return isNaN(n) ? null : n;
  }

  // union length of [start,end) intervals (µs), inputs need not be sorted
  function unionLength(intervals) {
    if (!intervals.length) return 0;
    const arr = intervals.slice().sort((a, b) => a[0] - b[0]);
    let total = 0, curS = arr[0][0], curE = arr[0][1];
    for (let i = 1; i < arr.length; i++) {
      const [s, e] = arr[i];
      if (s > curE) { total += curE - curS; curS = s; curE = e; }
      else if (e > curE) curE = e;
    }
    total += curE - curS;
    return total;
  }

  function intervalsIntersect(a, b) {
    return a[0] < b[1] && b[0] < a[1];
  }

  function unwrapTrace(traceJson) {
    if (!traceJson || typeof traceJson !== 'object') return { events: [], meta: {} };
    if (Array.isArray(traceJson.traceEvents)) return { events: traceJson.traceEvents, meta: traceJson };
    if (traceJson.trace && Array.isArray(traceJson.trace.traceEvents)) {
      return { events: traceJson.trace.traceEvents, meta: traceJson.trace };
    }
    if (Array.isArray(traceJson)) return { events: traceJson, meta: {} };
    return { events: [], meta: traceJson };
  }

  function parse(traceJson, opts) {
    opts = opts || {};
    const problemMap = opts.problemMap || null;
    const { events } = unwrapTrace(traceJson);

    const gaps = [];
    if (!Array.isArray(events) || !events.length) {
      gaps.push({ laneId: '*', kind: 'all', reason: 'no trace events' });
      return { timeRange: { startUs: 0, endUs: 0 }, steps: [], lanes: [], byNode: {}, gaps };
    }

    // ---- pass 1: metadata maps -----------------------------------------
    const procName = {};            // pid -> name
    const threadName = {};          // pid|tid -> name
    const tkey = (pid, tid) => pid + '|' + tid;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      if (!ev || ev.ph !== 'M') continue;
      const nm = ev.args && (ev.args.name != null ? ev.args.name : ev.args.Name);
      if (ev.name === 'process_name' && ev.pid != null) procName[ev.pid] = nm;
      else if (ev.name === 'thread_name' && ev.pid != null && ev.tid != null) threadName[tkey(ev.pid, ev.tid)] = nm;
    }

    // resolve a layer + display name per (pid,tid)
    function layerFor(ev) {
      const tn = threadName[tkey(ev.pid, ev.tid)];
      const pn = procName[ev.pid];
      return { layer: classifyLayer(tn, pn), threadName: tn, procName: pn };
    }

    // ---- pass 2: build tasks from X events ------------------------------
    const tasks = [];
    let tid = 0;
    let minTs = Infinity, maxEnd = -Infinity;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      if (!ev || ev.ph !== 'X') continue;
      try {
        const ts = numOrNull(ev.ts);
        const dur = numOrNull(ev.dur);
        if (ts == null || dur == null) continue; // skip malformed
        const a = ev.args || {};
        const li = layerFor(ev);

        let streamId = numOrNull(arg(a, ['Stream Id', 'Stream ID', 'streamId']));
        if (streamId == null) streamId = parseStreamFromName(li.threadName);
        const stepId = numOrNull(arg(a, ['Step Id', 'Step ID', 'stepId']));
        const taskTrcId = numOrNull(arg(a, ['Task Id', 'Task ID', 'taskId']));
        const waitUs = numOrNull(arg(a, ['Wait Time(us)', 'Wait Time(µs)', 'Wait Time', 'waitUs'])) || 0;
        const inputShapes = arg(a, ['Input Shapes', 'Input Shape', 'inputShapes']);
        const connectionId = numOrNull(arg(a, ['Connection Id', 'Connection ID', 'connectionId']));
        const rankId = numOrNull(arg(a, ['Rank Id', 'Rank ID', 'rankId']));

        // laneKind: what swimlane lane this task belongs in
        let laneKind;
        if (li.layer === 'communication') laneKind = 'communication';
        else if (li.layer === 'hardware') laneKind = 'stream';
        else laneKind = li.layer; // python / cann / other — kept but not a primary lane

        const opName = ev.name;
        const start = ts;
        const end = ts + dur;
        if (start < minTs) minTs = start;
        if (end > maxEnd) maxEnd = end;

        const status = (dur > 0 && waitUs > 0.3 * dur) ? 'wait' : 'ok'; // 'overlap' resolved later

        tasks.push({
          id: 't' + (tid++),
          label: opName,
          opName: opName,
          runtimeOpName: opName,
          nodeId: null,
          startUs: start,
          durUs: dur,
          waitUs: waitUs,
          streamId: streamId,
          stepId: stepId,
          rankId: rankId,
          status: status,
          sourceFile: 'trace_view.json',
          laneKind: laneKind,
          // internal extras (not part of contract, used for aggregation)
          _layer: li.layer,
          _taskTrcId: taskTrcId,
          _connectionId: connectionId,
          _inputShapes: inputShapes,
          _commChannel: li.layer === 'communication' ? (li.threadName || 'comm') : null,
          _end: end,
        });
      } catch (e) {
        // skip malformed event, continue
      }
    }

    if (!tasks.length) {
      gaps.push({ laneId: '*', kind: 'all', reason: 'no complete (ph:X) events with ts/dur' });
      return { timeRange: { startUs: 0, endUs: 0 }, steps: [], lanes: [], byNode: {}, gaps };
    }

    const timeRange = { startUs: minTs, endUs: maxEnd };

    // ---- resolve 'overlap' status (task intersects a task in another lane) ----
    // group tasks by lane bucket for overlap detection
    const hwTasks = tasks.filter((t) => t.laneKind === 'stream');
    const commTasks = tasks.filter((t) => t.laneKind === 'communication');
    // mark overlap: a hardware/comm task whose wall-clock intersects a task on the *other* side
    function markOverlap(setA, setB) {
      const sorted = setB.slice().sort((a, b) => a.startUs - b.startUs);
      setA.forEach((t) => {
        if (t.status === 'wait') return;
        for (let i = 0; i < sorted.length; i++) {
          const o = sorted[i];
          if (o.startUs >= t._end) break;
          if (intervalsIntersect([t.startUs, t._end], [o.startUs, o._end])) { t.status = 'overlap'; break; }
        }
      });
    }
    markOverlap(hwTasks, commTasks);
    markOverlap(commTasks, hwTasks);

    // ---- byNode mapping via problemMap (sets task.nodeId before lanes built) ----
    const byNode = {};
    if (problemMap && problemMap.problemNodes) {
	      Object.keys(problemMap.problemNodes).forEach((nodeId) => {
	        const pn = problemMap.problemNodes[nodeId] || {};
	        const wantOps = Array.isArray(pn.runtimeOpNames)
	          ? pn.runtimeOpNames
	          : (pn.runtimeOpName != null ? [pn.runtimeOpName] : []);
	        if (!wantOps.length) return; // need at least an op name to match
	        const wantStream = pn.streamId != null ? Number(pn.streamId) : null;
	        const wantStep = pn.stepId != null ? Number(pn.stepId) : null;
	        const matched = [];
	        tasks.forEach((t) => {
	          if (wantOps.indexOf(t.opName) < 0 && wantOps.indexOf(t.runtimeOpName) < 0) return;
	          if (wantStream != null && t.streamId !== wantStream) return;
	          if (wantStep != null && t.stepId !== wantStep) return;
	          matched.push(t.id);
          t.nodeId = nodeId; // back-link (lenient: last writer wins)
        });
        if (matched.length) byNode[nodeId] = matched;
      });
    }

    // ---- steps aggregation ---------------------------------------------
    const stepMap = new Map();
    tasks.forEach((t) => {
      if (t.stepId == null) return;
      if (!stepMap.has(t.stepId)) stepMap.set(t.stepId, []);
      stepMap.get(t.stepId).push(t);
    });

    const steps = Array.from(stepMap.keys()).sort((a, b) => a - b).map((sid) => {
      const list = stepMap.get(sid);
      let sStart = Infinity, sEnd = -Infinity, compute = 0, comm = 0;
      const busy = [];
      list.forEach((t) => {
        if (t.startUs < sStart) sStart = t.startUs;
        if (t._end > sEnd) sEnd = t._end;
        if (t.laneKind === 'stream') { compute += t.durUs; busy.push([t.startUs, t._end]); }
        else if (t.laneKind === 'communication') { comm += t.durUs; busy.push([t.startUs, t._end]); }
      });
      const span = sEnd - sStart;
      const busyUs = unionLength(busy);
      const free = Math.max(0, span - busyUs);
      const overlap = Math.max(0, (compute + comm) - busyUs);
      return { stepId: sid, startUs: sStart, durUs: span, compute, comm, free, overlap };
    });

    // ---- lanes ----------------------------------------------------------
    const lanes = [];

    // step lane: one bar per step
    lanes.push({
      id: 'lane-step',
      kind: 'step',
      label: 'Steps',
      tasks: steps.map((s) => ({
        id: 'step-' + s.stepId,
        label: 'Step ' + s.stepId,
        opName: 'Step ' + s.stepId,
        nodeId: null,
        startUs: s.startUs,
        durUs: s.durUs,
        waitUs: s.free,
        streamId: null,
        stepId: s.stepId,
        status: s.overlap > 0 ? 'overlap' : (s.free > 0.3 * s.durUs ? 'wait' : 'ok'),
        sourceFile: 'trace_view.json',
        laneKind: 'step',
      })),
    });
    if (!steps.length) gaps.push({ laneId: 'lane-step', kind: 'step', reason: 'no Step Id on any event' });

    // stream lanes: one per Stream Id
    const streamGroups = new Map();
    hwTasks.forEach((t) => {
      const key = t.streamId == null ? 'na' : t.streamId;
      if (!streamGroups.has(key)) streamGroups.set(key, []);
      streamGroups.get(key).push(t);
    });
    const streamKeys = Array.from(streamGroups.keys()).sort((a, b) => {
      if (a === 'na') return 1; if (b === 'na') return -1; return a - b;
    });
    streamKeys.forEach((key) => {
      lanes.push({
        id: 'lane-stream-' + key,
        kind: 'stream',
        label: key === 'na' ? 'Stream ?' : 'Stream ' + key,
        tasks: streamGroups.get(key),
      });
    });
    if (!hwTasks.length) gaps.push({ laneId: 'lane-stream', kind: 'stream', reason: 'no Ascend Hardware stream tasks' });

    // communication lane(s): one per comm channel/thread (e.g. HCCL, hcom)
    const commGroups = new Map();
    commTasks.forEach((t) => {
      const key = t._commChannel || 'comm';
      if (!commGroups.has(key)) commGroups.set(key, []);
      commGroups.get(key).push(t);
    });
    const commKeys = Array.from(commGroups.keys()).sort();
    if (commKeys.length) {
      commKeys.forEach((key) => {
        lanes.push({
          id: 'lane-communication-' + slug(key),
          kind: 'communication',
          label: key,
          tasks: commGroups.get(key),
        });
      });
    } else {
      lanes.push({ id: 'lane-communication', kind: 'communication', label: 'Communication', tasks: [] });
      gaps.push({ laneId: 'lane-communication', kind: 'communication', reason: 'no HCCL/hcom tasks' });
    }

    // overlap lane: computed segments where compute & comm coincide, plus free gaps per step
    const overlapTasks = [];
    steps.forEach((s) => {
      if (s.overlap > 0) {
        overlapTasks.push({
          id: 'ov-' + s.stepId,
          label: 'overlap',
          opName: 'compute∩comm',
          nodeId: null,
          startUs: s.startUs,
          durUs: s.durUs,
          waitUs: s.free,
          streamId: null,
          stepId: s.stepId,
          status: 'overlap',
          sourceFile: 'trace_view.json',
          laneKind: 'overlap',
        });
      }
    });
    lanes.push({ id: 'lane-overlap', kind: 'overlap', label: 'Overlap / Free', tasks: overlapTasks });
    if (!overlapTasks.length) gaps.push({ laneId: 'lane-overlap', kind: 'overlap', reason: 'no compute/comm overlap detected' });

    // coverage lane: stub — populated by the swimlane agent from byNode mapping.
    lanes.push({ id: 'lane-coverage', kind: 'coverage', label: 'Coverage', tasks: [] });

    // ---- final pass: strip internal underscore fields from all lane tasks ----
    lanes.forEach((lane) => { lane.tasks = stripInternal(lane.tasks); });

    return { timeRange, steps, lanes, byNode, gaps };
  }

  function slug(s) {
    return String(s == null ? '' : s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'x';
  }

  // remove internal underscore-prefixed fields from emitted lane tasks
  function stripInternal(list) {
    return list.map((t) => {
      const o = {};
      Object.keys(t).forEach((k) => { if (k[0] !== '_') o[k] = t[k]; });
      return o;
    });
  }

  GEW.traceParser = { parse: parse, _classifyLayer: classifyLayer };
})(window);
