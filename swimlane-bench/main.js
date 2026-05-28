const dom = {
  taskCount: document.getElementById('benchTaskCount'),
  laneCount: document.getElementById('benchLaneCount'),
  span: document.getElementById('benchSpan'),
  seed: document.getElementById('benchSeed'),
  runBtn: document.getElementById('benchRunBtn'),
  downloadBtn: document.getElementById('benchDownloadBtn'),
  preview: document.getElementById('benchPreview'),
  metrics: document.getElementById('benchMetrics'),
  log: document.getElementById('benchLog'),
};

let lastDataset = null;

const metricDefinitions = [
  ['Dataset Build', 'datasetBuildMs'],
  ['Lane Index', 'laneIndexMs'],
  ['Search Index', 'searchIndexMs'],
  ['Visible Slice', 'visibleSliceMs'],
  ['JSON Size', 'jsonSizeMb'],
  ['Estimated DOM', 'estimatedDomNodes'],
];

function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function generateSyntheticDataset({ taskCount, laneCount, span, seed }) {
  const random = mulberry32(seed);
  const lanes = Array.from({ length: laneCount }, (_, index) => ({
    threadName: `lane_${String(index).padStart(4, '0')}`,
    laneKind: index % 3 === 0 ? 'aic' : index % 3 === 1 ? 'aiv' : 'aicpu',
    tasks: [],
  }));

  for (let index = 0; index < taskCount; index += 1) {
    const lane = lanes[index % laneCount];
    const start = Math.floor(random() * Math.max(span - 80, 1));
    const duration = Math.max(4, Math.floor(random() * 160));
    lane.tasks.push({
      id: `task_${index}`,
      rawName: `kernel_${index}`,
      label: index % 5 === 0 ? 'Query-Linear' : index % 5 === 1 ? 'Weight-Linear' : 'Compute',
      threadName: lane.threadName,
      laneKind: lane.laneKind,
      seqNo: index % 16,
      relTs: start,
      relEnd: start + duration,
      dur: duration,
      inputRawMagic: [`in_${index % 256}`],
      outputRawMagic: [`out_${index % 256}`],
    });
  }

  lanes.forEach((lane) => {
    lane.tasks.sort((a, b) => a.relTs - b.relTs);
  });

  return {
    meta: { taskCount, laneCount, span, seed, generatedAt: new Date().toISOString() },
    lanes,
  };
}

function buildSearchIndex(dataset) {
  const index = new Map();
  dataset.lanes.forEach((lane) => {
    lane.tasks.forEach((task) => {
      const text = `${task.rawName} ${task.label} ${task.threadName} ${task.seqNo} ${task.inputRawMagic.join(' ')} ${task.outputRawMagic.join(' ')}`.toLowerCase();
      index.set(task.id, text);
    });
  });
  return index;
}

function buildLaneSlices(dataset, viewport) {
  const timeStart = viewport.timeStart;
  const timeEnd = viewport.timeEnd;
  return dataset.lanes.map((lane) => {
    let start = 0;
    let end = lane.tasks.length;
    while (start < lane.tasks.length && lane.tasks[start].relEnd < timeStart) start += 1;
    while (end > start && lane.tasks[end - 1].relTs > timeEnd) end -= 1;
    return end - start;
  });
}

function renderMetrics(metrics) {
  dom.metrics.innerHTML = metricDefinitions.map(([label, key]) => {
    return `
      <div class="bench-metric">
        <div class="bench-metric-label">${label}</div>
        <div class="bench-metric-value">${metrics[key]}</div>
      </div>`;
  }).join('');
}

function appendLog(entry) {
  const node = document.createElement('div');
  node.className = 'bench-log-entry';
  node.textContent = `[${new Date().toLocaleTimeString()}] ${entry}`;
  dom.log.prepend(node);
}

function formatMetrics(result) {
  return {
    datasetBuildMs: `${result.datasetBuildMs.toFixed(1)} ms`,
    laneIndexMs: `${result.laneIndexMs.toFixed(1)} ms`,
    searchIndexMs: `${result.searchIndexMs.toFixed(1)} ms`,
    visibleSliceMs: `${result.visibleSliceMs.toFixed(1)} ms`,
    jsonSizeMb: `${result.jsonSizeMb.toFixed(2)} MB`,
    estimatedDomNodes: result.estimatedDomNodes.toLocaleString(),
  };
}

function benchmarkScenario(config) {
  const t0 = performance.now();
  const dataset = generateSyntheticDataset(config);
  const t1 = performance.now();

  const laneIndexStart = performance.now();
  const laneIndex = dataset.lanes.map((lane) => ({
    name: lane.threadName,
    taskCount: lane.tasks.length,
    firstTs: lane.tasks[0]?.relTs ?? 0,
    lastTs: lane.tasks[lane.tasks.length - 1]?.relEnd ?? 0,
  }));
  const laneIndexEnd = performance.now();

  const searchIndexStart = performance.now();
  const searchIndex = buildSearchIndex(dataset);
  const searchIndexEnd = performance.now();

  const visibleSliceStart = performance.now();
  const visibleCounts = buildLaneSlices(dataset, {
    timeStart: config.span * 0.25,
    timeEnd: config.span * 0.45,
  });
  const visibleSliceEnd = performance.now();

  const json = JSON.stringify(dataset);
  const jsonSizeMb = new Blob([json]).size / (1024 * 1024);
  const estimatedDomNodes = dataset.lanes.length * 2 + config.taskCount * 2 + visibleCounts.reduce((sum, count) => sum + count, 0);

  return {
    dataset,
    laneIndex,
    searchIndex,
    metrics: formatMetrics({
      datasetBuildMs: t1 - t0,
      laneIndexMs: laneIndexEnd - laneIndexStart,
      searchIndexMs: searchIndexEnd - searchIndexStart,
      visibleSliceMs: visibleSliceEnd - visibleSliceStart,
      jsonSizeMb,
      estimatedDomNodes,
    }),
  };
}

function readConfig() {
  return {
    taskCount: Number(dom.taskCount.value),
    laneCount: Number(dom.laneCount.value),
    span: Number(dom.span.value),
    seed: Number(dom.seed.value),
  };
}

function renderPreview(dataset, metrics) {
  const preview = {
    meta: dataset.meta,
    firstLanes: dataset.lanes.slice(0, 3).map((lane) => ({
      threadName: lane.threadName,
      laneKind: lane.laneKind,
      taskCount: lane.tasks.length,
      firstTasks: lane.tasks.slice(0, 3),
    })),
    metrics,
  };
  dom.preview.textContent = JSON.stringify(preview, null, 2);
}

function downloadDataset() {
  if (!lastDataset) return;
  const blob = new Blob([JSON.stringify(lastDataset, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `synthetic_swimlane_${lastDataset.meta.taskCount}_${lastDataset.meta.laneCount}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function run() {
  const config = readConfig();
  const result = benchmarkScenario(config);
  lastDataset = result.dataset;
  renderMetrics(result.metrics);
  renderPreview(result.dataset, result.metrics);
  appendLog(`Generated ${config.taskCount.toLocaleString()} tasks across ${config.laneCount.toLocaleString()} lanes.`);
}

dom.runBtn.addEventListener('click', run);
dom.downloadBtn.addEventListener('click', downloadDataset);

run();
