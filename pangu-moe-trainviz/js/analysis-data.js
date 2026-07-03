const clamp01 = value => Math.max(0, Math.min(1, value));

function hash01(a, b = 0, c = 0, d = 0) {
  let h = ((a * 73856093) ^ (b * 19349663) ^ (c * 83492791) ^ (d * 2654435761)) >>> 0;
  h ^= h >>> 13;
  h = Math.imul(h, 1274126177) >>> 0;
  return h / 4294967295;
}

function quantile(values, q) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)));
  return sorted[idx];
}

function summarize(values) {
  const max = values.length ? Math.max(...values) : 0;
  const avg = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  return { max, avg, p95: quantile(values, 0.95) };
}

export function buildMoeRuntimeMetrics({
  firstMoeLayer = 4,
  lastMoeLayer = 49,
  expertsPerLayer = 80,
  capacityTokens = 192,
  step = 0,
  collapse = 0,   // 0=均衡；1=路由坍缩（少数专家吃光 token，其余空转）—— 由 load_balance_loss 驱动
} = {}) {
  const layers = [];
  const loadRatio = [];
  const assignedTokens = [];
  const acceptedTokens = [];
  const droppedTokens = [];
  const reroutedTokens = [];
  const routerProbMass = [];
  const allToAllSendTokens = [];
  const allToAllRecvTokens = [];
  const layerCount = Math.max(0, lastMoeLayer - firstMoeLayer + 1);

  for (let row = 0; row < layerCount; row++) {
    const layer = firstMoeLayer + row;
    const layerPhase = Math.sin((layer + step * 0.7) * 0.31) * 0.12;
    const rowLoads = [];
    let overloaded = 0;
    let idle = 0;
    let dropped = 0;
    let assigned = 0;
    let send = 0;
    let recv = 0;

    for (let expertId = 0; expertId < expertsPerLayer; expertId++) {
      const index = row * expertsPerLayer + expertId;
      const laneHotspot = expertId % 64 === (layer * 7 + step) % 64 ? 0.48 : 0;
      const groupHotspot = Math.floor(expertId / 32) === (layer + step) % 8 ? 0.20 : 0;
      const noise = (hash01(layer, expertId, step, 17) - 0.5) * 0.34;
      const wave = Math.sin((expertId * 0.23) + (layer * 0.41) + step) * 0.11;
      let ratio = Math.max(0, 0.64 + layerPhase + wave + noise + laneHotspot + groupHotspot);
      if (collapse > 0) {                                  // 坍缩：少数赢家专家暴增，其余被抽干
        const winner = (expertId % 41) === ((layer * 13 + 7) % 41);
        ratio = winner ? ratio + collapse * 3.0 : ratio * (1 - collapse * 0.85);
      }
      const tokens = Math.round(capacityTokens * ratio);
      const accepted = Math.min(tokens, capacityTokens);
      const droppedNow = Math.max(0, tokens - capacityTokens);
      const rerouted = Math.round(droppedNow * (0.45 + hash01(layer, expertId, step, 31) * 0.4));
      const probMass = clamp01(0.52 + ratio * 0.28 + (hash01(layer, expertId, step, 43) - 0.5) * 0.16);
      const sendTokens = Math.round(tokens * (0.72 + hash01(layer, expertId, step, 59) * 0.42));
      const recvTokens = Math.round(tokens * (0.70 + hash01(layer, expertId, step, 71) * 0.46));

      loadRatio[index] = ratio;
      assignedTokens[index] = tokens;
      acceptedTokens[index] = accepted;
      droppedTokens[index] = droppedNow;
      reroutedTokens[index] = rerouted;
      routerProbMass[index] = probMass;
      allToAllSendTokens[index] = sendTokens;
      allToAllRecvTokens[index] = recvTokens;

      rowLoads.push(ratio);
      assigned += tokens;
      dropped += droppedNow;
      send += sendTokens;
      recv += recvTokens;
      if (ratio > 1) overloaded++;
      if (tokens < capacityTokens * 0.08) idle++;
    }

    const loadSummary = summarize(rowLoads);
    layers.push({
      layer,
      row,
      expertCount: expertsPerLayer,
      assignedTokens: assigned,
      droppedTokens: dropped,
      overloadedExperts: overloaded,
      idleExperts: idle,
      maxLoadRatio: loadSummary.max,
      avgLoadRatio: loadSummary.avg,
      p95LoadRatio: loadSummary.p95,
      allToAllSendTokens: send,
      allToAllRecvTokens: recv,
      allToAllSkew: send && recv ? Math.max(send, recv) / Math.max(1, Math.min(send, recv)) : 1,
    });
  }

  return {
    schema: 'pangu.moe-runtime-metrics.mock.v1',
    step,
    firstMoeLayer,
    lastMoeLayer,
    layerCount,
    expertsPerLayer,
    capacityTokens,
    layers,
    arrays: {
      loadRatio,
      assignedTokens,
      acceptedTokens,
      droppedTokens,
      reroutedTokens,
      routerProbMass,
      allToAllSendTokens,
      allToAllRecvTokens,
    },
  };
}

export function buildMoeLoadViewModel(metrics) {
  const ratios = metrics.arrays.loadRatio;
  const summary = summarize(ratios);
  const overloaded = ratios.reduce((sum, value) => sum + (value > 1 ? 1 : 0), 0);
  const idle = metrics.arrays.assignedTokens.reduce((sum, value) => sum + (value < metrics.capacityTokens * 0.08 ? 1 : 0), 0);
  const dropped = metrics.arrays.droppedTokens.reduce((sum, value) => sum + value, 0);
  const worstLayer = metrics.layers.reduce((best, layer) => !best || layer.maxLoadRatio > best.maxLoadRatio ? layer : best, null);
  return {
    id: 'moe-load',
    title: 'MoE Load',
    meta: `L${metrics.firstMoeLayer}-L${metrics.lastMoeLayer} · ${metrics.layerCount} layers × ${metrics.expertsPerLayer} experts`,
    metrics,
    metricOptions: [
      { id: 'loadRatio', label: 'Load', array: metrics.arrays.loadRatio, unit: '× capacity' },
      { id: 'droppedTokens', label: 'Dropped', array: metrics.arrays.droppedTokens, unit: 'tokens' },
      { id: 'reroutedTokens', label: 'Reroute', array: metrics.arrays.reroutedTokens, unit: 'tokens' },
      { id: 'allToAllSendTokens', label: 'A2A send', array: metrics.arrays.allToAllSendTokens, unit: 'tokens' },
    ],
    stats: [
      { label: 'max load', value: `${summary.max.toFixed(2)}x` },
      { label: 'p95 load', value: `${summary.p95.toFixed(2)}x` },
      { label: 'overloaded', value: `${overloaded}` },
      { label: 'idle', value: `${idle}` },
      { label: 'dropped', value: `${dropped}` },
      { label: 'worst layer', value: worstLayer ? `L${worstLayer.layer}` : '-' },
    ],
  };
}

export function buildRankLoadViewModel(timelineRuntime) {
  const ranks = (timelineRuntime?.ranks || []).map(rank => {
    let computeUs = 0;
    let commUs = 0;
    let bubbleUs = 0;
    for (const task of rank.tasks || []) {
      if (task.kind === 'F' || task.kind === 'B') computeUs += task.durUs || 0;
      else if (task.kind === 'bubble') bubbleUs += task.durUs || 0;
      else commUs += task.durUs || 0;
    }
    const totalUs = Math.max(1, timelineRuntime?.timeRangeUs?.[1] || computeUs + commUs + bubbleUs);
    return {
      rank: rank.rank,
      dp: rank.dp,
      stage: rank.stage,
      tp: rank.tp,
      ep: rank.ep ?? 0,
      group: rank.group,
      computeUs,
      commUs,
      bubbleUs,
      utilRatio: computeUs / totalUs,
      commRatio: commUs / totalUs,
      bubbleRatio: bubbleUs / totalUs,
      totalUs,
    };
  });
  const utils = ranks.map(r => r.utilRatio);
  const comms = ranks.map(r => r.commRatio);
  const worst = ranks.reduce((best, rank) => !best || rank.bubbleRatio > best.bubbleRatio ? rank : best, null);
  return {
    id: 'rank-load',
    title: 'Rank Load',
    meta: `${ranks.length} ranks · compute / comm / bubble`,
    ranks,
    stats: [
      { label: 'avg util', value: `${Math.round((summarize(utils).avg || 0) * 100)}%` },
      { label: 'max util', value: `${Math.round((summarize(utils).max || 0) * 100)}%` },
      { label: 'avg comm', value: `${Math.round((summarize(comms).avg || 0) * 100)}%` },
      { label: 'worst wait', value: worst ? `R${worst.rank}` : '-' },
    ],
  };
}

// 每 step 一份卡占用快照：collapse(0..1) 来自 load_balance_loss，>0 时令牌路由坍缩 →
// 每个 DP·PP 组里令牌堆到 TP1（winner，util→97%、comm→60%），其余卡饥饿（util→12%）。
export function buildCardLoadViewModel(rankViewModel, collapse = 0) {
  const c = clamp01(collapse);
  const cards = (rankViewModel?.ranks || []).map(rank => {
    let util = rank.utilRatio;
    let comm = rank.commRatio;
    if (c > 0) {
      const winner = (rank.ep ?? rank.tp) === 1;
      util += ((winner ? 0.97 : 0.12) - util) * c;
      comm += ((winner ? 0.60 : 0.44) - comm) * c;
    }
    util = clamp01(util);
    comm = clamp01(comm);
    return {
      cardId: rank.rank,
      label: `Card ${rank.rank}`,
      dp: rank.dp,
      stage: rank.stage,
      tp: rank.tp,
      ep: rank.ep ?? 0,
      utilRatio: util,
      commRatio: comm,
      bubbleRatio: rank.bubbleRatio,
      pressure: util,
      state: util < 0.30 ? 'alert' : (util > 0.95 || comm > 0.5) ? 'warn' : 'ok',
    };
  });
  // 组排序：先 DP·PP 组，组内按 TP→EP；DP2×PP4×TP2×EP2 会自然铺成 8 组×4 卡。
  cards.sort((a, b) => (a.dp - b.dp) || (a.stage - b.stage) || (a.tp - b.tp) || (a.ep - b.ep));
  const tpCount = Math.max(1, ...cards.map(card => card.tp + 1));
  const epCount = Math.max(1, ...cards.map(card => card.ep + 1));
  const avgUtil = summarize(cards.map(card => card.utilRatio)).avg || 0;
  const starved = cards.filter(card => card.state === 'alert').length;
  const hot = cards.filter(card => card.state === 'warn').length;
  return {
    id: 'card-load',
    title: 'Card Load',
    meta: `${cards.length} cards · local ${tpCount}TP×${epCount}EP · 每 step 占用`,
    cards,
    tpCount,
    epCount,
    stats: [
      { label: 'cards', value: `${cards.length}` },
      { label: 'avg util', value: `${Math.round(avgUtil * 100)}%` },
      { label: 'starved', value: `${starved}` },
      { label: 'hot', value: `${hot}` },
    ],
  };
}

// ===== Layer Scan：逐层 × 逐 step 异常分（单跑自身信号，非双跑 diff） =====
// 语义：分值 0..1 = 该层该 step 的「异常强度」，由单跑自身信号（grad_norm 贡献 / 激活范数 / 负载偏移）合成。
// 故事自洽于 timeseries.js 的故障链：faultStep 起，faultLayer（Router 所在 MoE 层）最先点亮，
// 误差沿层深向下游累积（accumulation），collapseStep 后在故障带饱和。
const LAYER_SCAN_OPS_MOE = [
  { op: 'mla', label: 'MLA', sem: 'sem:attention', w: 0.07 },
  { op: 'moe_prenorm', label: 'Pre-RMSNorm', sem: 'sem:norm', w: 0.05 },
  { op: 'gate', label: 'Router', sem: 'sem:gate', w: 0.30 },
  { op: 'a2a_dispatch', label: 'A2A dispatch', sem: 'sem:comm', w: 0.14 },
  { op: 'experts', label: 'Experts', sem: 'sem:moe', w: 0.24 },
  { op: 'a2a_combine', label: 'A2A combine', sem: 'sem:comm', w: 0.14 },
  { op: 'moe_residual', label: 'Post-RMSNorm', sem: 'sem:norm', w: 0.06 },
];
const LAYER_SCAN_OPS_DENSE = [
  { op: 'mla', label: 'MLA', sem: 'sem:attention', w: 0.34 },
  { op: 'dense', label: 'Dense FFN', sem: 'module:decoder', w: 0.50 },
  { op: 'norm', label: 'RMSNorm', sem: 'sem:norm', w: 0.16 },
];

export function buildLayerScanMetrics({
  steps,
  faultStep,
  collapseStep,
  totalLayers = 61,
  firstMoeLayer = 3,
  lastMoeLayer = 60,
  faultLayer = 47,
  warnThreshold = 0.35,
} = {}) {
  const stepCount = steps.length;
  const lastStep = steps[stepCount - 1];
  const scores = new Float32Array(totalLayers * stepCount);
  const layers = [];
  for (let layer = 0; layer < totalLayers; layer++) {
    const isMoe = layer >= firstMoeLayer && layer <= lastMoeLayer;
    layers.push({ layer, isMoe, isDense: !isMoe });
  }
  const gauss = (x, mu, sigma) => Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma));

  for (let layer = 0; layer < totalLayers; layer++) {
    const isMoe = layers[layer].isMoe;
    const origin = gauss(layer, faultLayer, 3.2);   // 故障层邻域（尖锐峰）
    const downstream = layer > faultLayer ? clamp01((layer - faultLayer) / Math.max(1, lastMoeLayer - faultLayer)) : 0;  // 向下游累积
    for (let si = 0; si < stepCount; si++) {
      const s = steps[si];
      const base = 0.05 + hash01(layer, s, 5) * 0.05;   // 健康基线噪声
      let score = base;
      if (isMoe && s >= faultStep) {
        const ramp = clamp01((s - faultStep) / Math.max(1, collapseStep - faultStep));                       // fault→collapse 爬升
        const post = s >= collapseStep ? clamp01((s - collapseStep) / Math.max(1, lastStep - collapseStep)) : 0;  // 坍缩后累积
        const noise = (hash01(layer, s, 23) - 0.5) * 0.06;
        score = base
          + ramp * (0.50 * origin + 0.28 * downstream)
          + post * (0.42 * origin + 0.50 * downstream + 0.08)
          + (ramp + post) * noise;
      }
      scores[layer * stepCount + si] = clamp01(score);
    }
  }

  // 逐层：首超标 step / 峰值
  const perLayer = layers.map(({ layer, isMoe }) => {
    let firstDivergeStep = null, peak = 0, peakStep = steps[0];
    for (let si = 0; si < stepCount; si++) {
      const v = scores[layer * stepCount + si];
      if (firstDivergeStep == null && v >= warnThreshold) firstDivergeStep = steps[si];
      if (v > peak) { peak = v; peakStep = steps[si]; }
    }
    return { layer, isMoe, firstDivergeStep, peak, peakStep };
  });
  // epicenter = 最早超标的层（并列取峰值更高）；无超标则取全局峰值层
  const diverged = perLayer.filter(l => l.firstDivergeStep != null);
  const epicenter = diverged.length
    ? diverged.reduce((best, l) => (l.firstDivergeStep < best.firstDivergeStep
        || (l.firstDivergeStep === best.firstDivergeStep && l.peak > best.peak)) ? l : best)
    : perLayer.reduce((best, l) => l.peak > best.peak ? l : best, perLayer[0]);
  const maxPeak = perLayer.reduce((m, l) => Math.max(m, l.peak), 0);

  const scoreAt = (layer, step) => {
    const si = steps.indexOf(step);
    if (si < 0 || layer < 0 || layer >= totalLayers) return 0;
    return scores[layer * stepCount + si];
  };
  // 层内算子分解：把该层该 step 的异常分按算子权重摊开。
  // share = 占该层异常的比例（条长）；score = 强度 0..1（颜色，主导算子≈该层分值）。
  const opBreakdown = (layer, step) => {
    const total = scoreAt(layer, step);
    const defs = layers[layer]?.isMoe ? LAYER_SCAN_OPS_MOE : LAYER_SCAN_OPS_DENSE;
    const raw = defs.map(d => ({ ...d, w2: d.w * (0.82 + hash01(layer, step, d.op.length * 7) * 0.36) }));
    const sum = raw.reduce((acc, d) => acc + d.w2, 0) || 1;
    const maxW2 = raw.reduce((m, d) => Math.max(m, d.w2), 0) || 1;
    return raw.map(d => ({
      op: d.op, label: d.label, sem: d.sem,
      share: d.w2 / sum,
      score: clamp01(total * (d.w2 / maxW2)),
    }));
  };

  return {
    schema: 'pangu.layer-scan.mock.v1',
    steps, stepCount, totalLayers, firstMoeLayer, lastMoeLayer,
    faultStep, collapseStep, faultLayer, warnThreshold,
    scores, layers, perLayer, epicenter, maxPeak,
    scoreAt, opBreakdown,
    title: 'Layer Scan',
    meta: `L0-L${lastMoeLayer} × ${stepCount} steps · 逐层异常分（单跑信号）`,
    stats: [
      { label: '首问题层', value: `L${epicenter.layer}` },
      { label: '首超标', value: epicenter.firstDivergeStep != null ? `step ${epicenter.firstDivergeStep}` : '—' },
      { label: '峰值分', value: maxPeak.toFixed(2) },
      { label: '超标层数', value: `${diverged.length}` },
      { label: '故障算子', value: 'Router' },
      { label: 'MoE 层', value: `L${firstMoeLayer}-L${lastMoeLayer}` },
    ],
  };
}

function compDuration(stage, type, microbatch, forwardBaseUs, backwardBaseUs) {
  const base = type === 'F' ? forwardBaseUs : backwardBaseUs;
  const stageWeight = stage === 0 ? 0.9 : 1.08;
  const microbatchWeight = 0.74 + hash01(stage, type === 'F' ? 1 : 2, microbatch) * 0.62;
  return base * stageWeight * microbatchWeight;
}

function simulate1F1BSchedule(stageCount, microbatches, forwardBaseUs, backwardBaseUs, ppCommUs) {
  const compF = Array.from({ length: stageCount }, () => Array(microbatches).fill(null));
  const compB = Array.from({ length: stageCount }, () => Array(microbatches).fill(null));
  const stageFree = Array(stageCount).fill(0);
  const seq = [];
  const idx = Array(stageCount).fill(0);
  const stageOps = Array.from({ length: stageCount }, () => []);

  for (let stage = 0; stage < stageCount; stage++) {
    const warmupCount = stageCount - 1 - stage;
    const steadyCount = microbatches - warmupCount;
    const ops = [];
    let fMicro = 0;
    let bMicro = 0;
    for (let k = 0; k < warmupCount; k++) ops.push({ type: 'F', microbatch: fMicro++ });
    for (let k = 0; k < steadyCount; k++) {
      ops.push({ type: 'F', microbatch: fMicro++ });
      ops.push({ type: 'B', microbatch: bMicro++ });
    }
    while (bMicro < microbatches) ops.push({ type: 'B', microbatch: bMicro++ });
    seq.push(ops);
  }

  let scheduled = 0;
  const total = stageCount * microbatches * 2;
  let guard = 0;
  while (scheduled < total && guard++ < 20000) {
    let progressed = false;
    for (let stage = 0; stage < stageCount; stage++) {
      if (idx[stage] >= seq[stage].length) continue;
      const op = seq[stage][idx[stage]];
      let dep = 0;
      if (op.type === 'F') {
        if (stage > 0) {
          const upstream = compF[stage - 1][op.microbatch];
          if (!upstream) continue;
          dep = upstream.end + ppCommUs;
        }
      } else if (stage < stageCount - 1) {
        const downstream = compB[stage + 1][op.microbatch];
        if (!downstream) continue;
        dep = downstream.end + ppCommUs;
      } else {
        const localForward = compF[stage][op.microbatch];
        if (!localForward) continue;
        dep = localForward.end;
      }
      const start = Math.max(stageFree[stage], dep);
      const dur = compDuration(stage, op.type, op.microbatch, forwardBaseUs, backwardBaseUs);
      const rec = { start, end: start + dur, type: op.type, m: op.microbatch };
      (op.type === 'F' ? compF : compB)[stage][op.microbatch] = rec;
      stageOps[stage].push(rec);
      stageFree[stage] = rec.end;
      idx[stage]++;
      scheduled++;
      progressed = true;
    }
    if (!progressed) break;
  }

  let totalUs = 0;
  for (let stage = 0; stage < stageCount; stage++) {
    for (let microbatch = 0; microbatch < microbatches; microbatch++) {
      totalUs = Math.max(totalUs, compF[stage][microbatch]?.end || 0, compB[stage][microbatch]?.end || 0);
    }
  }
  return { compF, compB, stageOps, totalUs: totalUs + ppCommUs };
}

export function buildSimulated1F1BRuntime({
  dp = 2,
  pp = 4,
  tp = 2,
  ep = 2,
  microbatches = 8,
  forwardBaseUs = 420,
  backwardBaseUs = 780,
  ppCommUs = 72,
  tpCommUs = 58,
  epCommUs = 96,
  dpCommUs = 64,
  stageRanges = [[0, 12], [13, 25], [26, 37], [38, 49]],
} = {}) {
  const { compF, compB, stageOps, totalUs } = simulate1F1BSchedule(pp, microbatches, forwardBaseUs, backwardBaseUs, ppCommUs);
  const ranks = [];
  for (let dpIndex = 0; dpIndex < dp; dpIndex++) {
    for (let stage = 0; stage < pp; stage++) {
      for (let tpIndex = 0; tpIndex < tp; tpIndex++) {
        for (let epIndex = 0; epIndex < ep; epIndex++) {
          const rank = (((dpIndex * pp + stage) * tp + tpIndex) * ep) + epIndex;
          const jitter = offset => (((rank * 131 + offset * 977) % 100) / 100 - 0.5);
          const rankKey = variant => `rank:r${rank}:v${variant % 7}`;
          const range = stageRanges[stage] || stageRanges[stageRanges.length - 1] || [0, 0];
          const tasks = [];

        const emit = (rec, type, microbatch) => {
          if (!rec) return;
          const start = rec.start;
          const dur = rec.end - rec.start;
          const straggler = hash01(rank, type === 'F' ? 91 : 97, microbatch) > 0.82 ? 0.06 : 0;
          const load = Math.min(1, 0.70 + hash01(rank, type === 'F' ? 11 : 17, microbatch) * 0.26 + straggler);
          const visualDur = dur * load;
          tasks.push({
            startUs: start,
            durUs: visualDur,
            slotStartUs: start,
            slotDurUs: dur,
            rankLoad: load,
            kind: type,
            microbatch,
            status: 'ok',
            label: `${type} m${microbatch}`,
            opName: `${type === 'F' ? 'Forward' : 'Backward'} · micro ${microbatch} · L${range[0]}-L${range[1]} · 填充 ${(load * 100) | 0}%`,
          });
          tasks.push({
            startUs: start + dur * 0.58,
            durUs: tpCommUs * (0.72 + jitter(5) * 0.5),
            kind: 'tp',
            colorKey: 'sem:attention',
            rankColorKey: rankKey(5),
            microbatch,
            status: 'overlap',
            label: 'AR',
            opName: `TP All-Reduce · ${tp}-rank TP group · micro ${microbatch}`,
          });
          tasks.push({
            startUs: start + dur * 0.26,
            durUs: epCommUs * (0.7 + jitter(6) * 0.6),
            kind: 'ep',
            colorKey: 'sem:comm',
            rankColorKey: rankKey(6),
            microbatch,
            status: 'overlap',
            label: 'A2A',
            opName: `EP All-to-All · ${ep}-rank expert group · token dispatch/combine · micro ${microbatch}`,
          });
        };

        for (let microbatch = 0; microbatch < microbatches; microbatch++) {
          emit(compF[stage][microbatch], 'F', microbatch);
          emit(compB[stage][microbatch], 'B', microbatch);
        }
        for (let microbatch = 0; microbatch < microbatches; microbatch++) {
          const pushPp = (at, label) => tasks.push({
            startUs: Math.max(0, at),
            durUs: ppCommUs * 0.8,
            kind: 'pp',
            colorKey: 'sem:mlp',
            rankColorKey: rankKey(4),
            microbatch,
            status: 'ok',
            label: 'PP',
            opName: label,
          });
          if (stage > 0 && compF[stage][microbatch]) pushPp(compF[stage][microbatch].start - ppCommUs * 0.8, `PP recv activation ← PP${stage - 1} · micro ${microbatch}`);
          if (stage < pp - 1 && compF[stage][microbatch]) pushPp(compF[stage][microbatch].end, `PP send activation → PP${stage + 1} · micro ${microbatch}`);
          if (stage < pp - 1 && compB[stage][microbatch]) pushPp(compB[stage][microbatch].start - ppCommUs * 0.8, `PP recv gradient ← PP${stage + 1} · micro ${microbatch}`);
          if (stage > 0 && compB[stage][microbatch]) pushPp(compB[stage][microbatch].end, `PP send gradient → PP${stage - 1} · micro ${microbatch}`);
          if (compB[stage][microbatch]) {
            tasks.push({
              startUs: compB[stage][microbatch].end - dpCommUs * 0.38,
              durUs: dpCommUs * (0.78 + jitter(8) * 0.42),
              kind: 'dp',
              colorKey: 'sem:head',
              rankColorKey: rankKey(2),
              microbatch,
              status: 'overlap',
              label: 'DP',
              opName: `DP gradient sync · D0/D1 replica group · PP${stage} TP${tpIndex} EP${epIndex} · micro ${microbatch}`,
            });
          }
        }

        const ops = stageOps[stage];
        const bubbleThreshold = ppCommUs * 0.9;
        let prev = 0;
        ops.forEach(op => {
          if (op.start - prev > bubbleThreshold) {
            tasks.push({
              startUs: prev,
              durUs: op.start - prev,
              kind: 'bubble',
              status: 'wait',
              colorKey: 'pipeline:bubble',
              rankColorKey: rankKey(3),
              label: 'bubble',
              opName: `Pipeline bubble · PP${stage} 等待跨 stage 依赖（${prev < 1 ? 'warmup 填充' : '1F1B 空泡'}）`,
            });
          }
          prev = Math.max(prev, op.end);
        });
        if (totalUs - prev > bubbleThreshold) {
          tasks.push({
            startUs: prev,
            durUs: totalUs - prev,
            kind: 'bubble',
            status: 'wait',
            colorKey: 'pipeline:bubble',
            rankColorKey: rankKey(3),
            label: 'bubble',
            opName: `Pipeline bubble · PP${stage} drain 尾部空闲（流水线排空）`,
          });
        }

          ranks.push({
            rank,
            dp: dpIndex,
            stage,
            tp: tpIndex,
            ep: epIndex,
            label: `Rank ${rank}`,
            group: `D${dpIndex}·PP${stage}·TP${tpIndex}·EP${epIndex}`,
            tasks,
          });
        }
      }
    }
  }
  return { config: { dp, pp, tp, ep, microbatches }, timeRangeUs: [0, totalUs], ranks };
}
