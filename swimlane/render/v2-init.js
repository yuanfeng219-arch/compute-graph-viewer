import { SwimlaneRendererCore } from './core-renderer.js';
import { createTaskColormap } from './task-colormap.js';
import { createColorResolver } from './color-resolver.js';

const BAR_HEIGHT = 16;
const LINE_GAP = 4;
const TOP_PAD = 8;
const BOTTOM_PAD = 8;

function computeLaneHeight(lane) {
  const lineCount = Math.max(1, lane?.tasks?.length || 1);
  return TOP_PAD + BOTTOM_PAD + lineCount * BAR_HEIGHT + Math.max(0, lineCount - 1) * LINE_GAP;
}

function computeTotalDuration(lanes) {
  let maxEnd = 0;
  for (const lane of lanes) {
    for (const task of lane.tasks || []) {
      const end = (task.startUs || 0) + (task.durationUs || 0);
      if (end > maxEnd) maxEnd = end;
    }
  }
  return maxEnd || 1;
}

export function initV2Canvas(canvasEl, dataset) {
  const colorResolver = createColorResolver({ taskColormap: createTaskColormap() });
  const renderer = new SwimlaneRendererCore({ colorResolver });
  renderer.attachDataset(dataset);
  renderer.setViewportSize(canvasEl.width, canvasEl.height);

  const lanes = dataset?.lanes || [];
  const totalDurationUs = computeTotalDuration(lanes);
  const canvasW = canvasEl.width;
  const canvasH = canvasEl.height;

  function draw() {
    const ctx = canvasEl.getContext('2d');
    ctx.clearRect(0, 0, canvasW, canvasH);

    let cursorY = 0;
    lanes.forEach((lane, laneIndex) => {
      const laneH = computeLaneHeight(lane);
      const bg = laneIndex % 2 === 0 ? '#1b1714' : '#22201d';
      ctx.fillStyle = bg;
      ctx.fillRect(0, cursorY, canvasW, laneH);

      ctx.fillStyle = '#c2b7aa';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillText(lane.threadName || lane.id || `Lane ${laneIndex}`, 8, cursorY + laneH / 2 + 4);

      const tasks = lane.tasks || [];
      tasks.forEach((task, taskIndex) => {
        const x = (task.startUs || 0) / totalDurationUs * canvasW;
        const w = Math.max(1, (task.durationUs || 0) / totalDurationUs * canvasW);
        const y = cursorY + TOP_PAD + taskIndex * (BAR_HEIGHT + LINE_GAP);
        const color = colorResolver.colorForTask(task, 'engine');
        ctx.fillStyle = color;
        ctx.fillRect(x, y, w, BAR_HEIGHT);
      });

      cursorY += laneH;
    });
  }

  return {
    renderer,
    draw,
    setScroll(x, y) { renderer.setScroll(x, y); draw(); },
    setScale(s) { renderer.setScale(s); draw(); },
  };
}
