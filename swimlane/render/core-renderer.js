import { ViewportState } from './viewport-state.js';
import { computeLaneOffsets, findVisibleLaneWindow } from './lane-layout.js';
import { createColorResolver } from './color-resolver.js';
import { createTaskColormap } from './task-colormap.js';

export class SwimlaneRendererCore {
  constructor(options = {}) {
    this.viewport = options.viewport || new ViewportState();
    this.colorResolver = options.colorResolver || createColorResolver({
      taskColormap: createTaskColormap(),
    });
    this.metricsForLane = options.metricsForLane || defaultMetricsForLane;
    this.dataset = null;
    this.layout = { offsets: [], totalHeight: 0 };
  }

  attachDataset(dataset) {
    this.dataset = dataset;
    const lanes = dataset?.lanes || [];
    this.layout = computeLaneOffsets(lanes, this.metricsForLane);
    this.updateVisibleWindow();
  }

  setViewportSize(width, height) {
    this.viewport.setViewportSize(width, height);
    this.updateVisibleWindow();
  }

  setScroll(scrollLeft, scrollTop) {
    this.viewport.setScroll(scrollLeft, scrollTop);
    this.updateVisibleWindow();
  }

  setScale(pxPerUnit) {
    this.viewport.setScale(pxPerUnit);
  }

  updateVisibleWindow() {
    const range = findVisibleLaneWindow(
      this.layout.offsets,
      this.viewport.camera.scrollTop,
      this.viewport.viewportHeight,
      this.viewport.overscanLanes
    );
    this.viewport.updateVisibleLaneWindow(range.start, range.end);
    return range;
  }

  getVisibleLanes() {
    return this.layout.offsets.slice(this.viewport.visibleLaneStart, this.viewport.visibleLaneEnd);
  }

  getSceneSnapshot() {
    return {
      viewport: this.viewport.getSnapshot(),
      layout: this.layout,
      visibleLanes: this.getVisibleLanes(),
      laneCount: this.dataset?.lanes?.length || 0,
    };
  }
}

function defaultMetricsForLane(lane) {
  const lineCount = Math.max(1, lane?.lineCount || 1);
  const barHeight = 16;
  const lineGap = 4;
  const topPad = 8;
  const bottomPad = 8;
  return {
    lineCount,
    barHeight,
    lineGap,
    topPad,
    bottomPad,
    laneHeight: topPad + bottomPad + lineCount * barHeight + Math.max(0, lineCount - 1) * lineGap,
  };
}
