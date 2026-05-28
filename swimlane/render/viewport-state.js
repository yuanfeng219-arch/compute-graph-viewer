export const DEFAULT_DENSITY_MODE = 'analysis';
export const DEFAULT_OVERSCAN_LANES = 6;
export const DEFAULT_CAMERA = Object.freeze({
  timeStart: 0,
  timeEnd: 1,
  pxPerUnit: 8,
  scrollLeft: 0,
  scrollTop: 0,
  densityMode: DEFAULT_DENSITY_MODE,
});

export class ViewportState {
  constructor(initial = {}) {
    this.viewportWidth = initial.viewportWidth || 0;
    this.viewportHeight = initial.viewportHeight || 0;
    this.camera = {
      ...DEFAULT_CAMERA,
      ...initial.camera,
    };
    this.visibleLaneStart = 0;
    this.visibleLaneEnd = 0;
    this.overscanLanes = initial.overscanLanes || DEFAULT_OVERSCAN_LANES;
  }

  setViewportSize(width, height) {
    this.viewportWidth = Math.max(0, width || 0);
    this.viewportHeight = Math.max(0, height || 0);
  }

  setScroll(scrollLeft, scrollTop) {
    this.camera.scrollLeft = Math.max(0, scrollLeft || 0);
    this.camera.scrollTop = Math.max(0, scrollTop || 0);
  }

  setScale(pxPerUnit) {
    this.camera.pxPerUnit = Math.max(0.001, pxPerUnit || this.camera.pxPerUnit);
  }

  setTimeWindow(timeStart, timeEnd) {
    this.camera.timeStart = Math.max(0, timeStart || 0);
    this.camera.timeEnd = Math.max(this.camera.timeStart, timeEnd || this.camera.timeStart);
  }

  setDensityMode(mode) {
    this.camera.densityMode = mode || DEFAULT_DENSITY_MODE;
  }

  updateVisibleLaneWindow(startIndex, endIndex) {
    this.visibleLaneStart = Math.max(0, startIndex || 0);
    this.visibleLaneEnd = Math.max(this.visibleLaneStart, endIndex || this.visibleLaneStart);
  }

  getTimeWindow() {
    const start = this.camera.scrollLeft / this.camera.pxPerUnit;
    const span = this.viewportWidth / this.camera.pxPerUnit;
    return {
      start,
      end: start + span,
      span,
    };
  }

  getSnapshot() {
    return {
      viewportWidth: this.viewportWidth,
      viewportHeight: this.viewportHeight,
      visibleLaneStart: this.visibleLaneStart,
      visibleLaneEnd: this.visibleLaneEnd,
      overscanLanes: this.overscanLanes,
      camera: { ...this.camera },
      timeWindow: this.getTimeWindow(),
    };
  }
}
