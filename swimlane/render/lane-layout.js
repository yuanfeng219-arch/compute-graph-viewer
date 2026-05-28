export function computeLaneOffsets(lanes, metricsForLane) {
  const offsets = [];
  let cursorTop = 0;

  lanes.forEach((lane, index) => {
    const metrics = metricsForLane(lane, index);
    offsets.push({
      laneId: lane.threadName || lane.id || String(index),
      lane,
      index,
      top: cursorTop,
      height: metrics.laneHeight,
      metrics,
    });
    cursorTop += metrics.laneHeight;
  });

  return {
    totalHeight: cursorTop,
    offsets,
  };
}

export function findVisibleLaneWindow(offsets, scrollTop, viewportHeight, overscan = 0) {
  const viewTop = Math.max(0, scrollTop || 0);
  const viewBottom = viewTop + Math.max(0, viewportHeight || 0);

  let start = 0;
  while (start < offsets.length && offsets[start].top + offsets[start].height < viewTop) {
    start += 1;
  }

  let end = start;
  while (end < offsets.length && offsets[end].top <= viewBottom) {
    end += 1;
  }

  return {
    start: Math.max(0, start - overscan),
    end: Math.min(offsets.length, end + overscan),
  };
}
