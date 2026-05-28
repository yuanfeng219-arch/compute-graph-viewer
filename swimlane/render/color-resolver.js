const DEFAULT_UI_PALETTE = Object.freeze({
  neutralTask: '#76716a',
  neutralMuted: '#5f5a54',
  interactivePrimary: '#4369ef',
  diagnosticGap: '#d27a52',
});

export function createColorResolver(options = {}) {
  const uiPalette = {
    ...DEFAULT_UI_PALETTE,
    ...(options.uiPalette || {}),
  };
  const taskColormap = options.taskColormap || null;

  return {
    uiPalette,
    taskColormap,
    colorForLaneKind(kind) {
      if (taskColormap?.colorForLaneKind) {
        return taskColormap.colorForLaneKind(kind);
      }
      if (kind === 'fake') return uiPalette.neutralMuted;
      return uiPalette.neutralTask;
    },
    colorForTask(task, mode = 'engine') {
      if (taskColormap?.colorForTask) {
        return taskColormap.colorForTask(task, mode);
      }
      if (mode === 'gap') {
        return uiPalette.diagnosticGap;
      }
      return this.colorForLaneKind(task?.laneKind);
    },
  };
}
