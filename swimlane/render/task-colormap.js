// Swimlane task/lane coloring belongs to the visualization colormap layer,
// not the base design-system token layer. These defaults intentionally live
// next to the renderer so they can later be swapped to shared colormap.js
// adapters without coupling semantic task colors to UI surface tokens.

const STITCH_COLORS = ['#7b57bf', '#4d79d4', '#4da56d', '#d98f55', '#45b5c4', '#c86aa0', '#00a6fb', '#eab308'];

const LABEL_COLORS = {
  'Prolog-Quant': '#9b6bde',
  'Query-Linear': '#7b57bf',
  'Query-Dequant': '#4d79d4',
  'Query-Hadamard': '#6f63c5',
  'Weight-Linear': '#4da56d',
  'Key-Linear': '#d98f55',
  'Key-Hadamard': '#e39b63',
  'Key-LayerNorm': '#c86aa0',
  'Key-Rope2D': '#45b5c4',
  fake: '#6f6a64',
  unknown: '#6f6a64',
};

const LANE_KIND_COLORS = {
  fake: '#6f6a64',
  aic: '#7b57bf',
  aiv: '#4d79d4',
  aicpu: '#4da56d',
  other: '#8c847c',
};

function stableHash(input) {
  let hash = 2166136261;
  const value = String(input || '');
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hueColor(input, saturation, lightness) {
  const hash = stableHash(input);
  const hue = hash % 360;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

export function createTaskColormap() {
  return {
    colorForLaneKind(kind) {
      return LANE_KIND_COLORS[kind] || LANE_KIND_COLORS.other;
    },
    colorForTask(task, mode = 'semantic') {
      if (mode === 'stitch') {
        const index = Math.abs(task?.seqNo || 0) % STITCH_COLORS.length;
        return STITCH_COLORS[index];
      }
      if (mode === 'engine') {
        return this.colorForLaneKind(task?.laneKind);
      }
      if (mode === 'subgraph') {
        const key = task?.subgraphKey || task?.subGraphId || task?.leafHash || task?.label;
        return hueColor(key, 58, 56);
      }
      return LABEL_COLORS[task?.label] || hueColor(task?.label, 54, 54);
    },
  };
}
