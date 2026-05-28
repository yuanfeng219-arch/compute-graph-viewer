Swimlane v2 renderer scaffolding.

This directory intentionally contains framework-free ES modules so the legacy
DOM implementation can adopt the new pipeline incrementally.

Planned module roles:

- `viewport-state.js`: camera, scroll, zoom, and density state
- `lane-layout.js`: lane offsets and visible lane window calculations
- `task-colormap.js`: visualization semantics for swimlane task/lane coloring
- `color-resolver.js`: UI-vs-visualization color boundary wrapper
- `hit-test.js`: lane-local binary search helpers
- `core-renderer.js`: orchestration shell for future canvas rendering
