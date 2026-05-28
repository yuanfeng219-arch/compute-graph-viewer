# PTO Todo

## Design System Migration

### Shared System Layer

- `[done]` Target UI spec page is established in `/Users/yin/pto/design-system-preview.html`.
- `[done]` New-module workflow skill exists: `/Users/yin/.codex/skills/pto-new-module-design-system/`.
- `[done]` External-module onboarding skill exists: `/Users/yin/.codex/skills/pto-module-onboarding-to-design-system/`.
- `[done]` Shared button system has been narrowed to the current target variants and matching rules.
- `[done]` Shared `stat-chip` tokens and shared zoom-control pattern were added in:
  - `/Users/yin/pto/tokens/components.css`
  - `/Users/yin/pto/css/style.css`
- `[todo]` Continue moving repeated shared UI primitives out of module CSS and into shared tokens/classes.

### Module Status

- `/Users/yin/pto/swimlane`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/swimlane/existingUI_preview.html`
  - `[done]` Major button-system direction and target patterns were defined in the shared preview page.
  - `[in-progress]` Full migrate to shared system is not finished.
  - `[todo]` Complete real module migration against the approved target patterns.

- `/Users/yin/pto/pass-ir`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/pass-ir/existingUI_preview.html`
  - `[done]` Graph node target pattern was defined from `pass-ir/node-preview.html` and folded into `/Users/yin/pto/design-system-preview.html`.
  - `[in-progress]` Some shared pattern adoption exists, but full migration is not complete.
  - `[todo]` Continue migrating module UI to shared button/toggle/card patterns.

- `/Users/yin/pto/op-ide-assistant`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/op-ide-assistant/existingUI_preview.html`
  - `[todo]` Start actual module migration using the shared system mapping.

- `/Users/yin/pto/execution-overlay`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/execution-overlay/existingUI_preview.html`
  - `[done]` Component preview exists: `/Users/yin/pto/execution-overlay/component-preview.html`
  - `[done]` Playback CTA moved from old `.btn-primary` to shared `.btn.btn-solid`
  - `[done]` Filter strip moved to shared selected-state naming
  - `[done]` Viewport/panel neutral surfaces started consuming shared tokens
  - `[done]` Graph-viz-heavy parts were approved as data-viz exempt and will not enter the shared system
  - `[todo]` Keep only neutral shell/button/toggle pieces on the shared path; do not try to normalize `eo-node` special graph skin into shared UI

- `/Users/yin/pto/mem_viewer`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/mem_viewer/existingUI_preview.html`
  - `[in-progress]` Used as the reference source for the new CANN memory-viewer test module
  - `[todo]` Audit whether any of its card/shell pieces can be directly migrated to shared system without affecting memory-tier viz encoding

- `/Users/yin/pto/model-architecture`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/model-architecture/existingUI_preview.html`
  - `[done]` Earlier hard-coded color cleanup was completed for this module
  - `[todo]` Complete full component migration to shared system

- `/Users/yin/pto/graph-prototype-lab`
  - `[done]` Existing UI preview exists: `/Users/yin/pto/graph-prototype-lab/existingUI_preview.html`
  - `[done]` Earlier hard-coded color cleanup was completed for this module
  - `[todo]` Complete full component migration to shared system

- `/Users/yin/pto/cann-910b-mem-viewer`
  - `[done]` Test module created to validate skill 1
  - `[done]` Formal shell exists:
    - `/Users/yin/pto/cann-910b-mem-viewer/index.html`
    - `/Users/yin/pto/cann-910b-mem-viewer/styles.css`
  - `[done]` Preview gate exists for new memory-tier visual:
    - `/Users/yin/pto/cann-910b-mem-viewer/component-preview.html`
  - `[done]` Memory architecture pattern was approved and folded into `/Users/yin/pto/design-system-preview.html`
  - `[in-progress]` Shared shell/components are applied; formal module still needs to consume the approved shared preview pattern
  - `[todo]` Replace the formal module placeholder architecture with the approved shared memory-tier pattern

### Important / Not Urgent

- `execution-overlay` graph-viz special styles are approved exempt; keep them out of shared UI.
- After more modules migrate, add a lightweight project-level dashboard view for migration status so progress is visible without reading source files.
