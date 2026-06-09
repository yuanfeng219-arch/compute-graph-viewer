# AGENTS.md

## Local Rules

- Prefix shell commands with `rtk`.
- This folder is a static PTO Train module. Prefer direct HTML, CSS, and JS edits; no build step is required.
- Reuse PTO design-system tokens and components from `../../pto-design-system/`. Keep non-graph UI colors, spacing, borders, type, and radii token-derived.
- Keep train module top headers transparent by default. Do not add a filled header background or a page-level margin/gap below the header; put spacing inside panes or content shells instead.
- Keep `training-mental-model.html`, `training-mental-model.css`, and `training-mental-model.js` aligned: nav `data-visual-target`, section `data-visual-section`, and `visuals` keys must match.
- For training facts, prefer Ascend official documentation plus local `MindSpeed-LLM-master/examples` scripts. If adding model parameters, include short user-facing annotations explaining why the value matters.
- After layout changes, verify `training-mental-model.html` as a static browser page at desktop and mobile widths.
