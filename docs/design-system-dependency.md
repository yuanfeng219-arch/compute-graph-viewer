# PTO Design System Dependency

PTO treats `pto-design-system` as the canonical source for design-system tokens, shared CSS, reusable patterns, preview pages, and agent-facing skill documentation.

Canonical repository:

```text
https://github.com/yinyucheng0601/pto-design-system.git
```

Online preview:

```text
https://yinyucheng0601.github.io/pto-design-system/design-system-preview.html
```

## Local Layout

```text
pto/
├── vendor/pto-design-system/  # Git submodule; canonical source pinned by commit
└── scripts/sync-design-system.mjs
```

Do not edit generated mirrors as the source of truth for design-system changes. Edit and push `pto-design-system`. Runtime pages should reference `vendor/pto-design-system/...` directly unless a page has a specific compatibility reason to use a mirror.

## Initialize Or Update The Submodule

```bash
git submodule update --init --recursive vendor/pto-design-system
```

Update to the latest canonical design system:

```bash
git -C vendor/pto-design-system fetch origin main
git -C vendor/pto-design-system checkout main
git -C vendor/pto-design-system pull --ff-only origin main
```

## Sync Into PTO

Preview what would be copied:

```bash
node scripts/sync-design-system.mjs
```

Write the canonical files into `design-system-share/`:

```bash
node scripts/sync-design-system.mjs --write
```

Overwrite matching legacy mirror files under `tokens/`, `css`, `patterns`, and `assets/` only when an old page still needs those paths:

```bash
node scripts/sync-design-system.mjs --write --legacy-mirrors
```

Make `design-system-share/` a clean mirror of the canonical package:

```bash
node scripts/sync-design-system.mjs --write --clean-share
```

Use `--clean-share` only after checking `git status`, because it removes files in `design-system-share/` that are not present in `vendor/pto-design-system/`.

## Policy

- `vendor/pto-design-system/` is the dependency source.
- `design-system-share/` is an optional generated/shareable package for AI tools, not a required checked-in runtime dependency.
- `tokens/`, `css/`, `patterns/`, and `assets/` legacy mirrors should be generated only when compatibility requires them.
- Runtime pages should load CSS, tokens, and patterns from `vendor/pto-design-system/...`.
- New PTO pages should prefer local repository paths, not GitHub Pages URLs, so local development remains offline-capable and version-pinned.
- Avoid hand-copying design-system folders; use `scripts/sync-design-system.mjs` so drift is visible.
