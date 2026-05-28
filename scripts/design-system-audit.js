#!/usr/bin/env node
/**
 * PTO Design System Audit
 *
 * Checks:
 *   1. Hardcoded colors in module CSS (rgba/hsl/#hex outside tokens/)
 *   2. Unused design tokens (defined but never consumed)
 *   3. Module @import coverage (foundation / semantic / components)
 *
 * Usage:
 *   node scripts/design-system-audit.js
 *   node scripts/design-system-audit.js --check colors
 *   node scripts/design-system-audit.js --check tokens
 *   node scripts/design-system-audit.js --check imports
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

// Dirs to skip entirely
const SKIP_DIRS = new Set(['devui', 'archive', 'node_modules', 'scripts', '.git', 'assets', 'data', 'low-fi', 'pencil', 'tools', 'claude', '业务理解']);

// Token source dir — excluded from hardcode checks (they define raw values)
const TOKEN_DIR = path.join(ROOT, 'tokens');

// Token files
const TOKEN_FILES = ['foundation.css', 'semantic.css', 'components.css'].map(f => path.join(TOKEN_DIR, f));

// css/style.css is the shared scene token layer — also excluded from hardcode check
const STYLE_CSS = path.join(ROOT, 'css', 'style.css');

// ─── Utility ───────────────────────────────────────────────────────────────

function collectCssFiles(dir, results = []) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return results; }
  for (const entry of entries) {
    if (SKIP_DIRS.has(entry.name)) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) collectCssFiles(full, results);
    else if (entry.name.endsWith('.css')) results.push(full);
  }
  return results;
}

function readLines(file) {
  return fs.readFileSync(file, 'utf8').split('\n');
}

function relPath(p) {
  return path.relative(ROOT, p);
}

// ─── Check 1: Hardcoded Colors ─────────────────────────────────────────────

// Patterns that indicate raw color values in module CSS
// We allow them in token files (they're defining the values)
// We flag: rgb(...), rgba(...), hsl(...), hsla(...), #hex
// We do NOT flag: var(--*), color-mix(in srgb, var(--*), var(--*)) where both args are vars
const HARDCODE_PATTERNS = [
  { re: /#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?\b/g,    label: 'hex' },
  { re: /#[0-9a-fA-F]{3}\b/g,                       label: 'hex3' },
  { re: /\brgba?\s*\(/g,                             label: 'rgba' },
  { re: /\bhsla?\s*\(/g,                             label: 'hsl' },
];

// Lines to skip: comments, @import, content: lines
function isIgnoredLine(line) {
  const t = line.trim();
  return t.startsWith('/*') || t.startsWith('*') || t.startsWith('//') || t.startsWith('@import') || t.startsWith('content:');
}

// Check if a color-mix() call only references var() — that's OK
// e.g. color-mix(in srgb, var(--card-bg), var(--state-hover)) ← allowed
// e.g. color-mix(in srgb, var(--primary) 12%, rgba(0,0,0,0.5)) ← flagged
function hasRawColorInColorMix(line) {
  const colorMixRe = /color-mix\([^)]+\)/g;
  const matches = line.match(colorMixRe) || [];
  for (const m of matches) {
    // Remove all var(...) references, then check if raw color remains
    const stripped = m.replace(/var\(--[\w-]+\)/g, 'VAR');
    if (/rgba?\s*\(|hsla?\s*\(|#[0-9a-fA-F]{3,8}/.test(stripped)) return true;
  }
  return false;
}

function checkHardcodedColors(files) {
  const violations = [];

  for (const file of files) {
    // Skip token sources and the shared style.css
    if (file.startsWith(TOKEN_DIR) || file === STYLE_CSS) continue;

    const lines = readLines(file);
    lines.forEach((line, i) => {
      if (isIgnoredLine(line)) return;

      // Flag color-mix() containing raw values
      if (/color-mix/.test(line) && hasRawColorInColorMix(line)) {
        violations.push({ file, line: i + 1, match: line.trim(), label: 'color-mix with raw value' });
        return;
      }

      for (const { re, label } of HARDCODE_PATTERNS) {
        re.lastIndex = 0;
        const match = re.exec(line);
        if (match) {
          violations.push({ file, line: i + 1, match: line.trim(), label });
        }
      }
    });
  }

  return violations;
}

// ─── Check 2: Unused Tokens ────────────────────────────────────────────────

function extractDefinedTokens() {
  const tokens = new Map(); // name → { file, line }
  for (const tf of TOKEN_FILES) {
    const lines = readLines(tf);
    lines.forEach((line, i) => {
      const m = line.match(/^\s*(--[\w-]+)\s*:/);
      if (m) tokens.set(m[1], { file: tf, line: i + 1 });
    });
  }
  return tokens;
}

function findTokenUsages(files) {
  const used = new Set();
  const tokenRefRe = /var\((--[\w-]+)\)/g;
  for (const file of files) {
    const content = fs.readFileSync(file, 'utf8');
    let m;
    while ((m = tokenRefRe.exec(content)) !== null) {
      used.add(m[1]);
    }
  }
  return used;
}

function checkUnusedTokens(allCssFiles) {
  const defined = extractDefinedTokens();
  const used = findTokenUsages(allCssFiles);

  const unused = [];
  for (const [name, loc] of defined) {
    // Skip tokens that reference other tokens (they're intermediate aliases, valid even if "unused" in module CSS)
    if (!used.has(name)) {
      unused.push({ name, ...loc });
    }
  }
  return unused;
}

// ─── Check 3: Module Import Coverage ───────────────────────────────────────

// Detect module root CSS files (entry points, not deeply nested)
// We define modules as top-level directories that have at least one CSS file
function detectModules(allCssFiles) {
  const modules = new Map(); // moduleDir → [css files]
  for (const f of allCssFiles) {
    if (f.startsWith(TOKEN_DIR)) continue;
    if (f === STYLE_CSS) continue;
    const rel = relPath(f);
    const parts = rel.split('/');
    const mod = parts[0] === 'css' ? 'css' : parts[0];
    if (!modules.has(mod)) modules.set(mod, []);
    modules.get(mod).push(f);
  }
  return modules;
}

const IMPORT_MARKERS = {
  foundation: /foundation\.css/,
  semantic:   /semantic\.css/,
  components: /components\.css/,
  style:      /css\/style\.css|style\.css/,
};

function checkModuleImports(modules) {
  const results = [];

  for (const [mod, files] of modules) {
    const combined = files.map(f => fs.readFileSync(f, 'utf8')).join('\n');

    // Also check HTML files for <link> tags
    const modDir = path.join(ROOT, mod);
    let htmlContent = '';
    try {
      for (const f of fs.readdirSync(modDir)) {
        if (f.endsWith('.html')) {
          htmlContent += fs.readFileSync(path.join(modDir, f), 'utf8');
        }
      }
    } catch { /* skip */ }

    const everything = combined + '\n' + htmlContent;
    const has = {
      foundation: IMPORT_MARKERS.foundation.test(everything),
      semantic:   IMPORT_MARKERS.semantic.test(everything),
      components: IMPORT_MARKERS.components.test(everything),
      style:      IMPORT_MARKERS.style.test(everything),
    };

    // Full integration: has style (which implies others) or has all 3 token files
    const fullViaStyle = has.style;
    const fullViaTokens = has.foundation && has.semantic && has.components;
    const status = fullViaStyle || fullViaTokens ? '✅' : (has.foundation || has.semantic ? '△' : '✕');

    results.push({ mod, status, has });
  }

  return results.sort((a, b) => a.mod.localeCompare(b.mod));
}

// ─── Formatting ────────────────────────────────────────────────────────────

const RESET = '\x1b[0m';
const RED   = '\x1b[31m';
const YEL   = '\x1b[33m';
const GRN   = '\x1b[32m';
const DIM   = '\x1b[2m';
const BOLD  = '\x1b[1m';

function header(title) {
  console.log(`\n${BOLD}── ${title} ${RESET}${'─'.repeat(Math.max(0, 60 - title.length - 4))}`);
}

// ─── Main ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const checkArg = args.includes('--check') ? args[args.indexOf('--check') + 1] : 'all';

const allFiles = collectCssFiles(ROOT);

// ① Hardcoded colors
if (checkArg === 'all' || checkArg === 'colors') {
  header('① Hardcoded Colors');
  const violations = checkHardcodedColors(allFiles);

  if (violations.length === 0) {
    console.log(`${GRN}No hardcoded colors found.${RESET}`);
  } else {
    // Group by file
    const byFile = new Map();
    for (const v of violations) {
      if (!byFile.has(v.file)) byFile.set(v.file, []);
      byFile.get(v.file).push(v);
    }
    for (const [file, vs] of byFile) {
      console.log(`\n${YEL}${relPath(file)}${RESET}`);
      for (const v of vs) {
        console.log(`  ${DIM}${String(v.line).padStart(4)}${RESET}  ${RED}[${v.label}]${RESET}  ${v.match.slice(0, 100)}`);
      }
    }
    console.log(`\n${RED}${violations.length} violation(s)${RESET}`);
  }
}

// ② Unused tokens
if (checkArg === 'all' || checkArg === 'tokens') {
  header('② Unused Tokens');
  const unused = checkUnusedTokens(allFiles);

  if (unused.length === 0) {
    console.log(`${GRN}All tokens are consumed.${RESET}`);
  } else {
    const byFile = new Map();
    for (const u of unused) {
      if (!byFile.has(u.file)) byFile.set(u.file, []);
      byFile.get(u.file).push(u);
    }
    for (const [file, us] of byFile) {
      console.log(`\n${DIM}${relPath(file)}${RESET}`);
      for (const u of us) {
        console.log(`  ${DIM}${String(u.line).padStart(4)}${RESET}  ${YEL}${u.name}${RESET}`);
      }
    }
    console.log(`\n${YEL}${unused.length} unused token(s)${RESET} ${DIM}(may be intentional forward-definitions)${RESET}`);
  }
}

// ③ Module imports
if (checkArg === 'all' || checkArg === 'imports') {
  header('③ Module Import Coverage');
  const modules = detectModules(allFiles);
  const results = checkModuleImports(modules);

  const colW = Math.max(...results.map(r => r.mod.length)) + 2;
  console.log(`\n${'Module'.padEnd(colW)}  Status  foundation  semantic  components  style`);
  console.log('─'.repeat(colW + 50));
  for (const r of results) {
    const tick = v => v ? `${GRN}✓${RESET}` : `${DIM}✗${RESET}`;
    console.log(
      `${r.mod.padEnd(colW)}  ${r.status}       ${tick(r.has.foundation)}           ${tick(r.has.semantic)}         ${tick(r.has.components)}           ${tick(r.has.style)}`
    );
  }

  const counts = { ok: 0, partial: 0, none: 0 };
  for (const r of results) {
    if (r.status === '✅') counts.ok++;
    else if (r.status === '△') counts.partial++;
    else counts.none++;
  }
  console.log(`\n${GRN}✅ ${counts.ok}${RESET}  ${YEL}△ ${counts.partial}${RESET}  ${RED}✕ ${counts.none}${RESET}`);
}

console.log('');
