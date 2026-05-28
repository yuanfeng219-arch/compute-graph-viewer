(function () {
  'use strict';

  const DATA_ROOT = 'demo-data/indexer-prolog-quant';

  const state = {
    view: 'source',
    taskId: 'explain-operator',
    selectedSemantic: null,
    data: null,
  };

  const dom = {
    primaryPath: document.getElementById('demoPrimaryPath'),
    caseTitle: document.getElementById('demoCaseTitle'),
    caseSubtitle: document.getElementById('demoCaseSubtitle'),
    entryFunction: document.getElementById('demoEntryFunction'),
    casePath: document.getElementById('demoCasePath'),
    taskList: document.getElementById('demoTaskList'),
    stageList: document.getElementById('demoStageList'),
    viewTabs: Array.from(document.querySelectorAll('#demoViewTabs .tb-tab')),
    resetBtn: document.getElementById('demoResetBtn'),
    panelTitle: document.getElementById('demoPanelTitle'),
    panelMeta: document.getElementById('demoPanelMeta'),
    views: Array.from(document.querySelectorAll('.demo-view')),
    sourceMeta: document.getElementById('demoSourceMeta'),
    sourceCode: document.getElementById('demoSourceCode'),
    anchorList: document.getElementById('demoAnchorList'),
    passCards: document.getElementById('demoPassCards'),
    pathSummary: document.getElementById('demoPathSummary'),
    pathLabels: document.getElementById('demoPathLabels'),
    pathSubgraphs: document.getElementById('demoPathSubgraphs'),
    executionMeta: document.getElementById('demoExecutionMeta'),
    semanticTop: document.getElementById('demoSemanticTop'),
    taskTable: document.getElementById('demoTaskTable'),
    executionInsights: document.getElementById('demoExecutionInsights'),
    cardBadge: document.getElementById('demoCardBadge'),
    explainTitle: document.getElementById('demoExplainTitle'),
    conclusion: document.getElementById('demoConclusion'),
    evidence: document.getElementById('demoEvidence'),
    reasoning: document.getElementById('demoReasoning'),
    actions: document.getElementById('demoActions'),
    risks: document.getElementById('demoRisks'),
  };

  init().catch((error) => {
    console.error(error);
    dom.caseTitle.textContent = 'Failed to load demo data';
    dom.caseSubtitle.textContent = String(error.message || error);
  });

  async function init() {
    bindEvents();
    state.data = await loadAll();
    state.selectedSemantic = state.data.semantic_steps[0]?.semanticLabel || null;
    render();
  }

  function bindEvents() {
    dom.viewTabs.forEach((button) => {
      button.addEventListener('click', () => {
        state.view = button.dataset.view;
        render();
      });
    });

    dom.resetBtn.addEventListener('click', () => {
      state.view = 'source';
      state.taskId = 'explain-operator';
      render();
    });
  }

  async function loadAll() {
    const files = [
      'scenario',
      'source_excerpt',
      'semantic_steps',
      'pass_snapshots',
      'path_summary',
      'swimlane_summary',
      'agent_cards',
    ];

    const entries = await Promise.all(files.map(async (name) => {
      const response = await fetch(`${DATA_ROOT}/${name}.json`);
      if (!response.ok) {
        throw new Error(`Unable to load ${name}.json`);
      }
      return [name, await response.json()];
    }));

    return Object.fromEntries(entries);
  }

  function render() {
    renderSidebar();
    renderMain();
    renderExplain();
    syncTabs();
  }

  function renderSidebar() {
    const scenario = state.data.scenario;
    dom.primaryPath.textContent = scenario.primaryPath;
    dom.caseTitle.textContent = scenario.title;
    dom.caseSubtitle.textContent = scenario.subtitle;
    dom.entryFunction.textContent = scenario.entryFunction;
    dom.casePath.textContent = scenario.primaryPath;

    dom.taskList.innerHTML = state.data.agent_cards.map((card) => {
      const checked = card.taskId === state.taskId ? ' checked' : '';
      const active = card.taskId === state.taskId ? ' active' : '';
      return `
        <label class="radio-opt${active}" data-task-id="${card.taskId}">
          <input type="radio" name="demo-task"${checked}>
          ${escapeHtml(card.title)}
        </label>
      `;
    }).join('');

    dom.taskList.querySelectorAll('[data-task-id]').forEach((node) => {
      node.addEventListener('click', () => {
        state.taskId = node.dataset.taskId;
        if (state.taskId === 'trace-path') state.view = 'path';
        if (state.taskId === 'analyze-bottleneck') state.view = 'execution';
        if (state.taskId === 'explain-operator') state.view = 'source';
        render();
      });
    });

    dom.stageList.innerHTML = state.data.semantic_steps.map((step) => {
      const active = step.semanticLabel === state.selectedSemantic ? ' active' : '';
      return `
        <article class="demo-stage-item${active}" data-semantic="${escapeAttr(step.semanticLabel)}">
          <div class="demo-stage-head">
            <div class="demo-card-title">${escapeHtml(step.semanticLabel)}</div>
            <span class="demo-stage-kind">${escapeHtml(step.kind)}</span>
          </div>
          <div class="demo-stage-desc">${escapeHtml(step.sourceSummary)}</div>
        </article>
      `;
    }).join('');

    dom.stageList.querySelectorAll('[data-semantic]').forEach((node) => {
      node.addEventListener('click', () => {
        state.selectedSemantic = node.dataset.semantic;
        renderExplain();
        renderSidebar();
      });
    });
  }

  function renderMain() {
    const titles = {
      source: 'Source View',
      pass: 'Pass Storyboard',
      path: 'Path Explanation',
      execution: 'Execution Summary',
    };
    dom.panelTitle.textContent = titles[state.view];
    dom.panelMeta.textContent = `${state.data.scenario.caseName} / ${state.data.scenario.primaryPath}`;

    dom.views.forEach((view) => {
      view.classList.toggle('active', view.dataset.view === state.view);
    });

    renderSourceView();
    renderPassView();
    renderPathView();
    renderExecutionView();
  }

  function renderSourceView() {
    const source = state.data.source_excerpt;
    dom.sourceMeta.textContent = `${source.file} : ${source.focusRange.start}-${source.focusRange.end}`;
    dom.sourceCode.textContent = source.code;
    dom.anchorList.innerHTML = source.anchors.map((anchor) => `
      <article class="demo-anchor-item">
        <div class="demo-card-head">
          <div class="demo-card-title">${escapeHtml(anchor.semanticLabel)}</div>
          <div class="demo-card-meta">L${anchor.lineStart}-${anchor.lineEnd}</div>
        </div>
        <div class="demo-anchor-desc">${escapeHtml(anchor.description)}</div>
      </article>
    `).join('');
  }

  function renderPassView() {
    dom.passCards.innerHTML = state.data.pass_snapshots.map((snapshot) => `
      <section class="demo-card">
        <div class="demo-card-head">
          <div class="demo-card-title">${escapeHtml(snapshot.passName)}</div>
          <div class="demo-card-meta">${escapeHtml(snapshot.snapshotLabel)}</div>
        </div>
        ${renderKvRows([
          ['Operations', snapshot.stats.operations],
          ['Tensors', snapshot.stats.tensors],
          ['Raw Tensors', snapshot.stats.rawTensors],
          ['Subgraphs', snapshot.stats.subgraphs],
        ])}
        <div class="form-label">Top Semantic Labels</div>
        ${snapshot.topSemanticLabels.map((item) => `
          <div class="demo-bar-row">
            <div class="demo-kv-key">${escapeHtml(item.label)}</div>
            <div class="demo-bar-track"><div class="demo-bar-fill" style="width:${Math.max(8, (item.count / snapshot.topSemanticLabels[0].count) * 100)}%"></div></div>
            <div class="demo-card-meta">${item.count}</div>
          </div>
        `).join('')}
      </section>
    `).join('');
  }

  function renderPathView() {
    const path = state.data.path_summary;
    dom.pathSummary.innerHTML = renderKvRows([
      ['Path', path.pathId],
      ['Loop', path.loopName],
      ['Unroll Mode', path.unrollMode],
      ['Branch Meaning', path.branchMeaning],
      ['Why Focus', path.reasonToFocus],
    ]);
    dom.pathLabels.innerHTML = path.linkedSemanticLabels.map(renderChip).join('');
    dom.pathSubgraphs.innerHTML = path.linkedSubgraphs.map(renderChip).join('');
  }

  function renderExecutionView() {
    const execution = state.data.swimlane_summary;
    dom.executionMeta.textContent = `${execution.eventCount} summarized events`;
    dom.semanticTop.innerHTML = execution.semanticTop.map((item) => `
      <div class="demo-bar-row">
        <div class="demo-kv-key">${escapeHtml(item.label)}</div>
        <div class="demo-bar-track"><div class="demo-bar-fill" style="width:${Math.max(8, (item.count / execution.semanticTop[0].count) * 100)}%"></div></div>
        <div class="demo-card-meta">${item.count}</div>
      </div>
    `).join('');

    dom.taskTable.innerHTML = execution.exampleTasks.map((task) => `
      <article class="demo-task-row">
        <div class="demo-task-head">
          <div class="demo-card-title">${escapeHtml(task.semanticLabel)}</div>
          <div class="demo-card-meta">${escapeHtml(task.lane)}</div>
        </div>
        <div class="demo-task-meta">
          <span>ts ${formatNumber(task.ts)}</span>
          <span>dur ${task.dur.toFixed(2)}</span>
          <span>${escapeHtml(task.name)}</span>
        </div>
      </article>
    `).join('');

    dom.executionInsights.innerHTML = execution.insightSummary.map(renderBullet).join('');
  }

  function renderExplain() {
    const card = state.data.agent_cards.find((item) => item.taskId === state.taskId) || state.data.agent_cards[0];
    dom.cardBadge.textContent = state.taskId;
    dom.explainTitle.textContent = card.title;
    dom.conclusion.textContent = card.conclusion;
    dom.evidence.innerHTML = card.evidence.map(renderBullet).join('');
    dom.reasoning.innerHTML = card.reasoning.map(renderBullet).join('');
    dom.actions.innerHTML = card.suggestedActions.map(renderBullet).join('');
    dom.risks.innerHTML = card.riskNotes.map(renderBullet).join('');
  }

  function syncTabs() {
    dom.viewTabs.forEach((button) => {
      button.classList.toggle('active', button.dataset.view === state.view);
    });
  }

  function renderKvRows(rows) {
    return rows.map(([key, value]) => `
      <div class="demo-kv-row">
        <div class="demo-kv-key">${escapeHtml(String(key))}</div>
        <div class="demo-kv-val">${escapeHtml(String(value))}</div>
      </div>
    `).join('');
  }

  function renderChip(value) {
    return `<span class="demo-chip">${escapeHtml(String(value))}</span>`;
  }

  function renderBullet(text) {
    return `<div class="demo-bullet-item">${escapeHtml(String(text))}</div>`;
  }

  function formatNumber(value) {
    return Number(value).toLocaleString('en-US', { maximumFractionDigits: 2 });
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }
})();
