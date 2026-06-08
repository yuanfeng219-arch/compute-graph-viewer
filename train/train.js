(function initTrainLearningPage() {
  'use strict';

  const TRAIN_ASSET_ROOT = './assets/';
  const models = {
    deepseek: {
      label: 'DeepSeek V3.2',
      title: 'DeepSeek V3.2 架构图',
      meta: 'MoE · MLA · DSA · 671B',
      frame: `${TRAIN_ASSET_ROOT}deepseek_v32_modelviz_embed.html?theme=light&embed=1`,
    },
    qwen: {
      label: 'Qwen7B',
      title: 'Qwen7B 架构图',
      meta: 'Dense Transformer · 32 Layers · 7B',
      frame: `${TRAIN_ASSET_ROOT}qwen7b_modelviz_embed.html?theme=light&embed=1`,
    },
  };

  const focusCopy = {
    overview: {
      label: '学习地图',
      callout: '从文本到输出：Tokenizer 切 token，Embedding 转向量，多层 Transformer 加工，LM Head 输出词表概率。',
    },
    tokenizer: {
      label: 'Tokenizer',
      callout: 'Tokenizer 把文本切成 token id。图里通常从 Token IDs、input_ids 或 tokenizer 入口开始。',
    },
    embedding: {
      label: 'Embedding',
      callout: 'Embedding 把 token id 转成向量，是文本进入神经网络计算前的第一站。',
    },
    layer: {
      label: '层',
      callout: 'Transformer 层是重复加工工位：每层通常包含 Norm、Attention、残差连接、MLP/FFN。',
    },
    four: {
      label: '结构、权重、数据、框架',
      callout: '模型结构、模型权重、训练数据和运行框架分别对应架构图、参数文件、样本管线和执行工具链。',
    },
    attention: {
      label: 'Attention',
      callout: 'Attention 负责让当前 token 看上下文重点。Qwen7B 是学习 Dense Attention 主干的直观样例。',
    },
    mlp: {
      label: 'MLP',
      callout: 'MLP/FFN 对每个 token 的表示做进一步加工，常和 Attention 交替出现在 Transformer Block 内。',
    },
    lmhead: {
      label: 'LM Head',
      callout: 'LM Head 把最后隐藏向量映射到词表概率，用来选择或采样下一个 token。',
    },
    output: {
      label: '输出 / Logits',
      callout: '输出节点通常是 Logits：LM Head 给词表里的每个 token 打分，采样或贪心选择后再解码成文字。',
    },
    moe: {
      label: 'MoE',
      callout: 'MoE 用 router 选择部分 expert 参与计算。DeepSeek V3.2 的大规模能力来自专家协作。',
    },
    dsa: {
      label: 'DSA',
      callout: 'DSA 先用 Lightning Indexer 选择 TopK 相关 token，再执行 Sparse Flash Attention。',
    },
    training: {
      label: '训练',
      callout: '训练会计算 loss 并反向传播更新参数；预训练、SFT、LoRA 的更新范围和目标不同。',
    },
    checkpoint: {
      label: 'Checkpoint',
      callout: 'Checkpoint 是某次训练保存的权重和状态；继续训练、转换权重、推理验证都依赖它。',
    },
    weight: {
      label: '权重格式',
      callout: 'HF、Megatron-Mcore、推理框架权重格式服务于不同工具链，转换时并行参数必须匹配。',
    },
    data: {
      label: '数据预处理',
      callout: '.bin 保存 token 内容，.idx 保存索引和样本边界；训练脚本通常读取预处理后的前缀。',
    },
    inference: {
      label: '推理',
      callout: '推理分 Prefill 和 Decode。KV Cache 复用历史 Key/Value，减少重复计算。',
    },
    parallel: {
      label: '并行',
      callout: 'TP/PP/DP/EP/CP 分别拆矩阵、层、数据、专家和上下文，用来协同训练或推理同一个模型。',
    },
    split: {
      label: '拆分训练',
      callout: '工程上可以分布式拆分；语义上不能把层各自独立训练后随便拼起来。',
    },
    compare: {
      label: '模型对照',
      callout: 'Qwen7B 适合学 Dense Transformer 基础，DeepSeek V3.2 适合学 MoE 和大规模推理工程。',
    },
    route: {
      label: '学习路线',
      callout: '先读模型目录和 config，再做基础推理，再学训练流程，最后读大规模并行和推理优化脚本。',
    },
    script: {
      label: '读脚本',
      callout: '读脚本先按类别看：路径、结构、并行、精度/量化、数据、checkpoint、启动方式和功能开关。',
    },
    flow: {
      label: '完整流程',
      callout: '完整链路是文本、tokenizer、.bin/.idx、权重转换、训练或微调、checkpoint、推理和输出 token。',
    },
    conclusion: {
      label: '结论',
      callout: 'Qwen7B 用来建立 Dense Transformer 地基；DeepSeek V3.2 用来理解 MoE、DSA、并行和大规模推理工程。',
    },
  };

  const frameFocusTargets = {
    qwen: {
      tokenizer: ['input_tokens'],
      embedding: ['token_embedding'],
      layer: ['decoder-stack', 'decoder_layer'],
      attention: ['attention-block', 'attention', 'scaled_attention'],
      mlp: ['mlp-block', 'mlp', 'mlp_gate_linear', 'mlp_up_linear', 'mlp_down_linear'],
      lmhead: ['lm_head'],
      output: ['output_logits'],
      inference: ['kv_cache', 'output_logits'],
    },
    deepseek: {
      tokenizer: ['dsv32arch_token_ids_608f714b'],
      embedding: ['dsv32arch_embedding_2f81ba1b'],
      layer: ['dsv32arch_block_0214b4b3'],
      attention: [
        'dsv32arch_attention_call_612a5ae9',
        'dsv32arch_attention_d0278de5',
        'dsv32arch_q_path_b399f580',
        'dsv32arch_kv_path_a69fbb2b',
      ],
      dsa: [
        'dsv32arch_attention_d0278de5',
        'dsv32arch_indexer_dfa17da7',
        'dsv32arch_sparse_attn_f5c98e92',
      ],
      moe: [
        'dsv32arch_ffn_a2b90028',
        'dsv32arch_moe_63e885ca',
        'dsv32arch_router_77eb1db6',
        'dsv32arch_experts_cfbede3c',
      ],
      mlp: [
        'dsv32arch_ffn_a2b90028',
        'dsv32arch_dense_c8e834d8',
        'dsv32arch_dense_w1w3_0b7a312e',
        'dsv32arch_dense_w2_8e01455d',
      ],
      lmhead: ['dsv32arch_lm_head_2b98b847'],
      output: ['dsv32arch_logits_c7d6c722'],
      inference: ['dsv32arch_cache_b0359280', 'dsv32arch_logits_c7d6c722'],
    },
  };

  const focusGraphAliases = {
    overview: ['tokenizer', 'embedding', 'layer', 'lmhead', 'output'],
    flow: ['tokenizer', 'data', 'training', 'weight', 'inference', 'output'],
    four: ['layer'],
    training: ['layer'],
    split: ['layer'],
    script: ['layer'],
    conclusion: ['layer'],
  };

  const body = document.body;
  const primaryFrame = document.getElementById('primaryFrame');
  const compareFrame = document.getElementById('compareFrame');
  const graphTitle = document.getElementById('graphTitle');
  const graphMeta = document.getElementById('graphMeta');
  const primaryFrameLabel = document.getElementById('primaryFrameLabel');
  const compareFrameLabel = document.getElementById('compareFrameLabel');
  const graphCallout = document.getElementById('graphCallout');
  const focusReadout = document.getElementById('focusReadout');
  const readerScroll = document.getElementById('readerScroll');
  const beginnerToggle = document.getElementById('beginnerToggle');

  const state = {
    model: 'deepseek',
    mode: 'single',
    focus: 'overview',
    beginner: true,
  };
  let focusLockUntil = 0;
  const initialParams = new URLSearchParams(window.location.search);
  const initialModel = initialParams.get('model');
  const initialMode = initialParams.get('mode');
  const initialFocus = initialParams.get('focus');
  if (models[initialModel]) {
    state.model = initialModel;
  }
  if (initialMode === 'single' || initialMode === 'compare') {
    state.mode = initialMode;
  }
  if (focusCopy[initialFocus]) {
    state.focus = initialFocus;
  }

  function otherModel(model) {
    return model === 'deepseek' ? 'qwen' : 'deepseek';
  }

  function setButtonSelected(selector, valueAttr, value) {
    document.querySelectorAll(selector).forEach((button) => {
      const isSelected = button.getAttribute(valueAttr) === value;
      button.classList.toggle('is-selected', isSelected);
    });
  }

  function graphFocusKeys(focus) {
    return new Set([focus, ...(focusGraphAliases[focus] || [])]);
  }

  function frameDocument(frame) {
    try {
      return frame.contentDocument || frame.contentWindow?.document || null;
    } catch (error) {
      return null;
    }
  }

  function tagFrameNodes(frame) {
    const doc = frameDocument(frame);
    if (!doc) return;

    doc.querySelectorAll('[class*="node_"], [class*="boundary-box_"], .pto-model-graphviz-node, .pto-model-graphviz-cluster, [class*="modelviz-schema-node_"]').forEach((element) => {
      if (element.dataset.nodeId) return;
      const datum = element.__data__;
      const nodeId = Array.isArray(datum)
        ? datum[0]
        : datum?.id || datum?.nodeId || null;
      if (nodeId) {
        element.dataset.nodeId = nodeId;
      }
    });
  }

  function targetIdsForFocus(model, activeKeys) {
    const modelTargets = frameFocusTargets[model] || {};
    const ids = new Set();
    activeKeys.forEach((key) => {
      (modelTargets[key] || []).forEach((id) => ids.add(id));
    });
    return ids;
  }

  function syncFrameFocus() {
    [primaryFrame, compareFrame].forEach((frame) => {
      if (!frame) return;
      const model = frame.dataset.modelFrame || state.model;
      const doc = frameDocument(frame);
      if (!doc) return;
      tagFrameNodes(frame);
      const targetIds = targetIdsForFocus(model, graphFocusKeys(state.focus));
      doc.querySelectorAll('[data-pto-train-focus="true"]').forEach((element) => {
        element.dataset.ptoTrainFocus = 'false';
      });
      doc.querySelectorAll('[data-node-id]').forEach((element) => {
        if (targetIds.has(element.dataset.nodeId)) {
          element.dataset.ptoTrainFocus = 'true';
        }
      });
    });
  }

  function syncVisualFocus() {
    syncFrameFocus();
  }

  function assignFrame(frame, model) {
    const config = models[model];
    frame.dataset.modelFrame = model;
    frame.setAttribute('title', `${config.label} model architecture`);
    if (frame.getAttribute('src') !== config.frame) {
      frame.setAttribute('src', config.frame);
    } else {
      window.setTimeout(() => normalizeFrame(frame), 30);
    }
  }

  function updateModel(model) {
    state.model = model;
    const config = models[model];
    const compareConfig = models[otherModel(model)];
    body.dataset.model = model;
    graphTitle.textContent = config.title;
    graphMeta.textContent = config.meta;
    primaryFrameLabel.textContent = config.label;
    compareFrameLabel.textContent = compareConfig.label;
    assignFrame(primaryFrame, model);
    assignFrame(compareFrame, otherModel(model));
    setButtonSelected('[data-model-button]', 'data-model-button', model);
  }

  function updateMode(mode) {
    state.mode = mode;
    body.dataset.mode = mode;
    setButtonSelected('[data-mode-button]', 'data-mode-button', mode);
    if (mode === 'compare') {
      updateFocus('compare', { scroll: false });
    } else if (state.focus === 'compare') {
      updateFocus('overview', { scroll: false });
    }
  }

  function updateFocus(focus, options = {}) {
    state.focus = focus;
    if (options.scrollTarget || options.lock) {
      focusLockUntil = window.performance.now() + 900;
    }
    const copy = focusCopy[focus] || focusCopy.overview;
    body.dataset.focus = focus;
    focusReadout.textContent = `焦点：${copy.label}`;
    graphCallout.textContent = copy.callout;
    document.querySelectorAll('[data-focus-button]').forEach((button) => {
      const isActive = button.getAttribute('data-focus-button') === focus;
      button.classList.toggle('is-active', isActive);
      if (button.classList.contains('btn')) {
        button.classList.toggle('is-selected', isActive);
      }
    });
    document.querySelectorAll('[data-focus-section]').forEach((section) => {
      section.classList.toggle('is-active', section.getAttribute('data-focus-section') === focus);
    });
    document.querySelectorAll('[data-focus-row]').forEach((row) => {
      row.classList.toggle('is-active-row', row.getAttribute('data-focus-row') === focus);
    });
    syncVisualFocus();
    if (options.scrollTarget) {
      scrollToSection(options.scrollTarget);
    }
  }

  function scrollToSection(id) {
    const section = document.getElementById(id);
    if (!section) return;
    section.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }

  function normalizeFrame(frame) {
    let doc;
    try {
      doc = frame.contentDocument || frame.contentWindow?.document;
    } catch (error) {
      return;
    }
    if (!doc || !doc.documentElement) return;

    const model = frame.dataset.modelFrame || state.model;
    doc.documentElement.dataset.theme = 'light';
    if (doc.body) {
      doc.body.dataset.theme = 'light';
      doc.body.style.background = 'var(--background)';
      doc.body.style.color = 'var(--foreground)';
    }

    const oldStyle = doc.getElementById('pto-train-embed-style');
    if (oldStyle) oldStyle.remove();

    const style = doc.createElement('style');
    style.id = 'pto-train-embed-style';
    style.textContent = `${model === 'qwen' ? qwenEmbedCss() : deepseekEmbedCss()}\n${frameFocusCss()}`;
    doc.head?.appendChild(style) || doc.documentElement.appendChild(style);
    window.setTimeout(() => {
      tagFrameNodes(frame);
      syncFrameFocus();
    }, 120);
    window.setTimeout(() => {
      tagFrameNodes(frame);
      syncFrameFocus();
    }, 520);
  }

  function frameFocusCss() {
    return `
      [data-pto-train-focus="true"] > rect,
      [data-pto-train-focus="true"] rect:first-child {
        stroke: var(--primary, #2563eb) !important;
        stroke-width: 4px !important;
        filter: drop-shadow(0 0 8px color-mix(in srgb, var(--primary, #2563eb) 36%, transparent)) !important;
      }

      [data-pto-train-focus="true"] > text,
      [data-pto-train-focus="true"] text {
        fill: var(--foreground, #111827) !important;
        font-weight: 700 !important;
      }
    `;
  }

  function qwenEmbedCss() {
    return `
      html,
      body,
      .qwen-app {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        background: var(--background) !important;
        color: var(--foreground) !important;
        overflow: hidden !important;
      }

      .qwen-app {
        display: block !important;
      }

      .qwen-topbar,
      .qwen-actions,
      .qwen-status {
        display: none !important;
      }

      .qwen-main,
      .qwen-graph-shell,
      .qwen-graph-stage,
      #graphStage {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: var(--background) !important;
        box-shadow: none !important;
      }
    `;
  }

  function deepseekEmbedCss() {
    return `
      :root,
      html,
      body {
        width: 100% !important;
        height: 100% !important;
        min-height: 0 !important;
        margin: 0 !important;
        color-scheme: light !important;
        background: var(--background) !important;
        color: var(--foreground) !important;
        overflow: hidden !important;
      }

      [class^="logo_"],
      [class*=" logo_"],
      [class^="source-validation_"],
      [class*=" source-validation_"],
      [class^="modelviz-toolbar_"],
      [class*=" modelviz-toolbar_"],
      [class^="modelviz-status_"],
      [class*=" modelviz-status_"] {
        display: none !important;
      }

      [id^="graph-container_"],
      [id*=" graph-container_"],
      [id^="graph_"],
      [id*=" graph_"] {
        width: 100% !important;
        min-width: 0 !important;
        height: 100% !important;
        min-height: 0 !important;
        background: var(--background) !important;
        color: var(--foreground) !important;
      }

      .popup_8738e552-e764-4a57-839a-6321bc3ae7d3 {
        max-width: calc(100% - 24px) !important;
      }
    `;
  }

  function handleAction(action) {
    const modelTarget = action.getAttribute('data-model-target');
    if (modelTarget && models[modelTarget] && modelTarget !== state.model) {
      updateModel(modelTarget);
    }
    const target = action.getAttribute('data-section-target');
    const focus = action.getAttribute('data-focus-button')
      || document.getElementById(target)?.getAttribute('data-focus-section')
      || 'overview';
    updateFocus(focus, { scrollTarget: target });
  }

  document.addEventListener('click', (event) => {
    const action = event.target.closest('button');
    if (!action) return;
    if (action.hasAttribute('data-model-button')) {
      updateModel(action.getAttribute('data-model-button'));
      return;
    }
    if (action.hasAttribute('data-mode-button')) {
      updateMode(action.getAttribute('data-mode-button'));
      return;
    }
    if (action.hasAttribute('data-focus-button') || action.hasAttribute('data-section-target')) {
      handleAction(action);
    }
  });

  function toggleBeginner() {
    state.beginner = !state.beginner;
    body.classList.toggle('is-beginner', state.beginner);
    beginnerToggle.setAttribute('aria-pressed', String(state.beginner));
  }

  beginnerToggle.addEventListener('click', toggleBeginner);

  window.trainLearning = {
    updateModel,
    updateMode,
    updateFocus,
    toggleBeginner,
  };

  primaryFrame.addEventListener('load', () => normalizeFrame(primaryFrame));
  compareFrame.addEventListener('load', () => normalizeFrame(compareFrame));

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      if (window.performance.now() < focusLockUntil) return;
      const focus = visible.target.getAttribute('data-focus-section');
      if (!focus || focus === state.focus) return;
      updateFocus(focus, { scroll: false });
    }, {
      root: readerScroll,
      threshold: [0.45, 0.7],
    });
    document.querySelectorAll('.train-doc-section').forEach((section) => observer.observe(section));
  }

  updateModel(state.model);
  updateMode(state.mode);
  updateFocus(state.focus, { scroll: false });
})();
