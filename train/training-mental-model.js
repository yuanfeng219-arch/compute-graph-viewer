(function initAscendTrainingMentalModel() {
  'use strict';

  const SRC = '../../gitcode/MindSpeed-LLM-master/';

  function n(id, label, typeLabel, kind, x, y, width, height, extra) {
    return {
      id,
      label,
      typeLabel,
      kind,
      x,
      y,
      width,
      height,
      ...(extra || {}),
    };
  }

  function e(source, target, extra) {
    return { source, target, ...(extra || {}) };
  }

  const graphs = {
    ecosystem: {
      width: 760,
      height: 520,
      clusters: [
        { id: 'stack', label: 'Training Stack', x: 94, y: 52, width: 572, height: 386, colorKey: 'module:stack' },
      ],
      nodes: [
        n('hardware', 'Hardware Runtime', 'Layer', 'op', 380, 98, 270, 54, { colorKey: 'sem:runtime' }),
        n('core', 'MindSpeed Core', 'Layer', 'op', 380, 176, 260, 54, { colorKey: 'sem:core' }),
        n('llm', 'MindSpeed-LLM', 'Layer', 'op', 380, 254, 260, 54, { colorKey: 'sem:llm' }),
        n('task', 'Training Task', 'Module', 'op', 380, 332, 260, 54, { colorKey: 'sem:task' }),
        n('diag', 'msTT Diagnostics', 'Loop', 'op', 380, 410, 260, 54, { colorKey: 'sem:diag' }),
        n('user', 'Developer Goal', 'Input', 'tensor', 120, 254, 180, 48, { colorKey: 'io:input' }),
        n('result', 'Checkpoint Evidence', 'Output', 'tensor', 640, 254, 190, 48, { colorKey: 'io:output' }),
      ],
      edges: [
        e('hardware', 'core'),
        e('core', 'llm'),
        e('llm', 'task'),
        e('task', 'diag'),
        e('diag', 'llm', { dashed: true }),
        e('user', 'llm'),
        e('task', 'result'),
      ],
    },
    levels: {
      width: 760,
      height: 520,
      clusters: [
        { id: 'cases', label: 'Case Ladder', x: 72, y: 68, width: 616, height: 330, colorKey: 'module:cases' },
      ],
      nodes: [
        n('beginner', 'Beginner', 'Persona', 'tensor', 145, 126, 160, 48, { colorKey: 'io:input' }),
        n('qwen3', 'Qwen7B -> Qwen3', 'Dense Path', 'op', 365, 126, 230, 54, { colorKey: 'sem:dense' }),
        n('goal1', 'First Step', 'Outcome', 'tensor', 610, 126, 150, 48, { colorKey: 'io:output' }),
        n('advanced', 'Advanced', 'Persona', 'tensor', 145, 244, 160, 48, { colorKey: 'io:input' }),
        n('qwenmoe', 'Qwen3 MoE', 'Model', 'op', 365, 244, 210, 54, { colorKey: 'sem:moe' }),
        n('goal2', 'Correct Run', 'Outcome', 'tensor', 610, 244, 150, 48, { colorKey: 'io:output' }),
        n('expert', 'Expert', 'Persona', 'tensor', 145, 362, 160, 48, { colorKey: 'io:input' }),
        n('deepseek', 'DeepSeek MoE', 'Model', 'op', 365, 362, 210, 54, { colorKey: 'sem:deepseek' }),
        n('goal3', 'Efficient Run', 'Outcome', 'tensor', 610, 362, 150, 48, { colorKey: 'io:output' }),
      ],
      edges: [
        e('beginner', 'qwen3'),
        e('qwen3', 'goal1'),
        e('advanced', 'qwenmoe'),
        e('qwenmoe', 'goal2'),
        e('expert', 'deepseek'),
        e('deepseek', 'goal3'),
      ],
    },
    journey: {
      width: 760,
      height: 520,
      clusters: [
        { id: 'assets', label: 'Assets', x: 48, y: 70, width: 212, height: 318, colorKey: 'module:assets' },
        { id: 'train', label: 'Training Assembly', x: 286, y: 70, width: 424, height: 318, colorKey: 'module:train' },
      ],
      nodes: [
        n('hf', 'HF Weights', 'Input', 'tensor', 154, 132, 150, 46, { parent: 'assets' }),
        n('rawdata', 'Raw Data', 'Input', 'tensor', 154, 226, 150, 46, { parent: 'assets' }),
        n('tokenizer', 'Tokenizer', 'Input', 'tensor', 154, 320, 150, 46, { parent: 'assets' }),
        n('convert', 'Weight Convert', 'Tool', 'op', 386, 132, 190, 54, { parent: 'train', colorKey: 'sem:convert' }),
        n('preprocess', 'Data Convert', 'Tool', 'op', 386, 226, 190, 54, { parent: 'train', colorKey: 'sem:data' }),
        n('script', 'Train Script', 'Tool', 'op', 386, 320, 190, 54, { parent: 'train', colorKey: 'sem:script' }),
        n('run', 'Distributed Run', 'Runtime', 'op', 602, 226, 190, 54, { parent: 'train', colorKey: 'sem:run' }),
        n('ckpt', 'Checkpoint', 'Output', 'tensor', 602, 348, 166, 46, { parent: 'train', colorKey: 'io:output' }),
      ],
      edges: [
        e('hf', 'convert'),
        e('rawdata', 'preprocess'),
        e('tokenizer', 'preprocess'),
        e('convert', 'script'),
        e('preprocess', 'script'),
        e('tokenizer', 'script'),
        e('script', 'run'),
        e('run', 'ckpt'),
      ],
    },
    qwen7b: {
      width: 820,
      height: 660,
      clusters: [
        { id: 'source', label: 'Local Qwen7B Source', x: 42, y: 58, width: 252, height: 546, colorKey: 'module:source' },
        { id: 'model', label: 'Dense Decoder Model', x: 352, y: 58, width: 420, height: 546, colorKey: 'module:decoder', repeat: '32' },
      ],
      nodes: [
        n('readme', 'README', 'Official Note', 'tensor', 168, 128, 164, 44, { parent: 'source', colorKey: 'io:input' }),
        n('config', 'config.json', 'Structure', 'tensor', 168, 210, 164, 44, { parent: 'source', colorKey: 'io:input' }),
        n('modeling', 'modeling_qwen.py', 'Code', 'tensor', 168, 322, 190, 44, { parent: 'source', colorKey: 'io:input' }),
        n('weights', 'safetensors index', 'Weights', 'tensor', 168, 492, 190, 44, { parent: 'source', colorKey: 'io:output' }),
        n('generation', 'generation_config', 'Sampling', 'tensor', 168, 564, 190, 44, { parent: 'source', colorKey: 'io:input' }),
        n('tokens', 'Token IDs', 'Input', 'tensor', 562, 112, 160, 44, { parent: 'model', colorKey: 'io:input' }),
        n('embed', 'Embedding', '151936 x 4096', 'op', 562, 190, 214, 50, { parent: 'model', colorKey: 'sem:embedding' }),
        n('attention', 'Attention', '32 heads', 'op', 562, 284, 214, 50, { parent: 'model', colorKey: 'sem:attention' }),
        n('mlp', 'SwiGLU MLP', '22016', 'op', 562, 362, 214, 50, { parent: 'model', colorKey: 'sem:mlp' }),
        n('norm', 'RMSNorm', 'Pre + Final', 'op', 562, 440, 214, 50, { parent: 'model', colorKey: 'sem:norm' }),
        n('lmhead', 'LM Head', 'Logits', 'op', 562, 518, 214, 50, { parent: 'model', colorKey: 'sem:linear' }),
        n('sample', 'top_p / eos', 'Decode Rule', 'tensor', 562, 586, 180, 44, { parent: 'model', colorKey: 'io:output' }),
      ],
      edges: [
        e('readme', 'config'),
        e('config', 'embed'),
        e('modeling', 'attention'),
        e('modeling', 'mlp'),
        e('generation', 'sample'),
        e('weights', 'lmhead'),
        e('tokens', 'embed'),
        e('embed', 'attention'),
        e('attention', 'mlp'),
        e('mlp', 'norm'),
        e('norm', 'lmhead'),
        e('lmhead', 'sample'),
      ],
    },
    qwen3: {
      width: 760,
      height: 620,
      clusters: [
        { id: 'decoder', label: 'Decoder Layer', x: 150, y: 174, width: 460, height: 292, colorKey: 'module:decoder', repeat: '36' },
      ],
      nodes: [
        n('tokens', 'Token IDs', 'Input', 'tensor', 380, 58, 160, 46, { colorKey: 'io:input' }),
        n('embed', 'Embedding', 'Op', 'op', 380, 132, 210, 54, { colorKey: 'sem:embedding' }),
        n('attn_norm', 'Attention Norm', 'Op', 'op', 300, 236, 196, 52, { parent: 'decoder' }),
        n('attention', 'Attention', 'Module', 'op', 300, 326, 186, 52, { parent: 'decoder', colorKey: 'sem:attention' }),
        n('mlp_norm', 'MLP Norm', 'Op', 'op', 478, 236, 168, 52, { parent: 'decoder' }),
        n('mlp', 'MLP', 'Module', 'op', 478, 326, 168, 52, { parent: 'decoder', colorKey: 'sem:mlp' }),
        n('block_out', 'Block Output', 'Tensor', 'tensor', 380, 424, 180, 46, { parent: 'decoder' }),
        n('final_norm', 'Final Norm', 'Op', 'op', 380, 510, 180, 52, { colorKey: 'sem:norm' }),
        n('lm_head', 'LM Head', 'Op', 'op', 380, 574, 180, 52, { colorKey: 'sem:linear' }),
        n('logits', 'Logits', 'Output', 'tensor', 618, 574, 130, 46, { colorKey: 'io:output' }),
      ],
      edges: [
        e('tokens', 'embed'),
        e('embed', 'attn_norm'),
        e('attn_norm', 'attention'),
        e('attention', 'mlp_norm'),
        e('mlp_norm', 'mlp'),
        e('mlp', 'block_out'),
        e('block_out', 'final_norm'),
        e('final_norm', 'lm_head'),
        e('lm_head', 'logits'),
        e('embed', 'mlp_norm', { dashed: true }),
      ],
    },
    qwen3moe: {
      width: 760,
      height: 640,
      clusters: [
        { id: 'moe_layer', label: 'MoE Decoder Layer', x: 82, y: 144, width: 560, height: 334, colorKey: 'module:moe-layer', repeat: '48' },
        { id: 'dpo', label: 'DPO Task', x: 82, y: 504, width: 560, height: 98, colorKey: 'module:dpo' },
      ],
      nodes: [
        n('tokens', 'Token IDs', 'Input', 'tensor', 380, 64, 160, 46, { colorKey: 'io:input' }),
        n('embed', 'Embedding', 'Op', 'op', 380, 122, 190, 52, { colorKey: 'sem:embedding' }),
        n('attention', 'Attention', 'Module', 'op', 250, 220, 180, 52, { parent: 'moe_layer', colorKey: 'sem:attention' }),
        n('router', 'Router', 'Op', 'op', 500, 220, 156, 52, { parent: 'moe_layer', colorKey: 'sem:router' }),
        n('topk', 'TopK Select', 'Op', 'op', 500, 306, 156, 52, { parent: 'moe_layer', colorKey: 'sem:router' }),
        n('expert', 'Expert MLP', 'Op', 'op', 500, 392, 166, 52, { parent: 'moe_layer', colorKey: 'sem:expert' }),
        n('combine', 'Expert Combine', 'Op', 'op', 250, 392, 190, 52, { parent: 'moe_layer', colorKey: 'sem:combine' }),
        n('chosen', 'Chosen', 'Input', 'tensor', 190, 552, 128, 42, { parent: 'dpo', colorKey: 'io:input' }),
        n('rejected', 'Rejected', 'Input', 'tensor', 336, 552, 128, 42, { parent: 'dpo', colorKey: 'io:input' }),
        n('loss', 'DPO Loss', 'Task', 'op', 510, 552, 150, 48, { parent: 'dpo', colorKey: 'sem:loss' }),
      ],
      edges: [
        e('tokens', 'embed'),
        e('embed', 'attention'),
        e('attention', 'router'),
        e('router', 'topk'),
        e('topk', 'expert'),
        e('expert', 'combine'),
        e('combine', 'loss'),
        e('chosen', 'loss'),
        e('rejected', 'loss'),
      ],
    },
    deepseek: {
      width: 760,
      height: 660,
      clusters: [
        { id: 'decoder', label: 'DeepSeek Decoder Layer', x: 80, y: 126, width: 590, height: 410, colorKey: 'module:decoder', repeat: '64' },
      ],
      nodes: [
        n('tokens', 'Token IDs', 'Input', 'tensor', 380, 58, 160, 46, { colorKey: 'io:input' }),
        n('embed', 'Embedding', 'Op', 'op', 380, 110, 190, 52, { colorKey: 'sem:embedding' }),
        n('mla', 'MLA', 'Module', 'op', 244, 204, 160, 52, { parent: 'decoder', colorKey: 'sem:attention' }),
        n('indexer', 'Sparse Indexer', 'Module', 'op', 244, 286, 190, 52, { parent: 'decoder', colorKey: 'sem:indexer' }),
        n('attn', 'Attention', 'Op', 'op', 244, 368, 170, 52, { parent: 'decoder', colorKey: 'sem:attention' }),
        n('router', 'Router', 'Op', 'op', 520, 204, 156, 52, { parent: 'decoder', colorKey: 'sem:router' }),
        n('moe', 'MoE', 'Module', 'op', 520, 286, 156, 52, { parent: 'decoder', colorKey: 'sem:moe' }),
        n('mtp', 'MTP', 'Module', 'op', 520, 368, 156, 52, { parent: 'decoder', colorKey: 'sem:mtp' }),
        n('block_out', 'Block Output', 'Tensor', 'tensor', 380, 478, 180, 46, { parent: 'decoder' }),
        n('norm', 'Final Norm', 'Op', 'op', 380, 574, 180, 52, { colorKey: 'sem:norm' }),
        n('lmhead', 'LM Head', 'Op', 'op', 380, 628, 180, 52, { colorKey: 'sem:linear' }),
        n('logits', 'Logits', 'Output', 'tensor', 616, 628, 130, 46, { colorKey: 'io:output' }),
      ],
      edges: [
        e('tokens', 'embed'),
        e('embed', 'mla'),
        e('mla', 'indexer'),
        e('indexer', 'attn'),
        e('attn', 'router'),
        e('router', 'moe'),
        e('moe', 'mtp'),
        e('mtp', 'block_out'),
        e('block_out', 'norm'),
        e('norm', 'lmhead'),
        e('lmhead', 'logits'),
        e('embed', 'router', { dashed: true }),
      ],
    },
    rlhf: {
      width: 760,
      height: 560,
      clusters: [
        { id: 'ray', label: 'Ray RLHF Runtime', x: 74, y: 70, width: 612, height: 390, colorKey: 'module:ray' },
      ],
      nodes: [
        n('prompt', 'Prompt Data', 'Input', 'tensor', 156, 140, 150, 46, { parent: 'ray', colorKey: 'io:input' }),
        n('policy', 'Policy Model', 'Model', 'op', 370, 140, 190, 54, { parent: 'ray', colorKey: 'sem:policy' }),
        n('rollout', 'Rollout', 'Runtime', 'op', 582, 140, 154, 54, { parent: 'ray', colorKey: 'sem:run' }),
        n('reward', 'Reward Signal', 'Task', 'op', 582, 256, 170, 54, { parent: 'ray', colorKey: 'sem:reward' }),
        n('reference', 'Reference Model', 'Model', 'op', 370, 256, 190, 54, { parent: 'ray', colorKey: 'sem:reference' }),
        n('grpo', 'GRPO Update', 'Trainer', 'op', 370, 372, 190, 54, { parent: 'ray', colorKey: 'sem:update' }),
        n('ckpt', 'Updated Policy', 'Output', 'tensor', 582, 372, 170, 46, { parent: 'ray', colorKey: 'io:output' }),
      ],
      edges: [
        e('prompt', 'policy'),
        e('policy', 'rollout'),
        e('rollout', 'reward'),
        e('reference', 'grpo', { dashed: true }),
        e('reward', 'grpo'),
        e('policy', 'grpo'),
        e('grpo', 'ckpt'),
      ],
    },
    breakpoints: {
      width: 760,
      height: 540,
      clusters: [
        { id: 'checks', label: 'Before Run Checks', x: 68, y: 70, width: 624, height: 342, colorKey: 'module:checks' },
      ],
      nodes: [
        n('task', 'Task Choice', 'Input', 'tensor', 150, 140, 150, 46, { parent: 'checks', colorKey: 'io:input' }),
        n('backend', 'Backend', 'Check', 'op', 370, 110, 150, 48, { parent: 'checks', colorKey: 'sem:backend' }),
        n('weights', 'Weight Format', 'Check', 'op', 370, 178, 170, 48, { parent: 'checks', colorKey: 'sem:weights' }),
        n('data', 'Data Prefix', 'Check', 'op', 370, 246, 150, 48, { parent: 'checks', colorKey: 'sem:data' }),
        n('parallel', 'Parallel Topology', 'Check', 'op', 370, 314, 190, 48, { parent: 'checks', colorKey: 'sem:parallel' }),
        n('launch', 'Launch Rank', 'Check', 'op', 570, 246, 150, 48, { parent: 'checks', colorKey: 'sem:launch' }),
        n('diagnose', 'Diagnosis Path', 'Output', 'tensor', 570, 346, 170, 46, { parent: 'checks', colorKey: 'io:output' }),
      ],
      edges: [
        e('task', 'backend'),
        e('backend', 'weights'),
        e('weights', 'data'),
        e('data', 'parallel'),
        e('parallel', 'launch'),
        e('launch', 'diagnose'),
      ],
    },
    mvp: {
      width: 760,
      height: 540,
      clusters: [
        { id: 'product', label: 'Visual Training Workbench', x: 72, y: 62, width: 616, height: 370, colorKey: 'module:product' },
      ],
      nodes: [
        n('input', 'Model Task Card', 'Input', 'tensor', 158, 142, 170, 46, { parent: 'product', colorKey: 'io:input' }),
        n('wizard', 'Task Wizard', 'Module', 'op', 370, 118, 170, 50, { parent: 'product', colorKey: 'sem:wizard' }),
        n('checker', 'Script Checker', 'Module', 'op', 370, 196, 170, 50, { parent: 'product', colorKey: 'sem:checker' }),
        n('graph', 'Architecture View', 'Module', 'op', 370, 274, 190, 50, { parent: 'product', colorKey: 'sem:graph' }),
        n('explainer', 'Run Explainer', 'Module', 'op', 370, 352, 170, 50, { parent: 'product', colorKey: 'sem:explainer' }),
        n('mtt', 'msTT Profiler', 'Tool', 'op', 594, 274, 160, 50, { parent: 'product', colorKey: 'sem:diag' }),
        n('output', 'Next Action', 'Output', 'tensor', 594, 352, 150, 46, { parent: 'product', colorKey: 'io:output' }),
      ],
      edges: [
        e('input', 'wizard'),
        e('wizard', 'checker'),
        e('checker', 'graph'),
        e('graph', 'explainer'),
        e('mtt', 'explainer'),
        e('explainer', 'output'),
      ],
    },
  };

  const visuals = {
    ecosystem: {
      title: '生态分层图',
      meta: '硬件与运行时 -> MindSpeed Core -> MindSpeed-LLM -> 训练任务 -> msTT',
      chip: '主线',
      graph: graphs.ecosystem,
      insight: '这一层图的价值是把 Ascend 训练生态和竞品训练生态放到同一张地图里：Ascend 的优势是把 NPU 运行时、MindSpeed Core 加速、MindSpeed LLM 任务装配和 profiling 诊断形成端到端链路；竞品生态组件更成熟、更分散，开发者需要自己拼装约束。',
      nodeNotes: {
        hardware: 'Atlas A2/A3 训练系列、CANN、torch_npu、NNAL 等运行时约束决定脚本能否启动。',
        core: 'MindSpeed Core 承接并行、通信、融合算子和显存优化能力。',
        llm: 'MindSpeed-LLM 是训练任务总装层，官方架构包含功能模块、训练后端、训练算法和大模型四层。',
        task: '训练任务包括 pretrain、SFT、LoRA、DPO、RLHF、eval 与推理验证。',
        diag: 'msTT、profiler、msProbe 应回流到训练异常解释和性能优化闭环。',
        result: 'checkpoint 是训练链路的可验证产物，后续用于续训、评估、推理和格式转换。',
      },
      evidence: [
        ['官方四层', '功能模块、训练后端、训练算法、大模型：分别对应工具链、执行路线、训练方法和模型菜谱库。'],
        ['Ascend 路线', 'Atlas NPU + CANN/torch_npu/HCCL + MindSpeed Core + MindSpeed LLM，把训练约束收束到昇腾原生链路。'],
        ['竞品路线', 'CUDA/NCCL + Megatron/DeepSpeed/FSDP/Transformers 生态成熟，但权重、数据、并行和 profiling 常需要自行组合。'],
        ['做饭类比', 'NPU 是灶台，CANN/torch_npu 是锅具适配，MindSpeed Core 是高效厨具，MindSpeed LLM 是菜谱和备菜流程。'],
      ],
      sources: [
        ['MindSpeed LLM introduction', `${SRC}docs/zh/pytorch/introduction.md`],
        ['MindSpeed feature list', 'https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/modthirdparty/asdevguide/mindspeed_0007.html'],
      ],
    },
    levels: {
      title: '开发者分层图',
      meta: '从 Qwen7B 源码入门到 Qwen3 dense 训练，再到 MoE 与 DeepSeek-V3.2',
      chip: 'Level',
      graph: graphs.levels,
      insight: '同一套仓库要给不同 level 的用户暴露不同复杂度。初学者看结构和路径，进阶看数据和 MoE，专家看拓扑和性能证据。',
      nodeNotes: {
        beginner: '目标是从选模型到首个 step，少决策、少分叉。',
        qwen3: 'Qwen7B 负责本地源码和配置入门，Qwen3-8B 负责对接 Ascend 官方训练快速路径。',
        qwenmoe: 'Qwen3-MoE 30B-A3B 引入 router、top-k、expert、EP 和 DPO 数据。',
        deepseek: 'DeepSeek-V3.2 671B 用于解释超大 MoE、MLA、MTP、TP/PP/EP/CP 和性能证据。',
      },
      evidence: [
        ['初学者', '先用 Qwen7B 读懂本地模型目录和 Dense Transformer，再用 Qwen3-8B 跑通 Ascend 官方训练链路。'],
        ['进阶', 'Qwen3-MoE 30B-A3B 把 router、expert、DPO、长上下文放到同一条线。'],
        ['专家', 'DeepSeek-V3.2 671B 的价值是验证超大 MoE 的并行、通信和显存策略。'],
        ['产品策略', '默认路径少决策，专家模式保留完整控制面。'],
      ],
      sources: [
        ['Qwen3 examples', `${SRC}examples/mcore/qwen3/`],
        ['Qwen3 MoE examples', `${SRC}examples/mcore/qwen3_moe/`],
        ['DeepSeek V3.2 examples', `${SRC}examples/mcore/deepseek32/`],
      ],
    },
    journey: {
      title: '端到端训练旅程图',
      meta: 'HF 权重、tokenizer、数据、转换、训练脚本、checkpoint',
      chip: 'Flow',
      graph: graphs.journey,
      insight: '训练体验的断点通常发生在资产转换边界：HF 权重到 Mcore/FSDP2，原始数据到 bin/idx，脚本参数到真实 rank 拓扑。',
      nodeNotes: {
        hf: 'HF 权重通常不能直接进入 Mcore 训练，需要转换或选择 FSDP2 原生路径。',
        rawdata: '原始数据可能是纯文本、指令数据、ShareGPT 或 pairwise 数据。',
        tokenizer: 'tokenizer 必须和模型权重匹配，数据预处理和训练脚本都依赖它。',
        convert: '权重转换时的 TP/PP 等切分配置要和训练脚本保持一致。',
        preprocess: '数据转换产物通常是 bin/idx，训练脚本填写的是前缀。',
        script: '训练脚本汇总路径、模型结构、并行、优化开关和启动参数。',
        run: '分布式启动最容易错在 rank、IP、端口、网卡和节点数。',
        ckpt: 'checkpoint 需要被评估、推理验证，必要时转换回 HF 格式。',
      },
      evidence: [
        ['权重转换', 'Mcore 训练脚本常要求先执行 hf2mcore，转换切分与训练 TP/PP 必须一致。'],
        ['数据转换', '训练脚本里的 DATA_PATH 多数是预处理产物前缀，不是完整 bin 文件名。'],
        ['启动配置', 'NPUS_PER_NODE、NNODES、NODE_RANK、MASTER_ADDR 是训练旅程里的高频错误源。'],
        ['验证闭环', 'checkpoint 需要继续训练、评估、推理验证和可能的 mcore2hf 转换。'],
      ],
      sources: [
        ['Qwen3 quick start', `${SRC}docs/zh/pytorch/training/quick_start.md`],
        ['Mcore pretrain doc', `${SRC}docs/zh/pytorch/training/pretrain/mcore/pretrain.md`],
      ],
    },
    qwen7b: {
      title: 'Qwen7B 本地源码证据图',
      meta: 'README · config · modeling_qwen.py · generation_config · safetensors',
      chip: 'Qwen7B',
      graph: graphs.qwen7b,
      insight: 'Qwen7B 适合把模型学习从“听说是 7B”变成可验证证据链：README 给官方口径，config 给结构参数，modeling_qwen.py 给执行路径，generation_config 给采样规则，safetensors index 给权重分片。',
      nodeNotes: {
        readme: 'README 给出 32 layers、32 heads、4096 d_model、8192 sequence length，并说明 RoPE、SwiGLU、RMSNorm 和 FlashAttention。',
        config: 'config.json 是结构事实源：151936 vocab、4096 hidden、22016 FFN、32768 max position、use_cache=true。',
        modeling: 'modeling_qwen.py 把 Attention、MLP、Block、Model 和 LMHead 落成 Python 类和 forward 逻辑。',
        generation: 'generation_config.json 控制推理停止与采样：eos/pad 151643、max_new_tokens 512、top_p 0.8、top_k 0。',
        weights: 'model.safetensors.index.json 的 weight_map 显示 transformer.h.* 和 lm_head 分布在 8 个 shard。',
        embed: 'Embedding 维度由 vocab_size x hidden_size 决定，token id 在这里变成连续向量。',
        attention: 'QWenAttention 使用 32 heads，c_attn 一次投影出 Q/K/V，并支持 cache 与 flash attention 路径。',
        mlp: 'QWenMLP 使用 w1、w2 和 SiLU 门控，属于 SwiGLU 风格 MLP。',
        norm: 'RMSNorm 出现在每个 block 的 attention/MLP 前，以及模型最终 ln_f。',
        lmhead: 'LM Head 将 4096 hidden state 映射回 151936 词表 logits。',
        sample: '采样规则决定 decode 输出，不应和模型结构混为一谈。',
      },
      evidence: [
        ['结构参数', '32 layers、hidden size 4096、32 heads、intermediate size 22016、seq length 8192。'],
        ['代码路径', 'Token IDs -> wte Embedding -> 32x QWenBlock -> ln_f -> lm_head -> logits。'],
        ['推理配置', 'eos/pad token 151643、max_new_tokens 512、do_sample true、top_p 0.8、top_k 0。'],
        ['权重证据', 'safetensors index metadata total_size 15442649088，weight_map 指向 8 个 shard。'],
      ],
      sources: [
        ['Qwen7B README', '../../gitcode/qwen7b-source/README.md'],
        ['Qwen7B config', '../../gitcode/qwen7b-source/config.json'],
        ['Qwen7B modeling code', '../../gitcode/qwen7b-source/modeling_qwen.py'],
        ['Qwen7B generation config', '../../gitcode/qwen7b-source/generation_config.json'],
      ],
    },
    qwen3: {
      title: 'Qwen3-8B 架构与训练证据',
      meta: 'Dense decoder · Mcore · 4K · A3 脚本',
      chip: 'Qwen3',
      graph: graphs.qwen3,
      insight: 'Qwen3-8B 是最适合做训练任务向导首屏的真实案例。右图把模型主干和脚本里的结构参数对应起来，用户能知道自己为什么要填 tokenizer、权重、数据和 TP/PP。',
      nodeNotes: {
        tokens: '训练数据经过 tokenizer 后进入模型，形态是 token id。',
        embed: 'Qwen3-8B 脚本里 padded vocab size 是 151936，Embedding 将 token id 转为 hidden states。',
        attention: '脚本开启 flash attention、rotary position embedding 和 group-query attention。',
        mlp: 'Qwen3 使用 SwiGLU MLP，脚本开启 fused swiglu。',
        block_out: '36 个 decoder layer 折叠成一个重复层模板，避免展开成几十个重复节点。',
        lm_head: 'LM Head 输出词表 logits，用于 next token prediction loss。',
      },
      evidence: [
        ['模型结构', '36 layers、hidden size 4096、ffn hidden size 12288、32 attention heads。'],
        ['训练规模', 'A3 脚本中 NPUS_PER_NODE 16、NNODES 1、TP 2、PP 1。'],
        ['序列与 batch', 'SEQ_LENGTH 4096、MBS 1、GBS 128、TRAIN_ITERS 2000。'],
        ['关键开关', 'flash attention、fused rotary、fused swiglu、fused rmsnorm、distributed optimizer。'],
      ],
      sources: [
        ['Qwen3-8B A3 pretrain', `${SRC}examples/mcore/qwen3/pretrain_qwen3_8b_4K_ptd_A3.sh`],
        ['Qwen3 data convert', `${SRC}examples/mcore/qwen3/data_convert_qwen3_pretrain.sh`],
      ],
    },
    qwen3moe: {
      title: 'Qwen3-MoE 30B-A3B 架构与 DPO 证据',
      meta: 'MoE · Router · DPO · 16K · Mcore',
      chip: 'MoE',
      graph: graphs.qwen3moe,
      insight: 'MoE 模型让“架构图”和“并行参数”直接绑定：router/top-k/expert 对应 EP 和 all-to-all 通信，DPO 又把 chosen/rejected 数据接入后训练损失。',
      nodeNotes: {
        router: 'Router 为每个 token 计算 expert 路由分数。',
        topk: 'Qwen3-MoE 脚本中 router top-k 为 8。',
        expert: 'Expert MLP 对应 128 experts，训练时会引入 EP 和 all-to-all 通信。',
        combine: 'Expert 输出按路由权重合并回 token hidden state。',
        chosen: 'DPO pairwise 数据中的偏好答案。',
        rejected: 'DPO pairwise 数据中的非偏好答案。',
        loss: 'DPO 入口是 posttrain_gpt.py，stage=dpo，loss type 为 sigmoid。',
      },
      evidence: [
        ['MoE 参数', '128 experts、top-k 8、moe grouped gemm、alltoall_seq dispatcher。'],
        ['预训练拓扑', 'pretrain 脚本中 NNODES 2、NPUS_PER_NODE 8、TP 1、PP 2、EP 8。'],
        ['DPO 拓扑', 'DPO 脚本中 NNODES 2、NPUS_PER_NODE 16、TP 2、PP 8、SEQ_LENGTH 16384。'],
        ['数据任务', 'DPO 使用 pairwise 数据，训练入口是 posttrain_gpt.py。'],
      ],
      sources: [
        ['Qwen3-MoE pretrain', `${SRC}examples/mcore/qwen3_moe/pretrain_qwen3_30b_a3b_4K_ptd.sh`],
        ['Qwen3-MoE DPO', `${SRC}examples/mcore/qwen3_moe/dpo_qwen3_30b_a3b_16K_A3_ptd.sh`],
        ['Qwen3-MoE FSDP2 A5 quant', `${SRC}examples/fsdp2/qwen3_moe/pretrain_qwen3_30b_4k_fsdp2_quant_A5.sh`],
      ],
    },
    deepseek: {
      title: 'DeepSeek-V3.2 671B 架构与并行证据',
      meta: 'MLA · Sparse Indexer · MoE · MTP · 32x16 A3',
      chip: 'DeepSeek',
      graph: graphs.deepseek,
      insight: 'DeepSeek-V3.2 是专家模式样板。MLA、Sparse Indexer、MoE、MTP 和超大并行拓扑必须一起解释，单独列参数无法帮助用户判断性能瓶颈。',
      nodeNotes: {
        mla: 'MLA 通过 latent 表示降低注意力 KV 表示和长上下文成本。',
        indexer: 'Sparse Indexer / DSA 选择相关 token，配合 sparse flash attention。',
        router: 'Router 决定 token 进入哪些 expert；DeepSeek 脚本中 router top-k 为 8。',
        moe: 'DeepSeek-V3.2 训练脚本中 num experts 为 256，EP 为 64。',
        mtp: 'MTP 作为多 token prediction 路径，脚本中 mtp-num-layers 为 1。',
        norm: 'Final Norm 后接 LM Head 输出 logits。',
        logits: '输出 logits 用于预训练 next token prediction loss。',
      },
      evidence: [
        ['模型规模', '671B，脚本中 NUM_LAYERS 64、hidden size 7168、attention heads 128。'],
        ['集群规模', 'NNODES 32、NPUS_PER_NODE 16，支持表中标注 32x16。'],
        ['并行参数', 'TP 4、PP 8、EP 64、CP 1，GBS 7680。'],
        ['专家特性', '256 experts、router top-k 8、MoE all-to-all、MTP、DSA indexer。'],
      ],
      sources: [
        ['DeepSeek-V3.2 pretrain', `${SRC}examples/mcore/deepseek32/pretrain_deepseek32_671b_4k_A3_ptd.sh`],
        ['DeepSeek V3.2 architecture schema', '../../pto-design-system/patterns/model-graphviz/assets/deepseek_v32_model_architecture.json'],
      ],
    },
    rlhf: {
      title: 'Qwen2.5-7B GRPO 训练系统图',
      meta: 'RLHF · Ray · rollout · reward · policy update',
      chip: 'GRPO',
      graph: graphs.rlhf,
      insight: 'GRPO 不是单个模型结构图能讲清的内容。它需要把 policy、rollout、reward/reference 和训练器更新画成系统架构，才能解释 Ray、网卡、资源和日志问题。',
      nodeNotes: {
        prompt: 'GRPO 训练输入 prompt 数据，驱动 policy 生成候选输出。',
        policy: 'Policy Model 是被更新的 Qwen2.5-7B。',
        rollout: 'Ray 负责协调 rollout 与训练资源，脚本显式启动 Ray head/worker。',
        reward: 'Reward signal 或规则反馈用于构造优化目标。',
        reference: 'Reference Model 提供约束或对比信号，避免策略漂移。',
        grpo: 'GRPO Update 汇总 rollout、reward/reference 信号并更新 policy。',
        ckpt: '训练输出是更新后的 policy checkpoint。',
      },
      evidence: [
        ['模型案例', '脚本名为 grpo_qwen25_7b_A3.sh，对应 Qwen2.5-7B GRPO。'],
        ['启动系统', '先清理 python 与 Ray，再启动 Ray head 或 worker。'],
        ['硬件参数', 'NNODES 1、NPUS_PER_NODE 16，Ray resource 标记为 NPU。'],
        ['体验重点', 'MASTER_ADDR、SOCKET_IFNAME、Ray status 和 Hydra config 是主要体检点。'],
      ],
      sources: [
        ['Qwen2.5-7B GRPO', `${SRC}examples/rlhf/grpo/grpo_qwen25_7b_A3.sh`],
      ],
    },
    breakpoints: {
      title: '关键体验断点图',
      meta: '后端、权重、数据、并行、rank、诊断路径',
      chip: 'DX',
      graph: graphs.breakpoints,
      insight: '断点图的产品含义是：启动训练前就应该把隐性约束显性化，而不是等用户在日志里定位。',
      nodeNotes: {
        backend: '后端决定脚本形态：Mcore shell、FSDP2 YAML 或 MindSpore-Mcore。',
        weights: '权重格式与并行切分不一致会导致加载失败或训练异常。',
        data: 'DATA_PATH 常填预处理产物前缀，不是完整 .bin 文件名。',
        parallel: 'TP/PP/EP/CP/FSDP 共同决定显存、通信和权重切分。',
        launch: '多机启动需要检查 MASTER_ADDR、NODE_RANK、端口、网卡和 HCCL。',
        diagnose: '诊断路径应该把日志错误归因到环境、数据、权重、并行或算子性能。',
      },
      evidence: [
        ['后端选择', 'Mcore、FSDP2、MindSpore-Mcore 对数据、权重和配置格式有不同要求。'],
        ['权重格式', 'HF、Mcore、LoRA merge、FSDP2 原生路径应在 UI 中明确。'],
        ['并行拓扑', 'TP/PP/EP/CP/FSDP 应可视化解释切模型、切专家、切上下文和切数据。'],
        ['诊断工具', 'loss、OOM、通信慢和启动失败应能回到具体检查项。'],
      ],
      sources: [
        ['Training install guide', `${SRC}docs/zh/pytorch/training/install_guide.md`],
        ['FSDP2 finetune doc', `${SRC}docs/zh/pytorch/training/finetune/fsdp2/finetune_fsdp2.md`],
      ],
    },
    mvp: {
      title: '可视化训练工作台 MVP',
      meta: '任务向导 · 脚本体检器 · 架构视图 · 运行解释器',
      chip: 'MVP',
      graph: graphs.mvp,
      insight: 'MVP 要围绕真实案例闭环，不做泛泛平台。先让用户用 Qwen3-8B 成功跑首个 step，再用 Qwen3-MoE 和 DeepSeek-V3.2 解释进阶复杂度。',
      nodeNotes: {
        wizard: '任务向导输入模型、任务、卡型、节点数，输出推荐脚本和准备清单。',
        checker: '脚本体检器读取 shell/yaml，检查路径、DATA_PATH、TP/PP/EP/CP、rank 和端口。',
        graph: '架构视图解释脚本参数来自哪里，哪些和模型结构、硬件规模强相关。',
        explainer: '运行解释器读取日志、profiler、msTT 输出，并给出下一步动作。',
        mtt: 'msTT 和 profiler 是专家证据层，不应该和训练主流程割裂。',
      },
      evidence: [
        ['训练任务向导', '输入模型、任务、卡型、节点数，输出后端和脚本建议。'],
        ['脚本体检器', '读取 shell/yaml，检查路径、rank、DATA_PATH、TP/PP/EP/CP。'],
        ['架构视图', '把模型结构和训练参数同屏显示，解释参数为什么存在。'],
        ['运行解释器', '把日志、profiler、msTT 结果转成下一步动作。'],
      ],
      sources: [
        ['Developer experience document', '../../gitcode/ascend-training-developer-experience.md'],
      ],
    },
  };

  const body = document.body;
  const readerScroll = document.getElementById('readerScroll');
  const graphStage = document.getElementById('graphStage');
  const visualTitle = document.getElementById('visualTitle');
  const visualMeta = document.getElementById('visualMeta');
  const visualChip = document.getElementById('visualChip');
  const visualInsight = document.getElementById('visualInsight');
  const evidenceGrid = document.getElementById('evidenceGrid');
  const sourceList = document.getElementById('sourceList');
  const nodeTooltip = document.getElementById('nodeTooltip');
  const readerSections = Array.from(document.querySelectorAll('[data-visual-section]'));
  let activeVisual = 'ecosystem';
  let lockUntil = 0;
  let scrollSyncFrame = 0;

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function renderEvidence(items) {
    evidenceGrid.innerHTML = (items || []).map(([label, value]) => `
      <div class="mental-evidence-item">
        <strong>${label}</strong>
        <span>${value}</span>
      </div>
    `).join('');
  }

  function renderSources(items) {
    sourceList.innerHTML = (items || []).map(([label, href]) => `
      <a href="${href}">
        <small>${label}</small>
        <code>${href.replace(/^\.\.\/\.\.\//, '/Users/yin/')}</code>
      </a>
    `).join('');
  }

  function placeTooltip(event) {
    if (!nodeTooltip || nodeTooltip.hidden) return;
    const bodyRect = graphStage.parentElement.getBoundingClientRect();
    const tooltipRect = nodeTooltip.getBoundingClientRect();
    const gap = 14;
    const maxX = Math.max(gap, bodyRect.width - tooltipRect.width - gap);
    const maxY = Math.max(gap, bodyRect.height - tooltipRect.height - gap);
    let x = event.clientX - bodyRect.left + gap;
    let y = event.clientY - bodyRect.top + gap;
    if (x > maxX) x = event.clientX - bodyRect.left - tooltipRect.width - gap;
    if (y > maxY) y = event.clientY - bodyRect.top - tooltipRect.height - gap;
    nodeTooltip.style.left = `${Math.max(gap, Math.min(maxX, x))}px`;
    nodeTooltip.style.top = `${Math.max(gap, Math.min(maxY, y))}px`;
  }

  function showTooltip(event, visual, node) {
    if (!nodeTooltip || !node) return;
    const note = visual.nodeNotes?.[node.id] || visual.insight;
    nodeTooltip.innerHTML = `
      <strong>${escapeHtml(node.label || node.id)}</strong>
      <span>${escapeHtml(node.typeLabel || node.kind || 'Node')}</span>
      <span>${escapeHtml(note)}</span>
    `;
    nodeTooltip.hidden = false;
    placeTooltip(event);
  }

  function hideTooltip() {
    if (!nodeTooltip) return;
    nodeTooltip.hidden = true;
  }

  function bindGraphTooltips(visual) {
    hideTooltip();
    const nodeElements = Array.from(graphStage.querySelectorAll('.pto-model-graphviz-node'));
    nodeElements.forEach((element, index) => {
      const node = visual.graph.nodes?.[index];
      if (!node) return;
      element.dataset.nodeId = node.id;
      element.addEventListener('mouseenter', (event) => showTooltip(event, visual, node));
      element.addEventListener('mousemove', placeTooltip);
      element.addEventListener('mouseleave', hideTooltip);
      const label = node.label || node.id;
      const note = visual.nodeNotes?.[node.id] || '';
      element.setAttribute('aria-label', `${label} ${note}`.trim());
    });
    graphStage.addEventListener('mouseleave', hideTooltip, { once: true });
  }

  function setSelectedButtons(key) {
    document.querySelectorAll('[data-visual-target]').forEach((button) => {
      const selected = button.getAttribute('data-visual-target') === key;
      button.classList.toggle('is-selected', selected);
      if (selected) {
        button.setAttribute('aria-current', 'true');
      } else {
        button.removeAttribute('aria-current');
      }
    });
  }

  function setActiveSections(key) {
    document.querySelectorAll('[data-visual-section]').forEach((section) => {
      section.classList.toggle('is-active', section.getAttribute('data-visual-section') === key);
    });
  }

  function renderVisual(key, options) {
    const visual = visuals[key] || visuals.ecosystem;
    activeVisual = key;
    body.dataset.activeVisual = key;
    visualTitle.textContent = visual.title;
    visualMeta.textContent = visual.meta;
    visualChip.textContent = visual.chip;
    visualInsight.textContent = visual.insight;
    renderEvidence(visual.evidence);
    renderSources(visual.sources);
    setSelectedButtons(key);
    setActiveSections(key);

    if (window.PtoModelGraphvizPattern?.render) {
      window.PtoModelGraphvizPattern.render(graphStage, visual.graph, {
        ariaLabel: `${visual.title} visualization`,
      });
      bindGraphTooltips(visual);
    } else {
      graphStage.innerHTML = '<div class="mental-insight">model-graphviz renderer 未加载。</div>';
    }

    if (options?.scrollTarget) {
      const target = document.getElementById(options.scrollTarget);
      if (target) {
        lockUntil = window.performance.now() + 900;
        target.scrollIntoView({ block: 'start', behavior: 'smooth' });
      }
    }
  }

  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-visual-target]');
    if (!button) return;
    renderVisual(button.getAttribute('data-visual-target'), {
      scrollTarget: button.getAttribute('data-section-target'),
    });
  });

  function syncActiveSectionFromScroll() {
    scrollSyncFrame = 0;
    if (!readerScroll || window.performance.now() < lockUntil) return;
    const rootRect = readerScroll.getBoundingClientRect();
    const visible = readerSections
      .map((section) => {
        const rect = section.getBoundingClientRect();
        const visibleHeight = Math.min(rect.bottom, rootRect.bottom) - Math.max(rect.top, rootRect.top);
        return {
          section,
          visibleHeight,
          distance: Math.abs(rect.top - rootRect.top),
        };
      })
      .filter((item) => item.visibleHeight > 0)
      .sort((a, b) => a.distance - b.distance || b.visibleHeight - a.visibleHeight)[0];
    const key = visible?.section.getAttribute('data-visual-section');
    if (!key || key === activeVisual) return;
    renderVisual(key);
  }

  function requestScrollSync() {
    if (scrollSyncFrame) return;
    scrollSyncFrame = window.requestAnimationFrame(syncActiveSectionFromScroll);
  }

  if (readerScroll) {
    readerScroll.addEventListener('scroll', requestScrollSync, { passive: true });
  }

  if ('IntersectionObserver' in window && readerScroll) {
    const observer = new IntersectionObserver(requestScrollSync, {
      root: readerScroll,
      threshold: [0.05, 0.35, 0.65],
    });
    readerSections.forEach((section) => observer.observe(section));
  }

  window.ascendTrainingMentalModel = {
    renderVisual,
    visuals,
  };

  renderVisual(activeVisual);
})();
