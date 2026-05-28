#!/usr/bin/env python3
# coding: utf-8
"""
Reverse-build the operator-source bridge JSON for the DeepSeek V3.2 profiling
workbench. Real input would be the profiling tool's JSON; here we assemble it
from the actual DeepSeek V3.2 inference source plus the semantic annotations
extracted from ds3.2_report (9).html.

Output: deepseek_v32_operator_source.json
"""
import json
import os

SRC = "/Users/yin/gitcode/deepseekv3.2源码"
OUT = "/Users/yin/pto/graphviz/deepseek_v32_operator_source.json"

FILES = [
    {
        "id": "model",
        "name": "model.py",
        "path": os.path.join(SRC, "inference_副本/model.py"),
        "lang": "python",
        "desc": "DeepSeek V3.2 推理主模型（Transformer / Block / MLA / MoE）",
        "url": "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py",
    },
    {
        "id": "indexer",
        "name": "lightning_indexer_prolog_quant.py",
        "path": os.path.join(SRC, "lightning_indexer_prolog_quant.py"),
        "lang": "python",
        "desc": "昇腾 Lightning 稀疏索引 prolog-quant 算子（pypto）",
        "url": "https://gitcode.com/cann/cann-recipes-infer",
    },
    {
        "id": "kernel",
        "name": "kernel.py",
        "path": os.path.join(SRC, "inference_副本/kernel.py"),
        "lang": "python",
        "desc": "FP8 量化 / GEMM / index 的 TileLang kernel",
        "url": "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py",
    },
]

# annotation: line -> mapped architecture node + tag + semantic comment
# semantic text adapted from ds3.2_report (9).html node-semantic data
ANNOTATIONS = {
    "model": [
        {"line": 18, "nodeId": "", "tag": "Config",
         "semantic": "模型超参数：dim=2048、n_layers=27、n_routed_experts=64、n_activated_experts=6、index_topk=2048。架构图所有节点的形状都源自这里。"},
        {"line": 92, "nodeId": "dsv32arch_embedding_2f81ba1b", "tag": "Embedding",
         "semantic": "Main-model token embedding：按 world_size 切分词表的 ParallelEmbedding，world_size>1 时通过 all_reduce 恢复完整 hidden state。"},
        {"line": 110, "nodeId": "dsv32arch_embedding_2f81ba1b", "tag": "Embedding",
         "semantic": "Embedding forward：vocab-partial GatherV2 查表 + mask-out 无效位置 + all_reduce。报告里该段约 46 us、10 kernel，不是性能风险。"},
        {"line": 272, "nodeId": "", "tag": "Norm",
         "semantic": "RMSNorm：融合 residual add 的 RMS 归一化。对应 input_layernorm / ffn_norm / final norm，全图反复出现。"},
        {"line": 435, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "Lightning 稀疏索引器：为 sparse attention 生成 top-k key 索引（Indexer.forward → topk_indices）。对齐 LightningIndexerQuant，报告判定正常。"},
        {"line": 457, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "Indexer forward：wq_b/wk 投影 → rope → hadamard → fp8 量化 → fp8_index 打分 → topk。总耗时约 6,370 us。"},
        {"line": 483, "nodeId": "dsv32arch_topk_e1fd5d33", "tag": "Sparse Index",
         "semantic": "top-k index ≤ 2048：Indexer 到 sparse attention 的中间结果，更适合作为路径锚点而非单独瓶颈。"},
        {"line": 498, "nodeId": "dsv32arch_attention_call_612a5ae9", "tag": "MLA",
         "semantic": "Multi-Head Latent Attention：W_kv_b 吸收 + Lightning 稀疏索引 + KvQuantSparseFlashAttention。对齐 QuantBatchMatmulV3、MlaPrologV3。"},
        {"line": 545, "nodeId": "dsv32arch_attention_call_612a5ae9", "tag": "MLA",
         "semantic": "MLA forward：prefill 走 MHA、decode 走 MQA；两条路径都调用 indexer 产出 sparse mask。"},
        {"line": 560, "nodeId": "dsv32arch_q_path_b399f580", "tag": "Q Path",
         "semantic": "Q path：wq_a → q_norm → wq_b。对齐 QuantBatchMatmulV3（MIX_AIC 270 次），需区分小 shape 低 cube 利用率与真实异常。"},
        {"line": 583, "nodeId": "dsv32arch_sparse_attn_f5c98e92", "tag": "Sparse Attn",
         "semantic": "KvQuantSparseFlashAttention：用 indexer top-k 索引做 Q×sparse_K×sparse_V。总耗时约 9,222 us，报告判定正常但仍是路径关键段。"},
        {"line": 611, "nodeId": "dsv32arch_dense_c8e834d8", "tag": "Dense FFN",
         "semantic": "DenseMLP：SwiGLU 前馈（w1/w3 上投影 → SiLU 门控 → w2 下投影）。只覆盖 layer 0，约 812 us、34 kernel。"},
        {"line": 646, "nodeId": "dsv32arch_router_77eb1db6", "tag": "MoE Router",
         "semantic": "Gate：hidden → n_routed_experts 打分 → top-k 专家选择。对应 MoeGatingTopK / MoeDistributeDispatchV2，约 2,972 us。"},
        {"line": 712, "nodeId": "dsv32arch_experts_cfbede3c", "tag": "Routed Experts",
         "semantic": "Expert：分组批量专家计算 gate_up（GroupedMatmul）→ SwiGLU → down。GroupedMatmul 是全局 top-1 算子，约 29,903 us。"},
        {"line": 747, "nodeId": "dsv32arch_moe_63e885ca", "tag": "MoE FFN",
         "semantic": "Mixture-of-Experts：gate 路由 + shared_experts + dispatch + routed_experts + combine。占 top 算子 21.84%，报告标注正常。"},
        {"line": 778, "nodeId": "dsv32arch_shared_d18aac96", "tag": "Shared Experts",
         "semantic": "Shared experts：MoE 的常驻 FFN 分支，与 routed experts 并行在独立 stream 上。"},
        {"line": 807, "nodeId": "dsv32arch_block_0214b4b3", "tag": "Decoder Block",
         "semantic": "Transformer Block：fused residual RMSNorm → MLA → ffn_norm → FFN(dense 或 MoE)。重复 27 次，是主要耗时段。"},
        {"line": 854, "nodeId": "dsv32arch_transformer_9236e026", "tag": "Transformer",
         "semantic": "Transformer Core：embed → 27 层 Block → final norm → lm_head。正常步均值 HOST_BOUND，Free Time 34.4%。"},
        {"line": 886, "nodeId": "dsv32arch_lm_head_2b98b847", "tag": "LM Head",
         "semantic": "lm_head：末端 vocab-parallel 投影到 102400 词表。报告里约 192 us、7 kernel，不是主要性能风险。"},
        {"line": 908, "nodeId": "dsv32arch_logits_c7d6c722", "tag": "Wait Anchor",
         "semantic": "logits 计算：报告的 WAIT_ANCHOR_FALSE_HOTSPOT 在此附近——OnesLike wait 99.7%，是等待锚点不是真实 compute 热点。"},
    ],
    "indexer": [
        {"line": 102, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "Key-LayerNorm：indexer prolog 对 K 做的归一化（pypto 语义标签 Key-LayerNorm）。"},
        {"line": 146, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "Prolog-Quant：INT8 量化，对应报告里的 LightningIndexerQuant kernel。"},
        {"line": 205, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "昇腾 pypto 实现的 indexer prolog-quant 主体：mark_dynamic 动态轴 + pass options + 分档循环。"},
        {"line": 252, "nodeId": "dsv32arch_topk_e1fd5d33", "tag": "Sparse Index",
         "semantic": "按 token 分档 unroll 循环：Query/Key Linear → Rope → Hadamard → Quant，产出 sparse index 的输入。"},
        {"line": 363, "nodeId": "dsv32arch_indexer_dfa17da7", "tag": "Indexer",
         "semantic": "@pypto.jit 入口：编译 lightning_indexer_prolog_quant kernel。"},
    ],
    "kernel": [
        {"line": 37, "nodeId": "dsv32arch_q_path_b399f580", "tag": "Q Path",
         "semantic": "act_quant_kernel：block-wise FP8 量化 kernel（TileLang），把 bf16 激活量化到 float8_e4m3。"},
        {"line": 87, "nodeId": "dsv32arch_q_path_b399f580", "tag": "Q Path",
         "semantic": "act_quant：FP8 量化的 host 封装，linear() 在 fp8 路径上调用它。"},
        {"line": 114, "nodeId": "dsv32arch_dense_c8e834d8", "tag": "Dense FFN",
         "semantic": "fp8_gemm_kernel：FP8 矩阵乘 kernel——报告 top 算子 MatMul / QuantBatchMatmulV3 的底层实现。"},
        {"line": 199, "nodeId": "dsv32arch_topk_e1fd5d33", "tag": "Sparse Index",
         "semantic": "fp8_index_kernel：FP8 index-score kernel——q@k → relu → 加权求和，产出 indexer 的打分。"},
        {"line": 254, "nodeId": "dsv32arch_topk_e1fd5d33", "tag": "Sparse Index",
         "semantic": "fp8_index：index-score 的 host 封装，Indexer.forward 调用它得到 index_score。"},
    ],
}


def build():
    files_out = []
    for f in FILES:
        with open(f["path"], encoding="utf-8") as fh:
            lines = fh.read().split("\n")
        anns = sorted(ANNOTATIONS.get(f["id"], []), key=lambda a: a["line"])
        files_out.append({
            "id": f["id"],
            "name": f["name"],
            "lang": f["lang"],
            "desc": f["desc"],
            "url": f["url"],
            "lineCount": len(lines),
            "lines": lines,
            "annotations": anns,
        })
    data = {
        "schemaVersion": "1.0",
        "inputContract": "profiling_json (operator source bridge)",
        "note": "理想输入是 profiling 工具产出的 JSON；此文件由 DeepSeek V3.2 真实推理源码 + ds3.2_report 语义注释反向组装。",
        "files": files_out,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    total_ann = sum(len(f["annotations"]) for f in files_out)
    print(f"wrote {OUT}")
    print(f"  {len(files_out)} files, {total_ann} annotations")
    for f in files_out:
        print(f"  - {f['name']}: {f['lineCount']} lines, {len(f['annotations'])} annotations")


if __name__ == "__main__":
    build()
