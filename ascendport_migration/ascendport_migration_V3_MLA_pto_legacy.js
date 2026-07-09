// Extracted unchanged from ascendport_migration_V3_MLA.html for PTO shell refresh.
// Business data/state machine remains page-owned; visual shell lives in ascendport_migration_V3_MLA_pto.html.
/* ============================ 源代码 & 产物 ============================ */
const CUDA = String.raw`// flash_mla_decode.cu
// DeepSeek-V3 · Flash MLA Sparse Decode  ——  稀疏注意力解码核
// O[b,q,h] = Softmax( Q[b,q,h]·K[indices]^T / √d ) · V[indices]
// 带 FP8 KV Cache + TopK 稀疏索引 + Sink Token 支持
#include <cuda_fp8.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

constexpr int BLOCK_M   = 1;       // 每个 block 处理 1 个 query
constexpr int BLOCK_N   = 128;     // key 分块大小
constexpr int HEAD_DIM  = 576;     // DeepSeek-V3 head_dim (qk)
constexpr int HEAD_DIM_V= 512;     // value head_dim
constexpr int WARP      = 32;
constexpr int BLOCK_SIZE= 256;

using fp8   = __nv_fp8_e4m3;
using fp8x4 = __nv_fp8x4_e4m3;

// ---- 线程块级 Softmax 规约(依赖 warp shuffle + shared mem)-------------------
__device__ __forceinline__
float block_reduce_max(float val, float* shared) {
    const int lane = threadIdx.x & (WARP - 1);
    const int wid  = threadIdx.x / WARP;
    // warp 内规约
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    // 跨 warp 规约
    val = (threadIdx.x < BLOCK_SIZE / WARP) ? shared[threadIdx.x] : -CUDART_INF_F;
    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP / 2; offset > 0; offset /= 2)
            val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__
float block_reduce_sum(float val, float* shared) {
    const int lane = threadIdx.x & (WARP - 1);
    const int wid  = threadIdx.x / WARP;
    #pragma unroll
    for (int offset = WARP / 2; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < BLOCK_SIZE / WARP) ? shared[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int offset = WARP / 2; offset > 0; offset /= 2)
            val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    __syncthreads();
    return val;
}

// ---- Flash MLA Sparse Decode 主核 --------------------------------
extern "C" __global__ void flash_mla_sparse_decode_kernel(
        const fp8*   __restrict__ q,             // [B, num_heads_q, HEAD_DIM]
        const fp8*   __restrict__ kv_cache,      // [num_blocks, page_size, num_heads_k, 656]
        const int*   __restrict__ indices,       // [B, topk] 稀疏索引
        const float* __restrict__ attn_sink,     // [num_heads_q] sink token 权重
        float*       __restrict__ out,           // [B, num_heads_q, HEAD_DIM_V]
        float*       __restrict__ lse,           // [B, num_heads_q] log-sum-exp
        int B, int num_heads_q, int topk,
        int page_size, float softmax_scale)
{
    cg::thread_block cta = cg::this_thread_block();
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;

    __shared__ float s_qk[BLOCK_N];              // 共享内存: QK^T 分数
    __shared__ float s_reduce[BLOCK_SIZE / WARP];// 规约临时空间
    __shared__ float s_max, s_sum;               // Softmax 统计量

    // 加载 Q (本 head 的 query 向量)
    const fp8* q_ptr = q + (batch_idx * num_heads_q + head_idx) * HEAD_DIM;
    float local_q[HEAD_DIM / BLOCK_SIZE];        // 寄存器分块存 Q
    #pragma unroll
    for (int i = 0; i < HEAD_DIM / BLOCK_SIZE; ++i) {
        int offset = tid + i * BLOCK_SIZE;
        if (offset < HEAD_DIM) local_q[i] = float(q_ptr[offset]);
    }

    float m_prev = -CUDART_INF_F;                // 在线 Softmax 最大值
    float l_prev = 0.f;                          // 在线 Softmax 累加器
    float acc[HEAD_DIM_V / BLOCK_SIZE] = {0};    // 输出累加器 (寄存器)

    // 分块遍历 TopK 个 KV (稀疏)
    for (int tile = 0; tile < (topk + BLOCK_N - 1) / BLOCK_N; ++tile) {
        int tile_start = tile * BLOCK_N;
        int tile_size  = min(BLOCK_N, topk - tile_start);

        // 计算 QK^T (每个线程负责一部分 key)
        for (int k = tid; k < tile_size; k += BLOCK_SIZE) {
            int kv_idx = indices[batch_idx * topk + tile_start + k];
            // 解析 FP8 KV cache: 前 512B NoPE + 16B scale + 128B RoPE
            const fp8* k_ptr = kv_cache + kv_idx * 656;  // 简化:实际需解析 page_block
            float dot = 0.f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 4) {
                fp8x4 qv = *reinterpret_cast<const fp8x4*>(q_ptr + d);
                fp8x4 kv = *reinterpret_cast<const fp8x4*>(k_ptr + d);
                dot += float(qv.x)*float(kv.x) + float(qv.y)*float(kv.y)
                     + float(qv.z)*float(kv.z) + float(qv.w)*float(kv.w);
            }
            s_qk[k] = dot * softmax_scale;
        }
        cta.sync();

        // 在线 Softmax: 更新 max 与 sum
        float m_curr = -CUDART_INF_F;
        for (int k = tid; k < tile_size; k += BLOCK_SIZE)
            m_curr = fmaxf(m_curr, s_qk[k]);
        m_curr = block_reduce_max(m_curr, s_reduce);
        if (tid == 0) s_max = fmaxf(m_prev, m_curr);
        cta.sync();

        float alpha = expf(m_prev - s_max);
        float m_new = s_max;
        for (int i = 0; i < HEAD_DIM_V / BLOCK_SIZE; ++i)
            acc[i] *= alpha;

        float local_sum = 0.f;
        for (int k = tid; k < tile_size; k += BLOCK_SIZE) {
            s_qk[k] = expf(s_qk[k] - m_new);
            local_sum += s_qk[k];
        }
        local_sum = block_reduce_sum(local_sum, s_reduce);
        if (tid == 0) s_sum = l_prev * alpha + local_sum;
        cta.sync();

        // 累加 V
        for (int k = 0; k < tile_size; ++k) {
            int kv_idx = indices[batch_idx * topk + tile_start + k];
            const fp8* v_ptr = kv_cache + kv_idx * 656 + 512 + 16; // 跳到 V 部分
            float weight = s_qk[k];
            for (int i = 0; i < HEAD_DIM_V / BLOCK_SIZE; ++i) {
                int offset = tid + i * BLOCK_SIZE;
                if (offset < HEAD_DIM_V)
                    acc[i] += weight * float(v_ptr[offset]);
            }
        }
        m_prev = m_new;
        l_prev = s_sum;
        cta.sync();
    }

    // 写出结果
    float* out_ptr = out + (batch_idx * num_heads_q + head_idx) * HEAD_DIM_V;
    for (int i = 0; i < HEAD_DIM_V / BLOCK_SIZE; ++i) {
        int offset = tid + i * BLOCK_SIZE;
        if (offset < HEAD_DIM_V)
            out_ptr[offset] = acc[i] / l_prev;
    }
    if (tid == 0) lse[batch_idx * num_heads_q + head_idx] = logf(l_prev) + m_prev;
}
`;

const S3 = String.raw`// flash_mla_decode.cpp · AscendC 核  (AscendPort · S3 自动生成)
// 由 flash_mla_sparse_decode_kernel 迁移 —— SIMT grid → 分核 SPMD
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HEAD_DIM   = 576;
constexpr int32_t HEAD_DIM_V = 512;
constexpr int32_t BLOCK_N    = 128;

class FlashMLADecode {
public:
    __aicore__ inline FlashMLADecode() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kvCache, GM_ADDR indices,
                                GM_ADDR attnSink, GM_ADDR out, GM_ADDR lse,
                                int32_t B, int32_t numHeads, int32_t topk,
                                int32_t pageSize, float softmaxScale) {
        // CUDA: blockIdx.x = batch, blockIdx.y = head  →  昇腾:按 AI Core 数切分 (batch, head) 对
        this->batchIdx = GetBlockIdx() / numHeads;
        this->headIdx  = GetBlockIdx() % numHeads;
        this->B = B;  this->numHeads = numHeads;  this->topk = topk;
        this->softmaxScale = softmaxScale;
        qGm.SetGlobalBuffer((__gm__ fp8_t*)q);
        kvCacheGm.SetGlobalBuffer((__gm__ fp8_t*)kvCache);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        outGm.SetGlobalBuffer((__gm__ float*)out);
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        // TODO(S4): 分配 L1 / L0A / L0B / L0C / UB,插入逐级 DataCopy
        // TODO(S5): 沿 key(topk 维)选择分块长度
    }
    __aicore__ inline void Process() {
        if (batchIdx >= B || headIdx >= numHeads) return;
        ComputeAttention();     // QK^T (Cube) → Softmax (Vector) → 累加 V
    }
private:
    // TODO(S4): QK^T 走 Cube, Softmax 与 V 累加走 Vector
    __aicore__ inline void ComputeAttention() { /* 待 S4 填充 */ }
    // TODO(S6): 替代 block_reduce_max/sum (warp shuffle) → Vector 规约

    GlobalTensor<fp8_t>   qGm, kvCacheGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<float>   outGm, lseGm;
    int32_t batchIdx, headIdx, B, numHeads, topk;
    float softmaxScale;
};

extern "C" __global__ __aicore__ void flash_mla_sparse_decode(
        GM_ADDR q, GM_ADDR kvCache, GM_ADDR indices,
        GM_ADDR attnSink, GM_ADDR out, GM_ADDR lse, GM_ADDR tiling) {
    FlashMLADecode op;
    op.Init(q, kvCache, indices, attnSink, out, lse, /*B*/0, /*numHeads*/0, /*topk*/0, /*pageSize*/0, /*scale*/1.0f);
    op.Process();
}
`;

const S4 = String.raw`// flash_mla_decode.cpp · AscendC 核  (AscendPort · S4 内存层次已注入)
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HEAD_DIM   = 576;
constexpr int32_t HEAD_DIM_V = 512;
constexpr int32_t BLOCK_N    = 128;

class FlashMLADecode {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kvCache, GM_ADDR indices,
                                GM_ADDR attnSink, GM_ADDR out, GM_ADDR lse,
                                int32_t B, int32_t numHeads, int32_t topk,
                                int32_t pageSize, float softmaxScale, int32_t nTile) {
        this->batchIdx = GetBlockIdx() / numHeads;
        this->headIdx  = GetBlockIdx() % numHeads;
        this->B = B; this->numHeads = numHeads; this->topk = topk;
        this->softmaxScale = softmaxScale; this->nTile = nTile;
        qGm.SetGlobalBuffer((__gm__ fp8_t*)q);
        kvCacheGm.SetGlobalBuffer((__gm__ fp8_t*)kvCache);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        outGm.SetGlobalBuffer((__gm__ float*)out);
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        // === 片上缓冲层次(S4 注入)===
        pipe.InitBuffer(qL1,  1, HEAD_DIM * sizeof(fp8_t));          // Q: GM→L1→L0A
        pipe.InitBuffer(kL1,  1, BLOCK_N * HEAD_DIM * sizeof(fp8_t));// K: GM→L1→L0B
        pipe.InitBuffer(vL1,  1, BLOCK_N * HEAD_DIM_V * sizeof(fp8_t));// V: GM→L1
        pipe.InitBuffer(cO,   1, BLOCK_N * sizeof(float));           // QK^T logits: L0C
        pipe.InitBuffer(ubQK, 1, BLOCK_N * sizeof(float));           // Softmax 中间: UB
        pipe.InitBuffer(ubOut,1, HEAD_DIM_V * sizeof(float));        // 输出累加: UB
    }
    __aicore__ inline void Process() {
        if (batchIdx >= B || headIdx >= numHeads) return;
        // 加载 Q
        LocalTensor<fp8_t> qLoc = qL1.AllocTensor<fp8_t>();
        DataCopy(qLoc, qGm[(batchIdx * numHeads + headIdx) * HEAD_DIM], HEAD_DIM);
        qL1.EnQue(qLoc);
        LocalTensor<fp8_t> q = qL1.DeQue<fp8_t>();

        LocalTensor<float> outAcc = ubOut.Get<float>();
        SetValue(outAcc, HEAD_DIM_V, 0.f);                           // 初始化输出累加器
        float mPrev = -CUDART_INF_F, lPrev = 0.f;                    // Softmax 统计量

        // 分块遍历 TopK 个 KV
        for (int32_t tile = 0; tile < nTile; ++tile) {
            ComputeTile(q, tile, outAcc, mPrev, lPrev);
        }
        // 归一化并写回
        Div(outAcc, outAcc, lPrev, HEAD_DIM_V);                      // Vector: out /= lPrev
        DataCopy(outGm[(batchIdx * numHeads + headIdx) * HEAD_DIM_V], outAcc, HEAD_DIM_V);
        float lseVal = logf(lPrev) + mPrev;
        DataCopy(lseGm[batchIdx * numHeads + headIdx], &lseVal, 1); // 写 LSE
        qL1.FreeTensor(q);
    }
private:
    __aicore__ inline void ComputeTile(LocalTensor<fp8_t>& q, int32_t tile,
                                       LocalTensor<float>& outAcc, float& mPrev, float& lPrev) {
        int32_t tileStart = tile * BLOCK_N;
        int32_t tileSize  = min(BLOCK_N, topk - tileStart);

        // 加载 K
        LocalTensor<fp8_t> kLoc = kL1.AllocTensor<fp8_t>();
        for (int32_t k = 0; k < tileSize; ++k) {
            int32_t kvIdx = indicesGm[batchIdx * topk + tileStart + k];
            DataCopy(kLoc[k * HEAD_DIM], kvCacheGm[kvIdx * 656], HEAD_DIM);  // 简化: 实际需解析 page
        }
        kL1.EnQue(kLoc);
        LocalTensor<fp8_t> k = kL1.DeQue<fp8_t>();

        // Cube: QK^T
        LocalTensor<float> logits = cO.AllocTensor<float>();
        Mmad(logits, q, k, {1, tileSize, HEAD_DIM});                // [1, BLOCK_N, HEAD_DIM] → [1, BLOCK_N]
        Muls(logits, logits, softmaxScale, tileSize);               // logits *= scale
        cO.EnQue(logits);
        LocalTensor<float> lg = cO.DeQue<float>();

        // Softmax: 更新 max 与 sum (Vector 规约)
        LocalTensor<float> qkScores = ubQK.Get<float>();
        DataCopy(qkScores, lg, tileSize);
        float mCurr = ReduceMax(qkScores, tileSize);                // Vector: max
        float mNew  = fmaxf(mPrev, mCurr);
        float alpha = expf(mPrev - mNew);
        Muls(outAcc, outAcc, alpha, HEAD_DIM_V);                    // outAcc *= alpha

        Subs(qkScores, qkScores, mNew, tileSize);                   // qk -= mNew
        Exp(qkScores, qkScores, tileSize);                          // qk = exp(qk)
        float localSum = ReduceSum(qkScores, tileSize);             // Vector: sum
        float lNew = lPrev * alpha + localSum;

        // 加载 V 并累加
        LocalTensor<fp8_t> vLoc = vL1.AllocTensor<fp8_t>();
        for (int32_t k = 0; k < tileSize; ++k) {
            int32_t kvIdx = indicesGm[batchIdx * topk + tileStart + k];
            DataCopy(vLoc[k * HEAD_DIM_V], kvCacheGm[kvIdx * 656 + 512 + 16], HEAD_DIM_V);
        }
        vL1.EnQue(vLoc);
        LocalTensor<fp8_t> v = vL1.DeQue<fp8_t>();

        for (int32_t k = 0; k < tileSize; ++k) {
            float weight = qkScores[k];
            Axpy(outAcc, v[k * HEAD_DIM_V], weight, HEAD_DIM_V);    // outAcc += weight * v[k]
        }

        mPrev = mNew; lPrev = lNew;
        kL1.FreeTensor(k); cO.FreeTensor(lg); vL1.FreeTensor(v);
    }

    TPipe pipe;
    TQue<TPosition::A1, 1> qL1;
    TQue<TPosition::B1, 1> kL1;
    TQue<TPosition::VECIN,1> vL1;
    TQue<TPosition::CO1,1> cO;
    TBuf<TPosition::VECCALC> ubQK, ubOut;
    GlobalTensor<fp8_t>   qGm, kvCacheGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<float>   outGm, lseGm;
    int32_t batchIdx, headIdx, B, numHeads, topk, nTile;
    float softmaxScale;
};
`;

const S6 = String.raw`// flash_mla_decode.cpp · AscendC 核  (AscendPort · S6 双缓冲流水已编排)
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HEAD_DIM   = 576;
constexpr int32_t HEAD_DIM_V = 512;
constexpr int32_t BLOCK_N    = 128;
constexpr int32_t DEPTH      = 2;              // ← 双缓冲深度(Ping-Pong)

class FlashMLADecode {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kvCache, GM_ADDR indices,
                                GM_ADDR attnSink, GM_ADDR out, GM_ADDR lse,
                                int32_t B, int32_t numHeads, int32_t topk,
                                int32_t pageSize, float softmaxScale, int32_t nTile) {
        this->batchIdx = GetBlockIdx() / numHeads;
        this->headIdx  = GetBlockIdx() % numHeads;
        this->B = B; this->numHeads = numHeads; this->topk = topk;
        this->softmaxScale = softmaxScale; this->nTile = nTile;
        qGm.SetGlobalBuffer((__gm__ fp8_t*)q);
        kvCacheGm.SetGlobalBuffer((__gm__ fp8_t*)kvCache);
        indicesGm.SetGlobalBuffer((__gm__ int32_t*)indices);
        outGm.SetGlobalBuffer((__gm__ float*)out);
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        pipe.InitBuffer(qL1,  1,     HEAD_DIM * sizeof(fp8_t));
        pipe.InitBuffer(kL1,  DEPTH, BLOCK_N * HEAD_DIM * sizeof(fp8_t));    // 深度=2 双缓冲
        pipe.InitBuffer(vL1,  DEPTH, BLOCK_N * HEAD_DIM_V * sizeof(fp8_t));  // 深度=2
        pipe.InitBuffer(cO,   DEPTH, BLOCK_N * sizeof(float));
        pipe.InitBuffer(ubQK, DEPTH, BLOCK_N * sizeof(float));
        pipe.InitBuffer(ubOut,1,     HEAD_DIM_V * sizeof(float));
    }
    __aicore__ inline void Process() {
        if (batchIdx >= B || headIdx >= numHeads) return;
        LocalTensor<fp8_t> qLoc = qL1.AllocTensor<fp8_t>();
        DataCopy(qLoc, qGm[(batchIdx * numHeads + headIdx) * HEAD_DIM], HEAD_DIM);
        qL1.EnQue(qLoc);
        LocalTensor<fp8_t> q = qL1.DeQue<fp8_t>();

        LocalTensor<float> outAcc = ubOut.Get<float>();
        SetValue(outAcc, HEAD_DIM_V, 0.f);
        float mPrev = -CUDART_INF_F, lPrev = 0.f;

        // ---- 软件流水:预取 n+1  ∥  Cube/Vector 计算 n  ∥  V 累加 ----
        CopyInKV(0);                                        // 预热:载入第 0 块
        for (int32_t tile = 0; tile < nTile; ++tile) {
            if (tile + 1 < nTile) CopyInKV(tile + 1);       // 预取下一块(与计算重叠)
            ComputeTile(q, tile, outAcc, mPrev, lPrev);     // Cube QK^T → Vector Softmax
        }
        // 归一化并写回
        Div(outAcc, outAcc, lPrev, HEAD_DIM_V);
        DataCopy(outGm[(batchIdx * numHeads + headIdx) * HEAD_DIM_V], outAcc, HEAD_DIM_V);
        float lseVal = logf(lPrev) + mPrev;
        DataCopy(lseGm[batchIdx * numHeads + headIdx], &lseVal, 1);
        qL1.FreeTensor(q);
    }
private:
    __aicore__ inline void CopyInKV(int32_t tile) {
        int32_t tileStart = tile * BLOCK_N;
        int32_t tileSize  = min(BLOCK_N, topk - tileStart);
        // K
        LocalTensor<fp8_t> kLoc = kL1.AllocTensor<fp8_t>();
        for (int32_t k = 0; k < tileSize; ++k) {
            int32_t kvIdx = indicesGm[batchIdx * topk + tileStart + k];
            DataCopy(kLoc[k * HEAD_DIM], kvCacheGm[kvIdx * 656], HEAD_DIM);
        }
        kL1.EnQue(kLoc);                                    // 入队 → 与 Compute 并行
        // V
        LocalTensor<fp8_t> vLoc = vL1.AllocTensor<fp8_t>();
        for (int32_t k = 0; k < tileSize; ++k) {
            int32_t kvIdx = indicesGm[batchIdx * topk + tileStart + k];
            DataCopy(vLoc[k * HEAD_DIM_V], kvCacheGm[kvIdx * 656 + 512 + 16], HEAD_DIM_V);
        }
        vL1.EnQue(vLoc);
    }
    __aicore__ inline void ComputeTile(LocalTensor<fp8_t>& q, int32_t tile,
                                       LocalTensor<float>& outAcc, float& mPrev, float& lPrev) {
        int32_t tileStart = tile * BLOCK_N;
        int32_t tileSize  = min(BLOCK_N, topk - tileStart);

        LocalTensor<fp8_t> k = kL1.DeQue<fp8_t>();          // 取上一轮预取的块
        LocalTensor<float> logits = cO.AllocTensor<float>();
        Mmad(logits, q, k, {1, tileSize, HEAD_DIM});        // Cube
        Muls(logits, logits, softmaxScale, tileSize);
        cO.EnQue(logits);
        LocalTensor<float> lg = cO.DeQue<float>();

        LocalTensor<float> qkScores = ubQK.AllocTensor<float>();
        DataCopy(qkScores, lg, tileSize);
        // block_reduce_max/sum 在昇腾无对应物 → 改写为 Vector 片上归约
        float mCurr = ReduceMax(qkScores, tileSize);        // Vector 规约
        float mNew  = fmaxf(mPrev, mCurr);
        float alpha = expf(mPrev - mNew);
        Muls(outAcc, outAcc, alpha, HEAD_DIM_V);

        Subs(qkScores, qkScores, mNew, tileSize);
        Exp(qkScores, qkScores, tileSize);
        float localSum = ReduceSum(qkScores, tileSize);     // Vector 规约
        float lNew = lPrev * alpha + localSum;

        LocalTensor<fp8_t> v = vL1.DeQue<fp8_t>();
        for (int32_t k = 0; k < tileSize; ++k) {
            float weight = qkScores[k];
            Axpy(outAcc, v[k * HEAD_DIM_V], weight, HEAD_DIM_V);
        }
        ubQK.EnQue(qkScores);

        mPrev = mNew; lPrev = lNew;
        kL1.FreeTensor(k); cO.FreeTensor(lg); vL1.FreeTensor(v);
    }

    TPipe pipe;
    TQue<TPosition::A1, 1>        qL1;
    TQue<TPosition::B1, DEPTH>    kL1;      // ← Ping-Pong
    TQue<TPosition::VECIN, DEPTH> vL1;      // ← Ping-Pong
    TQue<TPosition::CO1, DEPTH>   cO;
    TQue<TPosition::VECOUT,DEPTH> ubQK;
    TBuf<TPosition::VECCALC>      ubOut;
    GlobalTensor<fp8_t>   qGm, kvCacheGm;
    GlobalTensor<int32_t> indicesGm;
    GlobalTensor<float>   outGm, lseGm;
    int32_t batchIdx, headIdx, B, numHeads, topk, nTile;
    float softmaxScale;
};
`;

/* view: {file, lang, text, hl(lineText,idx)->class} */
function riskHL(t){return /__shfl_xor_sync|block_reduce|cg::|__shared__|thread_block|__syncthreads/.test(t)?'hl-risk':''}
function todoHL(t){return /TODO\(/.test(t)?'hl-add':''}
function copyHL(t){return /DataCopy|InitBuffer|Mmad|Exp\(|ReduceMax|ReduceSum|AllocTensor|EnQue|DeQue/.test(t)?'hl-add':''}
// S4：内存层次注入的关键行更醒目
function s4HL(t){
  if(/InitBuffer|DataCopy|Mmad\(|Exp\(|ReduceMax|ReduceSum|Div\(/.test(t)) return 'hl-new';
  if(/AllocTensor|EnQue|DeQue|FreeTensor|\.Get<|SetValue/.test(t)) return 'hl-add';
  return '';
}
function bufHL(t){return /DEPTH|CopyInKV|预取|Ping-Pong|双缓冲|流水|EnQue|DeQue/.test(t)?'hl-buf':''}
// S6：双缓冲流水新增代码更醒目
function s6HL(t){
  if(/DEPTH|软件流水|预取|CopyInKV\(|Ping-Pong|ReduceMax|ReduceSum|nTile/.test(t)) return 'hl-new';
  if(/EnQue|DeQue|AllocTensor|FreeTensor|ComputeTile\(/.test(t)) return 'hl-buf';
  return '';
}
// tiling.h 高亮:nTile / TilingData 关键行醒目
function tilingHL(t){
  if(/nTile|TILING_KEY|TilingData|SetBlockDim|GET_TILING_DATA/.test(t)) return 'hl-new';
  if(/BEGIN_TILING|REGISTER_TILING|FIELD/.test(t)) return 'hl-add';
  return '';
}
// tiling.h 内容随所选分块方案变化
function tilingSrc(){
  const c=state.choices['S5']||'B';
  const nTile=(c==='A')?16:(c==='B')?8:4;
  const ubUtil=(c==='A')?58:(c==='B')?85:102;
  const cyc=(c==='A')?'1.00':(c==='B')?'0.68':'0.92';
  const note=(c==='C')?'// ⚠ nTile=4 超 L0C 容量,将触发回退搬运':'// ✓ 片上驻留最大化,回 GM 次数最小';
  return `// tiling.h · AscendC Tiling 结构  (AscendPort · S5 自动生成)
// 沿 TopK 维分块:每核每次处理 BLOCK_N 个 KV,贴合 L1/L0C/UB 容量
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {

BEGIN_TILING_DATA_DEF(FlashMLATiling)
  TILING_DATA_FIELD_DEF(int32_t, B);          // batch size
  TILING_DATA_FIELD_DEF(int32_t, numHeads);   // query heads
  TILING_DATA_FIELD_DEF(int32_t, topk);       // 稀疏 TopK 长度
  TILING_DATA_FIELD_DEF(int32_t, nTile);      // ← 分块数 = ceil(topk / BLOCK_N)
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(flash_mla_sparse_decode, FlashMLATiling)

// ---- 自动 Tiling:在 L0C / UB 容量约束下选定 BLOCK_N ----
constexpr int32_t BLOCK_N = ${(c==='A')?128:(c==='B')?256:512};  // UB 利用率 ${ubUtil}% · cycles ${cyc}×
${note}
static ge::graphStatus TilingFunc(gert::TilingContext* ctx) {
    FlashMLATiling t;
    int32_t B = ctx->GetInputShape(0)->GetStorageShape().GetDim(0);
    int32_t numHeads = ctx->GetInputShape(0)->GetStorageShape().GetDim(1);
    int32_t topk = ctx->GetInputShape(2)->GetStorageShape().GetDim(1);
    t.set_B(B);  t.set_numHeads(numHeads);  t.set_topk(topk);
    t.set_nTile((topk + BLOCK_N - 1) / BLOCK_N);   // 向上取整分块数
    ctx->SetBlockDim(B * numHeads);                 // 每个 (batch, head) 对一个核
    ctx->SetTilingKey(1);
    t.SaveToBuffer(ctx->GetRawTilingData()->GetData(),
                   ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(t.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
`;
}

const VIEWS = {
  cuda:{file:'flash_mla_decode.cu', lang:'cpp', text:CUDA, hl:riskHL},
  s3:{file:'flash_mla_decode.cpp', lang:'cpp', text:S3, hl:todoHL},
  s4:{file:'flash_mla_decode.cpp', lang:'cpp', text:S4, hl:s4HL},
  s6:{file:'flash_mla_decode.cpp', lang:'cpp', text:S6, hl:s6HL},
  get tiling(){ return {file:'tiling.h', lang:'cpp', text:tilingSrc(), hl:tilingHL}; },
};

/* ============================ 语法高亮 ============================ */
const KW = new Set(('for while if else return const void int float bool char class public private struct namespace using constexpr inline extern template this reinterpret_cast static true false __global__ __device__ __aicore__ __forceinline__ __restrict__ __shared__ __nv_fp8_e4m3 __nv_fp8x4_e4m3').split(' '));
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function highlight(line){
  let s = esc(line);
  const RE=/(\/\/[^\n]*)|("(?:[^"\\]|\\.)*")|(#[a-zA-Z]+)|\b(0x[0-9a-fA-F]+|\d+\.?\d*f?)\b|\b([A-Za-z_][A-Za-z0-9_]*)\b(\s*\()?/g;
  return s.replace(RE,(m,com,str,pp,num,word,paren)=>{
    if(com) return '<span class="c-com">'+com+'</span>';
    if(str) return '<span class="c-str">'+str+'</span>';
    if(pp)  return '<span class="c-pp">'+pp+'</span>';
    if(num!==undefined) return '<span class="c-num">'+num+'</span>';
    if(word){
      if(KW.has(word)) return '<span class="c-k">'+word+'</span>'+(paren||'');
      if(paren) return '<span class="c-fn">'+word+'</span>'+paren;
      if(/^[A-Z_]/.test(word)&&/[a-z]/.test(word)===false&&word.length>1) return '<span class="c-num">'+word+'</span>';
      if(/^[A-Z]/.test(word)) return '<span class="c-ty">'+word+'</span>';
      return word;
    }
    return m;
  });
}
function renderCode(key){
  const v=VIEWS[key]; const lines=v.text.replace(/\n$/,'').split('\n');
  const g=document.getElementById('gutter'), c=document.getElementById('codelines');
  g.textContent=lines.map((_,i)=>i+1).join('\n');
  c.innerHTML=lines.map(l=>{const cls=v.hl?v.hl(l):'';return '<span class="ln '+cls+'">'+(highlight(l)||' ')+'</span>'}).join('');
  document.getElementById('codewrap').scrollTop=0;
}
// 右侧对比面板：渲染生成的 AscendC 源码
function renderDiff(key){
  const v=VIEWS[key]; if(!v) return;
  const lines=v.text.replace(/\n$/,'').split('\n');
  const g=document.getElementById('diffGutter'), c=document.getElementById('diffLines');
  g.textContent=lines.map((_,i)=>i+1).join('\n');
  c.innerHTML=lines.map(l=>{const cls=v.hl?v.hl(l):'';return '<span class="ln '+cls+'">'+(highlight(l)||' ')+'</span>'}).join('');
  const f=document.getElementById('diffFile'); if(f) f.textContent=v.file;
  document.getElementById('diffwrap').scrollTop=0;
  // S4：新注入的内存层次行做一次入场闪烁
  if(key==='s4'){
    const news=c.querySelectorAll('.ln.hl-new');
    news.forEach(el=>el.classList.add('flash'));
    setTimeout(()=>news.forEach(el=>el.classList.remove('flash')),1100);
  }
}
// 开启 CUDA ↔ AscendC 同屏对比：左 CUDA、右生成的 cpp
function openCompare(diffKey){
  closeGraph(); closeTiling(); closePipe();        // 关闭计算图 / tiling / 流水对比
  activeTab='cuda';
  renderCode('cuda');                             // 左侧固定为 CUDA
  document.getElementById('leftPaneH').style.display='flex';
  renderDiff(diffKey);                            // 右侧为生成的 AscendC
  document.getElementById('split').classList.add('compare-open');
  renderTabs(); renderTree();
  const f=document.getElementById('etbFile'); if(f) f.textContent='lightning_indexer.cu ↔ .cpp';
  tagLinkGroups(diffKey);                          // 建立相同计算过程的联动呼应
}
function closeCompare(){
  const sp=document.getElementById('split');
  sp.classList.remove('compare-open'); sp.classList.remove('link-active');
  clearLinkHot();
  document.getElementById('leftPaneH').style.display='none';
}

/* ---------- S3 对比联动：相同计算过程的代码片段互相呼应 ---------- */
// 每组：cuda[起,止] ↔ asc[起,止]（1-based，含端点），label 为该计算过程。
let linkGroups=[]; // 当前对比视图的分组
const LINKMAP={
  s3:[
    {label:'内核入口 / 参数', cuda:[35,41], asc:[13,23]},
    {label:'grid → 分核 (blockIdx.x=t)', cuda:[44,44], asc:[16,18]},
    {label:'外层 query 循环', cuda:[57,57], asc:[27,32]},
    {label:'QKᵀ 点积 + ReLU·w 归约', cuda:[60,72], asc:[34,35]},
    {label:'warp 双调排序 → Vector Top-K', cuda:[76,83], asc:[36,37]},
    {label:'warp_bitonic_sort (SIMT 专属)', cuda:[17,32], asc:[36,37]},
    {label:'Top-K 输出', cuda:[87,94], asc:[22,23]},
  ],
  s4:[
    {label:'grid → 分核', cuda:[44,44], asc:[14,14]},
    {label:'头权重 __shared__ → UB', cuda:[48,51], asc:[30,30]},
    {label:'causal 分块循环 (s<=t)', cuda:[57,57], asc:[31,32]},
    {label:'kI / qI 载入 GM→L1', cuda:[62,63], asc:[38,45]},
    {label:'QKᵀ 点积 → Cube (Mmad)', cuda:[65,70], asc:[47,50]},
    {label:'ReLU → Vector', cuda:[71,71], asc:[53,53]},
    {label:'跨头加权归约 → Vector', cuda:[60,72], asc:[54,54]},
  ],
  s6:[
    {label:'causal 分块 + 软件流水', cuda:[57,57], asc:[29,40]},
    {label:'预取下一块 (双缓冲)', cuda:[57,57], asc:[33,37]},
    {label:'kI 载入 (CopyIn)', cuda:[62,63], asc:[42,46]},
    {label:'QKᵀ 点积 → Cube (Mmad)', cuda:[65,70], asc:[47,51]},
    {label:'ReLU → Vector', cuda:[71,71], asc:[54,54]},
    {label:'跨头加权归约 → Vector', cuda:[60,72], asc:[55,55]},
    {label:'warp 双调排序 → Vector TopK', cuda:[76,83], asc:[60,66]},
    {label:'warp_bitonic_sort (SIMT 专属)', cuda:[17,32], asc:[60,66]},
    {label:'Top-K 输出 UB→GM', cuda:[87,94], asc:[64,65]},
  ],
};
// 给两侧代码行打上分组标记（一行可属于多个分组）
function tagLinkGroups(diffKey){
  linkGroups = LINKMAP[diffKey] || [];
  const leftLns = document.querySelectorAll('#codelines .ln');
  const rightLns = document.querySelectorAll('#diffLines .ln');
  const reset=el=>{el.classList.remove('link-grp');el.removeAttribute('data-grp');};
  leftLns.forEach(reset); rightLns.forEach(reset);
  const add=(el,gi)=>{ if(!el) return; el.classList.add('link-grp');
    const cur=el.dataset.grp?el.dataset.grp.split(','):[]; if(!cur.includes(''+gi)){cur.push(''+gi);el.dataset.grp=cur.join(',');} };
  linkGroups.forEach((g,gi)=>{
    for(let i=g.cuda[0]-1;i<=g.cuda[1]-1;i++) add(leftLns[i],gi);
    for(let i=g.asc[0]-1;i<=g.asc[1]-1;i++) add(rightLns[i],gi);
  });
  bindLinkHover(leftLns,'left'); bindLinkHover(rightLns,'right');
}
function clearLinkHot(){
  document.querySelectorAll('.ln.link-hot').forEach(el=>el.classList.remove('link-hot'));
  document.getElementById('split').classList.remove('link-active');
}
function highlightGroup(grpAttr, originSide){
  clearLinkHot();
  if(grpAttr==null||grpAttr==='') return;
  const gis=(''+grpAttr).split(',').map(Number).filter(x=>!isNaN(x));
  if(!gis.length) return;
  const leftLns=document.querySelectorAll('#codelines .ln');
  const rightLns=document.querySelectorAll('#diffLines .ln');
  let firstOppLine=null;
  gis.forEach(gi=>{
    const g=linkGroups[gi]; if(!g) return;
    for(let i=g.cuda[0]-1;i<=g.cuda[1]-1;i++){ if(leftLns[i]) leftLns[i].classList.add('link-hot'); }
    for(let i=g.asc[0]-1;i<=g.asc[1]-1;i++){ if(rightLns[i]) rightLns[i].classList.add('link-hot'); }
    // 记录对侧首行用于滚动
    if(firstOppLine===null){
      firstOppLine = originSide==='left' ? rightLns[g.asc[0]-1] : leftLns[g.cuda[0]-1];
    }
  });
  const wrapId = originSide==='left' ? 'diffwrap' : 'codewrap';
  scrollLineIntoView(wrapId, firstOppLine);
  document.getElementById('split').classList.add('link-active');
}
function scrollLineIntoView(wrapId, lineEl){
  if(!lineEl) return;
  const wrap=document.getElementById(wrapId);
  const wr=wrap.getBoundingClientRect(), lr=lineEl.getBoundingClientRect();
  const offset=(lr.top-wr.top)+wrap.scrollTop;
  wrap.scrollTo({top:Math.max(0,offset - wrap.clientHeight/3), behavior:'smooth'});
}
function bindLinkHover(lns, side){
  lns.forEach(el=>{
    if(!el.classList.contains('link-grp')) return;
    if(el.__linkSide===side) return; el.__linkSide=side;
    el.onmouseenter=()=>highlightGroup(el.dataset.grp, side);
    el.onclick=()=>highlightGroup(el.dataset.grp, side); // 点击滚动到对侧
  });
  // 离开代码区清除高亮
  const wrap = side==='left'?document.getElementById('codewrap'):document.getElementById('diffwrap');
  wrap.onmouseleave=()=>clearLinkHot();
}

/* ============================ S4 硬件数据流动画 ============================ */
// 达芬奇内存层次 + 执行单元。坐标基于 viewBox 780×188。
const FUNITS={
  gm:  {x:14,  y:70, w:78, h:48, c:'--mem',    t:'Global Mem', s:'GM · HBM'},
  l1:  {x:150, y:70, w:74, h:48, c:'--mem',    t:'L1 Buffer',  s:'片上缓存'},
  l0a: {x:280, y:14, w:74, h:40, c:'--cube',   t:'L0A',        s:'Cube 输入·q'},
  l0b: {x:280, y:134,w:74, h:40, c:'--cube',   t:'L0B',        s:'Cube 输入·k'},
  cube:{x:410, y:60, w:86, h:66, c:'--cube',   t:'Cube',       s:'Mmad · QKᵀ'},
  l0c: {x:540, y:60, w:74, h:48, c:'--cube',   t:'L0C',        s:'矩阵输出·logits'},
  ub:  {x:664, y:14, w:102,h:48, c:'--vec',    t:'Unified Buffer', s:'UB · 头权重/打分'},
  vec: {x:664, y:118,w:102,h:52, c:'--vec',    t:'Vector',     s:'ReLU · Σw·(·)'},
};
const FEDGES={
  gm_l1:  ['gm','l1'], l1_l0a:['l1','l0a'], l1_l0b:['l1','l0b'],
  l0a_cube:['l0a','cube'], l0b_cube:['l0b','cube'], cube_l0c:['cube','l0c'],
  l0c_vec:['l0c','vec'], gm_ub:['gm','ub'], ub_vec:['ub','vec'],
};
// 每一步：亮起的单元、走的边、说明、颜色、对应 S4 代码行
const FLOW_STEPS=[
  {t:'DataCopy 头权重 GM→UB', units:['gm','ub'], edges:['gm_ub'], code:[30,30], col:'--mem',
   note:'w[t,·] 头权重从 Global Memory 搬入 Unified Buffer,供后续加权归约使用。'},
  {t:'DataCopy kI GM→L1→L0B', units:['gm','l1','l0b'], edges:['gm_l1','l1_l0b'], code:[38,40], col:'--mem',
   note:'key 分块 kI[s0:] 逐级搬运:GM → L1 → L0B,进入 Cube 的 B 矩阵入口。'},
  {t:'DataCopy qI GM→L1→L0A', units:['gm','l1','l0a'], edges:['gm_l1','l1_l0a'], code:[42,45], col:'--mem',
   note:'query qI[t] 同样 GM → L1 → L0A,进入 Cube 的 A 矩阵入口。'},
  {t:'Mmad 矩阵乘 → L0C', units:['l0a','l0b','cube','l0c'], edges:['l0a_cube','l0b_cube','cube_l0c'], code:[47,50], col:'--cube',
   note:'Cube 单元执行 QKᵀ = q·kᵀ(FP8),结果 logits 落入 L0C。这是算力主体。'},
  {t:'Relu 激活 · L0C→UB→Vector', units:['l0c','vec','ub'], edges:['l0c_vec'], code:[52,53], col:'--vec',
   note:'Vector 单元对 logits 做 ReLU(fmaxf(·,0)),逐元素激活,写入 UB。'},
  {t:'WeightedHeadReduce 加权归约', units:['ub','vec'], edges:['ub_vec'], code:[54,54], col:'--vec',
   note:'Vector 单元读 UB 中的头权重 w,做 Σ_j w[j]·ReLU(·) 跨头加权求和,得到每个 key 的分数。'},
];
let flowIdx=0, flowTimer=null, flowPlaying=false;

function unitCol(k){return getComputedStyle(document.documentElement).getPropertyValue(FUNITS[k].c).trim();}
function edgePath(ek){
  const [a,b]=FEDGES[ek]; const na=FUNITS[a], nb=FUNITS[b];
  // 竖直相邻(同列上下,如 UB↕Vector):走垂直连线
  const sameCol = Math.abs((na.x+na.w/2)-(nb.x+nb.w/2)) < 30;
  if(sameCol){
    const x=na.x+na.w/2;
    const y1=(na.y<nb.y)?na.y+na.h:na.y;
    const y2=(na.y<nb.y)?nb.y:nb.y+nb.h;
    return {d:`M${x},${y1} L${x},${y2}`, x1:x,y1,x2:x,y2};
  }
  // 头权重 GM→UB:从 GM 顶部绕行到 UB 左侧,避免横穿画布
  if(ek==='gm_ub'){
    const x1=na.x+na.w/2, y1=na.y, x2=nb.x, y2=nb.y+nb.h/2;
    return {d:`M${x1},${y1} C${x1},2 ${x2-40},2 ${x2},${y2}`, x1,y1,x2,y2};
  }
  const x1=na.x+na.w, y1=na.y+na.h/2, x2=nb.x, y2=nb.y+nb.h/2;
  const mx=(x1+x2)/2;
  return {d:`M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, x1,y1,x2,y2};
}
function buildFlowSVG(){
  let edges='', nodes='';
  Object.keys(FEDGES).forEach(ek=>{ const p=edgePath(ek);
    edges+=`<path class="fedge" id="fe_${ek}" d="${p.d}"/>`; });
  Object.keys(FUNITS).forEach(k=>{ const u=FUNITS[k]; const col=unitCol(k);
    nodes+=`<g class="fu-box" id="fu_${k}">
      <rect x="${u.x}" y="${u.y}" width="${u.w}" height="${u.h}" rx="9" fill="${col}22" stroke="${col}" stroke-width="1.5"/>
      <text class="fu-lbl" x="${u.x+u.w/2}" y="${u.y+u.h/2-2}" text-anchor="middle">${u.t}</text>
      <text class="fu-sub" x="${u.x+u.w/2}" y="${u.y+u.h/2+11}" text-anchor="middle">${u.s}</text>
    </g>`; });
  return `<svg viewBox="0 0 780 188" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    ${edges}
    <text class="fcap" x="53" y="134" text-anchor="middle">DDR / HBM</text>
    <text class="fcap" x="453" y="140" text-anchor="middle">矩阵单元</text>
    <text class="fcap" x="715" y="180" text-anchor="middle">向量单元</text>
    ${nodes}
    <g id="fpkts"></g>
  </svg>`;
}
function renderFlow(){
  const pane=document.getElementById('flowpane');
  pane.innerHTML=`
    <div class="flow-bar">
      <button class="fb-btn" id="flowPlay">▶ 播放</button>
      <button class="fb-btn" id="flowStep">⏭ 单步</button>
      <span class="fb-step">步骤 <b id="flowNo">1</b>/${FLOW_STEPS.length} · <span id="flowTitle">${FLOW_STEPS[0].t}</span></span>
      <span class="fb-spacer"></span>
      <span class="fb-legend">
        <span><i style="background:var(--mem)"></i>片上搬运 (MTE)</span>
        <span><i style="background:var(--cube)"></i>Cube 矩阵</span>
        <span><i style="background:var(--vec)"></i>Vector 向量</span>
      </span>
    </div>
    <div class="flow-stage" id="flowStage"></div>
    <div style="padding:6px 12px;border-top:1px solid #ffffff0a;font-size:14px;color:var(--dim)"><span id="flowNote">${FLOW_STEPS[0].note}</span></div>`;
  document.getElementById('flowStage').innerHTML=buildFlowSVG();
  document.getElementById('flowPlay').onclick=toggleFlowPlay;
  document.getElementById('flowStep').onclick=()=>{ stopFlow(); flowIdx=(flowIdx+1)%FLOW_STEPS.length; showFlowStep(flowIdx); };
  flowIdx=0; showFlowStep(0);
}
function clearFlowHot(){
  Object.keys(FUNITS).forEach(k=>document.getElementById('fu_'+k)?.classList.remove('active'));
  Object.keys(FEDGES).forEach(ek=>document.getElementById('fe_'+ek)?.classList.remove('lit'));
}
function spawnPacket(ek, col){
  const stage=document.getElementById('flowStage'); if(!stage) return;
  const svg=stage.querySelector('svg'); const layer=svg.querySelector('#fpkts');
  const path=svg.querySelector('#fe_'+ek); if(!path) return;
  const len=path.getTotalLength();
  const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
  c.setAttribute('r','4.5'); c.setAttribute('fill',col); c.setAttribute('filter','url(#glow)');
  c.setAttribute('opacity','0.95'); layer.appendChild(c);
  const t0=performance.now(), dur=620;
  (function move(now){
    const p=Math.min(1,(now-t0)/dur); const pt=path.getPointAtLength(p*len);
    c.setAttribute('cx',pt.x); c.setAttribute('cy',pt.y);
    if(p<1) requestAnimationFrame(move); else c.remove();
  })(t0);
}
function showFlowStep(i){
  const s=FLOW_STEPS[i]; if(!s) return;
  clearFlowHot();
  const col=getComputedStyle(document.documentElement).getPropertyValue(s.col).trim();
  s.units.forEach(u=>document.getElementById('fu_'+u)?.classList.add('active'));
  s.edges.forEach(ek=>{ document.getElementById('fe_'+ek)?.classList.add('lit'); spawnPacket(ek,col); });
  const no=document.getElementById('flowNo'), ti=document.getElementById('flowTitle'), nt=document.getElementById('flowNote');
  if(no) no.textContent=i+1; if(ti) ti.textContent=s.t; if(nt) nt.textContent=s.note;
  // 与右侧 AscendC 代码联动：高亮对应注入行
  if(s.code && document.getElementById('split').classList.contains('compare-open')){
    highlightDiffLines(s.code[0], s.code[1]);
  }
}
// 高亮/滚动 AscendC(右)面板的指定行
function highlightDiffLines(a,b){
  const lns=document.querySelectorAll('#diffLines .ln');
  lns.forEach(el=>el.classList.remove('hl-node'));
  for(let i=a-1;i<b && i<lns.length;i++) lns[i]?.classList.add('hl-node');
  if(lns[a-1]) scrollLineIntoView('diffwrap', lns[a-1]);
}
function toggleFlowPlay(){ flowPlaying?stopFlow():startFlow(); }
function startFlow(){
  flowPlaying=true; const b=document.getElementById('flowPlay'); if(b){b.textContent='⏸ 暂停';b.classList.add('on');}
  showFlowStep(flowIdx);
  flowTimer=setInterval(()=>{ flowIdx=(flowIdx+1)%FLOW_STEPS.length; showFlowStep(flowIdx); }, 1500);
}
function stopFlow(){
  flowPlaying=false; if(flowTimer){clearInterval(flowTimer);flowTimer=null;}
  const b=document.getElementById('flowPlay'); if(b){b.textContent='▶ 播放';b.classList.remove('on');}
}
// 供面板 tab 调用
function openFlowPanel(autoplay){
  document.getElementById('flowTab').style.display='';
  activatePanelTab('flow');
  renderFlow();
  if(autoplay) startFlow();
}
function activatePanelTab(p){
  document.querySelectorAll('.ptab').forEach(x=>x.classList.toggle('on',x.dataset.p===p));
  const showTerm = (p==='term'||p==='term2');
  document.getElementById('term').style.display=showTerm?'block':'none';
  document.getElementById('probs').style.display=p==='probs'?'block':'none';
  document.getElementById('flowpane').style.display=p==='flow'?'flex':'none';
  document.getElementById('accpane').style.display=p==='acc'?'block':'none';
  document.getElementById('perfpane').style.display=p==='perf'?'block':'none';
}

/* ============================ S7 精度报告 ============================ */
// 逐算子对齐 CUDA 黄金基准。fixed 表示已应用修复后的复测结果。
let accFixed=false;
const ACC_OPS=[
  {op:'DataCopy (GM→L1/UB)', kind:'搬运', err:'0',      pass:true},
  {op:'Mmad · QKᵀ',          kind:'Cube', err:'2.4e-4', pass:true},
  {op:'Relu',                kind:'Vector',err:'0',     pass:true},
  {op:'WeightedHeadReduce',  kind:'Vector',err:'3.1e-2',pass:false,   // ← 异常算子
    fixedErr:'8.0e-4', anomaly:true},
  {op:'TopK · Top-K 规约',   kind:'Vector',err:'—',     pass:true, note:'命中率 100%(2048/2048)'},
];
function accStats(){
  const anomaly = ACC_OPS.find(o=>o.anomaly);
  const maxErr = accFixed ? '8.0e-4' : '3.1e-2';
  const cos    = accFixed ? '0.99987' : '0.9962';
  const passN  = accFixed ? ACC_OPS.length : ACC_OPS.filter(o=>o.pass).length;
  return {anomaly, maxErr, cos, passN, total:ACC_OPS.length};
}
function renderAccReport(){
  const st=accStats();
  const pane=document.getElementById('accpane');
  const rows=ACC_OPS.map(o=>{
    const ok = o.pass || accFixed;
    const err = (o.anomaly && accFixed) ? o.fixedErr : o.err;
    const stCls = (o.anomaly && accFixed) ? 'fixed' : (ok?'pass':'fail');
    const stTxt = (o.anomaly && accFixed) ? '已修复' : (ok?'通过':'异常');
    const errCol = (!ok)?'color:var(--risk)':((o.anomaly&&accFixed)?'color:var(--mem)':'color:var(--txt)');
    return `<tr class="${(!ok)?'bad':''}">
      <td class="acc-op">${o.op}</td>
      <td style="color:var(--dim)">${o.kind}</td>
      <td class="acc-err" style="${errCol}">${err}${o.note?`<div style="font-size:12px;color:var(--dim);font-family:var(--sans)">${o.note}</div>`:''}</td>
      <td><span class="acc-st ${stCls}">${stTxt}</span></td>
    </tr>`;
  }).join('');

  const a=st.anomaly;
  const anomalyBlock = accFixed ? `
    <div class="acc-card ok">
      <div class="ac-h">✓ 精度对齐通过 <span class="tag" style="background:#48d59722;color:var(--ok);border:1px solid #48d59755">FIXED</span></div>
      <div class="ac-row"><div class="ac-k">复测</div><div class="ac-v">max_abs_err <code>8.0e-4</code> · cos_sim <code>0.99987</code>,已达 rtol 1e-3 阈值。</div></div>
      <div class="ac-row"><div class="ac-k">Top-K</div><div class="ac-v">命中一致率 <code>100%</code> (2048/2048),并列分数顺序已对齐。</div></div>
    </div>` : `
    <div class="acc-card">
      <div class="ac-h">⚠ 检测到精度异常算子 <span class="tag risk">FAIL</span></div>
      <div class="ac-row"><div class="ac-k">算子</div><div class="ac-v"><code>${a.op}</code>(${a.kind})</div></div>
      <div class="ac-row"><div class="ac-k">现象</div><div class="ac-v">max_abs_err <code>${a.err}</code>,超出 rtol <code>1e-3</code> 阈值 ~30×。</div></div>
      <div class="ac-row"><div class="ac-k">根因</div><div class="ac-v"><b>FP8 累加顺序不一致</b>:CUDA 里各 head 的 <code>Σ w·ReLU</code> 在 FP32 寄存器串行累加;昇腾 Vector 归约按不同次序、且中间以 <b>FP8/FP16 累加</b>,舍入误差在跨 head 求和时被放大。</div></div>
      <div class="ac-fix">
        <div class="fh">🔧 修复方案 · 累加提升 FP32 + 对齐归约次序</div>
        <div class="acc-diff"><span class="ctx">    // WeightedHeadReduce(sc, w, sTile);</span><span class="del">-   ReduceSum&lt;fp16_t&gt;(sc, prod, sTile);          // FP16 累加,舍入放大</span><span class="add">+   ReduceSum&lt;float&gt;(sc, prod, sTile);            // 提升 FP32 累加</span><span class="add">+   SetReduceOrder(HEAD_ORDER_FIXED);              // 对齐 CUDA head 归约次序</span></div>
        <div class="acc-apply" id="accApply">▶ 应用修复并复测</div>
      </div>
    </div>`;

  pane.innerHTML=`
    <div class="acc-top">
      <div class="acc-kpi"><div class="kv" style="color:${accFixed?'var(--ok)':'var(--risk)'}">${st.maxErr}</div><div class="kk">max_abs_err</div><div class="kd" style="color:${accFixed?'var(--ok)':'var(--risk)'}">阈值 rtol 1e-3</div></div>
      <div class="acc-kpi"><div class="kv" style="color:${accFixed?'var(--ok)':'var(--warn)'}">${st.cos}</div><div class="kk">cos_sim</div><div class="kd" style="color:var(--dim)">越接近 1 越好</div></div>
      <div class="acc-kpi"><div class="kv">${st.passN}/${st.total}</div><div class="kk">算子通过</div><div class="kd" style="color:${accFixed?'var(--ok)':'var(--risk)'}">${accFixed?'全部通过':'1 个异常'}</div></div>
    </div>
    <div class="acc-sec-h">逐算子精度对齐 · golden = CUDA</div>
    <table class="acc-table">
      <thead><tr><th>算子</th><th>单元</th><th>max_abs_err</th><th>状态</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${anomalyBlock}`;

  const ap=document.getElementById('accApply');
  if(ap) ap.onclick=()=>{
    accFixed=true; setProblems(0);
    document.getElementById('accCnt').textContent='✓';
    renderAccReport();
    notify('✓ 精度修复已应用','累加提升 FP32 · cos_sim 0.99987 · 问题清零','ok');
  };
}
function openAccPanel(){
  document.getElementById('accTab').style.display='';
  document.getElementById('accCnt').textContent = accFixed?'✓':'!';
  document.getElementById('accCnt').style.background = accFixed?'#48d59722':'#ff547033';
  document.getElementById('accCnt').style.color = accFixed?'var(--ok)':'#ff8ba0';
  activatePanelTab('acc');
  renderAccReport();
}

/* ============================ S8 性能报告 ============================ */
// 泳道图:每条泳道一个硬件单元,cell 为 {s起, w宽, cls, l标签}。时间以格为单位。
// 直译版(before):串行,单元间大量 idle 空转。
function perfSwimBefore(){
  const rows={mte:[],cube:[],vec:[]}; let t=0;
  for(let n=0;n<3;n++){
    rows.mte.push({s:t,w:3,cls:'mte',l:`搬${n}`});
    rows.cube.push({s:t,w:3,cls:'idle',l:''});          // Cube 空等搬运
    rows.cube.push({s:t+3,w:2,cls:'cube',l:`Cube${n}`});
    rows.vec.push({s:t,w:5,cls:'idle',l:''});           // Vector 长时间空等
    rows.vec.push({s:t+5,w:1,cls:'vec',l:`V${n}`});
    t+=6;
  }
  return {rows,total:t};
}
// 优化版(after):双缓冲重叠,MTE 隐藏在计算下,单元密排。
function perfSwimAfter(){
  const rows={mte:[],cube:[],vec:[]};
  for(let n=0;n<3;n++) rows.mte.push({s:n*2,w:2,cls:'mte',l:`搬${n}`});
  for(let n=0;n<3;n++) rows.cube.push({s:2+n*2,w:2,cls:'cube',l:`Cube${n}`});
  for(let n=0;n<3;n++) rows.vec.push({s:4+n*2,w:1,cls:'vec',l:`V${n}`});
  return {rows,total:4+3*2};
}
function swimRow(label, cells, total, play){
  const pct=x=>(x/total*100);
  let inner='';
  cells.forEach((c,i)=>{ inner+=`<div class="swim-cell ${c.cls} ${play?'play':''}" style="left:${pct(c.s)}%;width:${pct(c.w)}%;${play?`animation-delay:${i*70}ms`:''}">${c.l}</div>`; });
  return `<div class="swim-row"><div class="swim-lbl">${label}</div><div class="swim-track">${inner}</div></div>`;
}
function swimHTML(model, play){
  const {rows,total}=model;
  return `<div class="swim">
    ${swimRow('MTE 搬运', rows.mte, total, play)}
    ${swimRow('Cube', rows.cube, total, play)}
    ${swimRow('Vector', rows.vec, total, play)}
    <div class="swim-axis"><span>t=0</span><span>时间(cycle)→</span><span>t=${total}</span></div>
    <div class="swim-legend"><span><i style="background:var(--mem)"></i>MTE 搬运</span><span><i style="background:var(--cube)"></i>Cube</span><span><i style="background:var(--vec)"></i>Vector</span><span><i style="background:repeating-linear-gradient(45deg,#ffffff30,#ffffff30 3px,transparent 3px,transparent 6px)"></i>空转 idle</span></div>
  </div>`;
}
// 利用率对比条
function cmpBar(label, before, after){
  return `<div class="cmp"><div class="cl"><span>${label}</span><b><span style="color:var(--risk)">${before}%</span> → <span style="color:var(--ok)">${after}%</span></b></div>
    <div class="bars">
      <div class="barrow"><span class="brl">直译</span><div class="brt"><div class="brf" style="width:${before}%;background:var(--risk)">${before}%</div></div></div>
      <div class="barrow"><span class="brl">优化</span><div class="brt"><div class="brf" style="width:${after}%;background:var(--ok)">${after}%</div></div></div>
    </div></div>`;
}
function renderPerfReport(play){
  const before=perfSwimBefore(), after=perfSwimAfter();
  const speedup=(before.total/after.total).toFixed(1);
  const pane=document.getElementById('perfpane');
  pane.innerHTML=`
    <div class="perf-top">
      <div class="perf-kpi"><div class="kv" style="color:var(--ok)">3.1×</div><div class="kk">端到端加速</div></div>
      <div class="perf-kpi"><div class="kv"><span style="color:var(--risk)">31%</span><span class="arw">→</span><span style="color:var(--ok)">82%</span></div><div class="kk">aicore 利用率</div></div>
      <div class="perf-kpi"><div class="kv" style="color:var(--ok)">76%</div><div class="kk">Cube 占用</div></div>
      <div class="perf-kpi"><div class="kv" style="color:var(--ok)">94%</div><div class="kk">MTE 隐藏</div></div>
    </div>

    <div class="perf-sec-h">流水泳道图 · msProf<span class="tag old">直译版</span></div>
    <div class="perf-play" id="perfPlay">▶ 播放泳道时序</div>
    ${swimHTML(before, play)}
    <div style="font-size:14px;color:var(--dim);margin:2px 0 0">串行搬运-计算,Cube/Vector 大量空转(斜纹),总耗时 ${before.total} cycle。</div>

    <div class="perf-sec-h">流水泳道图 · msProf<span class="tag new">优化版</span></div>
    ${swimHTML(after, play)}
    <div style="font-size:14px;color:var(--dim);margin:2px 0 0">双缓冲重叠,MTE 搬运隐藏在计算下,总耗时 ${after.total} cycle(约 ${speedup}× 缩短)。</div>

    <div class="perf-sec-h">利用率对比 · 直译 → 优化</div>
    ${cmpBar('aicore 总利用率', 31, 82)}
    ${cmpBar('Cube 占用率', 22, 76)}
    ${cmpBar('MTE 隐藏率', 12, 94)}

    <div class="perf-sec-h">调优发现与建议</div>
    <div class="perf-tune">
      <div class="pt-item"><span class="ic" style="color:var(--ok)">✓</span><div><b>双缓冲重叠</b> <span class="pv">已消除搬运气泡,流水气泡 21%→4%(见 S6)。</span></div></div>
      <div class="pt-item"><span class="ic" style="color:var(--ok)">✓</span><div><b>Cube FP8 满流水</b> <span class="pv">Mmad 连续无断流,Cube 占用 76%。</span></div></div>
      <div class="pt-item"><span class="ic" style="color:var(--warn)">◐</span><div><b>Vector 仍有空隙</b> <span class="pv">ReLU/归约与 Cube 存在轻微串行,可进一步用 UB Ping-Pong 重叠(潜在 +6%)。</span></div></div>
      <div class="pt-item"><span class="ic" style="color:var(--warn)">◐</span><div><b>末块尾效应</b> <span class="pv">nTile 末块无预取对象,建议按 sTile 对齐 S 长度以摊薄尾延迟。</span></div></div>
    </div>

    <div class="perf-reg"><b>✓ 已注册 aclNN 算子:</b> <code>aclnnLightningIndexer</code> —— 可供图层直接调用。端到端相较直译版 <b>3.1×</b> 加速,精度 cos_sim 0.99987。</div>`;
  const pb=document.getElementById('perfPlay');
  if(pb) pb.onclick=()=>renderPerfReport(true);
}
function openPerfPanel(){
  document.getElementById('perfTab').style.display='';
  activatePanelTab('perf');
  renderPerfReport(true);
}

/* ============================ S5 Tiling 可视化 ============================ */
const TILING_OPTS={
  A:{sTile:128, ub:61,  l0c:48, gm:16, cyc:'1.00', tag:'',       tagCls:''},
  B:{sTile:256, ub:88,  l0c:96, gm:8,  cyc:'0.72', tag:'推荐',   tagCls:'rec'},
  C:{sTile:512, ub:103, l0c:128,gm:4,  cyc:'0.95', tag:'溢出',   tagCls:'wrn'},
};
const S_TOTAL=2048; // 演示用 key 总长
function openTiling(){
  closeGraph(); closeCompare(); closePipe();
  document.getElementById('split').classList.add('tiling-open');
  renderTilingViz();
}
function closeTiling(){ document.getElementById('split').classList.remove('tiling-open'); }
function renderTilingViz(){
  const c=state.choices['S5']||'B';
  const o=TILING_OPTS[c];
  const nTile=Math.ceil(S_TOTAL/o.sTile);
  const full=Math.floor(S_TOTAL/o.sTile), tail=S_TOTAL - full*o.sTile;
  // S 维分块条
  let blks='';
  for(let i=0;i<nTile;i++){
    const isTail=(tail>0 && i===nTile-1);
    blks+=`<div class="sblk ${isTail?'tail':''}" data-i="${i}">${o.sTile}</div>`;
  }
  const ubCol=o.ub>100?'var(--risk)':(o.ub>=85?'var(--ok)':'var(--warn)');
  const l0cCol=o.l0c>100?'var(--risk)':(o.l0c>=85?'var(--ok)':'var(--cube)');
  const body=document.getElementById('tpBody');
  body.innerHTML=`
    <div class="tp-sec">
      <div class="h">分块方案 · sTile</div>
      <div class="tp-opts">
        ${Object.entries(TILING_OPTS).map(([k,v])=>`
          <div class="tp-opt ${k===c?'on':''}" data-v="${k}">
            <b>${v.sTile}</b><span>UB ${v.ub}%</span>
            ${v.tag?`<span class="${v.tagCls}">${v.tag}</span>`:'<span>&nbsp;</span>'}
          </div>`).join('')}
      </div>
    </div>

    <div class="tp-sec">
      <div class="h">key(S=${S_TOTAL}) 维分块 · nTile = ⌈S/sTile⌉ = ${nTile}</div>
      <div class="tp-anim" id="tpPlay">▶ 演示分块搬运过程</div>
      <div class="sbar" id="sbar">${blks}</div>
      <div class="sbar-cap"><span>← 沿 key 维流式载入,每块 sTile=${o.sTile}</span><span>${tail>0?`末块 ${tail}`:'整除'}</span></div>
    </div>

    <div class="tp-sec">
      <div class="h">片上缓冲占用 · 容量约束</div>
      <div class="util">
        <div class="ul"><span>Unified Buffer (UB)</span><b style="color:${ubCol}">${o.ub}%</b></div>
        <div class="track"><div class="fill" style="width:${Math.min(o.ub,100)}%;background:${ubCol}"></div><div class="cap-line" style="left:100%"></div></div>
      </div>
      <div class="util">
        <div class="ul"><span>L0C (矩阵输出)</span><b style="color:${l0cCol}">${o.l0c}%</b></div>
        <div class="track"><div class="fill" style="width:${Math.min(o.l0c,100)}%;background:${l0cCol}"></div><div class="cap-line" style="left:100%"></div></div>
      </div>
      ${o.ub>100||o.l0c>100?`<div style="font-size:14px;color:var(--risk);margin-top:4px">⚠ 超出片上容量 → 触发回退搬运(spill),cycles 反而升高</div>`:`<div style="font-size:14px;color:var(--ok);margin-top:4px">✓ 恰好贴合片上容量,驻留最大化</div>`}
    </div>

    <div class="tp-sec">
      <div class="h">代价评估</div>
      <div class="tp-metrics">
        <div class="tp-metric"><div class="mv">${nTile}</div><div class="mk">回 GM 次数 / 行</div></div>
        <div class="tp-metric"><div class="mv" style="color:${o.cyc==='0.72'?'var(--ok)':'#eef'}">${o.cyc}×</div><div class="mk">相对 cycles</div></div>
        <div class="tp-metric"><div class="mv">${o.sTile}</div><div class="mk">sTile</div></div>
      </div>
    </div>`;
  // 选项联动:更新选择 → 重渲染 tiling.h 与可视化
  body.querySelectorAll('.tp-opt').forEach(el=>el.onclick=()=>{
    state.choices['S5']=el.dataset.v;
    renderTilingViz();
    if(activeTab==='tiling') renderCode('tiling');       // 同步 tiling.h 源码
    renderWizard();                                       // 同步向导选项
  });
  const play=document.getElementById('tpPlay');
  if(play) play.onclick=()=>animateTiling(nTile);
}
let tileAnimTimer=null;
function animateTiling(nTile){
  if(tileAnimTimer){clearInterval(tileAnimTimer);tileAnimTimer=null;}
  const blks=document.querySelectorAll('#sbar .sblk');
  blks.forEach(b=>b.classList.remove('act'));
  let i=0;
  tileAnimTimer=setInterval(()=>{
    blks.forEach(b=>b.classList.remove('act'));
    if(i>=blks.length){ clearInterval(tileAnimTimer); tileAnimTimer=null; return; }
    blks[i].classList.add('act'); i++;
  }, Math.max(120, 1200/Math.max(nTile,1)));
}

/* ============================ S6 流水线前后对比可视化 ============================ */
// 三个 tile,时间以「格」为单位。op:mte(搬运2格)/cube(2格)/vec(1格)
const PIPE_TILES=3;
// 串行:每 tile 依次 MTE→Cube→Vec,单元间空档形成气泡
function buildSerial(){
  const rows={mte:[],cube:[],vec:[]}; let t=0;
  for(let n=0;n<PIPE_TILES;n++){
    rows.mte.push({s:t,w:2,l:`搬${n}`,cls:'mte'});
    // Cube 需等 MTE 完成 → 气泡
    rows.cube.push({s:t,w:2,l:'',cls:'bub'});          // Cube 空转等待
    rows.cube.push({s:t+2,w:2,l:`Cube${n}`,cls:'cube'});
    rows.vec.push({s:t+4,w:1,l:`Vec${n}`,cls:'vec'});
    t+=5;
  }
  return {rows,total:t};
}
// 双缓冲流水:MTE 连续预取,Cube 紧接上一块搬运后连续执行,Vec 跟随
function buildPipe(){
  const rows={mte:[],cube:[],vec:[]};
  // MTE 预热块0(2格),之后每块提前预取,连续排布
  for(let n=0;n<PIPE_TILES;n++) rows.mte.push({s:n*2,w:2,l:`搬${n}`,cls:'mte'});
  // Cube 从块0搬完(t=2)起连续执行,每块2格
  for(let n=0;n<PIPE_TILES;n++) rows.cube.push({s:2+n*2,w:2,l:`Cube${n}`,cls:'cube'});
  // Vec 跟在各自 Cube 之后
  for(let n=0;n<PIPE_TILES;n++) rows.vec.push({s:4+n*2,w:1,l:`Vec${n}`,cls:'vec'});
  const total=4+PIPE_TILES*2; // 末块 Cube 结束 + Vec
  return {rows,total};
}
function tlRowHTML(label, cells, total, play){
  const pct=x=>(x/total*100);
  let inner='';
  cells.forEach((c,i)=>{
    inner+=`<div class="tl-cell ${c.cls} ${play?'play':''}" style="left:${pct(c.s)}%;width:${pct(c.w)}%;${play?`animation-delay:${i*90}ms`:''}">${c.l}</div>`;
  });
  return `<div class="tl-row"><div class="tl-lbl">${label}</div><div class="tl-track">${inner}</div></div>`;
}
function timelineHTML(model, play){
  const {rows,total}=model;
  return `<div class="tl-rows">
    ${tlRowHTML('MTE 搬运', rows.mte, total, play)}
    ${tlRowHTML('Cube', rows.cube, total, play)}
    ${tlRowHTML('Vector', rows.vec, total, play)}
  </div>
  <div class="tl-axis"><span>t=0</span><span>时间 →</span><span>t=${total}</span></div>`;
}
function openPipe(){
  closeGraph(); closeCompare(); closeTiling();
  document.getElementById('split').classList.add('pipe-open');
  renderPipeViz(false);
}
function closePipe(){ document.getElementById('split').classList.remove('pipe-open'); }
function renderPipeViz(play){
  const ser=buildSerial(), pip=buildPipe();
  const serBubbles=ser.rows.cube.filter(c=>c.cls==='bub').length;
  const body=document.getElementById('ppBody');
  body.innerHTML=`
    <div class="pp-play" id="ppPlay">▶ 播放流水时序</div>
    <div class="pp-block">
      <div class="h"><span class="badge old">编排前</span>串行:搬运→计算 顺序执行</div>
      ${timelineHTML(ser, play)}
      <div style="font-size:14px;color:var(--dim);margin-top:5px">Cube 每块都要空等 MTE 搬运完成(斜纹为气泡),单元利用率低。</div>
    </div>
    <div class="pp-block">
      <div class="h"><span class="badge new">编排后</span>双缓冲 Ping-Pong:预取 n+1 ∥ 计算 n</div>
      ${timelineHTML(pip, play)}
      <div style="font-size:14px;color:var(--dim);margin-top:5px">TQue 深度 1→2,MTE 预取与 Cube/Vector 计算重叠,气泡几乎消除。</div>
    </div>
    <div class="pp-metrics">
      <div class="pp-metric"><div class="mv"><span style="color:var(--risk)">${ser.total}</span><span class="arw">→</span><span style="color:var(--ok)">${pip.total}</span></div><div class="mk">总周期(格)</div></div>
      <div class="pp-metric"><div class="mv"><span style="color:var(--risk)">21%</span><span class="arw">→</span><span style="color:var(--ok)">4%</span></div><div class="mk">流水气泡</div></div>
      <div class="pp-metric"><div class="mv" style="color:var(--ok)">${(ser.total/pip.total).toFixed(2)}×</div><div class="mk">吞吐提升</div></div>
    </div>`;
  const pb=document.getElementById('ppPlay');
  if(pb) pb.onclick=()=>renderPipeViz(true);
}

/* ============================ 计算图 ============================ */
// unit: mem|cube|vector|scalar|risk
const GNODES=[
  {id:'q', x:26,  y:86,  w:120,h:40, unit:'mem', t:'Q[b,q,h] · FP8', s:'GM→L1→L0A', d:'Query 向量,FP8 e4m3,每个 (batch, head) 对加载一个 query。搬入 L1 后进 L0A 供 Cube 读取。', lines:[20,25]},
  {id:'kv', x:200, y:14,  w:120,h:40, unit:'mem', t:'KV Cache · FP8', s:'GM→L1→L0B', d:'FP8 量化的 KV cache,每个 token 656 字节(512B NoPE + 16B scale + 128B RoPE)。按 TopK 稀疏索引分块载入。', lines:[28,32]},
  {id:'idx', x:26,  y:14, w:120,h:40, unit:'mem', t:'indices[b,topk]', s:'GM→UB', d:'稀疏 TopK 索引,指示每个 query 应该 attend 到哪些 KV。原为 CPU/GPU 预计算,昇腾映射到 Unified Buffer。', lines:[28,32]},
  {id:'qk', x:200, y:86,  w:120,h:44, unit:'cube', t:'QK^T 点积', s:'Cube · Mmad', d:'Q·K^T 的 FP8 矩阵乘,是算力主体。CUDA 里是手写循环累加,昇腾直接映射到 Cube 矩阵单元(Mmad)。', lines:[35,45]},
  {id:'sm', x:200, y:158, w:120,h:44, unit:'vector', t:'Softmax', s:'Vector · Exp/Reduce', d:'在线 Softmax:逐块更新 max 与 sum,exp 归一化。CUDA 用 __shfl_xor_sync 规约,昇腾映射到 Vector ReduceMax/Sum。', lines:[48,60]},
  {id:'shf',x:26, y:158, w:120,h:44, unit:'risk', gpuOnly:true, t:'block_reduce 规约', s:'GPU-only · 无直接适配', d:'依赖 warp 内 lane 间硬件 shuffle 做 max/sum 规约。达芬奇无线程/warp 概念,不是可直接映射的昇腾算子,S2 决策需替代为 Vector 片上归约。', lines:[16,28]},
  {id:'sync',x:26, y:230, w:120,h:40, unit:'risk', gpuOnly:true, t:'__syncthreads', s:'GPU-only · 无直接适配', d:'CUDA 线程块级同步屏障。昇腾无线程块同步模型,不是可直接映射的昇腾算子,需改写为 EnQue/DeQue 的流水同步(见 S6)。', lines:[30,45]},
  {id:'vac', x:200, y:230, w:120,h:44, unit:'vector', t:'V 累加', s:'Vector · Axpy', d:'加权累加 V:out += weight * V[k]。逐 token 累加,映射到 Vector 单元的 Axpy 操作。', lines:[62,70]},
  {id:'out',x:200,y:300, w:120,h:44, unit:'mem', t:'Output + LSE', s:'UB→GM', d:'注意力输出与 log-sum-exp 写回 Global Memory。LSE 用于后续层或 loss 计算。', lines:[72,76]},
];
const GEDGES=[['q','qk'],['kv','qk'],['idx','kv'],['qk','sm'],['sm','vac'],['kv','vac'],['shf','sm'],['sync','sm'],['vac','out']];
const UNITC={mem:'--mem',cube:'--cube',vector:'--vec',scalar:'--scalar',risk:'--risk'};
let graphMapped=false; // 经 S2 后 risk→vector

function unitColor(u){return getComputedStyle(document.documentElement).getPropertyValue(UNITC[u]).trim()}
function renderGraph(animate){
  const W=346,H=430;
  const eff=id=>{const n=GNODES.find(x=>x.id===id);let u=n.unit;if(graphMapped&&u==='risk')u='vector';return u;};
  let edges='';
  GEDGES.forEach(([a,b])=>{
    const na=GNODES.find(n=>n.id===a),nb=GNODES.find(n=>n.id===b);
    const x1=na.x+na.w/2,y1=na.y+na.h,x2=nb.x+nb.w/2,y2=nb.y;
    const my=(y1+y2)/2;
    const hot=(eff(a)==='risk'||eff(b)==='risk');
    edges+=`<path class="gedge${hot?' hot':''}" d="M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}"/>`;
  });
  let nodes='';
  GNODES.forEach((n,i)=>{
    const u=eff(n.id); const col=unitColor(u);
    const risk=(u==='risk');
    nodes+=`<g class="gnode${animate?' enter':''}" data-id="${n.id}" style="${animate?`animation-delay:${i*70}ms`:''}">
      ${risk?`<rect x="${n.x-3}" y="${n.y-3}" width="${n.w+6}" height="${n.h+6}" rx="${n.h/2+3}" fill="none" stroke="${col}" stroke-width="1.4" class="risk-pulse"/>`:''}
      <rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${risk ? n.h/2 : 9}" fill="${risk ? 'var(--surface-3)' : `${col}22`}" stroke="${risk ? 'var(--danger)' : col}" stroke-width="${risk ? 2.2 : 1.4}"/>
      <text class="nt" x="${n.x+11}" y="${n.y+ (n.h>42?19:18)}" fill="#eef" font-size="13">${n.t}</text>
      <text class="ns2" x="${n.x+11}" y="${n.y+(n.h>42?33:31)}" fill="${risk ? 'var(--danger)' : col}" font-size="12" font-family="ui-monospace,Menlo,Consolas,monospace">${n.s}</text>
      ${(graphMapped&&n.unit==='risk')?`<text x="${n.x+n.w-8}" y="${n.y+13}" text-anchor="end" fill="${col}" font-size="12" font-weight="700">✓改写</text>`:''}
    </g>`;
  });
  document.getElementById('gcanvas').innerHTML=
    `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="min-height:${H}px">
      <defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L9,5 L0,10 z" fill="#5a6076"/></marker></defs>
      ${edges}${nodes}
    </svg>`;
  document.querySelectorAll('.gnode').forEach(el=>el.addEventListener('click',()=>selectNode(el.dataset.id)));
}
function selectNode(id){
  document.querySelectorAll('.gnode').forEach(e=>e.classList.toggle('sel',e.dataset.id===id));
  const n=GNODES.find(x=>x.id===id); let u=n.unit; if(graphMapped&&u==='risk')u='vector';
  const label={mem:'片上搬运',cube:'Cube 矩阵',vector:'Vector 向量',scalar:'Scalar 标量',risk:'GPU-only · 无直接适配'}[u];
  const col=unitColor(u);
  let note=n.d;
  if(graphMapped&&n.unit==='risk') note='【已在 S2 改写】'+n.d.replace(/S2 决策.*$/,'现映射为 Vector 片上归约,见 S6 的 SelectTopK / TopK 原语。');
  document.getElementById('gdetail').innerHTML=
    `<span class="badge" style="background:${col}22;color:${col};border:1px solid ${col}66">${label}</span>`+
    `<b>${n.t}</b> · <code style="font-family:var(--mono);font-size:13px">${n.s}</code><br>`+
    `<span style="display:block;margin-top:6px">${note}</span>`;

  // 源码联动：计算图节点来自 CUDA 解析，始终定位到 .cu 源码
  if(n.lines && n.lines.length >= 2){
    if(activeTab !== 'cuda'){ switchTab('cuda'); }
    // 切换 tab 后 DOM 需要重新渲染，稍作延迟再高亮滚动
    requestAnimationFrame(()=>highlightCodeLines(n.lines[0], n.lines[1]));
  }
}

/* ---------- S2 算子映射清单（主内容区展示） ---------- */
// CUDA 算子 → 昇腾算子/执行单元 对照。risk 项随 S2 决策变化。
const OPMAP=[
  {cuda:'Q·K^T 手写循环累加', op:'Mmad', unit:'cube', node:'qk', rewrite:false},
  {cuda:'Exp(logits - max)', op:'Exp', unit:'vector', node:'sm', rewrite:false},
  {cuda:'Axpy: out += w·V', op:'Axpy', unit:'vector', node:'vac', rewrite:false},
  {cuda:'FP8 KV cache 解析', op:'DataCopy + 地址计算', unit:'mem', node:'kv', rewrite:false},
  {cuda:'indices[b,topk] 索引', op:'Unified Buffer', unit:'mem', node:'idx', rewrite:false},
  {cuda:'GM 载入 Q / KV', op:'DataCopy (GM→L1→L0)', unit:'mem', node:'q', rewrite:false},
  {cuda:'__shfl_xor_sync 规约', op:null, unit:'risk', node:'shf', rewrite:true},
  {cuda:'__syncthreads 同步', op:null, unit:'risk', node:'sync', rewrite:true},
];
const UNIT_LABEL={mem:'片上搬运',cube:'Cube 矩阵',vector:'Vector 向量',scalar:'Scalar 标量',risk:'GPU-only · 无直接适配'};
function renderOpMapTable(){
  const choice = state.choices['S2'] || 'vector';
  let rows='';
  OPMAP.forEach(m=>{
    let unit=m.unit, op=m.op, st, stCls, isRw=false;
    if(m.rewrite){
      // 依据 S2 决策决定重写目标
      if(choice==='scalar'){ unit='scalar'; op='Scalar 逐元素模拟'; }
      else { unit='vector'; op=(m.node==='bit')?'Vector 归约 + TopK 原语':'Vector 片上归约'; }
      st='需重写'; stCls='rw'; isRw=true;
    } else {
      st='直接映射'; stCls='ok';
    }
    const col=unitColor(unit);
    rows+=`<tr class="${isRw?'rw':''}">
      <td class="cuda">${m.cuda}</td>
      <td class="op">${op}</td>
      <td><span class="unit" style="color:${col}"><i style="background:${col}"></i>${UNIT_LABEL[unit]}</span></td>
      <td><span class="st ${stCls}">${st}</span></td>
    </tr>`;
  });
  const rwN=OPMAP.filter(m=>m.rewrite).length, okN=OPMAP.length-rwN;
  return `<div class="opmap">
    <div class="opmap-h">🗺 算子映射清单 · CUDA → 昇腾<span class="cnt">${okN} 直接映射 · ${rwN} 需重写</span></div>
    <table>
      <thead><tr><th>CUDA 算子</th><th>昇腾算子</th><th>执行单元</th><th>状态</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}
function syncParseBtn(){const open=document.getElementById('split').classList.contains('graph-open');
  document.getElementById('parseBtn').classList.toggle('on',open);}
function openGraph(){closeCompare();closeTiling();closePipe();document.getElementById('split').classList.add('graph-open');renderGraph(true);syncParseBtn();}
function closeGraph(){document.getElementById('split').classList.remove('graph-open');syncParseBtn();}

// 源码高亮联动函数
function highlightCodeLines(startLine, endLine){
  // 清除之前的高亮
  document.querySelectorAll('.ln.hl-node').forEach(el => el.classList.remove('hl-node'));

  // 添加新的高亮
  const codelines = document.getElementById('codelines');
  if(!codelines) return;

  const lines = codelines.querySelectorAll('.ln');
  for(let i = startLine - 1; i < endLine && i < lines.length; i++){
    lines[i].classList.add('hl-node');
  }

  // 滚动到可视区域（用 rect 计算，兼容 sticky gutter 与内边距）
  const targetLine = lines[startLine - 1];
  if(targetLine){
    const codewrap = document.getElementById('codewrap');
    const wrapRect = codewrap.getBoundingClientRect();
    const lineRect = targetLine.getBoundingClientRect();
    // 目标行当前相对滚动容器顶部的偏移 + 已有滚动量
    const lineOffsetInWrap = (lineRect.top - wrapRect.top) + codewrap.scrollTop;
    const scrollTarget = lineOffsetInWrap - codewrap.clientHeight / 3; // 显示在上 1/3 位置
    codewrap.scrollTo({top: Math.max(0, scrollTarget), behavior: 'smooth'});
  }
}

document.getElementById('gclose').onclick=closeGraph;
document.getElementById('tpClose').onclick=closeTiling;
document.getElementById('ppClose').onclick=closePipe;
document.getElementById('parseBtn').onclick=()=>{
  const open=document.getElementById('split').classList.contains('graph-open');
  if(open){closeGraph();}
  else{ openGraph();
    termLine('解析算子 → 生成计算图(手动触发)','d');
    if(state.step===0) notify('已打开计算图','这是 S1 的解析结果预览 · 点「运行 S1」可写入迁移流程','ok');
  }
};

/* ============================ 文件树 / Tabs ============================ */
let hasCpp=false, activeTab='cuda', tilingReady=false;
function renderTree(){
  const t=document.getElementById('tree');
  t.innerHTML=`
   <div class="node"><svg class="fic" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.6"><path d="m6 9 6 6 6-6"/></svg><b style="font-weight:600;color:#cfd6ea">DEEPSEEK-V3 · FLASH MLA</b></div>
   <div class="node ind"><svg class="fic" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.5"><path d="m6 9 6 6 6-6"/></svg>ops/</div>
   <div class="node ind2 ${activeTab==='cuda'?'sel':''}" data-open="cuda"><span class="dot-c" style="background:var(--cube)"></span>flash_mla_decode.cu</div>
   ${hasCpp?`<div class="node ind2 ${(activeTab!=='cuda'&&activeTab!=='tiling')?'sel':''}" data-open="cpp"><span class="dot-c" style="background:var(--acc)"></span>flash_mla_decode.cpp<span class="tag new">NEW</span></div>`:''}
   ${tilingReady?`<div class="node ind2 ${activeTab==='tiling'?'sel':''}" data-open="tiling"><span class="dot-c" style="background:var(--vec)"></span>tiling.h<span class="tag new">NEW</span></div>`:''}
   <div class="node ind2"><span class="dot-c" style="background:var(--dim2)"></span>mla_ref.py</div>
   <div class="node ind"><svg class="fic" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.5"><path d="m9 18 6-6-6-6"/></svg>tests/</div>
   ${hasCpp?`<div class="node ind"><svg class="fic" viewBox="0 0 24 24" fill="none" stroke="var(--dim)" stroke-width="1.5"><path d="m9 18 6-6-6-6"/></svg>build/</div>`:''}
  `;
  t.querySelectorAll('[data-open]').forEach(n=>n.onclick=()=>{
    const d=n.dataset.open;
    if(d==='cuda') switchTab('cuda');
    else if(d==='tiling') openTilingFile();
    else switchTab(codeKey());
  });
}
function codeKey(){ if(state.step>=6)return's6'; if(state.step>=4)return's4'; if(state.step>=3)return's3'; return'cuda'; }
function renderTabs(){
  const tabs=document.getElementById('tabs');
  let html=`<div class="tab ${activeTab==='cuda'?'on':''}" data-t="cuda">
     <span class="dot-c" style="background:var(--cube)"></span>flash_mla_decode.cu<span class="x">×</span></div>`;
  if(hasCpp) html+=`<div class="tab ${(activeTab!=='cuda'&&activeTab!=='tiling')?'on':''}" data-t="cpp">
     <span class="dot-c" style="background:var(--acc)"></span>flash_mla_decode.cpp<span class="x">×</span></div>`;
  if(tilingReady) html+=`<div class="tab ${activeTab==='tiling'?'on':''}" data-t="tiling">
     <span class="dot-c" style="background:var(--vec)"></span>tiling.h<span class="x">×</span></div>`;
  tabs.innerHTML=html;
  tabs.querySelectorAll('[data-t]').forEach(el=>el.onclick=()=>{
    const d=el.dataset.t;
    if(d==='cuda') switchTab('cuda');
    else if(d==='tiling') openTilingFile();
    else switchTab(codeKey());
  });
}
function switchTab(key){ closeCompare(); closeTiling(); closePipe(); activeTab = (key==='cuda')?'cuda':key; renderCode(activeTab==='cuda'?'cuda':key); renderTabs(); renderTree();
  document.getElementById('leftPaneH').style.display='none';
  const f=document.getElementById('etbFile'); if(f) f.textContent=(activeTab==='cuda')?'flash_mla_decode.cu':'flash_mla_decode.cpp'; }
// 打开 tiling.h 文件 + 右侧 Tiling 可视化
function openTilingFile(){
  closeCompare(); closeGraph(); closePipe();
  activeTab='tiling';
  renderCode('tiling');                 // 左侧显示 tiling.h
  renderTabs(); renderTree();
  document.getElementById('leftPaneH').style.display='none';
  const f=document.getElementById('etbFile'); if(f) f.textContent='tiling.h';
  openTiling();                         // 右侧 Tiling 可视化
}
// S6:定位回 AscendC 源码页签 + 高亮新增流水代码 + 右侧前后对比
function openS6Source(){
  closeGraph(); closeCompare(); closeTiling();
  activeTab='s6';
  renderCode('s6');                     // 左侧显示 AscendC(S6)源码
  renderTabs(); renderTree();
  document.getElementById('leftPaneH').style.display='none';
  const f=document.getElementById('etbFile'); if(f) f.textContent='lightning_indexer.cpp';
  openPipe();                           // 右侧流水前后对比
  // 高亮并滚动到新增的软件流水代码块(Process 内)
  requestAnimationFrame(()=>flashCodeLines(31,38));
}
// 在左侧代码面板闪烁高亮一段行并滚动
function flashCodeLines(a,b){
  const lns=document.querySelectorAll('#codelines .ln');
  lns.forEach(el=>el.classList.remove('hl-node'));
  for(let i=a-1;i<b && i<lns.length;i++) lns[i]?.classList.add('hl-node');
  if(lns[a-1]){
    const wrap=document.getElementById('codewrap');
    const wr=wrap.getBoundingClientRect(), lr=lns[a-1].getBoundingClientRect();
    wrap.scrollTo({top:Math.max(0,(lr.top-wr.top)+wrap.scrollTop - wrap.clientHeight/3), behavior:'smooth'});
  }
}

/* ============================ 步骤定义 ============================ */
const STEPS=[
 {n:'S1',t:'解析算子',sub:'CUDA AST → 计算图',
  body:`扫描 <code>fused_lightning_indexer_kernel</code>,抽取算子结构并生成计算图。识别为<b>「indexer 打分 + Top-K 选择」融合算子</b>:QKᵀ 点积 + ReLU + 跨头加权归约,再做 warp 双调排序取 Top-K。
  <div class="inspector-soft-card is-info" style="margin-top:12px">
    <div style="font-size:14px;color:var(--dim);margin-bottom:6px">💡 提示</div>
    <div style="font-size:14px;color:var(--txt)">点击右上角「解析算子 · 计算图」按钮打开计算图画布，然后点击各个节点可查看对应的源码位置</div>
  </div>`,
  risk:{h:'检测到 SIMT 专属结构',p:'<code>__shfl_xor_sync</code> warp 洗牌、<code>warp_bitonic_sort</code> 双调排序、<code>cg::thread_block</code> 与 <code>__shared__</code> —— 均依赖 GPU 线程/warp 硬件模型,昇腾达芬奇架构<b>无直接对应物</b>,须在 S2 决策改写。'},
  log:[['','ascendport migrate ./ops/lightning_indexer.cu','p'],
       ['解析 CUDA translation unit … 148 行','d'],
       ['✓ 识别 kernel: fused_lightning_indexer_kernel','g'],
       ['  ├─ 融合级别: 打分 + Top-K (2 stage fused)','d'],
       ['  ├─ 精度: FP8 e4m3 (qI/kI) · FP32 累加','d'],
       ['  └─ 并行粒度: 1 block = 1 query token','d'],
       ['构建数据流图 … 10 节点 / 11 边','b'],
       ['⚠ 检测 SIMT 专属原语 ×3: __shfl_xor_sync, warp_bitonic_sort, cg::thread_block','r'],
       ['✓ 计算图已生成 → 右侧画布','a']],
  run(){ hasCpp=false; graphMapped=false; renderTree(); renderTabs(); switchTab('cuda'); openGraph(); }},

 {n:'S2',t:'算子映射',sub:'算子 → 达芬奇执行单元',
  body:`把计算图里的每个 CUDA 算子映射到目标昇腾算子与达芬奇执行单元。下方清单列出全部映射结果 —— 多数可直接映射,仅 SIMT 专属的 warp 洗牌 + 双调排序<b>无对应物、需重写</b>。`,
  choice:{q:'warp shuffle 规约 + 双调排序 Top-K 如何在昇腾重写?',
    opts:[
     {v:'vector',rec:'推荐',title:'重写为 Vector 片上归约 + TopK 原语',
      desc:'用 Vector 单元的树形归约替代 lane-shuffle,Top-K 用 AscendC TopK 原语。充分利用向量算力,吞吐最高。'},
     {v:'scalar',warn:'不推荐',title:'Scalar 单元逐元素模拟',
      desc:'用标量循环逐个比较模拟 shuffle。语义等价但 Vector 单元闲置,严重浪费算力。'}]},
  log:[['','ascendport map --target davinci','p'],
       ['映射计算图节点 → 执行单元 …','d'],
       ['  QKᵀ 点积         → Cube  (Mmad, FP8)','g'],
       ['  ReLU / 加权归约   → Vector','g'],
       ['  Causal 掩码       → Vector','g'],
       ['  头权重 __shared__ → Unified Buffer','g']],
  logVector:[['  warp shuffle + 双调排序 → Vector 片上归约 + TopK 原语','g'],
       ['✓ 计算图 risk 节点已更新: SIMT → Vector','a'],
       ['⚠ 注意:并列分数下 Top-K 顺序可能与 GPU 不同 → S7 校验命中率','y']],
  logScalar:[['  warp shuffle + 双调排序 → Scalar 逐元素模拟','y'],
       ['⚠ Vector 单元将闲置,预计算力利用率 < 40% —— 不推荐','r']]},

 {n:'S3',t:'代码生成',sub:'SIMT → SPMD 分核',
  body:`生成 AscendC 骨架 <code>lightning_indexer.cpp</code>,并在编辑器<b>左 CUDA · 右 AscendC 同屏对比</b>。<code>blockIdx.x=t</code> 的 grid 映射为按 AI Core 分核(<code>GetBlockIdx()</code> 认领 query 行);warp/lane 内的打分循环改为核内分块循环。<code>SelectTopK</code> 以 Vector 归约桩替代 warp 双调排序。`,
  log:[['','ascendport codegen --arch ascend910b','p'],
       ['生成 AscendC kernel 类 …','d'],
       ['✓ 新建 lightning_indexer.cpp','g'],
       ['  ├─ Init/Process/ComputeScores/SelectTopK','d'],
       ['  ├─ grid(blockIdx.x) → GetBlockIdx() 分核','g'],
       ['  └─ warp 双调排序 → SelectTopK() 桩 (Vector)','g'],
       ['插入 2 处 TODO 标记 (S4 内存 / S6 Top-K)','y'],
       ['✓ 已开启 CUDA ↔ AscendC 同屏对比视图 (计算图已收起)','a']],
  run(){ hasCpp=true; renderTree(); renderTabs(); openCompare('s3'); }},

 {n:'S4',t:'内存层次映射',sub:'显式片上缓冲 + DataCopy',
  body:`为每处数据流动生成逐级搬运:<code>kI</code> FP8 走 GM→L1→L0B、<code>qI</code>→L0A、<code>QKᵀ</code> logits 落 L0C、ReLU/归约在 UB。这是 GPU 隐式缓存与昇腾显式缓冲的核心落差。右侧 AscendC 中<b>新注入的内存层次代码已高亮标记</b>,底部「数据流」面板以硬件单元为基础动画演示数据如何在 GM ↔ L1 ↔ L0 ↔ Cube ↔ UB ↔ Vector 之间流动。`,
  log:[['','ascendport memmap --emit-datacopy','p'],
       ['分析数据生命周期 … 5 个张量','d'],
       ['✓ 注入 InitBuffer × 5 (L1/L0A/L0B/L0C/UB)','g'],
       ['✓ 注入 DataCopy: kI GM→L1, qI GM→L1, w GM→UB','g'],
       ['✓ Mmad→L0C, Relu/WeightedHeadReduce→UB','g'],
       ['✓ 新注入代码已在 AscendC 侧高亮','a'],
       ['▶ 已生成硬件数据流动画 → 底部「数据流」面板','a'],
       ['当前为串行搬运-计算,S6 将做双缓冲重叠','y']],
  run(){ openCompare('s4'); }},

 {n:'S5',t:'自动 Tiling',sub:'贴合缓冲容量的分块',
  body:`沿 key(S 维)搜索分块长度 <code>sTile</code>,在 L1/L0/UB 容量约束下最大化片上驻留、最小化回 GM 次数,结果写入 <code>tiling.h</code>(默认打开)。右侧 <b>Tiling 可视化</b>直观呈现:S 维如何被切成 nTile 块、各方案的缓冲占用与代价。给出候选,由你确认:`,
  choice:{q:'选择 key 维分块方案 sTile:',
    opts:[
     {v:'A',title:'sTile = 128',desc:'UB 利用率 61% · 回 GM 次数多 · cycles 基线 1.00×'},
     {v:'B',rec:'推荐',title:'sTile = 256',desc:'UB 利用率 88% · L0C 恰好容纳 · cycles 0.72× —— 综合最优'},
     {v:'C',warn:'溢出风险',title:'sTile = 512',desc:'UB 利用率 103% · 超 L0C 容量 → 触发回退搬运,cycles 0.95×'}]},
  log:[['','ascendport tiling --search --constraint l0c,ub','p'],
       ['枚举 sTile ∈ {128,256,512} …','d'],
       ['  sTile=128 → UB 61%  cycles 1.00×','d'],
       ['  sTile=256 → UB 88%  cycles 0.72×  ★','g'],
       ['  sTile=512 → UB 103% 溢出回退 0.95×','y']],
  logDone:[['✓ tiling.h 已生成 (sTile 写入 TilingData)','a'],
       ['▶ 已打开 tiling.h 并生成 Tiling 可视化 → 右侧','a']],
  run(){ tilingReady=true; renderTree(); renderTabs(); }},

 {n:'S6',t:'流水线编排',sub:'双缓冲 Ping-Pong 重叠',
  body:`把串行的「搬运→计算」重排为软件流水:<b>预取 n+1 ∥ 计算 n ∥ 写回 n-1</b>。<code>TQue</code> 深度 1→2,让 MTE 搬运与 Cube/Vector 计算重叠 —— 这是开箱性能翻倍的关键。同时把 <code>SelectTopK</code> 落地为 Vector <code>TopK</code> 原语。完成后定位回 <code>lightning_indexer.cpp</code>,<b>高亮新增流水代码</b>,右侧给出编排前后的流水时序对比。`,
  log:[['','ascendport pipeline --double-buffer','p'],
       ['构建软件流水 …','d'],
       ['✓ TQue depth 1→2 (kL1/cO/ubS) Ping-Pong','g'],
       ['✓ 预取 CopyIn(n+1) 与 Compute(n) 重叠','g'],
       ['✓ SelectTopK 落地: warp 双调排序 → Vector TopK 原语','g'],
       ['✓ 已定位回 AscendC 源码并高亮新增流水代码','a'],
       ['▶ 流水前后对比 → 右侧面板','a'],
       ['流水气泡 21% → 4%','a']],
  run(){ /* 完成后在回调中定位源码 */ }},

 {n:'S7',t:'精度对齐',sub:'以 GPU 为黄金基准',
  body:`用相同输入跑昇腾 kernel 与 CUDA 参考,逐元素比对,生成<b>精度报告</b>(见底部页签)。报告会定位精度异常的算子、给出根因与修复方案 —— 一键应用修复即可复测通过。`,
  log:[['','ascendport verify --golden cuda --rtol 1e-3','p'],
       ['运行昇腾 kernel vs CUDA 参考 …','d'],
       ['逐算子比对 … 5 个算子','d'],
       ['  Mmad·QKᵀ 2.4e-4 ✓ · Relu 0 ✓ · DataCopy 0 ✓','g'],
       ['✗ WeightedHeadReduce: max_abs_err 3.1e-2 (超阈值 30×)','r'],
       ['  根因: FP8 累加顺序不一致 → 误差放大','y'],
       ['▶ 精度报告已生成 → 底部「精度报告」页签,可查看根因与修复方案','a']],
  run(){ /* 报告在完成回调中打开 */ }},

 {n:'S8',t:'性能剖析与调优',sub:'msProf → aclNN 注册',
  body:`采集硬件流水,定位瓶颈并给出调优建议,最后把算子注册为 <code>aclNN</code> 供图层调用。完成后生成<b>性能报告</b>(见底部页签):含 msProf <b>流水泳道图</b>(直译 vs 优化)、利用率对比与调优建议。相比直译版,端到端 <b>3.1×</b> 加速。`,
  log:[['','ascendport profile --with msprof','p'],
       ['采集 aicore pipe utilization …','d'],
       ['  直译版 aicore 利用率: 31%  (Cube 空转, 串行搬运)','y'],
       ['  优化版 aicore 利用率: 82%  (双缓冲重叠)','g'],
       ['  端到端加速: 3.1× · Cube 占用 76% · MTE 隐藏 94%','g'],
       ['✓ 注册 aclNN 算子: aclnnLightningIndexer','a'],
       ['▶ 性能报告已生成 → 底部「性能报告」页签','a'],
       ['✓ 迁移完成 —— S1→S8 全流程通过','a']],
  run(){ if(!accFixed){ accFixed=true; setProblems(0); } setAicore('82%'); }},
];

/* ============================ 状态机 ============================ */
const state={step:1, choices:{}, viewStep:0}; // 初始 step=1：S1 已完成，按钮执行 S2
function renderProg(){
  const p=document.getElementById('prog'), l=document.getElementById('plabels');
  const viewIndex = Math.max(0, Math.min(STEPS.length-1, Number.isFinite(state.viewStep)?state.viewStep:Math.max(0,state.step-1)));
  p.innerHTML=STEPS.map((s,i)=>`<button class="pstep ${i<state.step?'done':''} ${i===state.step?'cur':''} ${i===viewIndex?'view':''}" type="button" data-step-index="${i}" title="${s.n} · ${s.t}｜${s.sub}" aria-label="查看 ${s.n} ${s.t}"></button>`).join('');
  l.innerHTML=STEPS.map((s,i)=>`<button class="plabel ${i===viewIndex?'view':''}" type="button" data-step-index="${i}" title="${s.n} · ${s.t}">${s.n}</button>`).join('');
  [...p.querySelectorAll('[data-step-index]'), ...l.querySelectorAll('[data-step-index]')].forEach(el=>el.onclick=()=>{
    state.viewStep=Number(el.dataset.stepIndex);
    renderProg();
    renderWizard();
  });
}
function renderWizard(){
  const sc=document.getElementById('wzScroll');
  // step 是已执行进度；viewStep 只控制右侧当前查看的阶段。
  const defaultView = Math.max(0, Math.min(STEPS.length-1, state.step-1));
  const viewIndex = Math.max(0, Math.min(STEPS.length-1, Number.isFinite(state.viewStep)?state.viewStep:defaultView));
  const viewedStep = STEPS[viewIndex];
  const viewedDone = viewIndex < state.step;
  const viewedNext = viewIndex === state.step;
  const completedStep = state.step > 0 ? STEPS[Math.min(state.step - 1, STEPS.length - 1)] : null;
  const nextStep = state.step < STEPS.length ? STEPS[state.step] : null;

  let html='';

  if(viewedStep){
    const status = viewedDone ? '已完成' : (viewedNext ? '下一步' : '待执行');
    const icon = viewedDone ? '✓' : (viewedNext ? '▶' : viewedStep.n.replace('S',''));
    const cardTone = viewedDone
      ? 'border-color:var(--ok);background:#48d59708'
      : (viewedNext ? 'border-color:var(--primary);background:var(--state-selected)' : 'background:color-mix(in srgb, var(--surface-2) 64%, transparent)');
    html+=`<div class="stepcard" style="${cardTone}">
      <div class="sc-h"><div class="sc-n" style="${viewedDone?'background:#48d59722;color:var(--ok)':(viewedNext?'background:var(--state-selected);color:var(--primary)':'')}">${icon}</div>
        <div class="sc-t"><b>${viewedStep.n} · ${viewedStep.t}</b><span>${viewedStep.sub} · ${status}</span></div></div>
      <div class="sc-body">${viewedStep.body}`;
    // S2：在主内容区展示"算子映射清单"，直观呈现 CUDA 算子 → 昇腾算子/单元
    if(viewedStep.n==='S2') html+=renderOpMapTable();
    if(viewedStep.risk) html+=`<div class="riskcard"><div class="rh">⚠ ${viewedStep.risk.h}</div><p>${viewedStep.risk.p}</p></div>`;
    if(viewedStep.choice && viewIndex <= state.step){
      const sel=state.choices[viewedStep.n]||viewedStep.choice.opts.find(o=>o.rec)?.v||viewedStep.choice.opts[0].v;
      state.choices[viewedStep.n]=sel;
      html+=`<div class="choice"><div class="q">${viewedStep.choice.q}</div>`+
        viewedStep.choice.opts.map(o=>`<div class="opt ${o.v===sel?'on':''}" data-step="${viewedStep.n}" data-v="${o.v}">
          <div class="rd"></div><div class="ot"><b>${o.title} ${o.rec?`<span class="pill rec">${o.rec}</span>`:''}${o.warn?`<span class="pill warn">${o.warn}</span>`:''}</b>
          <span>${o.desc}</span></div></div>`).join('')+`</div>`;
    }
    html+=`</div></div>`;
  }

  if(!nextStep){
    // 全部完成
    html+=`<div class="stepcard" style="border-color:var(--ok);background:#48d5970d">
      <div class="sc-h"><div class="sc-n" style="background:#48d59722;color:var(--ok)">✓</div>
      <div class="sc-t"><b>迁移完成</b><span>S1 → S8 全流程通过</span></div></div>
      <div class="sc-body">Flash MLA 已迁移为 AscendC 算子并注册为 <code>aclnnLightningIndexer</code>。端到端 <b>3.1×</b> 加速,aicore 利用率 31%→82%,精度对齐 cos_sim 0.99987。</div></div>`;
  }
  sc.innerHTML=html;
  sc.querySelectorAll('.opt').forEach(o=>o.onclick=()=>{
    state.choices[o.dataset.step]=o.dataset.v;
    // 若在 S2 卡片上改变映射决策，实时反映到计算图
    if(o.dataset.step==='S2'){ graphMapped=(o.dataset.v==='vector'); renderGraph(false); }
    // 若在 S5 卡片上改变 tiling 决策，实时反映到 tiling.h 与可视化
    if(o.dataset.step==='S5'){
      if(document.getElementById('split').classList.contains('tiling-open')) renderTilingViz();
      if(activeTab==='tiling') renderCode('tiling');
    }
    renderWizard();
  });

  // footer
  const btn=document.getElementById('runBtn'), hint=document.getElementById('footHint');
  if(state.step>=STEPS.length){
    btn.disabled=false; btn.textContent='↻ 重新开始迁移'; btn.className='run ghost';
    hint.textContent='全部 8 个阶段已完成';
  } else {
    btn.disabled=false; btn.className='run';
    btn.textContent=`▶ 运行 ${nextStep.n} · ${nextStep.t}`;
    hint.textContent=`共 8 个阶段 · 当前 ${state.step} / 8 完成`;
  }
  document.getElementById('sbStep').textContent = state.step>=STEPS.length?'✓ 完成':(completedStep?`${completedStep.n} · 已完成`:'准备就绪');
}

/* ---------- terminal ---------- */
let termBusy=false;
function termLine(txt,cls){const d=document.createElement('div');d.className='tl';
  d.innerHTML=`<span class="t">$ </span><span class="${cls||''}">${txt}</span>`;
  if(cls==='p'){d.innerHTML=`<span class="t">➜ </span><span class="p">${txt}</span>`;}
  document.getElementById('term').appendChild(d);
  document.getElementById('term').scrollTop=1e9;}
function streamLog(lines,done){
  termBusy=true; let i=0; let finished=false;
  const term=document.getElementById('term');
  const cur=document.createElement('div');cur.className='tl';cur.innerHTML='<span class="cursor"></span>';
  term.appendChild(cur);
  const finish=()=>{ if(finished) return; finished=true; clearInterval(iv);
    if(cur.parentNode) cur.remove(); termBusy=false; done&&done(); };
  const iv=setInterval(()=>{
    if(i>=lines.length){ finish(); return; }
    const [txt,cls]=lines[i]; termLine(txt,cls); i++;
    term.appendChild(cur); term.scrollTop=1e9;
  },160);
  // 看门狗:无论中途发生什么,流式都会结束并恢复按钮/状态
  setTimeout(finish, lines.length*160 + 800);
}

/* ---------- problems ---------- */
let problems=2;
function setProblems(n){problems=n;const c=document.getElementById('probCnt');c.textContent=n;c.className='cnt'+(n>0?' err':'');
  const pl=document.getElementById('probs');
  if(n===0){pl.innerHTML=`<div class="prob" style="color:var(--ok)"><span class="pi">✓</span>无问题 —— 精度对齐通过</div>`;}
}
function initProblems(){
  const pl=document.getElementById('probs');
  pl.innerHTML=`
   <div class="prob"><span class="pi" style="color:var(--risk)">⚠</span><div><div><code style="font-family:var(--mono)">__shfl_xor_sync</code> 无昇腾对应物 —— 需重写为 Vector 归约</div><div class="pf">lightning_indexer.cu · 行 20</div></div></div>
   <div class="prob"><span class="pi" style="color:var(--risk)">⚠</span><div><div><code style="font-family:var(--mono)">warp_bitonic_sort</code> SIMT 双调排序 —— 需重写为 TopK 原语</div><div class="pf">lightning_indexer.cu · 行 27</div></div></div>`;
}
// S7：精度异常写入问题面板
function setAccProblem(){
  problems=1;const c=document.getElementById('probCnt');c.textContent=1;c.className='cnt err';
  document.getElementById('probs').innerHTML=`
   <div class="prob"><span class="pi" style="color:var(--risk)">⚠</span><div><div><code style="font-family:var(--mono)">WeightedHeadReduce</code> 精度异常 —— max_abs_err 3.1e-2 超阈值(FP8 累加序不一致)</div><div class="pf">lightning_indexer.cpp · 详见「精度报告」页签</div></div></div>`;
}

/* ---------- notifications ---------- */
function notify(title,msg,kind){
  const w=document.getElementById('notifs');const d=document.createElement('div');
  d.className='notif '+(kind||'');d.innerHTML=`<b>${title}</b><span>${msg}</span>`;
  w.appendChild(d);setTimeout(()=>{d.style.transition='opacity .4s,transform .4s';d.style.opacity=0;d.style.transform='translateX(20px)';setTimeout(()=>d.remove(),400)},3400);
}
function setAicore(v){document.getElementById('sbAicore').textContent='aicore '+v;}

/* ---------- panel tabs ---------- */
document.querySelectorAll('.ptab').forEach(t=>t.onclick=()=>{
  const p=t.dataset.p;
  if(p==='flow'){ if(document.getElementById('flowpane').innerHTML.trim()==='') renderFlow(); }
  else { stopFlow(); }
  if(p==='acc'){ renderAccReport(); }
  if(p==='perf'){ renderPerfReport(false); }
  activatePanelTab(p);
});

/* ---------- run a step ---------- */
function runStep(){
  if(termBusy) return;
  if(state.step>=STEPS.length){ reset(); return; }

  const s=STEPS[state.step]; // 执行下一步
  const btn=document.getElementById('runBtn');
  btn.disabled=true; btn.textContent=`⏳ ${s.n} · 运行中…`;
  document.getElementById('sbStep').textContent=`${s.n} · 运行中…`;
  stopFlow();

  // 切到终端标签,确保用户看到流式日志
  activatePanelTab('term');

  // assemble log with choice enrichment
  let lines=s.log.slice();
  if(s.n==='S2'){ const c=state.choices['S2']||'vector';
    lines=lines.concat(c==='vector'?s.logVector:s.logScalar); }
  if(s.n==='S5'){ const c=state.choices['S5']||'B';
    lines.push([`✓ 选定 sTile = ${c==='A'?128:c==='B'?256:512}`, c==='C'?'y':'g']);
    lines=lines.concat(s.logDone); }

  s.run && s.run();
  streamLog(lines,()=>{
    // graph mapping update after S2
    if(s.n==='S2'){ const c=state.choices['S2']||'vector'; if(c==='vector'){graphMapped=true;renderGraph(false);} }
    state.step++; // 完成后步骤+1
    state.viewStep=Math.min(state.step-1, STEPS.length-1);
    renderProg(); renderWizard();
    // S4：完成后打开硬件数据流动画并自动播放
    if(s.n==='S4'){ openFlowPanel(true); }
    // S5：完成后默认打开 tiling.h 并展示 Tiling 可视化
    if(s.n==='S5'){ openTilingFile(); }
    // S6：完成后定位回 AscendC 源码,高亮新增流水代码并展示前后对比
    if(s.n==='S6'){ openS6Source(); }
    // S7：完成后打开精度报告(异常态),用户可查看根因/修复方案并一键修复
    if(s.n==='S7'){ accFixed=false; setAccProblem(); openAccPanel(); }
    // S8：完成后打开性能报告(泳道图 + 对比)
    if(s.n==='S8'){ openPerfPanel(); }
    const done=state.step>=STEPS.length;
    notify(done?'🎉 迁移完成':`✓ ${s.n} 完成`, done?'Flash MLA 已注册为 aclNN 算子':`${s.t} —— ${s.sub}`, done?'ok':'ok');
    if(!done) document.getElementById('runBtn').disabled=false;
  });
}
function reset(){
  state.step=1; state.choices={}; state.viewStep=0; hasCpp=false; graphMapped=false; activeTab='cuda'; tilingReady=false; accFixed=false; // 重置到 S1 已完成状态
  document.getElementById('term').innerHTML='';
  closeCompare(); closeTiling(); closePipe(); stopFlow();
  document.getElementById('flowTab').style.display='none';
  document.getElementById('flowpane').style.display='none';
  document.getElementById('flowpane').innerHTML='';
  document.getElementById('accTab').style.display='none';
  document.getElementById('accpane').style.display='none';
  document.getElementById('accpane').innerHTML='';
  document.getElementById('perfTab').style.display='none';
  document.getElementById('perfpane').style.display='none';
  document.getElementById('perfpane').innerHTML='';
  initProblems(); setProblems(2); setAicore('—');
  renderTree(); renderTabs(); renderCode('cuda'); renderProg(); renderWizard();
  openGraph(); // S1 已完成，展示计算图
  termLine('AscendPort 迁移工作台 · 就绪。S1 解析已完成，点击右侧「运行 S2」继续。','d');
}
document.getElementById('runBtn').onclick=runStep;

/* ---------- boot ---------- */
initProblems();
renderTree(); renderTabs(); renderCode('cuda'); renderProg(); renderWizard();
termLine('AscendPort v0.9 · target=Atlas 800T A2 (Ascend 910B)','d');
termLine('✓ S1 解析算子已完成 — 已生成计算图，点击任意节点可定位源码。','g');
termLine('点击右侧「运行 S2 · 算子映射」继续迁移流程。','d');
// S1 已完成，打开计算图
openGraph();
