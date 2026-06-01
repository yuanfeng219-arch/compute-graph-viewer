(function () {
    const a3Kernel = (window.WB_KERNELS || []).find((kernel) => kernel.id === 'c_api_add') || {};

    const A5_REGBASE_ADD_SOURCE_LINES = String.raw`/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/


/* !
 * \file c_api_add.asc
 * \brief 
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include "acl/acl.h"
#include "c_api/asc_simd.h"

constexpr uint32_t TILE_LENGTH = 2048;
constexpr uint32_t NUM_BLOCKS = 8;
constexpr uint32_t BLK_NUM = 1;
constexpr uint32_t MASK = 32;

__simd_vf__ inline void AddVF(uint16_t rep, uint16_t one_rep_size, uint32_t blockLength, __ubuf__ float* xLocal, __ubuf__ float* yLocal, __ubuf__ float* zLocal)
{
    vector_bool vmask;
    vector_float reg_src0;
    vector_float reg_src1;
    vector_float reg_dst;
    uint32_t remaining = blockLength;
    for (uint16_t i = 0; i < rep; ++i) {
        vmask = asc_update_mask_b32(remaining);
        asc_loadalign(reg_src0, xLocal + i * one_rep_size);
        asc_loadalign(reg_src1, yLocal + i * one_rep_size);    
        asc_add(reg_dst, reg_src0, reg_src1, vmask);
        asc_storealign(zLocal + i * one_rep_size, reg_dst, vmask);
    }
}

__vector__ __global__ __aicore__ void add_custom(__gm__ float* x, __gm__ float* y, __gm__ float* z)
{
    asc_init();

    uint32_t blockLength = TILE_LENGTH * NUM_BLOCKS / asc_get_block_num();

    __gm__ float* xGm = x + get_block_idx() * blockLength;
    __gm__ float* yGm = y + get_block_idx() * blockLength;
    __gm__ float* zGm = z + get_block_idx() * blockLength;

    __ubuf__ float xLocal[TILE_LENGTH];
    __ubuf__ float yLocal[TILE_LENGTH];
    __ubuf__ float zLocal[TILE_LENGTH];

    const uint8_t cacheMode0 = static_cast<uint8_t>(((uint64_t)xGm) >> 60);
    const uint8_t cacheMode1 = static_cast<uint8_t>(((uint64_t)yGm) >> 60);
    const uint8_t cacheMode2 = static_cast<uint8_t>(((uint64_t)zGm) >> 60);
    uint32_t burstLength = blockLength * 4;
    uint64_t srcStride = 0;
    uint32_t dstStride = 0;

    asc_copy_gm2ub_align((__ubuf__ float*)xLocal, xGm, BLK_NUM, burstLength, 0, 0, true, cacheMode0, srcStride, dstStride);
    asc_copy_gm2ub_align((__ubuf__ float*)yLocal, yGm, BLK_NUM, burstLength, 0, 0, true, cacheMode1, srcStride, dstStride);
    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);

    uint16_t mask_bit_size = 256;
    uint16_t one_rep_size = mask_bit_size / sizeof(float);
    uint16_t rep = (blockLength + one_rep_size - 1) / one_rep_size;
    AddVF(rep, one_rep_size, blockLength, (__ubuf__ float*)xLocal, (__ubuf__ float*)yLocal, (__ubuf__ float*)zLocal );
    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);

    asc_copy_ub2gm_align(zGm, (__ubuf__ float*)zLocal, BLK_NUM, burstLength, cacheMode2, srcStride, dstStride);
}

std::vector<float> kernel_add(std::vector<float> &x, std::vector<float> &y)
{
    constexpr uint32_t numBlocks = NUM_BLOCKS;
    uint32_t totalLength = x.size();
    size_t totalByteSize = totalLength * sizeof(float);
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    uint8_t *xHost = reinterpret_cast<uint8_t *>(x.data());
    uint8_t *yHost = reinterpret_cast<uint8_t *>(y.data());
    uint8_t *zHost = nullptr;
    float *xDevice = nullptr;
    float *yDevice = nullptr;
    float *zDevice = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);

    aclrtMallocHost((void **)(&zHost), totalByteSize);
    aclrtMalloc((void **)&xDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&yDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&zDevice, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy((uint8_t*)xDevice, totalByteSize, xHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy((uint8_t*)yDevice, totalByteSize, yHost, totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    add_custom<<<numBlocks, nullptr, stream>>>(xDevice, yDevice, zDevice);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(zHost, totalByteSize, (uint8_t*)zDevice, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<float> z((float *)zHost, (float *)(zHost + totalByteSize));

    aclrtFree(xDevice);
    aclrtFree(yDevice);
    aclrtFree(zDevice);
    aclrtFreeHost(zHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return z;
}

uint32_t VerifyResult(std::vector<float> &output, std::vector<float> &golden)
{
    auto printTensor = [](std::vector<float> &tensor, const char *name) {
        constexpr size_t maxPrintSize = 20;
        std::cout << name << ": ";
        std::copy(tensor.begin(), tensor.begin() + std::min(tensor.size(), maxPrintSize),
            std::ostream_iterator<float>(std::cout, " "));
        if (tensor.size() > maxPrintSize) {
            std::cout << "...";
        }
        std::cout << std::endl;
    };
    printTensor(output, "Output");
    printTensor(golden, "Golden");
    if (std::equal(golden.begin(), golden.end(), output.begin())) {
        std::cout << "[Success] Case accuracy is verification passed." << std::endl;
        return 0;
    } else {
        std::cout << "[Failed] Case accuracy is verification failed!" << std::endl;
        return 1;
    }
}

int32_t main(int32_t argc, char *argv[])
{
    constexpr uint32_t totalLength = NUM_BLOCKS * TILE_LENGTH;
    std::vector<float> x(totalLength);
    std::vector<float> y(totalLength);

    for (uint32_t i = 0; i < totalLength; ++i) {
        x[i] = i * 0.1f;
        y[i] = i * 0.1f;
    }
    std::vector<float> output = kernel_add(x, y);
    std::vector<float> golden(totalLength);
    for (uint32_t i = 0; i < totalLength; ++i) {
            golden[i] = x[i] + y[i];
    }
    return VerifyResult(output, golden);
}`.split('\n');

    const compareAnnotations = [
      {
        id: 'add-compare-copy-in',
        compareLines: { a5: [67, 68], a3: [42, 43] },
        kind: 'memory',
        tag: 'copy -> aligned copy',
        sourceTag: 'copy delta',
        code: 'asc_copy_gm2ub_align(...) / asc_copy_gm2ub(...)',
        short: 'A5 版本把 GM→UB 搬运升级为 aligned copy，显式 cache mode、burst 和 stride。',
        selectors: ['[data-mem950-node="rail:GM"]', '[data-mem950-node="rail:L2"]', '#mem950-aiv1 [data-aiv-node="buffer:UB"]'],
        routes: ['l2-to-aiv1'],
        path: 'A5 aligned GM→UB / A3 plain GM→UB',
        verdict: '访存入口差异',
        reasons: ['aligned copy', 'cache mode 显式化', 'burst/stride 可观察'],
        explanation: '同一个 Add 在 A5/950 版中把搬运参数显式化，便于对齐 128B sector、cache mode 和 stride 迁移检查。',
        rewrite: 'asc_copy_gm2ub_align(dst, src, BLK_NUM, burstLength, ..., cacheMode, srcStride, dstStride);',
        metrics: { confidence: '94%', cycles: '-12%', pressure: 'L2 可控' },
        deltaTags: ['copy -> aligned copy'],
        arch: {
          ascend950: {
            sourceTag: 'aligned copy',
            path: 'Global Memory → L2 → AIV UB · aligned copy',
            verdict: 'A5 aligned 搬运',
            instructionTags: ['MTE2'],
            deltaTags: ['A5: aligned copy', 'cache/stride explicit']
          },
          ascend910b: {
            sourceTag: 'plain copy',
            path: 'Global Memory → L2 → AIV UB · plain copy',
            verdict: 'A3 async baseline',
            deltaTags: ['A3: plain GM→UB', 'UB tensor baseline']
          }
        }
      },
      {
        id: 'add-compare-sync',
        compareLines: { a5: [69, 70, 76, 77], a3: [44, 47, 50] },
        kind: 'control',
        tag: 'plain sync -> pipe event',
        sourceTag: 'sync delta',
        code: 'asc_sync_notify/wait(...) / asc_sync()',
        short: 'A5 版本把全局 asc_sync 改成 MTE2/V、V/MTE3 的 producer-consumer event。',
        selectors: ['#mem950-aiv1 [data-aiv-node="scalar:Scalar"]', '#mem950-aiv1 [data-aiv-node="exec:SIMD"]', '#mem950-aiv1 [data-aiv-node="vector:Vector"]'],
        routes: [],
        path: 'Scalar event → MTE2/V → Vector → V/MTE3',
        verdict: '同步粒度收窄',
        reasons: ['pipe event', 'producer-consumer 边界', '减少保守同步'],
        explanation: 'A3 async 样例用 asc_sync 作为简单边界；A5 RegBase 样例明确 MTE2 到 Vector、Vector 到 MTE3 的 event 等待关系。',
        rewrite: 'asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0); asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);',
        metrics: { confidence: '91%', cycles: '-8%', pressure: '同步低' },
        deltaTags: ['plain sync -> pipe event'],
        arch: {
          ascend950: {
            sourceTag: 'pipe event',
            path: 'MTE2/V event → RegBase compute → V/MTE3 event',
            verdict: 'A5 event sync',
            instructionTags: ['MTE2', 'SIMD VF', 'MTE3'],
            deltaTags: ['A5: pipe event', 'MTE2/V + V/MTE3']
          },
          ascend910b: {
            sourceTag: 'plain sync',
            path: 'copy → asc_sync → vector → asc_sync → copy',
            verdict: 'A3 plain barrier',
            deltaTags: ['A3: asc_sync()', 'coarse barrier']
          }
        }
      },
      {
        id: 'add-compare-compute',
        compareLines: { a5: [30, 38, 39, 40, 41, 42, 75], a3: [46] },
        kind: 'compute',
        tag: 'UB tensor -> RegBase',
        sourceTag: 'compute delta',
        code: 'asc_loadalign + register asc_add + asc_storealign / asc_add(zLocal, xLocal, yLocal)',
        short: 'A5 版本把 UB tensor 计算展开为 RegBase/VF：loadalign 到寄存器、寄存器加法、storealign 回 UB。',
        selectors: ['#mem950-aiv1 [data-aiv-node="buffer:UB"]', '#mem950-aiv1 [data-aiv-node="exec:SIMD"]', '#mem950-aiv1 [data-aiv-node="vector:Vector"]'],
        routes: [],
        path: 'UB → Vector Register File → SIMD/VF add → UB',
        verdict: 'RegBase 计算路径',
        reasons: ['RegBase', 'LoadAlign/StoreAlign', 'mask 控制'],
        explanation: '同一 Add 计算从 A3 的 UB tensor API 变成 A5 的寄存器级 VF 函数，突出 950 AIV 从 memory based 到 register based 的迁移方向。',
        rewrite: 'asc_loadalign(reg_src0, xLocal + offset); asc_add(reg_dst, reg_src0, reg_src1, vmask);',
        metrics: { confidence: '97%', cycles: '-19%', pressure: 'Reg 中等' },
        deltaTags: ['UB tensor -> RegBase'],
        arch: {
          ascend950: {
            sourceTag: 'RegBase',
            path: 'AIV UB → Vector Register File → SIMD/VF',
            verdict: 'A5 RegBase',
            instructionTags: ['SIMD VF'],
            deltaTags: ['A5: RegBase/VF', 'loadalign/add/storealign']
          },
          ascend910b: {
            sourceTag: 'UB tensor',
            path: 'AIV UB → Vector tensor API',
            verdict: 'A3 UB tensor',
            deltaTags: ['A3: UB tensor API', 'asc_add(zLocal, xLocal, yLocal)']
          }
        }
      },
      {
        id: 'add-compare-copy-out',
        compareLines: { a5: [79], a3: [49] },
        kind: 'memory',
        tag: 'copy-out -> aligned copy-out',
        sourceTag: 'writeback delta',
        code: 'asc_copy_ub2gm_align(...) / asc_copy_ub2gm(...)',
        short: 'A5 版本把 UB→GM 写回也改成 aligned copy，和读入保持同一套 burst/stride 语义。',
        selectors: ['#mem950-aiv1 [data-aiv-node="buffer:UB"]', '[data-mem950-node="rail:L2"]', '[data-mem950-node="rail:GM"]'],
        routes: ['aiv1-to-l2'],
        path: 'AIV UB → L2 → GM · aligned writeback',
        verdict: '写回对齐差异',
        reasons: ['aligned writeback', 'burst length 显式', 'MTE3 边界清晰'],
        explanation: 'A5 写回路径和读入一样显式对齐参数，适合在架构图里把 MTE3 drain 与 L2/GM 写回观察点连起来。',
        rewrite: 'asc_copy_ub2gm_align(zGm, zLocal, BLK_NUM, burstLength, cacheMode2, srcStride, dstStride);',
        metrics: { confidence: '93%', cycles: '-10%', pressure: 'GM 低' },
        deltaTags: ['copy-out -> aligned copy-out'],
        arch: {
          ascend950: {
            sourceTag: 'aligned out',
            path: 'AIV UB → L2 → GM · aligned copy-out',
            verdict: 'A5 aligned writeback',
            instructionTags: ['MTE3'],
            deltaTags: ['A5: aligned writeback', 'MTE3 drain']
          },
          ascend910b: {
            sourceTag: 'plain out',
            path: 'AIV UB → L2 → GM · plain copy-out',
            verdict: 'A3 plain writeback',
            deltaTags: ['A3: plain UB→GM', 'baseline writeback']
          }
        }
      }
    ];

    window.registerWorkbenchKernel({
      id: 'c_api_add_a3_a5_compare',
      name: 'c_api_add.asc · A3/A5 对比',
      label: 'c_api_add A3/A5',
      path: 'A5 RegBase c_api_add.asc / A3 async c_api_add.asc',
      sourceUrl: 'https://gitcode.com/cann/asc-devkit/blob/master/examples/02_simd_c_api/00_introduction/04_reg_base_add_compute/c_api_simd_add/c_api_add.asc',
      summary: 'Official Add source comparison',
      target: 'Ascend 950 / Atlas A3',
      analysis: '官方源码 · A3/A5 对齐对比',
      verdict: 'UB tensor → RegBase',
      defaultTier: 't1',
      selectedId: 'add-compare-compute',
      sourceLines: A5_REGBASE_ADD_SOURCE_LINES,
      sourceCompare: {
        panes: [
          {
            id: 'a5',
            archId: 'ascend950',
            title: 'A5 / Ascend 950',
            subtitle: 'dav-3510 · RegBase C API',
            shortLabel: 'A5',
            badge: 'RegBase',
            sourceUrl: 'https://gitcode.com/cann/asc-devkit/blob/master/examples/02_simd_c_api/00_introduction/04_reg_base_add_compute/c_api_simd_add/c_api_add.asc',
            sourceLines: A5_REGBASE_ADD_SOURCE_LINES
          },
          {
            id: 'a3',
            archId: 'ascend910b',
            title: 'A3 / Atlas A3',
            subtitle: 'dav-2201 · async UB C API',
            shortLabel: 'A3',
            badge: 'UB tensor',
            sourceUrl: a3Kernel.sourceUrl,
            sourceLines: a3Kernel.sourceLines || []
          }
        ]
      },
      annotations: compareAnnotations
    });
})();
