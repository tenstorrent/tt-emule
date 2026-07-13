// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator implementation of sigmoid tile operations.
//
// Default: Layer-1 libm shadow (1/(1+exp(-x)) per element). Define
// EMULE_DEEP_SFPU_SIGMOID to promote to the deep path — the real silicon
// ckernel_sfpu_sigmoid.h (6-piece SFPLUTFP32 LUT) runs on emule's faithful sfpi
// backend. The default branch is left byte-for-byte unchanged.

#include "jit_hw/api/compute/eltwise_unary/deep_sfpu_registry.h"
#include <cmath>
#include <cstdint>

#if defined(EMULE_DEEP_SFPU_SIGMOID)
#include "jit_hw/internal/deep_sfpu.h"
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sigmoid.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sigmoid.h"
#else
#error "EMULE_DEEP_SFPU_SIGMOID requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif
#endif

namespace ckernel {

template <bool approx = false>
ALWI void sigmoid_tile_init() {
#if defined(EMULE_DEEP_SFPU_SIGMOID)
    ckernel::sfpu::_init_sigmoid_<approx>();
#endif
}

template <int vector_mode = (int)VectorMode::RC, bool approx = false>
ALWI void sigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "sigmoid_tile");
#if defined(EMULE_DEEP_SFPU_SIGMOID)
    __emule_deep::run_unary_sfpu(idst, [](int it) {
        ckernel::sfpu::_calculate_sigmoid_<approx, __emule_deep::kTileIterations>(it);
    });
#else
    // Real impl: 1 / (1 + exp(-x)) per element in __emule_compute_ctx().dst[idst].
    float* d = __emule_compute_ctx().dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        d[i] = 1.0f / (1.0f + std::exp(-d[i]));
    }
#endif
}

}  // namespace ckernel
