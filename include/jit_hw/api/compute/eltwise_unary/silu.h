// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for SiLU (x * sigmoid(x)) tile operations.
//
// Provides ckernel::silu_tile_init / silu_tile + the lowered LLK
// silu/sigmoid init+apply templates used by the MatmulFusedAct.
//
// Default: layer-1 libm shadow (x/(1+exp(-x))). Define EMULE_DEEP_SFPU_SILU to
// promote to the deep path — the real silicon ckernel_sfpu_silu.h (piecewise
// linear + 5th-order-poly sigmoid) runs on emule's faithful sfpi backend. The
// default branch is left byte-for-byte unchanged.

#include "jit_hw/api/compute/eltwise_unary/deep_sfpu_registry.h"
#include <cmath>
#include <cstdint>

#if defined(EMULE_DEEP_SFPU_SILU)
#include "jit_hw/internal/deep_sfpu.h"
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_silu.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_silu.h"
#else
#error "EMULE_DEEP_SFPU_SILU requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif
#endif

namespace ckernel {

template <bool approx = false>
ALWI void silu_tile_init() {}

template <int vector_mode = (int)VectorMode::RC, bool approx = false>
ALWI void silu_tile(uint32_t idst) {
    __emule_dst_check(idst, "silu_tile");
#if defined(EMULE_DEEP_SFPU_SILU)
    __emule_deep::run_unary_sfpu(idst, [](int) {
        ckernel::sfpu::_calculate_silu_<approx, __emule_deep::kTileIterations>();
    });
#else
    // Real impl: x * sigmoid(x) per element in __emule_compute_ctx().dst[idst].
    float* d = __emule_compute_ctx().dst[idst];
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; ++i) {
        float x = d[i];
        d[i] = x / (1.0f + std::exp(-x));
    }
#endif
}

}  // namespace ckernel
