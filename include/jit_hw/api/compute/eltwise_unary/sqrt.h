// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/eltwise_unary/deep_sfpu_registry.h"
// Emulator stub for sqrt SFPU tile op.
//
// Default: layer-1 libm shadow (std::sqrt per element). Define
// EMULE_DEEP_SFPU_SQRT (e.g. a JIT -D) to promote sqrt to the deep path — the
// real silicon ckernel_sfpu_sqrt.h SQRT_23 calculate runs on emule's faithful
// sfpi backend. The default (else) branch below is left byte-for-byte unchanged.

#include <cmath>
#include <cstdint>

#if defined(EMULE_DEEP_SFPU_SQRT)
#include "jit_hw/internal/deep_sfpu.h"
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_sqrt.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_sqrt.h"
#else
#error "EMULE_DEEP_SFPU_SQRT requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif
#endif

namespace ckernel {

template <bool FAST_APPROX = false>
ALWI void sqrt_tile_init() {
#if defined(EMULE_DEEP_SFPU_SQRT)
    ckernel::sfpu::_init_sqrt_</*APPROXIMATION_MODE*/ false>();
#endif
}

template <bool FAST_APPROX = false>
ALWI void sqrt_tile(uint32_t idst) {
    __emule_dst_check(idst, "sqrt_tile");
#if defined(EMULE_DEEP_SFPU_SQRT)
    // Deep path: run the real silicon SQRT_23 over the whole DST tile.
    __emule_deep::run_unary_sfpu(idst, [](int it) {
        ckernel::sfpu::_calculate_sqrt_</*APPROXIMATION_MODE*/ false,
                                        __emule_deep::kTileIterations,
                                        /*fp32_dest_acc_en*/ true,
                                        /*FAST_APPROX*/ false>(it);
    });
#else
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sqrt(__emule_dst[idst][i]);
#endif
}

} // namespace ckernel
