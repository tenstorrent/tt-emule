// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// sqrt SFPU tile op — DEEP-DEFAULT.
//
// sqrt is modelled by running the REAL silicon ckernel_sfpu_sqrt.h (the Newton
// SQRT_23 algorithm) on emule's faithful sfpi backend — not a hand-written libm
// shadow. There is no opt-in macro and no libm fallback: this is the deep path
// as the default for an op (the end state the deep-SFPU work enables, and the
// shape any unshadowed op takes via the deep bridge). The SQRT_23 result is
// accurate to ~fp32 ULP, so this is numerically equivalent to std::sqrt while
// being faithful to silicon. See docs/sfpu-deep-path.md.

#include "jit_hw/internal/deep_sfpu.h"
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h"
#else
#error "deep-default sqrt requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif

namespace ckernel {

// APPROXIMATION_MODE is fixed false: emule does not thread the upstream `APPROX`
// op define (it is not defined in the emule JIT), and sqrt's approximate-init
// algorithm is not modelled separately. FAST_APPROX (the template param) only
// affects calculate_sqrt's final-clamp, not the constants — so sqrt_init
// stays consistent with calculate_sqrt regardless of FAST_APPROX.
template <bool FAST_APPROX = false>
ALWI void sqrt_tile_init() {
    // Program the SQRT_23 constant registers (vConstIntPrgm0 / vConstFloatPrgm1/2).
    // 2nd template arg (legacy_compat=false) disambiguates from emule's 1-param
    // no-op sqrt_init stub in ckernel.h, selecting the real metal-layer init.
    ckernel::sfpu::sqrt_init</*APPROXIMATION_MODE*/ false, /*legacy_compat*/ false>();
}

template <bool FAST_APPROX = false>
ALWI void sqrt_tile(uint32_t idst) {
    __emule_dst_check(idst, "sqrt_tile");
    // Self-init the constant regs so the op is correct even if a caller skips
    // sqrt_tile_init (idempotent — just sets three cregs).
    // 2nd template arg (legacy_compat=false) disambiguates from emule's 1-param
    // no-op sqrt_init stub in ckernel.h, selecting the real metal-layer init.
    ckernel::sfpu::sqrt_init</*APPROXIMATION_MODE*/ false, /*legacy_compat*/ false>();
    // Run the real silicon SQRT_23 over the whole DST tile via the deep bridge.
    // FAST_APPROX is threaded through to the real calculate (it selects the
    // fast-approx final-clamp path). ITERATIONS is now a template arg.
    __emule_deep::run_unary_sfpu(idst, [](int) {
        ckernel::sfpu::calculate_sqrt</*APPROXIMATION_MODE*/ false,
                                      __emule_deep::kTileIterations,
                                      /*fp32_dest_acc_en*/ true,
                                      FAST_APPROX>();
    });
}

} // namespace ckernel
