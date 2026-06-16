// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/eltwise_unary/deep_sfpu_registry.h"
// Emulator stubs for trigonometric SFPU tile ops.
// All ops operate on DST (float32 storage) element-wise via std:: math.
//
// tanh has an optional deep path: define EMULE_DEEP_SFPU_TANH to run the real
// silicon ckernel_sfpu_tanh.h (3-piece SFPLUT) on emule's sfpi backend instead
// of the libm shadow. (The silicon tanh LUT is coarse — ~0.14 abs error near
// |x|=1 — so the deep path reproduces silicon, not torch.)

#include <cmath>
#include <cstdint>

#if defined(EMULE_DEEP_SFPU_TANH)
#include "jit_hw/internal/deep_sfpu.h"
#if defined(ARCH_BLACKHOLE)
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_tanh.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_tanh.h"
#else
#error "EMULE_DEEP_SFPU_TANH requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif
#endif

namespace ckernel {

ALWI void sin_tile_init() {}
ALWI void sin_tile(uint32_t idst) {
    __emule_dst_check(idst, "sin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sin(__emule_dst[idst][i]);
}

ALWI void cos_tile_init() {}
ALWI void cos_tile(uint32_t idst) {
    __emule_dst_check(idst, "cos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::cos(__emule_dst[idst][i]);
}

ALWI void tan_tile_init() {}
ALWI void tan_tile(uint32_t idst) {
    __emule_dst_check(idst, "tan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tan(__emule_dst[idst][i]);
}

// Templated on fast_and_approx to match upstream compute_kernel_api.h; the
// kernel SFPU_OP_CHAIN calls tanh_tile<0u>(0) / tanh_tile_init<0u>().
template <bool fast_and_approx = false>
ALWI void tanh_tile_init() {
#if defined(EMULE_DEEP_SFPU_TANH)
    ckernel::sfpu::_init_tanh_<fast_and_approx>();
#endif
}
template <bool fast_and_approx = false>
ALWI void tanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "tanh_tile");
#if defined(EMULE_DEEP_SFPU_TANH)
    __emule_deep::run_unary_sfpu(idst, [](int it) {
        ckernel::sfpu::_calculate_tanh_<fast_and_approx, __emule_deep::kTileIterations>(it);
    });
#else
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::tanh(__emule_dst[idst][i]);
#endif
}

ALWI void asin_tile_init() {}
ALWI void asin_tile(uint32_t idst) {
    __emule_dst_check(idst, "asin_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::asin(__emule_dst[idst][i]);
}

ALWI void acos_tile_init() {}
ALWI void acos_tile(uint32_t idst) {
    __emule_dst_check(idst, "acos_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::acos(__emule_dst[idst][i]);
}

ALWI void atan_tile_init() {}
ALWI void atan_tile(uint32_t idst) {
    __emule_dst_check(idst, "atan_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::atan(__emule_dst[idst][i]);
}

ALWI void acosh_tile_init() {}
ALWI void acosh_tile(uint32_t idst) {
    __emule_dst_check(idst, "acosh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::acosh(__emule_dst[idst][i]);
}

ALWI void asinh_tile_init() {}
ALWI void asinh_tile(uint32_t idst) {
    __emule_dst_check(idst, "asinh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::asinh(__emule_dst[idst][i]);
}

ALWI void atanh_tile_init() {}
ALWI void atanh_tile(uint32_t idst) {
    __emule_dst_check(idst, "atanh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::atanh(__emule_dst[idst][i]);
}

ALWI void cosh_tile_init() {}
ALWI void cosh_tile(uint32_t idst) {
    __emule_dst_check(idst, "cosh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::cosh(__emule_dst[idst][i]);
}

ALWI void sinh_tile_init() {}
ALWI void sinh_tile(uint32_t idst) {
    __emule_dst_check(idst, "sinh_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::sinh(__emule_dst[idst][i]);
}

} // namespace ckernel
