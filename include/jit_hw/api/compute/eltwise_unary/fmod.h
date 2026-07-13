// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stub for fmod SFPU tile op.
// Upstream: tt_metal/hw/inc/api/compute/eltwise_unary/fmod.h
// LLK family: llk_math_eltwise_unary_sfpu_fmod
//
// param0 is the uint32 bit-pattern of an fp32 divisor.
// param1 is the uint32 bit-pattern of the host-computed reciprocal (1/divisor).
// The reciprocal is an HW micro-optimization on silicon; the emule fallback
// uses std::fmod directly so the reciprocal is unused here.

#include <cmath>
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void fmod_tile_init(uint32_t param0, uint32_t param1) {
    // Upstream stages the divisor (param0) + reciprocal (param1) into SFPU const registers here;
    // mirror by staging the divisor per-fiber for fmod_tile (recip unused — emule uses std::fmod).
    (void)param1;
    __emule_compute_ctx().fmod_divisor_bits = param0;
}

ALWI void fmod_tile(uint32_t idst) {
    __emule_dst_check(idst, "fmod_tile");
    float divisor;
    const uint32_t param0 = __emule_compute_ctx().fmod_divisor_bits;
    std::memcpy(&divisor, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        __emule_compute_ctx().dst[idst][i] = std::fmod(__emule_compute_ctx().dst[idst][i], divisor);
    }
}

}  // namespace ckernel
