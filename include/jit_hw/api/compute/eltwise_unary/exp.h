// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for exponential tile operations.

#include <cmath>
#include <cstdint>

enum class InputClamping : uint8_t {
    ClampToNegative = 1,
    None = 0,
};

// VectorMode lives in ckernel::; see api/compute/vector_mode.h.
#include "jit_hw/api/compute/vector_mode.h"

// Raw-TTI compile-only surface (TTI_SFP* / addr_mod_t / p_sfpu LREGs /
// InstrModLoadStore) for kernels whose never-instantiated branches reference it
// (e.g. SDPA's calculate_exponential_polynomial). Parse-only; never executed.
#include "jit_hw/ckernel_tti_stubs.h"

// SFPU functor dispatcher `_llk_math_eltwise_unary_sfpu_params_` used by SDPA's
// first-column exp/recip/softplus helpers. NOTE: currently a no-op (does not invoke
// the functor) — the executed first-column SFPU paths therefore won't compute yet;
// making it functional (cursor → __emule_dst[idst], run functor) is Issue-3 work.
#include "jit_hw/llk_math_eltwise_unary_sfpu_params.h"

// p_sfpu constants used by real exp_tile callers
namespace p_sfpu {
constexpr uint16_t kCONST_1_FP16B = 0x3F80;
} // namespace p_sfpu

namespace ckernel {

// Signatures mirror current upstream api/compute/eltwise_unary/exp.h.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {}

// exp(x), optionally exp(x * scale) when scale_en (scale is a bf16 bit pattern).
// vector_mode gates which faces are written (default RC = whole tile). approx /
// input_clamping / iterations are silicon SFPU-approximation knobs with no effect
// on emule's exact std::exp.
template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                   uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    __emule_dst_check(idst, "exp_tile");
    float s = 1.0f;
    if constexpr (scale_en) {
        uint32_t b = static_cast<uint32_t>(scale) << 16;
        std::memcpy(&s, &b, sizeof(s));
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (!__emule_vector_mode_active(i, vector_mode)) continue;
        __emule_dst[idst][i] = std::exp(__emule_dst[idst][i] * s);
    }
}

// Pack-thread variants: silicon runs the exp on the PACK thread instead of MATH.
// emule executes UNPACK/MATH/PACK inline in one thread, so these are identical to
// exp_tile{,_init} — delegate.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_packthread_tile_init() {}

template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_packthread_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                              uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    exp_tile<approx, scale_en, input_clamping, iterations>(idst, vector_mode, scale);
}

} // namespace ckernel
