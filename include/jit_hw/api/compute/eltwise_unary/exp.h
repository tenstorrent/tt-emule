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

template <bool approx = false, bool fast_and_approx = true,
          bool scale_en = false, bool skip_positive_check = false,
          InputClamping input_clamping = InputClamping::ClampToNegative,
          int iterations = 8>
ALWI void exp_tile_init(uint32_t = 0, uint32_t = 0) {}

template <bool approx = false, bool fast_and_approx = true,
          bool scale_en = false, bool skip_positive_check = false,
          InputClamping input_clamping = InputClamping::ClampToNegative,
          int iterations = 8>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC,
                   uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    __emule_dst_check(idst, "exp_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::exp(__emule_dst[idst][i]);
}

} // namespace ckernel
