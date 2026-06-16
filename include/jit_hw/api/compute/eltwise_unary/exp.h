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

// Scale baked into the SFPU exp constant by exp_tile_init's template `scale`
// parameter (an fp32 bit pattern, default 1.0f). On silicon's approx path,
// exp_init folds this into the 1/ln2 constant (A_scaled = A * scale_fp32 in
// ckernel_sfpu_exp.h:exp_init), so calculate_exponential computes exp(in * scale)
// regardless of the runtime scale_en flag. SDPA fuses the 1/sqrt(d) softmax scale
// here (exp_tile_init<true, scale_fp32, None> in compute_common.hpp). Persists
// across calls until the next init, mirroring the SFPU constant register.
static thread_local float __emule_exp_init_scale = 1.0f;

// Signatures mirror current upstream api/compute/eltwise_unary/exp.h.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    float s;
    uint32_t b = scale;
    std::memcpy(&s, &b, sizeof(s));
    __emule_exp_init_scale = s;
}

// exp(x * scale). The effective scale follows silicon:
//   - approx mode: the scale baked in by exp_tile_init (runtime scale_en ignored,
//     matching ckernel_sfpu_exp.h's approx path which uses the pre-loaded constant);
//   - non-approx mode: the runtime `scale` (a bf16/fp16b bit pattern) when scale_en,
//     else 1.0 — matching _ckernel_sfpu_exp_accurate_.
// vector_mode + iterations gate which faces are written (see vector_mode.h).
// input_clamping is a silicon SFPU-approximation knob with no effect on std::exp.
template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                   uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    __emule_dst_check(idst, "exp_tile");
    float s = 1.0f;
    if constexpr (approx) {
        s = __emule_exp_init_scale;
    } else if constexpr (scale_en) {
        uint32_t b = static_cast<uint32_t>(scale) << 16;
        std::memcpy(&s, &b, sizeof(s));
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (!__emule_vector_mode_active(i, vector_mode, iterations)) continue;
        float r = std::exp(__emule_dst[idst][i] * s);
        // Fully-masked-row degeneracy: SDPA masks out-of-window scores with bf16 -inf
        // (0xFF80). When an ENTIRE row of a K-chunk is masked, the row's running max is
        // also -inf, so the softmax max-subtraction computes exp(-inf - (-inf)) =
        // exp(NaN) = NaN. The mathematically-correct softmax numerator for a masked
        // entry is exp(-inf)=0, and that K-chunk must contribute nothing (a later
        // in-window K-chunk — at minimum the causal diagonal — supplies the row's real
        // output, and the online-softmax correction rescales this chunk's 0 away). So
        // collapse the NaN to 0, matching silicon's valid (non-NaN) output for these
        // rows. Partially-masked rows have a finite max, so exp(-inf - finite)=0 already
        // and never reach here; finite masks (e.g. -1e9) never produce -inf at all.
        __emule_dst[idst][i] = std::isnan(r) ? 0.0f : r;
    }
}

// Pack-thread variants: silicon runs the exp on the PACK thread instead of MATH.
// emule executes UNPACK/MATH/PACK inline in one thread, so these are identical to
// exp_tile{,_init} — delegate.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_packthread_tile_init() {
    exp_tile_init<approx, scale, input_clamping>();
}

template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_packthread_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                              uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    exp_tile<approx, scale_en, input_clamping, iterations>(idst, vector_mode, scale);
}

} // namespace ckernel
