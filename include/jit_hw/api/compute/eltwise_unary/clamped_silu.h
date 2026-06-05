// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Layer-1 shadow for the ClampedSilu activation used by the upstream kernel (GPT-OSS SwiGLU).
//
// Silicon implements clamped_silu via an SFPU functor (`calculate_clamped_silu_gate`
// / `calculate_clamped_up` in the upstream kernel)
// dispatched through `_llk_math_eltwise_unary_sfpu_params_`. The functors
// use `sfpi::vFloat` SIMD types, which emule's `sfpi.h` provides only as a
// scalar placeholder. A full Layer-1 lift would require building out the
// `_llk_math_eltwise_unary_sfpu_params_` dispatcher to invoke the functor
// with proper sfpi context — pending; tracked.
//
// This partial lift moves the scalar math from the consumer op.hpp's
// `#ifdef __EMULE_JIT_MODE` body into here, exposing a single
// `clamped_silu_tile<gate_mode, fp32>(idst, alpha_bits, limit_bits, vector_mode)`
// entry. The consumer op.hpp `#ifdef` shrinks from ~20 lines to one call.

#include "jit_hw/api/compute/common.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

// Mode codes match the upstream kernel
inline constexpr uint32_t CLAMPED_SILU_TILE_MODE_GATE = 0;
inline constexpr uint32_t CLAMPED_SILU_TILE_MODE_UP   = 1;

// vector_mode_int: passes through the existing VectorMode enum value.
//   VectorMode::R (1) → top half of tile (512 elements)
//   VectorMode::RC (0) → full tile (1024 elements)
// alpha_bits / limit_bits: bit patterns of `alpha` and `limit` as fp32.
template <uint32_t mode, bool fp32_dest_acc_en>
ALWI void clamped_silu_tile(uint32_t idst,
                            uint32_t alpha_bits,
                            uint32_t limit_bits,
                            int vector_mode_int) {
    __emule_dst_check(idst, "clamped_silu_tile");
    __emule_dst_mark_dirty(idst);

    const uint32_t n = (vector_mode_int == static_cast<int>(VectorMode::R))
        ? (__EMULE_TILE_ELEMS / 2)
        : __EMULE_TILE_ELEMS;

    float* d = __emule_dst[idst];
    float limit;
    std::memcpy(&limit, &limit_bits, sizeof(float));

    if constexpr (mode == CLAMPED_SILU_TILE_MODE_GATE) {
        // clamp(x, max=limit) * sigmoid(alpha * clamp(x, max=limit))
        float alpha;
        std::memcpy(&alpha, &alpha_bits, sizeof(float));
        for (uint32_t j = 0; j < n; ++j) {
            float x = d[j];
            if (x > limit) x = limit;
            float sig = 1.0f / (1.0f + std::exp(-alpha * x));
            d[j] = x * sig;
        }
    } else {
        // clamp(x, -limit, limit) + 1.0
        for (uint32_t j = 0; j < n; ++j) {
            float x = d[j];
            if (x > limit) x = limit;
            if (x < -limit) x = -limit;
            d[j] = x + 1.0f;
        }
    }
}

template <uint32_t /*mode*/, bool /*fp32_dest_acc_en*/>
ALWI void clamped_silu_tile_init() {
    // Silicon programs SFPU sigmoid LUT under PACK pipeline; emule has none.
}

} // namespace ckernel
