// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/dropout.h`.
// `int_probability` is INT_MAX * prob (host encodes as `(uint32_t)(INT_MAX * prob)`).
// `scale_factor` is the uint32 bit-pattern of an fp32 scale value.
// Real LLK: tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_dropout.h
#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

// xorshift32 — fast deterministic PRNG for dropout emulation. One state per
// thread, seeded by dropout_kernel_init().

inline uint32_t __emule_dropout_next() {
    uint32_t x = __emule_compute_ctx().dropout_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    __emule_compute_ctx().dropout_rng_state = x;
    return x;
}

ALWI void dropout_kernel_init(uint32_t seed = 0) {
    // Avoid the all-zero degenerate xorshift state.
    __emule_compute_ctx().dropout_rng_state = seed ? seed : 0x9E3779B9u;
}

ALWI void dropout_tile(uint32_t idst, uint32_t int_probability, uint32_t scale_factor) {
    float scale;
    std::memcpy(&scale, &scale_factor, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        // Mask to [0, INT_MAX) so the threshold encoding matches the host's
        // `INT_MAX * prob` conversion.
        uint32_t r = __emule_dropout_next() & 0x7FFFFFFFu;
        if (r < int_probability) {
            __emule_compute_ctx().dst[idst][i] = 0.0f;
        } else {
            __emule_compute_ctx().dst[idst][i] *= scale;
        }
    }
}

}  // namespace ckernel
