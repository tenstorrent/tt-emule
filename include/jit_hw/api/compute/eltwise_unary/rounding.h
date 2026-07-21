// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stubs for rounding SFPU tile ops.
// trunc, ceil, floor operate element-wise on DST float32 storage; frac = x - trunc(x).

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

ALWI void rounding_op_tile_init() {}

ALWI void round_tile_init() {}

// Round each element to `decimals` decimal places (round-to-nearest-even).
// Real LLK: _calculate_round_ = (1/10^d) * round_even(x * 10^d).
ALWI void round_tile(uint32_t idst, int32_t decimals) {
    __emule_dst_check(idst, "round_tile");
    float scale = std::pow(10.0f, (float)decimals);
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = std::nearbyint(x * scale) / scale;
    }
}

ALWI void trunc_tile(uint32_t idst) {
    __emule_dst_check(idst, "trunc_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = std::trunc(__emule_compute_ctx().dst[idst][i]);
}

ALWI void ceil_tile(uint32_t idst) {
    __emule_dst_check(idst, "ceil_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = std::ceil(__emule_compute_ctx().dst[idst][i]);
}

ALWI void floor_tile(uint32_t idst) {
    __emule_dst_check(idst, "floor_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_compute_ctx().dst[idst][i] = std::floor(__emule_compute_ctx().dst[idst][i]);
}

ALWI void frac_tile(uint32_t idst) {
    __emule_dst_check(idst, "frac_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_compute_ctx().dst[idst][i];
        __emule_compute_ctx().dst[idst][i] = x - std::trunc(x);
    }
}

// Stochastic rounding of each fp32 element to bf16 precision: round up one bf16 ULP
// with probability proportional to the dropped low-16 mantissa bits (unbiased in
// expectation). Real LLK: _calculate_stochastic_round_ (SFPU LFSR). Emule has no SFPU
// LFSR, so it draws from the per-fiber xorshift32 RNG — the algorithm is faithful
// (unbiased round-to-bf16); the exact random sequence is not bit-identical to silicon.
ALWI void stochastic_round_tile(uint32_t idst) {
    __emule_dst_check(idst, "stochastic_round_tile");
    uint32_t& rng = __emule_compute_ctx().dropout_rng_state;
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        uint32_t bits;
        std::memcpy(&bits, &__emule_compute_ctx().dst[idst][i], sizeof(bits));
        const uint32_t dropped = bits & 0x0000FFFFu;  // low 16 mantissa bits bf16 discards
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        const uint32_t draw = rng & 0x0000FFFFu;
        uint32_t rounded = bits & 0xFFFF0000u;
        if (dropped > draw) {
            rounded += 0x00010000u;  // round up one bf16 ULP (carries into exponent at max mantissa)
        }
        float out;
        std::memcpy(&out, &rounded, sizeof(out));
        __emule_compute_ctx().dst[idst][i] = out;
    }
}

} // namespace ckernel
