// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Compute kernel API stub for emulated mode.
//
// tt-mlir's TTKernelIncludesMap.h maps a set of SFPU ops directly to
// `api/compute/compute_kernel_api.h` (rather than to a more specific
// per-op header). On real silicon those ops are pulled in through
// transitive includes; in tt-emule we declare them here directly.
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "jit_hw/api/compute/experimental/fill_arange.h"
#include "api/compute/eltwise_unary/activations.h"  // abs_tile, abs_tile_int32

#include <cmath>
#include <cstdint>

namespace ckernel {

// --- square (x*x) ---
ALWI void square_tile_init() {}
ALWI void square_tile(uint32_t idst) {
    __emule_dst_check(idst, "square_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x * x;
    }
}

// --- exp2 (2^x) ---
ALWI void exp2_tile_init() {}
ALWI void exp2_tile(uint32_t idst) {
    __emule_dst_check(idst, "exp2_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::exp2(__emule_dst[idst][i]);
}

// --- expm1 (e^x - 1) ---
ALWI void expm1_tile_init() {}
ALWI void expm1_tile(uint32_t idst) {
    __emule_dst_check(idst, "expm1_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::expm1(__emule_dst[idst][i]);
}

// --- log ---
template <bool fast_and_approx = false>
ALWI void log_tile_init() {}
template <bool fast_and_approx = false>
ALWI void log_tile(uint32_t idst) {
    __emule_dst_check(idst, "log_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::log(__emule_dst[idst][i]);
}

// --- power (x^exponent_packed_float) ---
ALWI void power_tile_init() {}
ALWI void power_tile(uint32_t idst, uint32_t exponent_packed = 0) {
    __emule_dst_check(idst, "power_tile");
    if (exponent_packed == 0) {
        fprintf(stderr, "[EMULE] power_tile: exponent_packed=0; emule cannot "
                        "distinguish 'not provided' from packed 0.0f. Caller "
                        "must pass packed bits of the exponent.\n");
        std::abort();
    }
    float exponent;
    std::memcpy(&exponent, &exponent_packed, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::pow(__emule_dst[idst][i], exponent);
}

// --- sigmoid (1 / (1 + e^-x)) ---
ALWI void sigmoid_tile_init() {}
ALWI void sigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "sigmoid_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = 1.0f / (1.0f + std::exp(-x));
    }
}

// --- sign (-1, 0, +1) ---
ALWI void sign_tile_init() {}
ALWI void sign_tile(uint32_t idst) {
    __emule_dst_check(idst, "sign_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = (x > 0.0f) - (x < 0.0f);
    }
}

// --- signbit (1 if signed-bit set, else 0) ---
ALWI void signbit_tile_init() {}
ALWI void signbit_tile(uint32_t idst) {
    __emule_dst_check(idst, "signbit_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::signbit(__emule_dst[idst][i]) ? 1.0f : 0.0f;
}

// --- silu (x * sigmoid(x)) ---
ALWI void silu_tile_init() {}
ALWI void silu_tile(uint32_t idst) {
    __emule_dst_check(idst, "silu_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x / (1.0f + std::exp(-x));
    }
}

// --- sfpu_reduce — SFPU-based reduction helper; emule reduces via other paths.
// No-op stubs let JIT compile; actual reduction tests use llk_defs.h reduce helpers.
template <typename ...Ts>
ALWI void sfpu_reduce(Ts...) {}
template <typename ...Ts>
ALWI void sfpu_reduce_init(Ts...) {}

} // namespace ckernel
