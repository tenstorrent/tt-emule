// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulator stubs for activation / absolute-value tile operations.
// DST is __emule_dst[][] (float array); operations are host-side math.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ckernel {

ALWI void abs_tile_init() {}

ALWI void abs_tile(uint32_t idst) {
    __emule_dst_check(idst, "abs_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++)
        __emule_dst[idst][i] = std::fabs(__emule_dst[idst][i]);
}

ALWI void abs_tile_int32(uint32_t idst) {
    __emule_dst_check(idst, "abs_tile_int32");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        int32_t v = __emule_dst_load_i32(idst, i);
        if (v < 0) v = -v;
        __emule_dst_store_i32(idst, i, v);
    }
}

// hardsigmoid: clamp((x+3)/6, 0, 1)
ALWI void hardsigmoid_tile_init() {}

ALWI void hardsigmoid_tile(uint32_t idst) {
    __emule_dst_check(idst, "hardsigmoid_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = std::clamp((x + 3.0f) / 6.0f, 0.0f, 1.0f);
    }
}
// Pack-thread variant: upstream dispatches the same SFPU LLK via PACK() for
// matmul fused-activation overlap (#79). emule has no thread split — forward.
ALWI void hardsigmoid_tile_init_pack() { hardsigmoid_tile_init(); }
ALWI void hardsigmoid_tile_pack(uint32_t idst) { hardsigmoid_tile(idst); }

// softsign: x / (1 + |x|)
ALWI void softsign_tile_init() {}

ALWI void softsign_tile(uint32_t idst) {
    __emule_dst_check(idst, "softsign_tile");
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = x / (1.0f + std::fabs(x));
    }
}

// celu: max(0, x) + min(0, alpha * (exp(x/alpha) - 1)).
// alpha and alpha_recip arrive as uint32_t bit-patterns of fp32.
// We tried porting the upstream Cody-Waite expm1 form in round 6 but it
// regressed test_celu_allclose / test_celu_arange — reverted to the simpler
// formulation that the original round-5 shim used.
ALWI void celu_tile_init() {}

ALWI void celu_tile(uint32_t idst, uint32_t alpha, uint32_t alpha_recip) {
    __emule_dst_check(idst, "celu_tile");
    float a, ar;
    std::memcpy(&a, &alpha, sizeof(float));
    std::memcpy(&ar, &alpha_recip, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = std::max(0.0f, x) + std::min(0.0f, a * (std::exp(x * ar) - 1.0f));
    }
}

// softshrink: if |x| > lambda: x - sign(x)*lambda; else 0. param0 is fp32 bp of lambda.
ALWI void softshrink_tile_init() {}

ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "softshrink_tile");
    float lambda;
    std::memcpy(&lambda, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        if (x > lambda)       __emule_dst[idst][i] = x - lambda;
        else if (x < -lambda) __emule_dst[idst][i] = x + lambda;
        else                  __emule_dst[idst][i] = 0.0f;
    }
}

// hardshrink: if |x| > lambda: x; else 0. param0 is fp32 bp of lambda.
ALWI void hardshrink_tile_init() {}

ALWI void hardshrink_tile(uint32_t idst, uint32_t param0) {
    __emule_dst_check(idst, "hardshrink_tile");
    float lambda;
    std::memcpy(&lambda, &param0, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        __emule_dst[idst][i] = (std::fabs(x) > lambda) ? x : 0.0f;
    }
}

} // namespace ckernel
