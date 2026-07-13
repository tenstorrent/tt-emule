// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/eltwise_unary/polygamma.h`.
//   psi^(n)(x) ~= (-1)^(n+1) * n! * [ sum_{k=0}^{10} 1/(x+k)^(n+1) + R_EM(z=x+11, n) ]
// `n_packed` / `scale_packed` are uint32 bit-patterns of fp32 (n as float,
// scale = (-1)^(n+1) * n!).
// Real LLK: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_polygamma.h

#include <cstdint>
#include <cstring>

#include "jit_hw/api/compute/common.h"

namespace ckernel {

ALWI void polygamma_tile_init() {}

ALWI void polygamma_tile(uint32_t idst, uint32_t n_packed, uint32_t scale_packed) {
    __emule_dst_check(idst, "polygamma_tile");

    float n_float;
    std::memcpy(&n_float, &n_packed, sizeof(float));
    float scale;
    std::memcpy(&scale, &scale_packed, sizeof(float));

    const int n = static_cast<int>(n_float);
    const int n_plus_1 = n + 1;

    // Bernoulli-derived tail coefficients (matches ckernel_sfpu_polygamma.h):
    //   c_b2 = (n+1)/12
    //   c_b4 = -(n+1)(n+2)(n+3)/720
    //   c_b6 = (n+1)(n+2)(n+3)(n+4)(n+5)/30240
    const float n1 = static_cast<float>(n + 1);
    const float n2 = static_cast<float>(n + 2);
    const float n3 = static_cast<float>(n + 3);
    const float n4 = static_cast<float>(n + 4);
    const float n5 = static_cast<float>(n + 5);
    const float nf = static_cast<float>(n);
    const float inv_nf = (n != 0) ? (1.0f / nf) : 0.0f;
    const float c_b2 = n1 / 12.0f;
    const float c_b4 = -(n1 * n2 * n3) / 720.0f;
    const float c_b6 = (n1 * n2 * n3 * n4 * n5) / 30240.0f;

    constexpr int NUM_TERMS = 11;  // exact terms k = 0..10

    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        const float x = __emule_compute_ctx().dst[idst][i];

        // Part 1: exact summation of first NUM_TERMS reciprocal powers.
        float sum = 0.0f;
        for (int k = 0; k < NUM_TERMS; k++) {
            const float xi = x + static_cast<float>(k);
            const float inv_xi = 1.0f / xi;
            float inv_power = inv_xi;
            for (int j = 1; j < n_plus_1; j++) {
                inv_power *= inv_xi;
            }
            sum += inv_power;
        }

        // Part 2: Euler-Maclaurin tail at z = x + NUM_TERMS.
        const float z = x + static_cast<float>(NUM_TERMS);
        const float inv_z = 1.0f / z;
        const float inv_z2 = inv_z * inv_z;

        // inv_z^n by repeated multiplication.
        float inv_z_n = inv_z;
        for (int j = 1; j < n; j++) {
            inv_z_n *= inv_z;
        }

        // Polynomial in inv_z2: E = inv_nf + c_b2*y + c_b4*y^2 + c_b6*y^3
        const float y = inv_z2;
        const float E = inv_nf + y * (c_b2 + y * (c_b4 + y * c_b6));
        const float tail = inv_z_n * (E + 0.5f * inv_z);

        sum += tail;

        __emule_compute_ctx().dst[idst][i] = sum * scale;
    }
}

}  // namespace ckernel
