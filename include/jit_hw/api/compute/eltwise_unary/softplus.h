// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for api/compute/eltwise_unary/softplus.h. The real header chains
// through ckernel_sfpu_softplus.h → ckernel_sfpu_exp.h → sfpi.h, which require
// RISC-V SFPU vector intrinsics that don't exist on x86 host. Override with
// no-op LLK stubs + a host-side softplus computed via std::log1p / std::exp.

#include <cstdint>
#include <cmath>
#include "api/compute/common.h"

namespace ckernel {

inline void softplus_tile_init() {}

// softplus(x; beta, threshold) = (1/beta) * log(1 + exp(beta*x)) for beta*x < threshold,
// else x. Real Tensix path uses iterative polyval/exp; emule uses host math.
inline void softplus_tile(uint32_t idst, uint32_t beta_packed = 0x3f800000u,
                           uint32_t beta_reciprocal_packed = 0x3f800000u,
                           uint32_t threshold_packed = 0x41a00000u) {
    __emule_dst_check(idst, "softplus_tile");
    float beta;            std::memcpy(&beta, &beta_packed, sizeof(float));
    float beta_recip;      std::memcpy(&beta_recip, &beta_reciprocal_packed, sizeof(float));
    float threshold;       std::memcpy(&threshold, &threshold_packed, sizeof(float));
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        float x = __emule_dst[idst][i];
        float bx = beta * x;
        __emule_dst[idst][i] = (bx > threshold) ? x : beta_recip * std::log1p(std::exp(bx));
    }
}

}  // namespace ckernel
