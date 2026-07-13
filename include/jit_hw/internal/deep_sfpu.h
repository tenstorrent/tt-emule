// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Deep-SFPU bridge.
//
// The "deep path" runs the REAL silicon ckernel_sfpu_<op>.h calculate
// functions on emule's faithful sfpi backend (include/jit_hw/sfpi.h) instead
// of a hand-written layer-1 scalar shim. A real calculate function iterates:
//
//     for (d = 0; d < iterations; d++) {
//         vFloat v = dst_reg[0]; ...; dst_reg[0] = result; dst_reg++;
//     }
//
// reading/writing 32-lane windows of the active DST tile via sfpi::dst_reg.
// This helper points the sfpi cursor at __emule_compute_ctx().dst[idst] for the duration of
// the call, then restores it — so a compute-API <op>_tile(idst) can delegate
// straight to the silicon math.
//
// See docs/sfpu-deep-path.md and api/compute/eltwise_unary/deep_sfpu_registry.h.

#include "jit_hw/api/compute/common.h"
#include "sfpi.h"

namespace __emule_deep {

// A full DST tile is 1024 fp32 elements = 32 windows of 32 sfpi lanes.
inline constexpr int kTileIterations =
    static_cast<int>(__EMULE_TILE_ELEMS / __EMULE_SFPI_LANES);

// Run a silicon unary-SFPU calculate function over DST tile `idst`.
// `calc(iterations)` is a callable wrapping the real ckernel::sfpu::_calculate_<op>_.
template <typename CalcFn>
inline void run_unary_sfpu(uint32_t idst, CalcFn&& calc, int iterations = kTileIterations) {
    auto& sfpu = __emule_compute_ctx().sfpu;
    float* prev_base = sfpu.dst_base;
    uint32_t prev_cursor = sfpu.cursor;
    sfpu.dst_base = &__emule_compute_ctx().dst[idst][0];
    sfpu.cursor = 0;
    calc(iterations);
    sfpu.dst_base = prev_base;
    sfpu.cursor = prev_cursor;
}

}  // namespace __emule_deep
