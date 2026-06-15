// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// emule stub of the silicon LLK math eltwise unary SFPU params layer.
//
// Silicon provides `_llk_math_eltwise_unary_sfpu_params_<...>(fn_ptr, dst_idx,
// vector_mode, params...)` — a dispatcher that calls the supplied SFPU
// function pointer per face/iteration with appropriate sfpi context. an upstream kernel's
// clamped_silu kernel passes `ckernel::sfpu::calculate_clamped_silu_gate<...>`
// as the fn_ptr.
//
// emule version: point the sfpi cursor at __emule_dst[dst_idx], reset the cursor +
// active-lane mask, then invoke the functor. The functor (e.g. calculate_recip/
// exponential/softplus_first_column, calculate_fused_max_sub_exp_add_tile) reads and
// writes sfpi::dst_reg, which now resolves into the selected DST tile.

#include <cstdint>

#include "jit_hw/sfpi.h"
#include "jit_hw/api/compute/common.h"  // __emule_dst

// Variadic dispatcher: point sfpi::dst_reg at DST[dst_idx], run the functor with the
// forwarded params. vector_mode is templated (VM) so callers can pass either
// `ckernel::VectorMode` or an int form. (The functor does its own per-iteration
// dst_reg traversal; vector_mode coverage rides on that.)
template <bool /*APPROX*/ = false, int /*DST_ACCUM_MODE*/ = 0, typename Fn, typename VM, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Fn fn,
    uint32_t dst_idx,
    VM /*vector_mode*/,
    Args... params) {
    // __emule_sfpi_dst_base / _cursor are global scope; __emule_sfpi_mask is in sfpi::.
    ::__emule_sfpi_dst_base = &__emule_dst[dst_idx][0];
    ::__emule_sfpi_cursor = 0;
    ::sfpi::__emule_sfpi_mask.fill(true);
    fn(params...);
    ::__emule_sfpi_dst_base = nullptr;
    ::__emule_sfpi_cursor = 0;
}
