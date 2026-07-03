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
// emule version: point the sfpi cursor at __emule_compute_ctx().dst[dst_idx], reset the cursor +
// active-lane mask, then invoke the functor. The functor (e.g. calculate_recip/
// exponential/softplus_first_column, calculate_fused_max_sub_exp_add_tile) reads and
// writes sfpi::dst_reg, which now resolves into the selected DST tile.

#include <cstdint>

#include "jit_hw/sfpi.h"
#include "jit_hw/api/compute/common.h"       // __emule_compute_ctx().dst
#include "jit_hw/api/compute/vector_mode.h"  // ckernel::VectorMode

// Variadic dispatcher: point sfpi::dst_reg at DST[dst_idx], run the functor with the
// forwarded params. vector_mode is templated (VM) so callers can pass either
// `ckernel::VectorMode` or an int form.
//
// Every emule caller is an SDPA first-column COLUMN-VECTOR helper
// (calculate_{recip,exponential,softplus}_first_column, calculate_fused_max_sub_exp_add_tile),
// invoked with VectorMode::C. The per-row statistic they read/write lives in col 0 of all
// 32 rows (emule row-major, no face transpose — see __emule_compute_ctx().sfpu.first_col_mode in sfpi.h).
// The functor walks dst_reg[0] then `dst_reg += 2`, ITERATIONS_HALF_FACE=4 times — one
// functor call therefore covers only 4 four-row blocks = rows 0..15 (cursor 0,4,8,12).
// Silicon's _llk_math_eltwise_sfpu_apply_vector_mode_ invokes the functor once PER ACTIVE
// FACE (C → faces {0,2}), advancing the DST face address between, so the two invocations
// cover the whole column. emule mirrors that: under VectorMode::C, enable the linear col-0
// lane mode and call the functor twice WITHOUT resetting the cursor between calls — the 2nd
// call continues at cursor 16 and covers rows 16..31 of col 0. Other modes (RC for full-tile
// SFPU ops like clamped_silu) keep a single call + the general face-major mapping.
template <bool /*APPROX*/ = false, int /*DST_ACCUM_MODE*/ = 0, typename Fn, typename VM, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Fn fn,
    uint32_t dst_idx,
    VM vector_mode,
    Args... params) {
    const bool is_col = (static_cast<int>(vector_mode) == static_cast<int>(ckernel::VectorMode::C));
    // DST-slot contract before we take its address (matches the __emule_dst_check used
    // by pack_tile / functors; SFPU_UNARY_CALL's _sfpu_check_ defers the real bound here).
    __emule_dst_check(dst_idx, "_llk_math_eltwise_unary_sfpu_params_");
    // __emule_compute_ctx().sfpu.dst_base / _cursor / _first_col_mode are global scope; mask is in sfpi::.
    ::__emule_compute_ctx().sfpu.dst_base = &__emule_compute_ctx().dst[dst_idx][0];
    ::__emule_compute_ctx().sfpu.cursor = 0;
    ::__emule_compute_ctx().sfpu.first_col_mode = is_col;
    ::__emule_compute_ctx().sfpu.mask.fill(true);
    fn(params...);                       // covers col-0 rows 0..15 (cursor ends at 16)
    if (is_col) {
        fn(params...);                   // continues at cursor 16 → col-0 rows 16..31
    }
    ::__emule_compute_ctx().sfpu.dst_base = nullptr;
    ::__emule_compute_ctx().sfpu.cursor = 0;
    ::__emule_compute_ctx().sfpu.first_col_mode = false;
}
