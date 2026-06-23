// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Emule shadow for blaze/.../compute_kernel_api/sdpa.h.
//
// The real header pulls the custom-MM / deepseek-hw-startup / llk_math_sdpa_*
// chain (row-major SFPU LLKs emule can't consume). The only single-chip
// consumer here, batched_qk_norm, uses just sdpa_reduce_row_init from it (to
// arm the zero-increment ADDR_MOD_7 dest addressing). We provide that, and pull
// internal/sfpu_instr.h for the raw SFPU-instruction layer the inlined
// batched_qk_norm_detail helpers need (TTI_SFP*, p_sfpu, ADDR_MOD_7, TT_SETC16,
// get_dest_buffer_base). The jit_hw -I root resolves ahead of the blaze tree,
// so this shadows the real header.
#pragma once

// batched_qk_norm gates its inline SFPU helper namespace (batched_qk_norm_detail)
// on TRISC_MATH. On silicon the compute kernel is built per-TRISC with TRISC_MATH
// defined for the math core; emule runs all roles in one unified thread and only
// sets COMPILE_FOR_TRISC, so that math-role code would be compiled out while
// operator() (COMPILE_FOR_TRISC) still calls it. This header is pulled via op.hpp
// before that #if, so defining TRISC_MATH here makes the math-role code visible.
// Scoped to sdpa-including ops; the curated gate doesn't include sdpa.h.
#ifndef TRISC_MATH
#define TRISC_MATH 1
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "internal/sfpu_instr.h"

namespace ckernel {

// On silicon this programs the SFPU config + the zero-increment dest addressing
// mode (ADDR_MOD_7). Under emule the SFPLOAD/SFPSTORE decode is absolute
// (base + imm), so the only state needed is the DST origin binding.
template <DataFormat format>
ALWI void sdpa_reduce_row_init() {
    MATH((::__emule_sfpu_bind_dst(&__emule_dst[0][0])));
}

}  // namespace ckernel
