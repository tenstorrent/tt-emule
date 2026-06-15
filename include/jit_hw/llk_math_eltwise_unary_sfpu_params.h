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
// emule version: invoke the function pointer once (no per-face iteration since
// we don't model SFPU lanes). The function body uses our sfpi:: shim
// (placeholder scalar semantics — see sfpi.h TODO). Real implementation will
// need to map silicon's 32-lane vector dispatch onto __emule_dst[idst][1024].

#include <cstdint>

#include "jit_hw/sfpi.h"

// Variadic dispatcher: accept any params after dst_idx + vector_mode and
// forward them to the function pointer. vector_mode is templated (VM) so callers
// can pass either `ckernel::VectorMode` or an int form.
template <bool /*APPROX*/ = false, int /*DST_ACCUM_MODE*/ = 0, typename Fn, typename VM, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(
    Fn /*fn*/,
    uint32_t /*dst_idx*/,
    VM /*vector_mode*/,
    Args... /*params*/) {
    // TODO (Issue 3): invoke fn with the sfpi cursor pointed at __emule_dst[idst]
    // so the first-column exp/recip/softplus functors actually execute. For now a
    // no-op so kernels parse + link; executed SFPU paths leave DST unchanged.
}
