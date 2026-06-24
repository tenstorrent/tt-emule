// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// LLK stubs for the matmul-based reduce path used by ttnn::sum-on-W via
// ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl.  Most stubs are no-ops
// (MATH/UNPACK/PACK macros are identity in emule); llk_math_matmul delegates
// to ckernel::matmul_tiles for actual math.

#include <cstdint>

// Forward-declare the enums.  Including llk_defs.h here would also pull in its
// dummy get_local_cb_interface(), which would shadow dataflow_api.h's real
// CB-state-backed version in DM kernels and silently break NOC writes.
namespace ckernel {
enum class MathFidelity : uint8_t;
}
enum class PoolType  : uint8_t;
enum class ReduceDim : uint8_t;
#include "jit_hw/api/compute/common.h"  // p_dim_stride_target (template default)

// constexpr defaults; only fall back to #define if the kernel source has
// already #defined them (some upstream prologs do this), to avoid a redefinition.
#ifndef MM_THROTTLE
constexpr int MM_THROTTLE = 0;
#endif
#ifndef MATH_FIDELITY
#define MATH_FIDELITY ckernel::MathFidelity::HiFi4
#endif

// Bridges UNPACK's tile selection across to MATH on the same host thread
// (struct __emule_matmul_bridge + the state live in ComputeThreadCtx, reached
// via __emule_compute_ctx().matmul_state).

inline void state_configure(uint32_t /*in1_cb*/, uint32_t /*in0_cb*/) {}

template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args>
inline void llk_math_matmul_init(Args... /*ignored*/) {}
template <typename... Args>
inline void llk_unpack_AB_matmul_init(Args... /*ignored*/) {}
// Reconfig stubs. Upstream signatures vary — some calls bind explicit
// `<bool is_fp32_dest_acc_en, p_dim_stride_target, bool to_from_int8>`
// non-type template args, others omit some. Use `auto... TArgs` so any
// combination of non-type args compiles. p_dim_stride_target itself is
// declared in common.h at global scope; don't redeclare it here.
// emule's compute path is format-agnostic so the body is a no-op regardless.
template <auto... TArgs>
inline void llk_unpack_reconfig_data_format_srca(uint32_t /*old_or_new*/ = 0, uint32_t /*new_operand*/ = 0) {}
template <auto... TArgs>
inline void llk_math_reconfig_data_format_srca(uint32_t /*old_or_new*/ = 0, uint32_t /*new_operand*/ = 0) {}
template <auto... TArgs>
inline void llk_unpack_reconfig_data_format_srcb(uint32_t /*old_or_new*/ = 0, uint32_t /*new_operand*/ = 0) {}
template <auto... TArgs>
inline void llk_math_reconfig_data_format_srcb(uint32_t /*old_or_new*/ = 0, uint32_t /*new_operand*/ = 0) {}
// No-op on WH/Quasar silicon (only BH has a body); emule models no remap state.
inline void llk_math_reconfig_remap(const bool /*remap_enable*/) {}

inline void llk_unpack_AB_matmul(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_idx, uint32_t in1_idx) {
    __emule_compute_ctx().matmul_state = {in0_cb, in1_cb, in0_idx, in1_idx};
}

// Forward decl: ckernel::matmul_tiles is defined in jit_hw/api/compute/matmul.h,
// which transitively includes this header.  llk_math_matmul below is a template,
// so it instantiates lazily — by the time a kernel actually calls it, matmul.h
// has been fully parsed and the definition is in scope.  Don't call
// llk_math_matmul from a non-template context in this file or you'll hit an
// unresolved-reference at link time.
namespace ckernel {
void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile, uint32_t in1_tile, uint32_t idst);
}
template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args>
inline void llk_math_matmul(uint32_t idst, Args... /*ignored*/) {
    ckernel::matmul_tiles(
        __emule_compute_ctx().matmul_state.in0_cb,
        __emule_compute_ctx().matmul_state.in1_cb,
        __emule_compute_ctx().matmul_state.in0_idx,
        __emule_compute_ctx().matmul_state.in1_idx,
        idst);
}

template <PoolType Pool, ReduceDim Dim, typename... Args>
inline void llk_unpack_AB_reduce_init(Args... /*ignored*/) {}
template <PoolType Pool, ReduceDim Dim, int Mode, ckernel::MathFidelity Fidelity, typename... Args>
inline void llk_math_reduce_init(Args... /*ignored*/) {}
