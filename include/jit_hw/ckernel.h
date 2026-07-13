// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Minimal ckernel stub for JIT emulation.
// ThreadId and mailbox ops are now in jit_kernel_stubs.hpp.
// CSR emulation (NEO_ID, TRISC_ID) for Quasar compute kernels.
#include <cstdint>
#include "jit_hw/internal/emule_thread_ctx.h"

namespace ckernel {
enum class CSR : uint32_t {
    NEO_ID   = 0xBC2,
    TRISC_ID = 0xBC3,
};

template <CSR csr>
inline uint32_t csr_read();

template <>
inline uint32_t csr_read<CSR::NEO_ID>() { return __emule_self->neo_id; }

template <>
inline uint32_t csr_read<CSR::TRISC_ID>() { return __emule_self->trisc_id; }

// Hardware semaphore stubs. Silicon uses these to synchronize between
// UNPACK/MATH/PACK RISCs via T6 hardware semaphores. Emule runs all three
// on a unified compute thread, so semaphore ops are no-ops.
namespace semaphore {
constexpr uint32_t FPU_SFPU = 0;
constexpr uint32_t UNPACK_PACK = 1;
constexpr uint32_t UNPACK_TO_DEST = 2;
constexpr uint32_t UNPACK_MATH_DONE = 3;
constexpr uint32_t MATH_DONE = 4;
constexpr uint32_t MATH_PACK = 5;
constexpr uint32_t PACK_DONE = 6;  // SDPA streaming PACK/UNPACK sync
}  // namespace semaphore

// Note: deepseek_compute_kernel_hw_startup.h already defines
// `constexpr uint32_t SFPU_FPU = semaphore::UNPACK_MATH_DONE;` inside
// namespace ckernel — don't redefine here (would trip
// "redefinition of 'SFPU_FPU'"). Older kernels that reference
// ckernel::SFPU_FPU pick it up from the deepseek header transitively.

inline void t6_semaphore_init(uint32_t /*sem_idx*/, int /*min_val*/, int /*max_val*/) {}
template <uint32_t /*StallPoint*/ = 0>
inline void t6_semaphore_get(uint32_t /*sem_idx*/) {}
template <uint32_t /*StallPoint*/ = 0>
inline void t6_semaphore_post(uint32_t /*sem_idx*/) {}

// Operand enum (SRCA/SRCB/PACK) lives in silicon's
// api/compute/sentinel/sentinel_core.h. Kernels that use state_configure
// pull it in transitively via sentinel headers — no stub needed here.

// p_stall namespace — LLK stall point + stall trigger constants. Used in
// TTI_STALLWAIT() macro calls (which are no-op'd in emule). Values are
// bitmask-OR'd in some call sites, so distinct bits per constant.
namespace p_stall {
constexpr uint32_t NONE = 0;
constexpr uint32_t TRISC_SYNC = 1u << 0;
constexpr uint32_t THCON = 1u << 1;
constexpr uint32_t UNPACK = 1u << 2;
constexpr uint32_t PACK = 1u << 3;
constexpr uint32_t MATH = 1u << 4;
constexpr uint32_t SFPU = 1u << 5;
constexpr uint32_t XMOV = 1u << 6;
constexpr uint32_t WAIT_SFPU = 1u << 7;
constexpr uint32_t SRCA_VLD = 1u << 8;
constexpr uint32_t SRCB_VLD = 1u << 9;
constexpr uint32_t SRCA_CLR = 1u << 10;
constexpr uint32_t SRCB_CLR = 1u << 11;
constexpr uint32_t STALL_PACK = 1u << 12;
constexpr uint32_t STALL_MATH = 1u << 13;
constexpr uint32_t STALL_UNPACK = 1u << 14;
constexpr uint32_t STALL_SFPU = 1u << 15;
constexpr uint32_t STALL_CFG = 1u << 16;
constexpr uint32_t STALL_TDMA = 1u << 17;
constexpr uint32_t STALL_ON_ZERO = 1u << 18;
constexpr uint32_t STALL_ON_MAX = 1u << 19;
// TTI_STALLWAIT is a no-op in emule, so the value only needs to be distinct
// (silicon's is 0x20).
constexpr uint32_t STALL_THCON = 1u << 20;
constexpr uint32_t STALL_SYNC  = 1u << 21;  // SDPA streaming PACK/UNPACK sync stalls
}  // namespace p_stall

// state_configure lives in the silicon compute-kernel sentinel header alongside
// the Operand enum — kernels that call it pull both in transitively.

// Spin-wait on a semaphore reaching zero / max. No-ops (emule's unified
// compute thread doesn't have producer-consumer waits at the semaphore level).
template <uint32_t /*StallPoint*/>
inline void t6_semaphore_wait_on_zero(uint32_t /*sem_idx*/) {}
template <uint32_t /*StallPoint*/>
inline void t6_semaphore_wait_on_max(uint32_t /*sem_idx*/) {}

// ---- LLK type enums ----
// DataCopyType lives in llk_types.h (already in ckernel namespace).
// SfpuType + Transpose stubs for upstream kernels that reference them by name.
enum class SfpuType : uint8_t {
    unused = 0,
    exponential = 1,
    sqrt_op = 2,
    sigmoid = 3,
    tanh = 4,
    log = 5,
    silu = 6,
    relu = 7,
    gelu = 8,
    abs_op = 9,
    neg_op = 10,
};
enum class Transpose : uint8_t { Tile = 0, NoneT = 1, IntraFace = 2 };

// APPROX — upstream kernels uses this as a global compile-time flag for SFPU op
// approximation. Silicon sets it via a build define; emule defaults to false
// so kernels reference it via `if constexpr (APPROX) { ... }`.
#ifndef APPROX
inline constexpr bool APPROX = false;
#endif

// UnpackToDestEn lives in llk_types.h (global scope, not ckernel).

// SFPU op-name function stubs. upstream kernels uses `ckernel::sfpu::<op>_init` as a
// non-type template parameter for kernel init dispatch (e.g. via INIT_OP
// wrapper macros). Emule has no SFPU vector hardware — the templates just
// need to be valid for linkage; bodies are no-ops.
namespace sfpu {
template <bool APPROX = false> inline void silu_init() {}
template <bool APPROX = false> inline void exp_init() {}
template <bool APPROX = false> inline void gelu_init() {}
template <bool APPROX = false> inline void relu_init() {}
template <bool APPROX = false> inline void sqrt_init() {}
template <bool APPROX = false> inline void rsqrt_init() {}
template <bool APPROX = false> inline void sigmoid_init() {}
template <bool APPROX = false> inline void tanh_init() {}
template <bool APPROX = false> inline void log_init() {}
template <bool APPROX = false> inline void abs_init() {}
template <bool APPROX = false> inline void neg_init() {}
}  // namespace sfpu

}  // namespace ckernel
