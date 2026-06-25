// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stubs included at the TOP of every JIT wrapper before the kernel source.
// Provides everything a kernel needs that isn't in its own includes.

// Mark that we're in JIT compilation mode.  Headers like dataflow_api.hpp
// use this to call extern "C" bridge functions instead of inlining Device
// methods (which would have wrong ABI when TT_EMULE_USE_XY_PAIR is active
// in the host process but not in the JIT .so).
#define __EMULE_JIT_MODE 1
// TODO(#103): audit __EMULE_JIT_MODE usage across the codebase.

// Mark this TU as a kernel build so silicon host-device message headers
// take their firmware branch instead of the host-only #error branch.
#ifndef KERNEL_BUILD
#define KERNEL_BUILD
#endif

// ---- Host/target size_t reconciliation: mixed-type std::min / std::max ----
// Silicon's Tensix RISC cores are 32-bit, so size_t == uint32_t there, and kernels
// freely write e.g. std::min(uint32_t_expr, count * sizeof(T)) — both operands
// deduce to uint32_t and std::min's single-type template binds fine. On emule's
// 64-bit host, size_t is 64-bit, so that same call has mismatched operand types
// (uint32_t vs unsigned long) and std::min fails to deduce a common type
// (e.g. data_movement/scatter/.../writer_scatter.cpp). Add overloads that engage
// ONLY when the two operands are DIFFERENT integral types (SFINAE), promoting to
// their common type. Same-type calls are SFINAE'd out of these and still bind to
// the real std::min/std::max, so there is no ambiguity and no behavior change.
// Faithful in result (the same value silicon computes); the kernel source is
// unchanged. Lives in the JIT preamble (included before every kernel).
// Adding these to namespace std is formally [namespace.std] UB; accepted deliberately
// because the toolchain is pinned (clang-20 + libstdc++) and there is no conforming way
// to satisfy a *qualified* std::min(size_t, uint32_t) call without editing the pristine
// upstream kernel. The integral-only, different-type SFINAE keeps the blast radius to
// exactly the host/target size_t-width mismatch and never shadows a same-type call.
#include <algorithm>
#include <type_traits>
namespace std {
template <class A, class B,
          enable_if_t<!is_same_v<A, B> && is_integral_v<A> && is_integral_v<B>, int> = 0>
constexpr common_type_t<A, B> min(const A& a, const B& b) {
    using C = common_type_t<A, B>;
    return static_cast<C>(b) < static_cast<C>(a) ? static_cast<C>(b) : static_cast<C>(a);
}
template <class A, class B,
          enable_if_t<!is_same_v<A, B> && is_integral_v<A> && is_integral_v<B>, int> = 0>
constexpr common_type_t<A, B> max(const A& a, const B& b) {
    using C = common_type_t<A, B>;
    return static_cast<C>(a) < static_cast<C>(b) ? static_cast<C>(b) : static_cast<C>(a);
}
}  // namespace std

// NOC mode constants — silicon firmware sets these per-RISC build.
#ifndef DM_DEDICATED_NOC
#define DM_DEDICATED_NOC 0
#endif
#ifndef DM_DYNAMIC_NOC
#define DM_DYNAMIC_NOC 1
#endif
// noc_mode / noc_index are defined together further down, next to the other
// firmware-mirrored constants, once <cstdint> (uint8_t) is in scope.

// tensix_sync — silicon's TRISC sync barrier. emule runs all TRISCs on a
// unified compute thread, so this is a no-op. Universal because compute
// kernels call it directly from generated code (patched_kernel.cpp).
inline void tensix_sync() {}

// ALWI — silicon's "always inline" attribute used in tt-metal kernel
// headers (ttnn/kernel_lib/tilize_helpers.hpp etc.) that don't pull in
// api/compute/common.h before declaring ALWI-prefixed prototypes.
// Universal because the ttnn-side tilize kernel includes
// tilize_helpers.hpp before any common.h header.
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

// stdint.h — provides uint32_t in the global namespace (needed by ttnn
// kernel_lib headers that use uint32_t without std:: qualification).
#include <stdint.h>

#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"
#include "emule_cb_state.h"
#include "emule_dfb_state.h"
#include "tools/profiler/kernel_profiler.hpp"

// CB-interface surface. Silicon kernels (and consumer kernel-lib templates
// that wrap `reconfig_cbs_for_mask`) reference `cb_addr_shift`,
// `LocalCBInterface`, `get_local_cb_interface`, `get_cb_tiles_received_ptr`,
// and `get_cb_tiles_acked_ptr` directly. On silicon these come transitively
// via the per-RISC firmware include chain. Under emule the JIT wrapper has
// no such chain — pull the symbols in here so every JIT TU resolves them
// regardless of `COMPILE_FOR_*` defines (which emule's runner does not set).
#include "api/cb_api.h"
#include "internal/cb_interface.h"

// `firmware_common.h` provides invalidate_l1_cache / flush_l1_cache /
// WAYPOINT — needed by dataflow kernels via `experimental::semaphore_wait`.
// Pull it in here so every JIT kernel sees it. The heavier LLK surface
// (DataCopyType, UnpackToDestEn, llk_math_* templates, __llk_pack_* /
// __llk_unpack_* state, experimental::pack_untilize_block) is scoped to
// the compute entry point `api/compute/compute_kernel_hw_startup.h` —
// pulling it into every TU corrupts the SFPU INT32 unary tile-data path.
#include "internal/firmware_common.h"

#include <cstdint>
#include <cstring>

// compute_kernel_hw_startup lean overloads. Silicon's full version pulls in
// heavy LLK state that corrupts SFPU INT32 paths (see comment above), so
// we provide empty overloads here so kernels that call this from a path
// not transiting api/compute/common.h still resolve. The full LLK setup
// happens in api/compute/compute_kernel_hw_startup.h for kernels that
// explicitly include it.
// Guard so that api/compute/compute_kernel_hw_startup.h (which has its own
// definition with real reset of __llk_pack_offset / __emule_l1_acc_enabled)
// doesn't double-define. First-included wins.
#ifndef __EMULE_COMPUTE_KERNEL_HW_STARTUP_DEFINED
#define __EMULE_COMPUTE_KERNEL_HW_STARTUP_DEFINED
// #46346: SrcOrder (matmul uses Reverse) — accepted-and-ignored in emule.
enum class SrcOrder : uint8_t { Regular = 0, Reverse = 1 };
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {}
template <SrcOrder = SrcOrder::Regular>
inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t) { compute_kernel_hw_startup(a, b); }
#endif

// Forward declarations matching kernel_runner.cpp definitions.
// The main executable exports these with -rdynamic; the JIT .so resolves them
// at dlopen() time.
namespace tt_emule { class Core; }
// Per-thread L1 pointers set by the runner's kernel-launch lambda.
// nullptr = no args for this RISC.
extern thread_local uint32_t* __rt_args;
extern thread_local uint32_t* __common_rt_args;
extern thread_local tt_emule::Core*       __core;

// C-linkage bridge functions — resolve at dlopen time to the host process's
// implementations, avoiding ABI mismatch with Device layout.
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr);

// L1 bridge pointer — host pointer to L1 memory start.
// Used to convert between uint32_t L1 "addresses" (really: truncated pointers
// from mmap'd-below-4GB L1) and real host pointers.
extern thread_local uint8_t* __emule_bridge_l1;

// Translate a raw L1 firmware offset (or already-absolute host pointer) to a
// host uint8_t*.  Available to ALL JIT kernels so the l1_arg_ptr regex patch in
// emulated_program_runner can inject calls without requiring dataflow_api.h.
#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED
#define __EMULE_LOCAL_L1_TO_PTR_DEFINED
inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (l1_addr >= l1_base) {
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
    }
    return __emule_bridge_l1 + l1_addr;
}
#endif

// Bank mapping arrays — populated by emulated_program_runner, resolved at dlopen.
// Match firmware declarations from dataflow_api_common.h / firmware_common.h —
// sized by the same NUM_*_BANKS JIT defines so multi-extern type-mismatch
// errors don't fire when this header and dataflow_api_addrgen.h coexist.
extern uint16_t dram_bank_to_noc_xy[2][NUM_DRAM_BANKS];
extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];
extern uint16_t l1_bank_to_noc_xy[2][NUM_L1_BANKS];
extern int32_t bank_to_l1_offset[NUM_L1_BANKS];

// Per-core NOC coordinates (set per kernel thread by program runner).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];

// Per-core logical coordinates (for D2M get_absolute_logical_x/y).
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;

// Processor ID — substitutes RISC-V mhartid CSR in emulation.
extern thread_local uint8_t __processor_id;

// CSR emulation: NEO engine ID and TRISC core ID within engine.
extern thread_local uint8_t __emule_neo_id;
extern thread_local uint8_t __emule_trisc_id;

// Thread ID: number of compute/DM engines and this engine's index.
extern thread_local uint32_t __emule_num_threads;
extern thread_local uint32_t __emule_my_thread_id;

// tt_l1_ptr: type qualifier for L1 pointers. On real HW, this adds volatile.
// In emulation, L1 is normal host memory — the qualifier is empty.
#ifndef tt_l1_ptr
#define tt_l1_ptr
#endif

// noc_index / noc_mode mirror the firmware dataflow_api_common.h KERNEL_BUILD
// formula (noc_index = NOC_INDEX, noc_mode = NOC_MODE): the host emits these per
// kernel type (kernel.cpp::process_defines) — BRISC→NOC0, NCRISC→NOC1, mode
// DM_DEDICATED_NOC by default. emule's compute wrappers emit neither macro, so
// default them. See docs/noc-emulation.md §8.3.
#ifndef NOC_INDEX
#define NOC_INDEX 0
#endif
#ifndef NOC_MODE
#define NOC_MODE DM_DEDICATED_NOC
#endif
constexpr uint8_t noc_index = NOC_INDEX;
constexpr uint8_t noc_mode = NOC_MODE;

static inline uintptr_t get_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__rt_args[arg_idx]);
}

static inline uintptr_t get_common_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__common_rt_args[arg_idx]);
}

template<typename T = uint32_t>
inline T get_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    T val;
    std::memcpy(&val, &__rt_args[arg_idx], sizeof(T));
    return val;
}

template<typename T = uint32_t>
inline T get_common_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    T val;
    std::memcpy(&val, &__common_rt_args[arg_idx], sizeof(T));
    return val;
}

// Upstream tt::CBIndex (enum). Kernels that also include the upstream chain
// would otherwise hit a struct-vs-enum tag mismatch against an emule struct.
#include "hostdevcommon/kernel_structs.h"

// Routes ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp through its DM-kernel
// fallback path. Emule executes UNPACK/MATH/PACK serially in one thread, so
// half-sync optimization is moot.
#ifndef ENABLE_FP32_DEST_ACC
#  define ENABLE_FP32_DEST_ACC 0
#endif
#ifndef DST_SYNC_FULL
#  define DST_SYNC_FULL 1
#endif

// No-op assertion macro — kernel asserts reference hardware state not available in emulation.
#define ASSERT(...) ((void)0)

// Semaphore address helper — returns a uint32_t L1 address for the given
// semaphore ID.  Defined here so both compute and dataflow kernels can use it.
// EMULE_SEM_BASE is passed as a JIT compiler define by emulated_program_runner
// (computed from the HAL's KERNEL_CONFIG base + sem_offset).  When not defined
// (e.g. compute-only TU that doesn't pull in dataflow_api.h), get_semaphore is
// elided — kernels that need semaphores include dataflow_api.h, which hard-
// errors if EMULE_SEM_BASE is missing.
#ifndef EMULE_SEM_ALIGN
#define EMULE_SEM_ALIGN 16
#endif
#if defined(EMULE_SEM_BASE)
#ifndef __EMULE_GET_SEMAPHORE_DEFINED
#define __EMULE_GET_SEMAPHORE_DEFINED
inline uint32_t get_semaphore(uint32_t semaphore_id) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    return l1_base + EMULE_SEM_BASE + semaphore_id * EMULE_SEM_ALIGN;
}
#endif
#endif

// ---- ThreadId enum + mailbox stubs (used by both compute and dataflow kernels) ----
namespace ckernel {
enum ThreadId {
    BriscThreadId  = 0,
    UnpackThreadId = 1,
    MathThreadId   = 2,
    PackThreadId   = 3
};
inline uint32_t mailbox_read(uint8_t /*thread*/) { return 1; }
inline void mailbox_write(uint8_t /*thread*/, uint32_t /*data*/) {}
} // namespace ckernel
