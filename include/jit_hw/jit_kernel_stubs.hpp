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

// NOC mode constants — silicon firmware sets these per-RISC build. upstream kernels
// has static_assert(noc_mode == DM_DYNAMIC_NOC) in several ops, so emule
// reports dynamic mode. Universal because upstream code references them from both
// compute and dataflow paths.
#ifndef DM_DEDICATED_NOC
#define DM_DEDICATED_NOC 0
#endif
#ifndef DM_DYNAMIC_NOC
#define DM_DYNAMIC_NOC 1
#endif
inline constexpr int noc_mode = DM_DYNAMIC_NOC;

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
#include "emule_asan.h"
#include "tools/profiler/kernel_profiler.hpp"

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
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {}
inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t) { compute_kernel_hw_startup(a, b); }
#endif

// Forward declarations matching kernel_runner.cpp definitions.
// The main executable exports these with -rdynamic; the JIT .so resolves them
// at dlopen() time.
namespace tt_emule { class Core; class Device; }
// Per-thread L1 pointers set by the runner's kernel-launch lambda.
// nullptr = no args for this RISC.
extern thread_local uint32_t* __rt_args;
extern thread_local uint32_t* __common_rt_args;
extern thread_local tt_emule::Core*       __core;
extern thread_local tt_emule::Device*     __device;

// C-linkage bridge functions — resolve at dlopen time to the host process's
// implementations, avoiding ABI mismatch with Device layout.
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr);

// L1 bridge pointer — host pointer to L1 memory start.
// Used to convert between uint32_t L1 "addresses" (really: truncated pointers
// from mmap'd-below-4GB L1) and real host pointers.
extern thread_local uint8_t* __emule_bridge_l1;

// Sanitizer thread-locals — see ASAN.md for the full handshake.
extern thread_local uint32_t __emule_sem_l1_range_start;
extern thread_local uint32_t __emule_sem_l1_range_end;
extern thread_local uint32_t __emule_pending_noc_reads;
extern thread_local uint32_t __emule_l1_unreserved_base;
extern thread_local const uint64_t* __emule_l1_tensor_ranges;
extern thread_local uint32_t __emule_l1_tensor_ranges_count;
extern thread_local const uint64_t* __emule_l1_padding_ranges;
extern thread_local uint32_t __emule_l1_padding_ranges_count;
extern thread_local uint64_t* __emule_l1_resolved_ranges;
extern thread_local uint32_t* __emule_l1_resolved_ranges_count;
extern thread_local uint32_t __emule_l1_resolved_ranges_capacity;
extern thread_local uint32_t __emule_cb_reserved_pages[32];
extern thread_local uint32_t __emule_cb_waited_pages[32];
extern thread_local bool __emule_cb_boundary_strict;

// L1 access chokepoint — all kernel sanitizers run here. See ASAN.md.
#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED
#define __EMULE_LOCAL_L1_TO_PTR_DEFINED
inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr) {
    if (__emule_sem_l1_range_end > 0 &&
        l1_addr >= __emule_sem_l1_range_start && l1_addr < __emule_sem_l1_range_end) {
        __emule_asan_panic(
                "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
                l1_addr, __emule_sem_l1_range_start, __emule_sem_l1_range_end);
    }
    // CB ranges must be matched before OOB — CB memory isn't registered in
    // LiveL1Ranges and would otherwise look out-of-bounds.
    if (__emule_cbs != nullptr) {
        for (uint32_t cb_id = 0; cb_id < 32; ++cb_id) {
            auto& cb = __emule_cbs[cb_id];
            if (cb.num_pages == 0) continue;
            uint32_t cb_start = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(cb.base));
            uint32_t cb_size = cb.num_pages * cb.page_size;
            if (l1_addr < cb_start || l1_addr >= cb_start + cb_size) continue;
            if (__emule_cb_boundary_strict) {
                uint32_t access_page = (l1_addr - cb_start) / cb.page_size;
                uint32_t write_dist = (access_page + cb.num_pages - cb.write_idx) % cb.num_pages;
                uint32_t read_dist  = (access_page + cb.num_pages - cb.read_idx)  % cb.num_pages;
                uint32_t reserved = __emule_cb_reserved_pages[cb_id];
                uint32_t waited   = __emule_cb_waited_pages[cb_id];
                // Only meaningful when the kernel holds an ACTIVE reservation/wait
                // window. reserved==0 && waited==0 means raw get_write_ptr /
                // get_read_ptr addressing (globally-allocated/sharded CBs, single-
                // buffered scratch, output CBs written then DMA'd) — there is no
                // window to be "outside" of, so it is not a boundary violation.
                // (A genuine write past the CB's allocated region is still caught
                // downstream by the OOB-tensor check.)
                if ((reserved > 0 || waited > 0) && !(write_dist < reserved) && !(read_dist < waited)) {
                    __emule_asan_panic(
                            "[ASAN ERROR] CB Boundary Violation: Attempted to access CB %u at offset 0x%x "
                            "(byte %u of %u, page %u of %u). "
                            "Write window: write_idx=%u, %u page(s) reserved. "
                            "Read window: read_idx=%u, %u page(s) waited.\n",
                            cb_id, l1_addr, l1_addr - cb_start, cb_size,
                            access_page, cb.num_pages,
                            cb.write_idx, reserved,
                            cb.read_idx, waited);
                }
            }
            uint32_t l1_base_cb = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
            if (l1_addr >= l1_base_cb) {
                return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
            }
            return __emule_bridge_l1 + l1_addr;
        }
    }
    // Reduce to the within-slot L1 offset by masking the low 21 bits (2 MB worker
    // slot). The high bits just encode which core / absolute bridge base the
    // address came through, while the live tensor/padding ranges are stored as
    // buffer-relative offsets (buffer.address()). Masking means two addresses that
    // share their low 21 bits map to the same offset regardless of core — so a
    // legitimate access isn't flagged just because its high bits differ from this
    // thread's base. (Per-core L1 bases are 2 MB-aligned, so for an in-slot address
    // the mask equals base-subtraction but is robust for any high bits.)
    //
    // KNOWN LIMITATION (accepted, see review): because the ranges are offset-based
    // and a sharded tensor occupies the *same* offset on each of its shard cores,
    // a write to that offset on a core where the tensor is NOT sharded passes this
    // check (a false negative). Catching it would require per-core shard-placement
    // tracking, not just offsets.
    uint32_t l1_off = l1_addr & 0x1FFFFF;  // SLOT_MASK = 2 MB - 1
    if (__emule_l1_tensor_ranges != nullptr && l1_off >= __emule_l1_unreserved_base) {
        bool in_tensor = false;
        uint64_t matched_packed = 0;
        for (uint32_t i = 0; i < __emule_l1_tensor_ranges_count; ++i) {
            uint64_t packed = __emule_l1_tensor_ranges[i];
            uint32_t r_start = static_cast<uint32_t>(packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(packed);
            if (l1_off >= r_start && l1_off < r_end) {
                in_tensor = true;
                matched_packed = packed;
                break;
            }
        }
        if (!in_tensor) {
            __emule_asan_panic(
                    "[ASAN ERROR] Out-of-Bounds Write: Attempted to access address 0x%x which is not part of any allocated tensor\n",
                    l1_off);
        }
        if (__emule_l1_resolved_ranges != nullptr &&
            __emule_l1_resolved_ranges_count != nullptr) {
            uint32_t cur = *__emule_l1_resolved_ranges_count;
            bool already = false;
            for (uint32_t i = 0; i < cur; ++i) {
                if (__emule_l1_resolved_ranges[i] == matched_packed) {
                    already = true;
                    break;
                }
            }
            if (!already && cur < __emule_l1_resolved_ranges_capacity) {
                __emule_l1_resolved_ranges[cur] = matched_packed;
                *__emule_l1_resolved_ranges_count = cur + 1;
            }
        }
    }
    if (__emule_l1_padding_ranges != nullptr) {
        for (uint32_t i = 0; i < __emule_l1_padding_ranges_count; ++i) {
            uint64_t packed = __emule_l1_padding_ranges[i];
            uint32_t logical_end = static_cast<uint32_t>(packed >> 32);
            uint32_t physical_end = static_cast<uint32_t>(packed);
            if (l1_off >= logical_end && l1_off < physical_end) {
                __emule_asan_panic(
                        "[ASAN ERROR] Tensor Padding Violation: Attempted to write to a padded memory region at address 0x%x (logical_end=0x%x, physical_end=0x%x)\n",
                        l1_off, logical_end, physical_end);
            }
        }
    }
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

// NOC index — always 0 for emulation (real firmware sets this per core).
constexpr uint8_t noc_index = 0;

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
