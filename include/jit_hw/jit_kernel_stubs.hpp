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

#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"
#include "api/debug/dprint.h"
#include "api/debug/device_print.h"
#include "dev_mem_map.h"
#include "emule_cb_state.h"
#include "emule_dfb_state.h"
#include "tools/profiler/kernel_profiler.hpp"

// `firmware_common.h` provides invalidate_l1_cache / flush_l1_cache /
// WAYPOINT — needed by dataflow kernels via `experimental::semaphore_wait`.
// Pull it in here so every JIT kernel sees it. The heavier LLK surface
// (DataCopyType, UnpackToDestEn, llk_math_* templates, __llk_pack_* /
// __llk_unpack_* state, experimental::pack_untilize_block) is scoped to
// the compute entry point `api/compute/compute_kernel_hw_startup.h` —
// pulling it into every TU corrupts the SFPU INT32 unary tile-data path.
#include "internal/firmware_common.h"

#include <vector>
#include <cstdint>
#include <cstring>

// Forward declarations matching kernel_runner.cpp definitions.
// The main executable exports these with -rdynamic; the JIT .so resolves them
// at dlopen() time.
namespace tt_emule { class Core; class Device; }
extern thread_local std::vector<uint32_t> __rt_args;
extern thread_local std::vector<uint32_t> __common_rt_args;
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
        fprintf(stderr,
                "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
                l1_addr, __emule_sem_l1_range_start, __emule_sem_l1_range_end);
        abort();
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
                if (!(write_dist < reserved) && !(read_dist < waited)) {
                    fprintf(stderr,
                            "[ASAN ERROR] CB Boundary Violation: Attempted to access CB %u at offset 0x%x "
                            "(byte %u of %u, page %u of %u). "
                            "Write window: write_idx=%u, %u page(s) reserved. "
                            "Read window: read_idx=%u, %u page(s) waited.\n",
                            cb_id, l1_addr, l1_addr - cb_start, cb_size,
                            access_page, cb.num_pages,
                            cb.write_idx, reserved,
                            cb.read_idx, waited);
                    abort();
                }
            }
            uint32_t l1_base_cb = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
            if (l1_addr >= l1_base_cb) {
                return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
            }
            return __emule_bridge_l1 + l1_addr;
        }
    }
    // Normalize to a buffer-relative offset before comparing against the live
    // tensor / padding ranges (which are relative, from buffer.address()).
    // Sharded, CB/DFB, l1_alloc and >2MB bank accesses reach here as ABSOLUTE
    // bridge-based addresses; a raw absolute value can never fall in a relative
    // range -> false OOB. Mirrors the pointer-return path's absolute->relative
    // conversion (dataflow_api.h uses __emule_addr_to_offset).
    uint32_t l1_off_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    uint32_t l1_off = (l1_addr >= l1_off_base) ? (l1_addr - l1_off_base) : l1_addr;
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
            fprintf(stderr,
                    "[ASAN ERROR] Out-of-Bounds Write: Attempted to access address 0x%x which is not part of any allocated tensor\n",
                    l1_off);
            abort();
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
                fprintf(stderr,
                        "[ASAN ERROR] Tensor Padding Violation: Attempted to write to a padded memory region at address 0x%x (logical_end=0x%x, physical_end=0x%x)\n",
                        l1_off, logical_end, physical_end);
                abort();
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
// Match firmware declarations from dataflow_api_common.h / firmware_common.h.
extern uint16_t dram_bank_to_noc_xy[2][32];
extern int32_t bank_to_dram_offset[32];
extern uint16_t l1_bank_to_noc_xy[2][32];
extern int32_t bank_to_l1_offset[32];

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

// get_arg_addr(idx) — mirrors tt-metal's rta_l1_base-based implementation.
// Returns a pointer to the idx-th runtime arg (held in __rt_args).
// Signature matches real firmware: int param, uintptr_t return.
static inline uintptr_t get_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__rt_args[arg_idx]);
}

// get_common_arg_addr(idx) — pointer to common runtime arg.
static inline uintptr_t get_common_arg_addr(int arg_idx) {
    return reinterpret_cast<uintptr_t>(&__common_rt_args[arg_idx]);
}

// Per-core and common runtime argument value access.
// Signatures match real firmware: int param, template return.
template<typename T = uint32_t>
inline T get_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    if (arg_idx < 0 || static_cast<size_t>(arg_idx) >= __rt_args.size()) {
        fprintf(stderr, "EMULE BUG: get_arg_val(%d) out of bounds (size=%zu)\n",
                arg_idx, __rt_args.size());
        std::abort();
    }
    T val;
    std::memcpy(&val, &__rt_args[arg_idx], sizeof(T));
    return val;
}

template<typename T = uint32_t>
inline T get_common_arg_val(int arg_idx) {
    static_assert(sizeof(T) <= sizeof(uint32_t));
    if (arg_idx < 0 || static_cast<size_t>(arg_idx) >= __common_rt_args.size()) {
        fprintf(stderr, "EMULE BUG: get_common_arg_val(%d) out of bounds (size=%zu)\n",
                arg_idx, __common_rt_args.size());
        std::abort();
    }
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
// EMULE_SEM_BASE should be passed as a JIT compiler define (e.g. -DEMULE_SEM_BASE=0xFFE00).
// If not defined here, dataflow_api.h provides the default (0xFFE00).
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
