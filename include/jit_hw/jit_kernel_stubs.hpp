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

#include <cstdint>
#include <cstring>

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

// Tensor-padding membership test. Tensor padding is 2-D, not a single trailing
// band (see emule_live_ranges.hpp): a tensor with a logical extent smaller than
// its padded extent has pad bytes at the right edge of every data row AND in the
// trailing rows — an "L" shape. `desc` points at this buffer's 4 packed descriptor
// words (w0=start|physical_end, w1=layout|elem_size, w2=logical_rows|logical_cols,
// w3=padded_cols|reserved). Returns true if within-slot offset `l1_off` lands in
// the padding region. Closed-form modulo math, O(1) — no per-row range list.
// KEEP IN SYNC with the identical copy in dataflow_api.h.
#ifndef __EMULE_OFFSET_IN_PADDING_DEFINED
#define __EMULE_OFFSET_IN_PADDING_DEFINED
inline bool __emule_offset_in_padding(const uint64_t* desc, uint32_t l1_off) {
    uint32_t start = static_cast<uint32_t>(desc[0] >> 32);
    uint32_t physical_end = static_cast<uint32_t>(desc[0]);
    if (l1_off < start || l1_off >= physical_end) {
        return false;  // not inside the buffer this descriptor covers
    }
    uint32_t layout = static_cast<uint32_t>(desc[1] >> 32);
    uint32_t elem_size = static_cast<uint32_t>(desc[1]);
    uint32_t logical_rows = static_cast<uint32_t>(desc[2] >> 32);
    uint32_t logical_cols = static_cast<uint32_t>(desc[2]);
    uint32_t padded_cols = static_cast<uint32_t>(desc[3] >> 32);
    uint32_t padded_page_rows = static_cast<uint32_t>(desc[3]);  // per-page row period; 0 = no paging
    if (elem_size == 0 || padded_cols == 0) {
        return false;  // malformed descriptor — never fire
    }
    uint32_t elem = (l1_off - start) / elem_size;  // element index in storage order
    uint32_t row, col;
    if (layout == 0) {
        // Row-major: padded_cols-wide physical rows.
        row = elem / padded_cols;
        col = elem % padded_cols;
    } else {
        // Tiled: 32x32 tiles in row-major tile order; each tile is 4 16x16 nfaces
        // (face 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right),
        // row-major within a face. This inverts nfaces.h's rowmajor_to_nfaces.
        const uint32_t TILE = 32, FACE = 16, FACE_HW = 256, TILE_HW = 1024;
        uint32_t tiles_per_row = padded_cols / TILE;
        if (tiles_per_row == 0) {
            return false;
        }
        uint32_t tile_idx = elem / TILE_HW;
        uint32_t in_tile = elem % TILE_HW;  // == nfaces offset within this tile
        uint32_t tile_r = tile_idx / tiles_per_row;
        uint32_t tile_c = tile_idx % tiles_per_row;
        uint32_t face = in_tile / FACE_HW;  // 0..3
        uint32_t in_face = in_tile % FACE_HW;
        uint32_t fr = in_face / FACE;
        uint32_t fc = in_face % FACE;
        row = tile_r * TILE + (face / 2) * FACE + fr;
        col = tile_c * TILE + (face % 2) * FACE + fc;
    }
    // N-D / batched: the buffer stacks G pages of height padded_page_rows, each with
    // the same per-page logical height (logical_rows). Reset the row test per page so
    // page p's data rows (global rows p*padded_page_rows .. +logical_rows) aren't
    // mistaken for padding. Columns don't stack, so col needs no period.
    uint32_t row_in_page = (padded_page_rows != 0) ? (row % padded_page_rows) : row;
    return (row_in_page >= logical_rows) || (col >= logical_cols);
}
#endif

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
                // Only meaningful when the kernel holds an ACTIVE reservation/wait
                // window. reserved==0 && waited==0 means raw get_write_ptr /
                // get_read_ptr addressing (globally-allocated/sharded CBs, single-
                // buffered scratch, output CBs written then DMA'd) — there is no
                // window to be "outside" of, so it is not a boundary violation.
                // (A genuine write past the CB's allocated region is still caught
                // downstream by the OOB-tensor check.)
                if ((reserved > 0 || waited > 0) && !(write_dist < reserved) && !(read_dist < waited)) {
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
        // 4 packed words per descriptor (see emule_live_ranges.cpp); count is the
        // number of descriptors.
        for (uint32_t i = 0; i < __emule_l1_padding_ranges_count; ++i) {
            const uint64_t* desc = __emule_l1_padding_ranges + i * 4;
            if (__emule_offset_in_padding(desc, l1_off)) {
                fprintf(stderr,
                        "[ASAN ERROR] Tensor Padding Violation: Attempted to write to a padded memory region at address 0x%x (buffer base 0x%x)\n",
                        l1_off, static_cast<uint32_t>(desc[0] >> 32));
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
