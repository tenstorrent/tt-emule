// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// CB operations for JIT-compiled kernels.
// Delegates to shared tt_emule::cb_sync_* functions — single source of truth.

#include "jit_hw/emule_cb_state.h"
#include "jit_hw/emule_dfb_state.h"
#include "jit_hw/api/compute/common_globals.h"
#include "jit_hw/emule_wait.h"
#include "jit_hw/internal/emule_cb_ptr.h"   // per-RISC CB pointers + cb_addr_shift
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Forward declarations for hang diagnostics (defined in jit_kernel_stubs.hpp).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;

// cb_addr_shift (16-byte fifo-pointer encoding) is defined in emule_cb_ptr.h.

// ---- Constexpr tile metadata arrays (populated by JIT defines) ----
// EMULE_TILE_SIZES is defined by the JIT compiler as a comma-separated list of
// 32 page sizes (one per CB index), matching the real device's unpack_tile_size[].

#ifdef EMULE_TILE_SIZES
constexpr uint16_t unpack_tile_size[32] = { EMULE_TILE_SIZES };
#else
constexpr uint16_t unpack_tile_size[32] = {};
#endif

// Per-CB data format arrays (tt::DataFormat enum values) — emule's single source of
// truth, the analog of the device's compile-time unpack_src_format[]/pack_dst_format[]
// (generated into chlkc_descriptors.h by genfiles.cpp::compute_data_formats). The two
// L1-side arrays are populated from EMULE_CB_DATA_FORMATS (a 32-value list the JIT
// compiler builds from each CB's CircularBufferImpl::data_format(idx)); for a given CB id
// both equal the L1 tile's format. DataFormat::Invalid (0xFF) marks unconfigured slots
// (mirrors the host's std::optional<DataFormat> empty state) — format-dispatch consumers
// fall back to the page_size heuristic for those. The two DST-side arrays are all-zero
// (Float32) stubs: emule's DST register file is always fp32, so they are immaterial to
// the emulated pack/unpack math.
#ifdef EMULE_CB_DATA_FORMATS
constexpr uint8_t unpack_src_format[32] = { EMULE_CB_DATA_FORMATS };
constexpr uint8_t pack_dst_format[32]   = { EMULE_CB_DATA_FORMATS };
#else
constexpr uint8_t unpack_src_format[32] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
constexpr uint8_t pack_dst_format[32] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};
#endif
constexpr uint8_t unpack_dst_format[32] = {};  // DST-side (fp32) stub
constexpr uint8_t pack_src_format[32]   = {};  // DST-side (fp32) stub

// Standard 32×32 tiles: 2 faces in each dimension
constexpr uint8_t unpack_tile_r_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
};
constexpr uint8_t unpack_tile_c_dim[32] = {
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
    32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
};
constexpr uint8_t unpack_num_faces_r_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
};
constexpr uint8_t unpack_num_faces_c_dim[32] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
};

// ---- Per-RISC CB pointers ----
// The per-thread read/write pointers (the single source of truth that fixes the
// #139 race) live in jit_hw/internal/emule_cb_ptr.h (__emule_local_cb). This
// file's cb_push_back/cb_pop_front advance them via __emule_cb_advance_wr/rd and
// get_write_ptr/get_read_ptr read them via __emule_cb_wr_addr/__emule_cb_rd_addr.
// CBSyncState here owns only the shared occupied semaphore.

// ---- Circular Buffer sync operations ----

// CB timeout: large matmuls (e.g. 2048x2048x2048 f32) can keep a thread busy
// for tens of seconds between CB interactions. Default 120s; override via env.
inline int __emule_cb_timeout_sec() {
    static int val = []() {
        const char* env = std::getenv("TT_EMULE_CB_TIMEOUT");
        return env ? std::atoi(env) : 120;
    }();
    return val;
}

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    if (n > cb.num_pages) {
        fprintf(stderr, "EMULE BUG: cb_reserve_back(cb_id=%u, n=%u) requests more than capacity "
                "(num_pages=%u, page_size=%u) [phys (%u,%u) logical (%u,%u)]\n",
                cb_id, n, cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
    // Lock-free fast path (safe for SPSC — only consumer decrements occupied)
    if ((cb.num_pages - cb.occupied.load(std::memory_order_acquire)) >= n) {
        __emule_pack_offset[cb_id] = 0;
        return;
    }
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!__emule_cv_wait(cb.space_cv, lk, __emule_cb_timeout_sec(),
            [&]{ return (cb.num_pages - cb.occupied.load(std::memory_order_relaxed)) >= n; })) {
        fprintf(stderr, "EMULE HANG: cb_reserve_back(cb_id=%u, n=%u) timed out after %ds "
                "(occupied=%u, num_pages=%u, page_size=%u) "
                "[phys (%u,%u) logical (%u,%u)] "
                "(set TT_EMULE_CB_TIMEOUT=<secs> to adjust)\n",
                cb_id, n, __emule_cb_timeout_sec(),
                cb.occupied.load(), cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
    // Reset PACK engine auto-advance offset for this new batch.
    __emule_pack_offset[cb_id] = 0;
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    // Advance this thread's own write pointer (mirrors the per-RISC write ptr on
    // silicon), then bump the shared occupied semaphore.
    __emule_cb_advance_wr(cb_id, n);
    tt_emule::cb_sync_push(__emule_cbs[cb_id], n);
    // Bridge CB→DFB: update tile counters so DM's dfb_wait_front sees compute's output.
    // cb.mu already released; now safe to acquire tc.mu (consistent lock ordering).
    if (__emule_dfbs && __emule_tc_array && __emule_dfbs[cb_id].active) {
        auto& iface = __emule_dfbs[cb_id];
        if (iface.broadcast_tc) {
            for (uint8_t i = 0; i < iface.num_tcs_to_rr; ++i) {
                auto& slot = iface.tc_slots[i];
                __emule_tc_array->inc_posted(slot.neo_id, slot.counter_id, n);
                slot.wr_ptr += static_cast<uint32_t>(n) * iface.stride_size;
                if (slot.wr_ptr >= slot.limit)
                    slot.wr_ptr = slot.base_addr + (slot.wr_ptr - slot.limit);
            }
        } else {
            auto& slot = iface.tc_slots[iface.tc_idx];
            __emule_tc_array->inc_posted(slot.neo_id, slot.counter_id, n);
            slot.wr_ptr += static_cast<uint32_t>(n) * iface.stride_size;
            if (slot.wr_ptr >= slot.limit)
                slot.wr_ptr = slot.base_addr + (slot.wr_ptr - slot.limit);
            iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
        }
        iface.wr_entry_idx += n;
    }
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    if (n > cb.num_pages) {
        fprintf(stderr, "EMULE BUG: cb_wait_front(cb_id=%u, n=%u) requests more than capacity "
                "(num_pages=%u, page_size=%u) [phys (%u,%u) logical (%u,%u)]\n",
                cb_id, n, cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
    // Lock-free fast path (safe for SPSC — only producer increments occupied)
    if (cb.occupied.load(std::memory_order_acquire) >= n) return;
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!__emule_cv_wait(cb.data_cv, lk, __emule_cb_timeout_sec(),
            [&]{ return cb.occupied.load(std::memory_order_relaxed) >= n; })) {
        fprintf(stderr, "EMULE HANG: cb_wait_front(cb_id=%u, n=%u) timed out after %ds "
                "(occupied=%u, num_pages=%u, page_size=%u) "
                "[phys (%u,%u) logical (%u,%u)] "
                "(set TT_EMULE_CB_TIMEOUT=<secs> to adjust)\n",
                cb_id, n, __emule_cb_timeout_sec(),
                cb.occupied.load(), cb.num_pages, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    // Advance this thread's own read pointer (mirrors the per-RISC read ptr on
    // silicon), then drop the shared occupied semaphore.
    __emule_cb_advance_rd(cb_id, n);
    tt_emule::cb_sync_pop(__emule_cbs[cb_id], n);
    // Bridge CB→DFB: update tile counter acked so DM's dfb_reserve_back sees freed space.
    if (__emule_dfbs && __emule_tc_array && __emule_dfbs[cb_id].active) {
        auto& iface = __emule_dfbs[cb_id];
        auto& slot = iface.tc_slots[iface.tc_idx];
        __emule_tc_array->inc_acked(slot.neo_id, slot.counter_id, n);
        slot.rd_ptr += static_cast<uint32_t>(n) * iface.stride_size;
        if (slot.rd_ptr >= slot.limit)
            slot.rd_ptr = slot.base_addr + (slot.rd_ptr - slot.limit);
        iface.rd_entry_idx += n;
        iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
    }
}

// ---- int32_t overloads (D2M int32 support emits int32_t tile counts) ----
inline void cb_reserve_back(uint32_t cb_id, int32_t n) { cb_reserve_back(cb_id, static_cast<uint32_t>(n)); }
inline void cb_push_back(uint32_t cb_id, int32_t n)    { cb_push_back(cb_id, static_cast<uint32_t>(n)); }
inline void cb_wait_front(uint32_t cb_id, int32_t n)   { cb_wait_front(cb_id, static_cast<uint32_t>(n)); }
inline void cb_pop_front(uint32_t cb_id, int32_t n)    { cb_pop_front(cb_id, static_cast<uint32_t>(n)); }

// ---- Pointer accessors ----

// Return uint32_t (truncated host pointer). CB memory is mmap'd below 4 GB.
// Reads the calling thread's own per-RISC write/read pointer (emule_cb_ptr.h),
// so concurrent reader and writer threads each see their own view — matching
// silicon's per-RISC write/read pointer registers (the #139 fix).
inline uint32_t get_write_ptr(uint32_t cb_id) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_cb_wr_addr(cb_id)));
}

inline uint32_t get_read_ptr(uint32_t cb_id) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_cb_rd_addr(cb_id)));
}

// get_tile_size — return page size (bytes) for a CB.
// Uses constexpr unpack_tile_size[] array (populated by JIT EMULE_TILE_SIZES define).
constexpr inline uint32_t get_tile_size(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_tile_size[cb_id]);
}

// get_tile_hw — return tile height × width (elements).
constexpr inline uint32_t get_tile_hw(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_tile_r_dim[cb_id]) *
           static_cast<uint32_t>(unpack_tile_c_dim[cb_id]);
}

// get_tile_num_faces — return number of faces per tile.
constexpr inline uint32_t get_tile_num_faces(uint32_t cb_id) {
    return static_cast<uint32_t>(unpack_num_faces_r_dim[cb_id]) *
           static_cast<uint32_t>(unpack_num_faces_c_dim[cb_id]);
}

// get_dataformat — return the CB's real data format, faithful to the device's
// dataflow_api.h: `get_dataformat(operand) = unpack_src_format[operand]`. Falls back to
// the page_size heuristic (bf16 ≤ 2048B, else Float32) when the format is unset (Invalid).
constexpr inline DataFormat get_dataformat(uint32_t cb_id) {
    const uint8_t fmt = unpack_src_format[cb_id];
    if (fmt != static_cast<uint8_t>(DataFormat::Invalid)) {
        return static_cast<DataFormat>(fmt);
    }
    return unpack_tile_size[cb_id] > 2048 ? DataFormat::Float32 : DataFormat::Float16_b;
}
