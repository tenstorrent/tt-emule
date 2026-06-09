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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// Forward declarations for hang diagnostics (defined in jit_kernel_stubs.hpp).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;
extern thread_local uint32_t __emule_pending_noc_reads;

// ---- cb_addr_shift ----
// Silicon convention: addresses stored in 16-byte units (matches fifo_rd_ptr
// encoding in LocalCBInterface). the cb_reconfig kernel reads
// cb_config[] entries and right-shifts by this constant to get the encoded
// pointer form.
inline constexpr uint32_t cb_addr_shift = 4;

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

// Master ASAN switch (TT_METAL_EMULE_ASAN). Re-read every call; see ASAN.md.
inline bool __emule_asan_enabled() {
    const char* v = std::getenv("TT_METAL_EMULE_ASAN");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

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
    // Always on: gating this would deadlock the CV wait below. See ASAN.md.
    if (n > cb.num_pages) {
        fprintf(stderr,
                "[ASAN ERROR] CB Reservation Overflow: CB %u has %u total pages, "
                "but kernel requested to reserve %u pages. This would hang on silicon! "
                "(page_size=%u) [phys (%u,%u) logical (%u,%u)]\n",
                cb_id, cb.num_pages, n, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
        std::abort();
    }
    // Lock-free fast path (safe for SPSC — only consumer decrements occupied).
    if ((cb.num_pages - cb.occupied.load(std::memory_order_acquire)) >= n) {
        __emule_pack_offset[cb_id] = 0;
        __emule_cb_reserved_pages[cb_id] += n;
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
    __emule_cb_reserved_pages[cb_id] += n;
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    // Shrink the reserved window before the FIFO advance — keeps any concurrent
    // boundary check consistent.
    if (n <= __emule_cb_reserved_pages[cb_id]) {
        __emule_cb_reserved_pages[cb_id] -= n;
    } else {
        __emule_cb_reserved_pages[cb_id] = 0;
    }
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
    // Lock-free fast path (safe for SPSC — only producer increments occupied).
    if (cb.occupied.load(std::memory_order_acquire) >= n) {
        // max(), not += — overlapping waits before a pop don't grow the
        // consumable region.
        if (n > __emule_cb_waited_pages[cb_id]) {
            __emule_cb_waited_pages[cb_id] = n;
        }
        return;
    }
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
    if (n > __emule_cb_waited_pages[cb_id]) {
        __emule_cb_waited_pages[cb_id] = n;
    }
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    // All outstanding NoC reads must be barriered before a pop frees the page:
    // popping releases the page for the producer to refill, and a read still
    // landing into it would race the refill (consumer reads stale/torn data on
    // silicon). cb_push_back has no such requirement — only writes precede a
    // push, so an unbarriered read there is harmless.
    if (__emule_asan_enabled() && __emule_pending_noc_reads > 0) {
        fprintf(stderr,
                "[ASAN ERROR] Race Condition: cb_pop_front(cb_id=%u) called while a NoC read is still pending "
                "(%u outstanding) — missing noc_async_read_barrier()\n",
                cb_id, __emule_pending_noc_reads);
        abort();
    }
    // Saturate at 0; popping more than waited is suspect but mustn't underflow.
    if (n <= __emule_cb_waited_pages[cb_id]) {
        __emule_cb_waited_pages[cb_id] -= n;
    } else {
        __emule_cb_waited_pages[cb_id] = 0;
    }
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
inline uint32_t get_write_ptr(uint32_t cb_id) {
    uint8_t* ptr = tt_emule::cb_sync_write_ptr(__emule_cbs[cb_id]);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline uint32_t get_read_ptr(uint32_t cb_id) {
    uint8_t* ptr = tt_emule::cb_sync_read_ptr(__emule_cbs[cb_id]);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
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
