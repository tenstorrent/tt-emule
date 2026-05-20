// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// CB operations for JIT-compiled kernels.
// Delegates to shared tt_emule::cb_sync_* functions — single source of truth.

#include "jit_hw/emule_cb_state.h"
#include "jit_hw/emule_dfb_state.h"
#include "jit_hw/api/compute/common_globals.h"
#include <chrono>
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

// ---- Constexpr tile metadata arrays (populated by JIT defines) ----
// EMULE_TILE_SIZES is defined by the JIT compiler as a comma-separated list of
// 32 page sizes (one per CB index), matching the real device's unpack_tile_size[].

#ifdef EMULE_TILE_SIZES
constexpr uint16_t unpack_tile_size[32] = { EMULE_TILE_SIZES };
#else
constexpr uint16_t unpack_tile_size[32] = {};
#endif

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
        // CB-boundary sanitizer: record how many pages are now reservable
        // starting at the current cb.write_idx. cb_push_back decrements this.
        __emule_cb_reserved_pages[cb_id] += n;
        return;
    }
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!cb.space_cv.wait_for(lk, std::chrono::seconds(__emule_cb_timeout_sec()),
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
    if (__emule_pending_noc_reads > 0) {
        fprintf(stderr,
                "[ASAN ERROR] Race Condition: cb_push_back(cb_id=%u) called while a NoC read is still pending "
                "(%u outstanding) — missing noc_async_read_barrier()\n",
                cb_id, __emule_pending_noc_reads);
        abort();
    }
    // CB-boundary sanitizer: shrink the reserved window before the FIFO
    // advance so any concurrent boundary check sees a consistent view.
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
    // Lock-free fast path (safe for SPSC — only producer increments occupied)
    if (cb.occupied.load(std::memory_order_acquire) >= n) {
        // CB-boundary sanitizer: widen the read window to the largest count
        // the kernel has waited on since the last pop. max(), not +=, because
        // overlapping waits before a pop don't grow the consumable region.
        if (n > __emule_cb_waited_pages[cb_id]) {
            __emule_cb_waited_pages[cb_id] = n;
        }
        return;
    }
    std::unique_lock<std::mutex> lk(cb.mu);
    if (!cb.data_cv.wait_for(lk, std::chrono::seconds(__emule_cb_timeout_sec()),
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
    // CB-boundary sanitizer: shrink the read window before the FIFO advance
    // so any concurrent boundary check sees a consistent view. Saturate at 0
    // — a kernel that pops more than it waited on is suspect but we don't
    // want to underflow the sanitizer state.
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

// ---- Dirty-CB sanitizer (called by emulated_program_runner) ----
//
// On silicon, a kernel that finishes with pages still on a CB (push count >
// pop count, i.e. occupied > 0) leaves the CB pointers offset for the next
// program launch — which then immediately back-pressures on cb_reserve_back.
// The emulator detects this by walking __emule_cbs at kernel exit and
// aborting on any leftover pages, attributing the leak to the kernel that
// just finished. The runner also performs a final whole-program sweep in
// case multi-kernel programs cancel out per-kernel.
inline void __emule_check_kernel_cb_dirty(uint32_t lx, uint32_t ly, uint8_t processor_id) {
    if (__emule_cbs == nullptr) {
        return;
    }
    for (uint32_t cb_id = 0; cb_id < 32; ++cb_id) {
        auto& cb = __emule_cbs[cb_id];
        if (cb.num_pages == 0) {
            continue;  // CB not configured on this core
        }
        uint32_t occupied = cb.occupied.load(std::memory_order_acquire);
        if (occupied > 0) {
            fprintf(stderr,
                    "[ASAN ERROR] Dirty CB Detected: Core (%u, %u) CB %u was not flushed! "
                    "Kernel (processor %u) ended with %u/%u pages still on the CB "
                    "(push > pop) — this back-pressures the next program launch on silicon.\n",
                    lx, ly, cb_id, processor_id, occupied, cb.num_pages);
            std::abort();
        }
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

// get_dataformat — infer DataFormat from tile size.
// bf16 = 2048 bytes (Float16_b), 32-bit = 4096 bytes (Float32), else Float16_b.
constexpr inline DataFormat get_dataformat(uint32_t cb_id) {
    uint32_t sz = unpack_tile_size[cb_id];
    if (sz > 2048) return DataFormat::Float32;
    return DataFormat::Float16_b;
}
