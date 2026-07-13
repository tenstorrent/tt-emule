// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CBInterface + get_local_cb_interface — mirror silicon's per-RISC CB register
// file so kernels written against the real device API compile and run unchanged
// under emule. `LocalCBInterface` (the struct) and the authoritative per-RISC
// storage `__emule_self->local_cb` live in jit_hw/internal/emule_cb_ptr.h.
//
// AUTHORITATIVE, NOT a snapshot (#139). get_local_cb_interface returns a
// reference into the per-RISC storage that get_write_ptr/get_read_ptr (cb_api.h)
// and the compute pack/unpack helpers (compute/common.h) also read. So:
//   - the fifo_{rd,wr}_ptr it returns is the SAME per-RISC pointer all accessors
//     use (no divergence between the CB API and this interface), and
//   - a kernel that WRITES fifo_{wr,rd}_ptr through the returned reference
//     (silicon-style override_cb_rd_ptr) takes effect — the next cb_push_back /
//     get_*_ptr advances/reads from the written value (pointer write-back).
// Scope of write-back: ONLY fifo_{wr,rd}_ptr. The address/advance helpers in
// emule_cb_ptr.h derive base/page_size/num_pages from the shared CBSyncState
// (__emule_self->cbs[cb_id]), so writing the view's geometry fields
// (fifo_page_size/fifo_size/fifo_limit/fifo_num_pages) does NOT affect
// wrap/advance — geometry cannot be reconfigured through this reference today.
// Per call only the live `tiles_received` (the shared occupied semaphore) is
// refreshed; the fifo_{wr,rd}_ptr persist as authoritative per-RISC state.
// First-touch seating at the shard base is handled by __emule_cb_view_init.

#include <atomic>
#include <cstdint>

#include "jit_hw/internal/emule_thread_ctx.h"
#include "jit_hw/internal/emule_cb_ptr.h"  // LocalCBInterface, __emule_self->local_cb, view helpers

struct CBInterface {
    union {
        LocalCBInterface local_cb_interface;
    };
};

// CB tile counter pointers. Silicon returns pointers into L1-resident
// `pages_received` / `pages_acked` counters (one cacheline per CB). an upstream kernel's
// cb_scratch_reset kernel writes 0 to these at init. Emule's canonical CB
// sync uses `CBSyncState.occupied` (host-side atomic) rather than L1 counters —
// these pointer accessors give kernels a place to write that doesn't affect
// emule's real CB state. Thread_local backing storage so multiple CBs
// don't alias.
inline volatile uint32_t* get_cb_tiles_received_ptr(uint32_t cb_id) {
    thread_local volatile uint32_t s_received[NUM_CIRCULAR_BUFFERS]{};
    return &s_received[cb_id];
}
inline volatile uint32_t* get_cb_tiles_acked_ptr(uint32_t cb_id) {
    thread_local volatile uint32_t s_acked[NUM_CIRCULAR_BUFFERS]{};
    return &s_acked[cb_id];
}

#ifndef __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#define __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
inline LocalCBInterface& get_local_cb_interface(uint32_t cb_id) {
    // Return a reference into the per-RISC register file shared with all CB
    // pointer accessors — NOT a recomputed snapshot. Reads reflect this RISC's
    // own pointer; writes through the reference persist (write-back).
    auto& view = __emule_self->local_cb[cb_id];
    if (__emule_self->cbs == nullptr) {
        return view;
    }
    // First touch seats fifo_{rd,wr}_ptr at the shard base + fills geometry
    // (16-byte units) from the shared CBSyncState; no-op once seated, so any
    // kernel-written pointer/geometry persists.
    __emule_cb_view_init(cb_id);
    // Refresh only the live semaphore count; pointer + geometry are authoritative.
    view.tiles_received = static_cast<uint16_t>(
        __emule_self->cbs[cb_id].occupied.load(std::memory_order_relaxed));
    view.tiles_acked = 0;
    return view;
}
#endif
