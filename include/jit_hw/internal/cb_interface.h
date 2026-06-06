// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// LocalCBInterface / CBInterface — mirror silicon's struct shape from
// the upstream circular-buffer interface so kernels written
// against the real device API compile and run unchanged under emule.
//
// Field semantics:
// - fifo_size      : total CB capacity, in bytes (page_size * num_pages).
// - fifo_limit     : end-of-CB 16-byte index (host pointer + size, >> 4).
// - fifo_page_size : bytes per page.
// - fifo_num_pages : page count.
// - fifo_rd_ptr    : current read position as a 16-byte-encoded host address
//                    (kernel does `<< 4` to reconstitute a byte pointer).
// - fifo_wr_ptr    : current write position, same encoding as fifo_rd_ptr.
// - fifo_wr_tile_ptr : silicon tracks separately; emule mirrors fifo_wr_ptr.
// - tiles_acked / tiles_received : silicon producer/consumer counters; emule
//                    exposes occupied as tiles_received, tiles_acked = 0.
//
// CAVEAT — snapshot semantics. Fields are recomputed from __emule_cbs[cb_id]
// (the canonical CB sync state) on every call. Mutation through the returned
// reference does NOT persist: re-reads see fresh values from __emule_cbs.
// Emule's cb_push_back/cb_pop_front (api/cb_api.h) advances __emule_cbs
// directly, not through this interface, so kernels using emule's CB API see
// consistent state. Silicon-style code that mutates fifo_wr_ptr via this
// reference is not currently supported — if a future kernel requires it,
// switch to lazy-init-once + write-back semantics here.

#include <atomic>
#include <cstdint>

#include "jit_hw/emule_cb_state.h"
#include "tt-metalium/circular_buffer_constants.h"

struct LocalCBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;  // range is inclusive of the limit
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    uint32_t fifo_wr_tile_ptr;
};

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
    thread_local LocalCBInterface s_cbs[NUM_CIRCULAR_BUFFERS]{};
    auto& view = s_cbs[cb_id];
    if (__emule_cbs == nullptr) {
        return view;
    }
    auto& sync = __emule_cbs[cb_id];
    const uint32_t r = sync.page_mask ? (sync.read_idx & sync.page_mask)
                                      : (sync.num_pages ? (sync.read_idx % sync.num_pages) : 0);
    const uint32_t w = sync.page_mask ? (sync.write_idx & sync.page_mask)
                                      : (sync.num_pages ? (sync.write_idx % sync.num_pages) : 0);
    // Encode host addresses as 16-byte indices so kernel-side `<< 4` reconstitutes
    // a byte pointer into mmap'd L1.
    const uintptr_t base_host = sync.base ? reinterpret_cast<uintptr_t>(sync.base) : 0;
    const uintptr_t rp_host = sync.base ? reinterpret_cast<uintptr_t>(sync.base + r * sync.page_size) : 0;
    const uintptr_t wp_host = sync.base ? reinterpret_cast<uintptr_t>(sync.base + w * sync.page_size) : 0;
    view.fifo_size = sync.page_size * sync.num_pages;
    view.fifo_limit = static_cast<uint32_t>((base_host + view.fifo_size) >> 4);
    view.fifo_page_size = sync.page_size;
    view.fifo_num_pages = sync.num_pages;
    view.fifo_rd_ptr = static_cast<uint32_t>(rp_host >> 4);
    view.fifo_wr_ptr = static_cast<uint32_t>(wp_host >> 4);
    view.tiles_received = static_cast<uint16_t>(sync.occupied.load(std::memory_order_relaxed));
    view.tiles_acked = 0;
    view.fifo_wr_tile_ptr = view.fifo_wr_ptr;
    return view;
}
#endif
