// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Shared circular buffer synchronization state and operations.
//
// SCOPE (issue #139): this struct is the per-core, thread-SHARED CB state. It
// owns ONLY what is genuinely shared across RISCs on silicon — the geometry
// (base/page_size/num_pages/page_mask) and the pages-occupied **semaphore**
// (occupied + cvs), the analog of L1 pages_received/acked. The per-RISC read/
// write *pointers* do NOT live here: they are per-thread, in
// jit_hw/internal/emule_cb_ptr.h (__emule_local_cb), mirroring silicon's
// per-RISC pointer registers. Keeping the pointers out of this shared struct is
// what fixes the pad_rm_sharded_stickwise race (see emule_cb_ptr.h).
//
// Consumed by JIT-compiled kernels via jit_hw/emule_cb_state.h
// (__emule_cb_state / __emule_cbs).

#include <atomic>
#include <cstdint>

#include "jit_hw/internal/emule_fiber_bridge.h"

namespace tt_emule {

struct CBSyncState {
    uint8_t*  base      = nullptr;  // Host pointer to start of CB memory region
    uint32_t  page_size = 0;        // Bytes per page (tile size)
    uint32_t  num_pages = 0;        // Capacity
    uint32_t  page_mask = 0;        // num_pages - 1 (for bitmask modulo; 0 if non-power-of-2)
    bool      globally_allocated = false;  // exempt from CB-Boundary window check; see docs/ASAN.md
    std::atomic<uint32_t> occupied{0};  // Number of occupied pages (the shared semaphore)
    // No mutex/CV here: under the fiber engine cb_reserve_back/cb_wait_front park on &cb
    // and are woken by cb_sync_push/pop below. See docs/fiber-engine.md.
};

// ---- Semaphore operations on CBSyncState ----
//
// These manage ONLY the shared pages-occupied count + producer/consumer wakeups.
// Per-RISC pointer advance lives in jit_hw/internal/emule_cb_ptr.h. The wake key is
// &cb (the CBSyncState host address) — the same key cb_reserve_back/cb_wait_front
// park on.

inline void cb_sync_push(CBSyncState& cb, uint32_t n) {
    // Producer published n pages: bump the semaphore and wake a waiting consumer.
    cb.occupied.fetch_add(n, std::memory_order_release);
    __emule_fiber_note_publish(n);  // tier-2 watchdog: real forward progress
    __emule_fiber_wake(&cb);
}

inline void cb_sync_pop(CBSyncState& cb, uint32_t n) {
    // Consumer freed n pages: drop the semaphore and wake a waiting producer.
    cb.occupied.fetch_sub(n, std::memory_order_release);
    __emule_fiber_wake(&cb);
}

} // namespace tt_emule
