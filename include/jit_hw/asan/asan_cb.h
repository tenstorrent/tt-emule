// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CB-operation sanitizer bookkeeping, lifted out of the cb_reserve_back /
// cb_push_back / cb_wait_front / cb_pop_front bodies in api/cb_api.h so the CB
// sync logic there stays readable. Each `__emule_asan_cb_on_*` helper bundles
// the per-op ASAN state updates for one CB op:
//   - the Dirty-CB leak flags (reserve/wait *dangling*) + their call sites,
//   - the CB-Boundary *window* counters (reserved / waited pages),
//   - the always-on CB Reservation Overflow check (reserve),
//   - the gated NoC-read-pending race check (pop).
// The window counters and dangling flags are thread_local and only read after
// the owning op returns, so folding their update into one call per op is
// behavior-identical to the previous inline updates. See docs/ASAN.md.

#include <cstdint>
#include <cstdlib>

#include "jit_hw/emule_cb_state.h"   // __emule_cbs (CBSyncState*)
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self (fiber ctx)
#include "jit_hw/asan/emule_asan.h"  // __emule_asan_enabled / __emule_asan_panic

// Identity thread-locals (for the panic diagnostics) + the NoC-read counter.
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;
extern thread_local uint32_t __emule_pending_noc_reads;

// Per-CB call site of the most recent reserve/wait, so the post-exit Dirty-CB
// check (no kernel frame left to backtrace) can name the kernel file:line.
extern thread_local const char* __emule_cb_reserve_file[64];
extern thread_local uint32_t __emule_cb_reserve_line[64];
extern thread_local const char* __emule_cb_wait_file[64];
extern thread_local uint32_t __emule_cb_wait_line[64];

// CB-Boundary *window* counters: pages reserved ahead of / waited behind the
// FIFO pointer (read by the CB Boundary check in asan/asan_l1_checks.h).
extern thread_local uint32_t __emule_cb_reserved_pages[64];
extern thread_local uint32_t __emule_cb_waited_pages[64];

// Dirty-CB leak signal (decoupled from the window counters above): a flag still
// set at kernel exit means a reserve with no following push (or a wait with no
// following pop). See docs/ASAN.md §11 for why a flag, not a net count.
extern thread_local bool __emule_cb_reserve_dangling[64];
extern thread_local bool __emule_cb_wait_dangling[64];

// cb_reserve_back: record the site, mark a reserve outstanding, run the
// always-on Reservation Overflow check, and grow the reserved window. The
// overflow check is never gated — gating it would let an over-reserve deadlock
// on the space wait instead of reporting a clear error (see ASAN.md §8).
inline void __emule_asan_cb_on_reserve(
        uint32_t cb_id, uint32_t n, const char* site_file, uint32_t site_line) {
    __emule_cb_reserve_file[cb_id] = site_file;
    __emule_cb_reserve_line[cb_id] = site_line;
    __emule_cb_reserve_dangling[cb_id] = true;
    const auto& cb = __emule_self->cbs[cb_id];
    if (n > cb.num_pages) {
        __emule_asan_panic(
                "[ASAN ERROR] CB Reservation Overflow: CB %u has %u total pages, "
                "but kernel requested to reserve %u pages. This would hang on silicon! "
                "(page_size=%u) [phys (%u,%u) logical (%u,%u)]\n",
                cb_id, cb.num_pages, n, cb.page_size,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
    }
    __emule_cb_reserved_pages[cb_id] += n;
}

// cb_push_back: any push clears the dangling reserve and shrinks the reserved
// window (saturating at 0).
inline void __emule_asan_cb_on_push(uint32_t cb_id, uint32_t n) {
    __emule_cb_reserve_dangling[cb_id] = false;
    if (n <= __emule_cb_reserved_pages[cb_id]) {
        __emule_cb_reserved_pages[cb_id] -= n;
    } else {
        __emule_cb_reserved_pages[cb_id] = 0;
    }
}

// cb_wait_front: record the site, mark a wait outstanding, and grow the waited
// window with max() (overlapping waits before a pop don't grow the region).
inline void __emule_asan_cb_on_wait(
        uint32_t cb_id, uint32_t n, const char* site_file, uint32_t site_line) {
    __emule_cb_wait_file[cb_id] = site_file;
    __emule_cb_wait_line[cb_id] = site_line;
    __emule_cb_wait_dangling[cb_id] = true;
    if (n > __emule_cb_waited_pages[cb_id]) {
        __emule_cb_waited_pages[cb_id] = n;
    }
}

// cb_pop_front: a pop frees the page for the producer to refill, so all NoC
// reads must be barriered first (gated NoC-read-pending race check); then clear
// the dangling wait and shrink the waited window (saturating at 0).
inline void __emule_asan_cb_on_pop(uint32_t cb_id, uint32_t n) {
    if (__emule_asan_enabled() && __emule_pending_noc_reads > 0) {
        __emule_asan_panic(
                "[ASAN ERROR] Race Condition: cb_pop_front(cb_id=%u) called while a NoC read is still pending "
                "(%u outstanding) — missing noc_async_read_barrier()\n",
                cb_id, __emule_pending_noc_reads);
    }
    __emule_cb_wait_dangling[cb_id] = false;
    if (n <= __emule_cb_waited_pages[cb_id]) {
        __emule_cb_waited_pages[cb_id] -= n;
    } else {
        __emule_cb_waited_pages[cb_id] = 0;
    }
}
