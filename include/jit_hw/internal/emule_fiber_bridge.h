// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Fiber-scheduler bridge — the cooperative scheduler lives in the host runner
// (tt_metal/impl/emulation/emule_fiber_scheduler.{hpp,cpp}), but the blocking
// sync primitives that must yield (cb_wait_front, noc_semaphore_wait, dfb_*, …)
// are jit_hw headers compiled INTO the dlopen'd kernel .so. They reach the one
// scheduler instance through the same extern-C + `-rdynamic` mechanism the runner
// already uses for __emule_dram_ptr / __emule_resolve_noc_addr / __emule_self:
// the runner defines these symbols, exports them, and the .so resolves them at
// dlopen. So tt-emule stays header-only (declarations + an inline helper here).
//
// See docs/fiber-engine.md for the concurrency model. The lost-wakeup guard
// (needed at K>1) is the classic condvar pattern: the predicate re-check must
// happen UNDER the scheduler lock, so the lock is exported and the re-check runs
// here in the .so (it owns the predicate). __emule_fiber_wait() wraps the loop.
//
//   key = the host address of the sync object (CBSyncState* / TileCounter* / the
//   semaphore atom). Unique per process (the eventual multi-process multichip case
//   extends the key — see docs/fiber-engine.md).

#include <cstdint>
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self (per-fiber Dirty-CB snapshot)

extern "C" {

// Scheduler lock — guards the ready/parked structures. Acquire it around the
// predicate re-check + park so a concurrent wake cannot slip between them.
void __emule_fiber_lock(void);
void __emule_fiber_unlock(void);

// Park the CURRENT fiber on `key`. PRECONDITION: the scheduler lock is held (via
// __emule_fiber_lock). The fiber is registered parked and yields its worker; the
// lock is handed to the scheduler across the switch. POSTCONDITION on return
// (after a wake re-queued and rescheduled the fiber): the lock is NOT held. The
// caller must re-check its predicate (a wake means "look again", not "proceed").
void __emule_fiber_park_locked(const void* key);

// Re-queue every fiber parked on `key` (no-op if none). Self-contained (takes the
// lock internally). Called by a producer after its state-changing atomic store.
void __emule_fiber_wake(const void* key);

// Voluntary reschedule — yield the worker to another runnable fiber without
// parking (where a thread previously spun/slept to let a peer run).
void __emule_fiber_yield(void);

// Tier-2 watchdog: note genuine forward progress (data published — cb_push /
// dfb_push page counts). Fiber completions are counted by the scheduler itself.
void __emule_fiber_note_publish(unsigned pages);

}  // extern "C"

// ---- Per-fiber Dirty-CB snapshot swap (fixes the co-scheduled false positive) ----
//
// The Dirty-CB sanitizer bookkeeping (dangling reserve/wait flags, reserved/waited
// window counters, and reserve/wait call sites) lives in per-*worker* thread_locals
// because the tt-metal exit-time sweep (sweep_per_kernel_dirty_cbs) reads it there.
// But those flags track per-*kernel* (per-fiber) progress. A worker cooperatively
// hosts many fibers; whenever a fiber parks (or yields) it hands its worker to a
// peer that shares those thread_locals. If the peer runs its exit-time sweep while
// this fiber's `cb_wait_front` is still legitimately outstanding (parked, not yet
// popped), the peer reads THIS fiber's flag and aborts with a false "Dirty CB".
//
// Fix: swap the thread_local set with a per-fiber snapshot (ThreadCommonCtx) across
// every cooperative hand-off — save+clear before yielding the worker, restore on
// resume — so the thread_local always reflects the fiber currently running on the
// worker. A genuine leak (an exiting kernel with its OWN un-popped wait) is
// unaffected: the exiting fiber never parked between its wait and its exit, so its
// flags sit in the thread_local at sweep time exactly as before. Widths are 32 ==
// EMULE_NUM_CBS (emule_sanitizers.hpp), matching ThreadCommonCtx::SAN_CB_SLOTS.
extern thread_local bool __emule_cb_reserve_dangling[32];
extern thread_local bool __emule_cb_wait_dangling[32];
extern thread_local uint32_t __emule_cb_reserved_pages[32];
extern thread_local uint32_t __emule_cb_waited_pages[32];
extern thread_local const char* __emule_cb_reserve_file[32];
extern thread_local uint32_t __emule_cb_reserve_line[32];
extern thread_local const char* __emule_cb_wait_file[32];
extern thread_local uint32_t __emule_cb_wait_line[32];

// Save this fiber's Dirty-CB thread_local set into its ctx, then CLEAR the
// thread_local so a peer that STARTS fresh on this worker (no snapshot to restore)
// begins from a clean slate. Called immediately before the worker is handed off.
inline void __emule_fiber_cb_asan_save() {
    ThreadCommonCtx* self = __emule_self;
    if (!self) {
        return;  // off-fiber (dispatch thread); nothing to snapshot
    }
    for (uint32_t i = 0; i < ThreadCommonCtx::SAN_CB_SLOTS; ++i) {
        self->san_cb_reserve_dangling[i] = __emule_cb_reserve_dangling[i];
        self->san_cb_wait_dangling[i]    = __emule_cb_wait_dangling[i];
        self->san_cb_reserved_pages[i]   = __emule_cb_reserved_pages[i];
        self->san_cb_waited_pages[i]     = __emule_cb_waited_pages[i];
        self->san_cb_reserve_file[i]     = __emule_cb_reserve_file[i];
        self->san_cb_reserve_line[i]     = __emule_cb_reserve_line[i];
        self->san_cb_wait_file[i]        = __emule_cb_wait_file[i];
        self->san_cb_wait_line[i]        = __emule_cb_wait_line[i];
        __emule_cb_reserve_dangling[i] = false;
        __emule_cb_wait_dangling[i]    = false;
        __emule_cb_reserved_pages[i]   = 0;
        __emule_cb_waited_pages[i]     = 0;
    }
}

// Restore this fiber's Dirty-CB thread_local set from its ctx when it resumes on the
// worker, overwriting whatever a peer left behind.
inline void __emule_fiber_cb_asan_restore() {
    ThreadCommonCtx* self = __emule_self;
    if (!self) {
        return;
    }
    for (uint32_t i = 0; i < ThreadCommonCtx::SAN_CB_SLOTS; ++i) {
        __emule_cb_reserve_dangling[i] = self->san_cb_reserve_dangling[i];
        __emule_cb_wait_dangling[i]    = self->san_cb_wait_dangling[i];
        __emule_cb_reserved_pages[i]   = self->san_cb_reserved_pages[i];
        __emule_cb_waited_pages[i]     = self->san_cb_waited_pages[i];
        __emule_cb_reserve_file[i]     = self->san_cb_reserve_file[i];
        __emule_cb_reserve_line[i]     = self->san_cb_reserve_line[i];
        __emule_cb_wait_file[i]        = self->san_cb_wait_file[i];
        __emule_cb_wait_line[i]        = self->san_cb_wait_line[i];
    }
}

// Park the current fiber on `key` with the Dirty-CB snapshot swapped across the
// worker hand-off (pre: scheduler lock held, as __emule_fiber_park_locked requires;
// post: lock NOT held). Use this instead of the raw thunk everywhere a blocking
// primitive parks.
inline void __emule_fiber_park_locked_shielded(const void* key) {
    __emule_fiber_cb_asan_save();
    __emule_fiber_park_locked(key);   // yields the worker; returns unlocked when woken
    __emule_fiber_cb_asan_restore();
}

// Voluntary reschedule (no park) with the same snapshot shielding: yield hands the
// worker to a peer and returns to THIS fiber, so the same save/restore applies.
inline void __emule_fiber_yield_shielded(void) {
    __emule_fiber_cb_asan_save();
    __emule_fiber_yield();
    __emule_fiber_cb_asan_restore();
}

// Block the current fiber until `pred()` is true. Keeps the caller's lock-free
// fast path effective (pred() is checked before any lock), then runs the
// register-then-recheck-under-lock loop. Inline in the .so; pred is the caller's
// predicate over the sync object (e.g. `cb.occupied.load() >= n`).
template <class Pred>
inline void __emule_fiber_wait(const void* key, Pred pred) {
    if (pred()) {
        return;  // fast path: already satisfied, no lock, no park
    }
    for (;;) {
        __emule_fiber_lock();
        if (pred()) {  // re-check under the lock — closes the lost-wakeup window
            __emule_fiber_unlock();
            return;
        }
        __emule_fiber_park_locked_shielded(key);  // park + per-fiber Dirty-CB snapshot swap
    }
}
