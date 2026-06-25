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
        __emule_fiber_park_locked(key);  // yields the worker; returns unlocked when woken
    }
}
