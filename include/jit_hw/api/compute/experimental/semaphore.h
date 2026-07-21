// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulation stub for ckernel::Semaphore (Quasar compute kernels).
// Same L1-resident atomic counter as experimental::Semaphore but in the
// ckernel namespace and used by TRISC/compute-side code.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>

#include "jit_hw/internal/emule_fiber_bridge.h"  // __emule_fiber_wait / _wake (park/wake)

// get_semaphore() is defined in jit_kernel_stubs.hpp (included by all JIT
// kernels).  It returns a 0-based uint32 L1 offset for the given semaphore ID.

namespace ckernel {

class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id)
        : local_l1_addr_(static_cast<uintptr_t>(get_semaphore(semaphore_id))) {}

    void up(uint32_t value) {
        atom()->fetch_add(value, std::memory_order_release);
        __emule_fiber_wake(atom());
    }

    void down(uint32_t value) {
        atom()->fetch_sub(value, std::memory_order_release);
        __emule_fiber_wake(atom());
    }

    void wait(uint32_t target) {
        auto* a = atom();
        // Fiber park/wake (replaces the busy-spin): yield the worker until the
        // semaphore reaches the target. For a monotonic count-up handshake
        // (target > 0), "reached" means >= target, not exact ==: emule's
        // increments are zero-latency atomics, so a peer can advance the counter
        // past `target` between our wake and re-check and an equality wait would
        // miss it and hang. (Silicon paces increments over the NOC, so == never
        // misses there.) For the VALID->0 release toggle (target == 0) keep exact
        // equality; a count-up target is never 0, so the split is unambiguous.
        __emule_fiber_wait(a, [&] {
            uint32_t cur = a->load(std::memory_order_acquire);
            return target > 0 ? cur >= target : cur == target;
        });
    }

    void wait_min(uint32_t min_val) {
        auto* a = atom();
        __emule_fiber_wait(a, [&] { return a->load(std::memory_order_acquire) >= min_val; });
    }

    void set(uint32_t value) {
        atom()->store(value, std::memory_order_release);
        __emule_fiber_wake(atom());
    }

private:
    uintptr_t local_l1_addr_;

    std::atomic<uint32_t>* atom() const {
        return reinterpret_cast<std::atomic<uint32_t>*>(local_l1_addr_);
    }
};

}  // namespace ckernel
