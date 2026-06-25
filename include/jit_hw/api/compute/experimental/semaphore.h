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
// kernels).  It returns a uint32_t L1 address (truncated host pointer) for
// the given semaphore ID.

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
        __emule_fiber_wait(a, [&] { return a->load(std::memory_order_acquire) == target; });
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
