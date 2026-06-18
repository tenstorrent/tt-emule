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
    }

    void down(uint32_t value) {
        atom()->fetch_sub(value, std::memory_order_release);
    }

    void wait(uint32_t target) {
        auto* a = atom();
        // Mirror the free-function noc_semaphore_wait (dataflow_api.h): for a
        // monotonic count-up handshake (target > 0), "reached" means >= target,
        // not exact ==. emule's increments are zero-latency atomics, so a peer can
        // advance the counter past `target` between two of our polls and an
        // equality wait would miss it and spin to the watchdog. (Silicon paces
        // increments over the NOC, so == never misses there.) For the VALID->0
        // release toggle (target == 0) keep exact equality; a count-up target is
        // never 0, so the split is unambiguous.
        auto reached = [target](uint32_t cur) { return target > 0 ? cur >= target : cur == target; };
        uint64_t spins = 0;
        while (!reached(a->load(std::memory_order_acquire))) {
            if (spins < 64) {
                // busy-spin
            } else if (spins < 1024) {
                sched_yield();
            } else {
                usleep(1);
            }
            if (++spins > 10'000'000ULL) {
                fprintf(stderr,
                    "EMULE HANG: ckernel::Semaphore::wait(%u) stuck at %u "
                    "after %llu spins\n",
                    target, a->load(std::memory_order_relaxed),
                    (unsigned long long)spins);
                std::abort();
            }
        }
    }

    void wait_min(uint32_t min_val) {
        auto* a = atom();
        uint64_t spins = 0;
        while (a->load(std::memory_order_acquire) < min_val) {
            if (spins < 64) {
                // busy-spin
            } else if (spins < 1024) {
                sched_yield();
            } else {
                usleep(1);
            }
            if (++spins > 10'000'000ULL) {
                fprintf(stderr,
                    "EMULE HANG: ckernel::Semaphore::wait_min(%u) stuck at %u "
                    "after %llu spins\n",
                    min_val, a->load(std::memory_order_relaxed),
                    (unsigned long long)spins);
                std::abort();
            }
        }
    }

    void set(uint32_t value) {
        atom()->store(value, std::memory_order_release);
    }

private:
    uintptr_t local_l1_addr_;

    std::atomic<uint32_t>* atom() const {
        return reinterpret_cast<std::atomic<uint32_t>*>(local_l1_addr_);
    }
};

}  // namespace ckernel
