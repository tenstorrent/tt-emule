// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulation stub for experimental::Semaphore.
// Semaphore operations use atomic memory ops and spin-waits for cross-thread
// synchronization.  up/down are atomic fetch_add/fetch_sub; wait/wait_min
// spin with exponential back-off and hang detection.
//
// Remote operations (up with noc coords, set_multicast, inc_multicast) resolve
// the target core's L1 via __emule_resolve_noc_addr / __emule_multicast_write.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>
#include "jit_hw/experimental/noc.h"

extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern thread_local uint8_t* __emule_bridge_l1;

namespace experimental {

template <int core_type = 0>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) : local_l1_addr_(get_semaphore(semaphore_id)) {
        // Store the L1 offset for remote addressing.
        l1_offset_ = static_cast<uint32_t>(
            local_l1_addr_ - reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    }

    void up(uint32_t value) {
        atom()->fetch_add(value, std::memory_order_release);
    }

    // Remote atomic increment: resolve target core's semaphore via NOC address.
    void up(const Noc&, uint32_t remote_noc_x, uint32_t remote_noc_y,
            uint32_t value, uint8_t = 0) {
        uint64_t noc_addr = (static_cast<uint64_t>(remote_noc_y) << 42) |
                            (static_cast<uint64_t>(remote_noc_x) << 36) |
                            l1_offset_;
        uint8_t* ptr = __emule_resolve_noc_addr(noc_addr);
        if (ptr) {
            reinterpret_cast<std::atomic<uint32_t>*>(ptr)->fetch_add(
                value, std::memory_order_release);
        } else {
            fprintf(stderr, "EMULE WARN: Semaphore::up remote (%u,%u) offset=0x%x "
                    "failed to resolve\n", remote_noc_x, remote_noc_y, l1_offset_);
        }
    }

    void down(uint32_t value) {
        atom()->fetch_sub(value, std::memory_order_release);
    }

    void wait(uint32_t target) {
        auto* a = atom();
        uint64_t spins = 0;
        while (a->load(std::memory_order_acquire) != target) {
            if (spins < 64) {
                // busy-spin
            } else if (spins < 1024) {
                sched_yield();
            } else {
                usleep(1);
            }
            if (++spins > 10'000'000ULL) {
                fprintf(stderr,
                    "EMULE HANG: experimental::Semaphore::wait(%u) stuck at %u "
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
                    "EMULE HANG: experimental::Semaphore::wait_min(%u) stuck at %u "
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

    // Multicast set: write local semaphore value to all cores in the rectangle.
    template <typename M = void>
    void set_multicast(const Noc&, uint32_t x_start, uint32_t y_start,
                       uint32_t x_end, uint32_t y_end,
                       [[maybe_unused]] uint32_t num_cores, bool linked = false) {
        uint64_t mcast_addr =
            (static_cast<uint64_t>(y_start) << 54) |
            (static_cast<uint64_t>(x_start) << 48) |
            (static_cast<uint64_t>(y_end) << 42) |
            (static_cast<uint64_t>(x_end) << 36) |
            l1_offset_;
        // Load atomic value before passing to multicast to avoid data race.
        uint32_t val = atom()->load(std::memory_order_acquire);
        __emule_multicast_write(mcast_addr,
                                reinterpret_cast<const uint8_t*>(&val),
                                sizeof(uint32_t));
    }

    // Multicast increment: atomically add value to semaphore on all cores.
    void inc_multicast(const Noc&, uint32_t x_start, uint32_t y_start,
                       uint32_t x_end, uint32_t y_end,
                       [[maybe_unused]] uint32_t num_cores, uint32_t value) {
        // Iterate rectangle and atomic-add to each core's semaphore.
        for (uint32_t x = std::min(x_start, x_end); x <= std::max(x_start, x_end); x++) {
            for (uint32_t y = std::min(y_start, y_end); y <= std::max(y_start, y_end); y++) {
                uint64_t noc_addr = (static_cast<uint64_t>(y) << 42) |
                                    (static_cast<uint64_t>(x) << 36) |
                                    l1_offset_;
                uint8_t* ptr = __emule_resolve_noc_addr(noc_addr);
                if (ptr) {
                    reinterpret_cast<std::atomic<uint32_t>*>(ptr)->fetch_add(
                        value, std::memory_order_release);
                } else {
                    fprintf(stderr, "EMULE WARN: Semaphore::inc_multicast (%u,%u) "
                            "offset=0x%x failed to resolve\n", x, y, l1_offset_);
                }
            }
        }
    }

private:
    uintptr_t local_l1_addr_;
    uint32_t l1_offset_;  // L1 offset of semaphore (for NOC address construction)

    std::atomic<uint32_t>* atom() const {
        return reinterpret_cast<std::atomic<uint32_t>*>(local_l1_addr_);
    }
};

}  // namespace experimental
