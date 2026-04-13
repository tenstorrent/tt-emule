#pragma once
// Emulation stub for experimental::Semaphore.
// Delegates to the real noc_semaphore_* functions from dataflow_api.h for
// proper cross-core synchronization (atomic ops, spin-waits, multicast).

#include <atomic>
#include <cstdint>
#include "jit_hw/experimental/noc.h"
#include "jit_hw/api/dataflow/dataflow_api.h"

namespace experimental {

template <int core_type = 0>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) : local_l1_addr_(get_semaphore(semaphore_id)) {}

    // Local atomic increment (thread-safe for emulation).
    void up(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        __emule_sem_atomic(sem_addr)->fetch_add(value, std::memory_order_acq_rel);
    }

    // Remote atomic increment on another core's semaphore.
    void up(const Noc& noc, uint32_t noc_x, uint32_t noc_y, uint32_t value, uint8_t vc = 0) {
        uint64_t dest_noc_addr = get_noc_addr(noc_x, noc_y, static_cast<uint32_t>(local_l1_addr_), noc.get_noc_id());
        noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id());
    }

    // Spin-wait until semaphore >= value, then atomically subtract.
    void down(uint32_t value) {
        auto* sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_);
        noc_semaphore_wait_min(sem_addr, value);
        __emule_sem_atomic(sem_addr)->fetch_sub(value, std::memory_order_acq_rel);
    }

    // Block until semaphore equals value.
    void wait(uint32_t value) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    // Block until semaphore >= value.
    void wait_min(uint32_t value) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    // Set semaphore value (local atomic store).
    void set(uint32_t value) {
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_l1_addr_), value);
    }

    // Multicast semaphore value to a rectangle of cores.
    template <Noc::McastMode mcast_mode = Noc::McastMode::EXCLUDE_SRC>
    void set_multicast(const Noc& noc, uint32_t noc_x_start, uint32_t noc_y_start,
                       uint32_t noc_x_end, uint32_t noc_y_end,
                       uint32_t num_dests, bool linked = false) {
        uint64_t mcast_addr = get_noc_multicast_addr(
            noc_x_start, noc_y_start, noc_x_end, noc_y_end,
            static_cast<uint32_t>(local_l1_addr_), noc.get_noc_id());
        if constexpr (mcast_mode == Noc::McastMode::EXCLUDE_SRC) {
            noc_semaphore_set_multicast(
                static_cast<uint32_t>(local_l1_addr_), mcast_addr,
                num_dests, linked, noc.get_noc_id());
        } else {
            noc_semaphore_set_multicast_loopback_src(
                static_cast<uint32_t>(local_l1_addr_), mcast_addr,
                num_dests, linked, noc.get_noc_id());
        }
    }

    // Multicast atomic increment to a rectangle of cores.
    void inc_multicast(const Noc& noc, uint32_t noc_x_start, uint32_t noc_y_start,
                       uint32_t noc_x_end, uint32_t noc_y_end,
                       uint32_t value, uint32_t num_dests) {
        // Iterate over the rectangle and atomically increment each core's semaphore.
        for (uint32_t x = std::min(noc_x_start, noc_x_end); x <= std::max(noc_x_start, noc_x_end); x++) {
            for (uint32_t y = std::min(noc_y_start, noc_y_end); y <= std::max(noc_y_start, noc_y_end); y++) {
                uint64_t dest_noc_addr = get_noc_addr(x, y, static_cast<uint32_t>(local_l1_addr_), noc.get_noc_id());
                noc_semaphore_inc(dest_noc_addr, value, noc.get_noc_id());
            }
        }
    }

private:
    uintptr_t local_l1_addr_;
};

}  // namespace experimental
