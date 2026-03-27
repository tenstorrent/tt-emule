#pragma once
// Emulation stub for experimental::Semaphore.
// In emulation, kernels run sequentially so semaphore synchronization
// is trivial — waits are no-ops, increments/decrements are direct memory ops.

#include <cstdint>
#include "jit_hw/experimental/noc.h"

namespace experimental {

template <int core_type = 0>
class Semaphore {
public:
    explicit Semaphore(uint32_t semaphore_id) : local_l1_addr_(get_semaphore(semaphore_id)) {}

    void up(uint32_t value) {
        volatile uint32_t* sem = reinterpret_cast<volatile uint32_t*>(local_l1_addr_);
        *sem += value;
    }

    void up(const Noc&, uint32_t, uint32_t, uint32_t value, uint8_t = 0) {
        up(value);
    }

    void down(uint32_t value) {
        volatile uint32_t* sem = reinterpret_cast<volatile uint32_t*>(local_l1_addr_);
        *sem -= value;
    }

    void wait(uint32_t) {}
    void wait_min(uint32_t) {}

    void set(uint32_t value) {
        volatile uint32_t* sem = reinterpret_cast<volatile uint32_t*>(local_l1_addr_);
        *sem = value;
    }

    template <typename M = void>
    void set_multicast(const Noc&, uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, bool = false) {}

    void inc_multicast(const Noc&, uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, uint32_t) {}

private:
    uintptr_t local_l1_addr_;
};

}  // namespace experimental
