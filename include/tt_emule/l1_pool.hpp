// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <sys/mman.h>

#include "tt_emule/worker_l1_mmap.hpp"

namespace tt_emule {

/// L1Pool — allocates all worker L1 regions from a single contiguous mmap
/// (see worker_l1_mmap.hpp) with power-of-2 aligned slots, enabling bitmask
/// offset extraction.
///
/// Each slot is 2 MB (next power of 2 above max L1: 1.5 MB on Blackhole), so
/// callers extract an in-slot offset with a single `addr & (SLOT_SIZE-1)` mask.
class L1Pool {
public:
    static constexpr size_t SLOT_SIZE = 2 * 1024 * 1024;  // 2 MB

    explicit L1Pool(size_t num_slots) : num_slots_(num_slots) {
        if (num_slots == 0) return;
        size_t total = num_slots * SLOT_SIZE;
        // Request SLOT_SIZE-aligned memory by over-allocating and aligning. No upfront memset:
        // MAP_ANONYMOUS pages zero-fill on first fault, so only touched cores consume RAM.
        size_t alloc_size = total + SLOT_SIZE;  // extra for alignment
        void* raw = __emule_mmap_worker_l1(alloc_size);
        if (raw == MAP_FAILED)
            throw std::runtime_error("L1Pool: mmap failed for " +
                                     std::to_string(num_slots) + " slots");
        raw_ = static_cast<uint8_t*>(raw);
        raw_size_ = alloc_size;

        // Align base_ to SLOT_SIZE boundary
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_);
        uintptr_t aligned = (raw_addr + SLOT_SIZE - 1) & ~(SLOT_SIZE - 1);
        base_ = reinterpret_cast<uint8_t*>(aligned);
    }

    ~L1Pool() {
        if (raw_) munmap(raw_, raw_size_);
    }

    L1Pool(const L1Pool&) = delete;
    L1Pool& operator=(const L1Pool&) = delete;

    /// Get pointer to start of slot `index`.
    uint8_t* slot_ptr(size_t index) const {
        if (index >= num_slots_)
            throw std::out_of_range("L1Pool: slot index out of range");
        return base_ + index * SLOT_SIZE;
    }

    size_t num_slots() const { return num_slots_; }

private:
    size_t   num_slots_ = 0;
    uint8_t* raw_  = nullptr;  // raw mmap region (before alignment)
    size_t   raw_size_ = 0;
    uint8_t* base_ = nullptr;  // SLOT_SIZE-aligned base
};

}  // namespace tt_emule
