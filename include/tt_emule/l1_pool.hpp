#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>

#include "tt_emule/asan.h"

namespace tt_emule {

/// L1Pool — allocates all worker L1 regions from a single contiguous MAP_32BIT
/// mmap with power-of-2 aligned slots, enabling bitmask offset extraction.
///
/// Each slot is 2 MB (next power of 2 above max L1: 1.5 MB on Blackhole).
/// Offset extraction: `addr & SLOT_MASK` — one AND instruction, no TLS lookup.
class L1Pool {
public:
    static constexpr size_t SLOT_SIZE = 2 * 1024 * 1024;  // 2 MB
    static constexpr size_t SLOT_MASK = SLOT_SIZE - 1;

    /// Construct a pool of `num_slots` 2 MB slots. When `live_size_per_slot`
    /// is smaller than SLOT_SIZE, the unused tail of every slot is poisoned
    /// for AddressSanitizer so kernels writing past their L1 boundary trip
    /// ASan immediately. Pass 0 (the default) or SLOT_SIZE to disable
    /// tail-poisoning.
    explicit L1Pool(size_t num_slots, size_t live_size_per_slot = 0)
        : num_slots_(num_slots),
          live_size_per_slot_(live_size_per_slot == 0 ? SLOT_SIZE : live_size_per_slot) {
        if (num_slots == 0) return;
        size_t total = num_slots * SLOT_SIZE;
        // MAP_32BIT ensures all addresses fit in 32-bit pointers.
        // We request SLOT_SIZE-aligned memory by over-allocating and aligning.
        size_t alloc_size = total + SLOT_SIZE;  // extra for alignment
        void* raw = mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
        if (raw == MAP_FAILED)
            throw std::runtime_error("L1Pool: mmap failed for " +
                                     std::to_string(num_slots) + " slots");
        raw_ = static_cast<uint8_t*>(raw);
        raw_size_ = alloc_size;

        // Align base_ to SLOT_SIZE boundary
        uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_);
        uintptr_t aligned = (raw_addr + SLOT_SIZE - 1) & ~(SLOT_SIZE - 1);
        base_ = reinterpret_cast<uint8_t*>(aligned);

        // Zero out the usable region
        std::memset(base_, 0, total);

        if (live_size_per_slot_ < SLOT_SIZE) {
            const size_t tail = SLOT_SIZE - live_size_per_slot_;
            for (size_t i = 0; i < num_slots_; ++i) {
                EMULE_ASAN_POISON(base_ + i * SLOT_SIZE + live_size_per_slot_, tail);
            }
        }
    }

    ~L1Pool() {
        if (raw_) {
            // Unpoison before munmap so ASan's shadow doesn't outlive the mapping.
            EMULE_ASAN_UNPOISON(raw_, raw_size_);
            munmap(raw_, raw_size_);
        }
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

    /// Extract L1 offset from a host address via bitmask (one AND instruction).
    static uint32_t to_offset(uint32_t addr) { return addr & static_cast<uint32_t>(SLOT_MASK); }

    /// Bytes per slot considered "live" (the rest of the 2 MB is poisoned for ASan).
    size_t live_size_per_slot() const { return live_size_per_slot_; }

private:
    size_t   num_slots_ = 0;
    size_t   live_size_per_slot_ = SLOT_SIZE;
    uint8_t* raw_  = nullptr;  // raw mmap region (before alignment)
    size_t   raw_size_ = 0;
    uint8_t* base_ = nullptr;  // SLOT_SIZE-aligned base
};

}  // namespace tt_emule
