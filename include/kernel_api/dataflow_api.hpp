#pragma once
#include "common.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/circular_buffer.hpp"
#include <cstring>
#include <cstdint>

// NOC address type (prototype: just a raw pointer)
using NocAddr = uint8_t*;

// ---- Circular Buffer operations (delegate to shared CBSyncState logic) ----

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    // check if there are enough pages
    auto& cb_ptr = __core->cb(cb_id);
    uint32_t max_pages = cb_ptr->num_pages();
    
    if (n > max_pages) {
        fprintf(stderr, "[ASAN ERROR] CB Reservation Overflow: CB %u has %u total pages, "
                        "but kernel requested to reserve %u pages. This would hang on silicon!\n", 
                cb_id, max_pages, n);
        abort();
    }

    tt_emule::cb_sync_reserve(__core->cb(cb_id)->sync_state(), n);
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_push(__core->cb(cb_id)->sync_state(), n);
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_wait(__core->cb(cb_id)->sync_state(), n);
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    tt_emule::cb_sync_pop(__core->cb(cb_id)->sync_state(), n);
}

inline uint8_t* get_write_ptr(uint32_t cb_id) {
    // Check if the resulting pointer is word-aligned
    uint8_t* ptr = tt_emule::cb_sync_write_ptr(__core->cb(cb_id)->sync_state());
    
    // Check if the resulting pointer is word-aligned
    if ((uintptr_t)ptr % 4 != 0) {
        fprintf(stderr, "[ASAN ERROR] CB Write Ptr Misaligned: Ptr %p for CB %u is not 4-byte aligned!\n", (void*)ptr, cb_id);
        abort();
    }
    return ptr;
}

inline const uint8_t* get_read_ptr(uint32_t cb_id) {
    return tt_emule::cb_sync_read_ptr_at(__core->cb(cb_id)->sync_state(), 0);
}

// ---- NOC operations ----

// Resolve a (x, y, addr) tuple into a raw pointer.
#ifdef __EMULE_JIT_MODE
// In JIT mode, use extern "C" bridge to avoid Device ABI mismatch.
inline NocAddr get_noc_addr(uint32_t x, uint32_t y, uint64_t addr) {
    return __emule_noc_resolve(x, y, addr);
}
#else
inline NocAddr get_noc_addr(uint32_t x, uint32_t y, uint64_t addr) {
    return __device->noc_resolve(x, y, addr);
}
#endif

// Synchronous memcpy in prototype (emulates async DMA read: src DRAM -> dst L1)
inline void noc_async_read(NocAddr src, uint8_t* dst_l1, size_t size) {
    std::memcpy(dst_l1, src, size);
}

// Synchronous memcpy in prototype (emulates async DMA write: src L1 -> dst DRAM)
inline void noc_async_write(const uint8_t* src_l1, NocAddr dst, size_t size) {
    // Transfers must be 16 byte aligned
    if (((uintptr_t)src_l1 % 16 != 0) || ((uintptr_t)dst % 16 != 0)) {
        fprintf(stderr, "[ASAN ERROR] NOC DMA Write Alignment: Src(0x%p) and Dst(0x%p) must be 16-byte aligned\n", 
                (void*)src_l1, (void*)dst);
        abort();
    }
    
    // Size must be a multiple of 16
    if (size % 16 != 0) {
        fprintf(stderr, "[ASAN ERROR] NOC DMA Write Size: Size (%zu) must be a multiple of 16\n", size);
        abort();
    }

    std::memcpy(dst, src_l1, size);
}

// No-ops in prototype
inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}
