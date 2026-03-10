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
    return tt_emule::cb_sync_write_ptr(__core->cb(cb_id)->sync_state());
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
    std::memcpy(dst, src_l1, size);
}

// No-ops in prototype
inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}
