#pragma once
#include "common.hpp"
#include "tt_emule/device.hpp"
#include "tt_emule/circular_buffer.hpp"
#include <cstring>
#include <cstdint>

// NOC address type (prototype: just a raw pointer)
using NocAddr = uint8_t*;

// ---- Circular Buffer operations ----

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    __core->cb(cb_id)->reserve_back(n);
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    __core->cb(cb_id)->push_back(n);
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    __core->cb(cb_id)->wait_front(n);
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    __core->cb(cb_id)->pop_front(n);
}

inline uint8_t* get_write_ptr(uint32_t cb_id) {
    return __core->cb(cb_id)->get_write_ptr();
}

inline const uint8_t* get_read_ptr(uint32_t cb_id) {
    return __core->cb(cb_id)->get_read_ptr();
}

// ---- NOC operations ----

// Resolve a (x, y, addr) tuple into a raw pointer.
inline NocAddr get_noc_addr(uint32_t x, uint32_t y, uint64_t addr) {
    return __device->noc_resolve(x, y, addr);
}

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
