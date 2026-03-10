#pragma once
// JIT dataflow API — self-contained implementation for JIT-compiled kernels.
// Uses uint32_t L1 addresses (truncated host pointers from mmap'd-below-4GB L1)
// and __emule_cbs for circular buffer state.
//
// Does NOT include kernel_api/dataflow_api.hpp — this file provides all needed
// functions with signatures matching the real device API.

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/api/tensor/tensor_accessor.h"
#include <cstring>
#include <cstdint>

// ---- Circular Buffer operations ----
// CB state is in __emule_cbs (set per-thread, shared between threads on same core).

inline void cb_reserve_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.space_cv.wait(lk, [&]{ return (cb.num_pages - cb.occupied) >= n; });
}

inline void cb_push_back(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.write_idx = (cb.write_idx + n) % cb.num_pages;
    cb.occupied += n;
    cb.data_cv.notify_all();
}

inline void cb_wait_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.data_cv.wait(lk, [&]{ return cb.occupied >= n; });
}

inline void cb_pop_front(uint32_t cb_id, uint32_t n) {
    auto& cb = __emule_cbs[cb_id];
    std::unique_lock<std::mutex> lk(cb.mu);
    cb.read_idx = (cb.read_idx + n) % cb.num_pages;
    cb.occupied -= n;
    cb.space_cv.notify_all();
}

// get_write_ptr / get_read_ptr — return uint32_t (truncated host pointer).
// The CB memory is mmap'd below 4 GB so truncation to 32 bits is lossless.
// Kernels cast this to volatile pointers for direct memory access.
inline uint32_t get_write_ptr(uint32_t cb_id) {
    auto& cb = __emule_cbs[cb_id];
    uint8_t* ptr = cb.base + (cb.write_idx % cb.num_pages) * cb.page_size;
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

inline uint32_t get_read_ptr(uint32_t cb_id) {
    auto& cb = __emule_cbs[cb_id];
    uint8_t* ptr = cb.base + (cb.read_idx % cb.num_pages) * cb.page_size;
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
}

// get_tile_size — return page size (bytes) for a CB.
inline uint32_t get_tile_size(uint32_t cb_id) {
    return __emule_cbs[cb_id].page_size;
}

// ---- NOC operations ----

// noc_async_read_page: DRAM → L1
// Reads one page from addrgen's DRAM location and copies it to L1 address.
template<typename AddrGen>
FORCE_INLINE void noc_async_read_page(
        uint32_t id, const AddrGen& addrgen,
        uint32_t dst_local_l1_addr,
        uint32_t offset = 0, uint8_t noc = 0) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_local_l1_addr));
    uint64_t src_dram_offset =
        static_cast<uint64_t>(addrgen.bank_base_address) +
        static_cast<uint64_t>(id) * addrgen.page_size + offset;
    uint8_t* src = __emule_dram_ptr(src_dram_offset);
    std::memcpy(dst, src, addrgen.page_size);
}

// noc_async_write_page: L1 → DRAM
template<typename AddrGen>
FORCE_INLINE void noc_async_write_page(
        uint32_t id, const AddrGen& addrgen,
        uint32_t src_local_l1_addr,
        uint32_t size = 0, uint32_t offset = 0, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    uint64_t dst_dram_offset =
        static_cast<uint64_t>(addrgen.bank_base_address) +
        static_cast<uint64_t>(id) * addrgen.page_size + offset;
    uint8_t* dst = __emule_dram_ptr(dst_dram_offset);
    std::memcpy(dst, src, size ? size : addrgen.page_size);
}

// Raw NOC read/write with uint64_t NOC addresses and uint32_t L1 addresses.
inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
                           uint32_t size, uint8_t noc = 0) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_local_l1_addr));
    uint8_t* src = __emule_dram_ptr(src_noc_addr);
    std::memcpy(dst, src, size);
}

inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
                            uint32_t size, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    uint8_t* dst = __emule_dram_ptr(dst_noc_addr);
    std::memcpy(dst, src, size);
}

// Barriers — no-ops (all operations are synchronous memcpy).
inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}

// Semaphore no-ops
inline void noc_semaphore_set(uint32_t, uint32_t) {}
inline void noc_semaphore_wait(uint32_t, uint32_t) {}
