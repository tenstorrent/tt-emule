#pragma once
// JIT dataflow API — self-contained implementation for JIT-compiled kernels.
// Uses uint32_t L1 addresses (truncated host pointers from mmap'd-below-4GB L1)
// and __emule_cbs for circular buffer state.
//
// Does NOT include kernel_api/dataflow_api.hpp — this file provides all needed
// functions with signatures matching the real device API.

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/api/cb_api.h"
#include "jit_hw/api/tensor/tensor_accessor.h"
#include <cstring>
#include <cstdint>
#include <thread>

// ---- Bridge function declarations for cross-core access ----
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);

// ---- NOC address encoding ----
// Unicast: bits[63:48]=noc_x, [47:32]=noc_y, [31:0]=L1 address
inline uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr) {
    return (uint64_t(noc_x & 0xFFFF) << 48) |
           (uint64_t(noc_y & 0xFFFF) << 32) |
           uint64_t(addr);
}

// Overload: get NOC address for local core's L1 address
inline uint64_t get_noc_addr(uint32_t addr) {
    // For local access, encode with x=0, y=0 (resolves to own L1)
    return uint64_t(addr);
}

// Multicast: bits[63:56]=x_start, [55:48]=y_start, [47:40]=x_end, [39:32]=y_end, [31:0]=L1 address
inline uint64_t get_noc_multicast_addr(
    uint32_t x_start, uint32_t y_start,
    uint32_t x_end, uint32_t y_end,
    uint32_t addr) {
    return (uint64_t(x_start & 0xFF) << 56) |
           (uint64_t(y_start & 0xFF) << 48) |
           (uint64_t(x_end & 0xFF) << 40) |
           (uint64_t(y_end & 0xFF) << 32) |
           uint64_t(addr);
}

// ---- NOC page operations ----

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

// noc_async_read_tile — deprecated alias for noc_async_read_page.
template<typename AddrGen>
FORCE_INLINE void noc_async_read_tile(
        uint32_t id, const AddrGen& addrgen,
        uint32_t dst_local_l1_addr,
        uint32_t offset = 0, uint8_t noc = 0) {
    noc_async_read_page(id, addrgen, dst_local_l1_addr, offset, noc);
}

// ---- Raw NOC read/write ----

inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
                           uint32_t size, uint8_t noc = 0) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_local_l1_addr));
    // Try cross-core resolution first, fall back to DRAM
    uint8_t* src = __emule_resolve_noc_addr(src_noc_addr);
    if (!src) src = __emule_dram_ptr(src_noc_addr);
    if (src) std::memcpy(dst, src, size);
}

inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
                            uint32_t size, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    uint8_t* dst = __emule_resolve_noc_addr(dst_noc_addr);
    if (!dst) dst = __emule_dram_ptr(dst_noc_addr);
    if (dst) std::memcpy(dst, src, size);
}

// ---- Multicast write ----

inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    __emule_multicast_write(dst_mcast_noc_addr, src, size);
}

inline void noc_async_write_multicast_loopback_src(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    noc_async_write_multicast(src_local_l1_addr, dst_mcast_noc_addr, size, num_dests, linked, noc);
}

// ---- Barriers ----

inline void noc_async_read_barrier() {}
inline void noc_async_write_barrier() {}
inline void noc_async_writes_flushed() {}

// ---- Semaphore operations ----

// Get L1 address of semaphore by id.
// Uses EMULE_SEM_BASE + id * EMULE_SEM_ALIGN, matching the program runner.
#ifndef EMULE_SEM_BASE
#define EMULE_SEM_BASE 0xFFE00
#endif
#ifndef EMULE_SEM_ALIGN
#define EMULE_SEM_ALIGN 16
#endif

inline uint32_t get_semaphore(uint32_t semaphore_id) {
    return EMULE_SEM_BASE + semaphore_id * EMULE_SEM_ALIGN;
}

// Set semaphore value (local L1 store).
// addr is a uint32_t L1 address (truncated host pointer).
inline void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    *sem_addr = val;
}

// Wait for semaphore to reach expected value (blocking spin-wait).
inline void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    while (*sem_addr != val) {
        std::this_thread::yield();
    }
}

// Wait for semaphore to reach at least min_val (blocking spin-wait).
inline void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t min_val) {
    while (*sem_addr < min_val) {
        std::this_thread::yield();
    }
}

// Atomically increment a remote semaphore.
// noc_addr is a 64-bit encoded NOC address pointing to the semaphore.
inline void noc_semaphore_inc(uint64_t noc_addr, uint32_t incr, uint8_t noc = 0) {
    uint8_t* ptr = __emule_resolve_noc_addr(noc_addr);
    if (ptr) {
        __atomic_fetch_add(reinterpret_cast<uint32_t*>(ptr), incr, __ATOMIC_SEQ_CST);
    }
}

// Multicast a semaphore value to multiple cores.
inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    __emule_multicast_write(dst_mcast_noc_addr, src, sizeof(uint32_t));
}

inline void noc_semaphore_set_multicast_loopback_src(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    noc_semaphore_set_multicast(src_local_l1_addr, dst_mcast_noc_addr, num_dests, linked, noc);
}

// ---- CB interface struct (used by writer kernels) ----
struct CBInterface {
    uint32_t fifo_page_size;
};

inline CBInterface get_local_cb_interface(uint32_t cb_id) {
    return CBInterface{__emule_cbs[cb_id].page_size};
}
