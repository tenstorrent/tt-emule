#pragma once
// JIT dataflow API — self-contained implementation for JIT-compiled kernels.
// Uses uint32_t L1 addresses (truncated host pointers from mmap'd-below-4GB L1)
// and __emule_cbs for circular buffer state.
//
// Address translation model:
//   Encode: __emule_addr_to_offset(addr) — bitmask extracts L1 offset from host ptr
//   Fixup:  __emule_fixup_noc_addr(noc) — fixes OR-constructed NOC addrs (host ptr in lower bits)
//   Decode: __emule_resolve_noc_addr(noc_addr) — core_map lookup → host pointer
//
// Does NOT include kernel_api/dataflow_api.hpp — this file provides all needed
// functions with signatures matching the real device API.

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/api/cb_api.h"
#include "jit_hw/internal/dataflow/dataflow_api_addrgen.h"
#include "jit_hw/api/tensor/tensor_accessor.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>

// ---- Bridge function declarations for cross-core access ----
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);

// ---- L1 address conversion helper ----
// Extract L1 offset from a host address using bitmask.
// L1Pool allocates worker slots at 2 MB alignment, so addr & 0x1FFFFF
// gives the offset within the slot — one AND instruction, no TLS lookup.
// Standalone builds (without L1Pool) fall back to TLS subtraction.
inline uint32_t __emule_addr_to_offset(uint32_t addr) {
#ifdef TT_EMULE_USE_L1_POOL
    return addr & 0x1FFFFF;  // SLOT_MASK = 2 MB - 1
#else
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (addr >= l1_base) return addr - l1_base;
    return addr;
#endif
}

// ---- Coordinate translation tables ----
// On real hardware, these are L1-resident lookup tables populated by firmware.
// In emulation, EMULE_WORKER_COL_MAP / EMULE_WORKER_ROW_MAP provide the
// logical → virtual/physical coordinate mapping, generated from the device's
// virtual_core_from_logical_core() in the program runner.
#ifdef EMULE_WORKER_COL_MAP
inline uint32_t worker_logical_col_to_virtual_col[64] = {EMULE_WORKER_COL_MAP};
#else
inline uint32_t worker_logical_col_to_virtual_col[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
};
#endif
#ifdef EMULE_WORKER_ROW_MAP
inline uint32_t worker_logical_row_to_virtual_row[64] = {EMULE_WORKER_ROW_MAP};
#else
inline uint32_t worker_logical_row_to_virtual_row[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63
};
#endif

// Return the absolute logical x/y coordinate of the current core.
// D2M kernels use these to index into the translation tables above.
inline uint32_t get_absolute_logical_x() { return __emule_logical_x; }
inline uint32_t get_absolute_logical_y() { return __emule_logical_y; }

// ---- NOC address encoding (matches real firmware) ----
// Unicast: y in bits [47:42], x in bits [41:36], addr in bits [35:0]
// NOC_XY_ADDR(x, y, addr) = (y << 42) | (x << 36) | addr
inline uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc = 0) {
    uint32_t l1_off = __emule_addr_to_offset(addr);
    return (uint64_t(noc_y & 0x3F) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
           (uint64_t(noc_x & 0x3F) << NOC_ADDR_LOCAL_BITS) |
           uint64_t(l1_off);
}

// Overload: get NOC address for local core's L1 address
inline uint64_t get_noc_addr(uint32_t addr, uint8_t noc = 0) {
    return get_noc_addr(my_x[noc], my_y[noc], addr, noc);
}

// Multicast address encoding (matches real firmware NOC_MULTICAST_ADDR):
// addr in bits [35:0], x_end [41:36], y_end [47:42], x_start [53:48], y_start [59:54]
inline uint64_t get_noc_multicast_addr(
    uint32_t x_start, uint32_t y_start,
    uint32_t x_end, uint32_t y_end,
    uint32_t addr, uint8_t noc = 0) {
    uint32_t l1_off = __emule_addr_to_offset(addr);
    return (uint64_t(x_start & 0x3F) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) |
           (uint64_t(y_start & 0x3F) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) |
           (uint64_t(x_end & 0x3F) << NOC_ADDR_LOCAL_BITS) |
           (uint64_t(y_end & 0x3F) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
           uint64_t(l1_off);
}

// ---- Addrgen traits (matching real firmware) ----
template <typename, typename = void>
inline constexpr bool has_get_noc_addr_v = false;
template <typename T>
inline constexpr bool has_get_noc_addr_v<
    T, std::void_t<decltype(std::declval<T>().get_noc_addr(
           std::declval<uint32_t>(), std::declval<uint32_t>(), std::declval<uint8_t>()))>> = true;

template <typename, typename = void>
inline constexpr bool has_page_size_v = false;
template <typename T>
inline constexpr bool has_page_size_v<T, std::void_t<decltype(std::declval<T>().page_size)>> = true;

template <typename, typename = void>
inline constexpr bool has_log_base_2_of_page_size_v = false;
template <typename T>
inline constexpr bool has_log_base_2_of_page_size_v<
    T, std::void_t<decltype(std::declval<T>().log_base_2_of_page_size)>> = true;

// ---- NOC page operations (using addrgen.get_noc_addr) ----

template<typename AddrGen>
FORCE_INLINE void noc_async_read_page(
        uint32_t id, const AddrGen& addrgen,
        uint32_t dst_local_l1_addr,
        uint32_t offset = 0, uint8_t noc = 0) {
    uint32_t page_size;
    if constexpr (has_page_size_v<AddrGen>) {
        page_size = addrgen.page_size;
    } else {
        page_size = (1u << addrgen.log_base_2_of_page_size);
    }
    uint64_t noc_addr = addrgen.get_noc_addr(id, offset, noc);
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_local_l1_addr));
    uint8_t* src = __emule_resolve_noc_addr(noc_addr);
    if (src) std::memcpy(dst, src, page_size);
}

template<typename AddrGen>
FORCE_INLINE void noc_async_write_page(
        uint32_t id, const AddrGen& addrgen,
        uint32_t src_local_l1_addr,
        uint32_t size = 0, uint32_t offset = 0, uint8_t noc = 0) {
    uint32_t page_size;
    if constexpr (has_page_size_v<AddrGen>) {
        page_size = addrgen.page_size;
    } else {
        page_size = (1u << addrgen.log_base_2_of_page_size);
    }
    uint32_t sz = size ? size : page_size;
    uint64_t noc_addr = addrgen.get_noc_addr(id, offset, noc);
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    uint8_t* dst = __emule_resolve_noc_addr(noc_addr);
    if (dst) std::memcpy(dst, src, sz);
}

// noc_async_read_tile — deprecated alias for noc_async_read_page.
template<typename AddrGen>
FORCE_INLINE void noc_async_read_tile(
        uint32_t id, const AddrGen& addrgen,
        uint32_t dst_local_l1_addr,
        uint32_t offset = 0, uint8_t noc = 0) {
    noc_async_read_page(id, addrgen, dst_local_l1_addr, offset, noc);
}

// noc_async_write_tile — deprecated alias for noc_async_write_page.
template<typename AddrGen>
FORCE_INLINE void noc_async_write_tile(
        uint32_t id, const AddrGen& addrgen,
        uint32_t src_local_l1_addr,
        uint32_t size = 0, uint32_t offset = 0, uint8_t noc = 0) {
    noc_async_write_page(id, addrgen, src_local_l1_addr, size, offset, noc);
}

// ---- NOC address fixup for OR-constructed addresses ----
// Some kernels construct NOC addresses by ORing a host pointer (L1 address)
// into a pre-computed base: `noc_base | l1_host_ptr`. This embeds the host
// pointer in the address bits instead of an L1 offset. Fix it by masking
// the lower bits to extract just the L1 offset.
inline uint64_t __emule_fixup_noc_addr(uint64_t noc_addr) {
    constexpr uint64_t addr_mask = (1ULL << NOC_ADDR_LOCAL_BITS) - 1;
    uint32_t addr = static_cast<uint32_t>(noc_addr & addr_mask);
    uint32_t fixed = __emule_addr_to_offset(addr);
    return (noc_addr & ~addr_mask) | fixed;
}

// ---- Raw NOC read/write ----

inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
                           uint32_t size, uint8_t noc = 0) {
    uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_local_l1_addr));
    uint8_t* src = __emule_resolve_noc_addr(__emule_fixup_noc_addr(src_noc_addr));
    if (src) std::memcpy(dst, src, size);
}

inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
                            uint32_t size, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    uint8_t* dst = __emule_resolve_noc_addr(__emule_fixup_noc_addr(dst_noc_addr));
    if (dst) std::memcpy(dst, src, size);
}

// ---- Multicast write ----

inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    __emule_multicast_write(__emule_fixup_noc_addr(dst_mcast_noc_addr), src, size);
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
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    return l1_base + EMULE_SEM_BASE + semaphore_id * EMULE_SEM_ALIGN;
}

// Set semaphore value (local L1 store).
// addr is a uint32_t L1 address (truncated host pointer).
inline void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    *sem_addr = val;
}

// Wait for semaphore to reach expected value (blocking spin-wait with hang detection).
inline void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    uint64_t spins = 0;
    while (*sem_addr != val) {
        std::this_thread::yield();
        if (++spins > 100'000'000ULL) {
            fprintf(stderr, "EMULE HANG: noc_semaphore_wait(%p, %u) stuck at %u after %llu spins "
                    "[phys (%u,%u) logical (%u,%u)]\n",
                    (void*)sem_addr, val, *sem_addr, (unsigned long long)spins,
                    my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
            std::abort();
        }
    }
}

// Wait for semaphore to reach at least min_val (blocking spin-wait with hang detection).
inline void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t min_val) {
    uint64_t spins = 0;
    while (*sem_addr < min_val) {
        std::this_thread::yield();
        if (++spins > 100'000'000ULL) {
            fprintf(stderr, "EMULE HANG: noc_semaphore_wait_min(%p, %u) stuck at %u after %llu spins "
                    "[phys (%u,%u) logical (%u,%u)]\n",
                    (void*)sem_addr, min_val, *sem_addr, (unsigned long long)spins,
                    my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
            std::abort();
        }
    }
}

// Atomically increment a remote semaphore.
// noc_addr is a 64-bit encoded NOC address pointing to the semaphore.
inline void noc_semaphore_inc(uint64_t noc_addr, uint32_t incr, uint8_t noc = 0) {
    uint64_t fixed = __emule_fixup_noc_addr(noc_addr);
    uint8_t* ptr = __emule_resolve_noc_addr(fixed);
    if (ptr) {
        __atomic_fetch_add(reinterpret_cast<uint32_t*>(ptr), incr, __ATOMIC_SEQ_CST);
    } else {
        uint32_t noc_x = (fixed >> 36) & 0x3F;
        uint32_t noc_y = (fixed >> 42) & 0x3F;
        uint32_t offset = static_cast<uint32_t>(fixed & ((1ULL << 36) - 1));
        fprintf(stderr, "EMULE WARN: noc_semaphore_inc failed to resolve addr 0x%llx "
                "(target core (%u,%u) offset 0x%x) [from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)fixed, noc_x, noc_y, offset,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
    }
}

// Multicast a semaphore value to multiple cores.
inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    uint8_t* src = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(src_local_l1_addr));
    __emule_multicast_write(__emule_fixup_noc_addr(dst_mcast_noc_addr), src, sizeof(uint32_t));
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

// ---- Standalone NOC address helpers (matching firmware) ----

inline uint64_t get_dram_noc_addr(
    const uint32_t id, const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0, uint8_t noc = 0) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<true>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<true>(id, bank_offset_index);
    uint32_t aligned = align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<true>());
    uint32_t addr = (bank_offset_index * aligned) + bank_base_address + offset +
                    bank_to_dram_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<true>(bank_index, noc);
    return get_noc_addr_helper(noc_xy, addr);
}

inline uint64_t get_l1_noc_addr(
    const uint32_t id, const uint32_t page_size,
    const uint32_t bank_base_address,
    const uint32_t offset = 0, uint8_t noc = 0) {
    uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<false>(id);
    uint32_t bank_index = interleaved_addr_gen::get_bank_index<false>(id, bank_offset_index);
    uint32_t aligned = align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<false>());
    uint32_t addr = (bank_offset_index * aligned) + bank_base_address + offset +
                    bank_to_l1_offset[bank_index];
    uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<false>(bank_index, noc);
    return get_noc_addr_helper(noc_xy, addr);
}

// ---- Preempt tt-mlir verbatim injection of experimental_dataflow_api ----
// D2M code generation injects experimental_dataflow_api.h as verbatim C++
// containing firmware macros (NOC_MULTICAST_ADDR etc.) that don't exist in
// the emulation environment. By defining the header guard here and providing
// our own implementation, the verbatim injection is skipped.
#define TTMLIR_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DATAFLOW_API_H

namespace experimental {
inline std::uint64_t get_noc_multicast_addr(
    std::uint32_t noc_x_start, std::uint32_t noc_y_start,
    std::uint32_t noc_x_end, std::uint32_t noc_y_end,
    std::uint32_t addr, uint8_t noc = noc_index) {
    // noc 0: start=start, end=end; noc 1: start=end, end=start (matches firmware)
    if (noc) {
        return ::get_noc_multicast_addr(noc_x_end, noc_y_end, noc_x_start, noc_y_start, addr, noc);
    } else {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    }
}
} // namespace experimental
