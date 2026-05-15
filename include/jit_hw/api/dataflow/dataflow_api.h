#pragma once
// JIT dataflow API — self-contained implementation for JIT-compiled kernels.
// Uses uint32_t L1 addresses (truncated host pointers from mmap'd-below-4GB L1)
// and __emule_cbs for circular buffer state.
//
// Address translation model:
//   Encode: __emule_addr_to_offset(addr) — bitmask extracts L1 offset from host ptr
//   Decode: __emule_resolve_noc_addr(noc_addr) — core_map lookup → host pointer
//
// NOC addresses from get_noc_addr() have L1 offsets already converted by
// __emule_addr_to_offset.  Addresses from get_noc_addr_from_bank_id() contain
// firmware-style DRAM offsets (possibly > 2MB) and must NOT be masked.
//
// Does NOT include kernel_api/dataflow_api.hpp — this file provides all needed
// functions with signatures matching the real device API.

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/api/cb_api.h"
#include "jit_hw/internal/dataflow/dataflow_api_addrgen.h"
#include "jit_hw/api/tensor/tensor_accessor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <thread>

// ---- NOC virtual channel constants (values are unused in emulation) ----
#ifndef NOC_UNICAST_WRITE_VC
#define NOC_UNICAST_WRITE_VC 2
#endif
#ifndef NOC_MULTICAST_WRITE_VC
#define NOC_MULTICAST_WRITE_VC 8
#endif

// ---- Bridge function declarations for cross-core access ----
extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);

// ---- Debug logging (enabled by TT_EMULE_DEBUG_MULTICAST=1 env var) ----
inline bool __emule_debug_multicast() {
    static bool val = std::getenv("TT_EMULE_DEBUG_MULTICAST") != nullptr;
    return val;
}

// ---- L1 address conversion helpers ----
// Extract L1 offset from a host address using bitmask.
// L1Pool allocates worker slots at 2 MB alignment, so addr & 0x1FFFFF
// gives the offset within the slot — one AND instruction, no TLS lookup.
// Standalone builds (without L1Pool) fall back to TLS subtraction.
//
// IMPORTANT: This must only be called at NOC address construction time
// (get_noc_addr, get_noc_multicast_addr) where the input is a host L1
// pointer.  It must NOT be applied to firmware-style offsets like DRAM
// bank addresses from get_noc_addr_from_bank_id (which can exceed 2MB).
inline uint32_t __emule_addr_to_offset(uint32_t addr) {
#ifdef TT_EMULE_USE_L1_POOL
    return addr & 0x1FFFFF;  // SLOT_MASK = 2 MB - 1
#else
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (addr >= l1_base) return addr - l1_base;
    return addr;
#endif
}

// Inverse of __emule_addr_to_offset: convert a uint32_t L1 address (which
// may be either a firmware-style offset or an absolute host pointer from
// l1_alloc / CB / DFB) to a dereferenceable host pointer.
//
// l1_alloc() returns l1_base_ + bump  (>= l1_base, always a valid host ptr).
// Firmware HAL addresses (e.g. 0x19520) are offsets into the L1 buffer.
// We distinguish by comparing against __emule_bridge_l1's numeric address.
#ifndef __EMULE_LOCAL_L1_TO_PTR_DEFINED
#define __EMULE_LOCAL_L1_TO_PTR_DEFINED
inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr) {
    if (l1_addr % 4 != 0) {
        fprintf(stderr, "[ASAN ERROR] Local L1 Alignment: Offset 0x%x must be 4-byte aligned for scalar access\n", l1_addr);
        abort();
    }
    if (__emule_sem_l1_range_end > 0 &&
        l1_addr >= __emule_sem_l1_range_start && l1_addr < __emule_sem_l1_range_end) {
        fprintf(stderr,
                "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
                l1_addr, __emule_sem_l1_range_start, __emule_sem_l1_range_end);
        abort();
    }
    if (__emule_l1_tensor_ranges != nullptr && l1_addr >= __emule_l1_unreserved_base) {
        bool in_tensor = false;
        for (uint32_t i = 0; i < __emule_l1_tensor_ranges_count; ++i) {
            uint64_t packed = __emule_l1_tensor_ranges[i];
            uint32_t r_start = static_cast<uint32_t>(packed >> 32);
            uint32_t r_end = static_cast<uint32_t>(packed);
            if (l1_addr >= r_start && l1_addr < r_end) {
                in_tensor = true;
                break;
            }
        }
        if (!in_tensor) {
            fprintf(stderr,
                    "[ASAN ERROR] Out-of-Bounds Write: Attempted to access address 0x%x which is not part of any allocated tensor\n",
                    l1_addr);
            abort();
        }
    }
    // CB-boundary sanitizer. If the address lands inside a configured CB's
    // byte range, it must also land inside an active page window — either
    // the write reservation [write_idx, write_idx + reserved) (mod num_pages)
    // or the read wait window [read_idx, read_idx + waited) (mod num_pages).
    // Producer threads only populate `reserved`; consumer threads only
    // populate `waited`; the unused window has count 0 and contributes
    // nothing. Page-distance math is modular so wraparound (reservation
    // spanning the num_pages boundary) is handled naturally.
    if (__emule_cb_boundary_strict && __emule_cbs != nullptr) {
        for (uint32_t cb_id = 0; cb_id < 32; ++cb_id) {
            auto& cb = __emule_cbs[cb_id];
            if (cb.num_pages == 0) continue;
            uint32_t cb_start = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(cb.base));
            uint32_t cb_size = cb.num_pages * cb.page_size;
            if (l1_addr < cb_start || l1_addr >= cb_start + cb_size) continue;
            uint32_t access_page = (l1_addr - cb_start) / cb.page_size;
            uint32_t write_dist = (access_page + cb.num_pages - cb.write_idx) % cb.num_pages;
            uint32_t read_dist  = (access_page + cb.num_pages - cb.read_idx)  % cb.num_pages;
            uint32_t reserved = __emule_cb_reserved_pages[cb_id];
            uint32_t waited   = __emule_cb_waited_pages[cb_id];
            bool in_write_window = (write_dist < reserved);
            bool in_read_window  = (read_dist  < waited);
            if (!in_write_window && !in_read_window) {
                fprintf(stderr,
                        "[ASAN ERROR] CB Boundary Violation: Attempted to access CB %u at offset 0x%x "
                        "(byte %u of %u, page %u of %u). "
                        "Write window: write_idx=%u, %u page(s) reserved. "
                        "Read window: read_idx=%u, %u page(s) waited.\n",
                        cb_id, l1_addr, l1_addr - cb_start, cb_size,
                        access_page, cb.num_pages,
                        cb.write_idx, reserved,
                        cb.read_idx, waited);
                abort();
            }
            break;
        }
    }
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (l1_addr >= l1_base) {
        // Already an absolute host pointer (from l1_alloc / CB / DFB).
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
    }
    // Firmware L1 offset — translate via bridge pointer.
    return __emule_bridge_l1 + l1_addr;
}
#endif

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
// Guarded to avoid conflict with compute/common.h if both are included.
#ifndef __EMULE_GET_LOGICAL_COORDS_DEFINED
#define __EMULE_GET_LOGICAL_COORDS_DEFINED
inline uint32_t get_absolute_logical_x() { return __emule_logical_x; }
inline uint32_t get_absolute_logical_y() { return __emule_logical_y; }
#endif

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
    ++__emule_pending_noc_reads;
    uint64_t noc_addr = addrgen.get_noc_addr(id, offset, noc);
    uint8_t* dst = __emule_local_l1_to_ptr(dst_local_l1_addr);
    uint8_t* src = __emule_resolve_noc_addr(noc_addr);
    if (src) {
        std::memcpy(dst, src, page_size);
    } else {
        fprintf(stderr, "EMULE WARN: noc_async_read_page failed to resolve addr 0x%llx "
                "[from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)noc_addr, my_x[0], my_y[0],
                __emule_logical_x, __emule_logical_y);
    }
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
    uint8_t* src = __emule_local_l1_to_ptr(src_local_l1_addr);
    uint8_t* dst = __emule_resolve_noc_addr(noc_addr);
    if (dst) {
        std::memcpy(dst, src, sz);
    } else {
        fprintf(stderr, "EMULE WARN: noc_async_write_page failed to resolve addr 0x%llx "
                "[from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)noc_addr, my_x[0], my_y[0],
                __emule_logical_x, __emule_logical_y);
    }
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

// ---- Raw NOC read/write ----

inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
                           uint32_t size, uint8_t noc = 0, uint32_t vc = 0) {
    // NOC addresses are already properly constructed by get_noc_addr() or
    // get_noc_addr_from_bank_id() — no fixup needed here.  Applying
    // __emule_fixup_noc_addr would destroy DRAM bank offsets (> 2MB).
    ++__emule_pending_noc_reads;
    uint8_t* dst = __emule_local_l1_to_ptr(dst_local_l1_addr);
    uint8_t* src = __emule_resolve_noc_addr(src_noc_addr);
    if (__emule_debug_multicast()) {
        uint32_t nx = (src_noc_addr >> 36) & 0x3F;
        uint32_t ny = (src_noc_addr >> 42) & 0x3F;
        uint32_t off = static_cast<uint32_t>(src_noc_addr & ((1ULL << 36) - 1));
        fprintf(stderr, "EMULE DBG: noc_async_read src_core=(%u,%u) offset=0x%x size=%u resolved=%p "
                "[from logical (%u,%u)]\n",
                nx, ny, off, size, (void*)src,
                __emule_logical_x, __emule_logical_y);
    }
    if (src) {
        std::memcpy(dst, src, size);
    } else {
        fprintf(stderr, "EMULE WARN: noc_async_read failed to resolve addr 0x%llx "
                "[from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)src_noc_addr, my_x[0], my_y[0],
                __emule_logical_x, __emule_logical_y);
    }
}

inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr,
                            uint32_t size, uint8_t noc = 0, uint32_t vc = 0) {
    uint8_t* src = __emule_local_l1_to_ptr(src_local_l1_addr);
    uint8_t* dst = __emule_resolve_noc_addr(dst_noc_addr);
    if (dst) {
        std::memcpy(dst, src, size);
    } else {
        fprintf(stderr, "EMULE WARN: noc_async_write failed to resolve addr 0x%llx "
                "[from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)dst_noc_addr, my_x[0], my_y[0],
                __emule_logical_x, __emule_logical_y);
    }
}

// ---- Multicast write ----

inline void noc_async_write_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    if (__emule_debug_multicast()) {
        uint32_t x_end   = (dst_mcast_noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t y_end   = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t x_start = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t y_start = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t off     = static_cast<uint32_t>(dst_mcast_noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
        fprintf(stderr, "EMULE DBG: noc_async_write_multicast (%u,%u)->(%u,%u) offset=0x%x size=%u num_dests=%u "
                "[from logical (%u,%u)]\n",
                x_start, y_start, x_end, y_end, off, size, num_dests,
                __emule_logical_x, __emule_logical_y);
    }
    uint8_t* src = __emule_local_l1_to_ptr(src_local_l1_addr);
    __emule_multicast_write(dst_mcast_noc_addr, src, size);
}

inline void noc_async_write_multicast_loopback_src(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    noc_async_write_multicast(src_local_l1_addr, dst_mcast_noc_addr, size, num_dests, linked, noc);
}

// ---- Barriers ----

// Clears the pending-NOC-reads counter so subsequent publication points
// (cb_push_back, semaphore inc) won't flag a missing-barrier race. In emule
// the underlying reads are synchronous memcpys, so there's nothing to actually
// wait for — but the contract must still be observed for parity with silicon.
inline void noc_async_read_barrier() { __emule_pending_noc_reads = 0; }
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

#ifndef __EMULE_GET_SEMAPHORE_DEFINED
#define __EMULE_GET_SEMAPHORE_DEFINED
inline uint32_t get_semaphore(uint32_t semaphore_id) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    return l1_base + EMULE_SEM_BASE + semaphore_id * EMULE_SEM_ALIGN;
}
#endif

// Atomic helpers for semaphore operations.
// volatile reads are unreliable at -O3; use std::atomic for cross-thread visibility.
inline std::atomic<uint32_t>* __emule_sem_atomic(volatile tt_l1_ptr uint32_t* sem_addr) {
    return reinterpret_cast<std::atomic<uint32_t>*>(const_cast<uint32_t*>(sem_addr));
}

// Set semaphore value (local L1 store).
// addr is a uint32_t L1 address (truncated host pointer).
inline void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    __emule_sem_atomic(sem_addr)->store(val, std::memory_order_release);
}

// Wait for semaphore to reach expected value (spin with exponential backoff + hang detection).
inline void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val) {
    auto* atom = __emule_sem_atomic(sem_addr);
    if (__emule_debug_multicast()) {
        fprintf(stderr, "EMULE DBG: noc_semaphore_wait(%p, %u) current=%u "
                "[from logical (%u,%u)]\n",
                (void*)sem_addr, val, atom->load(std::memory_order_acquire),
                __emule_logical_x, __emule_logical_y);
    }
    uint64_t spins = 0;
    while (atom->load(std::memory_order_acquire) != val) {
        if (spins < 64) {
            // Busy-spin for fast wakeup
        } else if (spins < 1024) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (++spins > 10'000'000ULL) {
            fprintf(stderr, "EMULE HANG: noc_semaphore_wait(%p, %u) stuck at %u after %llu spins "
                    "[phys (%u,%u) logical (%u,%u)]\n",
                    (void*)sem_addr, val, atom->load(std::memory_order_relaxed), (unsigned long long)spins,
                    my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
            std::abort();
        }
    }
}

// Wait for semaphore to reach at least min_val (spin with exponential backoff + hang detection).
inline void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t min_val) {
    auto* atom = __emule_sem_atomic(sem_addr);
    if (__emule_debug_multicast()) {
        fprintf(stderr, "EMULE DBG: noc_semaphore_wait_min(%p, %u) current=%u "
                "[from logical (%u,%u)]\n",
                (void*)sem_addr, min_val, atom->load(std::memory_order_acquire),
                __emule_logical_x, __emule_logical_y);
    }
    uint64_t spins = 0;
    while (atom->load(std::memory_order_acquire) < min_val) {
        if (spins < 64) {
            // Busy-spin for fast wakeup
        } else if (spins < 1024) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        if (++spins > 10'000'000ULL) {
            fprintf(stderr, "EMULE HANG: noc_semaphore_wait_min(%p, %u) stuck at %u after %llu spins "
                    "[phys (%u,%u) logical (%u,%u)]\n",
                    (void*)sem_addr, min_val, atom->load(std::memory_order_relaxed), (unsigned long long)spins,
                    my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
            std::abort();
        }
    }
}

// Atomically increment a remote semaphore.
// noc_addr is a 64-bit encoded NOC address pointing to the semaphore.
inline void noc_semaphore_inc(uint64_t noc_addr, uint32_t incr, uint8_t noc = 0) {
    uint32_t noc_x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    uint32_t noc_y = (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    uint32_t offset = static_cast<uint32_t>(noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
    uint8_t* ptr = __emule_resolve_noc_addr(noc_addr);
    if (__emule_debug_multicast()) {
        fprintf(stderr, "EMULE DBG: noc_semaphore_inc target_core=(%u,%u) offset=0x%x incr=%u resolved=%p "
                "[from logical (%u,%u)]\n",
                noc_x, noc_y, offset, incr, (void*)ptr,
                __emule_logical_x, __emule_logical_y);
    }
    if (ptr) {
        __atomic_fetch_add(reinterpret_cast<uint32_t*>(ptr), incr, __ATOMIC_SEQ_CST);
    } else {
        fprintf(stderr, "EMULE WARN: noc_semaphore_inc failed to resolve addr 0x%llx "
                "(target core (%u,%u) offset 0x%x) [from phys (%u,%u) logical (%u,%u)]\n",
                (unsigned long long)noc_addr, noc_x, noc_y, offset,
                my_x[0], my_y[0], __emule_logical_x, __emule_logical_y);
    }
}

// Multicast a semaphore value to multiple cores.
inline void noc_semaphore_set_multicast(
    uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr,
    uint32_t num_dests, bool linked = false, uint8_t noc = 0) {
    if (__emule_debug_multicast()) {
        uint32_t x_end   = (dst_mcast_noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t y_end   = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t x_start = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t y_start = (dst_mcast_noc_addr >> (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
        uint32_t off     = static_cast<uint32_t>(dst_mcast_noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
        uint32_t sem_val;
        std::memcpy(&sem_val, __emule_local_l1_to_ptr(src_local_l1_addr), sizeof(uint32_t));
        fprintf(stderr, "EMULE DBG: noc_semaphore_set_multicast (%u,%u)->(%u,%u) offset=0x%x val=%u num_dests=%u "
                "[from logical (%u,%u)]\n",
                x_start, y_start, x_end, y_end, off, sem_val, num_dests,
                __emule_logical_x, __emule_logical_y);
    }
    uint8_t* src = __emule_local_l1_to_ptr(src_local_l1_addr);
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

// Guarded: llk_defs.h provides a dummy version for compute kernels.
// Dataflow kernels should use this one, which reads real CB state.
#ifndef __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
#define __EMULE_GET_LOCAL_CB_INTERFACE_DEFINED
inline CBInterface get_local_cb_interface(uint32_t cb_id) {
    return CBInterface{__emule_cbs[cb_id].page_size};
}
#endif

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

// ---- Barriers (DRAM) ----

inline void dram_barrier() {}  // No-op in emulation

// ---- NOC inline direct-write (32-bit value to remote L1) ----

// Thread-local state for the set_state / with_state API pair.
struct __emule_dw_state {
    uint64_t addr = 0;
    uint32_t val = 0;
};
inline thread_local __emule_dw_state __emule_dw_st;

// Apply byte-enable mask and write 32-bit value to a resolved host pointer.
inline void __emule_dw_write_be(uint8_t* dst, uint32_t val, uint8_t be) {
    if (be == 0xF) {
        std::memcpy(dst, &val, 4);
    } else {
        uint32_t existing;
        std::memcpy(&existing, dst, 4);
        uint8_t* vb = reinterpret_cast<uint8_t*>(&val);
        uint8_t* eb = reinterpret_cast<uint8_t*>(&existing);
        for (int i = 0; i < 4; i++) {
            if (be & (1 << i)) eb[i] = vb[i];
        }
        std::memcpy(dst, &existing, 4);
    }
}

// Unicast inline direct write — write a 32-bit value to a remote L1 address.
template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true>
inline void noc_inline_dw_write(
    uint64_t addr,
    uint32_t val,
    uint8_t be = 0xF,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t customized_src_addr = 0) {
    uint8_t* dst = __emule_resolve_noc_addr(addr);
    if (dst) {
        __emule_dw_write_be(dst, val, be);
    }
}

// Multicast inline direct write — write a 32-bit value to a rectangular core range.
template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true>
inline void noc_inline_mcast_dw_write(
    uint64_t addr,
    uint32_t val,
    uint8_t be = 0xF,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_MULTICAST_WRITE_VC,
    uint32_t customized_src_addr = 0,
    uint32_t num_dest = 1) {
    // Decode the multicast rectangle from the encoded address.
    constexpr uint32_t node_mask = (1u << NOC_ADDR_NODE_ID_BITS) - 1;
    constexpr uint64_t local_mask = (1ULL << NOC_ADDR_LOCAL_BITS) - 1;
    uint32_t x_end   = (addr >> NOC_ADDR_LOCAL_BITS) & node_mask;
    uint32_t y_end   = (addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & node_mask;
    uint32_t x_start = (addr >> (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) & node_mask;
    uint32_t y_start = (addr >> (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) & node_mask;
    uint32_t l1_off  = static_cast<uint32_t>(addr & local_mask);

    // Iterate over the rectangle and write to each core.
    uint32_t lo_x = (x_start < x_end) ? x_start : x_end;
    uint32_t hi_x = (x_start < x_end) ? x_end : x_start;
    uint32_t lo_y = (y_start < y_end) ? y_start : y_end;
    uint32_t hi_y = (y_start < y_end) ? y_end : y_start;
    for (uint32_t y = lo_y; y <= hi_y; y++) {
        for (uint32_t x = lo_x; x <= hi_x; x++) {
            // Build unicast NOC addr directly (l1_off is already a raw offset).
            uint64_t unicast_addr =
                (uint64_t(y & 0x3F) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
                (uint64_t(x & 0x3F) << NOC_ADDR_LOCAL_BITS) |
                uint64_t(l1_off);
            uint8_t* dst = __emule_resolve_noc_addr(unicast_addr);
            if (dst) {
                __emule_dw_write_be(dst, val, be);
            }
        }
    }
}

// Set state for stateful inline direct write.
template <bool posted = false, bool set_val = false>
inline void noc_inline_dw_write_set_state(
    uint64_t addr,
    uint32_t val = 0,
    uint8_t be = 0xF,
    uint8_t cmd_buf = write_at_cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    __emule_dw_st.addr = addr;
    if constexpr (set_val) {
        __emule_dw_st.val = val;
    }
}

// Issue write using previously set state, optionally updating address/value.
template <
    bool update_addr_lo = false,
    bool update_counter = true,
    bool posted = false,
    bool update_addr_hi = false,
    bool update_val = false,
    InlineWriteDst dst_type = InlineWriteDst::DEFAULT>
inline void noc_inline_dw_write_with_state(
    uint32_t val, uint32_t addr = 0, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index) {
    if constexpr (update_addr_lo) {
        // Replace the lower 32 bits of the address (the L1 offset).
        __emule_dw_st.addr = (__emule_dw_st.addr & ~uint64_t(0xFFFFFFFF)) | addr;
    }
    if constexpr (update_val) {
        __emule_dw_st.val = val;
    }
    uint8_t* dst = __emule_resolve_noc_addr(__emule_dw_st.addr);
    if (dst) {
        __emule_dw_write_be(dst, __emule_dw_st.val, 0xF);
    }
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
