// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The per-access L1 sanitizer checks, lifted out of the __emule_local_l1_to_ptr
// chokepoint (internal/emule_l1_to_ptr.h) so the chokepoint stays a thin
// translate-and-dispatch. Each helper either passes through or calls
// __emule_asan_panic(); none is reached unless TT_METAL_EMULE_ASAN is on (the
// chokepoint gates them behind __emule_asan_enabled()). See docs/ASAN.md
// "Kernel-side checks" for the order, the slot-mask rationale, and the accepted
// sharded-tensor false negative.

#include <cstdint>

#include "jit_hw/emule_cb_state.h"          // __emule_cbs (CBSyncState*)
#include "jit_hw/internal/emule_cb_ptr.h"   // __emule_cb_wr_page / __emule_cb_rd_page
#include "jit_hw/asan/emule_asan.h"         // __emule_asan_panic

// L1 bridge base + sanitizer thread-locals these checks (and the chokepoint's
// plain translation) consume. Defined in the runner (emulated_program_runner.cpp
// / kernel_runner.cpp) and threaded in per launch; see docs/ASAN.md
// "Thread-local handshake".
extern thread_local uint8_t* __emule_bridge_l1;
extern thread_local uint32_t __emule_sem_l1_range_start;
extern thread_local uint32_t __emule_sem_l1_range_end;
extern thread_local uint32_t __emule_l1_unreserved_base;
extern thread_local const uint64_t* __emule_l1_tensor_ranges;
extern thread_local uint32_t __emule_l1_tensor_ranges_count;
extern thread_local const uint64_t* __emule_l1_padding_ranges;
extern thread_local uint32_t __emule_l1_padding_ranges_count;
extern thread_local const uint64_t* __emule_l1_host_ranges;
extern thread_local uint32_t __emule_l1_host_ranges_count;
extern thread_local uint64_t* __emule_l1_resolved_ranges;
extern thread_local uint32_t* __emule_l1_resolved_ranges_count;
extern thread_local uint32_t __emule_l1_resolved_ranges_capacity;
extern thread_local uint32_t __emule_cb_reserved_pages[32];
extern thread_local uint32_t __emule_cb_waited_pages[32];
extern thread_local bool __emule_cb_boundary_strict;

// Plain address->host-pointer translation (the tail every chokepoint path ends
// in). A firmware-style offset is rebased onto __emule_bridge_l1; an already-
// absolute host pointer (l1_alloc / CB / DFB) passes through.
inline uint8_t* __emule_l1_translate(uint32_t l1_addr) {
    uint32_t l1_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    if (l1_addr >= l1_base) {
        return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(l1_addr));
    }
    return __emule_bridge_l1 + l1_addr;
}

// Check 1 — Illegal Semaphore Access: raw access into the reserved sem region.
inline void __emule_asan_check_semaphore(uint32_t l1_addr) {
    if (__emule_sem_l1_range_end > 0 &&
        l1_addr >= __emule_sem_l1_range_start && l1_addr < __emule_sem_l1_range_end) {
        __emule_asan_panic(
                "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
                l1_addr, __emule_sem_l1_range_start, __emule_sem_l1_range_end);
    }
}

// Check 2 — CB resolution + Boundary Violation. If `l1_addr` lands in a CB's
// backing memory, run the boundary check (when strict + an active window) and
// set `out` to the translated pointer, returning true. Matched before OOB: CB
// memory isn't in LiveL1Ranges and would otherwise look out-of-bounds.
inline bool __emule_asan_cb_resolve(uint32_t l1_addr, uint8_t*& out) {
    if (__emule_cbs == nullptr) return false;
    for (uint32_t cb_id = 0; cb_id < 32; ++cb_id) {
        auto& cb = __emule_cbs[cb_id];
        if (cb.num_pages == 0) continue;
        uint32_t cb_start = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(cb.base));
        uint32_t cb_size = cb.num_pages * cb.page_size;
        if (l1_addr < cb_start || l1_addr >= cb_start + cb_size) continue;
        // globally_allocated CBs reserve only nominally → exempt. See ASAN.md (CB Boundary Violation).
        if (__emule_cb_boundary_strict && !cb.globally_allocated) {
            uint32_t access_page = (l1_addr - cb_start) / cb.page_size;
            uint32_t write_idx = __emule_cb_wr_page(cb_id);
            uint32_t read_idx  = __emule_cb_rd_page(cb_id);
            uint32_t reserved = __emule_cb_reserved_pages[cb_id];
            uint32_t waited   = __emule_cb_waited_pages[cb_id];
            uint32_t write_dist = (access_page + cb.num_pages - write_idx) % cb.num_pages;
            uint32_t read_dist  = (access_page + cb.num_pages - read_idx)  % cb.num_pages;
            uint32_t produced  = (write_idx + cb.num_pages - read_idx) % cb.num_pages;
            // Fire only outside an ACTIVE window AND outside the produced region
            // [read_idx, write_idx) (reuse of valid data is legal).
            if ((reserved > 0 || waited > 0) &&
                !(write_dist < reserved) && !(read_dist < waited) && !(read_dist < produced)) {
                __emule_asan_panic(
                        "[ASAN ERROR] CB Boundary Violation: Attempted to access CB %u at offset 0x%x "
                        "(byte %u of %u, page %u of %u). "
                        "Write window: write_idx=%u, %u page(s) reserved. "
                        "Read window: read_idx=%u, %u page(s) waited.\n",
                        cb_id, l1_addr, l1_addr - cb_start, cb_size,
                        access_page, cb.num_pages,
                        write_idx, reserved,
                        read_idx, waited);
            }
        }
        out = __emule_l1_translate(l1_addr);
        return true;
    }
    return false;
}

// Check 3 — Out-of-Bounds Write (L1). For offsets at/above the unreserved base,
// require a live-tensor extent; also record the matched extent for the runner's
// Object Intent check.
inline void __emule_asan_check_oob_tensor(uint32_t l1_off) {
    if (__emule_l1_tensor_ranges == nullptr || l1_off < __emule_l1_unreserved_base) return;
    bool in_tensor = false;
    uint64_t matched_packed = 0;
    for (uint32_t i = 0; i < __emule_l1_tensor_ranges_count; ++i) {
        uint64_t packed = __emule_l1_tensor_ranges[i];
        uint32_t r_start = static_cast<uint32_t>(packed >> 32);
        uint32_t r_end = static_cast<uint32_t>(packed);
        if (l1_off >= r_start && l1_off < r_end) {
            in_tensor = true;
            matched_packed = packed;
            break;
        }
    }
    if (!in_tensor) {
        // Raw L1 the host designated via WriteToDeviceL1/ReadFromDeviceL1 is valid
        // data outside the Buffer allocator; accept it but DON'T record it as a
        // resolved tensor (it must stay invisible to Object Intent)
        for (uint32_t i = 0; i < __emule_l1_host_ranges_count; ++i) {
            uint64_t packed = __emule_l1_host_ranges[i];
            if (l1_off >= static_cast<uint32_t>(packed >> 32) && l1_off < static_cast<uint32_t>(packed)) {
                return;
            }
        }
        __emule_asan_panic(
                "[ASAN ERROR] Out-of-Bounds Write: Attempted to access address 0x%x which is not part of any allocated tensor\n",
                l1_off);
    }
    if (__emule_l1_resolved_ranges != nullptr && __emule_l1_resolved_ranges_count != nullptr) {
        uint32_t cur = *__emule_l1_resolved_ranges_count;
        bool already = false;
        for (uint32_t i = 0; i < cur; ++i) {
            if (__emule_l1_resolved_ranges[i] == matched_packed) {
                already = true;
                break;
            }
        }
        if (!already && cur < __emule_l1_resolved_ranges_capacity) {
            __emule_l1_resolved_ranges[cur] = matched_packed;
            *__emule_l1_resolved_ranges_count = cur + 1;
        }
    }
}

// Check 4 — Tensor Padding Violation: write into a buffer's [logical, physical) pad.
inline void __emule_asan_check_padding(uint32_t l1_off) {
    if (__emule_l1_padding_ranges == nullptr) return;
    for (uint32_t i = 0; i < __emule_l1_padding_ranges_count; ++i) {
        uint64_t packed = __emule_l1_padding_ranges[i];
        uint32_t logical_end = static_cast<uint32_t>(packed >> 32);
        uint32_t physical_end = static_cast<uint32_t>(packed);
        if (l1_off >= logical_end && l1_off < physical_end) {
            __emule_asan_panic(
                    "[ASAN ERROR] Tensor Padding Violation: Attempted to write to a padded memory region at address 0x%x (logical_end=0x%x, physical_end=0x%x)\n",
                    l1_off, logical_end, physical_end);
        }
    }
}
