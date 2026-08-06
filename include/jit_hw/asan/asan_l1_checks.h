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
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self (fiber ctx)

#include "jit_hw/emule_cb_state.h"          // __emule_cbs (CBSyncState*)
#include "jit_hw/internal/emule_cb_ptr.h"   // __emule_cb_wr_page / __emule_cb_rd_page
#include "jit_hw/asan/emule_asan.h"         // __emule_asan_panic

// The per-launch state these checks consume now lives in the per-fiber context
// (__emule_self->san, EmuleSanitizerState in internal/emule_thread_ctx.h) rather
// than worker-`thread_local`s, so it travels with a yielding fiber instead of being
// clobbered by a co-scheduled one. The host arms it per launch
// (set_sanitizer_thread_locals). See docs/ASAN.md + tt-emule #241.

// Address->host-pointer translation (the tail every chokepoint path ends in).
// L1 offset model: l1_addr is a 0-based firmware L1 offset (< l1_size); rebase
// onto this fiber's core L1. Cross-core / cross-chip access never reaches here —
// it goes through __emule_resolve_noc_addr. See docs/l1-emulation.md.
inline uint8_t* __emule_l1_translate(uint32_t l1_addr) {
#ifndef NDEBUG
    // Surface an un-migrated site (a value still carrying an absolute/aliased
    // address rather than a 0-based offset) as a named panic instead of a wild
    // store. Skipped only while l1_size is unset (0).
    if (__emule_self->l1_size != 0 && l1_addr >= __emule_self->l1_size) {
        __emule_asan_panic("[EMULE_L1] L1 offset 0x%x out of range (l1_size 0x%x)\n",
                           l1_addr, __emule_self->l1_size);
    }
#endif
    return __emule_self->bridge_l1 + l1_addr;
}

// Check 1 — Illegal Semaphore Access: raw access into the reserved sem region.
inline void __emule_asan_check_semaphore(uint32_t l1_addr) {
    if (__emule_self->san.sem_l1_range_end > 0 &&
        l1_addr >= __emule_self->san.sem_l1_range_start && l1_addr < __emule_self->san.sem_l1_range_end) {
        __emule_asan_panic(
                "[ASAN ERROR] Illegal Semaphore Access: Offset 0x%x is inside the reserved Semaphore region [0x%x, 0x%x)\n",
                l1_addr, __emule_self->san.sem_l1_range_start, __emule_self->san.sem_l1_range_end);
    }
}

// Check 2 — CB resolution + Boundary Violation. If `l1_addr` lands in a CB's
// backing memory, run the boundary check (when strict + an active window) and
// set `out` to the translated pointer, returning true. Matched before OOB: CB
// memory isn't in LiveL1Ranges and would otherwise look out-of-bounds.
inline bool __emule_asan_cb_resolve(uint32_t l1_addr, uint8_t*& out) {
    if (__emule_self->cbs == nullptr) return false;
    for (uint32_t cb_id = 0; cb_id < __EMULE_CTX_MAX_CBS; ++cb_id) {
        auto& cb = __emule_self->cbs[cb_id];
        if (cb.num_pages == 0) continue;
        // Offset model: l1_addr is a 0-based offset; cb.base is a host pointer,
        // so rebase it into offset space for the membership/boundary comparison.
        uint32_t cb_start = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(cb.base) -
                                                  reinterpret_cast<uintptr_t>(__emule_self->bridge_l1));
        uint32_t cb_size = cb.num_pages * cb.page_size;
        if (l1_addr < cb_start || l1_addr >= cb_start + cb_size) continue;
        // globally_allocated CBs reserve only nominally → exempt. See ASAN.md (CB Boundary Violation).
        if (__emule_self->san.cb_boundary_strict && !cb.globally_allocated) {
            uint32_t access_page = (l1_addr - cb_start) / cb.page_size;
            uint32_t write_idx = __emule_cb_wr_page(cb_id);
            uint32_t read_idx  = __emule_cb_rd_page(cb_id);
            uint32_t reserved = __emule_self->san.cb_reserved_pages[cb_id];
            uint32_t waited   = __emule_self->san.cb_waited_pages[cb_id];
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
    if (__emule_self->san.l1_tensor_ranges == nullptr || l1_off < __emule_self->san.l1_unreserved_base) return;
    bool in_tensor = false;
    uint64_t matched_packed = 0;
    for (uint32_t i = 0; i < __emule_self->san.l1_tensor_ranges_count; ++i) {
        uint64_t packed = __emule_self->san.l1_tensor_ranges[i];
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
        for (uint32_t i = 0; i < __emule_self->san.l1_host_ranges_count; ++i) {
            uint64_t packed = __emule_self->san.l1_host_ranges[i];
            if (l1_off >= static_cast<uint32_t>(packed >> 32) && l1_off < static_cast<uint32_t>(packed)) {
                return;
            }
        }
        __emule_asan_panic(
                "[ASAN ERROR] Out-of-Bounds Write: Attempted to access address 0x%x which is not part of any allocated tensor\n",
                l1_off);
    }
    // Record the resolved extent for the runner's Object Intent check. The log lives in
    // this fiber's ctx (__emule_self), so it needs no thread-local handshake and survives
    // a fiber swap. Inactive unless Object Intent is armed for this kernel.
    if (__emule_self->san_resolved_active) {
        uint32_t cur = __emule_self->san_resolved_count;
        bool already = false;
        for (uint32_t i = 0; i < cur; ++i) {
            if (__emule_self->san_resolved_log[i] == matched_packed) {
                already = true;
                break;
            }
        }
        if (!already && cur < __EMULE_SAN_RESOLVED_CAP) {
            __emule_self->san_resolved_log[cur] = matched_packed;
            __emule_self->san_resolved_count = cur + 1;
        }
    }
}

// Check 4 — Tensor Padding Violation: write into a buffer's [logical, physical) pad.
inline void __emule_asan_check_padding(uint32_t l1_off) {
    if (__emule_self->san.l1_padding_ranges == nullptr) return;
    for (uint32_t i = 0; i < __emule_self->san.l1_padding_ranges_count; ++i) {
        uint64_t packed = __emule_self->san.l1_padding_ranges[i];
        uint32_t logical_end = static_cast<uint32_t>(packed >> 32);
        uint32_t physical_end = static_cast<uint32_t>(packed);
        if (l1_off >= logical_end && l1_off < physical_end) {
            __emule_asan_panic(
                    "[ASAN ERROR] Tensor Padding Violation: Attempted to write to a padded memory region at address 0x%x (logical_end=0x%x, physical_end=0x%x)\n",
                    l1_off, logical_end, physical_end);
        }
    }
}
