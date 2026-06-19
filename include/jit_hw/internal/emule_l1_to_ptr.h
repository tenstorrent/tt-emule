// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Single source of truth for the kernel L1 access chokepoint. Every kernel L1
// access flows through __emule_local_l1_to_ptr, which converts a firmware-style
// L1 offset (or an absolute host pointer) to a dereferenceable host pointer.
// Previously this function was duplicated verbatim in jit_kernel_stubs.hpp and
// dataflow_api.h; both now include this header instead.
//
// The function stays thin: with ASAN off it does only the plain translation;
// with ASAN on it dispatches to the per-access check helpers in
// asan/asan_l1_checks.h. See docs/ASAN.md "Kernel-side checks".

#include <cstdint>

#include "jit_hw/asan/emule_asan.h"       // __emule_asan_enabled
#include "jit_hw/asan/asan_l1_checks.h"   // __emule_l1_translate + check helpers

inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr) {
    if (!__emule_asan_enabled()) {
        return __emule_l1_translate(l1_addr);
    }
    __emule_asan_check_semaphore(l1_addr);
    // CB memory isn't registered in LiveL1Ranges, so resolve it (and run the CB
    // boundary check) before the OOB-tensor check below.
    uint8_t* cb_ptr;
    if (__emule_asan_cb_resolve(l1_addr, cb_ptr)) {
        return cb_ptr;
    }
    // Mask to the within-slot offset (low 21 bits = 2 MB worker slot); live
    // ranges are stored buffer-relative. See docs/ASAN.md for the rationale.
    uint32_t l1_off = l1_addr & 0x1FFFFF;  // SLOT_MASK = 2 MB - 1
    __emule_asan_check_oob_tensor(l1_off);
    __emule_asan_check_padding(l1_off);
    return __emule_l1_translate(l1_addr);
}
