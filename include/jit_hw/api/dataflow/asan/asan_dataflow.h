// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOC Transfer Alignment sanitizer (gated by TT_METAL_EMULE_ASAN). Split out of
// dataflow_api.h so the NoC dataflow path isn't interleaved with sanitizer code.
// Included by dataflow_api.h AFTER the arch noc_parameters header and the
// __emule_noc_addr_is_dram declaration, so NOC_ADDR_LOCAL_BITS / ARCH_* are in
// scope. See docs/ASAN.md §10 for the per-side (not relative) alignment model.

#include <cstdint>
#include "jit_hw/asan/emule_asan.h"   // __emule_asan_enabled / __emule_asan_panic

// Resolved at dlopen from the runner (emulated_program_runner.cpp).
extern "C" bool __emule_noc_addr_is_dram(uint64_t noc_addr);

// Each endpoint is checked against its OWN memory-type alignment, not a relative
// "low bits must match" rule: L1 = 16B; DRAM read = 32B (WH) / 64B (BH); DRAM
// write = 16B. A DRAM read from a 32B-aligned source into a 16B- (but not 32B-)
// aligned L1 destination is legal even though their low bits differ.
inline void __emule_check_noc_read_alignment(uint64_t src_noc_addr, uint32_t dst_local_l1_addr) {
    if (!__emule_asan_enabled() || !__emule_asan_check_on(EMULE_ASAN_CHK_NOC_ALIGN)) return;
    uint32_t src_off = static_cast<uint32_t>(src_noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
    // L1 destination: 16-byte alignment.
    if ((dst_local_l1_addr & 0xF) != 0) {
        __emule_asan_panic(
                "[ASAN ERROR] NOC Transfer Alignment: L1 destination 0x%x must be 16-byte aligned\n",
                dst_local_l1_addr);
    }
    // Source alignment depends on its memory type.
    if (__emule_noc_addr_is_dram(src_noc_addr)) {
#ifdef ARCH_BLACKHOLE
        constexpr uint32_t src_mask = 0x3F;  // NOC_DRAM_READ_ALIGNMENT_BYTES = 64
#else
        constexpr uint32_t src_mask = 0x1F;  // NOC_DRAM_READ_ALIGNMENT_BYTES = 32
#endif
        if ((src_off & src_mask) != 0) {
            __emule_asan_panic(
                    "[ASAN ERROR] NOC Transfer Alignment: DRAM source 0x%x must be %u-byte aligned\n",
                    src_off, src_mask + 1);
        }
    } else if ((src_off & 0xF) != 0) {  // L1 source: 16-byte.
        __emule_asan_panic(
                "[ASAN ERROR] NOC Transfer Alignment: L1 source 0x%x must be 16-byte aligned\n",
                src_off);
    }
}

inline void __emule_check_noc_write_alignment(uint32_t src_local_l1_addr, uint64_t dst_noc_addr) {
    if (!__emule_asan_enabled() || !__emule_asan_check_on(EMULE_ASAN_CHK_NOC_ALIGN)) return;
    uint32_t dst_off = static_cast<uint32_t>(dst_noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
    // L1 source: 16-byte alignment.
    if ((src_local_l1_addr & 0xF) != 0) {
        __emule_asan_panic(
                "[ASAN ERROR] NOC Transfer Alignment: L1 source 0x%x must be 16-byte aligned\n",
                src_local_l1_addr);
    }
    // Destination: DRAM write and L1 are both 16-byte aligned (WH and BH).
    if ((dst_off & 0xF) != 0) {
        const char* dst_type = __emule_noc_addr_is_dram(dst_noc_addr) ? "DRAM" : "L1";
        __emule_asan_panic(
                "[ASAN ERROR] NOC Transfer Alignment: %s destination 0x%x must be 16-byte aligned\n",
                dst_type, dst_off);
    }
}
