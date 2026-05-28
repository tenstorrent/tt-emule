// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/tilize.h`. Provides tilize_init / tilize_block /
// tilize_uninit and fast_tilize_* stubs. Intercepts the include path so the
// JIT compile doesn't fall through to upstream's version (which references
// unstubbed LLK APIs).
#include <cstdint>

#include "jit_hw/internal/llk_state.h"
// compute_kernel_hw_startup is called directly by kernels that include only
// this header.
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"

inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb, uint32_t /*call_line*/) {
    tilize_init(icb, block, ocb);
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}

// tilize_block: copies tiles from input CB to output CB in tilized layout.
// In emulation the tile data is already in-order; just forward the pages.
inline void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb) {}

inline void tilize_uninit(uint32_t /*icb*/, uint32_t /*ocb*/ = 0) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = false;
}

// fast_tilize_* stubs (Blackhole-specific hardware path; no-ops in emulation).
inline void fast_tilize_init(uint32_t /*icb*/, uint32_t /*block*/, uint32_t /*ocb*/) {}
inline void fast_tilize_init_skip_remap(uint32_t /*icb*/, uint32_t /*block*/, uint32_t /*ocb*/) {}
inline void fast_tilize_block(uint32_t /*icb*/, uint32_t /*block*/, uint32_t /*ocb*/) {}
inline void fast_tilize_uninit(uint32_t /*icb*/, uint32_t /*ocb*/, uint32_t /*block*/ = 0) {}
