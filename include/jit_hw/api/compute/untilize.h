// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/untilize.h`. Owns `untilize_init` /
// `untilize_init_short`. Intercepts the include path so the JIT compile
// doesn't fall through to upstream's version (which references unstubbed
// LLK APIs).
#include <cstdint>

#include "jit_hw/internal/llk_state.h"
// compute_kernel_hw_startup is called directly by kernels that include only
// this header (e.g. untilize.cpp).  Pull it in so the symbol is visible.
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"

inline void untilize_init(uint32_t, uint32_t = 0) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}
inline void untilize_init_short(uint32_t) {
    __llk_unpack_is_tilize = false;
    __llk_pack_is_untilize = true;
}
