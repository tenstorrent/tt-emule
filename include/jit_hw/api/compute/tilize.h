// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/tilize.h`. Owns `tilize_init` /
// `tilize_init_short`. Intercepts the include path so the JIT compile
// doesn't fall through to upstream's version (which references unstubbed
// LLK APIs).
#include <cstdint>

#include "jit_hw/internal/llk_state.h"

inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
