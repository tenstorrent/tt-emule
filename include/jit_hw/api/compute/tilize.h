// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `api/compute/tilize.h`. Owns `tilize_init` /
// `tilize_init_short`. Intercepts the include path so the JIT compile
// doesn't fall through to upstream's version (which references unstubbed
// LLK APIs).
//
// Real tt-metal `api/compute/tilize.h` pulls in `api/compute/common.h` first
// to get ALWI/UNPACK/PACK/MATH macros. Downstream kernel-lib headers (e.g.
// ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp) assume ALWI is defined by the
// time they're parsed. Mirror that contract so emule users don't have to
// pre-include common.h themselves.
#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/compute/compute_kernel_hw_startup.h"
#include "jit_hw/internal/llk_state.h"
#include <cstdint>

inline void tilize_init(uint32_t, uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
inline void tilize_init_short(uint32_t, uint32_t) {
    __llk_unpack_is_tilize = true;
    __llk_pack_is_untilize = false;
}
