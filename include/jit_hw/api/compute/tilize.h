// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/tilize.h"`.
// Used to be an empty include-only intercept of the upstream tt-metal
// `api/compute/tilize.h` (which references unstubbed LLK APIs). Per PR #21
// review feedback ("intercept-only shim" smell), this shim now OWNS
// `tilize_init` / `tilize_init_short`, split out of the original monolithic
// `llk_defs.h`.
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
