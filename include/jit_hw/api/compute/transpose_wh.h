// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: deprecated transpose_wh — forwards to transpose.h, mirroring
// upstream's transpose_wh.h → transpose.h migration (#23835).
#pragma once

#include "api/compute/transpose.h"

namespace ckernel {

inline void transpose_wh_init(uint32_t icb, uint32_t ocb = 0) { transpose_init(icb); }
inline void transpose_wh_init_short(uint32_t icb = 0) { transpose_init(icb); }
inline void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t dst_idx) {
    transpose_tile(icb, itile, dst_idx);
}

}  // namespace ckernel

using namespace ckernel;
