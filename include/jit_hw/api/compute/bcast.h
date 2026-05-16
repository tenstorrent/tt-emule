// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// tt-emule stub: bcast (no-op in emulation)
#pragma once

#include "api/compute/common.h"

namespace ckernel {

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

template <EltwiseBinaryType op_type, BroadcastType bcast_type>
inline void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {}

// `unary_bcast<BroadcastType>` — broadcasts a single tile across DST positions.
// Added by tt-mlir PR #7926 era D2M codegen for scalar/row/col broadcasts.
// On emule we don't track per-thread DST layout; treat as copy_tile.
template <BroadcastType BCAST_T>
inline void unary_bcast_init(uint32_t /*icb0*/, uint32_t /*icb1*/) {}

template <BroadcastType BCAST_T>
inline void unary_bcast(uint32_t icb, uint32_t in_tile_index, uint32_t idst) {
    copy_tile(icb, in_tile_index, idst);
}

}  // namespace ckernel
