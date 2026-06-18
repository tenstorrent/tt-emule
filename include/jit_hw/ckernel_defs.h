// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Minimal ckernel_defs stub for JIT emulation.

#include <cstdint>

// Tile/face geometry, in `namespace ckernel` as on silicon
// (tt_llk_<arch>/common/inc/ckernel_defs.h). Single definition site so both
// dataflow kernels (which include only this) and compute kernels (via common.h)
// share it without a duplicate definition. TILE_C_DIM/TILE_R_DIM are omitted:
// llk_types.h #defines TILE_C_DIM, so a constexpr of that name wouldn't parse.
namespace ckernel {
constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH  = 16;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH  = 32;
}  // namespace ckernel
