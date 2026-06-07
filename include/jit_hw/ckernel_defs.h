// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Minimal ckernel_defs stub for JIT emulation.
//
// Dataflow kernels (e.g. padded_slice writer_unary_sharded_padded_tiled.cpp)
// include ckernel_defs.h purely for the tile/face dimension constants. The real
// header (tt_llk_*/common/inc/ckernel_defs.h) also pulls in LLK enums/intrinsics
// not buildable on a dataflow RISC, so emule mirrors just the dimensional block.

#include <cstdint>

namespace ckernel {

constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH  = 16;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH  = 32;

constexpr std::uint32_t FACE_R_DIM = FACE_HEIGHT;
constexpr std::uint32_t FACE_C_DIM = FACE_WIDTH;

constexpr std::uint32_t TILE_R_DIM = TILE_HEIGHT;
constexpr std::uint32_t TILE_C_DIM = TILE_WIDTH;

constexpr std::uint32_t TILE_NUM_FACES = ((TILE_R_DIM * TILE_C_DIM) / (FACE_R_DIM * FACE_C_DIM));
constexpr std::uint32_t FACE_SIZE = FACE_R_DIM * FACE_C_DIM;

}  // namespace ckernel
