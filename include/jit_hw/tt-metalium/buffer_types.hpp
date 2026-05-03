// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Local mirror of tt_metal/api/tt-metalium/buffer_types.hpp, kept verbatim.
// This file exists only because the JIT include path does not (today) reach
// tt_metal/api/, so kernels that pull <tt-metalium/buffer_types.hpp>
// (e.g. fill_pad_writer.cpp via sharded_tensor_addr_gen.hpp) would otherwise
// fail to resolve. Drift risk is real but low: these enums are public-API
// values and rarely change.
//
// TODO: drop this file once emulated_program_runner.cpp's JIT command line
// adds tt_metal/api as a `-I` path.

#include <cstdint>

namespace tt::tt_metal {

enum class TensorMemoryLayout {
    INTERLEAVED = 0,
    HEIGHT_SHARDED = 2,
    WIDTH_SHARDED = 3,
    BLOCK_SHARDED = 4,
    ND_SHARDED = 5,
};

enum class ShardOrientation {
    ROW_MAJOR = 0,
    COL_MAJOR,
};

enum class ShardDistributionStrategy {
    ROUND_ROBIN_1D = 0,
    GRID_2D = 1,
};

enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};

}  // namespace tt::tt_metal
