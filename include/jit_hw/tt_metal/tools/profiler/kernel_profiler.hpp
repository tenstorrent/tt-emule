// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Some kernels (e.g. line_reduce_scatter_minimal_async_reader.cpp) include this header via the
// "tt_metal/tools/profiler/kernel_profiler.hpp" spelling instead of the bare
// "tools/profiler/kernel_profiler.hpp" jit_hw/jit_kernel_stubs.hpp uses — a different #pragma-once
// identity, so without this file the real header (and its RISCV_DEBUG_REG_WALL_CLOCK_* reads,
// stubbed to address 0 under emule) gets pulled in too, redefining the no-op zone-scope macros
// back to their real (crashing) bodies. Mirror the fabric packet-header shims' pattern (see
// __emule_fabric_stubs.h) of shadowing every spelling a kernel might use.
#include "jit_hw/tools/profiler/kernel_profiler.hpp"
