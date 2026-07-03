// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Per-core state (owned by tt_emule::Core; the thread ctx borrows a pointer).
//
// Split out of emule_thread_ctx.h into this minimal, dependency-free header so
// that tt_emule::Core (device.hpp) can embed a CoreState member WITHOUT pulling
// in the full per-thread ThreadCommonCtx/ComputeThreadCtx definitions. The umd
// TU (sw_emule_chip.cpp -> device.hpp) only needs CoreState; it must not have to
// parse ComputeThreadCtx, whose members reference kernel-only types (sfpi, ...).
//
// Holds per-core emule-only state (the logical coordinates). The NOC coordinate
// globals my_x/my_y are deliberately NOT here — they are silicon-named symbols
// read directly by unmodified upstream firmware/kernels, so they stay as
// runner-set thread_local globals.

#include <cstdint>

namespace tt_emule {
struct CoreState {
    uint32_t logical_x = 0;  // logical core x (D2M get_absolute_logical_x)
    uint32_t logical_y = 0;  // logical core y
};
}  // namespace tt_emule
