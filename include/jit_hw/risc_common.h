// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub risc_common.h for JIT-compiled kernels.
// On real hardware this provides NOC coordinates, cache ops, register access.
// In the emulator most of these are no-ops.

#include <cstdint>
#include "jit_hw/internal/firmware_common.h"

// flush_l2_cache_line / flush_l2_cache_range live in jit_hw/internal/risc_attribs.h
// — that's the only shim file the JIT kernel chain transitively reaches via
// api/compute/reg_api.h and jit_kernel_stubs.hpp. This file is a near-empty
// shadow of upstream's risc_common.h that nothing currently includes; leaving
// it as a placeholder for future RISC-V-specific stubs that need a path-mirror
// shim (vs upstream's `#include "risc_common.h"`).
