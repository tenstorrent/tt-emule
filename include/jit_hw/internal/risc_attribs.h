// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// On the real RISC-V target these control memory qualifiers and inlining.
// In the JIT emulation host build, they are no-ops.
#include <cstdint>

#define tt_l1_ptr
#define tt_reg_ptr
#define FORCE_INLINE inline __attribute__((always_inline))

// Direct-write destination type (used by noc_inline_dw_write family).
enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };

// Command buffer index — unused in emulation, but kernels reference it.
constexpr uint32_t write_at_cmd_buf = 0;

// RISC-V debug-register addresses used by upstream
// `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp::spin()`.
// spin() is a debug-only busy-wait that emule kernels never execute; the
// addresses just need to be defined for the upstream header to compile.
#ifndef RISCV_DEBUG_REG_WALL_CLOCK_L
#define RISCV_DEBUG_REG_WALL_CLOCK_L 0
#endif
#ifndef RISCV_DEBUG_REG_WALL_CLOCK_H
#define RISCV_DEBUG_REG_WALL_CLOCK_H 0
#endif
