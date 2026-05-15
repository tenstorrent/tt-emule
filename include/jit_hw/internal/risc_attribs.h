// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// On the real RISC-V target these control memory qualifiers and inlining.
// In the JIT emulation host build, they are no-ops.
#include <cstdint>

#define tt_l1_ptr
#define FORCE_INLINE inline __attribute__((always_inline))

// Direct-write destination type (used by noc_inline_dw_write family).
enum class InlineWriteDst : uint8_t { DEFAULT = 0, L1 = 1, REG = 2 };

// Command buffer index — unused in emulation, but kernels reference it.
constexpr uint32_t write_at_cmd_buf = 0;
