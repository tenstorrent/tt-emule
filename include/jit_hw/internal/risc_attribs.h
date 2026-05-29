// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// On the real RISC-V target these control memory qualifiers and inlining.
// In the JIT emulation host build, they are no-ops.
#include <cstddef>
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

// L2 cache flush primitives. On Quasar RISC-V cores these MMIO-write the address
// to L2_FLUSH_ADDR to push dirty L1 cache lines out to TL1. The inline-asm bodies
// in upstream's risc_common.h are RISC-V-specific (`fence` mnemonic, mmio register
// pointer) and won't compile on the x86 JIT host. Emule has no L2 cache — kernel
// memory is plain host memory and any write is already coherent — so these are
// no-op stubs.
inline void flush_l2_cache_line(uintptr_t /*addr*/) {}
inline void flush_l2_cache_range(uintptr_t /*start_addr*/, std::size_t /*size*/) {}
