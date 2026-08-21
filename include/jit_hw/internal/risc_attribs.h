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

// RISC-V debug-register addresses used by upstream `common.hpp::spin()` and
// `device_delay_spin.cpp`. Both busy-wait by re-reading this pair as a live 64-bit counter until
// it advances past a target, so the backing storage must genuinely tick (a frozen value spins
// forever) and must outlive this kernel's JIT-compiled .so (dlclose()'d once its Program is torn
// down). Backed by tt_emule::wall_clock() (include/tt_emule/wall_clock.hpp), compiled into the
// long-lived runtime — resolved here via extern "C" the same way other runtime hooks are (see
// __emule_fabric_teleport in __emule_fabric_stubs.h).
extern "C" uintptr_t __emule_wall_clock_lo_addr();
extern "C" uintptr_t __emule_wall_clock_hi_addr();
#define RISCV_DEBUG_REG_WALL_CLOCK_L __emule_wall_clock_lo_addr()
#define RISCV_DEBUG_REG_WALL_CLOCK_H __emule_wall_clock_hi_addr()

// L2 cache flush primitives. On Quasar RISC-V cores these MMIO-write the address
// to L2_FLUSH_ADDR to push dirty L1 cache lines out to TL1. The inline-asm bodies
// in upstream's risc_common.h are RISC-V-specific (`fence` mnemonic, mmio register
// pointer) and won't compile on the x86 JIT host. Emule has no L2 cache — kernel
// memory is plain host memory and any write is already coherent — so these are
// no-op stubs. Hosted here (not in jit_hw/risc_common.h next to upstream's path-
// mirror) because nothing in the JIT include chain pulls in that shadow file.
inline void flush_l2_cache_line(uintptr_t /*addr*/) {}
inline void flush_l2_cache_range(uintptr_t /*start_addr*/, std::size_t /*size*/) {}
