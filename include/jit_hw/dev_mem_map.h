// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// emule shim for the per-arch firmware `dev_mem_map.h`.  Two purposes:
//
// 1. `MEM_L1_UNCACHED_BASE` — silicon kernels cast L1 offsets to pointers via
//    `(void*)(addr + MEM_L1_UNCACHED_BASE)`.  On silicon, that constant is the
//    fixed L1 base.  In emule, L1 is heap memory mmap'd per-core, so we
//    redirect to the per-thread `__emule_bridge_l1` pointer that the runner
//    sets before each kernel launch.
//
// 2. `MEM_ZEROS_BASE` / `MEM_ZEROS_SIZE` — the firmware-reserved L1 zeros
//    region kernels NOC-read for cheap zero-fill (see ttnn's
//    `kernel_lib/l1_helpers.hpp::zero_tile`).  Per-arch dispatch matches the
//    runner's ARCH_* JIT defines (see `build_kernel_defines` in
//    `tt_metal/impl/emulation/emulated_program_runner.cpp`).  Values mirror
//    upstream's `(MEM_MAILBOX_END + 31) & ~31`:
//      WH: tt_metal/hw/inc/internal/tt-1xx/wormhole/dev_mem_map.h
//      BH: tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h
//      Q:  tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h

extern thread_local uint8_t* __emule_bridge_l1;
#define MEM_L1_UNCACHED_BASE ((uintptr_t)__emule_bridge_l1)

// MEM_ZEROS_SIZE is typed `int` to match upstream's `#define MEM_ZEROS_SIZE 512`
// (defaults to int).  `std::min(bytes, MEM_ZEROS_SIZE)` where `bytes` is int
// would report a deduction ambiguity if this were `uint32_t`.
#ifndef MEM_ZEROS_SIZE
constexpr int MEM_ZEROS_SIZE = 512;
#endif

#ifndef MEM_ZEROS_BASE
#  if defined(ARCH_BLACKHOLE)
constexpr uint32_t MEM_ZEROS_BASE = 0x32E0;
#  elif defined(ARCH_QUASAR)
constexpr uint32_t MEM_ZEROS_BASE = 0xE180;
#  else  // ARCH_WORMHOLE (or unspecified)
constexpr uint32_t MEM_ZEROS_BASE = 0x3280;
#  endif
#endif

// Quasar-only: TRISC (compute) cores per Neo. Mirrors the real
// tt-2xx/quasar/dev_mem_map.h; compute kernels index per-thread state with it.
#ifndef NUM_TRISC_CORES
#  if defined(ARCH_QUASAR)
#    define NUM_TRISC_CORES 4
#  endif
#endif
