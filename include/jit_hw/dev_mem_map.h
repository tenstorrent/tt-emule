// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// In emulation, L1 is host memory (mmap'd via L1Pool).
// Kernels that cast L1 offsets to pointers (e.g. *(uint32_t*)(addr + MEM_L1_UNCACHED_BASE))
// need MEM_L1_UNCACHED_BASE to be the host base pointer of the core's L1 region,
// so that the sum is a valid host address.
// __emule_bridge_l1 is set per-thread before kernel launch in emulated_program_runner.
extern thread_local uint8_t* __emule_bridge_l1;
#define MEM_L1_UNCACHED_BASE ((uintptr_t)__emule_bridge_l1)

// Quasar core counts used by compute/DM kernels.
constexpr uint32_t NUM_TRISC_CORES = 4;
constexpr uint32_t NUM_DM_CORES = 8;

// L1 zeros block. Reads from MEM_ZEROS_BASE return MEM_ZEROS_SIZE bytes of
// zero. Mirrors upstream WH dev_mem_map.h: positioned immediately after the
// mailbox region (MEM_MAILBOX_END=0x3280 → 32-byte-aligned = 0x32A0), well
// below tt-metal's l1_unreserved_base. Core::reset_l1_bump() rezeros the
// region between program runs.
//
// Earlier emule choice (0xFFE00 = top of an assumed 1 MiB L1) silently
// corrupted user buffers once L1 grew to the real WH worker_l1_size of ~1.43
// MiB: user-allocator output buffers extended past 0xFFE00 and got the last
// 512 bytes zeroed mid-shard. See round12-closeout.md for repro.
//
// constexpr (not #define) so we don't accidentally shadow the upstream
// dev_mem_map.h declarations if jit_hw is ever linked into a target that also
// pulls in the real headers.
// MEM_ZEROS_SIZE is typed `int` to match upstream's `#define MEM_ZEROS_SIZE 512`
// (which defaults to int). full_kernel_common.hpp's `std::min(bytes, MEM_ZEROS_SIZE)`
// where `bytes` is `int` reports a deduction ambiguity if MEM_ZEROS_SIZE is
// `uint32_t` here.
#ifndef MEM_ZEROS_SIZE
constexpr int MEM_ZEROS_SIZE = 512;
#endif
#ifndef MEM_ZEROS_BASE
constexpr uint32_t MEM_ZEROS_BASE = 0x32A0;  // (MEM_MAILBOX_END + 31) & ~31, matches WH/BH/Quasar
#endif
