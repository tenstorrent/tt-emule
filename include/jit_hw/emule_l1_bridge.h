// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

// Per-thread host pointer to this core's L1 mmap region, set by the emulated
// program runner before each kernel launch.  Kernels that cast L1 offsets to
// pointers (e.g. `*(uint32_t*)(addr + MEM_L1_UNCACHED_BASE)`) resolve through
// this indirection into host memory.
//
// This is the only emule-specific symbol the mocked dev_mem_map.h used to
// provide; all other per-arch memory layout values come from upstream HAL or
// the SocDescriptor.
extern thread_local uint8_t* __emule_bridge_l1;
#define MEM_L1_UNCACHED_BASE ((uintptr_t)__emule_bridge_l1)
