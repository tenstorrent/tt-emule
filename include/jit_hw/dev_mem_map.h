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

// L1 zeros block. Reads from MEM_ZEROS_BASE return zero because the L1 mmap
// is zero-initialized and 0x10000 sits above the bump allocator's high-water
// mark for any tt-emule-tested kernel.
#ifndef MEM_ZEROS_SIZE
#define MEM_ZEROS_SIZE 512
#endif
#ifndef MEM_ZEROS_BASE
#define MEM_ZEROS_BASE 0x10000
#endif
