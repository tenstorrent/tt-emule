#pragma once
// Stubs included at the TOP of every JIT wrapper before the kernel source.
// Provides everything a kernel needs that isn't in its own includes.

#include "internal/risc_attribs.h"
#include "api/compile_time_args.h"

#include <vector>
#include <cstdint>

// Forward declarations matching kernel_runner.cpp definitions.
// The main executable exports these with -rdynamic; the JIT .so resolves them
// at dlopen() time.
namespace tt_emule { class Core; class Device; }
extern thread_local std::vector<uint32_t> __rt_args;
extern thread_local tt_emule::Core*       __core;
extern thread_local tt_emule::Device*     __device;

// get_arg_addr(idx) — mirrors tt-metal's rta_l1_base-based implementation.
// Returns a pointer to the idx-th runtime arg (held in __rt_args).
// The kernel casts this to (uint32_t*) to read the value.
inline void* get_arg_addr(uint32_t idx) {
    return static_cast<void*>(&__rt_args[idx]);
}

// No-op assertion macro used in some kernel headers.
#define ASSERT(...) ((void)0)
