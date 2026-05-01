#pragma once
// Stub risc_common.h for JIT-compiled kernels.
// On real hardware this provides NOC coordinates, cache ops, register access.
// In the emulator most of these are no-ops.

#include <cstdint>
#include "jit_hw/internal/firmware_common.h"

// Cache flush — no-op in emulation (no hardware cache).
inline void flush_l2_cache_line(uintptr_t) {}
