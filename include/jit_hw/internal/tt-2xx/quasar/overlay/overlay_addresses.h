#pragma once
#include <cstdint>

// Emulation stub for Quasar overlay addresses.
// L2_FLUSH_ADDR: On real hardware, writing to this register flushes an L2 cache line.
// In emulation, L1 is coherent host memory — the flush is a no-op.
// We allocate a scratch variable so that writes to L2_FLUSH_ADDR don't segfault.
namespace {
volatile uint64_t __emule_l2_flush_sink = 0;
}
#define L2_FLUSH_ADDR ((uint64_t)(uintptr_t)&__emule_l2_flush_sink)
