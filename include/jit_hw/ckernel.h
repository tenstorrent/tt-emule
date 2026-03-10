#pragma once
// Minimal ckernel stub for JIT emulation.
#include <cstdint>

namespace ckernel {
enum class ThreadId : uint8_t { Unpack = 0, Math = 1, Pack = 2 };
inline void mailbox_write(ThreadId, uint32_t) {}
} // namespace ckernel
