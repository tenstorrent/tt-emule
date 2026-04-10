#pragma once
// Stub debug print for JIT-compiled kernels.
// On the real device, DPRINT streams to a ring buffer read by the host.
// In the emulator, we send to stdout (or discard).

#include <cstdio>

namespace tt_emule_jit {
struct DPrintSink {
    template<typename T>
    const DPrintSink& operator<<(const T&) const { return *this; }
};
inline DPrintSink make_dprint() { return DPrintSink{}; }
} // namespace tt_emule_jit

#define DPRINT tt_emule_jit::make_dprint()
#define ENDL() '\n'

// Support for DEVICE_PRINT.
#include "device_print.h"
