#pragma once
// Stub debug print for JIT-compiled kernels.
// On the real device, DPRINT streams to a ring buffer read by the host.
// In the emulator, we discard all output (sink object).
//
// NOTE: hw/inc/api/debug/dprint.h has a different DPRINT definition ((void)0 &&).
// Both guard with #ifndef DPRINT, so whichever is included first wins.
// This JIT version should always be included first by JIT-compiled kernels.

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
