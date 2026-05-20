// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

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

// Stream manipulators referenced by upstream kernel debug prints.  All discarded
// by the sink, but they need to be valid expressions for the DPRINT chain to
// parse.  Match upstream tt_metal/hw/inc/api/debug/dprint.h shape.
struct HEX {};
struct DEC {};
struct OCT {};
struct BIN {};
struct FIXED {};
struct DEFAULTFLOAT {};
struct SETW { int v; constexpr SETW(int v_) : v(v_) {} };
struct SETPRECISION { int v; constexpr SETPRECISION(int v_) : v(v_) {} };

// Support for DEVICE_PRINT.
#include "device_print.h"
