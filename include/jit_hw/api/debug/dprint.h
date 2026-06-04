// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub debug print for JIT-compiled kernels.
//
// On the real device, DPRINT serializes its arguments to a ring buffer that
// the host print server drains. In the emulator we discard everything.
//
// Upstream removed the legacy stream-style API (`DPRINT << ... << ENDL();`)
// in tt-metal #44930 in favor of fmt-style `DPRINT(format, args...)`. Mirror
// upstream's per-thread variants (UNPACK/MATH/PACK/DATA0/DATA1) so kernels
// that target a specific RISC compile under emule too.

#include "device_print.h"

#define DPRINT(format, ...)        DEVICE_PRINT(format, ##__VA_ARGS__)
#define DPRINT_UNPACK(format, ...) DEVICE_PRINT_UNPACK(format, ##__VA_ARGS__)
#define DPRINT_MATH(format, ...)   DEVICE_PRINT_MATH(format, ##__VA_ARGS__)
#define DPRINT_PACK(format, ...)   DEVICE_PRINT_PACK(format, ##__VA_ARGS__)
#define DPRINT_DATA0(format, ...)  DEVICE_PRINT_DATA0(format, ##__VA_ARGS__)
#define DPRINT_DATA1(format, ...)  DEVICE_PRINT_DATA1(format, ##__VA_ARGS__)

#define ENDL() '\n'
