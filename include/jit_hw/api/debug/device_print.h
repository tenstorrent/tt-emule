// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Stub DEVICE_PRINT macros for JIT-compiled kernels.
// On real hardware these serialize arguments to a ring buffer.
// In the emulator we discard them.

#define DEVICE_PRINT(format, ...)
#define DEVICE_PRINT_UNPACK(format, ...)
#define DEVICE_PRINT_PACK(format, ...)
