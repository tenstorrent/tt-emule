// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `tensix_types.h`. The real header (in
// tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/) defines
// DataFormat and other arch-specific enums that conflict with emule's
// own DataFormat in jit_hw/api/compute/common_globals.h. Data-movement
// reader kernels include "tensix_types.h" but typically don't reference
// any symbol from it — including the empty shim lets the kernel parse.
//
// Add type aliases here if a kernel surface needs them. Avoid redefining
// DataFormat — use emule's existing one.
