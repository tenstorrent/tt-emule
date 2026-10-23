// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Mirrors tenstorrent/tt-metal#61991: tt-metalium/constants.hpp is host-only and #errors in kernel and
// firmware builds. Kernels take the same tt::constants values from api/tile_constants.h, which resolves to
// tt_metal/hw/inc on the JIT -I path. This shim exists so a kernel that still includes the host header
// fails here with the same message as on silicon, instead of compiling against an emule-only copy.
#pragma once

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#error "tt-metalium/constants.hpp is host-only; include \"api/tile_constants.h\" in device code"
#endif
