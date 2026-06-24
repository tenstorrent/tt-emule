// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `noc/noc_parameters.h`.  Upstream ships per-arch under
// `tt_metal/hw/inc/internal/tt-Nxx/<arch>/noc/`; the JIT include path only
// adds `tt_metal/hw/inc` (no arch-specific dir), so this shim dispatches on
// the ARCH_* JIT define that emulated_program_runner.cpp passes per program
// (see build_kernel_defines: ARCH_WORMHOLE / ARCH_BLACKHOLE / ARCH_QUASAR).
//
// WH vs BH noc_parameters differ in NUM_TENSIXES, VIRTUAL_TENSIX_START_X/Y,
// NOC_CMD_BUF_OFFSET, and several register address layouts.  Without the
// dispatch, a BH JIT kernel built with the WH header would compute wrong
// NOC commands.

// Drop emule's #ifndef fallback shims (pulled in transitively before this
// wrapper) so the authoritative, UNGUARDED per-arch header below wins without
// tripping -Wmacro-redefined on the semantically-equal redefinition. #undef of
// an undefined macro is a no-op.
#undef NOC_UNICAST_ADDR_X
#undef NOC_UNICAST_ADDR_Y
#undef NOC_XY_ADDR
#undef NOC_MULTICAST_ADDR
#undef NOC_ADDR_LOCAL_BITS
#undef NOC_ADDR_NODE_ID_BITS
#undef NOC_MAX_BURST_SIZE
#undef L1_ALIGNMENT
#undef DRAM_ALIGNMENT

#if defined(ARCH_BLACKHOLE)
#include "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h"
#elif defined(ARCH_QUASAR)
#include "tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h"
#else
#error "ARCH_{WORMHOLE,BLACKHOLE,QUASAR} not defined — the runner must pass one of these as a JIT compile define"
#endif
