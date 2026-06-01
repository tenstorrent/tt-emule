// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `tensix_types.h`. Some sharded data-movement kernels
// (e.g. reader_unary_sharded_blocks_interleaved_start_id.cpp) include
// this header defensively, then never reference any of its symbols.
// Without the shim, the bare `#include "tensix_types.h"` doesn't
// resolve and JIT compile fails.
//
// We don't forward to upstream's wormhole_b0_defines/tensix_types.h
// because its `enum class DataFormat` redefines the one emule already
// emits in `api/compute/common_globals.h` (pulled in via
// dataflow_api_addrgen.h). Trying to merge them would also drag in
// `<fmt/core.h>` (firmware-gated) and a host-only `io_queue_pointers_t`
// formatter chain we don't need.
//
// If a future kernel actually consumes a tensix_types symbol (xmov_*,
// packer_config_t, ...) the right move is to thin-add that symbol
// here directly, not to forward-include the upstream header.
//
// Real LLK reference:
//   tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h
