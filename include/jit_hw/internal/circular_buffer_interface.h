// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Shim for the tt-metal "internal/circular_buffer_interface.h" header that
// tt-metal kernel-lib headers (e.g. ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp)
// pull in. The real header at
//   /localdev/arminale/tt-metal/tt_metal/hw/inc/internal/circular_buffer_interface.h
// defines LocalCBInterface, CBInterface, get_local_cb_interface(uint32_t) —
// but emule's internal/llk_state.h already supplies a host-side CbInterface and
// inline get_local_cb_interface that wraps emule's CB sync state. Letting the
// real header through produces:
//   - "functions that differ only in their return type" conflict on
//     get_local_cb_interface (CbInterface& vs LocalCBInterface&)
//   - dependency on Tensix-only L1_ALIGNMENT/NUM_CIRCULAR_BUFFERS layout
//
// emule's llk_state.h is the source of truth on host. Pull it in so any code
// that #includes this path still gets the matching get_local_cb_interface
// declaration. The names diverge (CbInterface vs LocalCBInterface), but
// downstream code that only touches the .fifo_page_size etc. subset compiles
// against emule's struct without ever materializing LocalCBInterface.
#include "jit_hw/internal/llk_state.h"
