// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for upstream `api/compute/cb_api.h`. Intercepts the include so
// JIT compilation doesn't fall through to the upstream header that includes
// hardware-only LLK headers (`llk_io_pack.h` / `llk_io_unpack.h`) when the
// TRISC_UNPACK/MATH/PACK guard defines are set (which emule sets so that
// all three TRISC sections of a compute kernel execute on the single x86
// thread, per the per-arch real-HW build model).
//
// emule's CB operations live in `jit_hw/api/cb_api.h` and are exposed by
// the various headers compute kernels include (in particular,
// experimental::CircularBuffer in `experimental/circular_buffer.h`, which
// is what the tt-mlir-generated kernels use for CB ops). The free
// functions `cb_wait_front` / `cb_pop_front` / `cb_reserve_back` /
// `cb_push_back` that the upstream `cb_api.h` declares are already provided
// at the global scope by `jit_hw/api/cb_api.h` (included transitively via
// the JIT wrapper).

#include "jit_hw/api/cb_api.h"
