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
//
// Also pull in `compute/common.h` so a kernel that includes ONLY this
// header (no other compute API) still sees `ALWI` and the other shared
// compute macros it expects.

#include "jit_hw/api/compute/common.h"
#include "jit_hw/api/cb_api.h"

// In namespace ckernel to match upstream api/compute/cb_api.h, so both bare
// calls (via `using namespace ckernel;` in common.h) and ckernel::-qualified
// callers resolve.
namespace ckernel {

// get_tile_address — silicon returns the byte L1 address of a CB tile (via
// UNPACK→MATH/PACK mailbox sync) so callers can `reinterpret_cast<To*>` it and
// read tile data directly. Used by the normalization kernel_util
// (`ttnn/.../normalization/kernel_util/compute/memory.h::get_pointer_to_cb_data`,
// e.g. the layernorm-welford reciprocal LUT) and by reshuffle (embedding_backward).
// emule runs UNPACK/MATH/PACK on one thread, so no mailbox is needed. L1 offset
// model: return a 0-based L1 offset (like get_write_ptr/get_read_ptr) — the CB ring
// is maintained in host-pointer space, so subtract this fiber's bridge_l1. Callers
// that `reinterpret_cast<To*>` the result are rebased by the JIT patch pass
// (get_tile_address is registered there as an L1-address producer), and
// reshuffle_rows_tile rebases via __emule_local_l1_to_ptr.
ALWI uint32_t get_tile_address(uint32_t cb_id, uint32_t tile_index) {
    uint8_t* ptr = __emule_compute::cb_read_ptr_at(cb_id, tile_index);
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) -
                                 reinterpret_cast<uintptr_t>(__emule_self->bridge_l1));
}

// read_tile_value — silicon reads a uint32_t at element_offset from the CB
// tile's L1 bytes on UNPACK and mailboxes it to MATH/PACK (upstream
// api/compute/cb_api.h, CB mailbox helpers for tt-metal#27979; used by the
// embedding_backward compute kernel for the chunk-count handoff). emule runs
// all TRISC roles on one thread, so the mailbox collapses to a plain read of
// the same raw tile bytes.
ALWI uint32_t read_tile_value(uint32_t cb_id, uint32_t tile_index, uint32_t element_offset) {
    return reinterpret_cast<uint32_t*>(__emule_compute::cb_read_ptr_at(cb_id, tile_index))[element_offset];
}

}  // namespace ckernel
