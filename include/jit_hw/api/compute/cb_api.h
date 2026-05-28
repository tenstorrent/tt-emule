// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Shadow for api/compute/cb_api.h — intercepts the real Metal path (which
// references LLK symbols that don't exist in emulation) and replaces it with
// the emule CB-sync implementations from jit_hw/api/cb_api.h.
//
// Also declares the LLK stubs that the real Metal cb_api.h uses (llk_wait_tiles
// etc.) so that any downstream header that was already resolved before this
// shadow applies doesn't produce undeclared-identifier errors.

// Pull in the emule CB sync implementations (cb_reserve_back, cb_push_back,
// cb_wait_front, cb_pop_front, get_write_ptr, get_read_ptr, get_tile_size, ...).
#include "jit_hw/api/cb_api.h"

// The real api/compute/cb_api.h wraps every CB call in UNPACK(...)/PACK(...)
// macros that expand to the bare expression in emulation (common.h defines them
// as identity macros).  The LLK functions they reference are hardware-only; in
// emulation we provide no-op stubs so that any file that directly includes the
// real cb_api.h (e.g. via -I ordering that misses this shadow) still compiles.

inline void llk_wait_tiles(uint32_t /*cbid*/, uint32_t /*ntiles*/) {}
inline void llk_pop_tiles(uint32_t /*cbid*/, uint32_t /*ntiles*/) {}
template <bool A = false, bool B = false, bool C = false>
inline void llk_wait_for_free_tiles(uint32_t /*cbid*/, uint32_t /*ntiles*/) {}
template <bool A = false, bool B = false>
inline void llk_push_tiles(uint32_t /*cbid*/, uint32_t /*ntiles*/) {}

// Provide the ckernel-namespaced wrappers that mirror the real Metal cb_api.h.
// The real header puts cb_wait_front/cb_pop_front/cb_reserve_back/cb_push_back
// inside namespace ckernel (they delegate to the LLK fns above).  Our emule
// implementations are already in the global namespace; add ckernel aliases so
// code compiled with -I paths that see this shadow before the real header can
// use either spelling.
namespace ckernel {
    using ::cb_wait_front;
    using ::cb_pop_front;
    using ::cb_reserve_back;
    using ::cb_push_back;
    using ::get_write_ptr;
    using ::get_read_ptr;
    using ::get_tile_size;
}  // namespace ckernel
