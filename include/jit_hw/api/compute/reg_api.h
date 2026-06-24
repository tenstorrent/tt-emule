#pragma once
// JIT compute API — DST register state machine (tile_regs_*).
//
// Owns tile_regs_acquire / tile_regs_commit / tile_regs_wait /
// tile_regs_release. `api/compute/common.h` re-includes this header so
// existing `#include "api/compute/common.h"` callers continue to see the
// symbols.
//
// Dependencies: __emule_compute_ctx().dst (thread_local DST tile array),
// __EMULE_TILE_ELEMS, and __emule_dst_active_tiles() must be visible at
// the point of inclusion. They live in common.h; reg_api.h is included
// by common.h after those definitions are in scope. The __emule_compute_ctx().dst_fresh[]
// flag array is pulled in via common_globals.h.

#include "jit_hw/api/compute/common_globals.h"
#include "jit_hw/internal/risc_attribs.h"
#include <cstring>
#include <cstdint>

#ifndef ALWI
#define ALWI FORCE_INLINE
#endif

// ---- DST state machine (no-ops in single-thread-per-compute emulation) ----

ALWI void tile_regs_acquire() {
    // Zero DST on acquire (matches device behavior: acquire gives clean regs).
    // Mark all slots fresh so reduce_tile<MAX>/<MIN> knows to overwrite on
    // first use rather than max-accumulate against the zero-init.
    uint32_t active = __emule_dst_active_tiles();
    for (uint32_t s = 0; s < active; s++) {
        std::memset(__emule_compute_ctx().dst[s], 0, sizeof(__emule_compute_ctx().dst[s]));
        __emule_compute_ctx().dst_fresh[s] = true;
    }
}
ALWI void tile_regs_commit()  {}
ALWI void tile_regs_wait()    {}
ALWI void tile_regs_release() {}
