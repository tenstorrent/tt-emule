#pragma once
// JIT compute API — DST register state machine (tile_regs_*).
//
// History: this file used to be an intercept-only shim that delegated to
// api/compute/common.h via `#include "common.h"`.  Per PR #21 review
// feedback ("Should we move those APIs here then?"), the relationship is
// now flipped: reg_api.h OWNS tile_regs_acquire / tile_regs_commit /
// tile_regs_wait / tile_regs_release, and common.h re-includes this
// header for back-compat so existing `#include "api/compute/common.h"`
// callers continue to see the symbols.
//
// Dependencies: requires __emule_dst (the thread_local DST tile array),
// __EMULE_TILE_ELEMS, and __emule_dst_active_tiles() to be visible at
// the point of inclusion.  These currently live in common.h; reg_api.h
// is included by common.h after those definitions are in scope.  The
// __emule_dst_fresh[] flag array is pulled in via common_globals.h.

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
        std::memset(__emule_dst[s], 0, sizeof(__emule_dst[s]));
        __emule_dst_fresh[s] = true;
    }
}
ALWI void tile_regs_commit()  {}
ALWI void tile_regs_wait()    {}
ALWI void tile_regs_release() {}
