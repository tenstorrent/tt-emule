// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Single source of truth for ckernel::VectorMode.
//
// SFPU op stubs that accept a vector_mode argument (exp_tile, recip_tile, ...)
// must agree on the type's namespace; previously exp.h defined it at global
// scope while recip.h defined it inside ckernel, and both used the same
// __EMULE_VECTOR_MODE_DEFINED guard — so depending on include order only one
// of the two definitions was actually emitted. Consolidate here.
//
// Values mirror the canonical silicon definition in
//   tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0,quasar}/llk_lib/llk_defs.h
// (Quasar uses `enum class`; BH/WH use unscoped `enum`. We use `enum class`
// here — stricter than BH/WH, exactly matching Quasar.) Earlier emule
// stubs had RC=0 / R=1 / C=2 which would silently disagree with any code
// that compared `vector_mode` against the canonical integer constants.

#include <cstdint>

namespace ckernel {

enum class VectorMode : uint8_t {
    None      = 0,
    R         = 1,
    C         = 2,
    RC        = 4,
    RC_custom = 6,
    Invalid   = 0xFF,
};

// Whether SFPU element i (row-major: row = i/32, col = i%32) is written under a
// given VectorMode. Skipped faces keep prior DEST contents (data-path control, not
// a perf hint — confirmed via DeepWiki ISA). RC/RC_custom → all; R → faces {0,1}
// (row<16); C → faces {0,2} (col<16); None → face 0.
inline bool __emule_vector_mode_active(uint32_t i, VectorMode mode) {
    const uint32_t row = i / 32, col = i % 32;
    switch (mode) {
        case VectorMode::RC:
        case VectorMode::RC_custom: return true;
        case VectorMode::R:        return row < 16;
        case VectorMode::C:        return col < 16;
        default:                   return row < 16 && col < 16;  // None
    }
}

}  // namespace ckernel
