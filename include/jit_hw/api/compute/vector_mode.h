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

}  // namespace ckernel
