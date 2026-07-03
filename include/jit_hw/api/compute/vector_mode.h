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

enum VectorMode : uint8_t {
    None      = 0,
    R         = 1,
    C         = 2,
    RC        = 4,
    RC_custom = 6,
    Invalid   = 0xFF,
};

// Whether SFPU element i (row-major: row = i/32, col = i%32) is written under a
// given VectorMode and SFPU iteration count.
//
// Faithful to silicon's `_llk_math_eltwise_sfpu_apply_vector_mode_`
// (tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_sfpu_common.h): the dispatcher
// invokes the SFPU functor once per *active face base* — R→{0,1}, C→{0,2},
// RC→{0,1,2,3}, None→{0} — and each invocation walks `iterations`×32 DST lanes
// contiguously in face-major order from that base. One face = 256 lanes = 8
// iterations, so a single call covers ceil(iterations/8) contiguous faces.
//
// The common case iterations==8 covers exactly one face per base, reproducing the
// historical RC→all / R→row<16 / C→col<16 / None→face0 behavior. The SDPA softmax
// exp is the case that differs: it is dispatched as (VectorMode::None, iterations=32)
// — one functor call walking all 4 faces = the whole tile. With the old 1-face
// model, None dropped faces 1,2,3, leaving rows 16-31 (and cols 16-31) un-exp'd and
// the softmax structurally wrong.
//
// Faces (row-major 32x32): 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
// (face = (row/16)*2 + (col/16)). Skipped faces keep prior DEST contents (data-path
// control, not a perf hint — confirmed via DeepWiki ISA).
inline bool __emule_vector_mode_active(uint32_t i, VectorMode mode, int iterations) {
    const uint32_t row = i / 32, col = i % 32;
    const uint32_t face = (row / 16) * 2u + (col / 16);  // 0=TL,1=TR,2=BL,3=BR
    const uint32_t iters = iterations > 0 ? static_cast<uint32_t>(iterations) : 8u;
    const uint32_t faces_per_base = (iters * 32u + 255u) / 256u;  // ceil(iters/8), >=1
    auto covers = [&](uint32_t base) { return face >= base && face < base + faces_per_base; };
    switch (mode) {
        case VectorMode::RC:
        case VectorMode::RC_custom: return true;            // all four faces
        case VectorMode::R:        return covers(0) || covers(1);
        case VectorMode::C:        return covers(0) || covers(2);
        default:                   return covers(0);        // None
    }
}

// 2-arg form: default SFPU iteration count (8 = one face per active base). Matches
// the historical behavior for every caller that does not pass an explicit count.
inline bool __emule_vector_mode_active(uint32_t i, VectorMode mode) {
    return __emule_vector_mode_active(i, mode, 8);
}

}  // namespace ckernel
