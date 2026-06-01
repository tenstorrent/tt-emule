// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for upstream `api/compute/experimental/fast_untilize.h`. Intercepts
// the include so the JIT compile doesn't fall through to the upstream header,
// which under our TRISC_UNPACK/MATH/PACK defines would pull in riscv-target
// LLK headers (`experimental/llk_unpack_fast_untilize_api.h` and friends) that
// don't compile under the x86 JIT.
//
// Real LLK reference:
//   tt_metal/hw/inc/api/compute/experimental/fast_untilize.h
//   tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_fast_untilize.h
//
// Wormhole vs Blackhole: the kernel-lib's `compute_kernel_lib::untilize<>`
// template gates the call to fast_untilize_* on
// `static constexpr bool use_fast = can_use_fast_untilize<...>()`. On non-BH
// arches that predicate is `false`, so the `if constexpr (use_fast)` branch is
// statically discarded — bodies are never instantiated, only parsed for name
// lookup. On BH the predicate can be `true` and the bodies execute.
//
// Both arches handled by the same delegation: route fast_untilize_* to the
// regular `pack_untilize_*` path. The fast path's contract on real hardware is
// purely a packer-scheduling optimization (ALL_INTF_ACTIVE + STRIDED_MODE for
// wider PACR output) — invisible to a functional simulator. Upstream notes the
// BH fast fp32 path "narrows input through the current SrcA route" while the
// regular path is "bit-exact for Float32"; delegating to the regular path is
// therefore strictly more faithful for PCC.

#include <cstdint>

#include "jit_hw/api/compute/pack_untilize.h"
#include "jit_hw/api/compute/untilize.h"

namespace ckernel {

template <std::uint32_t full_ct_dim>
inline void fast_untilize_init(uint32_t icb, uint32_t ocb, uint32_t /*call_line*/ = 0) {
    pack_untilize_init<full_ct_dim, full_ct_dim>(icb, ocb);
}

template <std::uint32_t full_ct_dim>
inline void fast_untilize_init_skip_remap(uint32_t icb, uint32_t ocb, uint32_t /*call_line*/ = 0) {
    // Emule has no DEST remap state, so the skip-remap variant collapses to
    // the configure variant.
    pack_untilize_init<full_ct_dim, full_ct_dim>(icb, ocb);
}

template <std::uint32_t full_ct_dim>
inline void fast_untilize_block(uint32_t icb, uint32_t ocb,
                                uint32_t /*input_tile_index*/ = 0,
                                uint32_t /*output_tile_index*/ = 0) {
    // Delegate to the regular packer with block_rt_dim=1 (a single tile-row
    // block of width full_ct_dim), matching upstream's non-BH branch:
    //   #else  // !ARCH_BLACKHOLE
    //       pack_untilize_block<full_ct_dim, full_ct_dim>(icb, 1, ocb, 0);
    pack_untilize_block<full_ct_dim, full_ct_dim>(icb, 1, ocb, 0);
}

template <std::uint32_t full_ct_dim>
inline void fast_untilize_uninit(uint32_t ocb) {
    pack_untilize_uninit(ocb);
}

}  // namespace ckernel
