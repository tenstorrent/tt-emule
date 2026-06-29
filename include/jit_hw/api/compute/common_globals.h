// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Minimal DataFormat enum for JIT-compiled SFPU kernels.
// Matches the real tt-metal enum values we need.
#include <cstdint>
#include "tt-metalium/circular_buffer_constants.h"  // NUM_CIRCULAR_BUFFERS (32 WH / 64 BH)

enum class DataFormat : uint8_t {
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Tf32      = 4,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Int32     = 8,
    UInt16    = 9,
    Lf8       = 10,
    Bfp2      = 11,
    Int8      = 14,
    Bfp2_b    = 15,
    UInt32    = 24,
    UInt8     = 30,
    Fp8_e4m3  = 0x1A,
    RawUInt8  = 0xF0,
    RawUInt16 = 0xF1,
    RawUInt32 = 0xF2,
    Invalid   = 0xff
};

// PACK engine auto-advance: tracks the write offset within a pack batch.
// On real hardware the PACK engine auto-advances its L1 write pointer after each
// pack_tile.  In emulation, cb_write_ptr resolves the calling RISC's per-thread
// CB write pointer (jit_hw/internal/emule_cb_ptr.h), which only advances on
// push_back.  This counter emulates the hardware auto-advance: incremented by
// pack_tile, reset to 0 on cb_push_back (the batch commit — matching silicon
// pack.h, which resets the sequential pack pointer after cb_push_back, NOT on
// cb_reserve_back; a run of reserve_back calls with no push keeps advancing).
static thread_local uint32_t __emule_pack_offset[NUM_CIRCULAR_BUFFERS] = {};

// PACK blocked-width: silicon's llk_pack_init(ocb, pack_width) programs the packer
// MOP to emit `pack_width` consecutive tiles per single pack_tile issue (the SDPA
// streaming "blocked pack", used on Blackhole at width>=4; Wormhole packs single-
// tile below width 8). Recorded per-CB by llk_pack_init; a stored 0 means
// "not configured" and is treated as width 1. pack_tile reads DST[idst..idst+w-1]
// and writes w consecutive CB slots. Every kernel's pack_init re-establishes this
// (default 1), so a width set by one kernel can't leak into the next.
static thread_local uint32_t __emule_pack_width[NUM_CIRCULAR_BUFFERS] = {};

// Per-DST-slot "fresh since acquire" flag.  Set true by tile_regs_acquire,
// cleared by any op that writes meaningful values into the slot (copy_tile,
// add/sub/mul_tiles, matmul_tiles, etc.).  reduce_tile<MAX>/<MIN> uses it to
// distinguish "first call after acquire" from "accumulating into a slot the
// kernel pre-loaded via copy_tile".
//
// Why: emule's tile_regs_acquire zeroes DST (real HW leaves it undefined), so
// for negative-valued reductions like `min(x) = -max(-x)`, the first
// max-accumulation call would clamp to 0 instead of producing the negative
// per-tile max.  When fresh, the reduce op overwrites DST instead of
// max-accumulating; subsequent calls within the same acquire cycle (or after
// a copy_tile load of a running accumulator) see fresh=false and use the
// existing accumulator semantics.
static thread_local bool __emule_dst_fresh[16] = {true, true, true, true, true, true, true, true,
                                                  true, true, true, true, true, true, true, true};

inline void __emule_dst_mark_dirty(uint32_t slot) {
    if (slot < 16) {
        __emule_dst_fresh[slot] = false;
    }
}

// Per-DST-slot tag: does this slot hold a raw integer bit pattern (an index
// payload loaded from an integer-format CB) rather than a numeric float?
// emule's DST is untyped float32, so eqz consults this to emit an integer 1/0
// (silicon's integer-mode SFPU result) which then packs bit-exact, vs a float
// 1.0f/0.0f. Set by copy_tile from the source CB format; cleared on acquire.
static thread_local bool __emule_dst_holds_int[16] = {};

inline void __emule_dst_set_int(uint32_t slot, bool is_int) {
    if (slot < 16) {
        __emule_dst_holds_int[slot] = is_int;
    }
}

inline bool __emule_dst_is_int(uint32_t slot) {
    return slot < 16 && __emule_dst_holds_int[slot];
}

inline bool __emule_dst_take_fresh(uint32_t slot) {
    if (slot >= 16) {
        return false;
    }
    bool was_fresh = __emule_dst_fresh[slot];
    __emule_dst_fresh[slot] = false;
    return was_fresh;
}
