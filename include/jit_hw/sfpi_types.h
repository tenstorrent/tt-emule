// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// SFPI value types + the grouped per-RISC SFPU state struct.
//
// Split out of sfpi.h so that ComputeThreadCtx (emule_thread_ctx.h) can hold the
// SFPU state by-value WITHOUT a circular include: this header is dependency-free
// (only <array>/<cstdint>), so emule_thread_ctx.h — which is also parsed by the
// host runner and the umd TU — can include it. sfpi.h then includes
// emule_thread_ctx.h and reaches all SFPU state via __emule_compute_ctx().sfpu.
//
// Only `vUInt` (backs the LReg file) and `__MaskFrame` live here — the types that
// SfpuState holds by value. vFloat/vInt and all the SFPU ops stay in sfpi.h; the
// lane-masked `operator=` are declared here but DEFINED in sfpi.h (they read the
// predication mask, which is now SfpuState::mask reached via the ctx).

#include <cstdint>
#include <array>

static constexpr uint32_t __EMULE_SFPI_LANES = 32;
static constexpr uint32_t __EMULE_SFPI_TILE_ELEMS = 1024;

namespace sfpi {

// 32-lane unsigned vector — backs the SFPU LReg file (SfpuState::lreg). Plain
// storage + ctors here; the lane-masked operator= is defined in sfpi.h.
class vUInt {
public:
    std::array<uint32_t, __EMULE_SFPI_LANES> v{};
    constexpr vUInt() = default;
    constexpr vUInt(uint32_t x) { for (auto& lane : v) lane = x; }
    constexpr vUInt(int32_t x) { for (auto& lane : v) lane = static_cast<uint32_t>(x); }
    vUInt(const vUInt&) = default;
    vUInt& operator=(const vUInt& o);  // defined in sfpi.h (lane-masked; reads SfpuState::mask)
};

// One v_if/v_elseif/v_else/v_endif predication frame.
struct __MaskFrame {
    std::array<bool, __EMULE_SFPI_LANES> outer;   // outer-scope mask
    std::array<bool, __EMULE_SFPI_LANES> taken;   // OR of all branch conds matched so far
};

// All SFPI intrinsic state, grouped into one struct (held by ComputeThreadCtx as
// `sfpu`). Previously these were separate thread_local globals in sfpi.h.
struct SfpuState {
    float*   dst_base = nullptr;          // active DST tile element-0 (was __emule_sfpi_dst_base)
    uint32_t cursor = 0;                  // 32-lane window offset (was __emule_sfpi_cursor)
    std::array<bool, __EMULE_SFPI_LANES> mask = {  // active-lane predication mask (was __emule_sfpi_mask)
        true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true,
    };
    __MaskFrame frames[16] = {};          // predication stack (was __emule_sfpi_frames)
    int      frame_depth = 0;             // stack depth (was __emule_sfpi_frame_depth)
    float    dst_fallback[__EMULE_SFPI_TILE_ELEMS] = {};  // scratch DST (was __emule_sfpi_dst_fallback)
    vUInt    lreg[16] = {};               // SFPU LReg file (was __emule_lreg)
    uint32_t prgm_creg[3] = {0, 0, 0};    // programmable const regs (was __emule_prgm_creg)
    // multichip SDPA SFPU extensions (not in the original fiber-rebase SfpuState):
    bool     first_col_mode = false;      // first-column column-vector DST walk (was __emule_sfpi_first_col_mode)
    std::array<bool, __EMULE_SFPI_LANES> sfp_cc_outer = {};  // raw-TTI CC outer mask (was __emule_sfp_cc_outer)
    bool     sfp_cc_active = false;        // raw-TTI CC predication active (was __emule_sfp_cc_active)
};

}  // namespace sfpi
