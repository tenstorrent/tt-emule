// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Raw SFPU-instruction emulation layer (TTI_SFP* on a 32-lane LReg file).
//
// emule normally models compute at the tile-op level (`<op>_tile` over
// __emule_dst). batched_qk_norm hand-writes a Tensix SFPU instruction stream
// inline in its (pristine) op.hpp, so there is no high-level seam to shadow —
// the only way to run it is to execute the instructions. This header provides
// the six instructions that kernel uses plus the DST addressing they assume.
//
// Must be included where ::__emule_dst (api/compute/common.h) is visible.
//
// DST addressing (Blackhole, ADDR_MOD_7 = zero-increment; per
// tt-isa-documentation Vector Unit Instructions + tt_llk_blackhole binary_bcast):
//   address A = base(set by TT_SETC16) + imm, in datum units, decoded as
//     tile = A/64, face = (A%64)/16, inface = (A%64)%16,
//     row_band = inface>>2, even_odd = (inface>>1)&1
//   per lane: i = lane>>3 (sub-row 0..3), c = lane&7 (sub-col 0..7)
//     row = row_band*4 + i,  col = c*2 + even_odd   (even/odd INTERLEAVED cols)
//   so one 32-lane SFPLOAD covers 4 rows x 8 (even-or-odd) columns of a face;
//   +2 toggles even/odd, +4 next 4-row band, +16 next face, +64 next tile.
// SFP{MUL,ADD} are the MAD form VD = VA*VB + VC (LCONST_0=0.0, LCONST_1=1.0).
// SFPSHFT2 mod1=3 is a rotate-right-by-1 within each 8-lane sub-vector.
#pragma once

#include <cstdint>
#include "api/compute/common.h"

namespace p_sfpu {
constexpr uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3;
constexpr uint32_t LREG4 = 4, LREG5 = 5, LREG6 = 6, LREG7 = 7;
constexpr uint32_t LCONST_0 = 8;  // read-only 0.0
constexpr uint32_t LCONST_1 = 9;  // read-only 1.0
}  // namespace p_sfpu

#ifndef ADDR_MOD_7
constexpr uint32_t ADDR_MOD_7 = 7;
#endif
#ifndef DEST_TARGET_REG_CFG_MATH_Offset_ADDR32
constexpr uint32_t DEST_TARGET_REG_CFG_MATH_Offset_ADDR32 = 0;
#endif

// 32-lane float LReg file (indices 0..7 usable; 8/9 are the const registers).
inline thread_local float __emule_sfpu_freg[10][32] = {};
// Origin of the contiguous __emule_dst[][] tile storage; tiles are 1024 floats.
inline thread_local float* __emule_sfpu_dst_origin = nullptr;
// DEST_TARGET base offset (datum units) programmed by TT_SETC16.
inline thread_local uint32_t __emule_sfpu_dest_offset = 0;

inline void __emule_sfpu_bind_dst(float* origin) { __emule_sfpu_dst_origin = origin; }

inline float __emule_sfpu_rd(uint32_t reg, uint32_t lane) {
    if (reg == p_sfpu::LCONST_0) return 0.0f;
    if (reg == p_sfpu::LCONST_1) return 1.0f;
    return __emule_sfpu_freg[reg][lane];
}

inline float* __emule_sfpu_dst_ptr(uint32_t addr, uint32_t lane) {
    const uint32_t tile = addr / 64u;
    const uint32_t within = addr % 64u;
    const uint32_t face = within / 16u;          // 0..3
    const uint32_t inface = within % 16u;
    const uint32_t row_band = inface >> 2;        // 0..3
    const uint32_t even_odd = (inface >> 1) & 1u;
    const uint32_t i = lane >> 3;                 // sub-row 0..3
    const uint32_t c = lane & 7u;                 // sub-col 0..7
    const uint32_t face_row = row_band * 4u + i;  // 0..15
    const uint32_t face_col = c * 2u + even_odd;  // 0..15
    const uint32_t tile_row = (face >= 2u ? 16u : 0u) + face_row;
    const uint32_t tile_col = (face & 1u ? 16u : 0u) + face_col;
    return __emule_sfpu_dst_origin + tile * 1024u + tile_row * 32u + tile_col;
}

inline void __emule_sfpload(uint32_t vd, uint32_t /*mod0*/, uint32_t /*addr_mod*/, uint32_t offset) {
    const uint32_t addr = __emule_sfpu_dest_offset + offset;
    for (uint32_t lane = 0; lane < 32; ++lane) __emule_sfpu_freg[vd][lane] = *__emule_sfpu_dst_ptr(addr, lane);
}
inline void __emule_sfpstore(uint32_t vs, uint32_t /*mod0*/, uint32_t /*addr_mod*/, uint32_t offset) {
    const uint32_t addr = __emule_sfpu_dest_offset + offset;
    for (uint32_t lane = 0; lane < 32; ++lane) *__emule_sfpu_dst_ptr(addr, lane) = __emule_sfpu_freg[vs][lane];
}
// VD = VA*VB + VC  (SFPMAD form; both SFPMUL and SFPADD map here).
inline void __emule_sfpmad(uint32_t va, uint32_t vb, uint32_t vc, uint32_t vd) {
    float tmp[32];
    for (uint32_t l = 0; l < 32; ++l) tmp[l] = __emule_sfpu_rd(va, l) * __emule_sfpu_rd(vb, l) + __emule_sfpu_rd(vc, l);
    for (uint32_t l = 0; l < 32; ++l) __emule_sfpu_freg[vd][l] = tmp[l];
}
inline void __emule_sfpmov(uint32_t vc, uint32_t vd) {
    for (uint32_t l = 0; l < 32; ++l) __emule_sfpu_freg[vd][l] = __emule_sfpu_rd(vc, l);
}
// Rotate-right-by-1 within each 8-lane sub-vector: out[lane] = in[(lane&7)?lane-1:lane+7].
inline void __emule_sfpshft2_ror1(uint32_t vc, uint32_t vd) {
    float tmp[32];
    for (uint32_t l = 0; l < 32; ++l) {
        const uint32_t g = l & ~7u, k = l & 7u;
        tmp[l] = __emule_sfpu_rd(vc, g + (k == 0u ? 7u : k - 1u));
    }
    for (uint32_t l = 0; l < 32; ++l) __emule_sfpu_freg[vd][l] = tmp[l];
}

namespace ckernel {
// Returns the (logical, datum-unit) base of DST tile 0 and binds the SFPU DST
// origin to the contiguous __emule_dst storage. The kernel adds t*64 to select
// tile t, which the decode above resolves back to __emule_dst[t].
inline uint32_t get_dest_buffer_base() {
    ::__emule_sfpu_bind_dst(&__emule_dst[0][0]);
    return 0u;
}
}  // namespace ckernel

#ifndef TT_SETC16
#define TT_SETC16(reg, val)                                              \
    do {                                                                 \
        if ((reg) == DEST_TARGET_REG_CFG_MATH_Offset_ADDR32)             \
            ::__emule_sfpu_dest_offset = static_cast<uint32_t>(val);     \
    } while (0)
#endif
#ifndef TTI_STALLWAIT
#define TTI_STALLWAIT(stall_mask, wait_kind) do { (void)(stall_mask); (void)(wait_kind); } while (0)
#endif

// dest operand is the 4th arg for the MAD ops; 1st for load, source for store.
#ifndef TTI_SFPLOAD
#define TTI_SFPLOAD(vd, mod0, addr_mod, offset) ::__emule_sfpload((vd), (mod0), (addr_mod), (offset))
#endif
#ifndef TTI_SFPSTORE
#define TTI_SFPSTORE(vs, mod0, addr_mod, offset) ::__emule_sfpstore((vs), (mod0), (addr_mod), (offset))
#endif
#ifndef TTI_SFPMUL
#define TTI_SFPMUL(va, vb, vc, vd, mod) ::__emule_sfpmad((va), (vb), (vc), (vd))
#endif
#ifndef TTI_SFPADD
#define TTI_SFPADD(va, vb, vc, vd, mod) ::__emule_sfpmad((va), (vb), (vc), (vd))
#endif
#ifndef TTI_SFPMOV
#define TTI_SFPMOV(mod0, vc, vd, mod1) ::__emule_sfpmov((vc), (vd))
#endif
#ifndef TTI_SFPSHFT2
#define TTI_SFPSHFT2(mod0, vc, vd, imm) ::__emule_sfpshft2_ror1((vc), (vd))
#endif
