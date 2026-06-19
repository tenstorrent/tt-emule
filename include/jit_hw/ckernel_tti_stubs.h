// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Functional emulation of the raw Tensix SFPU intrinsic surface (TTI_SFP* macros)
// plus the addr_mod / p_sfpu / InstrModLoadStore types they reference.
//
// HISTORY: these were originally compile-only no-ops, on the assumption that the
// only raw-TTI code path — the SDPA compute kernel's `calculate_exponential_polynomial`
// (the `if constexpr` ELSE of the exp_approx_mode branch) — was never INSTANTIATED,
// because emule's tests ran exp_approx_mode=true and took the sfpi-based
// `_ckernel_sfpu_exp_accurate_` branch. That assumption was false: with
// exp_approx_mode=false, streaming SDPA's online-softmax rescale exp(prev_max-cur_max)
// (sub_exp_first_col_blocks -> exp_tile_first_column<false> -> calculate_exponential_polynomial)
// DOES execute this path. As no-ops the rescale factor was never computed, silently
// corrupting the cross-K-chunk correction (PCC ~0.76).
//
// So these macros now drive a small, faithful 32-lane SFPU interpreter (LReg file +
// LaneFlags CC mask + DST addressing via the existing sfpi cursor) sufficient to run
// the exp polynomial. Only the instructions that polynomial uses on Wormhole are
// modelled; Blackhole-only ops (SFPARECIP/SFPGT) and config-only ops stay no-ops.

#include <cstdint>
#include <cmath>
#include <cstring>
#include "jit_hw/sfpi.h"                 // __emule_sfpi_dst_base / _cursor / _lane_index / sfpi::__emule_sfpi_mask
#include "jit_hw/api/compute/common.h"   // __emule_dst (for the math:: addr-mod-base hook)

// The SDPA exp polynomial kernel declares `constexpr float M_LN2 = ...` as a local
// identifier. On bare-metal silicon M_LN2 is not a macro; emule's hosted <cmath>
// pulls glibc's `#define M_LN2 0.69314...` (math.h), which would mangle that
// declaration into a parse error. Undef to restore the silicon condition.
#ifdef M_LN2
#undef M_LN2
#endif

// SFP_STOCH_RND mod1 selectors. Silicon defines these in `sfpi::` (sfpi_constants.h);
// emule's sfpi.h omitted them because the raw-TTI exp polynomial — their only user —
// was never compiled. Add them now (values mirror upstream sfpi_constants.h).
namespace sfpi {
constexpr unsigned int SFPSTOCHRND_MOD1_FP32_TO_FP16B = 1;
constexpr unsigned int SFPSTOCHRND_MOD1_FP32_TO_UINT8 = 2;
constexpr unsigned int SFPSTOCHRND_MOD1_FP32_TO_INT8  = 3;
}  // namespace sfpi

// p_sfpu register indices. Global namespace, matching emule's existing p_sfpu in
// eltwise_unary/exp.h. Values mirror upstream ckernel_instr_params.h
// (LCONST_0 = LReg9 = 0.0, LCONST_1 = LReg10 = 1.0).
namespace p_sfpu {
constexpr std::uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3,
                        LREG4 = 4, LREG5 = 5, LREG6 = 6, LREG7 = 7;
constexpr std::uint32_t LCONST_0 = 9, LCONST_1 = 10;
}  // namespace p_sfpu

namespace ckernel {

// DEST/Src address-mode programming constants + descriptor. Config-only on silicon;
// the descriptor's .set() programs an ADDR_MOD slot. No-op here (DST addressing is
// driven explicitly by SFPLOAD/SFPSTORE + INCRWC through the sfpi cursor).
constexpr std::uint8_t ADDR_MOD_0 = 0, ADDR_MOD_1 = 1, ADDR_MOD_2 = 2, ADDR_MOD_3 = 3,
                       ADDR_MOD_4 = 4, ADDR_MOD_5 = 5, ADDR_MOD_6 = 6, ADDR_MOD_7 = 7;

struct addr_mod_t {
    struct incr_t { std::int16_t incr = 0; };
    incr_t srca{};
    incr_t srcb{};
    incr_t dest{};
    void set(std::uint32_t /*mod_index*/) const {}
};

// SFPLOAD/SFPSTORE data-format modifier. Values mirror upstream llk_defs.h.
enum class InstrModLoadStore : std::uint8_t {
    DEFAULT       = 0,
    FP16A         = 1,
    FP16B         = 2,
    FP32          = 3,
    INT32         = 4,
    INT8          = 5,
    LO16          = 6,
    HI16          = 7,
    INT32_2S_COMP = 12,
    INT8_2S_COMP  = 13,
    LO16_ONLY     = 14,
    HI16_ONLY     = 15,
};

struct p_setrwc {
    constexpr static std::uint32_t CLR_NONE = 0x0;
    constexpr static std::uint32_t CR_D     = 0x4;
    constexpr static std::uint32_t SET_D    = 0x4;
};

// ===========================================================================
// Faithful 32-lane SFPU interpreter for the raw-TTI exp polynomial.
// ===========================================================================
namespace __tti {

constexpr std::uint32_t NLANE = 32;

// LReg file: 16 registers x 32 lanes of raw 32-bit storage. LReg9 (LCONST_0) and
// LReg10 (LCONST_1) are silicon constant registers (0.0 / 1.0) — read-only.
inline thread_local std::uint32_t lreg[16][NLANE] = {};

// Per-lane condition flags (the SFPU CC). SFPSETCC sets them + enables masking;
// SFPENCC clears them + re-enables all lanes. While enabled, register writebacks
// only land on flagged-true lanes.
inline thread_local bool laneflag[NLANE] = {};
inline thread_local bool use_flags = false;

inline bool lane_active(std::uint32_t i) { return !use_flags || laneflag[i]; }

inline float as_f(std::uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
inline std::uint32_t as_u(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return b; }

// Read LReg as raw bits / float, honoring the constant registers.
inline std::uint32_t ru(std::uint32_t idx, std::uint32_t lane) {
    if (idx == p_sfpu::LCONST_0) return 0x00000000u;  // 0.0f
    if (idx == p_sfpu::LCONST_1) return 0x3f800000u;  // 1.0f
    if (idx >= 16) return 0u;  // bounds guard, symmetric with wu() (idx is always LREG0..7 in valid code)
    return lreg[idx][lane];
}
inline float rf(std::uint32_t idx, std::uint32_t lane) { return as_f(ru(idx, lane)); }

// Write LReg (constants are read-only on silicon — ignore writes to them).
inline void wu(std::uint32_t idx, std::uint32_t lane, std::uint32_t bits) {
    if (idx == p_sfpu::LCONST_0 || idx == p_sfpu::LCONST_1 || idx >= 16) return;
    lreg[idx][lane] = bits;
}
inline void wf(std::uint32_t idx, std::uint32_t lane, float f) { wu(idx, lane, as_u(f)); }

// Round to nearest integer, ties away from zero (the SFPU stochastic-round-EVEN
// behavior for FP32->int is round-half-away, not C rint's round-half-even).
inline float round_ties_away(float v) {
    return (v >= 0.0f) ? std::floor(v + 0.5f) : std::ceil(v - 0.5f);
}

// ---- instructions ----------------------------------------------------------

// SFPMAD(a, b, c, d, mod): d = a*b + c (fp32, lanewise). mod 0 only (no NEGATE on WH).
inline void sfpmad(std::uint32_t a, std::uint32_t b, std::uint32_t c, std::uint32_t d, std::uint32_t /*mod*/) {
    for (std::uint32_t i = 0; i < NLANE; ++i)
        if (lane_active(i)) wf(d, i, rf(a, i) * rf(b, i) + rf(c, i));
}

// SFPLOADI(vd, mod, imm16): build/patch the 32-bit LReg from a 16-bit immediate.
//   mod 0  -> bf16 immediate widened to fp32  (imm << 16)
//   mod 10 -> write LOW 16 bits, keep high     ; mod 8 -> write HIGH 16, keep low
inline void sfploadi(std::uint32_t vd, std::uint32_t mod, std::uint32_t imm) {
    for (std::uint32_t i = 0; i < NLANE; ++i) {
        if (!lane_active(i)) continue;
        std::uint32_t cur = ru(vd, i), nv;
        if (mod == 0)        nv = (imm & 0xFFFFu) << 16;
        else if (mod == 10)  nv = (cur & 0xFFFF0000u) | (imm & 0xFFFFu);
        else if (mod == 8)   nv = (cur & 0x0000FFFFu) | ((imm & 0xFFFFu) << 16);
        else                 nv = imm;
        wu(vd, i, nv);
    }
}

// TT_SFPADDI(imm16, vd, mod): vd += bf16(imm16) widened to fp32.
inline void sfpaddi(std::uint32_t imm, std::uint32_t vd, std::uint32_t /*mod*/) {
    const float add = as_f((imm & 0xFFFFu) << 16);
    for (std::uint32_t i = 0; i < NLANE; ++i)
        if (lane_active(i)) wf(vd, i, rf(vd, i) + add);
}

// SFP_STOCH_RND(rnd, imm, _, src, dst, mod1): convert src -> dst per mod1.
//   1 = FP32_TO_FP16B (round-nearest to bf16 precision, kept as fp32)
//   2 = FP32_TO_UINT8 (|x| -> clamp [0,255], sign-magnitude int bits)
//   3 = FP32_TO_INT8  (x   -> clamp [-127,127], sign-magnitude int bits)
inline void stochrnd(int /*rnd*/, int /*imm*/, int /*unused*/,
                     std::uint32_t src, std::uint32_t dst, unsigned mod1) {
    for (std::uint32_t i = 0; i < NLANE; ++i) {
        if (!lane_active(i)) continue;
        const float v = rf(src, i);
        if (mod1 == 3) {  // FP32_TO_INT8
            int k = static_cast<int>(round_ties_away(v));
            if (k > 127) k = 127; if (k < -127) k = -127;
            std::uint32_t mag = static_cast<std::uint32_t>(k < 0 ? -k : k) & 0x7FFFFFFFu;
            wu(dst, i, (k < 0 ? 0x80000000u : 0u) | mag);
        } else if (mod1 == 2) {  // FP32_TO_UINT8
            int k = static_cast<int>(round_ties_away(std::fabs(v)));
            if (k > 255) k = 255; if (k < 0) k = 0;
            wu(dst, i, static_cast<std::uint32_t>(k));
        } else if (mod1 == 1) {  // FP32_TO_FP16B (round-to-nearest-even to bf16)
            std::uint32_t b = as_u(v);
            std::uint32_t r = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
            wu(dst, i, r);
        }
    }
}

// SFPCAST(r0, r1, mod): sign-magnitude int32 -> fp32 (mod 0). Both args are the
// same reg in the exp polynomial, so r0/r1 order is immaterial here.
inline void sfpcast(std::uint32_t r0, std::uint32_t r1, std::uint32_t /*mod*/) {
    for (std::uint32_t i = 0; i < NLANE; ++i) {
        if (!lane_active(i)) continue;
        std::uint32_t b = ru(r1, i);
        float f = static_cast<float>(b & 0x7FFFFFFFu);
        if (b & 0x80000000u) f = -f;
        wf(r0, i, f);
    }
}

// SFPSETEXP(imm, vc, vd, mod): set vd's fp32 exponent field from the low byte of
// vd's own integer value (the biased exponent k+127), taking sign+mantissa from vc.
// With vc=LCONST_0 (=0) this yields exactly 2^k (sign 0, mantissa 0).
inline void sfpsetexp(std::uint32_t /*imm*/, std::uint32_t vc, std::uint32_t vd, std::uint32_t /*mod*/) {
    for (std::uint32_t i = 0; i < NLANE; ++i) {
        if (!lane_active(i)) continue;
        std::uint32_t exp_byte = ru(vd, i) & 0xFFu;
        std::uint32_t base = ru(vc, i) & 0x807FFFFFu;  // sign + mantissa from vc
        wu(vd, i, base | (exp_byte << 23));
    }
}

// SFPSETCC(imm, vc, _, mod1): set per-lane condition flags from a compare of vc, and
// enable lane masking. mod1: 0=<0, 2=!=0, 4=>=0, 6==0.
inline void sfpsetcc(std::uint32_t /*imm*/, std::uint32_t vc, std::uint32_t /*vc2*/, std::uint32_t mod1) {
    for (std::uint32_t i = 0; i < NLANE; ++i) {
        float v = rf(vc, i);
        bool cond;
        switch (mod1) {
            case 0:  cond = (v < 0.0f);  break;
            case 2:  cond = (v != 0.0f); break;
            case 4:  cond = (v >= 0.0f); break;
            case 6:  cond = (v == 0.0f); break;
            default: cond = true;        break;
        }
        laneflag[i] = cond;
    }
    use_flags = true;
}

// SFPENCC(...): clear lane flags, re-enable all lanes.
inline void sfpencc(int = 0, int = 0, int = 0, int = 0) {
    for (std::uint32_t i = 0; i < NLANE; ++i) laneflag[i] = true;
    use_flags = false;
}

inline void sfpnop() {}

// SFPLOAD/SFPSTORE address DST via the sfpi cursor + the shared lane->element map,
// so the col-0 first-column walk reuses the proven sfpi addressing. The data-format
// (fmt) and addr_mod args are ignored: emule's DST is fp32, and DST advancement is
// driven explicitly by INCRWC, not by addr_mod post-increment.
template <typename F, typename A>
inline void sfpload(std::uint32_t vd, F /*fmt*/, A /*addr_mod*/, int imm) {
    float* base = ::sfpi::__emule_sfpi_active_dst();
    const std::uint32_t addr = ::__emule_sfpi_cursor + static_cast<std::uint32_t>(imm);
    for (std::uint32_t i = 0; i < NLANE; ++i)
        wu(vd, i, as_u(base[::sfpi::__emule_sfpi_lane_index(addr, i)]));
}

template <typename F, typename A>
inline void sfpstore(std::uint32_t vc, F /*fmt*/, A /*addr_mod*/, int imm) {
    float* base = ::sfpi::__emule_sfpi_active_dst();
    const std::uint32_t addr = ::__emule_sfpi_cursor + static_cast<std::uint32_t>(imm);
    for (std::uint32_t i = 0; i < NLANE; ++i)
        if (lane_active(i) && ::sfpi::__emule_sfpi_mask[i])
            base[::sfpi::__emule_sfpi_lane_index(addr, i)] = rf(vc, i);
}

// INCRWC(cr, d, b, a): advance the DST row write counter by `d` (the sfpi cursor),
// matching dst_reg += d/2 in the sfpi path.
inline void incrwc(int /*cr*/, int d, int /*b*/, int /*a*/) {
    ::__emule_sfpi_cursor += static_cast<std::uint32_t>(d);
}

}  // namespace __tti

// math:: address-mode base programming. On silicon this configures the SFPU's DST
// address-mode base. emule uses it as the hook that aims the sfpi cursor at the DST
// region for the raw-TTI direct-SFPU path (recip_tile_first_column_wh_idst0_direct).
namespace math {
inline void set_addr_mod_base() {
    ::__emule_sfpi_dst_base = &__emule_dst[0][0];
    ::__emule_sfpi_cursor = 0;
    ::sfpi::__emule_sfpi_mask.fill(true);
    ::__emule_sfpi_first_col_mode = true;
}
inline void clear_addr_mod_base() {
    ::__emule_sfpi_dst_base = nullptr;
    ::__emule_sfpi_cursor = 0;
    ::__emule_sfpi_first_col_mode = false;
}
}  // namespace math

}  // namespace ckernel

// ---- Raw SFPU intrinsic macros -> functional interpreter -------------------
// Wormhole exp-polynomial instruction set is modelled; Blackhole-only and
// config-only ops remain no-ops.
//
// This header is pulled in (via exp.h) AFTER sfpi.h / common.h, which transitively
// include ckernel_ops.h — and ckernel_ops.h defines some of these macros (e.g.
// TTI_SFPLOADI -> sfpi::__emule_sfploadi) behind an #ifndef. So `#undef` each before
// (re)defining: this header is the authoritative TTI surface for the exp-polynomial
// path, the override is intentional, and it avoids a macro-redefinition warning.
#undef TTI_SFPNOP
#undef TTI_INCRWC
#undef TT_SFPADDI
#undef TTI_SFPLOADI
#undef TTI_SFPLOAD
#undef TTI_SFPSTORE
#undef TTI_SFPMAD
#undef TTI_SFPCAST
#undef TTI_SFPENCC
#undef TTI_SFPSETCC
#undef TTI_SFPSETEXP
#undef TTI_SFP_STOCH_RND
#undef TTI_SFPGT
#undef TTI_SFPARECIP
#undef TTI_SETRWC
#undef TTI_STALLWAIT
#define TTI_SFPNOP             (::ckernel::__tti::sfpnop())
#define TTI_INCRWC(...)        (::ckernel::__tti::incrwc(__VA_ARGS__))
#define TT_SFPADDI(...)        (::ckernel::__tti::sfpaddi(__VA_ARGS__))
#define TTI_SFPLOADI(...)      (::ckernel::__tti::sfploadi(__VA_ARGS__))
#define TTI_SFPLOAD(...)       (::ckernel::__tti::sfpload(__VA_ARGS__))
#define TTI_SFPSTORE(...)      (::ckernel::__tti::sfpstore(__VA_ARGS__))
#define TTI_SFPMAD(...)        (::ckernel::__tti::sfpmad(__VA_ARGS__))
#define TTI_SFPCAST(...)       (::ckernel::__tti::sfpcast(__VA_ARGS__))
#define TTI_SFPENCC(...)       (::ckernel::__tti::sfpencc(__VA_ARGS__))
#define TTI_SFPSETCC(...)      (::ckernel::__tti::sfpsetcc(__VA_ARGS__))
#define TTI_SFPSETEXP(...)     (::ckernel::__tti::sfpsetexp(__VA_ARGS__))
#define TTI_SFP_STOCH_RND(...) (::ckernel::__tti::stochrnd(__VA_ARGS__))
// Blackhole-only fast paths (SFPARECIP/SFPGT) and config-only ops: still no-ops.
#define TTI_SFPGT(...)         ((void)0)
#define TTI_SFPARECIP(...)     ((void)0)
#define TTI_SETRWC(...)        ((void)0)
#define TTI_STALLWAIT(...)     ((void)0)
