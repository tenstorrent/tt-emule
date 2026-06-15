// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Compile-only stubs for the raw Tensix SFPU intrinsic surface (TTI_SFP* macros)
// plus the addr_mod / p_sfpu / InstrModLoadStore types they reference.
//
// These appear only in raw-TTI code paths that emule never EXECUTES: the SDPA
// compute kernel's `calculate_exponential_polynomial` (the `if constexpr` ELSE of
// the exp_approx_mode path — never instantiated, since emule's tests run
// exp_approx_mode=true and take the sfpi-based `_ckernel_sfpu_exp_accurate_`
// branch) and `INSERT_SFPNOP()`. Because the names are non-dependent, C++ still
// requires them to be DECLARED for the template to parse — but never to function.
// No-op definitions are therefore faithful: a real SFPU ISA interpreter would only
// be needed if a raw-TTI path were actually instantiated (e.g. exp_approx_mode=false).

#include <cstdint>

// The SDPA exp polynomial kernel declares `constexpr float M_LN2 = ...` as a local
// identifier. On bare-metal silicon M_LN2 is not a macro; emule's hosted <cmath>
// pulls glibc's `#define M_LN2 0.69314...` (math.h), which would mangle that
// declaration into a parse error. Undef to restore the silicon condition. (cmath is
// already included upstream of this header via exp.h.)
#ifdef M_LN2
#undef M_LN2
#endif

// ---- Raw SFPU intrinsic macros (no-ops; raw-TTI paths are never instantiated here) ----
#define TTI_SFPNOP             ((void)0)
#define TTI_INCRWC(...)        ((void)0)
#define TT_SFPADDI(...)        ((void)0)
#define TTI_SFPLOADI(...)      ((void)0)
#define TTI_SFPLOAD(...)       ((void)0)
#define TTI_SFPSTORE(...)      ((void)0)
#define TTI_SFPMAD(...)        ((void)0)
#define TTI_SFPCAST(...)       ((void)0)
#define TTI_SFPENCC(...)       ((void)0)
#define TTI_SFPGT(...)         ((void)0)
#define TTI_SFPSETCC(...)      ((void)0)
#define TTI_SFPSETEXP(...)     ((void)0)
#define TTI_SFPARECIP(...)     ((void)0)
#define TTI_SFP_STOCH_RND(...) ((void)0)
#define TTI_SETRWC(...)        ((void)0)
#define TTI_STALLWAIT(...)     ((void)0)

// p_sfpu register indices. Global namespace, matching emule's existing p_sfpu in
// eltwise_unary/exp.h (silicon keeps it in ckernel::, but emule's surface is global
// and that is where callers resolve `p_sfpu::LREGn`). Values mirror upstream
// ckernel_instr_params.h.
namespace p_sfpu {
constexpr std::uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3,
                        LREG4 = 4, LREG5 = 5, LREG6 = 6, LREG7 = 7;
constexpr std::uint32_t LCONST_0 = 9, LCONST_1 = 10;
}  // namespace p_sfpu

namespace ckernel {

// DEST/Src address-mode programming constants + descriptor. Config-only on silicon;
// the descriptor's .set() programs an ADDR_MOD slot. No-op here.
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

// Instruction-modifier parameter constants for raw SETRWC / STALLWAIT. They are
// only ever passed to the no-op TTI_* macros above, so the values are opaque to
// emule (never interpreted) — names mirror upstream ckernel_instr_params.h.
// p_setrwc instruction-modifier constants (p_stall already lives in ckernel.h).
struct p_setrwc {
    constexpr static std::uint32_t CLR_NONE = 0x0;
    constexpr static std::uint32_t CR_D     = 0x4;
    constexpr static std::uint32_t SET_D    = 0x4;
};

// math:: address-mode base programming — HW addrmod register config; no-op here.
namespace math {
inline void set_addr_mod_base() {}
inline void clear_addr_mod_base() {}
}  // namespace math

}  // namespace ckernel
