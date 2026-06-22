// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for ckernel_ops.h. The real header emits raw Tensix instructions
// (instrn_buffer writes / INSTRUCTION_WORD) that don't exist on x86. Emule only
// needs the SFPU immediate-load + config macros used by the LUT init path
// (ckernel_sfpu_load_config.h, pulled by the deep tanh/sigmoid ckernels). It is
// resolved ahead of the real header because jit_hw is the first JIT -I path.
#include "sfpi.h"

// SFPLOADI writes a uniform immediate into an LReg (insmod selects lower/upper/
// zero-extend); emule routes it to the LReg bit storage backing lut()/lut2().
#ifndef TT_SFPLOADI
#define TT_SFPLOADI(lreg_ind, instr_mod0, imm16)  ::sfpi::__emule_sfploadi((lreg_ind), (instr_mod0), (imm16))
#endif
#ifndef TTI_SFPLOADI
#define TTI_SFPLOADI(lreg_ind, instr_mod0, imm16) ::sfpi::__emule_sfploadi((lreg_ind), (instr_mod0), (imm16))
#endif

// Raw-TTI SFP* instruction set for the Wormhole exp polynomial (SDPA's
// calculate_exponential_polynomial, exp_approx_mode=false). All route to the
// single deep-SFPU backend in sfpi.h (one LReg file + one CC mask). Genuinely-
// unmodeled-but-reachable ops fail loud; true config/sync ops are no-ops.
#ifndef TTI_SFPNOP
#define TTI_SFPNOP             (::sfpi::__emule_sfp_nop())
#endif
#ifndef TTI_INCRWC
#define TTI_INCRWC(...)        (::sfpi::__emule_sfp_incrwc(__VA_ARGS__))
#endif
#ifndef TT_SFPADDI
#define TT_SFPADDI(...)        (::sfpi::__emule_sfp_addi(__VA_ARGS__))
#endif
#ifndef TTI_SFPMAD
#define TTI_SFPMAD(...)        (::sfpi::__emule_sfp_mad(__VA_ARGS__))
#endif
#ifndef TTI_SFPLOAD
#define TTI_SFPLOAD(...)       (::sfpi::__emule_sfp_load(__VA_ARGS__))
#endif
#ifndef TTI_SFPSTORE
#define TTI_SFPSTORE(...)      (::sfpi::__emule_sfp_store(__VA_ARGS__))
#endif
#ifndef TTI_SFPCAST
#define TTI_SFPCAST(...)       (::sfpi::__emule_sfp_cast(__VA_ARGS__))
#endif
#ifndef TTI_SFPENCC
#define TTI_SFPENCC(...)       (::sfpi::__emule_sfp_encc(__VA_ARGS__))
#endif
#ifndef TTI_SFPSETCC
#define TTI_SFPSETCC(...)      (::sfpi::__emule_sfp_setcc(__VA_ARGS__))
#endif
#ifndef TTI_SFPSETEXP
#define TTI_SFPSETEXP(...)     (::sfpi::__emule_sfp_setexp(__VA_ARGS__))
#endif
#ifndef TTI_SFP_STOCH_RND
#define TTI_SFP_STOCH_RND(...) (::sfpi::__emule_sfp_stoch_rnd(__VA_ARGS__))
#endif
// Reachable-but-unmodeled SFPU ops (Blackhole-only fast paths). Fail loud rather
// than silently no-op (matching the TTI_SFPCONFIG convention) so an op that
// actually needs them aborts with a clear message instead of miscomputing.
#ifndef TTI_SFPGT
#define TTI_SFPGT(...)         ::sfpi::__emule_sfpu_unsupported("TTI_SFPGT (not modeled)")
#endif
#ifndef TTI_SFPARECIP
#define TTI_SFPARECIP(...)     ::sfpi::__emule_sfpu_unsupported("TTI_SFPARECIP (not modeled)")
#endif
// True config / sync ops — genuine no-ops in emule (DST addressing is explicit).
#ifndef TTI_SETRWC
#define TTI_SETRWC(...)        ((void)0)
#endif
#ifndef TTI_STALLWAIT
#define TTI_STALLWAIT(...)     ((void)0)
#endif

// SFPCONFIG programs the shared programmable-constant LRegs (11-14) from LReg0
// via the CONFIG path. The tanh/sigmoid LUT inits use SFPLOADI directly and do
// NOT use this, so it is not yet modeled. It FAILS LOUD (not a silent no-op):
// if a deep op actually executes SFPCONFIG, it aborts with a clear message
// rather than silently producing wrong constants. (load_config.h's
// _sfpu_load_config32_ / _init_sfpu_config_reg still compile; they only abort
// if invoked.) Implement the LReg0 -> creg copy here when a target op needs it.
#ifndef TTI_SFPCONFIG
#define TTI_SFPCONFIG(imm16_math, config_dest, instr_mod1) \
    ::sfpi::__emule_sfpu_unsupported("TTI_SFPCONFIG (SFPU config-reg path not modeled)")
#endif
