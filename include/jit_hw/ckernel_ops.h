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

// SFPCONFIG programs the shared programmable-constant LRegs (11-14) from LReg0
// via the CONFIG path. The tanh/sigmoid LUT inits use SFPLOADI directly and do
// NOT use this; it is a no-op so load_config.h's _sfpu_load_config32_ /
// _init_sfpu_config_reg compile. A future deep op that programs constants this
// way must implement the LReg0 -> creg copy here (don't leave it silent).
#ifndef TTI_SFPCONFIG
#define TTI_SFPCONFIG(imm16_math, config_dest, instr_mod1) ((void)0)
#endif
