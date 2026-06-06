// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _RISC_COMMON_H_
#define _RISC_COMMON_H_

// Stub of silicon's internal/tt-1xx/risc_common.h. The real file defines
// RISC-V firmware-side helpers (NOC_X/NOC_Y macros, register pointer
// helpers, MMIO wrappers) that pull in a deep chain of hardware register
// layout headers. emule's x86 JIT path doesn't model any of that, so we
// stop the chain here and provide stubs only for the symbols actual
// emule-consumed kernels reference.
//
// Add macros / inline helpers as needed (search for the specific symbol
// in the failing JIT compile log and add it here).

#include <cstdint>

// NOC coordinate helpers — silicon uses NOC_0_X / NOC_1_X tables; emule's
// dataflow_api_addrgen.h provides the logical-to-physical mapping it
// needs. These no-op fallbacks let any kernel that incidentally
// references NOC_X / NOC_Y compile.
#ifndef NOC_X
#define NOC_X(x) (x)
#endif
#ifndef NOC_Y
#define NOC_Y(y) (y)
#endif

// tt_reg_ptr: silicon uses an address-space attribute on RISC-V pointers
// to MMIO regions. On x86 it's a no-op type qualifier.
#ifndef tt_reg_ptr
#define tt_reg_ptr
#endif

// NOC mode constants — silicon firmware sets these per-RISC build. upstream
// kernels read `noc_mode` for dispatch policy.
#ifndef DM_DEDICATED_NOC
#define DM_DEDICATED_NOC 0
#endif
#ifndef DM_DYNAMIC_NOC
#define DM_DYNAMIC_NOC 1
#endif
// NOTE: the `noc_mode` variable itself is owned by jit_kernel_stubs.hpp
// (`inline constexpr int noc_mode = DM_DYNAMIC_NOC;`). We deliberately do NOT
// redefine it here: the `#ifndef noc_mode` guard only tests *macros*, not
// variables, so defining it in both headers would be an ODR/redefinition error
// in any TU including both — and the values would disagree (several upstream ops
// `static_assert(noc_mode == DM_DYNAMIC_NOC)`). jit_kernel_stubs.hpp is part of
// every kernel TU prelude, so the symbol is always available where kernels need it.

#endif  // _RISC_COMMON_H_
