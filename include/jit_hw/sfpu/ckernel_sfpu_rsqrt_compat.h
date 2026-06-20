// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Forwarder so the metal-layer ckernel_sfpu_sqrt.h's
//   #include "sfpu/ckernel_sfpu_rsqrt_compat.h"
// resolves on emule's JIT -I set (jit_hw is first; the LLK common/inc dir is not
// on the path). Only the legacy_compat=false path runs, so the compat bodies are
// parsed against emule's sfpi backend but never instantiated.

#if defined(ARCH_BLACKHOLE)
#include "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h"
#elif defined(ARCH_WORMHOLE)
#include "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_rsqrt_compat.h"
#else
#error "deep-default sqrt requires ARCH_BLACKHOLE or ARCH_WORMHOLE"
#endif
