// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's `hadamard` compute header. The emule JIT puts jit_hw
// FIRST on the -I list, so this shadows blaze's silicon header — whose
// hadamard_h128_init/_tile/_uninit bodies expand to the custom narrow-MOP LLKs
// (llk_unpack_hadamard_h128* / llk_math_hadamard_h128*) emule can't consume —
// and routes to the high-level emule model. blaze stays pristine.
#include "api/compute/hadamard.h"
