// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shadow for blaze's `custom_mm` compute header. The emule JIT puts
// jit_hw FIRST on the -I list, so this shadows blaze's silicon header — whose
// custom_mm_block_init/_block/_uninit bodies expand to the raw custom-mm LLK
// state machine (llk_unpack_AB_custom_mm*, llk_math_custom_mm*, the dense-pack
// cfg_reg_rmw_tensix<PCK0_*_Wstride_RMW> stride RMW) that emule can't consume —
// and routes the custom_mm_block_* API to the high-level emule model. blaze
// stays pristine. (split_acc / finalize / dense_packing are silicon packer-
// scheduling concerns; emule accumulates each k-partial straight into DST, so
// they are faithful no-ops here — see docs/tilize-untilize-pack.md.)
#include "api/compute/matmul_fused_act_emule.h"
