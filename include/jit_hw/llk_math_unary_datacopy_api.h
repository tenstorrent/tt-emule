// SPDX-License-Identifier: Apache-2.0
// Emule shadow for the silicon LLK math unary datacopy API.
// Custom upstream kernels (e.g. create_q_heads/custom_tilize.h) include this
// header directly under `#ifdef TRISC_MATH`. In emule that guard is also
// never defined, but some kernels include unconditionally — provide the
// shadow either way to forward to the existing eltwise datacopy shim.
#pragma once

#include "llk_math_eltwise_unary_datacopy.h"
