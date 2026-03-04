#pragma once
// Conditional SFPU includes for emulation (mirrors real sfpu_split_includes.h).
// Only includes headers for operations that have emule stub implementations.

#if SFPU_OP_BINOP_WITH_SCALAR_INCLUDE
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

#if SFPU_OP_FILL_INCLUDE
#include "api/compute/eltwise_unary/fill.h"
#endif

#if SFPU_OP_UNARY_COMP_INCLUDE
#include "api/compute/eltwise_unary/comp.h"
#endif
