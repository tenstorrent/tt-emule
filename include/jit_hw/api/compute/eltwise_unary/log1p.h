#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/eltwise_unary/log1p.h"`.
// log1p_tile / log1p_tile_init are SFPU ops; the emulator's eltwise_unary path
// handles them through the common SFPU framework. Empty shim suffices when
// only the include is required for name lookup.
#include "eltwise_unary.h"
