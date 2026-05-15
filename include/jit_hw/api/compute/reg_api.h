#pragma once
// Shim: tt-metal's reg_api.h defines tile_regs_acquire/commit/wait/release.
// tt-emule's common.h already provides these (with emule-aware DST zero-init).
// This shim intercepts `#include "api/compute/reg_api.h"` so the JIT compile
// uses tt-emule's definitions, avoiding the ambiguity that arises when the
// preprocessor falls through to tt-metal's version (which would coexist with
// tt-emule's, producing "call ... is ambiguous" errors).
#include "common.h"
