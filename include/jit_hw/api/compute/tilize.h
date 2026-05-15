// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// tt-emule shim for tt-mlir-emitted `#include "api/compute/tilize.h"`.
// `tilize_init`, `tilize_block`, etc. are defined in `llk_defs.h`, which is
// pulled in by `jit_kernel_stubs.hpp`. This file intercepts the include so the
// JIT compile's `-I` search doesn't fall through to the upstream tt-metal
// `tilize.h` (which references unstubbed LLK APIs).
#include <cstdint>
