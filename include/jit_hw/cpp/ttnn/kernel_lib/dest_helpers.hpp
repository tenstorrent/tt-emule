// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim. Upstream kernel
//   tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp
// includes `cpp/ttnn/kernel_lib/dest_helpers.hpp` (with the leading `cpp/`), which only
// resolves under `-I tt-metal/ttnn/`. emule's JIT include paths use `-I tt-metal/ttnn/cpp/`
// instead, so the upstream form falls through without resolution. Redirect to the real
// upstream header reachable through emule's resolved prefix.
//
// The real header is portable C++ (DEST capacity / fp32-accum detection) that keys off
// JIT-define macros — emule already supplies the supporting infra: ENABLE_FP32_DEST_ACC /
// DST_SYNC_FULL fallback defines (jit_kernel_stubs.hpp) and ckernel::DstSync (llk_types.h).

#include "ttnn/kernel_lib/dest_helpers.hpp"
