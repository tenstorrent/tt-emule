// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim. Upstream kernel
//   tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp
// includes `cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp`
// (with the leading `cpp/`), which only resolves under `-I tt-metal/ttnn/`.
// emule's JIT include paths use `-I tt-metal/ttnn/cpp/` instead, so the
// upstream form falls through without resolution. Redirect to the same
// real upstream header reachable through emule's resolved prefix — the
// portable chunk/tile index helpers stay owned by tt-metal, compiled verbatim.

#include "ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp"
