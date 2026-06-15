// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim. Upstream kernel
//   tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp
// includes `cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp`
// (with the leading `cpp/`), which only resolves under `-I tt-metal/ttnn/`. emule's JIT
// include paths use `-I tt-metal/ttnn/cpp/`, so the upstream form falls through without
// resolution. Redirect to the real upstream header (portable C++, only <cstdint>).

#include "ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp"
