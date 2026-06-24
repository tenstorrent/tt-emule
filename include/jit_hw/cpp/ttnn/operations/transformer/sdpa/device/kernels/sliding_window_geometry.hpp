// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim (see sdpa_streaming_qktv.hpp in this directory). Upstream SDPA
//   tt-metal/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp
// includes `cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp`
// (with the leading `cpp/`), which only resolves under `-I tt-metal/ttnn/`. emule's JIT
// include paths use `-I tt-metal/ttnn/cpp/`, so the upstream form falls through. Redirect
// to the real upstream header (portable C++).

#include "ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp"
