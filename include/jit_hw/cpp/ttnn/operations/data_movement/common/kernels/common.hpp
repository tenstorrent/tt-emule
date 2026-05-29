// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Redirect shim. Upstream kernel
//   tt-metal/ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/redistribute_pages_row_major_reader.cpp
// includes `cpp/ttnn/operations/data_movement/common/kernels/common.hpp`
// (with the leading `cpp/`), which only resolves under `-I tt-metal/ttnn/`.
// emule's JIT include paths use `-I tt-metal/ttnn/cpp/` instead, so the
// upstream form falls through without resolution. Redirect to the same
// kernel-lib common header reachable through emule's resolved prefix.

#include "ttnn/operations/data_movement/common/kernels/common.hpp"
