// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Empty shim. The reduce_w_neg / reduce_h_neg compute kernels include this
// header but never reference any symbol from it (likely a refactor leftover
// in upstream). Providing a no-op resolution lets the JIT compile succeed
// without pulling in the real ~700-line LLK header.
//
// TODO: file an upstream cleanup PR to drop the include from
// ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_*_neg.cpp.
