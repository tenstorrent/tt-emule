// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hosted JIT shadow for api/tensor/tensor_accessor_args.h.
//
// Emule provides TensorAccessorArgs from tensor_accessor.h. Keep this header in
// the include path so generated kernels that include the upstream split headers
// do not pull duplicate tt-metal definitions into the same translation unit.
#include "api/tensor/tensor_accessor.h"
