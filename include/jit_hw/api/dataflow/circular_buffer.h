// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// D2M-generated kernels use the path "api/dataflow/circular_buffer.h".
// Forward to the jit_hw emulation stub which implements the full
// experimental::CircularBuffer interface backed by cb_api.h.
// D2M emits "CircularBuffer cb_ctarg_N(...)" without namespace qualification,
// so we bring it into the global namespace with a using-declaration.
#include "experimental/circular_buffer.h"
using experimental::CircularBuffer;
