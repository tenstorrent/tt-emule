// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hosted JIT shadow for api/tensor/tensor_binding_token.h.
//
// TensorBindingToken is defined by emule's tensor_accessor.h shim. This file
// preserves the generated-kernel include surface while avoiding conflicts with
// tt-metal's in-tree TensorBindingToken definition.
#include "api/tensor/tensor_accessor.h"
