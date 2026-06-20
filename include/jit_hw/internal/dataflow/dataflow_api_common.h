// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for the firmware `internal/dataflow/dataflow_api_common.h`.
// Metal 2.0 dataflow kernels include this directly; its KERNEL_BUILD branch
// defines noc_index/noc_mode, which jit_kernel_stubs.hpp (in every wrapper
// prelude) already defines with the identical firmware formula. Pulling the real
// header in here too would be a redefinition. emule already supplies the pieces
// kernels use (noc_index/noc_mode + bank tables via jit_kernel_stubs.hpp, NOC
// VC/clear-mask macros via api/dataflow/dataflow_api.h), so this shim is empty.
