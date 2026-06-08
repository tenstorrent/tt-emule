// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon NOC parameters. Real silicon defines NOC field
// widths and helpers (NOC_0_X, NOC_ADDR_LOCAL_BITS, NOC_ADDR_NODE_ID_BITS,
// etc.). emule's dataflow_api_addrgen.h already provides these as needed
// for NOC address encoding; this file exists so the silicon risc_common header
// can include it without failing. Add macros only if a upstream kernel
// references them and the existing emule address-gen header doesn't
// satisfy.

#include <cstdint>
