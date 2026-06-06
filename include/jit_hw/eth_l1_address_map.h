// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon ethernet L1 address map. Real silicon defines per-arch
// ethernet L1 memory layout constants under `eth_l1_mem::address_map`.
// emule has no ethernet sub-engine, so most constants are inert here —
// but tt-metal's risc_common.h unconditionally #includes this file, so we
// need it to exist on the JIT include path.
//
// Add real constants only as upstream kernels (or other emule consumers) demand them.

#include <cstdint>

namespace eth_l1_mem::address_map {

constexpr std::uint32_t MAX_NUM_CONCURRENT_TRANSACTIONS = 4;

}  // namespace eth_l1_mem::address_map
