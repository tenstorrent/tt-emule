// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Stub of silicon's tt-metalium/experimental/fabric/fabric_edm_types.hpp. emule has no ethernet model.
// Single source of truth for tt::tt_fabric::Topology (ccl_host_types.hpp does
// `using tt::tt_fabric::Topology;`; __emule_fabric_stubs.h includes this header rather than redefining it).
#include <cstdint>
namespace tt::tt_fabric {
// Values MUST match silicon (NeighborExchange=0, Linear=1, ...) — kernels static_cast a host-passed
// compile-time arg to this enum, so a mismatch silently picks the wrong topology branch.
enum class Topology : uint8_t { NeighborExchange = 0, Linear = 1, Ring = 2, Mesh = 3, Torus = 4 };
}  // namespace tt::tt_fabric
