// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Stub of silicon's tt-metalium/experimental/fabric/fabric_edm_types.hpp. emule has no ethernet model.
// Provides tt::tt_fabric::Topology (ccl_host_types.hpp does `using tt::tt_fabric::Topology;`). Guarded so
// it agrees with the same enum in __emule_fabric_stubs.h (no redefinition regardless of include order).
#include <cstdint>
namespace tt::tt_fabric {
#ifndef __EMULE_TT_FABRIC_TOPOLOGY_DEFINED
#define __EMULE_TT_FABRIC_TOPOLOGY_DEFINED
// Values MUST match silicon (NeighborExchange=0, Linear=1, ...) — kernels static_cast a host-passed
// compile-time arg to this enum, so a mismatch silently picks the wrong topology branch.
enum class Topology : uint8_t { NeighborExchange = 0, Linear = 1, Ring = 2, Mesh = 3, Torus = 4 };
#endif
}  // namespace tt::tt_fabric
