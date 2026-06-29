// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of silicon's tt-metalium/experimental/fabric/fabric_edm_types.hpp.
// Real silicon defines EDM (Ethernet Data Mover) packet types and routing
// constants. emule has no ethernet model — short-circuit the chain.
// Add types/enums only if a upstream fabric kernel references them by name.

#include <cstdint>

namespace tt::tt_fabric {

// Mirrors silicon's tt::tt_fabric::Topology. ccl_host_types.hpp re-exports it as
// ttnn::ccl::Topology, which prefill matmul kernels (minimal_matmul's
// strided_all_gather_async fused receiver) reference by name and reconstruct from
// a runtime arg via static_cast<Topology>(uint32_t) — so the enumerator values
// must match silicon. On single-chip emule there is no gather, so the value only
// has to compile and round-trip correctly through the cast.
enum class Topology { NeighborExchange = 0, Linear = 1, Ring = 2, Mesh = 3, Torus = 4 };

}  // namespace tt::tt_fabric
