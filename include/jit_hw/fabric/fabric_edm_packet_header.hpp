// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Stub of the silicon fabric EDM packet header. emule has no
// fabric model (multichip is out of scope) — this file exists so silicon
// fabric headers that #include it transitively parse. Add types only as
// single-chip ops require them.

#include <cstdint>

// Fabric packet types — declared as opaque structs so kernels that
// reference them by type-name (PacketHeader, LowLatencyPacketHeader,
// HybridMeshPacketHeader) parse. No real implementation in emule.
struct PacketHeader {};
struct LowLatencyPacketHeader {};
struct HybridMeshPacketHeader {};

// Fabric send-type sentinels — silicon enum values; emule provides the
// trailing _LAST sentinel that switch statements / array sizes depend on.
constexpr uint32_t NOC_SEND_TYPE_LAST = 0;
constexpr uint32_t CHIP_SEND_TYPE_LAST = 0;
