// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// emule shadow of fabric_config.h. The real get_fabric_max_packet_size() reads the per-chip fabric
// connection table at MEM_TENSIX_FABRIC_CONNECTIONS_BASE in device L1 — which emule never populates
// (fabric firmware init is skipped). emule teleports each send in full (no real EDM buffer), so the
// "max packet size" only bounds the op's packetization; return a fixed, ample payload size.
#include <cstdint>

namespace tt::tt_fabric {
// 4 KB payload (2 bf16 tiles). Ample for emule (no buffer limit); keeps packetization simple.
inline uint32_t get_fabric_max_packet_size() { return 4096u; }
}  // namespace tt::tt_fabric
