// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Comprehensive fabric stub for emule. Multichip / fabric is out of scope —
// this header provides just enough types + function shapes so upstream ops
// that include silicon fabric headers parse cleanly. All operations are
// no-ops; calling fabric functions in emule has no effect.
//
// Each silicon fabric header is shadowed by an
// emule version that #includes this single file.

#include <cstdint>

// ---- Packet header types ----
using PACKET_HEADER_TYPE = struct PacketHeader;
struct PacketHeader {
    uint32_t pad[16] = {};
    uint32_t noc_send_type = 0;

    // Header constructor methods — silicon: configures the packet for a
    // specific NOC op. emule: no-op (returns *this for chaining).
    template <typename... Args>
    PacketHeader& to_noc_fused_unicast_write_atomic_inc(Args&&...) { return *this; }
    template <typename... Args>
    PacketHeader& to_noc_unicast_atomic_inc(Args&&...) { return *this; }
    template <typename... Args>
    PacketHeader& to_noc_unicast_write(Args&&...) { return *this; }
    template <typename... Args>
    PacketHeader& to_noc_unicast_inline_write(Args&&...) { return *this; }
    template <typename... Args>
    PacketHeader& to_chip_unicast(Args&&...) { return *this; }
    template <typename... Args>
    PacketHeader& to_chip_multicast(Args&&...) { return *this; }
};
struct LowLatencyPacketHeader : PacketHeader {};
struct HybridMeshPacketHeader : PacketHeader {};

// Make PacketHeader's "members" referenced by silicon parse — opaque getters
// are accessed by name in some debug paths; provide as no-ops returning 0.
namespace tt::tt_fabric {

// Send-type sentinels (silicon enum-class values; emule keeps them as
// constexpr ints since some references take them as integer values).
constexpr uint32_t NOC_SEND_TYPE_LAST = 0;
constexpr uint32_t CHIP_SEND_TYPE_LAST = 0;

// Command-header structs (constructor-only — emule never reads these).
struct NocUnicastCommandHeader {
    template <typename... Args>
    NocUnicastCommandHeader(Args&&...) {}
};
struct NocUnicastAtomicIncCommandHeader {
    template <typename... Args>
    NocUnicastAtomicIncCommandHeader(Args&&...) {}
};
struct NocUnicastInlineWriteCommandHeader {
    template <typename... Args>
    NocUnicastInlineWriteCommandHeader(Args&&...) {}
};
struct NocUnicastAtomicIncFusedCommandHeader {
    template <typename... Args>
    NocUnicastAtomicIncFusedCommandHeader(Args&&...) {}
};

// Connection manager — no-op stub.
class RoutingPlaneConnectionManager {
public:
    // `.sender` member — upstream code accesses fabric_connection.get(slot).sender
    // to get the per-slot sender handle. Self-reference so chained member
    // access still no-ops.
    RoutingPlaneConnectionManager& sender = *this;

    template <typename... Args>
    RoutingPlaneConnectionManager(Args&&...) {}
    template <typename... Args>
    void open(Args&&...) {}
    template <typename... Args>
    void close(Args&&...) {}
    template <typename... Args>
    void wait_for_empty_write_slot(Args&&...) {}
    template <typename Header>
    void send_payload_with_header(Header&&, const uint8_t* /*src*/, uint32_t /*size*/) {}
    template <typename Header>
    void send_header(Header&&) {}
    template <typename... Args>
    void send_payload_flush_blocking_from_address(Args&&...) {}
    template <typename... Args>
    void send_payload_non_blocking_from_address(Args&&...) {}
    template <typename... Args>
    void send_payload_blocking_from_address(Args&&...) {}
    template <typename... Args>
    void send_payload_without_header_non_blocking_from_address(Args&&...) {}
    template <typename... Args>
    void send_payload_without_header_blocking_from_address(Args&&...) {}
    // `get` member — upstream code accesses per-route connection handles via .get(i).
    template <typename... Args>
    RoutingPlaneConnectionManager& get(Args&&...) { return *this; }
};

// Free-function fabric API used by upstream ops.
template <typename... Args> inline void open_connections(Args&&...) {}
template <typename... Args> inline void close_connections(Args&&...) {}
template <typename... Args> inline void fabric_set_unicast_route(Args&&...) {}
template <typename... Args> inline void fabric_set_mcast_route(Args&&...) {}

// Route flag types referenced by upstream kernels.
struct UnicastFusedAtomicIncUpdateMask {
    template <typename... Args>
    UnicastFusedAtomicIncUpdateMask(Args&&...) {}
};

// Alias FabricConnectionManager → same type (some kernels use either name).
using FabricConnectionManager = RoutingPlaneConnectionManager;

// Experimental sub-namespaces — empty so `using namespace ...experimental;`
// statements parse.
namespace linear::experimental {}
namespace common::experimental {}
namespace mesh::experimental {}

}  // namespace tt::tt_fabric

// ---- PacketHeaderPool ----
// Silicon: a static pool of PacketHeader entries with reset() + allocate
// functions. emule: returns a static dummy header.
struct HeaderTableEntry_t {
    PacketHeader* first = nullptr;
    PacketHeader* second = nullptr;
};
struct PacketHeaderPool {
    static inline PacketHeader _dummy[4]{};
    static inline HeaderTableEntry_t header_table[16]{};

    static void reset() {}
    static PacketHeader* allocate_header(uint32_t /*count*/ = 1) { return &_dummy[0]; }
    static uint32_t allocate_header_n(uint32_t /*count*/) { return 0; }
};

// ---- sdpa_fabric helpers (used by sdpa op) ----
namespace sdpa_fabric {
template <typename... Args> inline void send_to_neighbor(Args&&...) {}
template <typename... Args> inline void send_to_neighbor_no_sem(Args&&...) {}
template <typename... Args> inline void send_sem_inc_to_neighbor(Args&&...) {}
}  // namespace sdpa_fabric
