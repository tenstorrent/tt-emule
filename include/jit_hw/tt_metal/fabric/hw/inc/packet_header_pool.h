// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
// emule shadow of tt_metal/fabric/hw/inc/packet_header_pool.h.
//
// The silicon pool hands out L1 addresses from the reserved MEM_PACKET_HEADER_POOL_BASE region. emule
// must return a HOST pointer into the CURRENT worker's real L1 (via __emule_local_l1_to_ptr), because
// the fabric shim narrows the header to a bridge_l1-relative offset (__emule_fabric_l1_off) and the
// teleport re-widens it against bridge_l1 — a pointer outside the worker's L1 region would not
// round-trip. We therefore map the same reserved L1 region through __emule_local_l1_to_ptr.
//
// Routes: silicon allocates contiguous blocks of headers and registers each block under a route_id
// (header_table[route_id] = {first_header, num}); the route-variant fabric send API iterates a route's
// headers via for_each_header. emule mirrors this with per-(core,risc)-thread state. Partitioning is by
// RISC (__emule_self->processor_id) — mirroring silicon's proc_type partition — so a core's DM0/DM1 riscs never
// alias a header; cores are already separate L1 (__emule_local_l1_to_ptr resolves to the running
// thread's core), so every (core, risc) thread's region is disjoint (race-free, no global counter).
#include <cstdint>
#include "jit_hw/__emule_fabric_stubs.h"
#include "dev_mem_map.h"
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self->processor_id (per-RISC pool key)

// Both defined in the runner / dataflow_api.h; declared here so the pool can map a reserved L1 offset to
// its low-2GB host alias and partition by the running RISC.
uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr);

class PacketHeaderPool {
    static constexpr uint32_t HEADER_STRIDE = sizeof(PACKET_HEADER_TYPE);  // 48B, already 16B-aligned
    static constexpr uint32_t POOL_SLOTS = MEM_PACKET_HEADER_POOL_SIZE / HEADER_STRIDE;
    static constexpr uint32_t SLOTS_PER_RISC = 16;                      // headers per RISC partition
    static constexpr uint32_t MAX_RISCS = POOL_SLOTS / SLOTS_PER_RISC;  // 48/16=3: BRISC/NCRISC/TRISC
    static constexpr uint8_t MAX_ROUTES = ThreadCommonCtx::PHDR_MAX_ROUTES;

    // Allocation state (cursor + route table) is homed in the running fiber's ThreadCommonCtx, NOT a
    // thread_local: a persistent emule worker hosts many fibers across many program launches, and silicon
    // gets a fresh pool per launch (kernel .bss re-zero). The ctx is a fresh, value-initialized object per
    // launch (see launch_cores), so keying off it resets the pool per program — without this the cursor +
    // route_id table leak across ops and overflow, corrupting fabric multicast routes (tt-emule #221).
    static uint32_t risc_partition_base() {
        return (static_cast<uint32_t>(__emule_self->processor_id) % MAX_RISCS) * SLOTS_PER_RISC;
    }
    static PACKET_HEADER_TYPE* header_at(uint32_t index_in_partition) {
        const uint32_t slot = risc_partition_base() + (index_in_partition % SLOTS_PER_RISC);
        return reinterpret_cast<PACKET_HEADER_TYPE*>(
            __emule_local_l1_to_ptr(MEM_PACKET_HEADER_POOL_BASE + slot * HEADER_STRIDE));
    }

public:
    // Reset this fiber's allocation cursor + route table (mirrors silicon reset() for loop reuse).
    static void reset() {
        __emule_self->phdr_cursor = 0;
        __emule_self->phdr_route_count = 0;
    }

    // Allocate `num` contiguous headers; register them as a new route; return the FIRST header pointer.
    static PACKET_HEADER_TYPE* allocate_header(uint32_t num = 1) {
        const uint32_t first = __emule_self->phdr_cursor;
        __emule_self->phdr_cursor += num;
        if (__emule_self->phdr_route_count < MAX_ROUTES) {
            __emule_self->phdr_route_first[__emule_self->phdr_route_count] = first;
            __emule_self->phdr_route_num[__emule_self->phdr_route_count] = static_cast<uint8_t>(num);
            __emule_self->phdr_route_count++;
        }
        return header_at(first);
    }

    // Allocate `num` contiguous headers and return the route_id (mirrors silicon: route over `num` conns).
    static uint8_t allocate_header_n(uint32_t num) {
        allocate_header(num);
        return static_cast<uint8_t>(__emule_self->phdr_route_count - 1);
    }

    template <typename Func>
    static void for_each_header(uint8_t route_id, Func&& func) {
        const uint32_t first = __emule_self->phdr_route_first[route_id];
        const uint8_t num = __emule_self->phdr_route_num[route_id];
        for (uint8_t i = 0; i < num; i++) {
            func(header_at(first + i), i);
        }
    }

    static uint8_t get_num_headers(uint8_t route_id) { return __emule_self->phdr_route_num[route_id]; }
};

// Route-variant fabric send API (silicon: linear/api.h connection_manager+route_id overloads). They
// iterate a route's headers and apply the per-header set/with-state form (defined in the stub). emule's
// chip-multicast routing metadata is inert (the teleport reaches the neighbor regardless), so set_state
// ignores the connection; with_state sends each header through the matching connection slot's
// teleporting sender. Defined here (not in the stub) because they reference PacketHeaderPool.
namespace tt::tt_fabric::linear::experimental {

template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_write_set_state(
    ConnMgr& /*conn*/, uint8_t route_id, uint8_t* start, uint8_t* range, Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_unicast_write_set_state<Mask>(hdr, start[i], range[i], cmd, size);
    });
}
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_write_with_state(
    ConnMgr& conn, uint8_t route_id, uint32_t src_addr, Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_unicast_write_with_state<Mask>(&conn.get(i).sender, hdr, src_addr, cmd, size);
    });
}

template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_scatter_write_set_state(
    ConnMgr& /*conn*/, uint8_t route_id, uint8_t* start, uint8_t* range, Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_scatter_write_set_state<Mask>(hdr, start[i], range[i], cmd, size);
    });
}
template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_scatter_write_with_state(
    ConnMgr& conn, uint8_t route_id, uint32_t src_addr, Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_scatter_write_with_state<Mask>(&conn.get(i).sender, hdr, src_addr, cmd, size);
    });
}

template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_atomic_inc_set_state(
    ConnMgr& /*conn*/, uint8_t route_id, uint8_t* start, uint8_t* range, Cmd cmd = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_unicast_atomic_inc_set_state<Mask>(hdr, start[i], range[i], cmd);
    });
}
template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename ConnMgr, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_atomic_inc_with_state(ConnMgr& conn, uint8_t route_id, Cmd cmd = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_multicast_noc_unicast_atomic_inc_with_state<Mask>(&conn.get(i).sender, hdr, cmd);
    });
}

// Unicast route variants (silicon: linear/api.h connection_manager+route_id overloads). Same shape as the
// multicast forms above — iterate the route's headers and apply the per-header set/with-state form from the
// stub — but for a one-target (real unicast) route: the codegen writers take this path when a route has a
// single destination (broadcast's is_point_to_point, relay/ring's directional sends).
// set_state stamps each header's 1D unicast route via __emule_stamp_unicast_route: silicon records the hop
// count through packet_header->to_chip_unicast, which is INERT in emule, so the route table must be stamped
// explicitly (the read-side mirror of silicon's fabric_set_unicast_route call in these overloads, and the
// direct analogue of the MCAST_1D self-stamp in the multicast forms). Without it, __emule_fabric_resolve_targets
// (EMULE_FABRIC8) cannot resolve the destination chip and the barrier atomic-inc misroutes, hanging the fiber
// engine (quiescent deadlock). with_state re-uses that stamped route and sends each header through the matching
// connection slot's teleporting sender. The first argument is the concrete RoutingPlaneConnectionManager
// (matching silicon, and what every generated writer declares) so it never collides with the
// packet-header-first low-level overloads these delegate to.
inline void __emule_stamp_unicast_route(tt::tt_fabric::PacketHeader* hdr, uint8_t num_hops) {
#ifndef EMULE_FABRIC_2D
    tt::tt_fabric::__emule_set_unicast_route_1d(hdr, num_hops);
#else
    (void)hdr;
    (void)num_hops;  // 2D routes are stamped by the kernel's explicit fabric_set_unicast_route(hdr,chip,mesh) calls
#endif
}
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& /*conn*/, uint8_t route_id, uint8_t* num_hops,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        __emule_stamp_unicast_route(hdr, num_hops[i]);
        fabric_unicast_noc_unicast_write_set_state<Mask>(hdr, num_hops[i], cmd, size);
    });
}
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& conn, uint8_t route_id, uint32_t src_addr,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_unicast_noc_unicast_write_with_state<Mask>(&conn.get(i).sender, hdr, src_addr, cmd, size);
    });
}

template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_scatter_write_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& /*conn*/, uint8_t route_id, uint8_t* num_hops,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        __emule_stamp_unicast_route(hdr, num_hops[i]);
        fabric_unicast_noc_scatter_write_set_state<Mask>(hdr, num_hops[i], cmd, size);
    });
}
template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_scatter_write_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& conn, uint8_t route_id, uint32_t src_addr,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_unicast_noc_scatter_write_with_state<Mask>(&conn.get(i).sender, hdr, src_addr, cmd, size);
    });
}

template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& /*conn*/, uint8_t route_id, uint8_t* num_hops, Cmd cmd = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        __emule_stamp_unicast_route(hdr, num_hops[i]);
        fabric_unicast_noc_unicast_atomic_inc_set_state<Mask>(hdr, num_hops[i], cmd);
    });
}
template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& conn, uint8_t route_id, Cmd cmd = nullptr) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<Mask>(&conn.get(i).sender, hdr, cmd);
    });
}

template <UnicastFusedAtomicIncUpdateMask Mask = UnicastFusedAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& /*conn*/, uint8_t route_id, uint8_t* num_hops,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        __emule_stamp_unicast_route(hdr, num_hops[i]);
        fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<Mask>(hdr, num_hops[i], cmd, size);
    });
}
template <UnicastFusedAtomicIncUpdateMask Mask = UnicastFusedAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    tt::tt_fabric::RoutingPlaneConnectionManager& conn, uint8_t route_id, uint32_t src_addr,
    Cmd cmd = nullptr, uint16_t size = 0) {
    PacketHeaderPool::for_each_header(route_id, [&](tt::tt_fabric::PacketHeader* hdr, uint8_t i) {
        fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<Mask>(&conn.get(i).sender, hdr, src_addr, cmd, size);
    });
}

}  // namespace tt::tt_fabric::linear::experimental
