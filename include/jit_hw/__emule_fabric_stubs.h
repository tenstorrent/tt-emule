// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// emule fabric client-API shim (multi-chip CCL).
//
// emule does NOT run the ethernet/ERISC fabric router or model multi-hop. It intercepts the fabric
// *client API* (WorkerToFabricEdmSender + its connection managers) here, and routes each worker send
// to a runtime "teleport" hook (__emule_fabric_teleport, implemented in emulated_program_runner.cpp)
// that decodes the packet header, resolves the destination chip, and applies the terminal NOC command
// directly into that chip's L1. Delivery is synchronous.
//
// The packet header below uses the REAL silicon layout (command union @0, payload_size @40,
// noc_send_type @42) so the teleport can decode it. Every silicon fabric header is shadowed by an
// emule version that #includes this single file (including both the `fabric/` and `tt_metal/fabric/`
// packet-header include paths, so the real header is never pulled in and there is one definition).
//
// Layout coupling note: this duplicates the real fabric_edm_packet_header.hpp field offsets rather
// than including it (the real header drags in HW intrinsics + a host-path TT_THROW that don't fit the
// JIT). Keep these offsets in sync with tt_metal/fabric/fabric_edm_packet_header.hpp. (Tracked as a
// deliberate tradeoff in docs/fabric-ccl-emulation.md — revisit using the real header once its deps are emule-safe.)

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "jit_hw/tt-metalium/experimental/fabric/fabric_edm_types.hpp"  // tt::tt_fabric::Topology (single def)
#include "jit_hw/internal/emule_thread_ctx.h"  // ThreadCommonCtx + __emule_self (per-fiber conn open-seq counter)
#include "jit_hw/internal/emule_l1_to_ptr.h"  // __emule_local_l1_to_ptr + __emule_self (offset<->host ptr)

// L1 offset model: fabric L1 addresses cross the shim as 0-based offsets. Narrow a header/payload
// host pointer to its offset (widen with __emule_local_l1_to_ptr). Not a truncation — survives worker
// L1 mapped above 4 GB, which the old `(uint32_t)ptr` truncation does not (the >4 GB galaxy needs this).
inline uint32_t __emule_fabric_l1_off(const volatile void* p) {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p) -
                                 reinterpret_cast<uintptr_t>(__emule_self->bridge_l1));
}

// Runtime teleport hooks (resolved at JIT link time to the runner symbols).
extern "C" void __emule_fabric_teleport(const void* packet_header, const void* payload, uint32_t payload_size);
// Record the route metadata for a packet header (keyed by its L1-alias address); the teleport resolves it
// to the final destination chip(s). kind/fields layout matches the runner's EmuleRoute — KEEP IN SYNC.
extern "C" void __emule_fabric_set_route(
    uint32_t hdr, uint32_t kind, uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e, uint32_t f);
// Record which of the worker's fabric connections (fwd=0/bwd=1) a 1D send used, so the teleport can resolve
// the dst chip by direction. Set at send time by the sender (which knows its own direction).
extern "C" void __emule_fabric_set_route_dir(
    uint32_t hdr, uint32_t conn_index, uint32_t mux_x, uint32_t mux_y);
// Carry a stamped route from a source header address to a destination address when the header bytes are
// copied (a worker staging a packet header into a forwarder relay slot). On silicon the routing fields ride
// inside the header, so a byte copy carries them for free; emule keeps them in the address-keyed side-table,
// so the copy must replicate the entry. No-op when src carries no route. See docs/fabric-ccl-emulation.md.
extern "C" void __emule_fabric_route_follow(uint32_t src_key, uint32_t dst_key);

// ProgrammableCoreType — a single GLOBAL definition guarded by the same macro noc_semaphore.h uses, so
// there is exactly one definition regardless of include order. CCL kernels (moe_utils.hpp) reference it
// unqualified at global scope; noc_semaphore.h's templated Semaphore<ProgrammableCoreType> must see the
// same type (a separate namespaced copy would make Semaphore<ProgrammableCoreType::TENSIX> ambiguous).
#ifndef PROGRAMMABLE_CORE_TYPE_DEFINED
#define PROGRAMMABLE_CORE_TYPE_DEFINED
enum class ProgrammableCoreType : uint8_t { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2 };
#endif

namespace tt::tt_fabric {

// --- NOC send types (silicon enum values; teleport switches on these) ---
enum NocSendType : uint8_t {
    NOC_UNICAST_WRITE = 0,
    NOC_UNICAST_INLINE_WRITE = 1,
    NOC_UNICAST_ATOMIC_INC = 2,
    NOC_FUSED_UNICAST_ATOMIC_INC = 3,
    NOC_UNICAST_SCATTER_WRITE = 4,
    NOC_MULTICAST_WRITE = 5,
    NOC_MULTICAST_ATOMIC_INC = 6,
    NOC_UNICAST_READ = 7,
    NOC_SEND_TYPE_LAST = NOC_UNICAST_SCATTER_WRITE,
};
constexpr uint32_t CHIP_SEND_TYPE_LAST = 0;

// NOC index the fabric uses for local-chip delivery (silicon: edm_fabric_utils.hpp). Shadowed here.
static constexpr uint8_t edm_to_local_chip_noc = 1;

// (x,y) NOC coordinate pair (silicon: tt::tt_fabric::WorkerXY in noc_addr.h, which emule shadows).
struct WorkerXY {
    uint32_t x = 0;
    uint32_t y = 0;
    WorkerXY() = default;
    WorkerXY(uint32_t x_, uint32_t y_) : x(x_), y(y_) {}
};

// NOTE: ProgrammableCoreType is already defined by emule elsewhere (core_config.h / noc_semaphore.h);
// do NOT redefine it here or `using` it into global scope — that creates an ambiguity for the
// templated Semaphore<ProgrammableCoreType> path. CCL kernels resolve it to the existing global enum.

// tt::tt_fabric::Topology comes from fabric_edm_types.hpp (included above) — single definition.

// Endpoint teardown (mux) — emule has no real endpoint, so no-op.
template <typename... A>
inline void fabric_endpoint_terminate(A&&...) {}

// Multicast routing command header (silicon: worker_routing_utils.hpp) — opaque to emule.
struct MulticastRoutingCommandHeader {
    uint32_t start_distance_in_hops = 0;
    uint32_t range_hops = 0;
    template <typename... A>
    MulticastRoutingCommandHeader(A&&...) {}
    MulticastRoutingCommandHeader() = default;
};

// Fabric mux endpoint connection helpers (silicon: tt_fabric mux API). emule needs no real
// connection (teleport), so these are no-ops returning a teleport-backed sender.
template <uint8_t NUM_BUFFERS>
struct WorkerToFabricMuxSender;  // defined below; forward-declared for the helpers here
template <uint8_t NUM_BUFFERS = 0, typename... A>
inline WorkerToFabricMuxSender<NUM_BUFFERS> build_connection_to_fabric_endpoint(
    uint8_t fabric_mux_x, uint8_t fabric_mux_y, A&&...) {
    WorkerToFabricMuxSender<NUM_BUFFERS> s{};
    // Carry the mux's NOC coords so the teleport can recover the mux's recorded line direction (g_mux_dir).
    s.emule_mux_x = fabric_mux_x;
    s.emule_mux_y = fabric_mux_y;
    return s;
}
template <typename... A>
inline void wait_for_fabric_endpoint_ready(A&&...) {}
template <typename... A>
inline void fabric_client_connect(A&&...) {}
// Split connect handshake (silicon: open the connection in two phases so the worker can overlap other
// setup between start and finish). emule has no real EDM/mux connection to open, so both are no-ops.
template <typename... A>
inline void fabric_client_connect_start(A&&...) {}
template <typename... A>
inline void fabric_client_connect_finish(A&&...) {}
template <typename... A>
inline void fabric_client_disconnect(A&&...) {}

// --- Command headers (the union members the teleport reads; constructors match silicon) ---
struct NocUnicastCommandHeader {
    uint64_t noc_address;
};
struct NocUnicastInlineWriteCommandHeader {
    uint64_t noc_address;
    uint32_t value;
};
struct NocUnicastAtomicIncCommandHeader {
    uint64_t noc_address;
    uint32_t val;
    bool flush;
    NocUnicastAtomicIncCommandHeader(uint64_t addr, uint32_t v, bool f = true) :
        noc_address(addr), val(v), flush(f) {}
};
struct NocUnicastAtomicIncFusedCommandHeader {
    uint64_t noc_address;
    uint64_t semaphore_noc_address;
    uint32_t val;
    bool flush;
    NocUnicastAtomicIncFusedCommandHeader(uint64_t a, uint64_t s, uint32_t v, bool f = true) :
        noc_address(a), semaphore_noc_address(s), val(v), flush(f) {}
};
static constexpr uint8_t NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS = 2;
struct NocUnicastScatterAtomicIncFusedCommandHeader {
    uint64_t noc_address[NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS] = {};
    uint64_t semaphore_noc_address;
    uint16_t chunk_size[NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS - 1] = {};  // last chunk size is implicit
    uint16_t val;                                                             // semaphore increment value
    bool flush;
    NocUnicastScatterAtomicIncFusedCommandHeader(
        std::initializer_list<uint64_t> noc_addresses, uint64_t semaphore_noc_address,
        std::initializer_list<uint16_t> chunk_sizes, uint16_t val, bool flush = true) :
        semaphore_noc_address(semaphore_noc_address), val(val), flush(flush) {
        uint32_t i = 0;
        for (uint64_t a : noc_addresses) {
            if (i < NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS) {
                noc_address[i++] = a;
            }
        }
        uint32_t j = 0;
        for (uint16_t s : chunk_sizes) {
            if (j < NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS - 1) {
                chunk_size[j++] = s;
            }
        }
    }
};
// Per-chunk encoding for a scatter-write packet (silicon: fabric_edm_packet_header.hpp
// NocScatterWriteChunkEncoding — 2 bits per chunk). The teleport reads it to tell a payload write from a
// semaphore increment on a fused scatter-write + atomic-inc.
enum NocScatterWriteChunkEncoding : uint8_t {
    CHUNK_ENCODING_NOP = 0,
    CHUNK_ENCODING_UNICAST_WRITE = 1,
    CHUNK_ENCODING_SEMINC_NO_FLUSH = 2,
    CHUNK_ENCODING_SEMINC_FLUSH = 3,
};
static constexpr uint8_t NOC_SCATTER_WRITE_MAX_CHUNKS = 4;
struct NocUnicastScatterCommandHeader {
    uint64_t noc_address[NOC_SCATTER_WRITE_MAX_CHUNKS] = {};
    uint16_t chunk_size[NOC_SCATTER_WRITE_MAX_CHUNKS - 1] = {};
    uint8_t chunk_count = 0;
    uint8_t chunk_encoding = 0;
    NocUnicastScatterCommandHeader(
        std::initializer_list<uint64_t> addresses, std::initializer_list<uint16_t> chunk_sizes = {}) {
        uint32_t i = 0;
        for (uint64_t a : addresses) {
            if (i < NOC_SCATTER_WRITE_MAX_CHUNKS) {
                noc_address[i++] = a;
            }
        }
        chunk_count = static_cast<uint8_t>(addresses.size());
        uint32_t j = 0;
        for (uint16_t s : chunk_sizes) {
            if (j < NOC_SCATTER_WRITE_MAX_CHUNKS - 1) {
                chunk_size[j++] = s;
            }
        }
    }
    // Array form (silicon: NocUnicastScatterCommandHeader(addrs[], sizes[], count)). `count` chunks =>
    // `count` destination addresses and `count-1` explicit chunk sizes (the last is the remainder).
    NocUnicastScatterCommandHeader(const uint64_t* addrs, const uint16_t* sizes, uint32_t count) {
        chunk_count = static_cast<uint8_t>(count);
        for (uint32_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; ++i) {
            noc_address[i] = (i < count) ? addrs[i] : 0;
        }
        for (uint32_t i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; ++i) {
            chunk_size[i] = (i + 1 < count) ? sizes[i] : 0;
        }
    }
};

// --- Packet header (REAL decodable layout, 48B = silicon LowLatencyPacketHeader) ---
struct PacketHeader {
    uint64_t cmd_noc_address;     // @0   command_fields union: NocUnicast*.noc_address
    uint8_t cmd_rest[32];         // @8..40: val / sem_address (union tail)
    uint16_t payload_size_bytes;  // @40
    uint8_t noc_send_type;        // @42
    uint8_t src_ch_id;            // @43
    uint32_t routing_fields;      // @44  (1D distance/range — emule resolves dst via neighbor instead)

    PacketHeader& to_noc_unicast_write(const NocUnicastCommandHeader& h, size_t size) volatile {
        cmd_noc_address = h.noc_address;
        payload_size_bytes = static_cast<uint16_t>(size);
        noc_send_type = NOC_UNICAST_WRITE;
        return const_cast<PacketHeader&>(*this);
    }
    PacketHeader& to_noc_unicast_inline_write(const NocUnicastInlineWriteCommandHeader& h) volatile {
        cmd_noc_address = h.noc_address;
        *reinterpret_cast<volatile uint32_t*>(&cmd_rest[0]) = h.value;  // value @8
        payload_size_bytes = 0;
        noc_send_type = NOC_UNICAST_INLINE_WRITE;
        return const_cast<PacketHeader&>(*this);
    }
    PacketHeader& to_noc_unicast_atomic_inc(const NocUnicastAtomicIncCommandHeader& h) volatile {
        cmd_noc_address = h.noc_address;
        *reinterpret_cast<volatile uint32_t*>(&cmd_rest[0]) = h.val;  // val @8
        noc_send_type = NOC_UNICAST_ATOMIC_INC;
        return const_cast<PacketHeader&>(*this);
    }
    PacketHeader& to_noc_fused_unicast_write_atomic_inc(
        const NocUnicastAtomicIncFusedCommandHeader& h, size_t size) volatile {
        cmd_noc_address = h.noc_address;
        *reinterpret_cast<volatile uint64_t*>(&cmd_rest[0]) = h.semaphore_noc_address;  // sem @8
        *reinterpret_cast<volatile uint32_t*>(&cmd_rest[8]) = h.val;                    // val @16
        payload_size_bytes = static_cast<uint16_t>(size);
        noc_send_type = NOC_FUSED_UNICAST_ATOMIC_INC;
        return const_cast<PacketHeader&>(*this);
    }
    PacketHeader& to_noc_unicast_scatter_write(const NocUnicastScatterCommandHeader& h, size_t size) volatile {
        // Lay the scatter command into the 40B command union: noc_address[4]@0..32, chunk_size[3]@32,
        // chunk_count@38, chunk_encoding@39 (matches the real NocUnicastScatterCommandHeader layout).
        volatile uint64_t* na = reinterpret_cast<volatile uint64_t*>(this);  // @0
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; ++i) {
            na[i] = h.noc_address[i];
        }
        for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; ++i) {
            *reinterpret_cast<volatile uint16_t*>(&cmd_rest[24 + i * 2]) = h.chunk_size[i];  // @32
        }
        cmd_rest[30] = h.chunk_count;     // @38
        cmd_rest[31] = h.chunk_encoding;  // @39
        payload_size_bytes = static_cast<uint16_t>(size);
        noc_send_type = NOC_UNICAST_SCATTER_WRITE;
        return const_cast<PacketHeader&>(*this);
    }
    // Mask-aware scatter field patch (mirrors api_common.h populate_unicast_scatter_write_fields<UpdateMask>):
    // patch ONLY the fields named in MaskBits, preserve the rest. This is what makes the stateful API work —
    // a with_state<DstAddrs> after a set_state<ChunkSizes|PayloadSize> must keep the size/chunks set_state laid
    // down (the kernel reuses one header across both calls). MaskBits = UnicastScatterWriteUpdateMask:
    // DstAddrs=1, ChunkSizes=2, PayloadSize=4.
    template <uint32_t MaskBits>
    void apply_scatter_fields(uint16_t payload_size, const NocUnicastScatterCommandHeader& h) volatile {
        constexpr bool upd_addr = (MaskBits & 0x1u) != 0;
        constexpr bool upd_chunks = (MaskBits & 0x2u) != 0;
        constexpr bool upd_psize = (MaskBits & 0x4u) != 0;
        if constexpr (upd_addr || upd_chunks) {
            cmd_rest[30] = h.chunk_count;  // @38
        }
        if constexpr (upd_addr) {
            volatile uint64_t* na = reinterpret_cast<volatile uint64_t*>(this);  // @0
            for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS; ++i) {
                na[i] = (i < h.chunk_count) ? h.noc_address[i] : 0;
            }
        }
        if constexpr (upd_psize) {
            payload_size_bytes = payload_size;
        }
        if constexpr (upd_chunks) {
            for (int i = 0; i < NOC_SCATTER_WRITE_MAX_CHUNKS - 1; ++i) {
                *reinterpret_cast<volatile uint16_t*>(&cmd_rest[24 + i * 2]) = h.chunk_size[i];  // @32
            }
        }
    }
    // Mask-aware fused unicast-write+atomic-inc patch (mirrors api_common.h populate_unicast_fused_atomic_inc
    // _fields<UpdateMask>): layout is noc_address@0, semaphore_noc_address@8, val@16 — the same slots the
    // NOC_FUSED_UNICAST_ATOMIC_INC teleport (runner case 3) reads back. Bits = UnicastFusedAtomicIncUpdateMask.
    template <uint32_t MaskBits>
    void apply_fused_fields(uint16_t payload_size, const NocUnicastAtomicIncFusedCommandHeader& h) volatile {
        constexpr bool upd_waddr = (MaskBits & 0x1u) != 0;   // WriteDstAddr
        constexpr bool upd_saddr = (MaskBits & 0x2u) != 0;   // SemaphoreAddr
        constexpr bool upd_val = (MaskBits & 0x4u) != 0;     // Val
        constexpr bool upd_psize = (MaskBits & 0x10u) != 0;  // PayloadSize (Flush=0x8 is inert — teleport is synchronous)
        if constexpr (upd_waddr) {
            cmd_noc_address = h.noc_address;  // @0
        }
        if constexpr (upd_saddr) {
            *reinterpret_cast<volatile uint64_t*>(&cmd_rest[0]) = h.semaphore_noc_address;  // @8
        }
        if constexpr (upd_val) {
            *reinterpret_cast<volatile uint32_t*>(&cmd_rest[8]) = h.val;  // @16
        }
        if constexpr (upd_psize) {
            payload_size_bytes = payload_size;
        }
    }
    // Mask-aware fused scatter-write + atomic-inc patch (mirrors api_common.h
    // populate_unicast_fused_scatter_write_atomic_inc_fields<UpdateMask>). Silicon encodes this op as a
    // NOC_UNICAST_SCATTER_WRITE packet — N unicast-write chunks + 1 semaphore-increment chunk, distinguished
    // by the per-chunk chunk_encoding, NOT a dedicated send-type. So the layout is the scatter-write command's:
    // noc_address[4]@0 (write dsts, then the seminc dst), chunk_size[3]@32 (write sizes, then the seminc val),
    // chunk_count@38, chunk_encoding@39 — exactly what the NOC_UNICAST_SCATTER_WRITE teleport (runner case 4)
    // reads back. Bits = UnicastFusedScatterWriteAtomicIncUpdateMask.
    template <uint32_t MaskBits>
    void apply_fused_scatter_fields(uint16_t payload_size, const NocUnicastScatterAtomicIncFusedCommandHeader& h) volatile {
        constexpr bool upd_waddr = (MaskBits & 0x1u) != 0;   // WriteDstAddrs
        constexpr bool upd_saddr = (MaskBits & 0x2u) != 0;   // SemaphoreDstAddr
        constexpr bool upd_chunks = (MaskBits & 0x4u) != 0;  // WriteChunkSizes
        constexpr bool upd_val = (MaskBits & 0x8u) != 0;     // Val
        constexpr bool upd_flush = (MaskBits & 0x10u) != 0;  // Flush — sets the chunk encodings (init)
        constexpr bool upd_psize = (MaskBits & 0x20u) != 0;  // PayloadSize
        constexpr uint8_t write_chunks = NOC_SCATTER_WRITE_ATOMIC_INC_FUSED_WRITE_CHUNKS;  // 2
        volatile uint64_t* na = reinterpret_cast<volatile uint64_t*>(this);               // noc_address[]@0
        cmd_rest[30] = static_cast<uint8_t>(write_chunks + 1);  // @38 chunk_count = writes + the seminc (constant)
        if constexpr (upd_waddr) {
            for (int i = 0; i < write_chunks; ++i) {
                na[i] = h.noc_address[i];  // write chunk dsts @0, @8
            }
        }
        if constexpr (upd_saddr) {
            na[write_chunks] = h.semaphore_noc_address;  // the seminc chunk's dst = the semaphore @16
        }
        if constexpr (upd_chunks) {
            uint16_t accumulated = 0;
            for (int i = 0; i < write_chunks - 1; ++i) {
                *reinterpret_cast<volatile uint16_t*>(&cmd_rest[24 + i * 2]) = h.chunk_size[i];  // @32
                accumulated = static_cast<uint16_t>(accumulated + h.chunk_size[i]);
            }
            // last write chunk size is implicit on silicon (remaining payload); make it explicit @34
            *reinterpret_cast<volatile uint16_t*>(&cmd_rest[24 + (write_chunks - 1) * 2]) =
                static_cast<uint16_t>(payload_size - accumulated);
        }
        if constexpr (upd_val) {
            // seminc value packed into the seminc chunk's size slot, after the write chunks @36
            *reinterpret_cast<volatile uint16_t*>(&cmd_rest[24 + write_chunks * 2]) = h.val;
        }
        if constexpr (upd_flush) {
            uint8_t enc = 0;  // @39: 2 bits/chunk — writes = UNICAST_WRITE, last = SEMINC_(NO_)FLUSH
            for (int i = 0; i < write_chunks; ++i) {
                enc |= static_cast<uint8_t>(CHUNK_ENCODING_UNICAST_WRITE << (i * 2));
            }
            enc |= static_cast<uint8_t>((h.flush ? CHUNK_ENCODING_SEMINC_FLUSH : CHUNK_ENCODING_SEMINC_NO_FLUSH)
                                        << (write_chunks * 2));
            cmd_rest[31] = enc;
        }
        if constexpr (upd_psize) {
            payload_size_bytes = payload_size;
        }
    }
    // Routing setters — emule resolves the destination chip via the cluster neighbor table, not the
    // routing fields, so these are accepted but otherwise inert.
    template <typename... A>
    PacketHeader& to_chip_unicast(A&&...) volatile {
        return const_cast<PacketHeader&>(*this);
    }
    template <typename... A>
    PacketHeader& to_chip_multicast(A&&...) volatile {
        return const_cast<PacketHeader&>(*this);
    }
};
static_assert(sizeof(PacketHeader) == 48, "emule PacketHeader must match silicon LowLatencyPacketHeader size");

// One 48B packet-header layout. The 2D (HybridMesh/UDMHybridMesh) variants are distinct empty-derived types,
// not aliases, so a kernel's `if constexpr (is_same_v<PACKET_HEADER_TYPE, …>)` route dispatch resolves to the
// branch matching the fabric config, as silicon does. Empty non-virtual inheritance keeps the layout and
// makes base⇄derived pointer casts no-ops. See docs/fabric-ccl-emulation.md.
using LowLatencyPacketHeader = PacketHeader;
struct HybridMeshPacketHeader : PacketHeader {};
struct UDMHybridMeshPacketHeader : PacketHeader {};
static_assert(sizeof(HybridMeshPacketHeader) == 48 && sizeof(UDMHybridMeshPacketHeader) == 48,
              "emule 2D packet-header variants must keep the 48B silicon layout");

// --- Worker -> fabric sender. Staged payload + teleport on header flush. ---
struct WorkerToFabricEdmSender {
    const uint8_t* pending_payload = nullptr;
    uint32_t pending_size = 0;
    uint8_t emule_conn_index = 0;  // fwd=0/bwd=1 (set by FabricConnectionManager) — for 1D dst resolution
    // Mux-path direction hint (0xFFFF = unset): the mux's NOC coords, set on the fabric MUX path
    // (build_connection_to_fabric_endpoint) so the teleport can recover the mux's host-recorded direction.
    uint16_t emule_mux_x = 0xFFFF, emule_mux_y = 0xFFFF;

    // Accepts both build_from_args<ProgrammableCoreType::TENSIX>(idx) and build_from_args(args...) forms.
    // Tags the sender with the fiber's connection open-sequence index, which the teleport uses as the
    // direct-path 1D dst direction (the mux path / FabricConnectionManager overwrite emule_conn_index with
    // their own signal). See docs/fabric-ccl-emulation.md.
    template <auto CoreType = 0, typename... A>
    static WorkerToFabricEdmSender build_from_args(A&&...) {
        WorkerToFabricEdmSender s{};
        if (__emule_self != nullptr) {
            s.emule_conn_index = static_cast<uint8_t>(__emule_self->fabric_open_conn_seq++);
        }
        return s;
    }

    // Connection lifecycle — bookkeeping no-ops (no real EDM connection to open/close).
    template <typename... A> void open(A&&...) {}
    template <typename... A> void open_start(A&&...) {}
    template <typename... A> void open_finish(A&&...) {}
    template <typename... A> void close(A&&...) {}
    template <typename... A> void close_start(A&&...) {}
    template <typename... A> void close_finish(A&&...) {}

    // Flow control — always free (synchronous teleport has no buffer pressure).
    void wait_for_empty_write_slot() const {}
    uint32_t get_num_free_write_slots() const { return 8; }
    template <size_t N>
    bool edm_has_space_for_packet() const {
        return true;
    }

    // Stage the payload; the subsequent header flush performs the teleport.
    void send_payload_without_header_non_blocking_from_address(uint32_t src_addr, uint32_t size) {
        pending_payload = __emule_local_l1_to_ptr(src_addr);  // src_addr is a 0-based L1 offset
        pending_size = size;
    }
    // Header sends → teleport using any staged payload.
    void send_payload_flush_blocking_from_address(uint32_t hdr_addr, uint32_t /*size*/) { teleport_(hdr_addr); }
    void send_payload_flush_non_blocking_from_address(uint32_t hdr_addr, uint32_t /*size*/) { teleport_(hdr_addr); }
    void send_payload_non_blocking_from_address(uint32_t hdr_addr, uint32_t /*size*/) { teleport_(hdr_addr); }
    void send_payload_blocking_from_address(uint32_t hdr_addr, uint32_t /*size*/) { teleport_(hdr_addr); }
    void send_payload_blocking(uint32_t /*cb_id*/, uint32_t /*num_pages*/, uint32_t /*page_size*/) {}

    // Current-slot + stateful API surface (blaze). Covered minimally; the payload was staged above.
    void send_current_slot_non_blocking(uint32_t payload_addr, uint32_t payload_size, uint32_t header_addr) {
        send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
        teleport_(header_addr);
    }
    template <typename... A> void setup_stateful_send_cmd_bufs(A&&...) {}
    template <bool posted = false>
    void send_current_slot_stateful_non_blocking(
        uint32_t payload_addr, uint32_t payload_size, uint32_t header_addr, uint8_t /*noc*/ = 0) {
        send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
        teleport_(header_addr);
    }
    template <bool posted = false>
    void send_current_slot_stateful_non_blocking_from_address(
        uint32_t packet_addr, uint32_t /*packet_size*/, uint8_t /*noc*/ = 0) {
        teleport_(packet_addr);
    }

private:
    void teleport_(uint32_t header_addr) {
        // Record this connection's direction signals so the teleport can resolve a 1D dst by direction.
        __emule_fabric_set_route_dir(header_addr, emule_conn_index, emule_mux_x, emule_mux_y);
        __emule_fabric_teleport(
            __emule_local_l1_to_ptr(header_addr), pending_payload, pending_size);  // header_addr is a 0-based L1 offset
        pending_payload = nullptr;
        pending_size = 0;
    }
};

// --- RoutingPlaneConnectionManager (blaze N-slot; also reached via .get(i).sender) ---
class RoutingPlaneConnectionManager {
public:
    struct ConnectionSlot {
        WorkerToFabricEdmSender sender;
        uint8_t tag = 0;
        uint16_t dst_dev_id = 0;
        uint16_t dst_mesh_id = 0;
    };
    ConnectionSlot slots_[8] = {};
    WorkerToFabricEdmSender fwd_, bwd_;

    template <typename... A> RoutingPlaneConnectionManager(A&&...) {}
    template <typename... A> static RoutingPlaneConnectionManager build_from_args(A&&...) {
        return RoutingPlaneConnectionManager{};
    }
    template <typename... A> void open(A&&...) {}
    template <typename... A> void open_start(A&&...) {}
    template <typename... A> void open_finish(A&&...) {}
    template <typename... A> void close(A&&...) {}
    template <typename... A> void close_start(A&&...) {}
    template <typename... A> void close_finish(A&&...) {}
    ConnectionSlot& get(uint32_t i = 0) { return slots_[i & 7]; }
    uint8_t get_tag(uint32_t i = 0) { return slots_[i & 7].tag; }
    WorkerToFabricEdmSender& get_forward_connection() { return fwd_; }
    WorkerToFabricEdmSender& get_backward_connection() { return bwd_; }
    template <typename F> void for_each(F&&) {}
    template <typename F> void for_each_with_tag(uint32_t, F&&) {}
};

// --- Free-function fabric API ---
// emule resolves the destination CHIP in the teleport (host-side, via the control-plane route table). The
// kernel knows the dst only semantically — (2D) an explicit FabricNodeId, (1D) a hop distance, or a line
// MULTICAST extent — so the route setters RECORD it (keyed by the packet-header address) via
// __emule_fabric_set_route. KIND constants mirror the runner's emule_route_kind — KEEP IN SYNC.
namespace emule_route {
enum Kind : uint32_t { UNSET = 0, UNICAST_1D = 1, UNICAST_2D = 2, MCAST_1D = 3, MCAST_2D = 4 };
inline uint32_t key(volatile PacketHeader* hdr) {
    return __emule_fabric_l1_off(hdr);
}
}  // namespace emule_route

// Stamp the 1D unicast route (distance_in_hops) for a (packet_header, distance) pair.
template <typename HdrT, typename DistT>
inline void __emule_set_unicast_route_1d(HdrT&& hdr, DistT&& target_num) {
    __emule_fabric_set_route(
        emule_route::key(reinterpret_cast<volatile PacketHeader*>(hdr)),
        emule_route::UNICAST_1D, static_cast<uint32_t>(target_num), 0, 0, 0, 0, 0);
}
// Stamp the explicit-destination unicast route (packet_header, dst_chip, dst_mesh). Templated on the header
// pointer type: under a 2D fabric config the kernel's PACKET_HEADER_TYPE is HybridMeshPacketHeader (a type
// DERIVED from the 48B PacketHeader), so a fixed `volatile PacketHeader*` parameter would need a
// derived-to-base conversion and LOSE overload resolution to the exact-match variadic below — silently
// dropping the route (kind UNSET → the teleport falls back to the physical neighbor, the wrong chip on a
// submesh line). Deducing the header type keeps it an exact match. See docs/fabric-ccl-emulation.md.
template <typename HdrT, typename ChipT, typename MeshT>
inline void __emule_set_unicast_route_2d(HdrT&& hdr, ChipT&& dst_chip, MeshT&& dst_mesh) {
    const uint32_t k = emule_route::key(reinterpret_cast<volatile PacketHeader*>(hdr));
#ifdef EMULE_FABRIC_2D
    __emule_fabric_set_route(
        k, emule_route::UNICAST_2D, static_cast<uint32_t>(dst_chip), static_cast<uint32_t>(dst_mesh), 0, 0, 0, 0);
#else
    // 1D config's stray (chip, mesh) call: record the first arg as a hop distance, not a chip id.
    __emule_fabric_set_route(k, emule_route::UNICAST_1D, static_cast<uint32_t>(dst_chip), 0, 0, 0, 0, 0);
#endif
}
// Single variadic entry for fabric_set_unicast_route — dispatching on arity avoids the overload-resolution
// hazard that a fixed base-pointer overload has against a derived (2D) header type. worker_routing_utils'
// 2D branch passes (hdr, dst_chip_id, dst_mesh_id) [3 args]; the 1D branch / variadic reaches
// `fabric_set_unicast_route<false>(hdr, distance)` [2 args]. See docs/fabric-ccl-emulation.md.
template <bool RangeHopsEnabled = false, typename... A>
inline bool fabric_set_unicast_route(A&&... args) {
    if constexpr (sizeof...(A) == 2) {
        __emule_set_unicast_route_1d(std::forward<A>(args)...);
    } else if constexpr (sizeof...(A) == 3) {
        __emule_set_unicast_route_2d(std::forward<A>(args)...);
    }
    return true;
}

// Line multicast (worker_routing_utils 2D branch): (hdr, dst_chip, dst_mesh, e, w, n, s). Single variadic
// dispatching on arity for the same reason as fabric_set_unicast_route above — a fixed base-pointer overload
// would lose to the variadic no-op for a derived 2D header type and drop the route. The teleport walks the
// route table for each non-zero direction and replays the terminal op to every chip in the line.
template <typename HdrT>
inline void __emule_set_mcast_route_impl(
    HdrT&& hdr, uint16_t dst_chip, uint16_t dst_mesh, uint16_t e, uint16_t w, uint16_t n, uint16_t s) {
    const uint32_t k = emule_route::key(reinterpret_cast<volatile PacketHeader*>(hdr));
#ifdef EMULE_FABRIC_2D
    __emule_fabric_set_route(k, emule_route::MCAST_2D, dst_chip, dst_mesh, e, w, n, s);
#else
    // 1D line multicast: the route_info union maps the 2D args to {dst_chip=range_hops, dst_mesh=
    // start_distance_in_hops}; the per-direction hop counts {e,w,n,s} carry which way the line multicast
    // goes (exactly one is non-zero on a line) — the teleport reads that to pick the direction, so a middle
    // chip's backward barrier resolves backward even on the (untagged) mux path.
    __emule_fabric_set_route(k, emule_route::MCAST_1D, dst_mesh /*start*/, dst_chip /*range*/, e, w, n, s);
#endif
}
template <typename... A>
inline void fabric_set_mcast_route(A&&... args) {
    if constexpr (sizeof...(A) == 7) {
        __emule_set_mcast_route_impl(std::forward<A>(args)...);
    }
}
template <typename... A> inline bool fabric_set_sparse_multicast_route(A&&...) { return true; }
template <typename... A> inline void open_connections(A&&...) {}
template <typename... A> inline void close_connections(A&&...) {}

// Mux sender (tt_fabric_mux_interface.hpp) — same teleport-backed sender. Templated on the
// compile-time buffer count, as the real one is (kernels use WorkerToFabricMuxSender<N>).
template <uint8_t NUM_BUFFERS = 0>
struct WorkerToFabricMuxSender : WorkerToFabricEdmSender {};

// Mux façade: stage payload then teleport the header (matches fabric_async_write semantics).
template <typename Conn, typename Hdr>
inline void fabric_async_write(Conn& connection_handle, Hdr* packet_header, uint32_t src_addr, uint32_t size) {
    connection_handle.wait_for_empty_write_slot();
    connection_handle.send_payload_without_header_non_blocking_from_address(src_addr, size);
    connection_handle.send_payload_flush_blocking_from_address(
        __emule_fabric_l1_off(packet_header),
        sizeof(PacketHeader));
}

// Stateful-send update masks (silicon: api_common.h, which emule shadows). Used as combinable
// (OR'd) non-type template args to the *_with_state send API; emule's send shims are no-ops so the
// values only need to compile. Members match api_common.h.
enum class UnicastWriteUpdateMask : uint32_t { None = 0, DstAddr = 1u << 0, PayloadSize = 1u << 1, All = 3 };
enum class UnicastInlineWriteUpdateMask : uint32_t { None = 0, DstAddr = 1u << 0, Value = 1u << 1, All = 3 };
enum class UnicastAtomicIncUpdateMask : uint32_t {
    None = 0, DstAddr = 1u << 0, Val = 1u << 1, Flush = 1u << 2, All = 7
};
enum class UnicastScatterWriteUpdateMask : uint32_t {
    None = 0, DstAddrs = 1u << 0, ChunkSizes = 1u << 1, PayloadSize = 1u << 2, All = 7
};
enum class UnicastFusedAtomicIncUpdateMask : uint32_t {
    None = 0, WriteDstAddr = 1u << 0, SemaphoreAddr = 1u << 1, Val = 1u << 2, Flush = 1u << 3,
    PayloadSize = 1u << 4, All = 31
};
enum class UnicastFusedScatterWriteAtomicIncUpdateMask : uint32_t {
    None = 0, WriteDstAddrs = 1u << 0, SemaphoreDstAddr = 1u << 1, WriteChunkSizes = 1u << 2,
    Val = 1u << 3, Flush = 1u << 4, PayloadSize = 1u << 5, All = 63
};
#define __EMULE_FABRIC_MASK_OR(T)                                                            \
    constexpr T operator|(T a, T b) {                                                        \
        return static_cast<T>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));          \
    }
__EMULE_FABRIC_MASK_OR(UnicastWriteUpdateMask)
__EMULE_FABRIC_MASK_OR(UnicastInlineWriteUpdateMask)
__EMULE_FABRIC_MASK_OR(UnicastAtomicIncUpdateMask)
__EMULE_FABRIC_MASK_OR(UnicastScatterWriteUpdateMask)
__EMULE_FABRIC_MASK_OR(UnicastFusedAtomicIncUpdateMask)
__EMULE_FABRIC_MASK_OR(UnicastFusedScatterWriteAtomicIncUpdateMask)
#undef __EMULE_FABRIC_MASK_OR

// Experimental sub-namespaces. linear::experimental holds the fabric send free-functions CCL kernels
// call; emule routes through the connection's teleport-backed sends, so the stateless atomic-inc forms
// are accepted as no-ops here (the data-moving send_payload/scatter paths teleport via the sender).
namespace linear::experimental {
// Re-export the parent-namespace fabric types so the kernel's
// `using namespace tt::tt_fabric::linear::experimental;` makes NocUnicast*CommandHeader and the
// Unicast*UpdateMask enums visible unqualified — matching silicon, where the linear API namespace
// pulls these in transitively.
using namespace tt::tt_fabric;

// Compile-time mask test (silicon: api_common.h has_flag).
template <typename M>
constexpr bool __mask_has(M mask, M bit) {
    return (static_cast<uint32_t>(mask) & static_cast<uint32_t>(bit)) != 0;
}

// The stateful send API splits a fabric send into a one-time set_state (configure the static header
// fields: command type + the fields NOT in the update mask) and a per-send with_state (patch the
// masked dynamic fields, then stage payload + teleport the header). emule keeps a single packet-header
// layout + single teleport path; these wrappers write the same offsets the teleport decodes.

// ---- Unicast write ----
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_write_set_state(
    PacketHeader* hdr, uint8_t /*num_hops*/, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    hdr->noc_send_type = NOC_UNICAST_WRITE;
    if constexpr (__mask_has(Mask, UnicastWriteUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = payload_size;
    }
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        if constexpr (__mask_has(Mask, UnicastWriteUpdateMask::DstAddr)) {
            hdr->cmd_noc_address = cmd.noc_address;
        }
    }
}
template <
    UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_write_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        if constexpr (__mask_has(Mask, UnicastWriteUpdateMask::DstAddr)) {
            hdr->cmd_noc_address = cmd.noc_address;
        }
    }
    if constexpr (__mask_has(Mask, UnicastWriteUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = payload_size;
    }
    client->wait_for_empty_write_slot();
    client->send_payload_without_header_non_blocking_from_address(src_addr, hdr->payload_size_bytes);
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}

// ---- Scatter write (stateful: set_state lays the static fields once, with_state patches only the masked
// dynamic fields and reuses the rest — so both must honor the UpdateMask, not re-lay the whole command). ----
template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_scatter_write_set_state(
    PacketHeader* hdr, uint8_t /*num_hops*/, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    hdr->noc_send_type = NOC_UNICAST_SCATTER_WRITE;
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_scatter_fields<static_cast<uint32_t>(Mask)>(payload_size, cmd);
    } else if constexpr (__mask_has(Mask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = payload_size;
    }
}
template <
    UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_scatter_write_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_scatter_fields<static_cast<uint32_t>(Mask)>(payload_size, cmd);
    } else if constexpr (__mask_has(Mask, UnicastScatterWriteUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = payload_size;
    }
    client->wait_for_empty_write_slot();
    client->send_payload_without_header_non_blocking_from_address(src_addr, hdr->payload_size_bytes);
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}

// ---- Unicast atomic inc (no payload; the teleport increments val@8 at the dst noc address) ----
template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_atomic_inc_set_state(
    PacketHeader* hdr, uint8_t /*num_hops*/, Cmd cmd = nullptr) {
    hdr->noc_send_type = NOC_UNICAST_ATOMIC_INC;
    hdr->payload_size_bytes = 0;
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        if constexpr (__mask_has(Mask, UnicastAtomicIncUpdateMask::Val)) {
            *reinterpret_cast<uint32_t*>(&hdr->cmd_rest[0]) = cmd.val;  // val @8
        }
        if constexpr (__mask_has(Mask, UnicastAtomicIncUpdateMask::DstAddr)) {
            hdr->cmd_noc_address = cmd.noc_address;
        }
    }
}
template <
    UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_unicast_atomic_inc_with_state(Conn* client, PacketHeader* hdr, Cmd cmd = nullptr) {
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        if constexpr (__mask_has(Mask, UnicastAtomicIncUpdateMask::DstAddr)) {
            hdr->cmd_noc_address = cmd.noc_address;
        }
        if constexpr (__mask_has(Mask, UnicastAtomicIncUpdateMask::Val)) {
            *reinterpret_cast<uint32_t*>(&hdr->cmd_rest[0]) = cmd.val;
        }
    }
    client->wait_for_empty_write_slot();
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}

// ---- Fused unicast-write + atomic-inc (stateful): set_state lays the static fields (payload size, val,
// flush) once; with_state patches the per-packet write + semaphore dst addrs and reuses the rest. The
// teleport (runner case 3, NOC_FUSED_UNICAST_ATOMIC_INC) writes the payload then increments the semaphore. ----
template <UnicastFusedAtomicIncUpdateMask Mask = UnicastFusedAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state(
    PacketHeader* hdr, uint8_t /*num_hops*/, Cmd cmd = nullptr, uint16_t packet_size_bytes = 0) {
    hdr->noc_send_type = NOC_FUSED_UNICAST_ATOMIC_INC;
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_fused_fields<static_cast<uint32_t>(Mask)>(packet_size_bytes, cmd);
    } else if constexpr (__mask_has(Mask, UnicastFusedAtomicIncUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = packet_size_bytes;
    }
}
template <
    UnicastFusedAtomicIncUpdateMask Mask = UnicastFusedAtomicIncUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t packet_size_bytes = 0) {
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_fused_fields<static_cast<uint32_t>(Mask)>(packet_size_bytes, cmd);
    } else if constexpr (__mask_has(Mask, UnicastFusedAtomicIncUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = packet_size_bytes;
    }
    client->wait_for_empty_write_slot();
    client->send_payload_without_header_non_blocking_from_address(src_addr, hdr->payload_size_bytes);
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}

// ---- Fused scatter-write (2 chunks) + atomic-inc (stateful). Silicon encodes it as a NOC_UNICAST_SCATTER_WRITE
// packet — 2 write chunks + a 3rd semaphore-increment chunk, tagged via per-chunk chunk_encoding — so the
// teleport (runner case 4) does the 2 writes + the semaphore inc. set_state lays payload size / chunk sizes /
// val / encodings; with_state patches the write + sem dsts. ----
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask Mask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_scatter_write_atomic_inc_set_state(
    PacketHeader* hdr, uint8_t /*num_hops*/, Cmd cmd = nullptr, uint16_t packet_size_bytes = 0) {
    hdr->noc_send_type = NOC_UNICAST_SCATTER_WRITE;
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_fused_scatter_fields<static_cast<uint32_t>(Mask)>(packet_size_bytes, cmd);
    } else if constexpr (__mask_has(Mask, UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = packet_size_bytes;
    }
}
template <
    UnicastFusedScatterWriteAtomicIncUpdateMask Mask = UnicastFusedScatterWriteAtomicIncUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_unicast_noc_fused_scatter_write_atomic_inc_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t packet_size_bytes = 0) {
    if constexpr (!std::is_same_v<Cmd, std::nullptr_t>) {
        hdr->template apply_fused_scatter_fields<static_cast<uint32_t>(Mask)>(packet_size_bytes, cmd);
    } else if constexpr (__mask_has(Mask, UnicastFusedScatterWriteAtomicIncUpdateMask::PayloadSize)) {
        hdr->payload_size_bytes = packet_size_bytes;
    }
    client->wait_for_empty_write_slot();
    client->send_payload_without_header_non_blocking_from_address(src_addr, hdr->payload_size_bytes);
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}

// ---- Multicast atomic inc: emule has no real multicast; the teleport reaches the single neighbor
// chip (2-chip scope). The kernel issues one with_state per destination core, so each resolves to the
// neighbor's correct core. Bodies delegate to the unicast forms plus the extra range arg. ----
template <UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_atomic_inc_set_state(
    PacketHeader* hdr, uint8_t start_hops, uint8_t range, Cmd cmd = nullptr) {
    fabric_unicast_noc_unicast_atomic_inc_set_state<Mask>(hdr, start_hops, cmd);
    // Stamp the multicast extent the real set_state configures (emule had dropped `range`): the teleport
    // replays the atomic-inc to every chip in [start, start+range) in the worker's direction. Without this,
    // a line/ring barrier multicast (e.g. reduce_scatter's batch_ready_sem) reaches only one neighbor and
    // its wait-for-2*(ring_size-1) never completes. 1D form (the 2D extent is e/w/n/s, set elsewhere).
#ifndef EMULE_FABRIC_2D
    __emule_fabric_set_route(emule_route::key(hdr), emule_route::MCAST_1D, start_hops, range, 0, 0, 0, 0);
#endif
}
template <
    UnicastAtomicIncUpdateMask Mask = UnicastAtomicIncUpdateMask::None,
    typename Conn,
    typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_atomic_inc_with_state(Conn* client, PacketHeader* hdr, Cmd cmd = nullptr) {
    fabric_unicast_noc_unicast_atomic_inc_with_state<Mask>(client, hdr, cmd);
}

// ---- Per-header multicast unicast write / scatter write. The set_state form must stamp the multicast
// extent (start_distance, range) the same way the atomic-inc form above does: under EMULE_FABRIC8 the
// teleport resolves the destination chip(s) from this recorded route, so a data multicast write must
// replay to every chip in [start, start+range) in the worker's direction. Without this stamp the header
// carries no route (kind UNSET) and the teleport falls back to the single physical neighbor — which on a
// >2-chip mesh is the wrong chip, so all relayed shards are dropped (tt-emule #221). The 1D form is
// stamped here; the 2D extent (e/w/n/s) is set by the route setters. ----
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_write_set_state(
    PacketHeader* hdr, uint8_t start_hops, uint8_t range, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    fabric_unicast_noc_unicast_write_set_state<Mask>(hdr, start_hops, cmd, payload_size);
#ifndef EMULE_FABRIC_2D
    __emule_fabric_set_route(emule_route::key(hdr), emule_route::MCAST_1D, start_hops, range, 0, 0, 0, 0);
#endif
}
template <UnicastWriteUpdateMask Mask = UnicastWriteUpdateMask::None, typename Conn, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_unicast_write_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    fabric_unicast_noc_unicast_write_with_state<Mask>(client, hdr, src_addr, cmd, payload_size);
}
template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_scatter_write_set_state(
    PacketHeader* hdr, uint8_t start_hops, uint8_t range, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    fabric_unicast_noc_scatter_write_set_state<Mask>(hdr, start_hops, cmd, payload_size);
#ifndef EMULE_FABRIC_2D
    __emule_fabric_set_route(emule_route::key(hdr), emule_route::MCAST_1D, start_hops, range, 0, 0, 0, 0);
#endif
}
template <UnicastScatterWriteUpdateMask Mask = UnicastScatterWriteUpdateMask::None, typename Conn, typename Cmd = std::nullptr_t>
inline void fabric_multicast_noc_scatter_write_with_state(
    Conn* client, PacketHeader* hdr, uint32_t src_addr, Cmd cmd = nullptr, uint16_t payload_size = 0) {
    fabric_unicast_noc_scatter_write_with_state<Mask>(client, hdr, src_addr, cmd, payload_size);
}

// Stateless (single-shot) atomic-inc sends — CCL kernels call these directly (not the set/with-state pair):
// configure the packet header + teleport, faithful to linear/api.h. The route is already recorded by the
// caller's route setter (emule's to_chip_unicast is inert), so these don't re-stamp it.
// See docs/fabric-ccl-emulation.md.
template <typename Conn, typename Cmd>
inline void fabric_unicast_noc_unicast_atomic_inc(
    Conn* client, volatile PacketHeader* hdr, Cmd cmd, uint8_t /*num_hops*/) {
    hdr->to_noc_unicast_atomic_inc(cmd);
    hdr->payload_size_bytes = 0;
    client->wait_for_empty_write_slot();
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}
// Multicast atomic-inc: stamp the MCAST_1D extent (start_distance, range) — as the *_set_state form does —
// so the teleport replays the atomic-inc to every chip in [start, start+range), then teleport.
template <typename Conn, typename Cmd>
inline void fabric_multicast_noc_unicast_atomic_inc(
    Conn* client, volatile PacketHeader* hdr, Cmd cmd, uint8_t start_distance, uint8_t range) {
    hdr->to_noc_unicast_atomic_inc(cmd);
    hdr->payload_size_bytes = 0;
#ifndef EMULE_FABRIC_2D
    __emule_fabric_set_route(emule_route::key(hdr), emule_route::MCAST_1D, start_distance, range, 0, 0, 0, 0);
#endif
    client->wait_for_empty_write_slot();
    client->send_payload_flush_non_blocking_from_address(
        __emule_fabric_l1_off(hdr), sizeof(PacketHeader));
}
// Route-manager (RoutingPlaneConnectionManager&, route_id, ...) overloads and the stateless write forms
// are not reached by any emule-supported op today; the variadic no-ops below absorb them. The concrete
// client-pointer atomic-inc overloads above are preferred by partial ordering when a caller matches.
template <typename... A> inline void fabric_unicast_noc_unicast_atomic_inc(A&&...) {}
template <typename... A> inline void fabric_multicast_noc_unicast_atomic_inc(A&&...) {}
template <typename... A> inline void fabric_unicast_noc_unicast_write(A&&...) {}
template <typename... A> inline void fabric_multicast_noc_unicast_write(A&&...) {}
}  // namespace linear::experimental
namespace common::experimental {}
namespace mesh::experimental {}

}  // namespace tt::tt_fabric

// --- PACKET_HEADER_TYPE + global FabricConnectionManager ---
// Silicon exposes PACKET_HEADER_TYPE at global scope; keep that. It is the header type the fabric config
// selects — the low-latency (1D) header under a 1D fabric, the hybrid-mesh (2D) header under a 2D fabric —
// so a kernel's `is_same_v<PACKET_HEADER_TYPE, …>` route dispatch resolves to the matching branch (the 2D
// variants are distinct types above). Gated on the same EMULE_FABRIC_2D define the runner emits from the
// fabric config. See tt-emule #222.
#ifdef EMULE_FABRIC_2D
using PACKET_HEADER_TYPE = tt::tt_fabric::HybridMeshPacketHeader;
#else
using PACKET_HEADER_TYPE = tt::tt_fabric::LowLatencyPacketHeader;
#endif

// Silicon's noc_addr.h declares get_noc_address_components at GLOBAL scope (returning
// tt::tt_fabric::WorkerXY); CCL kernels call it unqualified. Decompose an emule NOC address
// (y@[42:48), x@[36:42), offset@[0:36)); mirrors __emule_resolve_noc_addr's encoding.
inline std::pair<tt::tt_fabric::WorkerXY, uint32_t> get_noc_address_components(uint64_t noc_addr) {
    uint32_t x = static_cast<uint32_t>((noc_addr >> 36) & 0x3f);
    uint32_t y = static_cast<uint32_t>((noc_addr >> 42) & 0x3f);
    uint32_t offset = static_cast<uint32_t>(noc_addr & 0xFFFFFFFFFull);
    return {tt::tt_fabric::WorkerXY(x, y), offset};
}

// CCL kernels (e.g. ttnn moe_utils.hpp) reference these unqualified; mirror that.
using tt::tt_fabric::WorkerToFabricEdmSender;
using tt::tt_fabric::WorkerToFabricMuxSender;
using tt::tt_fabric::MulticastRoutingCommandHeader;
using tt::tt_fabric::fabric_set_unicast_route;
using tt::tt_fabric::fabric_set_mcast_route;
using tt::tt_fabric::fabric_set_sparse_multicast_route;

// Ethernet routing direction (silicon: global enum in hostdevcommon/fabric_common.h). CCL kernels use
// it unqualified; emule resolves the destination via the cluster neighbor table, so the value is inert.
enum eth_chan_directions : uint8_t { EAST = 0, WEST = 1, NORTH = 2, SOUTH = 3, COUNT = 4 };

// Next-hop direction (silicon: reads device-L1 routing tables, which emule never populates). The
// teleport resolves the destination from the cluster neighbor table instead, so any direction is fine.
inline eth_chan_directions get_next_hop_router_direction(uint32_t /*dst_mesh_id*/, uint32_t /*dst_dev_id*/) {
    return eth_chan_directions::EAST;
}

// Real FabricConnectionManager is a global-scope class holding two senders (fwd/bwd ring directions).
class FabricConnectionManager final {
public:
    enum BuildFromArgsMode : uint8_t {
        BUILD_ONLY,
        BUILD_AND_OPEN_CONNECTION,
        BUILD_AND_OPEN_CONNECTION_START_ONLY,
    };
    tt::tt_fabric::WorkerToFabricEdmSender forward_fabric_sender;
    tt::tt_fabric::WorkerToFabricEdmSender backward_fabric_sender;

    template <BuildFromArgsMode build_mode = BUILD_ONLY>
    static FabricConnectionManager build_from_args(std::size_t& /*arg_idx*/) {
        return FabricConnectionManager{};
    }
    static constexpr bool is_logically_connected() { return true; }
    bool has_forward_connection() const { return true; }
    bool has_backward_connection() const { return true; }
    void open() {}
    void open_start() {}
    void open_finish() {}
    void close() {}
    void close_start() {}
    void close_finish() {}
    // Tag each direction so the teleport can resolve a 1D dst by the worker's connection direction
    // (the host recorded fwd-then-bwd per worker; index 0=fwd, 1=bwd).
    tt::tt_fabric::WorkerToFabricEdmSender& get_forward_connection() {
        forward_fabric_sender.emule_conn_index = 0;
        return forward_fabric_sender;
    }
    tt::tt_fabric::WorkerToFabricEdmSender& get_backward_connection() {
        backward_fabric_sender.emule_conn_index = 1;
        return backward_fabric_sender;
    }
};

// ---- PacketHeaderPool ----
// NOTE: PacketHeaderPool is defined in the `tt_metal/fabric/hw/inc/packet_header_pool.h` shadow (NOT
// here). It must hand out pointers into the worker's real L1 reserved region (via
// __emule_local_l1_to_ptr) so the fabric shim's bridge_l1-relative header narrowing round-trips
// through the teleport. The shadow includes dev_mem_map.h for the reserved MEM_PACKET_HEADER_POOL_BASE
// region, which this stub cannot reach.
struct HeaderTableEntry_t {
    PACKET_HEADER_TYPE* first = nullptr;
    PACKET_HEADER_TYPE* second = nullptr;
};

// ---- sdpa_fabric helpers (used by sdpa op) ----
namespace sdpa_fabric {
template <typename... Args> inline void send_to_neighbor(Args&&...) {}
template <typename... Args> inline void send_to_neighbor_no_sem(Args&&...) {}
template <typename... Args> inline void send_sem_inc_to_neighbor(Args&&...) {}
}  // namespace sdpa_fabric
