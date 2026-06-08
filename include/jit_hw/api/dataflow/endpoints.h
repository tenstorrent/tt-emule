// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for api/dataflow/endpoints.h.
//
// Adapted from jit_hw/experimental/endpoints.h — same functional logic but
// with the 'namespace experimental {}' wrapper removed to match the promoted
// API introduced by tt-metal commit a67eb83be3c.
//
// Bridge model: noc_traits_t specialisations always resolve to host pointers
// via __emule_resolve_noc_addr / __emule_local_l1_ptr / __emule_dram_ptr,
// regardless of the AddressType template parameter, because the emulation
// Noc class uses the returned pointer directly in std::memcpy.

#include <cstdint>
#include "jit_hw/api/dataflow/noc.h"

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset);
extern uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);

// ---- UnicastEndpoint ----

struct UnicastEndpoint {
    uint64_t get_noc_unicast_addr(uint32_t noc_x, uint32_t noc_y,
                                   uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, addr, noc);
    }
};

// ---- MulticastEndpoint ----

struct MulticastEndpoint {
    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start,
        uint32_t noc_x_end,   uint32_t noc_y_end,
        uint32_t addr, uint8_t noc) const {
        return ::get_noc_multicast_addr(
            noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    }
};

// ---- AllocatorBank ----

enum class AllocatorBankType { L1, DRAM };

template <AllocatorBankType bank_type>
struct AllocatorBank {
    // Mirrors silicon: delegate to the templated global helper, which routes
    // (bank_id, addr) to the owning core's NOC coords via the per-arch
    // l1_bank_to_noc_xy / dram_bank_to_noc_xy tables.
    uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr_from_bank_id<bank_type == AllocatorBankType::DRAM>(bank_id, addr, noc);
    }
};

// ---- noc_traits_t<UnicastEndpoint> ----

template <>
struct noc_traits_t<UnicastEndpoint> {
    struct src_args_type {
        uint32_t noc_x{};
        uint32_t noc_y{};
        uint32_t addr{};
    };
    struct dst_args_type {
        uint32_t noc_x{};
        uint32_t noc_y{};
        uint32_t addr{};
    };

    // In emulation, resolve to host pointer for both AddressTypes —
    // the Noc class uses the result directly in memcpy.
    // Mirrors the real noc_traits_t<UnicastEndpoint> (tt_metal api/dataflow/endpoints.h):
    // a LOCAL_L1 endpoint (async_write source / async_read destination) returns the raw
    // local L1 offset — emule's to_host_ptr<LOCAL_L1> then resolves it to the issuing
    // core's own L1 (__emule_local_l1_to_ptr via __emule_bridge_l1), exactly the silicon
    // contract. A NOC endpoint (remote source/dest) uses the caller-supplied coords.
    // Only emule adaptation: the NOC branch resolves to a host pointer (silicon returns
    // the raw NOC address for the NIU; emule's to_host_ptr<NOC> casts directly).
    template <Noc::AddressType AT>
    static uintptr_t src_addr(const UnicastEndpoint& src, const Noc& noc,
                               const src_args_type& args) {
        if constexpr (AT == Noc::AddressType::LOCAL_L1) {
            return args.addr;
        } else {
            uint64_t noc_addr = src.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
            return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
        }
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const UnicastEndpoint& dst, const Noc& noc,
                               const dst_args_type& args) {
        if constexpr (AT == Noc::AddressType::LOCAL_L1) {
            return args.addr;
        } else {
            uint64_t noc_addr = dst.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
            return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
        }
    }
};

// ---- noc_traits_t<MulticastEndpoint> ----
// Returns the raw multicast NOC address; the Noc class passes it to
// __emule_multicast_write which iterates the rectangle.

template <>
struct noc_traits_t<MulticastEndpoint> {
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t addr{};
    };

    template <Noc::AddressType AT>
    static uint64_t dst_addr_mcast(const MulticastEndpoint& dst, const Noc& noc,
                                    const dst_args_mcast_type& args) {
        return dst.get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start,
            args.noc_x_end,   args.noc_y_end,
            args.addr, noc.get_noc_id());
    }
};

// ---- noc_traits_t<AllocatorBank<...>> ----
// Both L1 and DRAM resolve per-bank: (bank_id, addr) → NOC address via
// get_noc_addr_from_bank_id<bank_type==DRAM>, then __emule_resolve_noc_addr to
// a host pointer. The previous L1 specialization dropped bank_id and returned
// __emule_local_l1_ptr(args.addr) directly — only correct if bank_id == 0 (and
// even then only if the local core happens to be the owner). Drop that drift;
// match silicon's single templated specialization.

template <AllocatorBankType bank_type>
struct noc_traits_t<AllocatorBank<bank_type>> {
    struct src_args_type { uint32_t bank_id{}; uint32_t addr{}; };
    struct dst_args_type { uint32_t bank_id{}; uint32_t addr{}; };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const AllocatorBank<bank_type>& src,
                               const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr_from_bank_id(args.bank_id, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const AllocatorBank<bank_type>& dst,
                               const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr_from_bank_id(args.bank_id, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};
