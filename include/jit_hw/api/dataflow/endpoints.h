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
extern "C" uint8_t* __emule_dram_ptr_at(uint32_t bank_id, uint64_t offset);
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
    // In emulation, bank routing is handled by __emule_dram_ptr / __emule_local_l1_ptr.
    // This method exists for API compatibility; the actual resolution is in noc_traits_t.
    uint64_t get_noc_addr_from_bank_id(uint32_t /*bank_id*/, uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr(my_x[noc], my_y[noc], addr, noc);
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
    template <Noc::AddressType AT>
    static uintptr_t src_addr(const UnicastEndpoint& src, const Noc& noc,
                               const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_unicast_addr(
            args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const UnicastEndpoint& dst, const Noc& noc,
                               const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_unicast_addr(
            args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
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

// ---- noc_traits_t<AllocatorBank<L1>> ----

template <>
struct noc_traits_t<AllocatorBank<AllocatorBankType::L1>> {
    struct src_args_type { uint32_t bank_id{}; uint32_t addr{}; };
    struct dst_args_type { uint32_t bank_id{}; uint32_t addr{}; };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const AllocatorBank<AllocatorBankType::L1>&,
                               const Noc&, const src_args_type& args) {
        return reinterpret_cast<uintptr_t>(__emule_local_l1_ptr(args.addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const AllocatorBank<AllocatorBankType::L1>&,
                               const Noc&, const dst_args_type& args) {
        return reinterpret_cast<uintptr_t>(__emule_local_l1_ptr(args.addr));
    }
};

// ---- noc_traits_t<AllocatorBank<DRAM>> ----
//
// Route through `__emule_dram_ptr_at(bank_id, addr)` so each DRAM bank lands
// at its own slice of emule's flat DRAM buffer (per
// `bank_to_dram_offset[]`). Using the bank-agnostic `__emule_dram_ptr`
// aliased every bank to bank 0's slice — fine for single-bank tensors
// (Llama-3.2-1B fits) but wrong as soon as anything spans multiple banks.

template <>
struct noc_traits_t<AllocatorBank<AllocatorBankType::DRAM>> {
    struct src_args_type { uint32_t bank_id{}; uint32_t addr{}; };
    struct dst_args_type { uint32_t bank_id{}; uint32_t addr{}; };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const AllocatorBank<AllocatorBankType::DRAM>&,
                               const Noc&, const src_args_type& args) {
        return reinterpret_cast<uintptr_t>(__emule_dram_ptr_at(args.bank_id, args.addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const AllocatorBank<AllocatorBankType::DRAM>&,
                               const Noc&, const dst_args_type& args) {
        return reinterpret_cast<uintptr_t>(__emule_dram_ptr_at(args.bank_id, args.addr));
    }
};
