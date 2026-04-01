#pragma once
// Emulation stubs for experimental endpoint types + noc_traits_t.
// Bridge functions resolve L1/DRAM offsets to host pointers directly.

#include <cstdint>
#include "jit_hw/experimental/noc.h"

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset);
extern uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);

namespace experimental {

struct UnicastEndpoint {
    uint64_t get_noc_unicast_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc) const {
        return ::get_noc_addr(noc_x, noc_y, addr, noc);
    }
};

struct MulticastEndpoint {
    uint64_t get_noc_multicast_addr(
        uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end,
        uint32_t addr, uint8_t noc) const {
        return ::get_noc_multicast_addr(noc_x_start, noc_y_start, noc_x_end, noc_y_end, addr, noc);
    }
};

enum AllocatorBankType { L1, DRAM };

template <AllocatorBankType>
struct AllocatorBank {};

struct ReadSpec {
    uint32_t bank_id = 0;
    uint32_t addr = 0;
};

struct WriteSpec {
    uint32_t bank_id = 0;
    uint32_t addr = 0;
};

// --- noc_traits_t<UnicastEndpoint> ---
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

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const UnicastEndpoint& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const UnicastEndpoint& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_unicast_addr(args.noc_x, args.noc_y, args.addr, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};

// --- noc_traits_t<MulticastEndpoint> ---
// Returns the raw multicast NOC address (encoded with start/end coords and L1 offset).
// The Noc::async_write_multicast method passes this to __emule_multicast_write.
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
    static uint64_t dst_addr_mcast(const MulticastEndpoint& dst, const Noc& noc, const dst_args_mcast_type& args) {
        return dst.get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start, args.noc_x_end, args.noc_y_end, args.addr, noc.get_noc_id());
    }
};

// --- noc_traits_t<AllocatorBank<L1>> ---
template <>
struct noc_traits_t<AllocatorBank<L1>> {
    using src_args_type = ReadSpec;
    using dst_args_type = WriteSpec;

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const AllocatorBank<L1>&, const Noc&, const ReadSpec& args) {
        return reinterpret_cast<uintptr_t>(__emule_local_l1_ptr(args.addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const AllocatorBank<L1>&, const Noc&, const WriteSpec& args) {
        return reinterpret_cast<uintptr_t>(__emule_local_l1_ptr(args.addr));
    }
};

// --- noc_traits_t<AllocatorBank<DRAM>> ---
template <>
struct noc_traits_t<AllocatorBank<DRAM>> {
    using src_args_type = ReadSpec;
    using dst_args_type = WriteSpec;

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const AllocatorBank<DRAM>&, const Noc&, const ReadSpec& args) {
        return reinterpret_cast<uintptr_t>(__emule_dram_ptr(args.addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const AllocatorBank<DRAM>&, const Noc&, const WriteSpec& args) {
        return reinterpret_cast<uintptr_t>(__emule_dram_ptr(args.addr));
    }
};

}  // namespace experimental
