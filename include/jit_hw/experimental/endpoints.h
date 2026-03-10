#pragma once
// Emulation stubs for experimental AllocatorBank endpoints + noc_traits_t.
// Bridge functions resolve L1/DRAM offsets to host pointers directly.

#include <cstdint>
#include "jit_hw/experimental/noc.h"

// C-linkage bridge functions (resolved at dlopen time from emulated_program_runner).
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset);

namespace experimental {

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

// --- noc_traits_t<AllocatorBank<L1>> ---
// L1 bank: resolve via __emule_local_l1_ptr (current core's mmap'd L1).
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
// DRAM bank: resolve via __emule_dram_ptr.
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
