// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation shadow for api/core_local_mem.h.
//
// The real silicon header chains through debug/sanitize headers that include
// dev_msgs.h which is not available in the x86 JIT environment, so emule
// provides its own inline body. Body mirrors silicon's ::CoreLocalMem with
// the address-translation hook for raw L1 firmware offsets.

#include <cstddef>
#include <cstdint>
#include "jit_hw/api/dataflow/noc.h"  // ::noc_traits_t primary template and ::Noc
#include "jit_hw/internal/emule_thread_ctx.h"

// Raw L1 offsets are always < 16 MB (L1 is at most 4 MB).
// MAP_32BIT host pointers are >= 0x40000000.
static constexpr uintptr_t CORE_LOCAL_MEM_RAW_OFFSET_THRESHOLD = 0x1000000;

template <typename T, typename AddressType = uintptr_t>
class CoreLocalMem {
    using difference_type = std::ptrdiff_t;

    static uintptr_t translate(uintptr_t addr) {
        if (addr < CORE_LOCAL_MEM_RAW_OFFSET_THRESHOLD && __emule_self->bridge_l1) {
            return reinterpret_cast<uintptr_t>(__emule_self->bridge_l1) + addr;
        }
        return addr;
    }

public:
    CoreLocalMem(AddressType address) : address_(translate(static_cast<uintptr_t>(address))) {}
    CoreLocalMem(T* ptr) : address_(reinterpret_cast<uintptr_t>(ptr)) {}
    CoreLocalMem(const CoreLocalMem&) = default;
    CoreLocalMem& operator=(const CoreLocalMem&) = default;

    T* get_unsafe_ptr() const { return reinterpret_cast<T*>(address_); }
    AddressType get_address() const { return static_cast<AddressType>(address_); }
    T* data() { return get_unsafe_ptr(); }

    T& operator[](uint32_t index) const { return get_unsafe_ptr()[index]; }
    T& operator*() const { return get_unsafe_ptr()[0]; }
    T* operator->() const { return get_unsafe_ptr(); }

    CoreLocalMem& operator+=(difference_type offset) { address_ += offset * sizeof(T); return *this; }
    CoreLocalMem& operator-=(difference_type offset) { address_ -= offset * sizeof(T); return *this; }
    CoreLocalMem& operator++() { address_ += sizeof(T); return *this; }
    CoreLocalMem& operator--() { address_ -= sizeof(T); return *this; }
    CoreLocalMem operator++(int) { auto tmp = *this; ++(*this); return tmp; }
    CoreLocalMem operator--(int) { auto tmp = *this; --(*this); return tmp; }
    CoreLocalMem operator+(difference_type offset) const { return CoreLocalMem(get_unsafe_ptr() + offset); }
    CoreLocalMem operator-(difference_type offset) const { return CoreLocalMem(get_unsafe_ptr() - offset); }
    difference_type operator-(const CoreLocalMem& other) const {
        return (static_cast<difference_type>(address_) - static_cast<difference_type>(other.address_)) / sizeof(T);
    }

    bool operator==(const CoreLocalMem& other) const { return address_ == other.address_; }
    bool operator!=(const CoreLocalMem& other) const { return address_ != other.address_; }
    bool operator<(const CoreLocalMem& other) const { return address_ < other.address_; }
    bool operator<=(const CoreLocalMem& other) const { return address_ <= other.address_; }
    bool operator>(const CoreLocalMem& other) const { return address_ > other.address_; }
    bool operator>=(const CoreLocalMem& other) const { return address_ >= other.address_; }
    explicit operator bool() const { return address_ != 0; }

    operator uint8_t*() { return reinterpret_cast<uint8_t*>(address_); }

private:
    uintptr_t address_;
};

// Specialise global ::noc_traits_t<CoreLocalMem<T,AddressType>>.
// CoreLocalMem::get_address() already returns a host pointer
// (translates L1 firmware offset → host via __emule_self->bridge_l1 for small addrs).
template <typename T, typename AddressType>
struct noc_traits_t<CoreLocalMem<T, AddressType>> {
    struct src_args_type       { uintptr_t offset_bytes = 0; };
    struct dst_args_type       { uintptr_t offset_bytes = 0; };
    struct dst_args_mcast_type { uintptr_t offset_bytes = 0; };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const CoreLocalMem<T, AddressType>& src, const Noc&,
                              const src_args_type& args) {
        return static_cast<uintptr_t>(src.get_address()) + args.offset_bytes;
    }
    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc&,
                              const dst_args_type& args) {
        return static_cast<uintptr_t>(dst.get_address()) + args.offset_bytes;
    }
    template <Noc::AddressType AT>
    static uintptr_t dst_addr_mcast(const CoreLocalMem<T, AddressType>& dst, const Noc&,
                                    const dst_args_mcast_type& args) {
        // Multicast in emulation: return the local host address; the multicast
        // write in __emule_multicast_write iterates all registered cores.
        return static_cast<uintptr_t>(dst.get_address()) + args.offset_bytes;
    }
};
