#pragma once
// Emulation stub for experimental::CoreLocalMem<T>.
// Wraps an L1 address as a typed pointer into Core's mmap'd L1 region.

#include <cstdint>
#include <cstddef>
#include "tt_emule/device.hpp"
#include "jit_hw/experimental/noc.h"

extern thread_local tt_emule::Core* __core;

namespace experimental {

template <typename T, typename AddressType = uintptr_t>
class CoreLocalMem {
    using difference_type = std::ptrdiff_t;

public:
    CoreLocalMem(AddressType address) : address_(static_cast<uintptr_t>(address)) {}
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

template <typename T, typename AddressType>
struct noc_traits_t<CoreLocalMem<T, AddressType>> {
    struct src_args_type {
        uintptr_t offset_bytes = 0;
    };
    struct dst_args_type {
        uintptr_t offset_bytes = 0;
    };
    struct dst_args_mcast_type {};

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const CoreLocalMem<T, AddressType>& src, const Noc&, const src_args_type& args) {
        return static_cast<uintptr_t>(src.get_address()) + args.offset_bytes;
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc&, const dst_args_type& args) {
        return static_cast<uintptr_t>(dst.get_address()) + args.offset_bytes;
    }
};

}  // namespace experimental
