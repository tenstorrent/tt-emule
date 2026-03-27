#pragma once
// Emulation stub for experimental::CircularBuffer + noc_traits_t specialization.
// Delegates CB operations to jit_hw/api/cb_api.h (the shared sync functions).

#include "jit_hw/api/cb_api.h"
#include "jit_hw/experimental/noc.h"

namespace experimental {

class CircularBuffer {
public:
    enum class AddrSelector { WRITE_PTR, READ_PTR };

    explicit CircularBuffer(uint32_t cb_id) : cb_id_(cb_id) {}
    uint32_t get_cb_id() const { return cb_id_; }

    void reserve_back(int32_t n) { cb_reserve_back(cb_id_, n); }
    void push_back(int32_t n)    { cb_push_back(cb_id_, n); }
    void wait_front(int32_t n)   { cb_wait_front(cb_id_, n); }
    void pop_front(int32_t n)    { cb_pop_front(cb_id_, n); }

    bool pages_reservable_at_back(int32_t n) const { return true; }
    bool pages_available_at_front(int32_t n) const { return true; }

    uint32_t get_tile_size() const { return ::get_tile_size(cb_id_); }
    uint32_t get_write_ptr() const { return ::get_write_ptr(cb_id_); }
    uint32_t get_read_ptr() const  { return ::get_read_ptr(cb_id_); }

private:
    uint32_t cb_id_;
};

// noc_traits_t<CircularBuffer>:
//   As dst (Noc reads INTO CB): use write pointer.
//   As src (Noc writes FROM CB): use read pointer.
//   All addresses are 32-bit host pointers (L1 mmap'd below 4 GB).
template <>
struct noc_traits_t<CircularBuffer> {
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const CircularBuffer& cb, const Noc&, const src_args_type& args) {
        return static_cast<uintptr_t>(cb.get_read_ptr()) + args.offset_bytes;
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const CircularBuffer& cb, const Noc&, const dst_args_type& args) {
        return static_cast<uintptr_t>(cb.get_write_ptr()) + args.offset_bytes;
    }
};

template <CircularBuffer::AddrSelector AddrSel>
struct CircularBufferView {
    const CircularBuffer& cb;
    explicit constexpr CircularBufferView(const CircularBuffer& c) : cb(c) {}
};

template <CircularBuffer::AddrSelector AddrSel>
constexpr auto use(const CircularBuffer& cb) {
    return CircularBufferView<AddrSel>(cb);
}

template <CircularBuffer::AddrSelector AddrSel>
struct noc_traits_t<CircularBufferView<AddrSel>> {
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const CircularBufferView<AddrSel>& view, const Noc&, const src_args_type& args) {
        return static_cast<uintptr_t>(get_local_addr(view)) + args.offset_bytes;
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const CircularBufferView<AddrSel>& view, const Noc&, const dst_args_type& args) {
        return static_cast<uintptr_t>(get_local_addr(view)) + args.offset_bytes;
    }

private:
    static constexpr auto get_local_addr(const CircularBufferView<AddrSel>& view) {
        if constexpr (AddrSel == CircularBuffer::AddrSelector::READ_PTR) {
            return view.cb.get_read_ptr();
        } else {
            return view.cb.get_write_ptr();
        }
    }
};

}  // namespace experimental
