#pragma once
// Emulation stub for experimental::Noc and noc_traits_t.
// Dispatches async_read/write through noc_traits_t specializations, then does
// a direct memcpy between the resolved host pointers.

#include <cstdint>
#include <cstring>

namespace experimental {

template <typename T>
struct noc_traits_t {
    static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
};

class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };

    Noc() : noc_id_(0) {}
    explicit Noc(uint8_t noc_id) : noc_id_(noc_id) {}
    uint8_t get_noc_id() const { return noc_id_; }

    // async_read: NOC src → LOCAL_L1 dst.
    // In emulation, traits resolve to host pointers directly (uintptr_t).
    template <typename Src, typename Dst>
    void async_read(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const typename noc_traits_t<Src>::src_args_type& src_args,
        const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
        uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::NOC>(src, *this, src_args);
        uintptr_t d = noc_traits_t<Dst>::template dst_addr<AddressType::LOCAL_L1>(dst, *this, dst_args);
        if (s && d) {
            std::memcpy(reinterpret_cast<uint8_t*>(d), reinterpret_cast<uint8_t*>(s), size_bytes);
        }
    }

    // async_write: LOCAL_L1 src → NOC dst.
    template <typename Src, typename Dst>
    void async_write(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const typename noc_traits_t<Src>::src_args_type& src_args,
        const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
        uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args);
        uintptr_t d = noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args);
        if (s && d) {
            std::memcpy(reinterpret_cast<uint8_t*>(d), reinterpret_cast<uint8_t*>(s), size_bytes);
        }
    }

    void async_read_barrier() const {}
    void async_write_barrier() const {}
    void async_full_barrier() const {}

private:
    uint8_t noc_id_;
};

}  // namespace experimental
