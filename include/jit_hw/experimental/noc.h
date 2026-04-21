#pragma once
// Emulation stub for experimental::Noc and noc_traits_t.
// Dispatches async_read/write through noc_traits_t specializations, then does
// a direct memcpy between the resolved host pointers.

#include <cstdint>
#include <cstring>
#include <type_traits>

extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);

namespace experimental {

class DataflowBuffer;

// Matches upstream tt_metal/hw/inc/experimental/noc.h: concrete arg struct
// used by the DFB implicit-sync Noc overloads.
struct DataflowBufferArgs {
    uint32_t offset_bytes{};
};

template <typename T>
struct noc_traits_t {
    static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
};

class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };
    // Matches upstream tt-metal: TxnIdMode::ENABLED selects the DFB implicit-sync
    // overloads that internally manage reserve_back/wait_front and omit size_bytes.
    enum class TxnIdMode { ENABLED, DISABLED };

    Noc() : noc_id_(0) {}
    explicit Noc(uint8_t noc_id) : noc_id_(noc_id) {}
    uint8_t get_noc_id() const { return noc_id_; }

    // async_read: NOC src → LOCAL_L1 dst.
    // In emulation, traits resolve to host pointers directly (uintptr_t).
    template <TxnIdMode txn_id_mode = TxnIdMode::DISABLED, typename Src, typename Dst>
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
    template <TxnIdMode txn_id_mode = TxnIdMode::DISABLED, typename Src, typename Dst>
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

    // Implicit-sync overloads (Quasar only in real HW, always available here).
    // When TxnIdMode::ENABLED is used with a DataflowBuffer side, the size is
    // inferred from dfb.get_entry_size() and the push/pop bookkeeping is folded
    // in so the kernel does not call reserve_back/wait_front explicitly.
    // Defined out-of-line below because DataflowBuffer is incomplete here.
    template <TxnIdMode txn_id_mode, typename Src>
    std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED>
    async_read(
        const Src& src,
        DataflowBuffer& dst,
        const typename noc_traits_t<Src>::src_args_type& src_args,
        const DataflowBufferArgs& dst_args = {}) const;

    template <TxnIdMode txn_id_mode, typename Dst>
    std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED>
    async_write(
        DataflowBuffer& src,
        const Dst& dst,
        const DataflowBufferArgs& src_args,
        const typename noc_traits_t<Dst>::dst_args_type& dst_args) const;

    // async_write_multicast: Delegates to __emule_multicast_write which
    // iterates over the rectangle of target cores and copies data to each.
    template <typename... Extra, typename Src, typename Dst>
    void async_write_multicast(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        uint32_t num_dsts,
        const typename noc_traits_t<Src>::src_args_type& src_args,
        const typename noc_traits_t<Dst>::dst_args_mcast_type& dst_args,
        bool linked = false,
        uint32_t trid = 0) const {
        uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args);
        auto mcast_noc_addr = noc_traits_t<Dst>::template dst_addr_mcast<AddressType::NOC>(dst, *this, dst_args);
        if (s) {
            __emule_multicast_write(static_cast<uint64_t>(mcast_noc_addr),
                                    reinterpret_cast<const uint8_t*>(s), size_bytes);
        }
    }

    void async_read_barrier() const {}
    void async_write_barrier() const {}
    void async_writes_flushed() const {}
    void async_atomic_barrier() const {}
    void async_full_barrier() const {}

private:
    uint8_t noc_id_;
};

}  // namespace experimental
