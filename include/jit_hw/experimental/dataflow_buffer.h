// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emulation stub for experimental::DataflowBuffer.
// Wraps the dfb_api.h free functions in the class API that upstream kernels use.
// The real implementation (tt_metal/hw/inc/experimental/dataflow_buffer.h) uses
// hardware tile counters and LLK interfaces; this version delegates to the
// emulation TileCounterArray via thread-local __emule_dfbs / __emule_tc_array.

#include "jit_hw/api/dfb_api.h"
#include "jit_hw/experimental/noc.h"
#include "jit_hw/experimental/lock.h"

namespace experimental {

struct DFBAccessor {
    explicit constexpr DFBAccessor(uint16_t id) noexcept : id(id) {}
    uint16_t id;
};

class DataflowBuffer {
public:
    DataflowBuffer(uint16_t logical_dfb_id) : logical_dfb_id_(logical_dfb_id) {}
    DataflowBuffer(DFBAccessor accessor) : logical_dfb_id_(accessor.id) {}

    uint16_t get_id() const { return logical_dfb_id_; }

    uint32_t get_entry_size() const { return dfb_get_entry_size(logical_dfb_id_); }

    uint32_t get_stride_size() const { return __emule_dfbs[logical_dfb_id_].stride_size; }

    void reserve_back(uint16_t num_entries) { dfb_reserve_back(logical_dfb_id_, num_entries); }

    void push_back(uint16_t num_entries) { dfb_push_back(logical_dfb_id_, num_entries); }

    void wait_front(uint16_t num_entries) { dfb_wait_front(logical_dfb_id_, num_entries); }

    void pop_front(uint16_t num_entries) { dfb_pop_front(logical_dfb_id_, num_entries); }

    void finish() { dfb_finish(logical_dfb_id_); }

    void write_barrier(const Noc& noc) const { noc.async_write_barrier(); }

    uint32_t get_write_ptr() const { return dfb_get_write_ptr(logical_dfb_id_); }

    uint32_t get_read_ptr() const { return dfb_get_read_ptr(logical_dfb_id_); }

    template <typename Src>
    void read_in(const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args) {
        reserve_back(1);
        noc.async_read(src, *this, get_entry_size(), src_args, {});
        noc.async_read_barrier();
        push_back(1);
    }

    template <typename Dst>
    void write_out(const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args) {
        wait_front(1);
        noc.async_write(*this, dst, get_entry_size(), {}, dst_args);
        noc.async_write_barrier();
        pop_front(1);
    }

    [[nodiscard]] auto scoped_lock() {
        return Lock([this]() { /* no-op release in emulation */ });
    }

private:
    uint16_t logical_dfb_id_;
};

template <>
struct noc_traits_t<DataflowBuffer> {
    struct src_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_type {
        uint32_t offset_bytes{};
    };
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };

    // DFB wr_ptr/rd_ptr are absolute host addresses (l1_base_ + bump), set by
    // emulated_program_runner via l1_alloc().  MAP_32BIT guarantees they fit in
    // uint32_t; zero-extending to uintptr_t yields a valid host pointer.
    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const DataflowBuffer& src, const Noc&, const src_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "DataflowBuffer without mcast range can only be used as L1 source");
        return static_cast<uintptr_t>(src.get_read_ptr() + args.offset_bytes);
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const DataflowBuffer& dst, const Noc&, const dst_args_type& args) {
        static_assert(
            address_type == Noc::AddressType::LOCAL_L1,
            "DataflowBuffer without mcast range can only be used as L1 destination");
        return static_cast<uintptr_t>(dst.get_write_ptr() + args.offset_bytes);
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr_mcast(const DataflowBuffer& dst, const Noc&, const dst_args_mcast_type& args) {
        static_assert(
            address_type == Noc::AddressType::NOC, "DataflowBuffer with mcast range cannot be used as L1 source");
        return static_cast<uintptr_t>(dst.get_write_ptr() + args.offset_bytes);
    }
};

// Implicit-sync overloads: single-entry read/write that folds the DFB
// reserve_back/push_back (producer) or wait_front/pop_front (consumer) into
// the noc call itself.  In emulation the transfer is a synchronous memcpy,
// so the bookkeeping happens inline here.
template <Noc::TxnIdMode txn_id_mode, typename Src>
inline std::enable_if_t<txn_id_mode == Noc::TxnIdMode::ENABLED>
Noc::async_read(
    const Src& src,
    DataflowBuffer& dst,
    const typename noc_traits_t<Src>::src_args_type& src_args,
    const DataflowBufferArgs& dst_args) const {
    const uint32_t size_bytes = dst.get_entry_size();
    dst.reserve_back(1);
    uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::NOC>(src, *this, src_args);
    uintptr_t d = static_cast<uintptr_t>(dst.get_write_ptr() + dst_args.offset_bytes);
    if (s && d) {
        std::memcpy(reinterpret_cast<uint8_t*>(d), reinterpret_cast<uint8_t*>(s), size_bytes);
    }
    dst.push_back(1);
}

template <Noc::TxnIdMode txn_id_mode, typename Dst>
inline std::enable_if_t<txn_id_mode == Noc::TxnIdMode::ENABLED>
Noc::async_write(
    DataflowBuffer& src,
    const Dst& dst,
    const DataflowBufferArgs& src_args,
    const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    const uint32_t size_bytes = src.get_entry_size();
    src.wait_front(1);
    uintptr_t s = static_cast<uintptr_t>(src.get_read_ptr() + src_args.offset_bytes);
    uintptr_t d = noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args);
    if (s && d) {
        std::memcpy(reinterpret_cast<uint8_t*>(d), reinterpret_cast<uint8_t*>(s), size_bytes);
    }
    src.pop_front(1);
}

}  // namespace experimental
