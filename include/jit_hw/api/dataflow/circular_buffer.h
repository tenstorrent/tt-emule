// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for api/dataflow/circular_buffer.h.
//
// Self-contained replacement: the real header includes profiler headers from
// tools/profiler/ (not on the JIT include path) and depends on a richer
// CBInterface struct than our emulation provides. This stub adapts
// jit_hw/experimental/circular_buffer.h — same CB operation delegates,
// namespace removed, full noc_traits_t surface (including dst_args_mcast_type),
// and unconditional get_tile_size/get_tile_hw/get_dataformat.

#include "jit_hw/api/cb_api.h"
#include "jit_hw/api/dataflow/noc.h"
#include "api/lock.h"    // RAII Lock — reachable via tt_metal/hw/inc/

class CircularBuffer {
public:
    enum class AddrSelector { WRITE_PTR, READ_PTR };

    explicit CircularBuffer(uint32_t cb_id) : cb_id_(cb_id) {}
    uint32_t get_cb_id() const { return cb_id_; }

    // ---- CB FIFO operations ----
    void reserve_back(int32_t n)  { cb_reserve_back(cb_id_, n); }
    void push_back(int32_t n)     { cb_push_back(cb_id_, n); }
    void wait_front(int32_t n)    { cb_wait_front(cb_id_, n); }
    void pop_front(int32_t n)     { cb_pop_front(cb_id_, n); }

    // Non-blocking availability checks — in emulation always return true
    // (the SPSC locking in cb_reserve_back/cb_wait_front handles backpressure).
    bool pages_reservable_at_back(int32_t /*num_pages*/) const { return true; }
    bool pages_available_at_front(int32_t /*num_pages*/) const { return true; }

    // ---- Pointer queries ----
    uint32_t get_write_ptr() const { return ::get_write_ptr(cb_id_); }
    uint32_t get_read_ptr()  const { return ::get_read_ptr(cb_id_); }

    // Read one uint32 element (raw, un-permuted) from a tile at the front of the
    // CB. Mirrors silicon CircularBuffer::read_tile_value (api/dataflow/
    // circular_buffer.h): byte addr = fifo_rd_ptr + page_size*tile_index, then
    // index element_offset uint32s in. Used by reduction/manual_seed compute
    // kernels. emule reads directly (no UNPACK→mailbox→MATH/PACK relay needed).
    uint32_t read_tile_value(uint32_t tile_index, uint32_t element_offset) const {
        uint8_t* p = __emule_local_l1_to_ptr(
            get_read_ptr() + tile_index * get_tile_size() + element_offset * sizeof(uint32_t));
        uint32_t v;
        std::memcpy(&v, p, sizeof(uint32_t));
        return v;
    }

    // ---- Tile metadata ----
    // Provided unconditionally (real header guards with DATA_FORMATS_DEFINED,
    // but JIT kernels in the regression call these without that guard set).
    uint32_t   get_tile_size()   const { return ::get_tile_size(cb_id_); }
    uint32_t   get_tile_hw()     const { return ::get_tile_hw(cb_id_); }
    DataFormat get_dataformat()  const { return ::get_dataformat(cb_id_); }

    // ---- Scoped lock ----
    // Returns an RAII Lock wrapping the CB region.  In emulation, the CB sync
    // operations (reserve_back / wait_front) already use mutexes, so the lock
    // itself is a no-op; it exists purely for API compatibility.
    [[nodiscard]] auto scoped_lock() {
        return Lock([]() {});
    }

private:
    uint32_t cb_id_;
};

// ---- noc_traits_t<CircularBuffer> ----
// As NOC src: read from the current read pointer.
// As NOC dst (local L1 write): write to the current write pointer.
// As NOC dst (multicast): encode the write pointer into a multicast NOC addr.

template <>
struct noc_traits_t<CircularBuffer> {
    struct src_args_type      { uint32_t offset_bytes{}; };
    struct dst_args_type      { uint32_t offset_bytes{}; };
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };

    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const CircularBuffer& cb, const Noc&,
                               const src_args_type& args) {
        return static_cast<uintptr_t>(cb.get_read_ptr()) + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const CircularBuffer& cb, const Noc& noc,
                               const dst_args_type& args) {
        return static_cast<uintptr_t>(cb.get_write_ptr()) + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static uint64_t dst_addr_mcast(const CircularBuffer& cb, const Noc& noc,
                                    const dst_args_mcast_type& args) {
        uint32_t local_addr = cb.get_write_ptr() + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start,
            args.noc_x_end,   args.noc_y_end,
            local_addr, noc.get_noc_id());
    }
};

// ---- CircularBufferView<AddrSel> + traits ----

template <CircularBuffer::AddrSelector AddrSel>
struct CircularBufferView {
    const CircularBuffer& cb;
    explicit constexpr CircularBufferView(const CircularBuffer& c) : cb(c) {}
};

// Convenience factory — use<AddrSel>(cb)
template <CircularBuffer::AddrSelector AddrSel>
constexpr auto use(const CircularBuffer& cb) {
    return CircularBufferView<AddrSel>(cb);
}

template <CircularBuffer::AddrSelector AddrSel>
class noc_traits_t<CircularBufferView<AddrSel>> {
public:
    struct src_args_type      { uint32_t offset_bytes{}; };
    struct dst_args_type      { uint32_t offset_bytes{}; };
    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };

    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const CircularBufferView<AddrSel>& view, const Noc&,
                               const src_args_type& args) {
        return static_cast<uintptr_t>(get_local_addr(view)) + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const CircularBufferView<AddrSel>& view, const Noc&,
                               const dst_args_type& args) {
        return static_cast<uintptr_t>(get_local_addr(view)) + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static uint64_t dst_addr_mcast(const CircularBufferView<AddrSel>& view, const Noc& noc,
                                    const dst_args_mcast_type& args) {
        uint32_t local_addr = get_local_addr(view) + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start,
            args.noc_x_end,   args.noc_y_end,
            local_addr, noc.get_noc_id());
    }

private:
    static constexpr uint32_t get_local_addr(const CircularBufferView<AddrSel>& view) {
        if constexpr (AddrSel == CircularBuffer::AddrSelector::READ_PTR) {
            return view.cb.get_read_ptr();
        } else {
            return view.cb.get_write_ptr();
        }
    }
};
