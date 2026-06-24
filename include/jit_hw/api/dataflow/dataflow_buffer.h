// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for api/dataflow/dataflow_buffer.h.
//
// The real header has two arch-specific include chains that do not compile on x86:
//
//   WH/BH (!ARCH_QUASAR):
//     internal/circular_buffer_interface.h  — redefines CBInterface (conflict with ours)
//     api/debug/waypoint.h → hostdev/dev_msgs.h  — forbidden in JIT context
//     internal/tt-1xx/dataflow_buffer.inl → stream_io_map.h  — hardware register map
//
//   Quasar (ARCH_QUASAR):
//     internal/tt-2xx/dataflow_buffer/dataflow_buffer_init.h
//       → dataflow_buffer_isr.h → llk_intf_api.hpp
//       → asm volatile with RISC-V register names (x10/x11/x12) — invalid on x86
//
// This self-contained stub follows the same split:
//   !ARCH_QUASAR — DataflowBuffer wraps CB API (cb_reserve_back, get_write_ptr, …)
//   ARCH_QUASAR  — DataflowBuffer wraps DFB tile-counter API (dfb_reserve_back, …)
//                  identical functional logic to jit_hw/experimental/dataflow_buffer.h
//                  but without the namespace experimental {} wrapper.

#include "jit_hw/api/dataflow/noc.h"  // Noc, noc_traits_t, DataflowBufferArgs
#include "api/lock.h"                  // RAII Lock — reachable via tt_metal/hw/inc/
#include "jit_hw/internal/emule_thread_ctx.h"  // __emule_self->dfbs (Quasar path)

#ifndef ARCH_QUASAR
#include "jit_hw/api/cb_api.h"                  // get_tile_size/hw/dataformat
#include "jit_hw/api/dataflow/dataflow_api.h"   // cb_reserve_back, get_write_ptr, …
#else
#include "jit_hw/api/dfb_api.h"                 // dfb_reserve_back, dfb_get_write_ptr, …
#endif

// ---- DFBAccessor --------------------------------------------------------
// Opaque compile-time binding handle; auto-generated into kernel_bindings_generated.h.
struct DFBAccessor {
    explicit constexpr DFBAccessor(uint16_t id) noexcept : id(id) {}
    // Implicitly convertible to uint32_t so it can be passed to LLK compute APIs.
    constexpr operator uint32_t() const noexcept { return id; }
    uint16_t id;
};

// ---- DataflowBuffer -----------------------------------------------------

class DataflowBuffer {
public:
    DataflowBuffer(DFBAccessor accessor) : dfb_id_(accessor.id) {}
    explicit DataflowBuffer(uint16_t id) : dfb_id_(id) {}

    uint16_t get_id() const { return dfb_id_; }

#ifndef ARCH_QUASAR
    // ----------------------------------------------------------------
    // WH/BH path: thin wrapper around the existing CB emulation.
    // The real internal/tt-1xx/dataflow_buffer.inl does the same
    // (calls cb_reserve_back, cb_push_back, etc.) but chains through
    // hardware headers we cannot include in JIT context.
    // ----------------------------------------------------------------

    uint32_t get_entry_size()  const { return ::get_tile_size(dfb_id_); }
    uint32_t get_stride_size() const { return ::get_tile_size(dfb_id_); }

    void reserve_back(uint16_t n) { cb_reserve_back(dfb_id_, n); }
    void push_back(uint16_t n)    { cb_push_back(dfb_id_, n); }
    void wait_front(uint16_t n)   { cb_wait_front(dfb_id_, n); }
    void pop_front(uint16_t n)    { cb_pop_front(dfb_id_, n); }

    // Non-blocking availability checks.  In emulation, blocking is handled
    // by cb_reserve_back / cb_wait_front; these always report "available".
    bool pages_reservable_at_back(int32_t /*n*/) const { return true; }
    bool pages_available_at_front(int32_t /*n*/) const { return true; }

    uint32_t get_write_ptr() const { return ::get_write_ptr(dfb_id_); }
    uint32_t get_read_ptr()  const { return ::get_read_ptr(dfb_id_); }

    // Drain barrier: no-op on WH (cb_wait_front / cb_pop_front handle drain).
    void finish() {}

    // Tile metadata — provided unconditionally (kernels may call without
    // DATA_FORMATS_DEFINED guard, matching the behaviour of our CircularBuffer stub).
    uint32_t   get_tile_size()   const { return ::get_tile_size(dfb_id_); }
    uint32_t   get_tile_hw()     const { return ::get_tile_hw(dfb_id_); }
    DataFormat get_dataformat()  const { return ::get_dataformat(dfb_id_); }

#else
    // ----------------------------------------------------------------
    // Quasar path: backed by DFB tile-counter API from dfb_api.h.
    // Mirrors jit_hw/experimental/dataflow_buffer.h but without namespace.
    // ----------------------------------------------------------------

    uint32_t get_entry_size()  const { return dfb_get_entry_size(dfb_id_); }
    uint32_t get_stride_size() const { return __emule_self->dfbs[dfb_id_].stride_size; }

    void reserve_back(uint16_t n) { dfb_reserve_back(dfb_id_, n); }
    void push_back(uint16_t n)    { dfb_push_back(dfb_id_, n); }
    void wait_front(uint16_t n)   { dfb_wait_front(dfb_id_, n); }
    void pop_front(uint16_t n)    { dfb_pop_front(dfb_id_, n); }

    uint32_t get_write_ptr() const { return dfb_get_write_ptr(dfb_id_); }
    uint32_t get_read_ptr()  const { return dfb_get_read_ptr(dfb_id_); }

    // Drain all tile counters (producer calls after pushing all entries).
    void finish() { dfb_finish(dfb_id_); }

#endif  // ARCH_QUASAR

    // ---- Common (both arches) ----------------------------------------

    void write_barrier(const Noc& noc) const { noc.async_write_barrier(); }

    [[nodiscard]] auto scoped_lock() {
        return Lock([]() {});
    }

    friend class Noc;  // grants access for implicit-sync async_read/write overloads

private:
    uint16_t dfb_id_;
};

// ---- noc_traits_t<DataflowBuffer> ---------------------------------------
// src_addr / dst_addr resolve via get_read_ptr / get_write_ptr, which are
// already arch-dispatched above.

template <>
struct noc_traits_t<DataflowBuffer> {
    using src_args_type = DataflowBufferArgs;
    using dst_args_type = DataflowBufferArgs;

    struct dst_args_mcast_type {
        uint32_t noc_x_start{};
        uint32_t noc_y_start{};
        uint32_t noc_x_end{};
        uint32_t noc_y_end{};
        uint32_t offset_bytes{};
    };

    template <Noc::AddressType address_type>
    static auto src_addr(const DataflowBuffer& src, const Noc&,
                         const src_args_type& args) {
        static_assert(address_type == Noc::AddressType::LOCAL_L1,
                      "DataflowBuffer without mcast range can only be used as L1 source");
        return src.get_read_ptr() + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static auto dst_addr(const DataflowBuffer& dst, const Noc&,
                         const dst_args_type& args) {
        static_assert(address_type == Noc::AddressType::LOCAL_L1,
                      "DataflowBuffer without mcast range can only be used as L1 destination");
        return dst.get_write_ptr() + args.offset_bytes;
    }

    template <Noc::AddressType address_type>
    static auto dst_addr_mcast(const DataflowBuffer& dst, const Noc& noc,
                               const dst_args_mcast_type& args) {
        static_assert(address_type == Noc::AddressType::NOC,
                      "DataflowBuffer with mcast range cannot be used as L1 destination");
        auto local_addr = dst.get_write_ptr() + args.offset_bytes;
        return ::get_noc_multicast_addr(
            args.noc_x_start, args.noc_y_start,
            args.noc_x_end,   args.noc_y_end,
            local_addr, noc.get_noc_id());
    }
};

// ---- Noc implicit-sync overloads for DataflowBuffer ---------------------
// Declared in jit_hw/api/dataflow/noc.h; defined here after DataflowBuffer
// is complete (same pattern as the real api/dataflow/dataflow_buffer.h).
//
// In the real header these are gated on #ifdef ARCH_QUASAR.  We define them
// unconditionally here: on WH the `if constexpr (implicit_sync)` branch that
// calls them is pruned at compile time, so they are never instantiated.

template <NocOptions opts, typename Src>
inline std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_read(const Src& src,
                DataflowBuffer& dst,
                const typename noc_traits_t<Src>::src_args_type& src_args,
                const DataflowBufferArgs& dst_args) const {
    const uint32_t size = dst.get_entry_size();
    dst.reserve_back(1);
    uint8_t* src_ptr = to_host_ptr<AddressType::NOC>(
        noc_traits_t<Src>::template src_addr<AddressType::NOC>(src, *this, src_args));
    uint8_t* dst_ptr = __emule_local_l1_to_ptr(dst.get_write_ptr() + dst_args.offset_bytes);
    if (src_ptr && dst_ptr && size)
        std::memcpy(dst_ptr, src_ptr, size);
    dst.push_back(1);
}

template <NocOptions opts, typename Dst>
inline std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
Noc::async_write(DataflowBuffer& src,
                 const Dst& dst,
                 const DataflowBufferArgs& src_args,
                 const typename noc_traits_t<Dst>::dst_args_type& dst_args) const {
    const uint32_t size = src.get_entry_size();
    src.wait_front(1);
    uint8_t* src_ptr = __emule_local_l1_to_ptr(src.get_read_ptr() + src_args.offset_bytes);
    uint8_t* dst_ptr = to_host_ptr<AddressType::NOC>(
        noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args));
    if (src_ptr && dst_ptr && size)
        std::memcpy(dst_ptr, src_ptr, size);
    src.pop_front(1);
}
