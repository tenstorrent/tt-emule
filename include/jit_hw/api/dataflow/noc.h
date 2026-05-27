// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for api/dataflow/noc.h.
//
// Self-contained replacement: provides class Noc with the same public API as
// tt_metal/hw/inc/api/dataflow/noc.h but implemented with direct memcpy and
// host-pointer resolution instead of hardware NOC transactions.
//
// Relationship to jit_hw/experimental/noc.h:
//   The real tt-metal commit a67eb83be3c ("Moving new device apis out of
//   experimental namespace") is a pure namespace promotion — the only diff is
//   removal of 'namespace experimental {}'. This file extends that promotion
//   with the wider template/enum surface that the new class Noc exposes.

#include <cstdint>
#include <cstring>
#include <type_traits>
#include "jit_hw/api/dataflow/dataflow_api.h"   // bridge functions, my_x/y, noc_index,
                                                  // NOC_MAX_BURST_SIZE, NOC_UNICAST_ADDR_X/Y,
                                                  // barrier stubs, __emule_dw_write_be

// ---- Forward declarations (mirror real noc.h) ----
struct MulticastEndpoint;
class DataflowBuffer;

// Concrete arg struct for the DFB-specific Noc overloads.
struct DataflowBufferArgs {
    uint32_t offset_bytes{};
};

// Base noc_traits_t — specialised in endpoints.h per endpoint type.
template <typename T>
struct noc_traits_t {
    static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type");
};

// ---- class Noc ----

class Noc {
public:
    // ----- Enums (full set matching api/dataflow/noc.h) -----

    enum class AddressType { NOC, LOCAL_L1 };
    enum class TxnIdMode  { ENABLED, DISABLED };
    enum class ResponseMode { NON_POSTED, POSTED };
    enum class BarrierMode  { TXN_ID, FULL };
    enum class McastMode    { INCLUDE_SRC, EXCLUDE_SRC };
    enum class VcSelection  { DEFAULT, CUSTOM };

    static constexpr uint32_t INVALID_TXN_ID = 0xFFFFFFFF;

private:
    template <typename T> using src_args_t = typename noc_traits_t<T>::src_args_type;
    template <typename T> using dst_args_t = typename noc_traits_t<T>::dst_args_type;
    template <typename T> using dst_args_mcast_t = typename noc_traits_t<T>::dst_args_mcast_type;

    // Convert an address value (returned by noc_traits_t::src_addr / dst_addr) to a
    // real host uint8_t* that can be passed to std::memcpy.
    //
    // Design contract: noc_traits_t specialisations ALWAYS return host pointers
    // as their uintptr_t return value.  For NOC-type addresses this means the trait
    // already resolved the NOC address (e.g. via __emule_dram_ptr or
    // __emule_resolve_noc_addr) before returning — we simply cast here.
    //
    // LOCAL_L1 addresses are raw CB/DFB firmware pointers (uint32_t) or truncated
    // host pointers that must be run through __emule_local_l1_to_ptr, which handles
    // both "firmware offset from L1 base" and "truncated 32-bit host pointer" cases.
    template <AddressType addr_type>
    static uint8_t* to_host_ptr(uintptr_t addr) {
        if constexpr (addr_type == AddressType::NOC) {
            // Trait already resolved to a host pointer — cast directly.
            return reinterpret_cast<uint8_t*>(addr);
        } else {
            return __emule_local_l1_to_ptr(static_cast<uint32_t>(addr));
        }
    }

public:
    // ----- Constructors -----

    Noc() : noc_id_(noc_index) {}
    explicit Noc(uint8_t noc_id) : noc_id_(noc_id) {}

    uint8_t get_noc_id() const { return noc_id_; }

    // ----- Local-core queries -----

    bool is_local_bank(uint32_t virtual_x, uint32_t virtual_y) const {
        return virtual_x == my_x[noc_id_] && virtual_y == my_y[noc_id_];
    }

    bool is_local_addr(uint64_t noc_addr) const {
        return is_local_bank(
            static_cast<uint32_t>(NOC_UNICAST_ADDR_X(noc_addr)),
            static_cast<uint32_t>(NOC_UNICAST_ADDR_Y(noc_addr)));
    }

    // ----- async_read -----
    // In emulation: resolve both addresses via traits, then memcpy.
    // Template params (txn_id_mode, max_page_size, enable_noc_tracing) are
    // accepted for API compatibility but have no effect.

    template <
        TxnIdMode txn_id_mode = TxnIdMode::DISABLED,
        uint32_t  max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool      enable_noc_tracing = true,
        typename Src, typename Dst>
    void async_read(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint32_t vc  = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        uint8_t* src_ptr = to_host_ptr<AddressType::NOC>(
            noc_traits_t<Src>::template src_addr<AddressType::NOC>(src, *this, src_args));
        uint8_t* dst_ptr = to_host_ptr<AddressType::LOCAL_L1>(
            noc_traits_t<Dst>::template dst_addr<AddressType::LOCAL_L1>(dst, *this, dst_args));
        if (src_ptr && dst_ptr) {
            std::memcpy(dst_ptr, src_ptr, size_bytes);
        }
    }

    // ----- set_async_read_state + async_read_with_state -----
    // Mirrors upstream contract: cache size for subsequent async_read_with_state.

    template <
        VcSelection vc_selection = VcSelection::DEFAULT,
        uint32_t    max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src>
    void set_async_read_state(
        const Src& /*src*/,
        uint32_t size_bytes,
        const src_args_t<Src>& /*src_args*/,
        uint8_t /*vc*/ = 0) const {
        cached_size_ = size_bytes;
    }

    template <
        VcSelection vc_selection = VcSelection::DEFAULT,
        uint32_t    max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src, typename Dst>
    void async_read_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint8_t /*vc*/ = 0) const {
        // When max_page_size fits in one packet, size comes from cached state.
        constexpr bool fits_in_one_packet = max_page_size <= NOC_MAX_BURST_SIZE;
        const uint32_t bytes = fits_in_one_packet ? cached_size_ : size_bytes;
        uint8_t* src_ptr = to_host_ptr<AddressType::NOC>(
            noc_traits_t<Src>::template src_addr<AddressType::NOC>(src, *this, src_args));
        uint8_t* dst_ptr = to_host_ptr<AddressType::LOCAL_L1>(
            noc_traits_t<Dst>::template dst_addr<AddressType::LOCAL_L1>(dst, *this, dst_args));
        if (src_ptr && dst_ptr && bytes) {
            std::memcpy(dst_ptr, src_ptr, bytes);
        }
    }

    // ----- async_write -----

    template <
        TxnIdMode    txn_id_mode   = TxnIdMode::DISABLED,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool         enable_noc_tracing = true,
        typename Src, typename Dst>
    void async_write(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint32_t vc   = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        uint8_t* src_ptr = to_host_ptr<AddressType::LOCAL_L1>(
            noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args));
        uint8_t* dst_ptr = to_host_ptr<AddressType::NOC>(
            noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args));
        if (src_ptr && dst_ptr) {
            std::memcpy(dst_ptr, src_ptr, size_bytes);
        }
    }

    // ----- set_async_write_state + async_write_with_state -----

    template <
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Dst>
    void set_async_write_state(
        const Dst& dst,
        uint32_t size_bytes,
        const dst_args_t<Dst>& dst_args,
        uint8_t /*vc*/ = NOC_UNICAST_WRITE_VC) const {
        cached_size_      = size_bytes;
        cached_write_dst_ = reinterpret_cast<uintptr_t>(to_host_ptr<AddressType::NOC>(
            noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args)));
    }

    template <
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src, typename Dst>
    void async_write_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        uint8_t /*vc*/ = NOC_UNICAST_WRITE_VC) const {
        constexpr bool fits_in_one_packet = max_page_size <= NOC_MAX_BURST_SIZE;
        const uint32_t bytes = fits_in_one_packet ? cached_size_ : size_bytes;
        uint8_t* src_ptr = to_host_ptr<AddressType::LOCAL_L1>(
            noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args));
        // Prefer the pre-resolved destination from set_async_write_state when available;
        // fall back to re-resolving from the dst argument for safety.
        // cached_write_dst_ is already a host pointer (resolved in set_async_write_state).
        uint8_t* dst_ptr = cached_write_dst_
            ? reinterpret_cast<uint8_t*>(cached_write_dst_)
            : to_host_ptr<AddressType::NOC>(
                noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args));
        if (src_ptr && dst_ptr && bytes) {
            std::memcpy(dst_ptr, src_ptr, bytes);
        }
    }

    // ----- async_write_multicast -----
    // Delegates to __emule_multicast_write which iterates the rectangle.
    // McastMode::INCLUDE_SRC is handled by the multicast write loop visiting
    // all cores including the sender's (same host memory, no special case needed).

    template <
        McastMode    mcast_mode    = McastMode::EXCLUDE_SRC,
        TxnIdMode    txn_id_mode   = TxnIdMode::DISABLED,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool         enable_noc_tracing = true,
        typename Src, typename Dst>
    void async_write_multicast(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        uint32_t num_dsts,
        const src_args_t<Src>& src_args,
        const dst_args_mcast_t<Dst>& dst_args,
        bool linked = false,
        uint32_t trid = INVALID_TXN_ID) const {
        uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args);
        auto mcast_noc_addr = noc_traits_t<Dst>::template dst_addr_mcast<AddressType::NOC>(dst, *this, dst_args);
        if (s) {
            __emule_multicast_write(static_cast<uint64_t>(mcast_noc_addr),
                                    reinterpret_cast<const uint8_t*>(s), size_bytes);
        }
    }

    // ----- inline_dw_write -----
    // Resolves the destination NOC address via traits, then writes a 32-bit
    // value with byte-enable masking via __emule_dw_write_be.

    template <
        TxnIdMode    txn_id_mode   = TxnIdMode::DISABLED,
        InlineWriteDst dst_type    = InlineWriteDst::DEFAULT,
        ResponseMode response_mode = ResponseMode::NON_POSTED,
        typename Dst>
    void inline_dw_write(
        const Dst& dst,
        uint32_t val,
        const dst_args_t<Dst>& dst_args,
        uint8_t  be  = 0xF,
        uint32_t vc  = NOC_UNICAST_WRITE_VC,
        uint32_t trid = INVALID_TXN_ID) const {
        static_assert(txn_id_mode == TxnIdMode::DISABLED);
        uint8_t* ptr = to_host_ptr<AddressType::NOC>(
            noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args));
        if (ptr) {
            __emule_dw_write_be(ptr, val, be);
        }
    }

    // ----- Barriers -----
    // All are no-ops in emulation; template params accepted for API compatibility.

    template <BarrierMode barrier_type = BarrierMode::FULL>
    void async_read_barrier(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (barrier_type == BarrierMode::TXN_ID) {
            noc_async_read_barrier_with_trid(trid, noc_id_);
        } else {
            noc_async_read_barrier(noc_id_);
        }
    }

    template <BarrierMode barrier_type = BarrierMode::FULL>
    void async_write_barrier(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (barrier_type == BarrierMode::TXN_ID) {
            noc_async_write_barrier_with_trid(trid, noc_id_);
        } else {
            noc_async_write_barrier(noc_id_);
        }
    }

    template <ResponseMode response_mode = ResponseMode::NON_POSTED,
              BarrierMode  barrier_type  = BarrierMode::FULL>
    void async_writes_flushed(uint32_t trid = INVALID_TXN_ID) const {
        if constexpr (response_mode == ResponseMode::POSTED) {
            noc_async_posted_writes_flushed(noc_id_);
        } else {
            if constexpr (barrier_type == BarrierMode::TXN_ID) {
                noc_async_write_flushed_with_trid(trid, noc_id_);
            } else {
                noc_async_writes_flushed(noc_id_);
            }
        }
    }

    void async_atomic_barrier() const { noc_async_atomic_barrier(noc_id_); }
    void async_full_barrier()   const { noc_async_full_barrier(noc_id_); }

    // ----- DFB implicit-sync overloads -----
    // Declared here; defined out-of-line in api/dataflow/dataflow_buffer.h
    // which includes this header.

    template <TxnIdMode txn_id_mode, typename Src>
    std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED>
    async_read(
        const Src& src,
        DataflowBuffer& dst,
        const src_args_t<Src>& src_args,
        const DataflowBufferArgs& dst_args = {}) const;

    template <TxnIdMode txn_id_mode, typename Dst>
    std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED>
    async_write(
        DataflowBuffer& src,
        const Dst& dst,
        const DataflowBufferArgs& src_args,
        const dst_args_t<Dst>& dst_args) const;

private:
    uint8_t  noc_id_;
    mutable uint32_t  cached_size_      = 0;  // for set_async_read_state / set_async_write_state
    mutable uintptr_t cached_write_dst_ = 0;  // for set_async_write_state
};
