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
// Option surface mirrors tt-metal #45509 ("Replace diff noc txn template args
// with one NocOptions enum"): the per-method nested enums (TxnIdMode,
// VcSelection, BarrierMode, McastMode, ResponseMode) were collapsed into the
// single file-scope NocOptions flag enum + NocOptVals runtime struct.

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

// ---- NocOptions (file scope; mirrors api/dataflow/noc.h #45509) ----
// Bit-flag combination chosen at a template call site, e.g.
//   noc.async_write<NocOptions::TXN_ID | NocOptions::CUSTOM_VC>(...)
enum class NocOptions : uint32_t {
    DEFAULT        = 0,
    TXN_ID         = 1u << 0,
    POSTED         = 1u << 1,
    CUSTOM_VC      = 1u << 2,
    MCAST_INCL_SRC = 1u << 3,
    INLINE_L1      = 1u << 4,
    INLINE_REG     = 1u << 5,
};

constexpr NocOptions operator|(NocOptions a, NocOptions b) noexcept {
    return static_cast<NocOptions>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

constexpr bool has_flag(NocOptions opts, NocOptions flag) noexcept {
    return (static_cast<uint32_t>(opts) & static_cast<uint32_t>(flag)) != 0;
}

// Runtime values for the optional flags (vc when CUSTOM_VC, trid when TXN_ID).
struct NocOptVals {
    uint32_t vc   = NOC_UNICAST_WRITE_VC;
    uint32_t trid = 0;
};

// ---- class Noc ----

class Noc {
public:
    enum class AddressType { NOC, LOCAL_L1 };

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
    // Template params (opts, max_page_size, enable_noc_tracing) and noc_opts are
    // accepted for API compatibility but have no effect (no real VC / trid).

    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t  max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool      enable_noc_tracing = true,
        typename Src, typename Dst>
    void async_read(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& /*noc_opts*/ = {}) const {
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
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t    max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src>
    void set_async_read_state(
        const Src& /*src*/,
        uint32_t size_bytes,
        const src_args_t<Src>& /*src_args*/,
        const NocOptVals& /*noc_opts*/ = {}) const {
        cached_size_ = size_bytes;
    }

    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t    max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src, typename Dst>
    void async_read_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& /*noc_opts*/ = {}) const {
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
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        bool         enable_noc_tracing = true,
        typename Src, typename Dst>
    void async_write(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& /*noc_opts*/ = {}) const {
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
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Dst>
    void set_async_write_state(
        const Dst& dst,
        uint32_t size_bytes,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& /*noc_opts*/ = {}) const {
        cached_size_      = size_bytes;
        cached_write_dst_ = reinterpret_cast<uintptr_t>(to_host_ptr<AddressType::NOC>(
            noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args)));
    }

    template <
        NocOptions opts = NocOptions::DEFAULT,
        uint32_t     max_page_size = NOC_MAX_BURST_SIZE + 1,
        typename Src, typename Dst>
    void async_write_with_state(
        const Src& src,
        const Dst& dst,
        uint32_t size_bytes,
        const src_args_t<Src>& src_args,
        const dst_args_t<Dst>& dst_args,
        const NocOptVals& /*noc_opts*/ = {}) const {
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
    // NocOptions::MCAST_INCL_SRC selects silicon's NOC_CMD_BRCST_SRC_INCLUDE bit:
    // when set, the sender's own coordinates inside the rectangle receive the
    // packet; when clear, the sender NIU drops at itself. This MUST be
    // forwarded to __emule_multicast_write — leaving it to register garbage
    // produces non-deterministic data delivery for dual-role workers.

    template <
        NocOptions   opts = NocOptions::DEFAULT,
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
        bool linked = false) const {
        uintptr_t s = noc_traits_t<Src>::template src_addr<AddressType::LOCAL_L1>(src, *this, src_args);
        auto mcast_noc_addr = noc_traits_t<Dst>::template dst_addr_mcast<AddressType::NOC>(dst, *this, dst_args);
        if (s) {
            __emule_multicast_write(static_cast<uint64_t>(mcast_noc_addr),
                                    reinterpret_cast<const uint8_t*>(s), size_bytes,
                                    has_flag(opts, NocOptions::MCAST_INCL_SRC));
        }
    }

    // ----- inline_dw_write -----
    // Resolves the destination NOC address via traits, then writes a 32-bit
    // value with byte-enable masking via __emule_dw_write_be.

    template <
        NocOptions opts = NocOptions::DEFAULT,
        typename Dst>
    void inline_dw_write(
        const Dst& dst,
        uint32_t val,
        const dst_args_t<Dst>& dst_args,
        uint8_t  be  = 0xF,
        const NocOptVals& /*noc_opts*/ = {}) const {
        uint8_t* ptr = to_host_ptr<AddressType::NOC>(
            noc_traits_t<Dst>::template dst_addr<AddressType::NOC>(dst, *this, dst_args));
        if (ptr) {
            __emule_dw_write_be(ptr, val, be);
        }
    }

    // ----- Barriers -----
    // All are no-ops in emulation; template params accepted for API compatibility.

    template <NocOptions opts = NocOptions::DEFAULT>
    void async_read_barrier(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_read_barrier_with_trid(noc_opts.trid, noc_id_);
        } else {
            noc_async_read_barrier(noc_id_);
        }
    }

    template <NocOptions opts = NocOptions::DEFAULT>
    void async_write_barrier(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_write_barrier_with_trid(noc_opts.trid, noc_id_);
        } else {
            noc_async_write_barrier(noc_id_);
        }
    }

    template <NocOptions opts = NocOptions::DEFAULT>
    void async_writes_flushed(const NocOptVals& noc_opts = {}) const {
        if constexpr (has_flag(opts, NocOptions::POSTED)) {
            noc_async_posted_writes_flushed(noc_id_);
        } else if constexpr (has_flag(opts, NocOptions::TXN_ID)) {
            noc_async_write_flushed_with_trid(noc_opts.trid, noc_id_);
        } else {
            noc_async_writes_flushed(noc_id_);
        }
    }

    void async_atomic_barrier() const { noc_async_atomic_barrier(noc_id_); }
    void async_full_barrier()   const { noc_async_full_barrier(noc_id_); }

    // ----- DFB implicit-sync overloads -----
    // Declared here; defined out-of-line in api/dataflow/dataflow_buffer.h
    // which includes this header.

    template <NocOptions opts, typename Src>
    std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
    async_read(
        const Src& src,
        DataflowBuffer& dst,
        const src_args_t<Src>& src_args,
        const DataflowBufferArgs& dst_args = {}) const;

    template <NocOptions opts, typename Dst>
    std::enable_if_t<has_flag(opts, NocOptions::TXN_ID)>
    async_write(
        DataflowBuffer& src,
        const Dst& dst,
        const DataflowBufferArgs& src_args,
        const dst_args_t<Dst>& dst_args) const;

private:
    uint8_t  noc_id_;
    // NOC read/write state mirrors per-RISC hardware cmd-buf registers, NOT
    // per-Noc-object state. Kernel helpers (e.g. experimental_device_api.hpp's
    // set_read_state / read_with_state, pool clear_out_tiles / zero_out_page)
    // take `Noc` BY VALUE, so set_async_read_state and async_read_with_state run
    // on SEPARATE Noc copies — the cached size must persist across copies or the
    // state-read copies 0 bytes. thread_local = per-RISC, matching silicon.
    inline static thread_local uint32_t  cached_size_      = 0;
    inline static thread_local uintptr_t cached_write_dst_ = 0;
};
