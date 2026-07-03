// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation shadow for api/tensor/noc_traits.h.
//
// The real header defines noc_traits_t specialisations for:
//   TensorAccessor<DSpecT>           — provided here
//   AbstractTensorAccessorWrapper    — provided here (type-erased accessor, #45671)
//   tensor_accessor::Page            — provided here
//   ShardView<Accessor>              — provided here
//   PageView<Accessor>               — omitted (no JIT kernel uses it)
//
// Address resolution contract:
//   noc_traits_t in emulation always returns HOST POINTERS as uintptr_t —
//   not raw NOC addresses.  For TensorAccessor (DRAM-backed), we extract the
//   firmware DRAM offset from the lower 36 bits of the NOC address and pass it
//   to __emule_dram_ptr() which maps it to the backing host memory.

#include "jit_hw/api/dataflow/noc.h"          // noc_traits_t primary template, Noc
#include "api/tensor/tensor_accessor.h" // TensorAccessor<DSpecT>
#include "jit_hw/api/dataflow/dataflow_api.h"  // __emule_dram_ptr

// NOC_ADDR_LOCAL_BITS is the bit-width of the local address portion of a NOC
// address.  Defined in noc_parameters.h on hardware; we use the wormhole value.
#ifndef NOC_ADDR_LOCAL_BITS
#define NOC_ADDR_LOCAL_BITS 36u
#endif

// ---- noc_traits_t<TensorAccessor<DSpecT>> --------------------------------
// As NOC src: read from page_id-th page of the tensor (get_noc_addr).
// As NOC dst: write to page_id-th page of the tensor.
//
// In emulation the NOC address from InterleavedAddrGen::get_noc_addr has the
// firmware DRAM offset in the lower NOC_ADDR_LOCAL_BITS bits.  We pass this
// offset to __emule_dram_ptr() to obtain the real host pointer.
//
// Both LOCAL_L1 and NOC address_type specialisations return the same host
// pointer — the distinction only affects real hardware routing.

template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };

private:
    // Resolve a firmware NOC address from TensorAccessor::get_noc_addr() to a
    // host pointer into the emulation's DRAM backing memory.
    //
    // The NOC address encodes the DRAM core's (x,y) coordinates in the upper bits
    // and the local byte offset in the lower NOC_ADDR_LOCAL_BITS bits.  We use
    // __emule_resolve_noc_addr which looks up the correct DRAM core from the
    // NOC coordinate map — essential for multi-bank architectures like Quasar
    // where different DRAM banks are at distinct NOC (x,y) coordinates (e.g.
    // bank 0 at (0,0), bank 1 at (0,2)).  Using the old __emule_dram_ptr which
    // always references Core(0,0) would silently mis-route bank 1 reads/writes.
    static uint8_t* dram_host_ptr(const TensorAccessor<DSpecT>& acc,
                                   uint32_t page_id, uint32_t offset_bytes,
                                   uint8_t noc_id) {
        uint64_t noc_addr = acc.get_noc_addr(page_id, offset_bytes, noc_id);
        // __emule_resolve_noc_addr decodes the NOC XY from the upper bits and
        // the local offset from the lower NOC_ADDR_LOCAL_BITS bits, then looks
        // up the correct core in __emule_self->core_map.
        return __emule_resolve_noc_addr(noc_addr);
    }

public:
    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const TensorAccessor<DSpecT>& src, const Noc& noc,
                              const src_args_type& args) {
        return reinterpret_cast<uintptr_t>(
            dram_host_ptr(src, args.page_id, args.offset_bytes, noc.get_noc_id()));
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const TensorAccessor<DSpecT>& dst, const Noc& noc,
                              const dst_args_type& args) {
        return reinterpret_cast<uintptr_t>(
            dram_host_ptr(dst, args.page_id, args.offset_bytes, noc.get_noc_id()));
    }
};

// ---- noc_traits_t<tensor_accessor::Page> ---------------------------------
// Iterator-yielded Page carries a pre-baked uint64_t NOC address. To use it
// as a NOC src/dst, resolve `Page::noc_addr() + offset_bytes` through the
// same __emule_resolve_noc_addr that the TensorAccessor specialization uses.

template <>
struct noc_traits_t<tensor_accessor::Page> {
    struct src_args_type { uint32_t offset_bytes = 0; };
    struct dst_args_type { uint32_t offset_bytes = 0; };

    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const tensor_accessor::Page& src, const Noc&,
                              const src_args_type& args) {
        uint64_t noc_addr = src.noc_addr() + args.offset_bytes;
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const tensor_accessor::Page& dst, const Noc&,
                              const dst_args_type& args) {
        uint64_t noc_addr = dst.noc_addr() + args.offset_bytes;
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};

// ---- noc_traits_t<ShardView<Accessor>> -----------------------------------
// Accessor variant that addresses by shard_id rather than page_id. Delegates
// to acc.get_shard_noc_addr() — which upstream's TensorAccessor provides
// for free post-shim-removal.

template <typename Accessor>
struct noc_traits_t<ShardView<Accessor>> {
    struct src_args_type {
        uint32_t shard_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t shard_id{};
        uint32_t offset_bytes = 0;
    };

    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const ShardView<Accessor>& src, const Noc& noc,
                              const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.shard_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const ShardView<Accessor>& dst, const Noc& noc,
                              const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.shard_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};

// ---- noc_traits_t<AbstractTensorAccessorWrapper> -------------------------
// Type-erased accessor (#45671): make_abstract_tensor_accessor_wrappers wraps a
// tuple of TensorAccessors into an array the concat readers iterate over. Page-
// addressed via get_noc_addr — same host-pointer resolution as TensorAccessor.

template <>
struct noc_traits_t<AbstractTensorAccessorWrapper> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };

    template <Noc::AddressType address_type>
    static uintptr_t src_addr(const AbstractTensorAccessorWrapper& src, const Noc& noc,
                              const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType address_type>
    static uintptr_t dst_addr(const AbstractTensorAccessorWrapper& dst, const Noc& noc,
                              const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};
