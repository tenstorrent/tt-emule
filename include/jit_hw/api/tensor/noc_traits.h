// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation shadow for api/tensor/noc_traits.h.
//
// The real header defines noc_traits_t specialisations for:
//   TensorAccessor<DSpecT>   — provided here (needed by DFB kernel tests)
//   PageView<Accessor>       — omitted (not used in JIT kernels)
//   ShardView<Accessor>      — omitted
//   tensor_accessor::Page    — omitted
//
// Omitting the Page/PageView/ShardView specialisations avoids including the
// real tensor_accessor.h (which chains through many hardware headers), and
// avoids the partial-specialisation parse error that occurs when TensorAccessor
// is not yet known to be a class template.
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
        // up the correct core in __emule_core_map.
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
