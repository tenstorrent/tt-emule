#pragma once
// Emulation stub for experimental noc_traits_t<TensorAccessor>.
// Shadows the real tt_metal/hw/inc/experimental/tensor.h which references
// templated TensorAccessor<DSpecT>, PageView, ShardView, etc. that are not
// available in the jit_hw context.
//
// In emulation, TensorAccessor::get_noc_addr() returns a NOC-encoded address
// (with core coordinates in upper bits). We must resolve it via
// __emule_resolve_noc_addr() to get a host-accessible pointer.

#include "jit_hw/experimental/noc.h"
#include "jit_hw/api/tensor/tensor_accessor.h"

extern uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);

namespace experimental {

template <>
struct noc_traits_t<TensorAccessor> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };

    template <Noc::AddressType AT>
    static uintptr_t src_addr(const TensorAccessor& src, const Noc& noc, const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }

    template <Noc::AddressType AT>
    static uintptr_t dst_addr(const TensorAccessor& dst, const Noc& noc, const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        return reinterpret_cast<uintptr_t>(__emule_resolve_noc_addr(noc_addr));
    }
};

}  // namespace experimental
