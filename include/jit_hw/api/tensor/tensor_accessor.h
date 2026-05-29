// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT TensorAccessor stub.
//
// The real api/tensor/tensor_accessor.h defines:
//   template <typename DSpecT> struct TensorAccessor { ... };
//
// Our stub matches that signature so partial specialisations in
// api/tensor/noc_traits.h (shadowed by jit_hw/api/tensor/noc_traits.h) compile.
// Implementation is simplified: we only store bank_base_address + page_size and
// delegate to InterleavedAddrGen for NOC address computation — sufficient for
// JIT data-movement kernels.

#include <cstdint>
#include <cstddef>
#include <tuple>
#include <utility>
#include <array>

// Real TensorAccessorArgs (pure C++17 templates, compiles without KERNEL_BUILD).
// Dependency chain: tensor_accessor_args.h → arg_config.hpp → flags.hpp, const.h
#include "api/tensor/tensor_accessor_args.h"

// Full definition of InterleavedAddrGen needed for TensorAccessor::get_noc_addr.
#include "internal/dataflow/dataflow_api_addrgen.h"

// ---- TensorAccessorBindingToken ----------------------------------------
// Metal 2.0 compile-time binding token.  In the real build this is in
// tensor_accessor.h under namespace tensor_accessor; the actual value is
// emitted by host codegen into kernel_bindings_generated.h.
// In emulation the JIT wrapper emits "namespace ta { constexpr name_t name{}; }"
// where name_t is this type.

namespace tensor_accessor {
    template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
    struct TensorAccessorBindingToken {};
}

// ---- TensorAccessor -----------------------------------------------------
// Template parameter DSpecT mirrors the real class template parameter.
// Default to void so plain 'TensorAccessor' without a template arg still names
// the primary template (used in partial specialisations in noc_traits.h).

template <typename DSpecT = void>
struct TensorAccessor {
    size_t   bank_base_address = 0;
    uint32_t page_size         = 0;

    TensorAccessor() = default;

    // Metal 2.0 binding-token constructor.
    // Bank base address comes from CRTA at ADDR_CRTA_OFFSET/sizeof(uint32_t).
    // Page size is the compile-time arg at CTA_OFFSET + 1 (see tensor_accessor_args.h:43).
    template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
    TensorAccessor(tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>) noexcept :
        bank_base_address(static_cast<size_t>(
            get_common_arg_val<uint32_t>(ADDR_CRTA_OFFSET / sizeof(uint32_t)))),
        page_size(get_compile_time_arg_val(CTA_OFFSET + 1)) {}

    // Legacy TensorAccessorArgs constructor (Metal 1.x compatibility).
    template <std::size_t CTA, std::size_t CRTA>
    TensorAccessor(const TensorAccessorArgs<CTA, CRTA>& /*args*/,
                   size_t   addr,
                   uint32_t ps = TensorAccessorArgs<CTA, CRTA>::AlignedPageSize) noexcept :
        bank_base_address(addr), page_size(ps) {}

    // Real (single-template-arg) TensorAccessorArgs<CTA_OFFSET> constructor.
    // The real tt_metal/hw/inc/api/tensor/tensor_accessor_args.h uses one template
    // param; concat readers call make_tensor_accessor_tuple which constructs via
    // (TensorAccessorArgs<CTA_OFFSET>, addr). page_size comes from the Args type's
    // ::AlignedPageSize static member (also defined in the real header).
    template <uint32_t CTA_OFFSET>
    TensorAccessor(const ::TensorAccessorArgs<CTA_OFFSET>& /*args*/,
                   size_t   addr,
                   uint32_t ps = ::TensorAccessorArgs<CTA_OFFSET>::AlignedPageSize) noexcept :
        bank_base_address(addr), page_size(ps) {}

    // Direct constructor — used by host-side helpers and test code.
    explicit TensorAccessor(size_t addr, uint32_t ps = 0) noexcept :
        bank_base_address(addr), page_size(ps) {}

    // get_noc_addr — delegates to InterleavedAddrGen<true> (DRAM-style banking).
    inline uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = 0) const {
        InterleavedAddrGen<true> addrgen{static_cast<uint32_t>(bank_base_address), page_size};
        return addrgen.get_noc_addr(page_id, offset, noc);
    }
};

// ---- make_tensor_accessor_tuple ----------------------------------------
// Mirrors tt_metal/hw/inc/api/tensor/tensor_accessor.h:567-578. Used by
// concat's reader_concat_interleaved_start_id.cpp (and any kernel that
// builds an N-way fan-in of TensorAccessor instances from a tuple of args).
namespace tensor_accessor::detail {
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple(
    const std::tuple<Args...>& args, uint32_t address_rt_arg_index_start, std::integer_sequence<uint32_t, Indexes...>) {
    return std::make_tuple(
        TensorAccessor(std::get<Indexes>(args), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes))...);
}
}  // namespace tensor_accessor::detail

template <typename... Args>
auto make_tensor_accessor_tuple(const std::tuple<Args...>& args, uint32_t address_rt_arg_index_start) {
    return tensor_accessor::detail::make_tensor_accessor_tuple(
        args, address_rt_arg_index_start, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}

// ---- AbstractTensorAccessorWrapper -------------------------------------
// Mirrors tt_metal/hw/inc/api/tensor/tensor_accessor.h:627-664. Type-erased
// wrapper over a templated TensorAccessor, used by concat readers to iterate
// uniformly over a heterogeneous tuple of accessors at runtime.
class AbstractTensorAccessorWrapper {
public:
    AbstractTensorAccessorWrapper() = default;

    template <typename Accessor>
    AbstractTensorAccessorWrapper(const Accessor& accessor) :
        accessor_ptr(&accessor),
        get_noc_addr_fn([](const void* a, uint32_t page_idx, uint32_t offset, uint8_t noc) {
            return static_cast<const Accessor*>(a)->get_noc_addr(page_idx, offset, noc);
        }) {}

    uint64_t get_noc_addr(uint32_t page_idx, uint32_t offset = 0, uint8_t noc = 0) const {
        return get_noc_addr_fn(accessor_ptr, page_idx, offset, noc);
    }

private:
    using GetNocAddrFn = uint64_t (*)(const void*, uint32_t, uint32_t, uint8_t);
    const void* accessor_ptr = nullptr;
    GetNocAddrFn get_noc_addr_fn = nullptr;
};

namespace tensor_accessor::detail {
template <typename... Accessors, uint32_t... Indexes>
auto make_abstract_tensor_accessor_wrappers(
    const std::tuple<Accessors...>& accessors, std::integer_sequence<uint32_t, Indexes...>)
    -> std::array<AbstractTensorAccessorWrapper, sizeof...(Accessors)> {
    return {AbstractTensorAccessorWrapper(std::get<Indexes>(accessors))...};
}
}  // namespace tensor_accessor::detail

template <typename... Accessors>
auto make_abstract_tensor_accessor_wrappers(const std::tuple<Accessors...>& accessors) {
    return tensor_accessor::detail::make_abstract_tensor_accessor_wrappers(
        accessors, std::make_integer_sequence<uint32_t, sizeof...(Accessors)>());
}
