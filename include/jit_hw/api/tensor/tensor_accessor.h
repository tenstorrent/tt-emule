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

    // Direct constructor — used by host-side helpers and test code.
    explicit TensorAccessor(size_t addr, uint32_t ps = 0) noexcept :
        bank_base_address(addr), page_size(ps) {}

    // get_noc_addr — delegates to InterleavedAddrGen<true> (DRAM-style banking).
    inline uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = 0) const {
        InterleavedAddrGen<true> addrgen{static_cast<uint32_t>(bank_base_address), page_size};
        return addrgen.get_noc_addr(page_id, offset, noc);
    }
};
