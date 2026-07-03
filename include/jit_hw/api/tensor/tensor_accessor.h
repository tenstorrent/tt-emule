// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/internal/dataflow/dataflow_api_addrgen.h"

uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc);
uint64_t get_noc_addr(uint32_t addr, uint8_t noc);

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET = 0>
struct TensorAccessorArgs {
    static constexpr uint32_t ArgsConfig = get_compile_time_arg_val(CTA_OFFSET);
    static constexpr bool is_sharded = (ArgsConfig & (1u << 0)) != 0;
    static constexpr bool is_dram = (ArgsConfig & (1u << 1)) != 0;
    static constexpr uint32_t AlignedPageSize = get_compile_time_arg_val(CTA_OFFSET + 1);

    static constexpr uint32_t get_aligned_page_size() { return AlignedPageSize; }
    static constexpr uint32_t num_compile_time_args() { return 2; }
    static constexpr uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }
};

namespace tensor_accessor {

template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
struct TensorAccessorBindingToken {
    static constexpr uint32_t ArgsConfig = get_compile_time_arg_val(CTA_OFFSET);
    static constexpr bool is_sharded = (ArgsConfig & (1u << 0)) != 0;
    static constexpr bool is_dram = (ArgsConfig & (1u << 1)) != 0;
    static constexpr uint32_t AlignedPageSize = get_compile_time_arg_val(CTA_OFFSET + 1);
};

template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
using TensorBindingToken = TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>;

struct Page {
    uint64_t noc_addr_value = 0;

    constexpr Page() = default;
    constexpr explicit Page(uint64_t noc_addr) : noc_addr_value(noc_addr) {}

    constexpr uint64_t noc_addr() const { return noc_addr_value; }
};

}  // namespace tensor_accessor

template <typename DSpecT>
struct TensorAccessor {
    using DSpec = DSpecT;
    static constexpr bool is_dram = DSpec::is_dram;

    static constexpr uint32_t default_aligned_page_size() {
        if constexpr (requires { DSpec::AlignedPageSize; }) {
            return DSpec::AlignedPageSize;
        } else {
            return 0;
        }
    }

    DSpec dspec_instance{};
    size_t bank_base_address = 0;
    uint32_t aligned_page_size = 0;

    constexpr explicit TensorAccessor(
        DSpec dspec, const size_t bank_base_address_in, const uint32_t page_size_in = 0) :
        dspec_instance(dspec),
        bank_base_address(bank_base_address_in),
        aligned_page_size(page_size_in == 0 ? default_aligned_page_size() : page_size_in) {}

    constexpr explicit TensorAccessor(const size_t bank_base_address_in = 0, uint32_t page_size_in = 0) :
        bank_base_address(bank_base_address_in),
        aligned_page_size(page_size_in == 0 ? default_aligned_page_size() : page_size_in) {}

    template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET>
    constexpr TensorAccessor(
        const TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>&,
        const size_t bank_base_address_in,
        const uint32_t page_size_in = TensorAccessorArgs<CTA_OFFSET, CRTA_OFFSET>::AlignedPageSize) :
        bank_base_address(bank_base_address_in), aligned_page_size(page_size_in) {}

    template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
    TensorAccessor(tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>) :
        TensorAccessor(static_cast<size_t>(get_common_arg_val<uint32_t>(ADDR_CRTA_OFFSET / sizeof(uint32_t)))) {}

    constexpr const DSpec& dspec() const { return dspec_instance; }
    constexpr DSpec& dspec() { return dspec_instance; }

    FORCE_INLINE uint32_t get_aligned_page_size() const { return aligned_page_size; }
    FORCE_INLINE constexpr uint32_t get_bank_base_address() const { return static_cast<uint32_t>(bank_base_address); }

    FORCE_INLINE uint64_t get_noc_addr(
        const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        const uint32_t page_size = aligned_page_size == 0 ? 1u : aligned_page_size;
        return InterleavedAddrGen<is_dram>{
                   .bank_base_address = static_cast<uint32_t>(bank_base_address), .page_size = page_size}
            .get_noc_addr(page_id, offset, noc);
    }

    FORCE_INLINE uint64_t get_shard_noc_addr(
        const uint32_t shard_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr(shard_id, offset, noc);
    }
};

template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
TensorAccessor(tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>)
    -> TensorAccessor<tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>>;

template <typename Accessor>
struct ShardView {
    const Accessor& accessor;

    FORCE_INLINE uint64_t get_noc_addr(
        const uint32_t shard_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return accessor.get_shard_noc_addr(shard_id, offset, noc);
    }
};

struct AbstractTensorAccessorWrapper {
    using GetNocAddrFn = uint64_t (*)(const void*, uint32_t, uint32_t, uint8_t);

    const void* accessor = nullptr;
    GetNocAddrFn get_noc_addr_fn = nullptr;

    constexpr AbstractTensorAccessorWrapper() = default;

    template <typename Accessor>
    constexpr explicit AbstractTensorAccessorWrapper(const Accessor& acc) :
        accessor(&acc),
        get_noc_addr_fn([](const void* ptr, uint32_t page_id, uint32_t offset, uint8_t noc) -> uint64_t {
            return static_cast<const Accessor*>(ptr)->get_noc_addr(page_id, offset, noc);
        }) {}

    FORCE_INLINE uint64_t get_noc_addr(
        const uint32_t page_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        return get_noc_addr_fn == nullptr ? 0 : get_noc_addr_fn(accessor, page_id, offset, noc);
    }
};

template <typename... Accessors>
auto make_abstract_tensor_accessor_wrappers(const std::tuple<Accessors...>& accessors) {
    return std::apply(
        [](const auto&... acc) { return std::array<AbstractTensorAccessorWrapper, sizeof...(Accessors)>{
            AbstractTensorAccessorWrapper(acc)...}; },
        accessors);
}
