// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "jit_hw/jit_kernel_stubs.hpp"
#include "jit_hw/internal/dataflow/dataflow_api_addrgen.h"

uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc);
uint64_t get_noc_addr(uint32_t addr, uint8_t noc);

namespace tensor_accessor {

constexpr std::size_t UNKNOWN = static_cast<std::size_t>(-1);

}  // namespace tensor_accessor

template <std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET = tensor_accessor::UNKNOWN>
struct TensorAccessorArgs {
    static constexpr std::size_t Cta = CTA_OFFSET;
    static constexpr uint32_t ArgsConfig = get_compile_time_arg_val(CTA_OFFSET);
    static constexpr bool is_sharded = (ArgsConfig & (1u << 0)) != 0;
    static constexpr bool is_dram = (ArgsConfig & (1u << 1)) != 0;
    static constexpr bool is_interleaved = !is_sharded;
    static constexpr uint32_t AlignedPageSize = get_compile_time_arg_val(CTA_OFFSET + 1);

    // ArgsConfig runtime-relaxation bits (hostdevcommon/tensor_accessor/arg_config.hpp).
    // The shadow only models the fully-static sharded case (all shard metadata is CTA),
    // which is what the sharded reshape/tilize/concat kernels emit. If any field is a
    // runtime (CRTA) arg, is_static_sharded is false and get_noc_addr falls back to the
    // interleaved path (no worse than the pre-fix behavior).
    static constexpr bool rank_is_crta         = (ArgsConfig & (1u << 2)) != 0;
    static constexpr bool num_banks_is_crta    = (ArgsConfig & (1u << 3)) != 0;
    static constexpr bool tensor_shape_is_crta = (ArgsConfig & (1u << 4)) != 0;
    static constexpr bool shard_shape_is_crta  = (ArgsConfig & (1u << 5)) != 0;
    static constexpr bool bank_coords_is_crta  = (ArgsConfig & (1u << 6)) != 0;
    static constexpr bool page_size_is_crta    = (ArgsConfig & (1u << 7)) != 0;
    static constexpr bool is_static_sharded = is_sharded && !rank_is_crta && !num_banks_is_crta &&
                                              !tensor_shape_is_crta && !shard_shape_is_crta &&
                                              !bank_coords_is_crta && !page_size_is_crta;

    // Static-sharded CTA layout (mirrors tt_metal .../tensor_accessor_args.h):
    //   [Cta+0]=config [Cta+1]=page_size [Cta+2]=rank [Cta+3]=num_banks(raw)
    //   [Cta+4 .. +4+rank)      = tensor_shape
    //   [+4+rank .. +4+2*rank)  = shard_shape
    //   [+4+2*rank ..)          = bank coords (2 packed u16 per u32 word)
    static constexpr uint32_t ShardedRank        = is_static_sharded ? get_compile_time_arg_val(CTA_OFFSET + 2) : 0;
    static constexpr uint32_t ShardedNumBanksRaw = is_static_sharded ? get_compile_time_arg_val(CTA_OFFSET + 3) : 0;
    static constexpr uint32_t ShardedNumBanks    = ShardedNumBanksRaw & 0x7FFFFFFFu;
    static constexpr bool ShardedContiguous      = (ShardedNumBanksRaw & 0x80000000u) != 0;
    static constexpr std::size_t ShardedTensorShapeCta = CTA_OFFSET + 4;
    static constexpr std::size_t ShardedShardShapeCta  = ShardedTensorShapeCta + ShardedRank;
    static constexpr std::size_t ShardedBankCoordsCta  = ShardedShardShapeCta + ShardedRank;

    uint32_t crta_offset_rt = 0;

    constexpr TensorAccessorArgs() = default;
    constexpr explicit TensorAccessorArgs(uint32_t crta_offset) : crta_offset_rt(crta_offset) {}

    static constexpr uint32_t get_aligned_page_size() { return AlignedPageSize; }
    static constexpr uint32_t num_compile_time_args() {
        if constexpr (is_static_sharded) {
            return 4 + 2 * ShardedRank + (ShardedNumBanks + 1) / 2;
        } else {
            return 2;
        }
    }
    static constexpr uint32_t next_compile_time_args_offset() { return CTA_OFFSET + num_compile_time_args(); }

    constexpr uint32_t crta_offset() const {
        if constexpr (CRTA_OFFSET != tensor_accessor::UNKNOWN) {
            return CRTA_OFFSET;
        } else {
            return crta_offset_rt;
        }
    }

    constexpr uint32_t num_common_runtime_args() const { return 0; }
    constexpr uint32_t next_common_runtime_args_offset() const { return crta_offset() + num_common_runtime_args(); }

    constexpr uint32_t rank() const { return 1; }
    constexpr std::array<uint32_t, 1> tensor_shape() const { return {1}; }
    constexpr std::array<uint32_t, 1> shard_shape() const { return {1}; }
};

namespace tensor_accessor::detail {

template <uint32_t TENSOR_IDX, uint32_t CTA_OFFSET>
constexpr uint32_t get_tensor_accessor_args_cta_offset() {
    if constexpr (TENSOR_IDX == 0) {
        return CTA_OFFSET;
    } else {
        constexpr auto prev_offset = get_tensor_accessor_args_cta_offset<TENSOR_IDX - 1, CTA_OFFSET>();
        constexpr auto accessor_args = TensorAccessorArgs<prev_offset>();
        return accessor_args.next_compile_time_args_offset();
    }
}

template <uint32_t CTA_OFFSET, uint32_t... INDEXES>
constexpr auto get_tensor_accessor_args_cta_offsets(std::integer_sequence<uint32_t, INDEXES...>) {
    return std::integer_sequence<uint32_t, get_tensor_accessor_args_cta_offset<INDEXES, CTA_OFFSET>()...>();
}

template <uint32_t... CTA_OFFSETS>
constexpr auto make_tensor_accessor_args_tuple_from_cta_offsets(std::integer_sequence<uint32_t, CTA_OFFSETS...>) {
    return std::make_tuple(TensorAccessorArgs<CTA_OFFSETS>()...);
}

}  // namespace tensor_accessor::detail

template <uint32_t NUM_TENSORS, uint32_t CTA_OFFSET>
constexpr auto make_tensor_accessor_args_tuple() {
    constexpr auto cta_offsets = tensor_accessor::detail::get_tensor_accessor_args_cta_offsets<CTA_OFFSET>(
        std::make_integer_sequence<uint32_t, NUM_TENSORS>());
    return tensor_accessor::detail::make_tensor_accessor_args_tuple_from_cta_offsets(cta_offsets);
}

namespace tensor_accessor {

namespace detail {

template <typename T, typename = void>
struct has_aligned_page_size : std::false_type {};

template <typename T>
struct has_aligned_page_size<T, std::void_t<decltype(T::AlignedPageSize)>> : std::true_type {};

}  // namespace detail

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
        if constexpr (tensor_accessor::detail::has_aligned_page_size<DSpec>::value) {
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
        if constexpr (requires { DSpec::is_static_sharded; }) {
            if constexpr (DSpec::is_static_sharded && !DSpec::is_dram) {
                return sharded_noc_addr(page_id, offset, noc);
            }
        }
        const uint32_t page_size = aligned_page_size == 0 ? 1u : aligned_page_size;
        return InterleavedAddrGen<is_dram>{
                   .bank_base_address = static_cast<uint32_t>(bank_base_address), .page_size = page_size}
            .get_noc_addr(page_id, offset, noc);
    }

    FORCE_INLINE uint64_t get_shard_noc_addr(
        const uint32_t shard_id, const uint32_t offset = 0, uint8_t noc = noc_index) const {
        if constexpr (requires { DSpec::is_static_sharded; }) {
            if constexpr (DSpec::is_static_sharded && !DSpec::is_dram) {
                // A shard's first page: page offset within shard is 0.
                return sharded_bank_noc_addr(shard_id, /*page_offset_within_shard=*/0, offset, noc);
            }
        }
        return get_noc_addr(shard_id, offset, noc);
    }

  private:
    // L1-sharded page addressing. Ports the static-rank path of
    // tt_metal/hw/inc/api/tensor/tensor_accessor.h::get_bank_and_offset onto emule's
    // NoC helper. Only reached for is_static_sharded && !is_dram (guarded above); the
    // shadow otherwise routes through InterleavedAddrGen, which ignores sharding and
    // was the cause of the "column-0 only" corruption on sharded reshape/tilize.
    FORCE_INLINE uint64_t sharded_noc_addr(uint32_t page_id, uint32_t offset, uint8_t noc) const {
        constexpr uint32_t R = DSpec::ShardedRank;
        uint32_t ts[R]{}, ss[R]{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((ts[I] = get_compile_time_arg_val(DSpec::ShardedTensorShapeCta + I)), ...);
            ((ss[I] = get_compile_time_arg_val(DSpec::ShardedShardShapeCta + I)), ...);
        }(std::make_index_sequence<R>{});

        // shard strides (row-major within a shard) and shard-grid strides (over the shard grid)
        uint32_t shard_strides[R]{}, shard_grid_strides[R]{}, shard_volume = 1, num_shards = 1;
        {
            uint32_t s = 1;
            for (int i = static_cast<int>(R) - 1; i >= 0; --i) { shard_strides[i] = s; s *= ss[i]; }
            shard_volume = s;
        }
        {
            uint32_t s = 1;
            for (int i = static_cast<int>(R) - 1; i >= 0; --i) {
                shard_grid_strides[i] = s;
                s *= (ts[i] + ss[i] - 1) / ss[i];  // div_up: number of shards along dim i
            }
            num_shards = s;
        }

        uint32_t flattened_shard_id = 0, page_offset_within_shard = 0, pid = page_id;
        for (int i = static_cast<int>(R) - 1; i >= 0; --i) {
            const uint32_t page_coord = pid % ts[i];
            pid /= ts[i];
            flattened_shard_id += (page_coord / ss[i]) * shard_grid_strides[i];
            page_offset_within_shard += (page_coord % ss[i]) * shard_strides[i];
        }

        return bank_addr_from_shard(flattened_shard_id, page_offset_within_shard, shard_volume, num_shards, offset, noc);
    }

    // Shared shard->bank->NoC-address tail, used by both page and whole-shard entry points.
    FORCE_INLINE uint64_t sharded_bank_noc_addr(
        uint32_t flattened_shard_id, uint32_t page_offset_within_shard, uint32_t offset, uint8_t noc) const {
        // whole-shard entry: derive shard_volume from the shard shape.
        uint32_t sv = 1, num_shards = 1;
        constexpr uint32_t R = DSpec::ShardedRank;
        uint32_t ss[R]{}, ts[R]{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((ss[I] = get_compile_time_arg_val(DSpec::ShardedShardShapeCta + I)), ...);
            ((ts[I] = get_compile_time_arg_val(DSpec::ShardedTensorShapeCta + I)), ...);
        }(std::make_index_sequence<R>{});
        for (uint32_t i = 0; i < R; ++i) { sv *= ss[i]; num_shards *= (ts[i] + ss[i] - 1) / ss[i]; }
        return bank_addr_from_shard(flattened_shard_id, page_offset_within_shard, sv, num_shards, offset, noc);
    }

    FORCE_INLINE uint64_t bank_addr_from_shard(
        uint32_t flattened_shard_id, uint32_t page_offset_within_shard, uint32_t shard_volume,
        uint32_t num_shards, uint32_t offset, uint8_t noc) const {
        constexpr uint32_t NB = DSpec::ShardedNumBanks;
        uint16_t bank_xy[NB]{};
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((bank_xy[I] = static_cast<uint16_t>(
                  (I % 2 == 0) ? (get_compile_time_arg_val(DSpec::ShardedBankCoordsCta + I / 2) & 0xffffu)
                               : (get_compile_time_arg_val(DSpec::ShardedBankCoordsCta + I / 2) >> 16))),
             ...);
        }(std::make_index_sequence<NB>{});

        uint32_t bank_id, shard_in_bank;
        if constexpr (DSpec::ShardedContiguous) {
            const uint32_t shards_per_bank = num_shards / NB;
            bank_id = flattened_shard_id / shards_per_bank;
            shard_in_bank = flattened_shard_id % shards_per_bank;
        } else {  // round-robin
            bank_id = flattened_shard_id % NB;
            shard_in_bank = flattened_shard_id / NB;
        }

        const uint32_t bank_page_offset = shard_in_bank * shard_volume + page_offset_within_shard;
        const uint16_t xy = bank_xy[bank_id];
        const uint32_t bank_x = (xy >> 8) & 0xFF;  // packed as (x << 8) | y
        const uint32_t bank_y = xy & 0xFF;
        const uint32_t page_size = aligned_page_size == 0 ? 1u : aligned_page_size;
        const uint32_t addr = static_cast<uint32_t>(bank_base_address) + bank_page_offset * page_size + offset;
        return ::get_noc_addr(bank_x, bank_y, addr, noc);
    }

  public:
};

template <uint32_t CTA_OFFSET, uint32_t ADDR_CRTA_OFFSET>
TensorAccessor(tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>)
    -> TensorAccessor<tensor_accessor::TensorAccessorBindingToken<CTA_OFFSET, ADDR_CRTA_OFFSET>>;

namespace tensor_accessor::detail {

// Mirrors tt_metal/hw/inc/api/tensor/tensor_accessor.h: builds one TensorAccessor
// per TensorAccessorArgs in the tuple, reading each tensor's base address from a
// contiguous run of runtime args starting at address_rt_arg_index_start.
template <typename... Args, uint32_t... Indexes>
auto make_tensor_accessor_tuple(
    const std::tuple<Args...>& args,
    uint32_t address_rt_arg_index_start,
    std::integer_sequence<uint32_t, Indexes...>) {
    return std::make_tuple(
        TensorAccessor(std::get<Indexes>(args), get_arg_val<uint32_t>(address_rt_arg_index_start + Indexes))...);
}

}  // namespace tensor_accessor::detail

template <typename... Args>
auto make_tensor_accessor_tuple(const std::tuple<Args...>& args, uint32_t address_rt_arg_index_start) {
    return tensor_accessor::detail::make_tensor_accessor_tuple(
        args, address_rt_arg_index_start, std::make_integer_sequence<uint32_t, sizeof...(Args)>());
}

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
