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
    // Each shard-metadata field is either compile-time (CTA) or runtime (CRTA, i.e. a
    // common runtime arg). The shadow supports any combination as long as rank,
    // num_banks and page_size are compile-time (always true for the sharded kernels
    // emule runs — reshape/tilize/concat static; ttnn.permute makes only tensor_shape
    // runtime). tensor_shape / shard_shape / bank_coords are read per-field from CTA
    // or CRTA in TensorAccessor. Anything outside this (runtime rank/num_banks) falls
    // back to the interleaved path.
    static constexpr bool rank_is_crta         = (ArgsConfig & (1u << 2)) != 0;
    static constexpr bool num_banks_is_crta    = (ArgsConfig & (1u << 3)) != 0;
    static constexpr bool tensor_shape_is_crta = (ArgsConfig & (1u << 4)) != 0;
    static constexpr bool shard_shape_is_crta  = (ArgsConfig & (1u << 5)) != 0;
    static constexpr bool bank_coords_is_crta  = (ArgsConfig & (1u << 6)) != 0;
    static constexpr bool page_size_is_crta    = (ArgsConfig & (1u << 7)) != 0;
    static constexpr bool is_sharded_supported =
        is_sharded && !rank_is_crta && !num_banks_is_crta && !page_size_is_crta;

    // Sharded CTA layout (mirrors tt_metal .../tensor_accessor_args.h). Fields that
    // live in CRTA are skipped in the CTA stream, so later CTA offsets shift down.
    //   [Cta+0]=config [Cta+1]=page_size [Cta+2]=rank [Cta+3]=num_banks(raw)
    //   then, in order and only if NOT crta: tensor_shape[rank], shard_shape[rank],
    //   bank_coords[ceil(num_banks/2)] (2 packed u16 per u32 word).
    static constexpr uint32_t ShardedRank        = is_sharded_supported ? get_compile_time_arg_val(CTA_OFFSET + 2) : 0;
    static constexpr uint32_t ShardedNumBanksRaw = is_sharded_supported ? get_compile_time_arg_val(CTA_OFFSET + 3) : 0;
    static constexpr uint32_t ShardedNumBanks    = ShardedNumBanksRaw & 0x7FFFFFFFu;
    static constexpr bool ShardedContiguous      = (ShardedNumBanksRaw & 0x80000000u) != 0;
    static constexpr std::size_t ShardedTensorShapeCta = CTA_OFFSET + 4;
    static constexpr std::size_t ShardedShardShapeCta =
        ShardedTensorShapeCta + (tensor_shape_is_crta ? 0 : ShardedRank);
    static constexpr std::size_t ShardedBankCoordsCta =
        ShardedShardShapeCta + (shard_shape_is_crta ? 0 : ShardedRank);

    uint32_t crta_offset_rt = 0;

    constexpr TensorAccessorArgs() = default;
    constexpr explicit TensorAccessorArgs(uint32_t crta_offset) : crta_offset_rt(crta_offset) {}

    // CRTA offsets for the runtime shard-metadata fields (rank & num_banks are CT here,
    // so they contribute 0 to the CRTA stream). Matches tensor_accessor_args.h.
    constexpr uint32_t sharded_tensor_shape_crta_offset() const { return crta_offset(); }
    constexpr uint32_t sharded_shard_shape_crta_offset() const {
        return sharded_tensor_shape_crta_offset() + (tensor_shape_is_crta ? ShardedRank : 0);
    }
    constexpr uint32_t sharded_bank_coords_crta_offset() const {
        return sharded_shard_shape_crta_offset() + (shard_shape_is_crta ? ShardedRank : 0);
    }

    static constexpr uint32_t get_aligned_page_size() { return AlignedPageSize; }
    static constexpr uint32_t num_compile_time_args() {
        if constexpr (is_sharded_supported) {
            return 4 + (tensor_shape_is_crta ? 0 : ShardedRank) + (shard_shape_is_crta ? 0 : ShardedRank) +
                   (bank_coords_is_crta ? 0 : (ShardedNumBanks + 1) / 2);
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
        if constexpr (requires { DSpec::is_sharded_supported; }) {
            if constexpr (DSpec::is_sharded_supported) {
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
        if constexpr (requires { DSpec::is_sharded_supported; }) {
            if constexpr (DSpec::is_sharded_supported) {
                // A shard's first page: page offset within shard is 0.
                return sharded_bank_noc_addr(shard_id, /*page_offset_within_shard=*/0, offset, noc);
            }
        }
        return get_noc_addr(shard_id, offset, noc);
    }

  private:
    // Read the rank shard-metadata vectors from wherever they live: compile-time (CTA,
    // unrolled) or runtime (CRTA, a common runtime arg — e.g. ttnn.permute makes
    // tensor_shape runtime). rank & num_banks are always compile-time (is_sharded_supported).
    FORCE_INLINE void load_tensor_shape(uint32_t* ts) const {
        constexpr uint32_t R = DSpec::ShardedRank;
        if constexpr (DSpec::tensor_shape_is_crta) {
            const uint32_t base = dspec_instance.sharded_tensor_shape_crta_offset();
            for (uint32_t i = 0; i < R; ++i) ts[i] = get_common_arg_val<uint32_t>(base + i);
        } else {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((ts[I] = get_compile_time_arg_val(DSpec::ShardedTensorShapeCta + I)), ...);
            }(std::make_index_sequence<R>{});
        }
    }
    FORCE_INLINE void load_shard_shape(uint32_t* ss) const {
        constexpr uint32_t R = DSpec::ShardedRank;
        if constexpr (DSpec::shard_shape_is_crta) {
            const uint32_t base = dspec_instance.sharded_shard_shape_crta_offset();
            for (uint32_t i = 0; i < R; ++i) ss[i] = get_common_arg_val<uint32_t>(base + i);
        } else {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((ss[I] = get_compile_time_arg_val(DSpec::ShardedShardShapeCta + I)), ...);
            }(std::make_index_sequence<R>{});
        }
    }
    FORCE_INLINE void load_bank_xy(uint16_t* bank_xy) const {
        constexpr uint32_t NB = DSpec::ShardedNumBanks;
        if constexpr (DSpec::bank_coords_is_crta) {
            const uint32_t base = dspec_instance.sharded_bank_coords_crta_offset();
            for (uint32_t i = 0; i < NB; ++i) {
                const uint32_t w = get_common_arg_val<uint32_t>(base + i / 2);
                bank_xy[i] = static_cast<uint16_t>((i % 2 == 0) ? (w & 0xffffu) : (w >> 16));
            }
        } else {
            [&]<std::size_t... I>(std::index_sequence<I...>) {
                ((bank_xy[I] = static_cast<uint16_t>(
                      (I % 2 == 0) ? (get_compile_time_arg_val(DSpec::ShardedBankCoordsCta + I / 2) & 0xffffu)
                                   : (get_compile_time_arg_val(DSpec::ShardedBankCoordsCta + I / 2) >> 16))),
                 ...);
            }(std::make_index_sequence<NB>{});
        }
    }

    // Sharded page addressing (L1 or DRAM). Ports the static-rank path of
    // tt_metal/hw/inc/api/tensor/tensor_accessor.h::get_bank_and_offset onto emule's
    // NoC helper. Reached for any is_sharded_supported tensor; the L1 and DRAM banks
    // then diverge in bank_addr_from_shard (worker-coord get_noc_addr for L1 vs the
    // physical dram_bank_to_noc_xy table for DRAM). Previously the whole sharded path
    // was gated to !is_dram, so DRAM-sharded tensors wrongly fell through to
    // InterleavedAddrGen — the cause of the width-sharded-DRAM tilize round-trip
    // corruption (#45331).
    FORCE_INLINE uint64_t sharded_noc_addr(uint32_t page_id, uint32_t offset, uint8_t noc) const {
        constexpr uint32_t R = DSpec::ShardedRank;
        uint32_t ts[R]{}, ss[R]{};
        load_tensor_shape(ts);
        load_shard_shape(ss);

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
        load_shard_shape(ss);
        load_tensor_shape(ts);
        for (uint32_t i = 0; i < R; ++i) { sv *= ss[i]; num_shards *= (ts[i] + ss[i] - 1) / ss[i]; }
        return bank_addr_from_shard(flattened_shard_id, page_offset_within_shard, sv, num_shards, offset, noc);
    }

    FORCE_INLINE uint64_t bank_addr_from_shard(
        uint32_t flattened_shard_id, uint32_t page_offset_within_shard, uint32_t shard_volume,
        uint32_t num_shards, uint32_t offset, uint8_t noc) const {
        constexpr uint32_t NB = DSpec::ShardedNumBanks;
        uint16_t bank_xy[NB]{};
        load_bank_xy(bank_xy);

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
        uint32_t addr = static_cast<uint32_t>(bank_base_address) + bank_page_offset * page_size + offset;
        if constexpr (DSpec::is_dram) {
            // DRAM path — mirror InterleavedAddrGen<DRAM> exactly (both live over the
            // same runner-populated bank tables in internal/dataflow/dataflow_api_addrgen.h):
            //   1) fold in the per-bank DRAM base offset (bank_to_dram_offset);
            //   2) take the *physical* DRAM bank NoC coords from dram_bank_to_noc_xy —
            //      NOT the tensor's logical bank_coords metadata (bank_xy), which is a
            //      worker-style (x<<8)|y encoding that does not match emule's physical
            //      DRAM bank placement for banks > 0; and
            //   3) build the NoC address directly, bypassing ::get_noc_addr whose
            //      __emule_addr_to_offset step slot-masks to 2 MB (only valid for local
            //      L1 pointers; a DRAM bank offset can exceed 2 MB).
            // Using bank_xy + the 2 MB mask was the width-sharded-DRAM tilize corruption
            // (#45331). bank_id indexes the tensor's bank list = the DRAM bank index.
            addr += static_cast<uint32_t>(bank_to_dram_offset[bank_id]);
            const uint32_t dram_noc_xy = dram_bank_to_noc_xy[noc][bank_id];
            return (static_cast<uint64_t>(dram_noc_xy) << NOC_ADDR_COORD_SHIFT) | static_cast<uint64_t>(addr);
        }
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
