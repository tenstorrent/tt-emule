#pragma once
// Simplified TensorAccessor for JIT emulation.
// Supports interleaved (non-sharded) DRAM/L1 buffers with single-bank mapping:
//   page_addr = bank_base_address + page_id * page_size
//
// The real device header lives at hw/inc/api/tensor/tensor_accessor.h and has
// a complex DistributionSpec template.  We collapse everything into a single
// non-templated struct that noc_async_read_page/write_page can operate on.

#include <cstdint>
#include <cstddef>

// ---- ArgsConfig bitfield (mirrors hostdevcommon/tensor_accessor/arg_config.hpp) ----

namespace tensor_accessor {

enum class ArgConfig : uint8_t {
    None              = 0,
    Sharded           = 1 << 0,
    IsDram            = 1 << 1,
    RuntimeRank       = 1 << 2,
    RuntimeNumBanks   = 1 << 3,
    RuntimeTensorShape = 1 << 4,
    RuntimeShardShape  = 1 << 5,
    RuntimeBankCoords  = 1 << 6,
};

struct ArgsConfig {
    using Underlying = uint8_t;
    Underlying value;
    constexpr ArgsConfig() : value(0) {}
    constexpr explicit ArgsConfig(Underlying v) : value(v) {}
    constexpr bool test(ArgConfig bit) const { return (value & static_cast<Underlying>(bit)) != 0; }
};

} // namespace tensor_accessor

// ---- TensorAccessorArgs ----
// Decodes compile-time and common-runtime args to determine how many CTA/CRTA
// slots this tensor occupies, and whether it is DRAM / sharded.

template<std::size_t CTA_OFFSET, std::size_t CRTA_OFFSET = 0>
struct TensorAccessorArgs {
    // The first compile-time arg is the ArgsConfig bitfield.
    static constexpr auto args_config = tensor_accessor::ArgsConfig(
        static_cast<tensor_accessor::ArgsConfig::Underlying>(get_compile_time_arg_val(CTA_OFFSET)));

    static constexpr bool is_sharded           = args_config.test(tensor_accessor::ArgConfig::Sharded);
    static constexpr bool is_dram              = args_config.test(tensor_accessor::ArgConfig::IsDram);
    static constexpr bool rank_is_runtime      = args_config.test(tensor_accessor::ArgConfig::RuntimeRank);
    static constexpr bool num_banks_is_runtime = args_config.test(tensor_accessor::ArgConfig::RuntimeNumBanks);
    static constexpr bool tensor_shape_is_crta = args_config.test(tensor_accessor::ArgConfig::RuntimeTensorShape);
    static constexpr bool shard_shape_is_crta  = args_config.test(tensor_accessor::ArgConfig::RuntimeShardShape);
    static constexpr bool bank_coords_is_crta  = args_config.test(tensor_accessor::ArgConfig::RuntimeBankCoords);

    // ---- CTA layout (after ArgsConfig at CTA_OFFSET) ----
    // Slot 0:          ArgsConfig
    // Slot 1:          rank           (if !RuntimeRank)
    // Slot 1 or 2:     num_banks      (if !RuntimeNumBanks)
    // Then:            tensor_shape[rank] (if !RuntimeTensorShape)
    // Then:            shard_shape    (if !RuntimeShardShape && is_sharded)
    // Then:            bank_coords    (if !RuntimeBankCoords)

    static constexpr std::size_t rank_cta_offset = CTA_OFFSET + 1;

    static constexpr uint32_t get_rank() {
        if constexpr (!rank_is_runtime) {
            return get_compile_time_arg_val(rank_cta_offset);
        } else {
            return get_common_arg_val<uint32_t>(CRTA_OFFSET);
        }
    }

    static constexpr std::size_t num_banks_cta_offset =
        rank_cta_offset + (rank_is_runtime ? 0 : 1);

    static constexpr uint32_t get_num_banks() {
        if constexpr (!num_banks_is_runtime) {
            return get_compile_time_arg_val(num_banks_cta_offset);
        } else {
            return 1; // simplified
        }
    }

    // Count compile-time arg slots consumed by this accessor.
    static constexpr std::size_t num_compile_time_args() {
        std::size_t n = 1; // ArgsConfig
        if (!rank_is_runtime)       n += 1; // rank
        if (!num_banks_is_runtime)  n += 1; // num_banks
        // tensor_shape goes to CTA only if NOT RuntimeTensorShape
        if (!tensor_shape_is_crta)  n += get_rank();
        // shard_shape, bank_coords omitted in simplified version
        return n;
    }

    static constexpr std::size_t next_compile_time_args_offset() {
        return CTA_OFFSET + num_compile_time_args();
    }

    // Count common-runtime-arg slots consumed by this accessor.
    constexpr std::size_t num_common_runtime_args() const {
        std::size_t n = 0;
        if (rank_is_runtime)         n += 1;
        if (num_banks_is_runtime)    n += 1;
        if (tensor_shape_is_crta)    n += get_rank();
        return n;
    }

    constexpr std::size_t next_common_runtime_args_offset() const {
        return CRTA_OFFSET + num_common_runtime_args();
    }
};

// ---- TensorAccessor (simplified) ----
// Wraps bank_base_address + page_size for use by noc_async_read/write_page.

struct TensorAccessor {
    uint32_t bank_base_address;
    uint32_t page_size;

    template<std::size_t CTA, std::size_t CRTA>
    TensorAccessor(const TensorAccessorArgs<CTA, CRTA>&,
                   std::size_t addr, uint32_t ps)
        : bank_base_address(static_cast<uint32_t>(addr)), page_size(ps) {}
};

// Deduction guide so `TensorAccessor(args, addr, ps)` deduces the right type.
template<std::size_t CTA, std::size_t CRTA>
TensorAccessor(const TensorAccessorArgs<CTA, CRTA>&, std::size_t, uint32_t)
    -> TensorAccessor;

// NOC index constant (always 0 in emulation).
constexpr uint8_t noc_index = 0;
