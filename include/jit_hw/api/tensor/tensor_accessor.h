#pragma once
// JIT TensorAccessor stub.
// Uses the REAL TensorAccessorArgs from tt-metal/hw/inc (pure C++17 templates,
// compiles without KERNEL_BUILD). Keeps a simplified TensorAccessor struct
// since the real one requires hardware NOC/bank lookup tables.

#include <cstdint>
#include <cstddef>

// Real TensorAccessorArgs — resolves via -I<tt_metal>/tt_metal/hw/inc
// Dependency chain: tensor_accessor_args.h → arg_config.hpp → flags.hpp, const.h
#include "api/tensor/tensor_accessor_args.h"

// Simplified TensorAccessor for emulation.
// Wraps bank_base_address + page_size for noc_async_read/write_page.
struct TensorAccessor {
    uint32_t bank_base_address;
    uint32_t page_size;

    template<std::size_t CTA, std::size_t CRTA>
    TensorAccessor(const TensorAccessorArgs<CTA, CRTA>&,
                   size_t addr, uint32_t ps)
        : bank_base_address(static_cast<uint32_t>(addr)), page_size(ps) {}
};

// NOC index constant (always 0 in emulation).
constexpr uint8_t noc_index = 0;
