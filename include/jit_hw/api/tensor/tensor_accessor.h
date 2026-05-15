// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT TensorAccessor stub.
// Uses the REAL TensorAccessorArgs from tt-metal/hw/inc (pure C++17 templates,
// compiles without KERNEL_BUILD). Keeps a simplified TensorAccessor struct
// that delegates to InterleavedAddrGen for NOC address computation.

#include <cstdint>
#include <cstddef>

// Real TensorAccessorArgs — resolves via -I<tt_metal>/tt_metal/hw/inc
// Dependency chain: tensor_accessor_args.h -> arg_config.hpp -> flags.hpp, const.h
#include "api/tensor/tensor_accessor_args.h"

// Full definition of InterleavedAddrGen needed for TensorAccessor::get_noc_addr.
#include "internal/dataflow/dataflow_api_addrgen.h"

// Simplified TensorAccessor for emulation.
// Wraps bank_base_address + page_size, delegates to InterleavedAddrGen for
// proper banked address generation.
struct TensorAccessor {
    uint32_t bank_base_address;
    uint32_t page_size;

    template<std::size_t CTA, std::size_t CRTA>
    TensorAccessor(const TensorAccessorArgs<CTA, CRTA>&,
                   size_t addr,
                   uint32_t ps = TensorAccessorArgs<CTA, CRTA>::AlignedPageSize)
        : bank_base_address(static_cast<uint32_t>(addr)), page_size(ps) {}

    // get_noc_addr — delegates to InterleavedAddrGen<true> (DRAM) for proper banking.
    inline uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = 0) const {
        InterleavedAddrGen<true> addrgen{bank_base_address, page_size};
        return addrgen.get_noc_addr(page_id, offset, noc);
    }
};
