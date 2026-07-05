// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the concat reader kernels
// (ttnn/.../concat/device/kernels/dataflow/reader_concat_*interleaved_start_id.cpp),
// which build a tuple of TensorAccessors via:
//
//   auto args    = make_tensor_accessor_args_tuple<num_tensors, CTA_OFFSET>();
//   auto tuple   = make_tensor_accessor_tuple(args, src_addr_base_idx);
//   auto wrapped = make_abstract_tensor_accessor_wrappers(tuple);
//
// The shim was missing make_tensor_accessor_tuple, so these kernels failed JIT
// compile ("use of undeclared identifier 'make_tensor_accessor_tuple'") when
// tt-metal was built against tt-emule.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <tuple>

#define NUM_DRAM_BANKS 12
#define NUM_L1_BANKS 64
// Two interleaved DRAM tensors: {config=0b10 (dram), page=64}, {config=0b10, page=128}.
#define KERNEL_COMPILE_TIME_ARGS 2, 64, 2, 128

#include "jit_hw/api/tensor/tensor_accessor.h"

thread_local ThreadCommonCtx* __emule_self = nullptr;
thread_local uint8_t my_x[2] = {};
thread_local uint8_t my_y[2] = {};
thread_local uint32_t __emule_pending_noc_reads = 0;

extern "C" uint8_t* __emule_dram_ptr(uint64_t) { return nullptr; }
extern "C" uint8_t* __emule_noc_resolve(uint32_t, uint32_t, uint64_t) { return nullptr; }
uint16_t dram_bank_to_noc_xy[2][NUM_DRAM_BANKS] = {};
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] = {};
uint16_t l1_bank_to_noc_xy[2][NUM_L1_BANKS] = {};
int32_t bank_to_l1_offset[NUM_L1_BANKS] = {};

namespace {

bool make_tensor_accessor_tuple_builds_per_tensor_accessors() {
    ThreadCommonCtx ctx(ThreadCommonCtx::Kind::Datamovement);
    // Runtime args: base addresses start at index src_addr_base_idx (=3), one per tensor.
    std::array<uint32_t, 8> rt_args{};
    constexpr uint32_t src_addr_base_idx = 3;
    rt_args[src_addr_base_idx + 0] = 0x2000;
    rt_args[src_addr_base_idx + 1] = 0x9000;
    ctx.rt_args = rt_args.data();
    __emule_self = &ctx;

    constexpr uint32_t num_tensors = 2;
    // CTA layout mirrors the concat readers: page-size/config CTAs start at 0.
    auto args_tuple = make_tensor_accessor_args_tuple<num_tensors, 0>();
    auto accessors = make_tensor_accessor_tuple(args_tuple, src_addr_base_idx);
    auto wrappers = make_abstract_tensor_accessor_wrappers(accessors);

    static_assert(std::tuple_size<decltype(accessors)>::value == num_tensors);
    static_assert(wrappers.size() == num_tensors);

    const auto& a0 = std::get<0>(accessors);
    const auto& a1 = std::get<1>(accessors);

    if (a0.get_bank_base_address() != 0x2000 || a1.get_bank_base_address() != 0x9000) {
        std::cerr << "make_tensor_accessor_tuple decoded wrong base address(es): "
                  << std::hex << a0.get_bank_base_address() << ", "
                  << a1.get_bank_base_address() << "\n";
        return false;
    }
    if (a0.get_aligned_page_size() != 64 || a1.get_aligned_page_size() != 128) {
        std::cerr << "make_tensor_accessor_tuple decoded wrong page size(s): "
                  << std::dec << a0.get_aligned_page_size() << ", "
                  << a1.get_aligned_page_size() << "\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    return make_tensor_accessor_tuple_builds_per_tensor_accessors() ? EXIT_SUCCESS : EXIT_FAILURE;
}
