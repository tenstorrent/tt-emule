// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#define NUM_DRAM_BANKS 12
#define NUM_L1_BANKS 64
#define KERNEL_COMPILE_TIME_ARGS 2, 64

#include "jit_hw/api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "api/tensor/tensor_binding_token.h"

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

bool tensor_binding_token_matches_current_metal2_binding_name() {
    ThreadCommonCtx ctx(ThreadCommonCtx::Kind::Datamovement);
    std::array<uint32_t, 1> common_args{0x1000};
    ctx.common_rt_args = common_args.data();
    __emule_self = &ctx;

    using Token = tensor_accessor::TensorBindingToken<0, 0>;
    static_assert(Token::is_dram);
    static_assert(Token::AlignedPageSize == 64);

    TensorAccessor accessor(Token{});
    if (accessor.get_bank_base_address() != 0x1000) {
        std::cerr << "TensorAccessor binding token decoded wrong base address\n";
        return false;
    }
    if (accessor.get_aligned_page_size() != 64) {
        std::cerr << "TensorAccessor binding token decoded wrong page size\n";
        return false;
    }
    return true;
}

bool tensor_accessor_args_matches_current_offset_api() {
    using Args = TensorAccessorArgs<0, 7>;
    static_assert(Args::is_dram);
    static_assert(!Args::is_sharded);
    static_assert(Args::is_interleaved);
    static_assert(Args::AlignedPageSize == 64);
    static_assert(Args::next_compile_time_args_offset() == 2);

    constexpr Args args;
    static_assert(args.num_common_runtime_args() == 0);
    static_assert(args.next_common_runtime_args_offset() == 7);

    TensorAccessorArgs<0> runtime_offset_args(11);
    if (runtime_offset_args.next_common_runtime_args_offset() != 11) {
        std::cerr << "TensorAccessorArgs decoded wrong runtime offset\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    return tensor_binding_token_matches_current_metal2_binding_name() &&
                   tensor_accessor_args_matches_current_offset_api()
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}
