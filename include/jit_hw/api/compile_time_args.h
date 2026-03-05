#pragma once
#include <cstdint>

// Compile-time kernel arguments.  The JIT compiler passes
//   -DKERNEL_COMPILE_TIME_ARGS=val0,val1,...
// on the command line.  If not provided, a single zero placeholder is used so
// that kernels which never call get_compile_time_arg_val still compile cleanly.

#ifndef KERNEL_COMPILE_TIME_ARGS
#define KERNEL_COMPILE_TIME_ARGS 0
#endif

namespace {
constexpr uint32_t kernel_compile_time_args_arr[] = {KERNEL_COMPILE_TIME_ARGS};
template<int N>
constexpr uint32_t get_ct_arg() { return kernel_compile_time_args_arr[N]; }
} // anonymous namespace

#define get_compile_time_arg_val(N) get_ct_arg<N>()
