#pragma once
#include "tt_emule/program.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace tt_emule {

// Compile a kernel source file at runtime using the host g++.
// Returns a KernelFn (std::function<void()>) backed by a dlopen'd .so.
//
// kernel_src_path  - absolute or relative (to cwd) path to kernel .cpp
// compile_args     - compile-time args; passed as -DKERNEL_COMPILE_TIME_ARGS=val0,val1,...
// jit_include_dir  - path to the jit_hw/ directory (set by CMake via TT_EMULE_JIT_INCLUDE_DIR)
KernelFn jit_compile_kernel(const std::string& kernel_src_path,
                             const std::vector<uint32_t>& compile_args,
                             const std::string& jit_include_dir);

} // namespace tt_emule
