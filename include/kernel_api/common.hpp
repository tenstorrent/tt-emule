#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

// Forward declarations
namespace tt_emule {
    class Core;
    class Device;
}

// Thread-local context injected by kernel_runner before each kernel thread starts.
extern thread_local std::vector<uint32_t> __rt_args;
extern thread_local tt_emule::Core*       __core;
extern thread_local tt_emule::Device*     __device;

// Read a runtime argument by index.
template<typename T = uint32_t>
inline T get_arg_val(uint32_t idx) {
    if (idx >= __rt_args.size())
        throw std::out_of_range("get_arg_val: index out of range");
    return static_cast<T>(__rt_args[idx]);
}
