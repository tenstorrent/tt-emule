#pragma once
// Emulation stub for experimental::CoreLocalMem<T>.
// Wraps an L1 address as a typed pointer into Core's mmap'd L1 region.

#include <cstdint>
#include "tt_emule/device.hpp"

extern thread_local tt_emule::Core* __core;

namespace experimental {

template<typename T>
class CoreLocalMem {
    T* ptr_;
public:
    explicit CoreLocalMem(uint32_t l1_addr)
        : ptr_(reinterpret_cast<T*>(__core->l1_ptr(l1_addr))) {}
    CoreLocalMem(uint32_t l1_addr, uint32_t /*size*/) : CoreLocalMem(l1_addr) {}
    T* data() { return ptr_; }
    operator uint8_t*() { return reinterpret_cast<uint8_t*>(ptr_); }
};

} // namespace experimental
