#pragma once

// Allocator-driven ASan poisoning bridge.
//
// tt-metal's AllocatorImpl::allocate_buffer and ::deallocate_buffer call
// these to mark a buffer's per-core L1 / DRAM region as live (unpoisoned)
// or dead (poisoned). Defined in src/kernel_runner.cpp; symbols are
// exported via -rdynamic so dlopen'd JIT .so files can also call them
// (currently only the host process calls — JIT side does not).
//
// When the host binary was not built with ASan, both calls compile to
// no-ops at link time inside tt-emule (see asan.h), so tt-metal can
// invoke them unconditionally.

#include <cstddef>
#include <cstdint>

extern "C" void __emule_buffer_alloc(uint8_t* base, std::size_t size);
extern "C" void __emule_buffer_free (uint8_t* base, std::size_t size);
