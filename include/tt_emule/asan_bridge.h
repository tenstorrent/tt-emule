#pragma once

// Allocator-driven ASan poisoning bridge.
//
// tt-metal's AllocatorImpl::allocate_buffer and ::deallocate_buffer call
// these to mark a buffer's per-core L1 / DRAM region as live (unpoisoned)
// or dead (poisoned). Defined here as static inline so every TU that
// includes the header gets its own internal-linkage copy — no exported
// symbol, no weak/strong resolution, no chance of an out-of-tree provider
// silently shadowing the real implementation.
//
// History: previously defined as exported `extern "C"` symbols in
// tt-emule/src/kernel_runner.cpp with weak no-op fallbacks in libtt-umd's
// sw_emule_chip.cpp. The intent was that any consumer linking tt-emule
// would see the strong def; standalone UMD tools without tt-emule would
// fall back to the no-op. In practice, libtt_metal.so consumed only
// tt-emule::headers (interface target, no .o files), so the strong def
// was never linked into the runtime image and libtt-umd's weak no-op
// won at runtime — silently making per-buffer poisoning a no-op for the
// entire JIT path. The header-inline form sidesteps that class of bug
// because there is no symbol to resolve against.
//
// When ASan is off, EMULE_ASAN_POISON/UNPOISON expand to (void)0, the
// function bodies become trivially-empty, and the compiler elides every
// call site. Zero overhead.

#include <cstddef>
#include <cstdint>

#include "tt_emule/asan.h"

static inline void __emule_buffer_alloc(uint8_t* base, std::size_t size) {
    if (!base || size == 0) {
        return;
    }
    EMULE_ASAN_UNPOISON(base, size);
}

static inline void __emule_buffer_free(uint8_t* base, std::size_t size) {
    if (!base || size == 0) {
        return;
    }
    EMULE_ASAN_POISON(base, size);
}
