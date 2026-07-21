// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>

namespace tt_emule {

/// Map a worker-L1 region. Worker L1 is addressed by 0-based offsets — a kernel's 32-bit L1 address
/// is rebased onto the per-fiber `bridge_l1` at the deref site (see jit_hw/asan/asan_l1_checks.h), so
/// the backing may live anywhere in the 64-bit address space; a plain mmap suffices. `MAP_NORESERVE`
/// keeps a large multi-chip pool (a 32-chip Blackhole galaxy is ≈ 9 GB of worker L1) lazy-faulted:
/// RAM is consumed only by the cores a program actually touches.
///
/// EMULE_L1_MMAP_DEBUG logs each region's address + size — used to confirm placement above 4 GB on
/// large (galaxy-scale) meshes.
inline void* __emule_mmap_worker_l1(size_t size) {
    void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    static const bool dbg = std::getenv("EMULE_L1_MMAP_DEBUG") != nullptr;
    if (dbg && p != MAP_FAILED) {
        std::fprintf(stderr, "[EMULE_L1_MMAP] base=%p size=0x%zx (%.1f MB) %s 4GB\n", p, size,
                     size / (1024.0 * 1024.0),
                     reinterpret_cast<uintptr_t>(p) >= 0x100000000ULL ? "ABOVE" : "below");
    }
    return p;
}

}  // namespace tt_emule
