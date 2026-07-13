// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <sys/mman.h>

namespace tt_emule {

/// Map a worker-L1 region anywhere in the low 4 GB.
///
/// Worker L1 must be uint32-addressable: a kernel's 32-bit L1 address is
/// dereferenced *directly* as a host pointer (tt-metal kernels do
/// `reinterpret_cast<T*>(get_write_ptr())` and read/write through it — a
/// load-bearing contract we can't change), so the backing must live in [0,4 GB).
///
/// `MAP_32BIT` only reaches [0,2 GB) — enough for ~7-8 Blackhole chips. To use the
/// full [0,4 GB) (≈16 chips), fall back to a free gap in [2 GB,4 GB) found from
/// /proc/self/maps and placed with plain `MAP_FIXED`. `MAP_FIXED_NOREPLACE` is
/// intentionally avoided: the host kernel (3.10/el7) predates it (Linux 4.17) and
/// would silently treat the flag as an advisory hint, possibly landing above 4 GB.
inline void* __emule_mmap_low4g(size_t size) {
    constexpr int RW = PROT_READ | PROT_WRITE;
    // Fast path: kernel-placed in [0,2 GB).
    void* p = mmap(nullptr, size, RW, MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT | MAP_NORESERVE, -1, 0);
    if (p != MAP_FAILED) {
        return p;
    }

    // [0,2 GB) exhausted: find a free [2 GB,4 GB) gap from /proc/self/maps and MAP_FIXED it.
    // Serialized so the scan + map is atomic w.r.t. other worker-L1 allocations.
    static std::mutex g_mtx;
    std::lock_guard<std::mutex> lk(g_mtx);
    constexpr uintptr_t LO = 0x80000000ULL;   // 2 GB
    constexpr uintptr_t HI = 0x100000000ULL;  // 4 GB
    FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) {
        return MAP_FAILED;
    }
    uintptr_t prev_end = LO;
    uintptr_t chosen = 0;
    char line[512];
    while (std::fgets(line, sizeof(line), f)) {
        uintptr_t s = 0, e = 0;
        if (std::sscanf(line, "%lx-%lx", &s, &e) != 2) {
            continue;
        }
        if (e <= LO) {
            continue;  // entirely below the window
        }
        if (s >= HI) {
            break;  // maps are ascending — past the window
        }
        if (s > prev_end && prev_end + size <= s) {
            chosen = prev_end;  // free gap [prev_end, s) is large enough
            break;
        }
        if (e > prev_end) {
            prev_end = e;
        }
    }
    if (chosen == 0 && prev_end + size <= HI) {
        chosen = prev_end;  // trailing gap up to 4 GB
    }
    std::fclose(f);
    if (chosen == 0) {
        return MAP_FAILED;
    }

    void* q = mmap(reinterpret_cast<void*>(chosen), size, RW,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    if (q == MAP_FAILED) {
        return MAP_FAILED;
    }
    // The uint32 aliasing only holds if the region is below 4 GB. Fail loudly otherwise.
    if (reinterpret_cast<uintptr_t>(q) + size > HI) {
        munmap(q, size);
        return MAP_FAILED;
    }
    return q;
}

}  // namespace tt_emule
