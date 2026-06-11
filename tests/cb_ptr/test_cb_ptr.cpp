// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Per-RISC circular-buffer pointer regression coverage (issue #139).
//
// Exercises jit_hw/internal/emule_cb_ptr.h + cb_interface.h directly — the
// per-RISC CB pointer model that fixes the pad_rm_sharded_stickwise race:
//
//   1. PER-RISC ISOLATION (the race fix). Two threads share one CBSyncState
//      array (as reader/writer RISCs on a core do). One thread advances its
//      write pointer; the other thread's get_write_ptr / get_local_cb_interface
//      must still report the shard base. Under the old shared-index model the
//      advancing thread corrupted the other's base — the #139 bug.
//
//   2. WRITE-BACK. get_local_cb_interface returns a reference into the per-RISC
//      register file, so a kernel that writes fifo_rd_ptr / fifo_wr_ptr (e.g.
//      override_cb_rd_ptr, reconfig_cbs_for_mask) takes effect: the next
//      __emule_cb_rd_addr / advance reads/advances from the written value.
//
//   3. WRAP. Advancing past the ring end wraps back to base.
//
// CB memory is mmap'd below 4 GB (MAP_32BIT), like real emule L1, so the
// 16-byte-encoded fifo pointers (uint32, kernel does `<< 4`) round-trip exactly.
//
// The test links only tt_emule_lib, so it provides the one thread_local the
// JIT path normally gets from libtt_metal: __emule_cbs.

#include "tt_emule/cb_sync_state.hpp"
#include "jit_hw/internal/cb_interface.h"  // pulls emule_cb_ptr.h (LocalCBInterface, helpers)

#include <sys/mman.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// The per-RISC JIT path consumes this thread_local; libtt_metal defines it at
// runtime, here the test owns it (set per worker thread to the shared array).
thread_local tt_emule::CBSyncState* __emule_cbs = nullptr;

namespace {

#define CHECK(cond) do {                                                      \
    if (!(cond)) {                                                            \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

constexpr uint32_t kCb = 3;
constexpr uint32_t kPageSize = 64;   // 16-byte aligned (silicon CB constraint)
constexpr uint32_t kNumPages = 8;

}  // namespace

int main() {
    // Fake L1 region below 4 GB, matching emule's MAP_32BIT CB memory so the
    // uint32 16-byte pointer encoding is lossless.
    const size_t bytes = kPageSize * kNumPages;
    void* mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_32BIT, -1, 0);
    CHECK(mem != MAP_FAILED);
    auto* base = static_cast<uint8_t*>(mem);
    CHECK((reinterpret_cast<uintptr_t>(base) & 15) == 0);  // 16-byte aligned
    const uint32_t base16 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base) >> cb_addr_shift);

    // Shared per-core CB state array, indexed by cb id.
    std::vector<tt_emule::CBSyncState> shared(NUM_CIRCULAR_BUFFERS);
    shared[kCb].base      = base;
    shared[kCb].page_size = kPageSize;
    shared[kCb].num_pages = kNumPages;
    shared[kCb].page_mask = kNumPages - 1;  // power of two

    // ---- 1. Per-RISC isolation (the #139 race fix) -------------------------
    std::atomic<bool> writer_advanced{false};
    std::atomic<bool> reader_done{false};
    std::atomic<bool> reader_ok{false};

    std::thread writer([&] {
        __emule_cbs = shared.data();
        CHECK(__emule_cb_wr_addr(kCb) == base);          // own view starts at base
        __emule_cb_advance_wr(kCb, 3);                   // advance 3 pages
        CHECK(__emule_cb_wr_addr(kCb) == base + 3 * kPageSize);
        writer_advanced.store(true);
        while (!reader_done.load()) { std::this_thread::yield(); }
    });

    std::thread reader([&] {
        __emule_cbs = shared.data();
        while (!writer_advanced.load()) { std::this_thread::yield(); }
        // Reader never advanced: must STILL see base despite the writer's advance
        // (the old shared-index model corrupted this — the #139 bug).
        bool ok = (__emule_cb_wr_addr(kCb) == base);
        LocalCBInterface& iface = get_local_cb_interface(kCb);  // authoritative, same view
        ok = ok && ((iface.fifo_wr_ptr << cb_addr_shift) ==
                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base)));
        reader_ok.store(ok);
        reader_done.store(true);
    });

    writer.join();
    reader.join();
    CHECK(reader_ok.load());

    // ---- 2. Write-back -----------------------------------------------------
    std::thread wb([&] {
        __emule_cbs = shared.data();
        // Seat the view, then override the read pointer to page 5 (silicon-style
        // override_cb_rd_ptr writes fifo_rd_ptr in 16-byte units).
        get_local_cb_interface(kCb).fifo_rd_ptr = base16 + 5 * (kPageSize >> cb_addr_shift);
        CHECK(__emule_cb_rd_addr(kCb) == base + 5 * kPageSize);   // override took effect
        __emule_cb_advance_rd(kCb, 1);                            // advance from override
        CHECK(__emule_cb_rd_addr(kCb) == base + 6 * kPageSize);
    });
    wb.join();

    // ---- 3. Wrap -----------------------------------------------------------
    std::thread wrap([&] {
        __emule_cbs = shared.data();
        __emule_cb_advance_wr(kCb, kNumPages - 1);                // page 7
        CHECK(__emule_cb_wr_addr(kCb) == base + (kNumPages - 1) * kPageSize);
        __emule_cb_advance_wr(kCb, 1);                            // wraps to page 0
        CHECK(__emule_cb_wr_addr(kCb) == base);
        CHECK(__emule_cb_wr_addr(kCb, kNumPages) == base);        // full-ring offset -> base
    });
    wrap.join();

    munmap(mem, bytes);
    std::printf("cb_ptr: all checks passed\n");
    return 0;
}
