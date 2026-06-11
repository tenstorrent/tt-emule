// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ---------------------------------------------------------------------------
// Per-RISC circular-buffer read/write pointers — emule's single source of truth.
//
// FAITHFUL MODEL (issue #139). On silicon each RISC (DMR / DMW / Tensix) owns
// its *own* per-CB read and write pointer registers; a push_back on the writer
// RISC advances only that RISC's write pointer. The only cross-RISC CB state is
// the L1 pages_received/acked **semaphore** (emule: CBSyncState.occupied).
//
// Earlier emule kept the read/write *indices* inside the per-core, thread-shared
// CBSyncState, so a writer's cb_push_back advanced an index the reader was about
// to read as a base address — the pad_rm_sharded_stickwise race (#139), where
// reader and writer both call get_write_ptr() on the output-shard CB at startup.
//
// Here the per-RISC pointer lives in a per-thread `LocalCBInterface`
// (`__emule_local_cb`), which mirrors silicon's per-RISC CB register file. ALL
// pointer accessors — dataflow get_write_ptr/get_read_ptr (api/cb_api.h), compute
// cb_write_ptr*/cb_read_ptr_at (api/compute/common.h), and get_local_cb_interface
// (internal/cb_interface.h) — read this one storage. Because the returned
// LocalCBInterface IS the storage, kernels that write fifo_wr_ptr/fifo_rd_ptr
// (e.g. CB-reconfig paths) take effect automatically (write-back), and the next
// cb_push_back / get_*_ptr advances/reads from the written value.
//
// RESET INVARIANT (load-bearing): `thread_local` zero-initialises for every new
// std::thread. emule's runner creates a fresh std::thread per kernel per launch
// and joins it (no thread pool), so each RISC starts every launch with all CB
// pointers cleared — exactly mirroring silicon's per-RISC register reset at
// kernel launch. `fifo_page_size == 0` is the per-(thread,cb) "uninitialised"
// sentinel; first touch lazily seats the pointer at the shard base (index 0)
// from the shared CBSyncState geometry.
//
// SPSC INVARIANT: a CB has exactly one producer thread and one consumer thread
// (never same-thread produce+consume). Independent per-RISC pointers that both
// start at base and advance in FIFO lockstep, gated by the shared `occupied`
// semaphore, stay page-aligned — the same property that makes silicon's
// per-RISC rings correct.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <atomic>

#include "tt-metalium/circular_buffer_constants.h"  // NUM_CIRCULAR_BUFFERS (32 WH / 64 BH)
#include "jit_hw/emule_cb_state.h"                   // __emule_cbs (tt_emule::CBSyncState*)

// Silicon convention: CB fifo addresses are encoded in 16-byte units. The kernel
// reconstitutes a byte pointer with `<< cb_addr_shift`. (Mirror of the upstream
// LocalCBInterface fifo_{rd,wr}_ptr encoding.)
inline constexpr uint32_t cb_addr_shift = 4;

// Mirror of upstream's LocalCBInterface struct shape so kernels written against
// the real device API compile and run unchanged. fifo_{rd,wr}_ptr are 16-byte
// encoded host addresses (kernel does `<< 4`).
struct LocalCBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;  // 16-byte-encoded exclusive end = base + size; wrap when ptr >= fifo_limit.
                          // (Value matches upstream circular_buffer_init.h; upstream's struct comment
                          //  says "inclusive" but the value and the `>= fifo_limit` wrap are exclusive.)
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;

    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;

    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };

    uint32_t fifo_wr_tile_ptr;
};

// The per-RISC CB register file (single source of truth for CB pointers).
inline thread_local LocalCBInterface __emule_local_cb[NUM_CIRCULAR_BUFFERS]{};

// ---- internal helpers -----------------------------------------------------

// 16-byte-encoded base address of CB `cb_id`.
inline uint32_t __emule_cb_base16(uint32_t cb_id) {
    return static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(__emule_cbs[cb_id].base) >> cb_addr_shift);
}

// Total ring span of CB `cb_id` in 16-byte units (page_size * num_pages).
inline uint32_t __emule_cb_span16(uint32_t cb_id) {
    const auto& g = __emule_cbs[cb_id];
    return (g.page_size * g.num_pages) >> cb_addr_shift;
}

// Lazily seat this thread's pointers at the shard base (index 0) on first touch,
// mirroring the silicon per-RISC register reset at kernel launch. No-op once
// seated (so kernel writes to fifo_{wr,rd}_ptr persist) and when the CB is unset.
inline void __emule_cb_view_init(uint32_t cb_id) {
    auto& v = __emule_local_cb[cb_id];
    const auto& g = __emule_cbs[cb_id];
    if (v.fifo_page_size != 0 || g.page_size == 0) {
        return;
    }
    const uint32_t base16 = __emule_cb_base16(cb_id);
    const uint32_t span16 = __emule_cb_span16(cb_id);
    // Units match emule's established LocalCBInterface convention (kernels read
    // these directly): fifo_page_size / fifo_size in BYTES; fifo_{rd,wr}_ptr /
    // fifo_limit / fifo_wr_tile_ptr 16-byte-encoded (kernel does `<< 4`).
    v.fifo_page_size = g.page_size;                   // BYTES
    v.fifo_num_pages = g.num_pages;
    v.fifo_size      = g.page_size * g.num_pages;     // BYTES
    v.fifo_limit     = base16 + span16;               // 16-byte units
    v.fifo_rd_ptr    = base16;                        // 16-byte units
    v.fifo_wr_ptr    = base16;
    v.fifo_wr_tile_ptr = base16;
}

// Wrap a 16-byte-encoded pointer `p` back into [base16, base16+span16) using
// modulo (matches the old CBSyncState cb_sync_*_ptr semantics for ANY offset,
// not just off < num_pages).
inline uint32_t __emule_cb_wrap(uint32_t p, uint32_t base16, uint32_t span16) {
    return span16 ? base16 + ((p - base16) % span16) : p;
}

// Byte pointer for the write side at `off` pages past this thread's write ptr.
inline uint8_t* __emule_cb_wr_addr(uint32_t cb_id, uint32_t off = 0) {
    __emule_cb_view_init(cb_id);
    const uint32_t p = __emule_cb_wrap(
        __emule_local_cb[cb_id].fifo_wr_ptr + off * (__emule_cbs[cb_id].page_size >> cb_addr_shift),
        __emule_cb_base16(cb_id), __emule_cb_span16(cb_id));
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(p) << cb_addr_shift);
}

// Byte pointer for the read side at `off` pages past this thread's read ptr.
inline uint8_t* __emule_cb_rd_addr(uint32_t cb_id, uint32_t off = 0) {
    __emule_cb_view_init(cb_id);
    const uint32_t p = __emule_cb_wrap(
        __emule_local_cb[cb_id].fifo_rd_ptr + off * (__emule_cbs[cb_id].page_size >> cb_addr_shift),
        __emule_cb_base16(cb_id), __emule_cb_span16(cb_id));
    return reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(p) << cb_addr_shift);
}

// Advance this thread's write/read pointer by `n` pages, wrapping at the ring end.
inline void __emule_cb_advance_wr(uint32_t cb_id, uint32_t n) {
    __emule_cb_view_init(cb_id);
    auto& v = __emule_local_cb[cb_id];
    v.fifo_wr_ptr = __emule_cb_wrap(
        v.fifo_wr_ptr + n * (__emule_cbs[cb_id].page_size >> cb_addr_shift),
        __emule_cb_base16(cb_id), __emule_cb_span16(cb_id));
    v.fifo_wr_tile_ptr = v.fifo_wr_ptr;
}

inline void __emule_cb_advance_rd(uint32_t cb_id, uint32_t n) {
    __emule_cb_view_init(cb_id);
    auto& v = __emule_local_cb[cb_id];
    v.fifo_rd_ptr = __emule_cb_wrap(
        v.fifo_rd_ptr + n * (__emule_cbs[cb_id].page_size >> cb_addr_shift),
        __emule_cb_base16(cb_id), __emule_cb_span16(cb_id));
}
