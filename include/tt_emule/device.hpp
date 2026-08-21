// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cb_sync_state.hpp"
#include "dfb_sync_state.hpp"
#include "tile_counter.hpp"
#include "wall_clock.hpp"  // __emule_wall_clock_lo_addr/_hi_addr (RISCV_DEBUG_REG_WALL_CLOCK_* backing)
#include "jit_hw/internal/emule_cb_constants.h"  // EMULE_MAX_CBS
#include "jit_hw/internal/emule_core_state.h"  // tt_emule::CoreState (per-core coords)
#include "worker_l1_mmap.hpp"                  // __emule_mmap_worker_l1 (worker L1 backing)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <sys/mman.h>

namespace tt_emule {

// Role of a Core — determines how its mmap'd region is used.
enum class CoreRole { WORKER, DRAM };

struct CoreCoord {
    size_t x;
    size_t y;
    bool operator==(const CoreCoord& o) const { return x == o.x && y == o.y; }
};

class Core {
public:
    static constexpr size_t L1_SIZE = 1024 * 1024; // 1 MB
    // Public compatibility alias; storage derives from the shared emule limit.
    static constexpr size_t MAX_CBS = EMULE_MAX_CBS;

    // Default constructor: WORKER role, 1 MB L1.
    explicit Core(CoreCoord coord) : coord_(coord) {
        mmap_region(L1_SIZE);
    }

    // Role-aware constructor: mmap mem_size bytes.
    Core(CoreCoord coord, CoreRole role, size_t mem_size)
        : coord_(coord), role_(role), l1_size_(mem_size) {
        mmap_region(mem_size);
    }

    // Construct with external memory (no mmap, no munmap).
    // Used by the memory bridge to wrap EmulatedChip's backing store.
    Core(CoreCoord coord, uint8_t* external_l1, size_t l1_size)
        : coord_(coord), owns_l1_(false), l1_size_(l1_size) {
        l1_ = external_l1;
        l1_base_ = reinterpret_cast<uintptr_t>(l1_);
    }

    ~Core() {
        if (owns_l1_ && l1_) munmap(l1_, l1_size_);
    }

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    CoreRole  role()  const { return role_; }

    // Per-core NOC / logical coordinates. The per-thread ctx borrows a pointer
    // via __emule_self->core; all RISC threads on this core share these.
    CoreState& core_state() { return core_state_; }

    // Accepts uint64_t so DRAM banks (≤ 4 GB on BH, ≤ 2 GB on WH views) can be
    // addressed without uint32 truncation. Loud bounds check converts what was
    // previously silent UB into a clear failure during testing.
    uint8_t* l1_ptr(uint64_t offset) {
        if (offset >= l1_size_) {
            std::fprintf(stderr,
                "[EMULE] Core::l1_ptr OOB: role=%s coord=(%zu,%zu) offset=0x%llx size=0x%zx\n",
                role_ == CoreRole::DRAM ? "DRAM" : "WORKER",
                coord_.x, coord_.y,
                static_cast<unsigned long long>(offset),
                l1_size_);
            std::abort();
        }
        return l1_ + offset;
    }

    // Raw pointer to start of memory region (L1 or DRAM backing).
    uint8_t* l1_data() { return l1_; }

    // Size of the memory region (regardless of role).
    size_t l1_size() const { return l1_size_; }

    // Host address of the L1 base.
    uintptr_t l1_base_addr() const { return l1_base_; }

    // Bump allocate `bytes` from L1; returns a 0-based L1 offset (L1 offset
    // model — rebased to a host pointer at the deref via bridge_l1).
    // The bump region is the L1 below tt-metal's l1_unreserved_base, which is
    // firmware-reserved on silicon and unused in emule — so allocations here
    // don't collide with anything tt-metal's allocator hands out.
    uint32_t l1_alloc(size_t bytes) {
        if (l1_bump_ + bytes > l1_size_)
            throw std::runtime_error("L1 OOM");
        uint32_t addr = static_cast<uint32_t>(l1_bump_);
        l1_bump_ += bytes;
        return addr;
    }

    // Reset the bump allocator between program runs.  Only meaningful when
    // l1_alloc has actually been called (DFB fallback path, Quasar-only); on
    // WH/BH the bump never grows so this is a no-op.
    void reset_l1_bump() { l1_bump_ = 0; }

    // ---- CB sync state array (for JIT kernel threads) ----

    CBSyncState* cb_sync_array() { return cb_sync_states_; }

    void init_cb_sync(uint32_t idx, uint8_t* base, uint32_t page_size, uint32_t num_pages,
                      bool globally_allocated = false) {
        if (idx >= MAX_CBS) return;
        auto& s = cb_sync_states_[idx];
        s.base      = base;
        s.page_size = page_size;
        s.num_pages = num_pages;
        s.page_mask = (num_pages > 0 && (num_pages & (num_pages - 1)) == 0) ? num_pages - 1 : 0;
        s.globally_allocated = globally_allocated;
        s.occupied  = 0;
        // Per-RISC read/write pointers reset per-thread at kernel launch
        // (thread_local zero-init in jit_hw/internal/emule_cb_ptr.h), not here.
    }

    void reset_cb_sync() {
        for (auto& s : cb_sync_states_) {
            s.base      = nullptr;
            s.page_size = 0;
            s.num_pages = 0;
            s.page_mask = 0;
            s.globally_allocated = false;
            s.occupied  = 0;
        }
    }

    // ---- DFB / Tile Counter infrastructure (Quasar) ----

    void init_tile_counters(uint32_t num_neos) {
        tile_counters_ = std::make_unique<TileCounterArray>(num_neos);
    }

    TileCounterArray* tile_counters() { return tile_counters_.get(); }

    void init_dfb_sync(uint32_t idx, uint8_t* base, uint32_t entry_size,
                       uint32_t num_entries, uint32_t capacity) {
        if (idx >= MAX_DFBS) return;
        auto& s = dfb_sync_states_[idx];
        s.base            = base;
        s.entry_size      = entry_size;
        s.num_entries     = num_entries;
        s.capacity        = capacity;
        s.stride_in_entries = 1;
    }

    void reset_dfb_sync() {
        for (auto& s : dfb_sync_states_) {
            s.base = nullptr;
            s.entry_size = 0;
            s.num_entries = 0;
            s.capacity = 0;
            s.stride_in_entries = 1;
        }
        if (tile_counters_) tile_counters_->reset_all();
    }

private:
    void mmap_region(size_t size) {
        l1_size_ = size;
        // Both roles are plain 64-bit mmaps: worker L1 is offset-addressed (kernels deref via
        // bridge_l1) and DRAM via bridge functions, so neither backing is placement-constrained.
        // Worker L1 goes through the helper (MAP_NORESERVE, lazy-faulted for large multi-chip pools).
        void* p = (role_ == CoreRole::WORKER)
                      ? __emule_mmap_worker_l1(size)
                      : mmap(nullptr, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("mmap for Core memory failed");
        l1_ = static_cast<uint8_t*>(p);
        l1_base_ = reinterpret_cast<uintptr_t>(l1_);
        // MAP_ANONYMOUS guarantees zero-filled pages; no memset needed.
    }

    CoreCoord coord_;
    CoreRole  role_    = CoreRole::WORKER;
    bool      owns_l1_ = true;  // false when using external memory
    size_t    l1_size_ = L1_SIZE;
    uint8_t*  l1_      = nullptr;
    uintptr_t l1_base_ = 0;
    size_t    l1_bump_ = 0;
    CoreState core_state_;  // per-core NOC/logical coords (see emule_thread_ctx.h)
    CBSyncState cb_sync_states_[MAX_CBS] = {};
    // Quasar DFB state
    std::unique_ptr<TileCounterArray> tile_counters_;
    DFBSyncState dfb_sync_states_[MAX_DFBS] = {};
};

} // namespace tt_emule
