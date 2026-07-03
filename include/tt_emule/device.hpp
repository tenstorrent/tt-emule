// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "cb_sync_state.hpp"
#include "dfb_sync_state.hpp"
#include "tile_counter.hpp"
#include "jit_hw/internal/emule_core_state.h"  // tt_emule::CoreState (per-core coords)
#include "low4g_mmap.hpp"                      // __emule_mmap_low4g (worker L1 in low 4 GB)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <sys/mman.h>

namespace tt_emule {

// Optional caller-context hints that Core::l1_ptr will include in the OOB
// message before aborting. Each hint defaults to a distinct "unset" sentinel
// (UINT64_MAX for the NOC address, UINT32_MAX for coordinates, nullptr for
// the tag) so that valid values like `self=(0,0)` — which is a real caller
// coord in single-core / one-worker tests — are not suppressed. When a hint
// equals its sentinel it is omitted from the message; otherwise it appears.
//
// Intended use: a caller that has richer context than Core::l1_ptr sees (a
// full NOC address, the calling core's coord, a symbolic tag like a kernel
// name or a page_id) sets these thread_locals before dispatching L1 accesses,
// so a bounds-check abort points back to WHO made the request instead of only
// WHERE it landed. A typical wrapper:
//
//     tt_emule::l1_ptr_hint_noc_addr = noc_addr;
//     tt_emule::l1_ptr_hint_self_x   = my_x;
//     tt_emule::l1_ptr_hint_self_y   = my_y;
//     uint8_t* p = target_core->l1_ptr(offset);
//     tt_emule::l1_ptr_hint_noc_addr = tt_emule::L1_PTR_HINT_NOC_ADDR_UNSET;
//     tt_emule::l1_ptr_hint_self_x   = tt_emule::L1_PTR_HINT_COORD_UNSET;
//     tt_emule::l1_ptr_hint_self_y   = tt_emule::L1_PTR_HINT_COORD_UNSET;
//
// Motivation: on a matmul mcast_in1 OOB, the original message told us the
// target coord + offset but not the source coord or the raw NOC address. That
// forced hand-instrumentation of __emule_resolve_noc_addr / TensorAccessor to
// identify the offending write. Making these hints a first-class part of the
// abort output keeps the next debug session to minutes instead of an hour.
inline constexpr uint64_t L1_PTR_HINT_NOC_ADDR_UNSET = UINT64_MAX;
inline constexpr uint32_t L1_PTR_HINT_COORD_UNSET    = UINT32_MAX;
inline thread_local uint64_t l1_ptr_hint_noc_addr = L1_PTR_HINT_NOC_ADDR_UNSET;
inline thread_local uint32_t l1_ptr_hint_self_x   = L1_PTR_HINT_COORD_UNSET;
inline thread_local uint32_t l1_ptr_hint_self_y   = L1_PTR_HINT_COORD_UNSET;
inline thread_local const char* l1_ptr_hint_tag   = nullptr;

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
    static constexpr size_t MAX_CBS = 32;

    // Default constructor: WORKER role, 1 MB L1 mmap'd below 4 GB.
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
        l1_base_ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_));
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
                "[EMULE] Core::l1_ptr OOB: role=%s coord=(%zu,%zu) offset=0x%llx size=0x%zx",
                role_ == CoreRole::DRAM ? "DRAM" : "WORKER",
                coord_.x, coord_.y,
                static_cast<unsigned long long>(offset),
                l1_size_);
            // Emit caller-context hints if any have been populated. Each hint
            // is checked independently against its own sentinel so partial
            // context still prints and legitimately-zero coords / noc_addrs
            // are not suppressed.
            if (l1_ptr_hint_noc_addr != L1_PTR_HINT_NOC_ADDR_UNSET) {
                std::fprintf(stderr,
                    " noc_addr=0x%llx",
                    static_cast<unsigned long long>(l1_ptr_hint_noc_addr));
            }
            if (l1_ptr_hint_self_x != L1_PTR_HINT_COORD_UNSET ||
                l1_ptr_hint_self_y != L1_PTR_HINT_COORD_UNSET) {
                std::fprintf(stderr,
                    " self=(%u,%u)",
                    l1_ptr_hint_self_x, l1_ptr_hint_self_y);
            }
            if (l1_ptr_hint_tag != nullptr) {
                std::fprintf(stderr, " tag=%s", l1_ptr_hint_tag);
            }
            std::fprintf(stderr, "\n");
            std::abort();
        }
        return l1_ + offset;
    }

    // Raw pointer to start of memory region (L1 or DRAM backing).
    uint8_t* l1_data() { return l1_; }

    // Size of the memory region (regardless of role).
    size_t l1_size() const { return l1_size_; }

    // 32-bit absolute address of the L1 base (valid if mmap succeeded below 4 GB).
    uint32_t l1_base_addr() const { return l1_base_; }

    // Bump allocate `bytes` from L1; returns absolute host address.
    // The bump region is the L1 below tt-metal's l1_unreserved_base, which is
    // firmware-reserved on silicon and unused in emule — so allocations here
    // don't collide with anything tt-metal's allocator hands out.
    uint32_t l1_alloc(size_t bytes) {
        if (l1_bump_ + bytes > l1_size_)
            throw std::runtime_error("L1 OOM");
        uint32_t addr = l1_base_ + static_cast<uint32_t>(l1_bump_);
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
        // Worker cores must be uint32-addressable (kernels deref CB pointers directly),
        // so they live in the low 4 GB (low4g_mmap.hpp: MAP_32BIT, then a [2GB,4GB) gap).
        // DRAM cores are accessed via bridge functions (full 64-bit pointers), so they
        // use a regular mmap (anywhere) to avoid consuming the scarce low-4 GB space.
        void* p = (role_ == CoreRole::WORKER)
                      ? __emule_mmap_low4g(size)
                      : mmap(nullptr, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("mmap for Core memory failed");
        l1_ = static_cast<uint8_t*>(p);
        l1_base_ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_));
        // MAP_ANONYMOUS guarantees zero-filled pages; no memset needed.
    }

    CoreCoord coord_;
    CoreRole  role_    = CoreRole::WORKER;
    bool      owns_l1_ = true;  // false when using external memory
    size_t    l1_size_ = L1_SIZE;
    uint8_t*  l1_      = nullptr;
    uint32_t  l1_base_ = 0;
    size_t    l1_bump_ = 0;
    CoreState core_state_;  // per-core NOC/logical coords (see emule_thread_ctx.h)
    CBSyncState cb_sync_states_[MAX_CBS] = {};
    // Quasar DFB state
    std::unique_ptr<TileCounterArray> tile_counters_;
    DFBSyncState dfb_sync_states_[MAX_DFBS] = {};
};

} // namespace tt_emule
