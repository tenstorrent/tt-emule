// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// JIT emulation stub for api/dataflow/noc_semaphore.h.
//
// Adapted from jit_hw/experimental/noc_semaphore.h — same atomic implementation
// with the 'namespace experimental {}' wrapper removed and the API surface
// extended to match the promoted api/dataflow/noc_semaphore.h:
//   - template param is now ProgrammableCoreType (stubbed enum below)
//   - set_multicast gains a NocOptions template param (MCAST_INCL_SRC)
//   - inc_multicast argument order matches real API (value, num_dests)
//   - up(const Noc&, ...) delegates to noc_semaphore_inc (already in dataflow_api.h)
//
// Semaphore operations use C++ atomics for cross-thread visibility.
// Spin-waits share the backoff + wall-clock hang watchdog in emule_sem_wait.h.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "jit_hw/emule_sem_wait.h"
#include "jit_hw/api/dataflow/noc.h"

extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
extern thread_local uint8_t* __emule_bridge_l1;

// ---- ProgrammableCoreType stub ----
// Normally defined in tt-metal arch headers. In emulation all cores are
// treated identically so the template parameter is accepted but ignored.
#ifndef PROGRAMMABLE_CORE_TYPE_DEFINED
#define PROGRAMMABLE_CORE_TYPE_DEFINED
enum class ProgrammableCoreType : uint8_t { TENSIX = 0, ACTIVE_ETH = 1, IDLE_ETH = 2 };
#endif

// ---- class Semaphore ----

template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX>
class Semaphore {
    // Lets relay_unicast / relay_multicast read dst_sem's private members
    // without a public accessor — matches silicon's friend declaration.
    template <ProgrammableCoreType OT>
    friend class Semaphore;

public:
    // core_type is accepted for API compatibility but ignored in emulation —
    // get_semaphore is non-templated for two-phase template lookup reasons
    // (see comment in dataflow_api.h alongside the get_semaphore definition).
    explicit Semaphore(uint32_t semaphore_id)
        : local_l1_addr_(get_semaphore(semaphore_id)) {
        l1_offset_ = static_cast<uint32_t>(
            local_l1_addr_ - reinterpret_cast<uintptr_t>(__emule_bridge_l1));
    }

    // ---- Local operations ----

    void up(uint32_t value) {
        atom()->fetch_add(value, std::memory_order_release);
    }

    // Remote atomic increment via NOC address.
    void up(const Noc& noc, uint32_t noc_x, uint32_t noc_y,
            uint32_t value, uint8_t vc = NOC_UNICAST_WRITE_VC) {
        uint64_t noc_addr = (static_cast<uint64_t>(noc_y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
                            (static_cast<uint64_t>(noc_x) << NOC_ADDR_LOCAL_BITS) |
                            uint64_t(l1_offset_);
        noc_semaphore_inc(noc_addr, value, noc.get_noc_id(), vc);
    }

    void down(uint32_t value) {
        auto* a = atom();
        uint64_t spins = 0;
        uint64_t start_ns = __emule_now_ns();
        while (a->load(std::memory_order_acquire) < value) {
            __emule_sem_backoff(spins++);
            if (__emule_sem_watchdog_expired(start_ns)) {
                fprintf(stderr,
                    "EMULE HANG: Semaphore::down(%u) stuck at %u after %llu spins\n",
                    value, a->load(std::memory_order_relaxed),
                    (unsigned long long)spins);
                std::abort();
            }
        }
        a->fetch_sub(value, std::memory_order_release);
    }

    void wait(uint32_t target) {
        auto* a = atom();
        // Mirror the free-function noc_semaphore_wait (dataflow_api.h): for a
        // monotonic count-up handshake (target > 0), "reached" means >= target,
        // not exact ==. emule's increments are zero-latency atomics, so a peer can
        // advance the counter past `target` between two of our polls and an
        // equality wait would miss it and spin to the watchdog. (Silicon paces
        // increments over the NOC, so == never misses there.) For the VALID->0
        // release toggle (target == 0) keep exact equality; a count-up target is
        // never 0, so the split is unambiguous.
        auto reached = [target](uint32_t cur) { return target > 0 ? cur >= target : cur == target; };
        uint64_t spins = 0;
        uint64_t start_ns = __emule_now_ns();
        while (!reached(a->load(std::memory_order_acquire))) {
            __emule_sem_backoff(spins++);
            if (__emule_sem_watchdog_expired(start_ns)) {
                fprintf(stderr,
                    "EMULE HANG: Semaphore::wait(%u) stuck at %u after %llu spins\n",
                    target, a->load(std::memory_order_relaxed),
                    (unsigned long long)spins);
                std::abort();
            }
        }
    }

    void wait_min(uint32_t min_val) {
        auto* a = atom();
        uint64_t spins = 0;
        uint64_t start_ns = __emule_now_ns();
        while (a->load(std::memory_order_acquire) < min_val) {
            __emule_sem_backoff(spins++);
            if (__emule_sem_watchdog_expired(start_ns)) {
                fprintf(stderr,
                    "EMULE HANG: Semaphore::wait_min(%u) stuck at %u after %llu spins\n",
                    min_val, a->load(std::memory_order_relaxed),
                    (unsigned long long)spins);
                std::abort();
            }
        }
    }

    void set(uint32_t value) {
        atom()->store(value, std::memory_order_release);
    }

    // ---- Multicast operations ----
    // NocOptions::MCAST_INCL_SRC: sender is included in the rectangle — both modes
    // funnel through __emule_multicast_write which visits all cores in the
    // rectangle (including the local core if its coordinates fall inside).

    template <NocOptions opts = NocOptions::DEFAULT>
    void set_multicast(
        const Noc& noc,
        uint32_t noc_x_start, uint32_t noc_y_start,
        uint32_t noc_x_end,   uint32_t noc_y_end,
        uint32_t num_dests, bool linked = false) {
        uint64_t mcast_addr =
            (static_cast<uint64_t>(noc_x_start) << (NOC_ADDR_LOCAL_BITS + 2 * NOC_ADDR_NODE_ID_BITS)) |
            (static_cast<uint64_t>(noc_y_start) << (NOC_ADDR_LOCAL_BITS + 3 * NOC_ADDR_NODE_ID_BITS)) |
            (static_cast<uint64_t>(noc_x_end)   << NOC_ADDR_LOCAL_BITS) |
            (static_cast<uint64_t>(noc_y_end)   << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
            uint64_t(l1_offset_);
        uint32_t val = atom()->load(std::memory_order_acquire);
        __emule_multicast_write(mcast_addr,
                                reinterpret_cast<const uint8_t*>(&val),
                                sizeof(uint32_t),
                                has_flag(opts, NocOptions::MCAST_INCL_SRC));
    }

    // Argument order matches real api/dataflow/noc_semaphore.h: (value, num_dests).
    void inc_multicast(
        const Noc& noc,
        uint32_t noc_x_start, uint32_t noc_y_start,
        uint32_t noc_x_end,   uint32_t noc_y_end,
        uint32_t value, uint32_t num_dests) {
        uint32_t lo_x = (noc_x_start < noc_x_end) ? noc_x_start : noc_x_end;
        uint32_t hi_x = (noc_x_start < noc_x_end) ? noc_x_end : noc_x_start;
        uint32_t lo_y = (noc_y_start < noc_y_end) ? noc_y_start : noc_y_end;
        uint32_t hi_y = (noc_y_start < noc_y_end) ? noc_y_end : noc_y_start;
        for (uint32_t x = lo_x; x <= hi_x; x++) {
            for (uint32_t y = lo_y; y <= hi_y; y++) {
                uint64_t noc_addr =
                    (static_cast<uint64_t>(y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) |
                    (static_cast<uint64_t>(x) << NOC_ADDR_LOCAL_BITS) |
                    uint64_t(l1_offset_);
                uint8_t* ptr = __emule_resolve_noc_addr(noc_addr);
                if (ptr) {
                    reinterpret_cast<std::atomic<uint32_t>*>(ptr)->fetch_add(
                        value, std::memory_order_release);
                } else {
                    fprintf(stderr, "EMULE WARN: Semaphore::inc_multicast (%u,%u) "
                            "offset=0x%x failed to resolve\n", x, y, l1_offset_);
                }
            }
        }
    }

    // Relay this sem's local value to a *different* sem on a remote core.
    // Mirrors silicon: read our own sem, 4-byte write to dst_sem.get_l1_addr()
    // on the target core via noc_semaphore_set_remote.
    template <ProgrammableCoreType dst_core_type = core_type>
    void relay_unicast(const Noc& noc, const Semaphore<dst_core_type>& dst_sem,
                       uint32_t noc_x, uint32_t noc_y) {
        uint64_t dst_noc_addr = ::get_noc_addr(noc_x, noc_y,
                                               dst_sem.get_l1_addr(), noc.get_noc_id());
        noc_semaphore_set_remote(get_l1_addr(), dst_noc_addr, noc.get_noc_id());
    }

    // Multicast variant: relay this sem's local value to a different sem at the
    // same offset on every core in the rectangle. Mirrors silicon's branch on
    // NocOptions::MCAST_INCL_SRC between the loopback and non-loopback paths.
    template <NocOptions opts = NocOptions::DEFAULT,
              ProgrammableCoreType dst_core_type = core_type>
    void relay_multicast(const Noc& noc, const Semaphore<dst_core_type>& dst_sem,
                         uint32_t noc_x_start, uint32_t noc_y_start,
                         uint32_t noc_x_end,   uint32_t noc_y_end,
                         uint32_t num_dests, bool linked = false) {
        uint64_t mcast_addr = ::get_noc_multicast_addr(
            noc_x_start, noc_y_start, noc_x_end, noc_y_end,
            dst_sem.get_l1_addr(), noc.get_noc_id());
        if constexpr (has_flag(opts, NocOptions::MCAST_INCL_SRC)) {
            noc_semaphore_set_multicast_loopback_src(get_l1_addr(), mcast_addr,
                                                     num_dests, linked,
                                                     noc.get_noc_id());
        } else {
            noc_semaphore_set_multicast(get_l1_addr(), mcast_addr,
                                        num_dests, linked, noc.get_noc_id());
        }
    }

    // L1 offset accessor — returns the L1-relative offset used to construct
    // NOC addresses (matches silicon's get_l1_addr() semantics). Not the host
    // pointer (that's local_l1_addr_, private).
    uint32_t get_l1_addr() const { return l1_offset_; }

private:
    uintptr_t local_l1_addr_;
    uint32_t  l1_offset_;  // L1 offset of semaphore (for NOC address construction)

    std::atomic<uint32_t>* atom() const {
        return reinterpret_cast<std::atomic<uint32_t>*>(local_l1_addr_);
    }
};
