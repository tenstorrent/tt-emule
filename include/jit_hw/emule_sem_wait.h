// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Host-scheduling helpers shared by every emule semaphore spin-wait.
//
// Silicon waits on a semaphore by busy-spinning on an L1 word
// (noc_semaphore_wait: `do { invalidate_l1_cache(); } while (*sem != val);`)
// with no backoff and no timeout: every Tensix core is its own hardware, so a
// spin costs nothing and starves no one. emule runs each core as an OS thread
// and is heavily oversubscribed on a typical host, so a naive poll storm starves
// the producer threads a waiter depends on. These helpers cede the host CPU as a
// wait lengthens and bound a genuine deadlock with a wall-clock watchdog. They
// change host scheduling only, never emulated values or happens-before ordering,
// so all three spin-wait paths (noc_semaphore_wait/_min, Semaphore,
// ckernel::Semaphore) share them verbatim rather than each carrying a copy.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sched.h>
#include <time.h>
#include <unistd.h>

// Escalating backoff for a spin-wait that has already looped `spins` times:
// busy-spin for fast wakeup, then yield, then sleep progressively longer so a
// long-waiting thread cedes its host CPU to the producer it is waiting on.
// Uses usleep only, never <chrono>, which on libstdc++ pulls <format>/iostream
// and adds ~1s of frontend parse per JIT kernel (same rule as emule_wait.h).
inline void __emule_sem_backoff(uint64_t spins) {
    if (spins < 64) {
        // busy-spin for fast wakeup
    } else if (spins < 2048) {
        sched_yield();
    } else if (spins < 32768) {
        usleep(10);
    } else {
        usleep(200);
    }
}

// Monotonic host time in nanoseconds. clock_gettime, not std::chrono (see above).
inline uint64_t __emule_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

// Hang-watchdog timeout in nanoseconds, read once from TT_EMULE_SEM_TIMEOUT_SEC.
// A live peer can be starved past any fixed spin count without a real deadlock,
// so the watchdog is a wall-clock deadline, not a spin count: its meaning does
// not shift with the backoff tier. 0 disables the watchdog (never abort). A
// malformed value falls back to the default and warns once, so a typo can never
// silently turn hang detection off.
inline uint64_t __emule_sem_timeout_ns() {
    static uint64_t ns = []() -> uint64_t {
        constexpr uint64_t kDefaultSec = 120;
        const char* e = std::getenv("TT_EMULE_SEM_TIMEOUT_SEC");
        if (e == nullptr || *e == '\0') {
            return kDefaultSec * 1'000'000'000ULL;
        }
        char* end = nullptr;
        unsigned long long v = std::strtoull(e, &end, 10);
        if (*end != '\0') {
            fprintf(stderr,
                    "EMULE WARN: TT_EMULE_SEM_TIMEOUT_SEC=\"%s\" is not a number; "
                    "using default %llus\n",
                    e, (unsigned long long)kDefaultSec);
            return kDefaultSec * 1'000'000'000ULL;
        }
        return static_cast<uint64_t>(v) * 1'000'000'000ULL;  // 0 => disabled
    }();
    return ns;
}

// True once a wait that started at `start_ns` (from __emule_now_ns()) has run
// past the watchdog timeout; always false when the watchdog is disabled. The
// caller prints its own site-specific diagnostic and aborts.
inline bool __emule_sem_watchdog_expired(uint64_t start_ns) {
    uint64_t timeout = __emule_sem_timeout_ns();
    return timeout != 0 && (__emule_now_ns() - start_ns) > timeout;
}
