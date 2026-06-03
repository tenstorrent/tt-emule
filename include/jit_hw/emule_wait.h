// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Condition-variable wait wrapper for emule CB/DFB synchronization.
//
// Default: block on the predicate with no timeout (cv.wait). This keeps the
// kernel-facing include graph free of <chrono>, which on libstdc++ transitively
// pulls the C++20 <format>/iostream machinery (~1s of frontend parse per JIT
// kernel). Silicon kernels never include <chrono>, so this is zero divergence.
//
// Define EMULE_WAIT_TIMEOUT to restore the bounded wait_for + per-op hang
// diagnostic: this returns false on timeout and the caller's fprintf+abort runs.
// Use it when debugging a deadlock; off by default for fast JIT compile.

#include <condition_variable>
#include <mutex>
#if defined(EMULE_WAIT_TIMEOUT)
#include <chrono>
#endif

template <class Pred>
inline bool __emule_cv_wait(std::condition_variable& cv,
                            std::unique_lock<std::mutex>& lk,
                            int timeout_sec,
                            Pred pred) {
#if defined(EMULE_WAIT_TIMEOUT)
    return cv.wait_for(lk, std::chrono::seconds(timeout_sec), pred);
#else
    (void)timeout_sec;
    cv.wait(lk, pred);
    return true;
#endif
}
