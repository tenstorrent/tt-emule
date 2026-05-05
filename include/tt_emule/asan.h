#pragma once

// Lightweight wrappers around AddressSanitizer's manual poisoning API.
// Calls expand to the real __asan_* primitives when the host binary was
// compiled with -fsanitize=address; otherwise they compile to no-ops, so
// the call sites can stay in the source unconditionally without runtime
// cost on non-ASan builds.
//
// Usage:
//     EMULE_ASAN_POISON(ptr, size);
//     EMULE_ASAN_UNPOISON(ptr, size);
//     if (EMULE_ASAN_REGION_IS_POISONED(ptr, size)) { ... }

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define EMULE_ASAN_ENABLED 1
#  endif
#endif
#if !defined(EMULE_ASAN_ENABLED) && defined(__SANITIZE_ADDRESS__)
#  define EMULE_ASAN_ENABLED 1
#endif

#if defined(EMULE_ASAN_ENABLED)
#  include <sanitizer/asan_interface.h>
#  define EMULE_ASAN_POISON(p, n)              __asan_poison_memory_region((p), (n))
#  define EMULE_ASAN_UNPOISON(p, n)            __asan_unpoison_memory_region((p), (n))
#  define EMULE_ASAN_REGION_IS_POISONED(p, n)  (__asan_region_is_poisoned((void*)(p), (n)) != nullptr)
#else
#  define EMULE_ASAN_POISON(p, n)              ((void)0)
#  define EMULE_ASAN_UNPOISON(p, n)            ((void)0)
#  define EMULE_ASAN_REGION_IS_POISONED(p, n)  (false)
#endif

// ---- Bounds-check helpers (live regardless of ASan) ----
//
// Bridge functions (NOC resolve, DRAM ptr, ...) call __emule_bounds_fail()
// when an offset+size exceeds the target region. Default: print a diagnostic
// and std::abort() — same style as __emule_dst_check in compute/common.h.
// Set TT_EMULE_ASAN_WARN_ONLY=1 in the environment to log-and-continue
// (useful for first-pass triage when running regression).

#include <cstdio>
#include <cstdlib>

inline bool __emule_asan_warn_only() {
    static const bool warn = []() {
        const char* s = std::getenv("TT_EMULE_ASAN_WARN_ONLY");
        return s != nullptr && s[0] != '\0' && !(s[0] == '0' && s[1] == '\0');
    }();
    return warn;
}

inline void __emule_bounds_fail(const char* caller, const char* msg,
                                unsigned long long offset, unsigned long long size,
                                unsigned long long limit) {
    std::fprintf(stderr,
                 "[EMULE] %s: out-of-bounds — %s offset=0x%llx size=0x%llx limit=0x%llx\n",
                 caller, msg, offset, size, limit);
    std::fflush(stderr);
    if (__emule_asan_warn_only()) return;
    // Use _Exit (clean process termination, non-zero) rather than abort
    // (SIGABRT). Mirrors ASan's default die-on-error behavior and lets CTest
    // distinguish an expected bounds-check fire from a crash.
    std::_Exit(2);
}
