// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Unified diagnostic trace for every emule [ASAN ERROR]. When a sanitizer is
// about to abort, it calls __emule_asan_panic() instead of abort(): that prints
//   - which kernel + core/processor the failure happened on (when in kernel
//     context), and
//   - a symbolized backtrace — for frames inside a JIT'd kernel .so this resolves
//     to kernel-source file:line (the kernel is built with -g under ASAN; see
//     jit_compile_kernel) via llvm-symbolizer / addr2line.
//
// __emule_asan_panic is a single real (non-inline, extern "C") symbol: kernel
// .so files and the host-API translation unit only see the *declaration* and
// resolve it at link/dlopen (like the other __emule_* symbols), so neither has
// to pull <execinfo.h>/<dlfcn.h> or the tt-emule include path. The *definition*
// is emitted by the one runtime TU that defines EMULE_ASAN_IMPLEMENTATION before
// including this header (emulated_program_runner.cpp for libtt_metal, and
// kernel_runner.cpp for the standalone tt-emule runtime).

#include <cstdint>

// Human-readable source path of the kernel currently running on this thread, or
// nullptr when no kernel is on the stack (host-API checks). Set per launch.
extern thread_local const char* __emule_kernel_name;

// Report one [ASAN ERROR] and abort. `fmt`/args are the printf-style error line
// (printed verbatim, so existing message text and test regexes are unchanged);
// the unified context+backtrace follows. The whole report is emitted under one
// lock, exactly once, even when every core-thread trips a check at the same time
// (see the definition). Pass nullptr to print only the context+backtrace.
extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...);

#ifdef EMULE_ASAN_IMPLEMENTATION

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

// Identity thread-locals read by the trace (defined in the same runtime TUs).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];
extern thread_local uint32_t __emule_logical_x;
extern thread_local uint32_t __emule_logical_y;
extern thread_local uint8_t __processor_id;
extern thread_local uint8_t __emule_neo_id;
extern thread_local uint8_t __emule_trisc_id;

// Resolve one instruction address (as a module + file-relative offset) to a
// "func at file:line" string using llvm-symbolizer (preferred — matches the
// clang toolchain), falling back to addr2line. Both are asked for inline frames
// and pretty-printed; the multi-line result is flattened to one line with " <- "
// between inlined frames. Returns false if neither tool resolved real debug info.
static bool __emule_asan_symbolize(const char* module, uintptr_t file_offset, char* out, size_t out_sz) {
    if (module == nullptr || module[0] == '\0' || out_sz == 0) {
        return false;
    }
    // The JIT dlopen's the kernel from a temporary "<hash>.so.tmp.<pid>" that is
    // atomically renamed to "<hash>.so" right after load, so dladdr hands back a
    // path that no longer exists. If the reported module is gone, strip the
    // ".tmp.<pid>" suffix to recover the real, on-disk .so for the symbolizer.
    char modbuf[1024];
    std::snprintf(modbuf, sizeof(modbuf), "%s", module);
    if (access(modbuf, R_OK) != 0) {
        char* t = std::strstr(modbuf, ".so.tmp.");
        if (t != nullptr) {
            t[3] = '\0';  // keep "....so", drop ".tmp.<pid>"
        }
        if (access(modbuf, R_OK) != 0) {
            return false;  // can't find the object on disk; caller falls back
        }
    }
    module = modbuf;
    // Prefer llvm-symbolizer (reads clang's DWARF5; GNU addr2line cannot and only
    // yields "??:?"). The unversioned binary is often absent, so try the
    // clang-toolchain-versioned names too. addr2line is a last-ditch fallback.
    const char* tools[] = {
        "llvm-symbolizer --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-20 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-19 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "llvm-symbolizer-18 --obj=\"%s\" --pretty-print --inlines --demangle 0x%lx 2>/dev/null",
        "addr2line -f -C -i -p -e \"%s\" 0x%lx 2>/dev/null",
    };
    for (size_t t = 0; t < sizeof(tools) / sizeof(tools[0]); ++t) {
        char cmd[1100];
        std::snprintf(cmd, sizeof(cmd), tools[t], module, static_cast<unsigned long>(file_offset));
        FILE* pipe = popen(cmd, "r");
        if (pipe == nullptr) {
            continue;
        }
        char buf[2048];
        size_t used = 0;
        char line[512];
        while (used + 1 < sizeof(buf) && std::fgets(line, sizeof(line), pipe) != nullptr) {
            for (size_t i = 0; line[i] != '\0' && used + 1 < sizeof(buf); ++i) {
                char c = line[i];
                if (c == '\n' || c == '\r') {
                    if (used >= 4 && std::strcmp(buf + used - 4, " <- ") == 0) {
                        continue;
                    }
                    const char* sep = " <- ";
                    for (int k = 0; k < 4 && used + 1 < sizeof(buf); ++k) {
                        buf[used++] = sep[k];
                    }
                } else {
                    buf[used++] = c;
                }
            }
        }
        buf[used] = '\0';
        pclose(pipe);
        while (used >= 4 && std::strcmp(buf + used - 4, " <- ") == 0) {
            used -= 4;
            buf[used] = '\0';
        }
        if (used == 0) {
            continue;
        }
        bool has_file = std::strstr(buf, ".cpp") || std::strstr(buf, ".cc") || std::strstr(buf, ".h") ||
                        std::strstr(buf, ".hpp");
        if (!has_file && std::strstr(buf, "??") != nullptr) {
            continue;
        }
        std::snprintf(out, out_sz, "%s", buf);
        return true;
    }
    return false;
}

// Collapse balanced template-argument lists <...> to <> so frames stay readable
// (kernel NoC frames otherwise carry pages of TensorAccessor<...> spew). The
// " <- " inlined-frame separator is a '<' followed by '-', which is left intact.
static void __emule_asan_collapse_angles(char* s) {
    // In-place; only ever drops characters, so the write cursor never outruns the
    // read cursor. Keeps the outermost '<' and '>' and elides everything between.
    char* w = s;
    int depth = 0;
    for (const char* r = s; *r != '\0'; ++r) {
        if (*r == '<' && r[1] != '-') {
            if (depth == 0) {
                *w++ = '<';  // keep the outermost '<'
            }
            ++depth;
        } else if (*r == '>' && depth > 0) {
            --depth;
            if (depth == 0) {
                *w++ = '>';  // keep the outermost '>'
            }
        } else if (depth == 0) {
            *w++ = *r;
        }
    }
    *w = '\0';
}

static void __emule_asan_print_trace() {
    std::fflush(stdout);
    fprintf(stderr, "  --- emule ASAN context ---\n");
    if (__emule_kernel_name != nullptr && __emule_kernel_name[0] != '\0') {
        fprintf(stderr, "  kernel:    %s\n", __emule_kernel_name);
        fprintf(
            stderr,
            "  core:      logical (%u, %u)  physical (%u, %u)\n",
            __emule_logical_x,
            __emule_logical_y,
            static_cast<unsigned>(my_x[0]),
            static_cast<unsigned>(my_y[0]));
        fprintf(
            stderr,
            "  processor: %u  (neo %u, trisc %u)\n",
            static_cast<unsigned>(__processor_id),
            static_cast<unsigned>(__emule_neo_id),
            static_cast<unsigned>(__emule_trisc_id));
    }

    void* frames[128];
    int n = backtrace(frames, 128);
    char** syms = backtrace_symbols(frames, n);
    fprintf(stderr, "  backtrace (most recent call first):\n");

    int printed = 0;
    bool started = false;  // suppress the leading ASAN-machinery frames
    for (int i = 0; i < n && printed < 16; ++i) {
        Dl_info info;
        bool have_info = (dladdr(frames[i], &info) != 0) && info.dli_fname != nullptr;
        const char* module = have_info ? info.dli_fname : nullptr;
        // backtrace() yields return addresses (the instruction *after* each
        // call), so symbolize addr-1 to land on the calling instruction's line.
        uintptr_t off = have_info ? (reinterpret_cast<uintptr_t>(frames[i]) -
                                     reinterpret_cast<uintptr_t>(info.dli_fbase))
                                  : 0;
        char resolved[2048];
        bool ok =
            module != nullptr && __emule_asan_symbolize(module, off > 0 ? off - 1 : off, resolved, sizeof(resolved));

        // Skip print_trace / panic so the first frame shown is the check itself.
        if (!started) {
            const char* probe = ok ? resolved : ((have_info && info.dli_sname != nullptr) ? info.dli_sname : "");
            if (std::strstr(probe, "__emule_asan_") != nullptr) {
                continue;
            }
            started = true;
        }

        if (ok) {
            __emule_asan_collapse_angles(resolved);
            fprintf(stderr, "    #%-2d %s\n", printed, resolved);
        } else if (syms != nullptr && syms[i] != nullptr) {
            fprintf(stderr, "    #%-2d %s\n", printed, syms[i]);
        } else {
            fprintf(stderr, "    #%-2d %p\n", printed, frames[i]);
        }
        ++printed;

        // Stop at the kernel entry — everything below is runner/thread/libc glue.
        if (ok && (std::strstr(resolved, "kernel_main") != nullptr ||
                   std::strstr(resolved, "__emule_kernel_entry") != nullptr)) {
            break;
        }
    }
    if (syms != nullptr) {
        free(syms);
    }
    std::fflush(stderr);
}

extern "C" [[noreturn]] void __emule_asan_panic(const char* fmt, ...) {
    // emule runs one thread per core, so when a kernel bug hits every core they
    // all trip the check at once. Serialize: the first thread prints exactly one
    // full report (error line + context + backtrace) and aborts, which tears
    // down the whole process; every other thread blocks on this lock until that
    // abort kills it, so nothing interleaves and only ONE error is ever emitted.
    // The error line is printed here (not at the check site) precisely so the
    // lock covers it too.
    static std::mutex panic_mu;
    panic_mu.lock();  // intentionally never unlocked — the winner aborts while holding it
    if (fmt != nullptr) {
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stderr, fmt, ap);
        va_end(ap);
    }
    __emule_asan_print_trace();
    std::abort();
}

#endif  // EMULE_ASAN_IMPLEMENTATION
