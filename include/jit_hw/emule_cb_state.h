#pragma once
// Circular buffer state for JIT-compiled kernels.
// CBs are backed by L1 memory (mmap'd below 4 GB for 32-bit pointer compat).
// Shared between threads running on the same core; each thread's TLS points
// to the same array.  Defined in emulated_program_runner.cpp and resolved by
// the JIT .so at dlopen time.

#include <cstdint>
#include <mutex>
#include <condition_variable>

struct __emule_cb_state {
    uint8_t* base       = nullptr; // Host pointer to start of this CB's L1 region
    uint32_t page_size  = 0;       // Bytes per page (tile size)
    uint32_t num_pages  = 0;       // Capacity
    uint32_t write_idx  = 0;       // Current write index
    uint32_t read_idx   = 0;       // Current read index
    uint32_t occupied   = 0;       // Number of occupied pages
    std::mutex mu;
    std::condition_variable space_cv;
    std::condition_variable data_cv;
};

// Thread-local pointer to per-core CB state array (32 entries).
extern thread_local __emule_cb_state* __emule_cbs;
