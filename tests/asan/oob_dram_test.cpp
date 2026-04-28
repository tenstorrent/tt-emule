// Negative test: __emule_dram_ptr called with offset past the DRAM bank size.
//
// Expect: __emule_bounds_fail prints "[EMULE] __emule_dram_ptr: out-of-bounds"
// and std::abort()s. This bounds check is always-on (not gated by ASan).

#include "tt_emule/host_api.hpp"
#include "tt_emule/device.hpp"
#include <cstdio>

extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);

extern thread_local tt_emule::Device* __device;

int main() {
    tt_emule::Device dev;
    __device = &dev;

    // Read just past the bank size — should trip the bounds check.
    uint8_t* p = __emule_dram_ptr(static_cast<uint64_t>(dev.dram_size_per_channel()));
    std::fprintf(stderr, "ERROR: expected bounds-fail abort (got ptr=%p)\n", (void*)p);
    return 1;
}
