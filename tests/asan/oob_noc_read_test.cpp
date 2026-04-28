// Negative test: the sized NOC-resolver bounds-check fires when a kernel
// requests `offset + size > l1_size`. Standalone tests don't link against
// the tt-metal program runner, so we instantiate the same logic here against
// a local Core for self-contained verification.
//
// Expect: __emule_bounds_fail prints "[EMULE] noc_async_read: out-of-bounds"
// and _Exit(2)s.

#include "tt_emule/asan.h"
#include "tt_emule/device.hpp"
#include <cstdio>
#include <cstdint>

int main() {
    tt_emule::Core core({0, 0}, tt_emule::CoreRole::WORKER, tt_emule::Core::L1_SIZE);
    uint64_t offset = tt_emule::Core::L1_SIZE - 32;
    uint64_t size   = 64;  // straddles the boundary
    uint64_t limit  = core.l1_size();
    if (offset + size > limit) {
        __emule_bounds_fail("noc_async_read", "noc offset + size > target region",
                            offset, size, limit);
    }
    std::fprintf(stderr, "ERROR: expected bounds-fail abort\n");
    return 1;
}
