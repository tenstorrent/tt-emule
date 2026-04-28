// Negative test: write into the poisoned 1 MB tail of an L1Pool slot.
//
// L1Pool slots are 2 MB but only the first 1 MB is the "live" L1 region.
// The pool poisons the trailing 1 MB at construction, so a kernel that
// computes an offset past Core::L1_SIZE and writes into that tail trips ASan.
//
// Expect: ASan ERROR with "use-after-poison" or "use-after-scope" and abort.

#include "tt_emule/device.hpp"
#include "tt_emule/l1_pool.hpp"
#include <cstdio>

int main() {
    tt_emule::L1Pool pool(/*num_slots=*/2,
                          /*live_size_per_slot=*/tt_emule::Core::L1_SIZE);
    uint8_t* slot = pool.slot_ptr(0);

    // Write 1 byte at offset L1_SIZE — first byte of the poisoned tail.
    slot[tt_emule::Core::L1_SIZE] = 0x42;

    // Should not reach here under ASan.
    std::fprintf(stderr, "ERROR: expected ASan to fire on poisoned-tail write\n");
    return 1;
}
