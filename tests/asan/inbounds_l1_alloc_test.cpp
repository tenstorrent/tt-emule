// Positive control: write inside a region returned by Core::l1_alloc.
//
// Core::mmap_region poisons the entire WORKER L1 at construction. l1_alloc
// unpoisons each newly-bumped slice — bytes inside the returned slice MUST
// be writable without firing ASan. If this test trips ASan, the unpoison
// in l1_alloc regressed.
//
// Expect: clean exit 0.

#include "tt_emule/device.hpp"
#include <cstdio>

int main() {
    tt_emule::Core core({0, 0}, tt_emule::CoreRole::WORKER, tt_emule::Core::L1_SIZE);

    constexpr size_t live = 4096;
    uint32_t addr = core.l1_alloc(live);
    (void)addr;

    // Inside the unpoisoned slice — must NOT trip ASan.
    volatile uint8_t* p = core.l1_data();
    p[0] = 0x42;
    p[live - 1] = 0x55;

    std::fprintf(stderr, "OK: wrote inside l1_alloc'd region without ASan firing\n");
    return 0;
}
