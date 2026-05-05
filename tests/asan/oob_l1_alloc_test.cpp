// Negative test: read from a Core's L1 region that was never returned by l1_alloc.
//
// In standalone mode, Core::mmap_region poisons the entire WORKER L1 region
// at construction. l1_alloc unpoisons each newly-bumped slice. Bytes past
// the bump pointer remain poisoned, so a read targeting them trips ASan.
//
// Expect: ASan ERROR with "use-after-poison" and abort.

#include "tt_emule/device.hpp"
#include <cstdio>

int main() {
    tt_emule::Core core({0, 0}, tt_emule::CoreRole::WORKER, tt_emule::Core::L1_SIZE);

    // Allocate one buffer, leave the rest unallocated (still poisoned).
    constexpr size_t live = 4096;
    uint32_t addr = core.l1_alloc(live);
    (void)addr;

    // Read 1 byte past the bump pointer — first poisoned byte.
    volatile uint8_t v = core.l1_data()[live];
    std::fprintf(stderr, "ERROR: expected ASan to fire on read past l1_bump (got 0x%x)\n", v);
    return 1;
}
