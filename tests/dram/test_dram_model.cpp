// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM model regression coverage. Validates the tt_emule::Core DRAM role
// behaviors that the host bridges in tt-metal/.../emulated_program_runner.cpp
// (__emule_resolve_noc_addr, SWEmuleChip::write_to_device) rely on:
//
//   1. DRAM cores accept offsets ≥ 2 MB without truncation. The L1_SLOT_MASK
//      was previously applied unconditionally in __emule_resolve_noc_addr,
//      aliasing all DRAM accesses ≥ 2 MB into the first 2 MB of the bank.
//      This test writes well past 2 MB and verifies the data is preserved.
//
//   2. Multiple DRAM cores have isolated backing storage. SWEmuleChip
//      lazy-creates one mmap per DRAM core via get_core(xy). Writing to
//      one DRAM core must not affect another (proves per-bank routing in
//      Phase 2 host bridges).
//
//   3. Core::l1_ptr bounds check (Phase 4) catches out-of-range access.
//      A child-process death test verifies it aborts rather than
//      silently returning an OOB pointer.
//
//   4. WORKER cores are size-bounded to L1_SIZE (1 MB) and the role
//      accessor differentiates DRAM vs WORKER for the bridge's
//      mask-or-no-mask decision.

#include "tt_emule/device.hpp"

#include <sys/wait.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace tt_emule;

namespace {

#define CHECK(cond) do {                                                      \
    if (!(cond)) {                                                            \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
        std::exit(1);                                                         \
    }                                                                         \
} while (0)

// Test 1: DRAM core accepts offsets above L1's 2 MB slot mask threshold.
void test_dram_offset_above_2mb() {
    constexpr size_t DRAM_BANK_SIZE = 16 * 1024 * 1024;  // 16 MB; well above L1 slot
    Core dram(CoreCoord{0, 0}, CoreRole::DRAM, DRAM_BANK_SIZE);
    CHECK(dram.role() == CoreRole::DRAM);

    // Write a recognisable pattern at offset 4 MB (= 2 × L1_SLOT_SIZE) — would
    // alias to offset 0 under the old L1_SLOT_MASK truncation.
    constexpr uint64_t OFFSET = 4ull * 1024 * 1024;
    constexpr uint32_t MAGIC = 0xDEADBEEF;
    std::memcpy(dram.l1_ptr(OFFSET), &MAGIC, sizeof(MAGIC));

    // Read back from the same offset — must see MAGIC, not 0.
    uint32_t readback = 0;
    std::memcpy(&readback, dram.l1_ptr(OFFSET), sizeof(readback));
    CHECK(readback == MAGIC);

    // Verify offset 0 is still untouched — confirms no aliasing.
    uint32_t at_zero = 0;
    std::memcpy(&at_zero, dram.l1_ptr(0), sizeof(at_zero));
    CHECK(at_zero == 0);

    std::printf("  test_dram_offset_above_2mb: PASS\n");
}

// Test 2: Multiple DRAM cores have isolated backing.
void test_dram_multi_core_isolation() {
    constexpr size_t BANK = 8 * 1024 * 1024;  // 8 MB each
    Core bank0(CoreCoord{0, 0}, CoreRole::DRAM, BANK);
    Core bank1(CoreCoord{0, 1}, CoreRole::DRAM, BANK);

    constexpr uint64_t OFF = 3ull * 1024 * 1024;  // past L1 slot mask
    uint32_t val0 = 0xAAAA0000;
    uint32_t val1 = 0xBBBB0000;
    std::memcpy(bank0.l1_ptr(OFF), &val0, sizeof(val0));
    std::memcpy(bank1.l1_ptr(OFF), &val1, sizeof(val1));

    uint32_t rb0 = 0, rb1 = 0;
    std::memcpy(&rb0, bank0.l1_ptr(OFF), sizeof(rb0));
    std::memcpy(&rb1, bank1.l1_ptr(OFF), sizeof(rb1));
    CHECK(rb0 == val0);
    CHECK(rb1 == val1);

    // Cross-check: bank0 read at OFF must NOT show bank1's data, even though
    // they have the same offset. If the per-core mmap routing were broken
    // (single global DRAM view), both reads would return the last-written value.
    CHECK(rb0 != rb1);

    std::printf("  test_dram_multi_core_isolation: PASS\n");
}

// Test 3: Bounds check kicks in for out-of-range offset (Phase 4 safety).
// Death test — fork + check child abort()s.
void test_dram_bounds_check() {
    pid_t pid = fork();
    if (pid == 0) {
        // Child: deliberately go out-of-bounds, expect abort().
        // Quiet stderr/stdout to avoid noise in test logs.
        ::close(STDERR_FILENO);
        ::close(STDOUT_FILENO);
        constexpr size_t BANK = 2 * 1024 * 1024;  // 2 MB
        Core dram(CoreCoord{0, 0}, CoreRole::DRAM, BANK);
        (void)dram.l1_ptr(BANK);  // OOB — Phase 4 prints + abort()s.
        // Should not reach here.
        std::_Exit(0);
    }
    CHECK(pid > 0);
    int status = 0;
    waitpid(pid, &status, 0);
    // Child should have died via SIGABRT (abort()).
    CHECK(WIFSIGNALED(status));
    CHECK(WTERMSIG(status) == SIGABRT);

    std::printf("  test_dram_bounds_check: PASS\n");
}

// Test 4: WORKER vs DRAM role differentiation.
void test_role_accessor() {
    Core worker(CoreCoord{0, 0}, CoreRole::WORKER, Core::L1_SIZE);
    Core dram(CoreCoord{0, 0}, CoreRole::DRAM, 8 * 1024 * 1024);
    CHECK(worker.role() == CoreRole::WORKER);
    CHECK(dram.role() == CoreRole::DRAM);
    // size_t (WORKER L1 is 1 MB by default; DRAM is bank-sized)
    CHECK(worker.l1_size() == Core::L1_SIZE);
    CHECK(dram.l1_size() == 8ull * 1024 * 1024);

    std::printf("  test_role_accessor: PASS\n");
}

}  // namespace

int main() {
    std::printf("tt-emule DRAM model regression tests\n");
    test_dram_offset_above_2mb();
    test_dram_multi_core_isolation();
    test_dram_bounds_check();
    test_role_accessor();
    std::printf("ALL PASS\n");
    return 0;
}
