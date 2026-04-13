# Rebase Fix Report: tt-metal upstream `3fa4d75355`

## Overview

After rebasing tt-metal onto upstream commit `3fa4d75355` (1,669 commits newer), the DFB emulation regression went from **83 passed / 30 failed / 2 skipped** to **24 passed / 89 failed / 2 skipped**. A series of fixes recovered to the current state of **94 passed / 19 failed / 2 skipped**.

### Progression

| Stage | Passed | Failed | Skipped | Description |
|-------|--------|--------|---------|-------------|
| Pre-rebase baseline | 83 | 30 | 2 | Before rebase |
| Post-rebase (broken) | 24 | 89 | 2 | After rebase, no fixes |
| After fixes 1-3 | 58 | 55 | 2 | JIT stubs, HAL core count, finalize alloc_addr |
| After fix 4 (BLOCKED) | 80 | 33 | 2 | Correct BLOCKED stride/offset |
| After fixes 5-6 (TensixDM) | **94** | **19** | 2 | proc_bit + early DFB finalize |

The 19 remaining failures are all pre-existing (before the rebase).

---

## Committed Fixes (in tt-metal)

### Fix 1: JIT stubs for upstream API changes
**Commit:** `39eb510` (tt-emule)

The upstream rebase introduced new APIs (`TensorAccessor`, `DEVICE_PRINT`, `risc_common` changes) that JIT-compiled device kernels depend on. Added emulation stub headers so the JIT compiler can resolve these symbols.

### Fix 2: HAL core type count
**Commit:** `5f919b871c` (tt-metal)

`emulated_program_runner.cpp` used a hardcoded core type count. After the rebase, the HAL (Hardware Abstraction Layer) changed its core type enumeration. Fixed to query the runtime HAL for the correct count.

### Fix 3: DFB finalize alloc_addr
**Commit:** `95a5f23044` (tt-metal)

The DFB L1 allocation path was using an independent bump allocator, producing addresses that didn't match what `finalize_dataflow_buffer_configs()` computed. Fixed to read the finalize-allocated address from `dfb_impl->core_lookup_` so that the emulated DFB data lands at the same L1 offset the test expects.

---

## Uncommitted Fixes

### Fix 4: BLOCKED mode stride and initial offset

**Root cause:** The emulation computed BLOCKED mode layout as interleaved entries (`stride = num_producers * entry_size`, producer p starts at `base + p * entry_size`). The upstream Quasar serialization uses contiguous blocks (`stride = entry_size`, producer p starts at `base + p * capacity * entry_size` where `capacity = num_entries / num_producers`).

**Impact:** All 14 DM-DM BLOCKED tests failed with data corruption.

**Files changed:**

| File | Repo | Change |
|------|------|--------|
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | tt-metal | Fix BLOCKED producer/consumer stride and offset |
| `src/kernel_runner.cpp` | tt-emule | Same fix for standalone runner |
| `docs/DFB_EMULATION.md` | tt-emule | Update documentation to match upstream semantics |

**Detail (emulated_program_runner.cpp):**
```
BLOCKED producer:
- stride_size: num_producers * entry_size  -->  entry_size
- offset:      p * entry_size              -->  p * (num_entries / num_producers) * entry_size

BLOCKED consumer:
- stride_size: num_producers * entry_size  -->  entry_size
- rd_ptr/wr_ptr offset: same fix as producer
```

### Fix 5: WH/BH ComputeKernel proc_bit mismatch

**Root cause:** The upstream `BindDataflowBufferToProducerConsumerKernels()` sets different risc_mask bits depending on kernel type:
- **Quasar `QuasarComputeKernel`:** bits 8+ (`TENSIX_RISC_OFFSET = 8`)
- **WH/BH `ComputeKernel`:** bit 2

The emulation runner always computed `proc_bit` using the Quasar convention (`1 << (proc_id + 8)`), so on WH the Tensix producer thread never matched its DFB allocation. The DFB was never activated for the producer, and the DM consumer hung waiting for tiles that never came (120s timeout + abort).

**Impact:** All TensixDM tests failed with timeout/abort (22 tests affected, of which 14 were recoverable on the Quasar descriptor used by the regression suite).

**Files changed:**

| File | Repo | Change |
|------|------|--------|
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | tt-metal | Detect mask convention per-alloc; use bit 2 for WH/BH |
| `src/kernel_runner.cpp` | tt-emule | Same detection logic for standalone runner |

**Detail:** The fix moves `proc_bit` computation inside the per-alloc loop and auto-detects the convention:
```cpp
if (is_tensix) {
    bool quasar_masks = ((alloc.producer_risc_mask | alloc.consumer_risc_mask) & 0xFF00u) != 0;
    proc_bit = quasar_masks
        ? static_cast<uint16_t>(1u << (proc_id + TENSIX_RISC_OFFSET))  // Quasar
        : static_cast<uint16_t>(1u << 2);                               // WH/BH
} else {
    proc_bit = static_cast<uint16_t>(1u << proc_id);
}
```

### Fix 6: Early DFB finalize for TensixDM pre-fill

**Root cause:** The test pre-fills each core's DFB L1 with input data by iterating `dfb->groups[*].l1_by_core`. However, `groups` is only populated during `finalize_dataflow_buffer_configs()`, which runs inside `LaunchProgram`. The pre-fill runs *before* launch, so `groups` was empty and `WriteToDeviceL1` was never called. The kernel then read zeros from un-initialized L1.

**Impact:** All TensixDM tests produced all-zero output.

**File changed:**

| File | Repo | Change |
|------|------|--------|
| `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer.cpp` | tt-metal | Call `finalize_dataflow_buffer_configs()` + `allocate_dataflow_buffers()` before pre-fill |

**Detail:** Both functions are idempotent (they check `configs_finalized` and `allocation_needed_` flags), so the later call during `LaunchProgram` is a safe no-op:
```cpp
if (producer_type == DFBPorCType::TENSIX || consumer_type == DFBPorCType::TENSIX) {
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().allocate_dataflow_buffers(device);
}
```

### Other: TensorAccessor null-pointer diagnostic

**File:** `include/jit_hw/experimental/tensor.h` (tt-emule)

Added a diagnostic `fprintf` in `noc_traits_t<TensorAccessor>::dst_addr` that fires only when `__emule_resolve_noc_addr()` returns null. This is an error-path guard, not general instrumentation. It prints the NOC coordinates and offset to help diagnose future DRAM address resolution failures.

---

## Remaining Failures (19, all pre-existing)

| Category | Count | Notes |
|----------|-------|-------|
| DM-DM BLOCKED multi-P/C | 8 | `DMTest1xDFB{4Sx1B,4Sx4B,4Sx2B,2Sx4B}` x2 IS variants. Multi-producer/consumer BLOCKED on Quasar descriptor. |
| TensixDM BLOCKED multi-P/C | 8 | Same pattern as DM-DM but with Tensix producer. Quasar-only tests that pass through but have pre-existing BLOCKED issues with multi-P/C. |
| DFBEmuleDMTest | 1 | Pre-existing standalone DFB emulation test failure. |
| DFBEmuleBridgeTest | 1 | Pre-existing bridge test failure. |
| ttnn_add_int_silicon | 1 | Pre-existing silicon toggle test (always fails in emulation). |

---

## File Change Summary

### tt-metal (uncommitted)

```
 tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer.cpp  | +6
 tt_metal/impl/emulation/emulated_program_runner.cpp                   | +25 -13
```

### tt-emule (uncommitted)

```
 docs/DFB_EMULATION.md                 | +5 -5
 include/jit_hw/experimental/tensor.h  | +7
 src/kernel_runner.cpp                 | +24 -6
```
