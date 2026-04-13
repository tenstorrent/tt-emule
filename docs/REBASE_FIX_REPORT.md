# Rebase Fix Report: tt-metal upstream `3fa4d75355`

## Overview

After rebasing tt-metal onto upstream commit `3fa4d75355` (1,669 commits newer), the DFB emulation regression went from **83 passed / 30 failed / 2 skipped** to **24 passed / 89 failed / 2 skipped**. A series of fixes recovered to the current state of **109 passed / 4 failed / 2 skipped**.

### Progression

| Stage | Passed | Failed | Skipped | Description |
|-------|--------|--------|---------|-------------|
| Pre-rebase baseline | 83 | 30 | 2 | Before rebase |
| Post-rebase (broken) | 24 | 89 | 2 | After rebase, no fixes |
| After fixes 1-3 | 58 | 55 | 2 | JIT stubs, HAL core count, finalize alloc_addr |
| After fix 4 (BLOCKED) | 80 | 33 | 2 | Correct BLOCKED stride/offset |
| After fixes 5-6 (TensixDM) | 94 | 19 | 2 | proc_bit + early DFB finalize |
| After fix 7 (BLOCKED drain) | **109** | **4** | 2 | BLOCKED consumer drain_per_tc + per-slot limits |

The 4 remaining failures are all pre-existing (DFBEmuleDMTest, DFBEmuleBridgeTest, DmLoopbackPacketSizes, ttnn_add_int_silicon).

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

## Committed Fixes (round 2)

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

### Fix 7: BLOCKED consumer drain behavior and per-slot memory layout
**Commit:** `b342581` (tt-emule), `5d3617c0b2` (tt-metal)

**Root cause:** Two bugs in BLOCKED consumer emulation:
1. `pop_front()` advanced `tc_idx` on every call (round-robin), but BLOCKED consumers must drain each TC slot fully before moving to the next. The hardware reads all `capacity` entries from TC0 before TC1.
2. All TC slots shared the entire buffer range. Each slot must have its own sub-range: `base_addr = alloc_base + p * capacity * entry_size`, `limit = base_addr + capacity * entry_size`.

**Impact:** All 16 multi-P/C BLOCKED tests recovered (8 DM-DM + 8 TensixDM). Regression: 94 → 109 passed.

**Files changed:**

| File | Repo | Change |
|------|------|--------|
| `include/tt_emule/dfb_sync_state.hpp` | tt-emule | Added `drain_per_tc` field to `EmuleDFBInterface` |
| `include/jit_hw/api/dfb_api.h` | tt-emule | `dfb_pop_front`: conditional tc_idx advancement |
| `include/tt_emule/dataflow_buffer.hpp` | tt-emule | Same drain logic for standalone path |
| `src/kernel_runner.cpp` | tt-emule | Per-slot base_addr/limit for BLOCKED |
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | tt-metal | Per-slot base_addr/limit + drain_per_tc |

**Important:** Changing `EmuleDFBInterface` struct layout invalidates JIT cache. Clear `/tmp/tt_emule_jit_cache_*/` after this fix to avoid stale `.so` files that read the wrong field offsets.

### Other: TensorAccessor null-pointer diagnostic

**File:** `include/jit_hw/experimental/tensor.h` (tt-emule)

Added a diagnostic `fprintf` in `noc_traits_t<TensorAccessor>::dst_addr` that fires only when `__emule_resolve_noc_addr()` returns null. This is an error-path guard, not general instrumentation. It prints the NOC coordinates and offset to help diagnose future DRAM address resolution failures.

---

## Remaining Failures (4, all pre-existing)

| Category | Count | Notes |
|----------|-------|-------|
| DFBEmuleDMTest | 1 | Pre-existing standalone DFB emulation test failure (host-side verification mismatch). |
| DFBEmuleBridgeTest | 1 | Pre-existing bridge test failure (same root cause). |
| DmLoopbackPacketSizes | 1 | Pre-existing data movement test failure. |
| ttnn_add_int_silicon | 1 | Requires real hardware (always fails in emulation). |

---

## File Change Summary

### tt-metal

```
 tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer.cpp  | +6
 tt_metal/impl/emulation/emulated_program_runner.cpp                   | +39 -24
```

### tt-emule

```
 docs/DFB_EMULATION.md                    | updated
 include/tt_emule/dfb_sync_state.hpp      | +2 -1  (drain_per_tc field)
 include/jit_hw/api/dfb_api.h             | +8 -1  (conditional tc_idx)
 include/tt_emule/dataflow_buffer.hpp     | +7 -1  (same for standalone)
 include/jit_hw/experimental/tensor.h     | +7
 src/kernel_runner.cpp                    | +24 -6  (per-slot limits)
```
