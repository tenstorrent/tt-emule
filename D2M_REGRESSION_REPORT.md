# D2M Regression Report

**Date:** 2026-03-16 (updated)
**Build:** tt-metal `build_emule_clang` (clang-17, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (5 files run this cycle, 907 tests collected)
**Fixes since last run:** DRAM bank offset fix in NOC address translation; removed `__emule_fixup_noc_addr` from all NOC ops

## Summary

| Status | Count |
|--------|-------|
| Passed | 614 |
| Failed | 84 |
| Skipped | 48 |
| Not reached (OOM crash) | ~161 |

**Improvement vs previous run (2026-03-16 earlier):** +365 passed tests (614 vs 249). Three major wins:
1. **Batched matmul fixed:** All 10 `test_metal_matmul_higher_rank` tests now PASS (were 0/10, PCC 0.15→0.99+)
2. **Reductions unblocked:** 420 reduction tests PASS (were 12), including `test_sum`, `test_sum_3d`, `test_sum_4d`, `test_max`
3. **TMS fully passing:** 169/173 TMS tests PASS (were 68), including all permutes and all reshapes

## Per-File Results

| Test File | Total | Passed | Failed | Skipped | Status |
|-----------|-------|--------|--------|---------|--------|
| test_metal_tms | 173 | 169 | 4 | 0 | PASS (arange fails) |
| test_metal_reductions | 548 | 420 | 80 | 48 | PASS (unaligned fails) |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | **PASS** |
| test_metal_matmul | 127 | 15+ | 0 | 0 | OOM crash at test 16 (all executed pass) |
| test_metal_dma | 49 | 0 | 0 | 0 | Not reached (after matmul crash) |

**Previously tested files (not re-run this cycle, results from earlier 2026-03-16 run):**

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | PASS |
| test_metal_allocate | 6 | 6 | 0 | 0 | PASS |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_tilize | 11 | 3 | 4 | 4 skip | FAIL (golden PCC mismatch) |
| test_metal_masking | 20 | 2 | 18 | 0 | FAIL (JIT compile errors) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | FAIL (JIT compile errors) |
| test_metal_tensor_collapsing | 14 | 0 | 10 | 2 skip, 2 xfail | FAIL (mix: JIT errors + golden mismatch) |
| test_metal_virtual_grids | 39 | 0 | 39 | 0 | FAIL (JIT compile errors) |

## What Changed (2026-03-16 update)

### DRAM bank offset fix — root cause of batched matmul and reduction failures

**Root cause:** `__emule_fixup_noc_addr` applied `__emule_addr_to_offset` (2MB bitmask `& 0x1FFFFF`) to all NOC addresses before resolving them. On Wormhole N150, DRAM has 12 banks across 6 physical channels — odd banks (1,3,5,7,9,11) have `address_offset = 0x40000000` (1GB). The 2MB mask destroyed this offset, causing all odd-bank DRAM reads to address offset 0x0 instead of 0x40000000.

**Impact:** Any operation reading interleaved DRAM buffers with >1 bank got corrupted data for ~50% of pages (all odd-bank pages read from wrong offset). This caused:
- Batched matmul: PCC 0.15 (half the input tiles were wrong)
- Reductions: incorrect sums (half the input data corrupted)
- Large single-batch matmul shapes that span many DRAM pages

**Fix:** Removed `__emule_fixup_noc_addr` from all 5 NOC operation functions:
- `noc_async_read` — uses `__emule_resolve_noc_addr` directly
- `noc_async_write` — uses `__emule_resolve_noc_addr` directly
- `noc_async_write_multicast` — passes raw addr to `__emule_multicast_write`
- `noc_semaphore_inc` — uses `__emule_resolve_noc_addr` directly
- `noc_semaphore_set_multicast` — passes raw addr to multicast

L1 offset extraction (`__emule_addr_to_offset`) is only needed at construction time in `get_noc_addr()` / `get_write_ptr()` — not at NOC operation time.

**Files modified:**
- `include/jit_hw/api/dataflow/dataflow_api.h` — removed fixup from 5 NOC functions
- `include/jit_hw/api/compute/matmul.h` — removed debug logging
- `include/jit_hw/llk_defs.h` — removed debug logging
- `tt_metal/impl/emulation/emulated_program_runner.cpp` — re-enabled inline source cleanup

## Failure Categories

### Category 1: Unaligned tensor operations (80 failures — pre-existing)

**test_sum_unaligned** (48 failures) + **test_max_unaligned** (32 failures): Reduction operations on non-tile-aligned tensor shapes. These require padding/masking logic in the reduction kernel that the emulator stubs don't handle.

### Category 2: Arange (4 failures — pre-existing)

**test_arange** (4 failures): `arange` op generates sequential values via a specialized compute kernel. Likely needs a dedicated stub.

### Category 3: OOM crash (161 tests not reached)

**test_metal_matmul** crashed at test 16 of 127 (`1024x1024x2048`) with `Fatal Python error: Aborted` during `execute_fb`. The same test passes in isolation (confirmed via `--forked` run). Cause: accumulated memory from running hundreds of tests in a single process without `--forked`. All 15 matmul tests that executed before the crash passed with zero failures.

**test_metal_dma** (49 tests) not reached — came after matmul in test ordering.

### Category 4: JIT Compile Errors (4 files, ~80 failures — unchanged)

Same as previous run: missing `abs_tile`, `typecast_tile`, `exp_tile` stubs; masking CB API path conflict.

### Category 5: Tilize PCC mismatch (4 failures — unchanged)

Non-tile-aligned tilize shapes. Tile-aligned shapes pass.

## Clean Passes

**test_metal_matmul_higher_rank** (10/10) — All 3D and 4D batched matmul variants pass with PCC > 0.99. **NEW: was 0/10.**

**test_metal_tms** (169/173) — All reshapes (f32 + bf16, 54 shapes), all permutes (50 shapes), all concatenate_heads (11 shapes) pass. Only 4 arange tests fail. **NEW: was 68/173.**

**test_metal_reductions** (420/548) — All `test_sum` (108), `test_sum_3d` (80), `test_sum_4d` (160), `test_max` (72) pass. Only `test_sum_unaligned` (48) and `test_max_unaligned` (32) fail. 48 skipped (keepdim=False variants). **NEW: was 12/548.**

**test_metal_matmul** (15/15 executed) — All single-buffered `matmul_tile` variants (f32 + bf16) pass including 2048x2048x2048. Run cut short by OOM crash.

**test_metal_layout** (94/94) — Unchanged.

**test_metal_allocate** (6/6) — Unchanged.

## Recommended Fixes by Priority

### P0: Fix OOM crash + run full matmul suite (unblocks ~161 tests)

Run matmul and DMA tests with `--forked` to isolate per-test memory. The 1024x1024x2048 crash is not a kernel bug — it passes in isolation. Alternatively, split the test invocation into smaller batches.

### P1: Add missing compute stubs (unblocks ~80 tests)

Add to `include/jit_hw/api/compute/`:
- `abs_tile_init()` / `abs_tile(dst_index)` — `fabsf` per element in DST
- `typecast_tile_init()` / `typecast_tile(DataFormat, dst_index)` — format conversion in DST
- `exp_tile_init()` / `exp_tile(dst_index)` — `expf` per element in DST
- `add_tiles_init()` / `mul_tiles_init()` — no-op init stubs
- `arange` compute stub

Fix masking CB API include path conflict.

### P2: Unaligned tensor support (unblocks ~80 tests)

`test_sum_unaligned` and `test_max_unaligned` need padding/masking in reduction stubs for non-tile-aligned shapes.

### P3: Non-aligned tilize (unblocks ~4 tests)

Fix tilize PCC for non-tile-aligned shapes.

## Historical Progress

| Date | Passed | Failed | Key Change |
|------|--------|--------|------------|
| 2026-03-11 | 128 | ~310 | Initial D2M regression |
| 2026-03-13 | 208 | ~251 | Multicast NOC fixes, parallel JIT, cross-program core_map |
| 2026-03-16a | 249 | ~244 | HAL-based semaphore base; matmul abort resolved (+41 matmul passes) |
| 2026-03-16b | 614 | 84 | **DRAM bank offset fix**; +365 passes; higher_rank matmul, reductions, TMS all pass |
