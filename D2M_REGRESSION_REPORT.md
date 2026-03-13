# D2M Regression Report

**Date:** 2026-03-13 (updated run)
**Build:** tt-metal `build_emule_clang` (clang-17, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (13 files, ~1146 total tests)
**Fixes since last run:** multicast worker-core filtering, cross-program NOC reads (core_map populated for all workers), L1 offset masking in multicast, parallel JIT compilation, PYTHONPATH fix for `builder` module

## Summary

| Status | Count |
|--------|-------|
| Passed | 208 |
| Failed | ~251 |
| Skipped/XFail | 51 |
| Aborted (SIGABRT) | 2 files (~168 tests unreachable) |
| Hung (timeout) | 2 files (~530+ tests unreachable) |

**Improvement vs previous run:** +80 passed tests, 2 fewer hung files (matmul and DMA now abort cleanly instead of hanging), reductions now runs 118 tests before timeout (was ~67).

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | PASS |
| test_metal_allocate | 6 | 6 | 0 | 0 | PASS |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_matmul | 127 | 15 | 0 | 2 xfail | ABORT at test 18 (2048x2048x2048 matmul_block) |
| test_metal_dma | 49 | 8 | 0 | 0 | ABORT at test 9 (first DRAM-backed DMA) |
| test_metal_matmul_higher_rank | 10 | 0 | 10 | 0 | FAIL (golden PCC mismatch) |
| test_metal_tilize | 11 | 3 | 4 | 4 skip | FAIL (golden PCC mismatch) |
| test_metal_masking | 20 | 2 | 18 | 0 | FAIL (JIT compile errors) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | FAIL (JIT compile errors) |
| test_metal_tensor_collapsing | 14 | 0 | 10 | 2 skip, 2 xfail | FAIL (mix: JIT errors + golden mismatch) |
| test_metal_virtual_grids | 39 | 0 | 39 | 0 | FAIL (JIT compile errors) |
| test_metal_tms | 173 | 68 | 67 | 0 | HUNG at 78% (reshapes pass, permutes fail) |
| test_metal_reductions | 548 | 12 | 90 | 16 skip | HUNG at 21% (all sum variants fail) |

## What Changed

### Hangs resolved -> Aborts
**test_metal_matmul** and **test_metal_dma** previously hung indefinitely. With the multicast worker-core filtering and NOC fixes, these now abort cleanly (SIGABRT) instead of deadlocking. The abort occurs on the same test case (matmul_block-f32-2048x2048x2048, DRAM DMA with multi-grid) but the process terminates within the timeout instead of hanging.

### Reductions progress
Previously ~67 tests ran before timeout. Now 118 tests execute (12 passed, 90 failed, 16 skipped) before timeout at 21%. The failures are golden PCC mismatches, not hangs -- the sum reduction compute produces incorrect results. The timeout is from the sheer volume of tests (548) each taking ~2.5s.

### TMS progress
Previously ~88 tests ran before timeout. Now 135 tests execute (68 passed, 67 failed) before timeout at 78%. Tile-aligned reshapes pass; non-tile-aligned reshapes and all permutes fail.

### Layout/Allocate stable
Still 100% pass (100 tests total).

## Failure Categories

### Category 1: Aborts (2 files, ~168 tests unreachable)

**test_metal_matmul** -- 15 passed (all `matmul_tile` bf16/f32 variants up to 1024x1024x2048), 2 xfail (bf16 2048x). Process aborts on `matmul_block-f32-2048x2048x2048`. The abort (not hang) suggests the multicast/semaphore deadlock is partially fixed but an assertion or memory error remains for large matmul_block shapes.

**test_metal_dma** -- 8 passed (all L1-to-L1 DMA transfers). Process aborts on first DRAM-backed test (`dram-end_grid1-start_grid3-shape0`). DRAM DMA path triggers an assertion failure during NOC address resolution for DRAM cores.

### Category 2: JIT Compile Errors (4 files, ~70 failures)

Missing compute stub declarations cause g++ to fail during JIT compilation.

**test_metal_virtual_grids** (39 failures): Missing `abs_tile_init()`, `abs_tile(dst_index)`

**test_metal_bfp8_typecast** (13 failures): Missing `typecast_tile_init()`, `typecast_tile(DataFormat, dst_index)`

**test_metal_masking** (16 of 18 failures): Masking kernel's compute thread resolves `cb_wait_front` etc. from real `hw/inc/api/compute/cb_api.h` (which calls `llk_*` functions) instead of the emulator's stub. Also `cb_reserve_back` overload conflict between `int32_t` and `uint32_t`.

**test_metal_tensor_collapsing** (8 of 10 failures): Mix of missing stubs (`add_tiles_init`, `mul_tiles_init`, `exp_tile_init`, `exp_tile`) and golden PCC mismatch for matmul variants.

### Category 3: Incorrect Numeric Results (4 files, ~171 failures)

Tests compile and run but produce wrong output (PCC well below 0.99 threshold).

**test_metal_reductions** (90 failures): All `test_sum_unaligned` and `test_sum_4d` variants fail. The reduction compute kernel produces near-zero PCC. Likely cause: the reduce_tile stub doesn't correctly accumulate partial sums across the reduction dimension.

**test_metal_matmul_higher_rank** (10 failures): PCC range 0.02--0.15 for all batched matmul (3D, 4D). The batch loop in D2M-generated kernels handles accumulation differently from the single-batch case that passes.

**test_metal_tms** (67 failures): Non-tile-aligned reshapes and all permute operations fail. Tile-aligned reshapes (where input/output tile boundaries match) pass. Permute failures are likely due to missing transpose/permute compute stubs.

**test_metal_tilize** (4 failures): PCC ~0.06--0.08 for certain shapes. 3 tile-aligned shapes pass, 4 non-aligned fail.

### Category 4: Timeouts (2 files, ~530+ tests unreachable)

**test_metal_reductions** -- 548 tests, reached 21% (118 tests) in 300s. Not a kernel hang -- each test takes ~2.5s and there are simply too many. Increasing timeout to 1500s would allow full completion.

**test_metal_tms** -- 173 tests, reached 78% (135 tests) in 300s. The last test was `test_permute[shape29]`. Each permute test takes ~3s; likely would complete in ~400s.

## Clean Passes

**test_metal_layout** (94/94) -- All layout transformation tests pass: tilize, untilize, L1/DRAM transfers, multi-core grid distribution.

**test_metal_allocate** (6/6) -- Buffer allocation with max-reduce and matmul operations.

**test_metal_virtual_grid_rowmajor** (27/27 skipped) -- All skipped (requires n300 multi-chip).

**test_metal_matmul** (15/15 non-xfail pass) -- Single-core matmul_tile (bf16 and f32) and matmul_block (f32, up to 1024x sizes) all pass.

**test_metal_dma** (8/8 before abort) -- All L1-to-L1 DMA roundtrip tests pass.

**test_metal_tms** (68 pass) -- All tile-aligned reshapes pass (e.g., 64x64->32x128, 96x64->64x96).

## Recommended Fixes by Priority

### P0: Increase timeout + fix abort (unblocks ~700+ tests)

1. **Increase timeout to 600s** for reductions and TMS -- these aren't hanging, just slow with many tests. This alone would expose ~430 additional test results.
2. **Fix SIGABRT in matmul_block-2048x** and **DRAM DMA** -- investigate the assertion failure that replaced the previous hang. Check L1 allocation size for large matmul_block intermediates and DRAM NOC address resolution for multi-grid DMA.

### P1: Add missing compute stubs (unblocks ~70 tests)

Add to `include/jit_hw/api/compute/`:
- `abs_tile_init()` / `abs_tile(dst_index)` -- `fabsf` per element in DST
- `typecast_tile_init()` / `typecast_tile(DataFormat, dst_index)` -- format conversion in DST
- `exp_tile_init()` / `exp_tile(dst_index)` -- `expf` per element in DST
- `add_tiles_init()` / `mul_tiles_init()` -- no-op init stubs

Fix masking CB API include path: ensure compute kernels resolve CB functions from emulator stubs, not real `hw/inc/`. Fix `cb_reserve_back` overload for `int32_t`/`uint32_t`.

### P2: Fix reduction numerics (unblocks ~90 tests)

The `reduce_tile` stub likely doesn't handle partial-sum accumulation correctly for D2M-generated reduction loops. Check whether the reduction dimension stride and accumulation target in DST match what the kernel expects.

### P3: Fix batched matmul + permute correctness (unblocks ~77 tests)

- Batched matmul: DST accumulation may not reset between batches, or the batch loop structure differs from single-batch.
- Permute: likely needs a `transpose_wh` or general permute stub that rearranges tile data in DST/CB.
