# D2M Regression Report

**Date:** 2026-03-13
**Build:** tt-metal `build_emule_clang` (clang-17, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (13 files, ~1146 total tests)

## Summary

| Status | Count |
|--------|-------|
| Passed | 128 |
| Failed | ~205 |
| Skipped/XFail | 37 |
| Hung (timeout) | 4 files (~750+ tests unreachable) |

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | PASS |
| test_metal_allocate | 6 | 6 | 0 | 0 | PASS |
| test_metal_matmul | 127 | 15 | 0 | 2 xfail | HUNG at test 18 |
| test_metal_matmul_higher_rank | 10 | 0 | 10 | 0 | FAIL |
| test_metal_tilize | 11 | 3 | 4 | 4 skip | FAIL |
| test_metal_dma | 49 | 8 | 0 | 0 | HUNG at test 9 |
| test_metal_masking | 20 | 2 | 18 | 0 | FAIL |
| test_metal_reductions | 548 | 0 | ~67 | 0 | HUNG at ~67 |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | FAIL |
| test_metal_tensor_collapsing | 14 | 0 | 10 | 2 skip, 2 xfail | FAIL |
| test_metal_tms | 173 | ~44 | ~44 | 0 | HUNG at ~88 |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_virtual_grids | 39 | 0 | 39 | 0 | FAIL |

## Failure Categories

### Category 1: Hangs (4 files, blocks ~750+ tests)

Four test files hit a 180s timeout and were killed mid-run. In each case, some early tests pass before a specific test pattern triggers an indefinite hang.

**test_metal_matmul** -- 15 passed (`matmul_tile` variants), then hung on `matmul_block-f32-2048x2048x2048`. All `matmul_tile` shapes up to 2048x2048x2048 work; the first `matmul_block` with a large shape hangs. Likely deadlock in multi-core CB synchronization for `matmul_block` (which uses `copy_block_matmul_partials` and cross-core semaphore sync).

**test_metal_dma** -- 8 passed (all L1-to-L1 transfers), then hung on first DRAM-backed test `dram-end_grid1-start_grid3`. DRAM DMA with multi-grid endpoints triggers a hang, likely in semaphore wait during DRAM-to-L1 multicast.

**test_metal_reductions** -- All ~67 tests that ran before timeout failed (PCC mismatch on `test_sum_*` variants). The hang likely occurs when a reduction kernel pattern triggers a deadlocking CB wait.

**test_metal_tms** -- Mixed results (~44 pass, ~44 fail) before timeout. Tile-aligned reshapes pass; non-tile-aligned ones fail with incorrect results. Hang likely on a specific reshape that triggers multi-core deadlock.

**Root cause hypothesis:** All hangs share a pattern -- they occur when kernels use multi-core execution with CB synchronization and the kernel waits on a semaphore or CB slot that never becomes available. The `matmul_block` and DRAM DMA cases are the most reproducible triggers.

### Category 2: Missing Compute Stubs (3 files, ~68 failures)

JIT compilation fails because the emulator's `jit_hw` headers don't provide stubs for certain compute intrinsics that D2M-generated kernels call.

**test_metal_virtual_grids** (39 failures):
```
error: 'abs_tile_init' was not declared in this scope
error: 'abs_tile' was not declared in this scope
```
Missing: `abs_tile_init()`, `abs_tile(dst_index)`

**test_metal_bfp8_typecast** (13 failures):
```
error: 'typecast_tile_init' was not declared in this scope
error: 'typecast_tile' was not declared in this scope
```
Missing: `typecast_tile_init()`, `typecast_tile(DataFormat, dst_index)`

**test_metal_masking** (16 of 18 failures):
```
error: 'llk_push_tiles' was not declared in this scope
error: 'llk_wait_tiles' was not declared in this scope
error: 'llk_pop_tiles' was not declared in this scope
error: 'llk_wait_for_free_tiles' was not declared in this scope
error: call of overloaded 'cb_reserve_back(uint32_t, int32_t&)' is ambiguous
```
The masking kernel's compute thread includes the real `tt_metal/hw/inc/api/compute/cb_api.h` (which calls `llk_*` functions) instead of the emulator's `jit_hw/api/cb_api.h`. Also, `cb_reserve_back` has an overload conflict between int32_t and uint32_t num_tiles parameter.

**test_metal_tensor_collapsing** (5 of 10 failures):
```
error: 'add_tiles_init' was not declared in this scope
error: 'mul_tiles_init' was not declared in this scope
error: 'exp_tile_init' was not declared in this scope
error: 'exp_tile' was not declared in this scope
```
Missing: `add_tiles_init()`, `mul_tiles_init()`, `exp_tile_init()`, `exp_tile(dst_index)`

### Category 3: Incorrect Numeric Results (3 files, ~19 failures)

Tests compile and run but produce wrong output (PCC well below 0.99 threshold).

**test_metal_matmul_higher_rank** (10/10 failures):
- PCC range: 0.08 -- 0.50 (expected >= 0.99)
- All batched matmul variants (3D, 4D) fail
- Single-core and multi-core variants both affected
- Likely cause: `matmul_tiles` stub doesn't handle batch dimension iteration -- the D2M-generated loop structure for batched matmul differs from the single-batch case

**test_metal_tilize** (4/11 failures):
- PCC range: 0.06 -- 0.08
- Failing shapes: shape0 through shape3 (the non-skipped, non-passing ones)
- 3 tests pass (aligned shapes), 4 skip (unsupported), 4 fail
- Likely cause: tilize kernel handles certain dimension combinations incorrectly

**test_metal_tensor_collapsing** (5 of 10 failures, after compile errors excluded):
- PCC ~0.50 for `non_collapsed` variants (matmul, 3d_add, 3d_multiply, 3d_exp, 4d_exp)
- Likely cause: element placement incorrect when tensor dimensions aren't collapsed to 2D

## Clean Passes

**test_metal_layout** (94/94) -- All layout transformation tests pass. This validates the core data movement infrastructure: tilize, untilize, L1/DRAM transfers, and multi-core grid distribution for layout operations.

**test_metal_allocate** (6/6) -- Buffer allocation with max-reduce and matmul operations pass. Validates allocator integration and basic compute-with-allocation flows.

## Recommended Fixes by Priority

### P0: Fix hangs (unblocks ~750+ tests)

Investigate the `matmul_block-f32-2048x2048x2048` hang first -- it's the most reproducible case. Check:
1. Whether `copy_block_matmul_partials` stub correctly signals CB completion
2. Whether semaphore increment/wait in multi-core `matmul_block` has a race condition
3. Whether DRAM DMA path correctly wakes waiting reader threads

### P1: Add missing compute stubs (unblocks ~68 tests)

Add to `include/jit_hw/api/compute/`:
- `abs_tile_init()` / `abs_tile(dst_index)` -- apply `fabsf` per element in DST
- `typecast_tile_init()` / `typecast_tile(DataFormat, dst_index)` -- format conversion in DST
- `exp_tile_init()` / `exp_tile(dst_index)` -- apply `expf` per element in DST
- `add_tiles_init()` / `mul_tiles_init()` -- no-op init stubs (actual math already in `add_tiles`/`mul_tiles`)

Fix masking CB API include: ensure compute kernels resolve `cb_wait_front` etc. from emulator stubs, not real `hw/inc/api/compute/cb_api.h`. Fix `cb_reserve_back` overload to accept both int32_t and uint32_t.

### P2: Fix batched matmul correctness (unblocks 10 tests)

The `matmul_tiles` stub likely needs to handle the case where D2M generates a batch loop around the matmul. Check whether the batch dimension is folded into the M dimension or iterated separately, and ensure DST accumulation resets between batches.

### P3: Fix tilize/tensor_collapsing numerics (unblocks ~9 tests)

Investigate specific failing shapes in tilize and tensor_collapsing to identify the dimension handling bug.
