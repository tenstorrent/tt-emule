# D2M Regression Report

**Date:** 2026-03-16
**Build:** tt-metal `build_emule_clang` (clang-17, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (13 files, ~1146 total tests)
**Fixes since last run:** HAL-based semaphore base (kernel_config_base + sem_offset from ProgramConfig, replacing dynamic max_cb_end scan)

## Summary

| Status | Count |
|--------|-------|
| Passed | 249 |
| Failed | ~244 |
| Skipped/XFail | 56 |
| Aborted (SIGABRT) | 1 file (~41 tests unreachable) |
| Hung (timeout) | 3 files (~597 tests unreachable) |

**Improvement vs previous run (2026-03-13):** +41 passed tests (249 vs 208). Matmul abort resolved — now passes 56 tests (up from 15) including `matmul_block-{bf16,f32}-2048x2048x2048` with L1 accumulation. Matmul now times out at 48% instead of aborting at 14%.

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Status |
|-----------|-------|--------|--------|------------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | PASS |
| test_metal_allocate | 6 | 6 | 0 | 0 | PASS |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | SKIP (needs n300) |
| test_metal_matmul | 127 | 56 | 0 | 5 xfail | HUNG at 48% (timeout, 0 failures — all executed tests pass) |
| test_metal_dma | 49 | 8 | 0 | 0 | ABORT at test 9 (first DRAM-backed DMA) |
| test_metal_matmul_higher_rank | 10 | 0 | 10 | 0 | FAIL (golden PCC mismatch) |
| test_metal_tilize | 11 | 3 | 4 | 4 skip | FAIL (golden PCC mismatch) |
| test_metal_masking | 20 | 2 | 18 | 0 | FAIL (JIT compile errors) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | FAIL (JIT compile errors) |
| test_metal_tensor_collapsing | 14 | 0 | 10 | 2 skip, 2 xfail | FAIL (mix: JIT errors + golden mismatch) |
| test_metal_virtual_grids | 39 | 0 | 39 | 0 | FAIL (JIT compile errors) |
| test_metal_tms | 173 | 68 | 64 | 0 | HUNG at ~76% (reshapes pass, permutes fail) |
| test_metal_reductions | 548 | 12 | 86 | 16 skip | HUNG at ~21% (all sum variants fail) |

## What Changed (2026-03-16)

### Semaphore base aligned with hardware kernel config layout

Replaced the dynamic `max_cb_end` pre-scan (~25 lines scanning all CBs to compute `EMULE_SEM_BASE`) with a 5-line HAL query:

```cpp
kernel_config_base + prog_config.sem_offset
```

Where `kernel_config_base` is `MEM_MAP_END` (~0x8730 on Wormhole) and `sem_offset` is computed by `finalize_sems()` (already called before `execute_program_emulated`). This matches exactly how real firmware computes semaphore addresses.

### Matmul abort resolved → timeout

**test_metal_matmul** previously aborted (SIGABRT) at test 18 (`matmul_block-f32-2048x2048x2048`). With the HAL-based semaphore base, this test now **passes** — the semaphore address is correctly placed in the kernel config region, eliminating the CB/semaphore overlap that caused the abort. The file now reaches 48% (56 passed, 5 xfail) before timeout. All executed tests pass with zero failures.

### Layout/Allocate/DMA stable

Layout (94/94) and Allocate (6/6) still 100% pass. DMA still 8/8 L1 tests pass before DRAM abort.

## Failure Categories

### Category 1: Abort (1 file, ~41 tests unreachable)

**test_metal_dma** — 8 passed (all L1-to-L1 DMA transfers). Process aborts on first DRAM-backed test (`dram-end_grid1-start_grid3-shape0`). DRAM DMA path triggers an assertion failure during NOC address resolution for DRAM cores.

### Category 2: JIT Compile Errors (4 files, ~70 failures)

Missing compute stub declarations cause g++ to fail during JIT compilation.

**test_metal_virtual_grids** (39 failures): Missing `abs_tile_init()`, `abs_tile(dst_index)`

**test_metal_bfp8_typecast** (13 failures): Missing `typecast_tile_init()`, `typecast_tile(DataFormat, dst_index)`

**test_metal_masking** (16 of 18 failures): Masking kernel's compute thread resolves `cb_wait_front` etc. from real `hw/inc/api/compute/cb_api.h` (which calls `llk_*` functions) instead of the emulator's stub. Also `cb_reserve_back` overload conflict between `int32_t` and `uint32_t`.

**test_metal_tensor_collapsing** (8 of 10 failures): Mix of missing stubs (`add_tiles_init`, `mul_tiles_init`, `exp_tile_init`, `exp_tile`) and golden PCC mismatch for matmul variants.

### Category 3: Incorrect Numeric Results (4 files, ~164 failures)

Tests compile and run but produce wrong output (PCC well below 0.99 threshold).

**test_metal_reductions** (86 failures): All `test_sum_unaligned` and `test_sum_4d` variants fail. The reduction compute kernel produces near-zero PCC. Likely cause: the reduce_tile stub doesn't correctly accumulate partial sums across the reduction dimension.

**test_metal_matmul_higher_rank** (10 failures): PCC range 0.02--0.15 for all batched matmul (3D, 4D). The batch loop in D2M-generated kernels handles accumulation differently from the single-batch case that passes.

**test_metal_tms** (64 failures): Non-tile-aligned reshapes and all permute operations fail. Tile-aligned reshapes (where input/output tile boundaries match) pass. Permute failures are likely due to missing transpose/permute compute stubs.

**test_metal_tilize** (4 failures): PCC ~0.06--0.08 for certain shapes. 3 tile-aligned shapes pass, 4 non-aligned fail.

### Category 4: Timeouts (3 files, ~597 tests unreachable)

**test_metal_matmul** — 127 tests, reached 48% (56 passed, 5 xfail) in 300s. All executed tests pass. The large matmul shapes (2048x2048x2048) take ~5s each. Increasing timeout to 600s would likely allow full completion.

**test_metal_reductions** — 548 tests, reached 21% (114 tests) in 300s. Not a kernel hang — each test takes ~2.5s and there are simply too many. Increasing timeout to 1500s would allow full completion.

**test_metal_tms** — 173 tests, reached ~76% (132 tests) in 300s. Each permute test takes ~3s; likely would complete in ~400s.

## Clean Passes

**test_metal_layout** (94/94) — All layout transformation tests pass: tilize, untilize, L1/DRAM transfers, multi-core grid distribution.

**test_metal_allocate** (6/6) — Buffer allocation with max-reduce and matmul operations.

**test_metal_virtual_grid_rowmajor** (27/27 skipped) — All skipped (requires n300 multi-chip).

**test_metal_matmul** (56/56 non-xfail pass) — All matmul_tile and matmul_block variants (bf16 and f32) pass, including L1 accumulation and 2048x2048x2048 shapes. This is up from 15 in the previous run.

**test_metal_dma** (8/8 before abort) — All L1-to-L1 DMA roundtrip tests pass.

**test_metal_tms** (68 pass) — All tile-aligned reshapes pass (e.g., 64x64->32x128, 96x64->64x96).

## Recommended Fixes by Priority

### P0: Increase timeout + fix DRAM DMA abort (unblocks ~638 tests)

1. **Increase timeout to 600s** for matmul, reductions, and TMS — these aren't hanging, just slow with many tests. This alone would expose ~597 additional test results. Matmul is particularly valuable as all executed tests currently pass.
2. **Fix SIGABRT in DRAM DMA** — investigate the assertion failure in NOC address resolution for DRAM cores. L1-to-L1 path works; DRAM-backed path fails.

### P1: Add missing compute stubs (unblocks ~70 tests)

Add to `include/jit_hw/api/compute/`:
- `abs_tile_init()` / `abs_tile(dst_index)` — `fabsf` per element in DST
- `typecast_tile_init()` / `typecast_tile(DataFormat, dst_index)` — format conversion in DST
- `exp_tile_init()` / `exp_tile(dst_index)` — `expf` per element in DST
- `add_tiles_init()` / `mul_tiles_init()` — no-op init stubs

Fix masking CB API include path: ensure compute kernels resolve CB functions from emulator stubs, not real `hw/inc/`. Fix `cb_reserve_back` overload for `int32_t`/`uint32_t`.

### P2: Fix reduction numerics (unblocks ~86 tests)

The `reduce_tile` stub likely doesn't handle partial-sum accumulation correctly for D2M-generated reduction loops. Check whether the reduction dimension stride and accumulation target in DST match what the kernel expects.

### P3: Fix batched matmul + permute correctness (unblocks ~74 tests)

- Batched matmul: DST accumulation may not reset between batches, or the batch loop structure differs from single-batch.
- Permute: likely needs a `transpose_wh` or general permute stub that rearranges tile data in DST/CB.

## Historical Progress

| Date | Passed | Failed | Key Change |
|------|--------|--------|------------|
| 2026-03-11 | 128 | ~310 | Initial D2M regression |
| 2026-03-13 | 208 | ~251 | Multicast NOC fixes, parallel JIT, cross-program core_map |
| 2026-03-16 | 249 | ~244 | HAL-based semaphore base; matmul abort resolved (+41 matmul passes) |
