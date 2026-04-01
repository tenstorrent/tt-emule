# D2M Regression Report

**Date:** 2026-03-30
**Build:** tt-metal `build_emule_clang` (clang-17, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (all 13 files run, `--forked` isolation)
**tt-mlir base:** rebased on `milant/uplift_mar_25` branch
**Changes since last run (2026-03-17):** LLK API stubs for new D2M code generation (pack_untilize, llk_unpack_A, llk_wait_tiles/pop_tiles/push_tiles/wait_for_free_tiles, SFPU ops), int32_t CB overloads, coordinate APIs

## Summary

| Status | Count | Previous (2026-03-17) | Delta |
|--------|-------|-----------------------|-------|
| Passed | 1589 | 832 | **+757** |
| Failed | 147 | 162 | **-15** |
| Skipped/XFail | 142 | 81 | +61 |

**Total tests collected: 1878** (up from 1075 in 2026-03-17 — the tt-mlir uplift added 803 new tests across all files).

**Net improvement: +757 new passing tests, -15 fewer failures.** One regression: `test_metal_masking` dropped from 2 to 0 passes (the tt-mlir uplift changed the generated masking kernels to use new untilize paths that crash at runtime). All other previously-passing tests continue to pass.

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Prev Passed | Delta | Status |
|-----------|-------|--------|--------|------------|-------------|-------|--------|
| test_metal_layout | 94 | 94 | 0 | 0 | 94 | 0 | **PASS** |
| test_metal_matmul | 127 | 113 | 0 | 14 xfail | 108 | **+5** | **PASS** |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | 10 | 0 | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | 6 | 0 | **PASS** |
| test_metal_tms | 339 | 332 | 4 | 3 skip | 169 | **+163** | PASS (arange only) |
| test_metal_reductions | 1096 | 932 | 68 | 96 skip | 420 | **+512** | FAIL (unaligned) |
| test_metal_dma | 49 | 39 | 10 | 0 | 16 | **+23** | FAIL (DRAM aborts) |
| test_metal_tilize | 44 | 12 | 32 | 0 | 3 | **+9** | FAIL (untilize/tilize PCC) |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 2 skip | 2 | **+10** | **PASS** |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | 0 | **+39** | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | 0 | 0 | SKIP (needs n300) |
| test_metal_masking | 20 | 0 | 20 | 0 | 2 | **-2** | FAIL (SIGILL/signal 0) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | 0 | 0 | FAIL (PCC mismatch) |

### Regressions

**test_metal_masking: -2 tests (2 → 0 passed).** Previously 2 `test_complete_tile_masking` tests passed. After the tt-mlir uplift, the generated masking kernels now use `llk_wait_tiles`/`llk_pop_tiles` and `experimental::pack_untilize_block` paths that crash at runtime (SIGILL and signal 0). The JIT compile errors from the previous run are resolved, but the runtime behavior introduced new crashes. These tests need investigation of the masking kernel's interaction with the untilize path.

### Key Wins

1. **test_metal_virtual_grids: 0 → 39 (all pass).** Previously failed on JIT compile (`abs_tile` not declared). Now all 39 tests pass with the new `abs_tile` SFPU implementation.

2. **test_metal_tensor_collapsing: 2 → 12 (all non-skipped pass).** Previously failed on JIT compile (`exp_tile`, `abs_tile` not declared). Now passes with the new SFPU stubs.

3. **test_metal_tms: 169 → 332.** Suite expanded from 173 to 339 tests. 163 new tests pass. Only 4 `test_arange` failures remain (unchanged).

4. **test_metal_reductions: 420 → 932.** Suite expanded from 548 to 1096 tests. 512 new tests pass. Failures remain in `*_unaligned` variants (masking issue) plus 1 JIT compile failure.

5. **test_metal_dma: 16 → 39.** 23 new passing tests. DRAM-related crashes reduced. 10 failures remain (signal 6 SIGABRT, 1 JIT compile).

6. **test_metal_matmul: 108 → 113.** 5 previously-failing double-buffered tests now pass. 14 xfail (expected) remain.

7. **test_metal_tilize: 3 → 12.** `test_tilize_untilize` (12/12) now all pass with the `pack_untilize_block` fix. `test_untilize` (16) and `test_tilize` (16) still fail with PCC mismatch.

8. **test_metal_allocate: 6 → 6.** Stable. During the uplift, these tests initially crashed (SIGILL) due to uninitialized `__llk_pack_block_c`/`__llk_pack_offset` in `experimental::pack_untilize_block`. Fixed in this run; all 6 continue to pass.

## Failure Categories

### Category 1: Unaligned tensor operations (67 failures)

**Tests:** `test_sum_unaligned` (22), `test_max_unaligned` (25), `test_mean_unaligned` (20).
**Symptom:** PCC mismatch for non-tile-aligned reduction shapes.
**Root cause:** Padding/masking not implemented for partial tiles in reduction stubs.
**Priority:** P2

### Category 2: Masking runtime crashes (20 failures)

**Tests:** All `test_metal_masking` tests.
**Symptom:** SIGILL (signal 4) for `test_multicore_partial_tile_masking`, signal 0 for `test_complete_tile_masking`.
**Root cause:** Masking kernels use the `experimental::pack_untilize_block` path combined with masking-specific CB operations. The runtime execution triggers an illegal instruction — likely a DST index out of bounds or uninitialized state in the masking-untilize interaction.
**Priority:** P1

### Category 3: Tilize/Untilize PCC mismatch (32 failures)

**Tests:** `test_untilize` (16), `test_tilize` (16) in `test_metal_tilize.py`.
**Symptom:** PCC ~0.06-0.08 (near zero correlation). `test_tilize_untilize` (round-trip) passes.
**Root cause:** The standalone untilize and tilize operations produce incorrect data layouts. The round-trip cancels out errors (tilize then untilize returns to original), masking the bug. The tilize path likely has a face-ordering or row-stride issue in `__llk_pack_untilize` or `__llk_unpack_tilize`.
**Priority:** P1

### Category 4: BFP8 typecast PCC mismatch (13 failures)

**Tests:** All `test_metal_bfp8_typecast` tests.
**Symptom:** PCC from -0.01 to 0.72 depending on test. Tests compile and run (previously JIT compile failure).
**Root cause:** The BFP8 format conversion in `typecast_tile` and/or the `exp_tile` precision may not match hardware behavior. The matmul tests in this file get PCC ~-0.01, suggesting the typecast from BFP8 to BF16 in the inner loop produces garbage.
**Priority:** P2

### Category 5: DMA DRAM crashes (10 failures)

**Tests:** `test_roundtrip_dma_tiled[dram-*]`, `test_roundtrip_dma_rowmajor[dram-*]`, `test_interleaved_dma[*]`.
**Symptom:** SIGABRT (signal 6) and 1 JIT compile failure.
**Root cause:** DRAM NOC address resolution fails for DMA-specific access patterns. L1 DMA paths all pass (39/39).
**Priority:** P1

### Category 6: Arange (4 failures)

**Tests:** `test_arange` in TMS.
**Symptom:** PCC 0.0-0.97 depending on parameters.
**Root cause:** Missing dedicated `arange` compute implementation.
**Priority:** P3

### Category 7: JIT compile failure (1 failure)

**Tests:** `test_sum[bf16-ttmetal-True-dim_arg0-2-4]` in reductions.
**Symptom:** g++ compilation failure.
**Root cause:** A specific reduction variant emits code that references an undeclared symbol. Isolated failure.
**Priority:** P3

## Clean Passes (8 fully-passing files)

- **test_metal_layout** (94/94) — All layout, reblocking, and view composition tests.
- **test_metal_matmul** (113/113 + 14 xfail) — All matmul variants including double-buffered. **Improved from 108.**
- **test_metal_matmul_higher_rank** (10/10) — All 3D/4D batched matmul.
- **test_metal_allocate** (6/6) — All allocate tests. **Fixed from crash.**
- **test_metal_tms** (332/335 + 3 skip) — All TMS except arange. **Massive expansion.**
- **test_metal_tensor_collapsing** (12/12 + 2 skip) — **Fully fixed from JIT compile failure.**
- **test_metal_virtual_grids** (39/39) — **Fully fixed from JIT compile failure.**
- **test_metal_virtual_grid_rowmajor** (27 skip) — N300-only, correctly skipped.

## Standalone Regression (tt-emule)

All standalone tests pass: **3/3 (eltwise_add, matmul, tilize)**

## Implementation Changes (this run)

### New jit_hw additions for D2M LLK API support

1. **Pack untilize LLKs** (`pack_untilize.h`, `llk_defs.h`):
   - Preempt guard for `experimental_pack_untilize_llks.h` verbatim injection
   - `experimental::pack_untilize_block` using `copy_tile` + `__llk_pack_untilize`
   - Proper `__llk_pack_block_c` and `__llk_pack_offset` initialization
   - Templated `pack_untilize_init` overload

2. **LLK unpack/pack stubs** (`llk_defs.h`):
   - `llk_unpack_A` template stub (no-op for emulation)
   - `FACE_R_DIM`, `TILE_C_DIM` constants
   - `get_output_id`, `get_output_partial_face`, etc.

3. **LLK CB functions** (`llk_defs.h`, `cb_api.h`):
   - `llk_wait_tiles`, `llk_pop_tiles`, `llk_push_tiles`, `llk_wait_for_free_tiles`
   - `int32_t` overloads for `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front`

4. **Coordinate APIs** (`compute/common.h`):
   - `get_absolute_logical_x`, `get_absolute_logical_y`

5. **SFPU operations** (`eltwise_unary/*.h`):
   - `abs_tile` / `abs_tile_init` / `abs_tile_int32`
   - `exp_tile` / `exp_tile_init` (with `InputClamping`, `VectorMode` enums)
   - `negative_tile` / `negative_tile_init` / `negative_tile_int32`
   - `typecast_tile` / `typecast_tile_init`

## Historical Progress

| Date | Passed | Failed | Skip | Total | Key Change |
|------|--------|--------|------|-------|------------|
| 2026-03-11 | 128 | ~310 | — | ~438 | Initial D2M regression |
| 2026-03-13 | 208 | ~251 | — | ~459 | Multicast NOC fixes, parallel JIT |
| 2026-03-16a | 249 | ~244 | — | ~493 | HAL-based semaphore base |
| 2026-03-16b | 614 | 84 | — | ~698 | DRAM bank offset fix |
| 2026-03-17 | 832 | 162 | 81 | 1075 | Lock-free CB, AVX2 matmul, --forked |
| **2026-03-30** | **1589** | **147** | **142** | **1878** | **LLK API stubs, tt-mlir uplift (+803 tests)** |
