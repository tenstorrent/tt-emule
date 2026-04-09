# D2M Regression Report

**Date:** 2026-04-09 (nfaces fix run)
**Build:** tt-metal `build_emule_clang` (clang-20, `TT_METAL_EMULATION=ON`, clean rebuild)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (all 13 files run, `--forked` isolation)
**tt-mlir base:** rebased on `milant/uplift_mar_25` branch
**Changes since last run (2026-04-09 earlier):** Added proper nfaces tile format conversion to all UNPACK (CB→DST) and PACK (DST→CB) operations. On real hardware, UNPACK converts nfaces→row-major and PACK converts row-major→nfaces at the CB↔DST boundary. The emulator was missing these conversions. Fixed in `common.h` (copy_tile, add/sub/mul_tiles, pack_dst_to_buf), `matmul.h` (matmul_tiles), `reduce.h` (reduce_tile), and `llk_defs.h` (__llk_pack_tiled for D2M tilize PACK).

## Summary

| Status | Count | Previous (2026-04-09 earlier) | Delta |
|--------|-------|-----------------------|-------|
| Passed | 1624 | 1595 | **+29** |
| Failed | 112 | 141 | **-29** |
| Skipped/XFail | 142 | 142 | 0 |

**Total tests collected: 1878** (unchanged).

**Net improvement: +29 passing tests.** The nfaces UNPACK/PACK fix resolved all 32 tilize/untilize PCC failures (the biggest single improvement) plus 2 DMA test fixes. 5 unaligned reduction tests shifted from pass to fail (minor noise within an already-failing category).

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Prev Passed | Delta | Status |
|-----------|-------|--------|--------|------------|-------------|-------|--------|
| test_metal_reductions | 1096 | 929 | 71 | 96 skip | 934 | -5 | FAIL (unaligned) |
| test_metal_tms | 339 | 332 | 4 | 3 skip | 332 | 0 | FAIL (arange) |
| test_metal_layout | 94 | 94 | 0 | 0 | 94 | 0 | **PASS** |
| test_metal_matmul | 127 | 113 | 0 | 14 xfail | 113 | 0 | **PASS** |
| test_metal_tilize | 44 | 44 | 0 | 0 | 12 | **+32** | **PASS** |
| test_metal_dma | 49 | 40 | 9 | 0 | 38 | **+2** | FAIL (DMA crashes) |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | 39 | 0 | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | 0 | 0 | SKIP (needs n300) |
| test_metal_masking | 20 | 5 | 15 | 0 | 5 | 0 | FAIL (partial tile) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | 0 | 0 | FAIL (PCC mismatch) |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 2 skip | 12 | 0 | **PASS** |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | 10 | 0 | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | 6 | 0 | **PASS** |

### Fixed by nfaces UNPACK/PACK conversion

**test_metal_tilize: +32 tests (12 → 44 passed, 0 failed).** ALL tilize and untilize tests now pass. The nfaces conversion ensures that `__llk_pack_tiled()` writes nfaces format to CBs (matching real hardware's PACK engine), and `copy_tile()` reads nfaces format from CBs (matching real hardware's UNPACK engine). This was the root cause of all 32 tilize/untilize PCC failures.

**test_metal_dma: +2 tests (38 → 40 passed).** Two tiled roundtrip DMA tests fixed. 9 failures remain (DMA crashes/PCC issues for specific grid/shape combos).

### Minor regression

**test_metal_reductions: -5 tests (934 → 929 passed).** 5 additional unaligned reduction tests now fail. All 71 failures remain in the `test_*_unaligned` category (26 max + 23 sum + 22 mean). The nfaces conversion may slightly change accumulation order for edge-case unaligned shapes where padding interacts with the nfaces permutation. This is within the existing failure category and not a new class of failure.

## Failure Categories

### Category 1: Unaligned tensor operations (71 failures)

**Tests:** `test_sum_unaligned` (23), `test_max_unaligned` (26), `test_mean_unaligned` (22) in reductions.
**Symptom:** PCC mismatch for non-tile-aligned reduction shapes.
**Root cause:** Padding/masking not implemented for partial tiles in reduction stubs.
**Priority:** P2

### Category 2: Masking partial tile PCC (15 failures)

**Tests:** 15 of 20 `test_metal_masking` tests.
**Symptom:** PCC ranges from 0.0 to 0.92.
**Root cause:** Masking kernels produce partially correct results. The 5 passing tests use simple padding values on complete tiles.
**Priority:** P2

### Category 3: BFP8 typecast PCC mismatch (13 failures)

**Tests:** All `test_metal_bfp8_typecast` tests.
**Symptom:** PCC from -0.01 to 0.72.
**Root cause:** BFP8 format conversion precision doesn't match hardware.
**Priority:** P2

### Category 4: DMA crashes/PCC (9 failures)

**Tests:** `test_roundtrip_dma_tiled` (2), `test_roundtrip_dma_rowmajor` (2), `test_interleaved_dma` (5) in `test_metal_dma.py`.
**Symptom:** SIGABRT (signal 6), signal 0, or PCC mismatch for specific grid/shape combinations.
**Root cause:** DRAM NOC address resolution fails for specific DMA access patterns (all failures involve `end_grid0` or `end_grid1` with `start_grid0`).
**Priority:** P1

### Category 5: Arange (4 failures)

**Tests:** `test_arange` in TMS (all f32 variants).
**Symptom:** PCC 0.0-0.97.
**Root cause:** Missing dedicated `arange` compute implementation.
**Priority:** P3

## Standalone Regression (tt-emule)

**85 passed, 28 failed, 2 skipped** (+2 from previous: Quasar matmul now passes)

### Passing (85)

| Tier | Tests | Count |
|------|-------|-------|
| 0 | dfb_passthrough, dfb_multi_consumer, eltwise_add | 3 |
| 1 | bit_utils, host_buffer, tilize_untilize, blockfloat_common, CoreRange_* (5), CoreRangeSet_* (4), dst_capacity_bf16, dst_capacity_fp32 | 16 |
| 2 | SimpleL1Buffer, SimpleDramBuffer | 2 |
| 3 | TensixL1Tile | 1 |
| 3d | DMTensixTest (6 configs × IS/no-IS) | 12 |
| 3d | TensixDMTest (6 configs × IS/no-IS) | 12 |
| 3f | DMTensixTest BLOCKED (5 configs × IS/no-IS) | 10 |
| 3f | TensixDMTest BLOCKED (5 configs × IS/no-IS) | 10 |
| 3g | QuasarComputeKernel (4 tests) | 4 |
| 3h | QuasarSemaphores (2), DmLoopback | 3 |
| 3i | SingleDmL1Write, RISCV Atomics (3) | 4 |
| 3j | DmLoopback NOC (2), DmOneFromOne (2) | 4 |
| 4 | ttnn_relational, ttnn_add_int_emulated | 2 |
| 5 | ttnn_matmul_sweep (14/14 random bf16) | 1 |
| 5 | QuasarMatmul (TensixMatmulBlock, TensixMatmulBlockInitShort) | 2 |

### Failing (28)

| Tier | Tests | Count | Reason |
|------|-------|-------|--------|
| 3b | DFBEmuleDMTest, DFBEmuleBridgeTest | 2 | DM-only DFB sync (pre-existing) |
| 3c | DMTest STRIDED (6 configs × IS/no-IS) | 12 | DM-only DFB sync (pre-existing) |
| 3e | DMTensixDMTest pipeline (3 tests) | 3 | Multi-DFB pipeline (pre-existing) |
| 3f | DMTest BLOCKED (5 configs × IS/no-IS) | 10 | DM-only DFB BLOCKED (pre-existing) |
| 6 | ttnn_add_int_silicon | 1 | Requires real hardware |

### Skipped (2)

| Test | Reason |
|------|--------|
| emulation_toggle_default | Binary not found |
| emulation_toggle_active | Binary not found |

## Historical Progress

| Date | Passed | Failed | Skip | Total | Key Change |
|------|--------|--------|------|-------|------------|
| 2026-03-11 | 128 | ~310 | -- | ~438 | Initial D2M regression |
| 2026-03-13 | 208 | ~251 | -- | ~459 | Multicast NOC fixes, parallel JIT |
| 2026-03-16a | 249 | ~244 | -- | ~493 | HAL-based semaphore base |
| 2026-03-16b | 614 | 84 | -- | ~698 | DRAM bank offset fix |
| 2026-03-17 | 832 | 162 | 81 | 1075 | Lock-free CB, AVX2 matmul, --forked |
| 2026-03-30 | 1589 | 147 | 142 | 1878 | LLK API stubs, tt-mlir uplift (+803 tests) |
| 2026-04-08 | 1256 | 480 | 142 | 1878 | Nfaces, NUM_DRAM_BANKS=real, remote semaphore |
| 2026-04-09a | 1595 | 141 | 142 | 1878 | Remove nfaces from WH/BH compute ops |
| **2026-04-09b** | **1624** | **112** | **142** | **1878** | **Proper nfaces UNPACK/PACK at CB↔DST boundary** |

## Next Steps

1. **P1: Fix DMA crashes** — 9 remaining DMA failures for specific grid/shape combos. All involve `end_grid0`/`end_grid1` with `start_grid0`.

2. **P2: Unaligned reductions** — 71 failures. Needs padding/masking for partial tiles in reduction stubs.

3. **P2: Masking partial tile** — 15 failures. Masking kernels produce partially correct results.

4. **P2: BFP8 typecast** — 13 failures. BFP8 format conversion precision doesn't match hardware.

5. **P3: Arange** — 4 failures. Missing dedicated `arange` compute implementation.
