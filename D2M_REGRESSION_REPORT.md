# D2M Regression Report

**Date:** 2026-04-09
**Build:** tt-metal `build_emule_clang` (clang-20, `TT_METAL_EMULATION=ON`, clean rebuild)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (all 13 files run, `--forked` isolation)
**tt-mlir base:** rebased on `milant/uplift_mar_25` branch
**Changes since last run (2026-04-08):** Removed nfaces tile layout conversion from WH/BH compute ops (kept PACK auto-advance and INT32 bit preservation). Nfaces permutation broke matmul (0/113) because it doesn't cancel for non-element-wise ops. Clean rebuild with correct cmake flags per AGENT_BUILD_GUIDE.md.

## Summary

| Status | Count | Previous (2026-04-08) | Delta |
|--------|-------|-----------------------|-------|
| Passed | 1595 | 1256 | **+339** |
| Failed | 141 | 480 | **-339** |
| Skipped/XFail | 142 | 142 | 0 |

**Total tests collected: 1878** (unchanged).

**Net improvement: +339 passing tests.** Root cause of previous regression was the unconditional nfaces tile layout conversion added in ec06172. Removing it restored all matmul tests (0→113), TMS tests (193→332), virtual_grids (0→39), tensor_collapsing (0→12), and others.

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Prev Passed | Delta | Status |
|-----------|-------|--------|--------|------------|-------------|-------|--------|
| test_metal_reductions | 1096 | 934 | 66 | 96 skip | 934 | 0 | FAIL (unaligned) |
| test_metal_tms | 339 | 332 | 4 | 3 skip | 193 | **+139** | FAIL (arange) |
| test_metal_layout | 94 | 94 | 0 | 0 | 91 | **+3** | **PASS** |
| test_metal_matmul | 127 | 113 | 0 | 14 xfail | 0 | **+113** | **PASS** |
| test_metal_tilize | 44 | 12 | 32 | 0 | 16 | -4 | FAIL (untilize/tilize PCC) |
| test_metal_dma | 49 | 38 | 11 | 0 | 15 | **+23** | FAIL (DMA crashes) |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | 0 | **+39** | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | 0 | 0 | SKIP (needs n300) |
| test_metal_masking | 20 | 5 | 15 | 0 | 3 | **+2** | FAIL (partial tile) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | 0 | 0 | FAIL (PCC mismatch) |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 2 skip | 0 | **+12** | **PASS** |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | 0 | **+10** | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | 4 | **+2** | **PASS** |

### Restored by nfaces removal

**test_metal_matmul: +113 tests (0 → 113 passed).** All matmul variants restored. The nfaces permutation scrambled matrix element positions, producing PCC ~0.20-0.30. With direct indexing, matmul accumulates correctly.

**test_metal_matmul_higher_rank: +10 tests (0 → 10 passed).** Same root cause as matmul.

**test_metal_tms: +139 tests (193 → 332 passed).** reshape, permute, and concat_heads tests restored. Only 4 arange failures remain (pre-existing).

**test_metal_virtual_grids: +39 tests (0 → 39 passed).** All eltwise multi-core tests restored.

**test_metal_tensor_collapsing: +12 tests (0 → 12 passed).** All non-skipped tests pass.

**test_metal_dma: +23 tests (15 → 38 passed).** Tiled DMA tests restored. 11 failures remain (DMA crashes — SIGABRT/signal 0).

**test_metal_layout: +3 tests (91 → 94 passed).** Tiled grid reblocking tests restored.

**test_metal_allocate: +2 tests (4 → 6 passed).** All allocate tests pass.

**test_metal_masking: +2 tests (3 → 5 passed).** Minor improvement.

### Regressions

**test_metal_tilize: -4 tests (16 → 12 passed).** 4 tilize/untilize tests regressed. These may have depended on the nfaces permutation cancelling errors in round-trip scenarios.

## Failure Categories

### Category 1: Unaligned tensor operations (66 failures)

**Tests:** `test_sum_unaligned`, `test_max_unaligned`, `test_mean_unaligned` in reductions.
**Symptom:** PCC mismatch for non-tile-aligned reduction shapes.
**Root cause:** Padding/masking not implemented for partial tiles in reduction stubs.
**Priority:** P2

### Category 2: Tilize/Untilize PCC mismatch (32 failures)

**Tests:** `test_untilize` (20), `test_tilize` (12) in `test_metal_tilize.py`.
**Symptom:** PCC near zero. Round-trip (`test_tilize_untilize`) may pass but standalone operations fail.
**Root cause:** Face-ordering or row-stride issues in tilize/untilize stubs.
**Priority:** P1

### Category 3: Masking partial tile PCC (15 failures)

**Tests:** 15 of 20 `test_metal_masking` tests.
**Symptom:** PCC ranges from 0.0 to 0.92.
**Root cause:** Masking kernels produce partially correct results. The 5 passing tests use simple padding values on complete tiles.
**Priority:** P2

### Category 4: BFP8 typecast PCC mismatch (13 failures)

**Tests:** All `test_metal_bfp8_typecast` tests.
**Symptom:** PCC from -0.01 to 0.72.
**Root cause:** BFP8 format conversion precision doesn't match hardware.
**Priority:** P2

### Category 5: DMA crashes (11 failures)

**Tests:** DRAM DMA tests in `test_metal_dma.py`.
**Symptom:** SIGABRT (signal 6) and signal 0.
**Root cause:** DRAM NOC address resolution fails for specific DMA access patterns.
**Priority:** P1

### Category 6: Arange (4 failures)

**Tests:** `test_arange` in TMS.
**Symptom:** PCC 0.0-0.97.
**Root cause:** Missing dedicated `arange` compute implementation.
**Priority:** P3

## Standalone Regression (tt-emule)

**83 passed, 28 failed, 2 skipped**

### Passing (83)

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
| 5 | ttnn_matmul_sweep | 1 |

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

### Quasar Matmul Integration (NOT in regression script)

`test_matmul_X_tile`: **0 passed, 2 failed, 3 skipped.** `TensixMatmulBlock` and `TensixMatmulBlockInitShort` fail with data corruption (result contains raw bf16 values where float32 expected). Needs nfaces conversion on the Quasar/DFB path — to be addressed as conditional nfaces support.

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
| **2026-04-09** | **1595** | **141** | **142** | **1878** | **Remove nfaces from WH/BH compute ops** |

## Next Steps

1. **P0: Conditional nfaces for Quasar DFB path.** The nfaces conversion is needed for Quasar matmul (`test_matmul_X_tile`) but breaks WH/BH. Need runtime-conditional nfaces gated on DFB activation (`__emule_dfbs != nullptr`). All 27 DM-only DFB failures are pre-existing and unrelated.

2. **P1: Fix tilize/untilize standalone PCC** — face-ordering issue (32 D2M failures).

3. **P1: Fix DMA crashes** — DRAM NOC address resolution (11 D2M failures).

4. **P2: Unaligned reductions, masking, BFP8** — precision/padding issues (94 D2M failures).
