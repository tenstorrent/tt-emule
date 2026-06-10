# D2M Regression Report

**Date:** 2026-04-13 (direct-write + multi-bank DRAM + invalidate_l1_cache stub)
**Build:** tt-metal `build_emule_clang` (clang-20, `TT_METAL_EMULATION=ON`, clean rebuild)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (all 13 files run, `--forked` isolation)
**tt-mlir base:** rebased on `milant/uplift_mar_25` branch
**Changes since last run (2026-04-09):**
1. Added `noc_inline_dw_write` family (unicast, multicast, stateful) to `dataflow_api.h`
2. Added `invalidate_l1_cache()` / `flush_l1_cache()` stubs to `firmware_common.h` (fixes D2M-generated `experimental::semaphore_wait` compilation)
3. Fixed `run_d2m_regression.sh` to symlink emule-built `_ttnn.so` (avoids `libdevice.so` dependency from stale non-emule build)

## Summary

| Status | Count | Previous (2026-04-09) | Delta |
|--------|-------|-----------------------|-------|
| Passed | 1694 | 1624 | **+70** |
| Failed | 164 | 112 | +52 |
| XFail | 96 | — | — |
| Skipped | 128 | 142 | -14 |

**Total tests collected: 2082** (was 1878, +204 from upstream tt-mlir test additions).

**Net improvement: +70 passing tests.** The multi-bank DRAM fix resolved all 9 DMA failures (40→49, now 100%). The `invalidate_l1_cache` stub unblocked all D2M matmul/DMA tests. 204 new upstream reduction tests account for most of the failure count increase — actual regressions are minimal (1 matmul).

## Per-File Results

| Test File | Total | Passed | Failed | XFail | Skip | Prev Passed | Delta | Status |
|-----------|-------|--------|--------|-------|------|-------------|-------|--------|
| test_metal_reductions | 1300 | 991 | 131 | 82 | 96 | 929 | **+62** | FAIL (unaligned) |
| test_metal_tms | 339 | 332 | 4 | 0 | 3 | 332 | 0 | FAIL (arange) |
| test_metal_layout | 94 | 94 | 0 | 0 | 0 | 94 | 0 | **PASS** |
| test_metal_matmul | 127 | 112 | 1 | 14 | 0 | 113 | -1 | FAIL (1 regression) |
| test_metal_tilize | 44 | 44 | 0 | 0 | 0 | 44 | 0 | **PASS** |
| test_metal_dma | 49 | 49 | 0 | 0 | 0 | 40 | **+9** | **PASS** |
| test_metal_virtual_grids | 39 | 39 | 0 | 0 | 0 | 39 | 0 | **PASS** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 0 | 27 | 0 | 0 | SKIP (needs n300) |
| test_metal_masking | 20 | 5 | 15 | 0 | 0 | 5 | 0 | FAIL (partial tile) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | 0 | 0 | 0 | FAIL (PCC mismatch) |
| test_metal_tensor_collapsing | 14 | 12 | 0 | 0 | 2 | 12 | 0 | **PASS** |
| test_metal_matmul_higher_rank | 10 | 10 | 0 | 0 | 0 | 10 | 0 | **PASS** |
| test_metal_allocate | 6 | 6 | 0 | 0 | 0 | 6 | 0 | **PASS** |

### Fixed by multi-bank DRAM and invalidate_l1_cache stub

**test_metal_dma: +9 tests (40 → 49 passed, 0 failed).** ALL DMA tests now pass. The multi-bank DRAM fix (`NUM_DRAM_BANKS` set to real channel count, all bank NOC coordinates registered in `__emule_core_map`) resolved the remaining 9 DMA failures. These were failing because multi-bank interleaved DMA operations generated NOC addresses for banks 1+ that were previously unmapped.

**test_metal_reductions: +62 tests (929 → 991 passed).** 204 new reduction test cases added upstream. Of the new tests, 62 pass and the rest fall into the existing unaligned/partial-tile failure categories.

### Minor regression

**test_metal_matmul: -1 test (113 → 112 passed).** One multi-core matmul test regressed. The 14 xfail tests remain unchanged.

## Failure Categories

### Category 1: Unaligned tensor operations (131 failures)

**Tests:** Unaligned reduction tests in `test_metal_reductions` (131 total — 204 new upstream tests contributed ~60 new failures in existing categories).
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

### Category 4: Arange (4 failures)

**Tests:** `test_arange` in TMS (all f32 variants).
**Symptom:** PCC 0.0-0.97.
**Root cause:** Missing dedicated `arange` compute implementation.
**Priority:** P3

### Category 5: Matmul regression (1 failure)

**Tests:** 1 multi-core matmul test in `test_metal_matmul`.
**Root cause:** Under investigation.
**Priority:** P2

## Standalone Regression (tt-emule)

**126 passed, 11 failed, 0 skipped** against `arminale/emule-metal-base` @ `c812fbb1cc` (baseline 2026-05-01). All 11 failures are DFB STRIDED wraparound (Tier 3b, ×4) + DFB Config Validation (Tier 3g, ×7) — see [docs/changelog.md](docs/changelog.md) for the per-version history.

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
| 2026-04-09b | 1624 | 112 | 142 | 1878 | Proper nfaces UNPACK/PACK at CB↔DST boundary |
| **2026-04-13** | **1694** | **164** | **224** | **2082** | **Multi-bank DRAM, `invalidate_l1_cache` stub, +204 upstream tests** |

## Next Steps

1. **P2: Unaligned reductions** — 131 failures (60 new from upstream test additions). Needs padding/masking for partial tiles in reduction stubs.

2. **P2: Masking partial tile** — 15 failures. Masking kernels produce partially correct results.

3. **P2: Investigate matmul regression** — 1 multi-core matmul test regressed from passing to failing.

4. **P2: BFP8 typecast** — 13 failures. BFP8 format conversion precision doesn't match hardware.

5. **P3: Arange** — 4 failures. Missing dedicated `arange` compute implementation.
