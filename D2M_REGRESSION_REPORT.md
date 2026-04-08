# D2M Regression Report

**Date:** 2026-04-08
**Build:** tt-metal `build_emule_clang` (clang-20, `TT_METAL_EMULATION=ON`)
**Target:** wormhole_N150 (emulated, slow dispatch)
**tt-mlir tests:** `test/python/golden/test_metal_*.py` (all 13 files run, `--forked` isolation)
**tt-mlir base:** rebased on `milant/uplift_mar_25` branch
**Changes since last run (2026-03-30):** Nfaces tile format (4x16x16 face layout), INT32 memcpy fix, NUM_DRAM_BANKS set to real device count (6 for wormhole), remote semaphore operations, multicast NOC address resolution, thread-local compute state reset, debug noise removal

## Summary

| Status | Count | Previous (2026-03-30) | Delta |
|--------|-------|-----------------------|-------|
| Passed | 1256 | 1589 | **-333** |
| Failed | 480 | 147 | **+333** |
| Skipped/XFail | 142 | 142 | 0 |

**Total tests collected: 1878** (unchanged from 2026-03-30).

**Net regression: -333 passing tests.** Root cause: the change from `NUM_DRAM_BANKS=1` to `NUM_DRAM_BANKS=6` (real wormhole bank count) fixed standalone tests but broke D2M-generated kernels. D2M kernels use `InterleavedAddrGen` for DRAM access; with 6 banks the address computation distributes pages across banks, but the emulated host-side buffer write/read uses a single contiguous allocation. This mismatch causes kernels to read from wrong offsets, producing PCC ~0.06-0.25 (scrambled data, not garbage). Additionally, the nfaces tile format change (row-major to 4x16x16 faces) affects how D2M kernels unpack/pack tile data.

## Per-File Results

| Test File | Total | Passed | Failed | Skip/XFail | Prev Passed | Delta | Status |
|-----------|-------|--------|--------|------------|-------------|-------|--------|
| test_metal_reductions | 1096 | 934 | 66 | 96 skip | 932 | **+2** | FAIL (unaligned) |
| test_metal_tms | 339 | 193 | 143 | 3 skip | 332 | **-139** | FAIL (reshape/permute/concat_heads) |
| test_metal_layout | 94 | 91 | 3 | 0 | 94 | **-3** | FAIL (tiled_grid_reblocking) |
| test_metal_matmul | 127 | 0 | 113 | 14 xfail | 113 | **-113** | **REGRESSED** |
| test_metal_tilize | 44 | 16 | 28 | 0 | 12 | **+4** | FAIL (untilize/tilize PCC) |
| test_metal_dma | 49 | 15 | 34 | 0 | 39 | **-24** | FAIL (tiled DMA) |
| test_metal_virtual_grids | 39 | 0 | 39 | 0 | 39 | **-39** | **REGRESSED** |
| test_metal_virtual_grid_rowmajor | 27 | 0 | 0 | 27 skip | 0 | 0 | SKIP (needs n300) |
| test_metal_masking | 20 | 3 | 17 | 0 | 0 | **+3** | FAIL (partial tile) |
| test_metal_bfp8_typecast | 13 | 0 | 13 | 0 | 0 | 0 | FAIL (PCC mismatch) |
| test_metal_tensor_collapsing | 14 | 0 | 12 | 2 skip | 12 | **-12** | **REGRESSED** |
| test_metal_matmul_higher_rank | 10 | 0 | 10 | 0 | 10 | **-10** | **REGRESSED** |
| test_metal_allocate | 6 | 4 | 2 | 0 | 6 | **-2** | FAIL |

### Regressions (NUM_DRAM_BANKS / nfaces related)

**test_metal_matmul: -113 tests (113 -> 0 passed).** All matmul variants now produce PCC ~0.20-0.30. The D2M matmul kernels use `InterleavedAddrGen` to read input tiles from DRAM. With `NUM_DRAM_BANKS=6`, tile page `id` maps to `bank_index = id % 6`, but the emulated DRAM buffer is a single flat allocation. Tiles land in wrong banks, producing scrambled accumulation results. This was previously masked by `NUM_DRAM_BANKS=1` (all tiles in bank 0, sequential).

**test_metal_matmul_higher_rank: -10 tests (10 -> 0 passed).** Same root cause as matmul.

**test_metal_virtual_grids: -39 tests (39 -> 0 passed).** All eltwise tests produce PCC ~0.06. Multi-core eltwise operations read from wrong DRAM offsets. Low PCC (~0.06 vs ~0.25 for matmul) indicates nearly uncorrelated data — consistent with reading from completely wrong tile locations.

**test_metal_tensor_collapsing: -12 tests (12 -> 0 passed).** PCC ranges from -0.025 to 0.08. Same DRAM bank offset mismatch.

**test_metal_tms: -139 tests (332 -> 193 passed).** Breakdown of 143 failures:
- 100 `test_reshape` failures across all 5 dtypes (i1, i32, i64, f32, bf16) x 20 shapes — new
- 28 `test_permute` failures — new
- 11 `test_concatenate_heads` failures — new
- 4 `test_arange` failures — unchanged from previous

All reshape/permute/concat_heads failures are PCC mismatches from DRAM bank addressing issues. The 193 passing tests are single-core operations that happen to use small enough buffers to fit in bank 0.

**test_metal_dma: -24 tests (39 -> 15 passed).** All `test_roundtrip_dma_tiled` (16 L1 + 8 DRAM = 24 new failures) now fail. DMA tiled tests move tile-formatted data through DRAM, which is affected by bank addressing. The 15 passing tests are all `test_roundtrip_dma_rowmajor` (14) + `test_host_interop_single_bank_dram_dma` (1). Rowmajor DMA uses byte-level addressing unaffected by tile-bank mapping.

**test_metal_layout: -3 tests (94 -> 91 passed).** The 3 failing tests are `test_tiled_grid_reblocking` variants (multi-core tiled grid reblocking). PCC ~0.063.

**test_metal_allocate: -2 tests (6 -> 4 passed).** 2 allocate tests now fail, likely from DRAM bank offset computation.

### Improvements

**test_metal_tilize: +4 tests (12 -> 16 passed).** 4 additional `test_tilize_untilize` round-trip tests now pass with the nfaces tile format fix.

**test_metal_masking: +3 tests (0 -> 3 passed).** Three `test_complete_tile_masking` tests now pass (inf, -inf, 1.0 padding values). Previously all 20 tests crashed (SIGILL/signal 0). 17 still fail with PCC mismatch but no longer crash.

**test_metal_reductions: +2 tests (932 -> 934 passed).** 2 additional reduction tests pass.

## Failure Categories

### Category 1: DRAM bank addressing mismatch (~310 failures) -- NEW

**Tests:** All matmul (113), virtual_grids (39), tensor_collapsing (12), matmul_higher_rank (10), TMS reshape/permute/concat_heads (139), DMA tiled (24), layout reblocking (3), allocate (2).
**Symptom:** PCC 0.06-0.30 (scrambled data, not garbage).
**Root cause:** `NUM_DRAM_BANKS` changed from 1 to 6 (real wormhole count). D2M kernels use `InterleavedAddrGen` which computes `bank_index = tile_id % NUM_DRAM_BANKS` and `offset = (tile_id / NUM_DRAM_BANKS) * aligned_page_size`. The emulated DRAM allocation is flat — pages are at `base + tile_id * page_size`. The bank-interleaved address doesn't match the flat layout, so kernels read wrong data.
**Fix direction:** The emulated DRAM allocator/accessor needs to match the bank-interleaved layout that `InterleavedAddrGen` expects, or the JIT-compiled `InterleavedAddrGen` needs a flat-mode fallback for emulation.
**Priority:** P0 (blocks 333 tests)

### Category 2: Unaligned tensor operations (65 failures)

**Tests:** `test_sum_unaligned`, `test_max_unaligned`, `test_mean_unaligned` in reductions.
**Symptom:** PCC mismatch for non-tile-aligned reduction shapes.
**Root cause:** Padding/masking not implemented for partial tiles in reduction stubs.
**Priority:** P2

### Category 3: Masking partial tile PCC (17 failures)

**Tests:** 17 of 20 `test_metal_masking` tests.
**Symptom:** PCC ranges from 0.0 to 0.92. No longer crashing (improved from SIGILL/signal 0).
**Root cause:** Masking kernels produce partially correct results. The 3 passing tests use simple padding values (inf, -inf, 1.0) on complete tiles.
**Priority:** P2

### Category 4: Tilize/Untilize PCC mismatch (28 failures)

**Tests:** `test_untilize` (16), `test_tilize` (12) in `test_metal_tilize.py`.
**Symptom:** PCC near zero. Round-trip (`test_tilize_untilize`) passes (16/16).
**Root cause:** Standalone untilize/tilize operations have face-ordering or row-stride issues. Round-trip cancels out errors.
**Priority:** P1

### Category 5: BFP8 typecast PCC mismatch (13 failures)

**Tests:** All `test_metal_bfp8_typecast` tests.
**Symptom:** PCC from -0.01 to 0.72.
**Root cause:** BFP8 format conversion precision doesn't match hardware.
**Priority:** P2

### Category 6: DMA crashes (11 failures)

**Tests:** DRAM DMA tests in `test_metal_dma.py`.
**Symptom:** SIGABRT (signal 6, 7 tests) and signal 0 (4 tests).
**Root cause:** DRAM NOC address resolution fails for specific DMA access patterns.
**Priority:** P1

### Category 7: Reduction JIT compile (1 failure)

**Tests:** `test_sum[bf16-ttmetal-True-dim_arg0-2-4]`.
**Symptom:** g++ compilation failure.
**Root cause:** Specific reduction variant references undeclared symbol.
**Priority:** P3

### Category 8: Arange (4 failures)

**Tests:** `test_arange` in TMS.
**Symptom:** PCC 0.0-0.97.
**Root cause:** Missing dedicated `arange` compute implementation.
**Priority:** P3

## Standalone Regression (tt-emule)

**110 passed, 1 failed (Tier 6 silicon toggle — requires real hardware), 2 skipped (emulation_toggle binary not found)**

| Tier | Description | Status |
|------|-------------|--------|
| Tier 1 | Eltwise (add, sub, mul, sqrt, relu, gelu) | PASS |
| Tier 2 | Matmul sweep (14 shapes, 1-4 cores) | PASS |
| Tier 3 | Tilize round-trip, reduction sweep | PASS |
| Tier 3b | INT32 add | PASS (8/8) |
| Tier 3c | Relational INT32 | PASS (66/66) |
| Tier 3d | Matmul L1 accumulation | PASS |
| Tier 3e | DFB emulation (DM + Bridge) | PASS |
| Tier 3f | DFB multi-P/C BLOCKED | PASS (30/30) |
| Tier 4 | Blackhole P100 eltwise | PASS |
| Tier 5 | Multi-device cluster | PASS |
| Tier 6 | Silicon toggle | FAIL (expected — no hardware) |

## Implementation Changes (this run)

### Nfaces tile format
- `include/jit_hw/nfaces.h`: constexpr LUT converting between row-major 32x32 and 4-face 16x16 tile layout
- `include/jit_hw/api/compute/common.h`: `copy_tile()`, `pack_dst_to_buf()`, `pack_tile()` updated to use nfaces index mapping
- `include/jit_hw/api/compute/matmul.h`: matmul tile I/O uses nfaces layout

### INT32 bit pattern preservation
- `copy_tile()` and `pack_dst_to_buf()` use `uint32_t*` + `memcpy` instead of `float*` assignment for 32-bit formats (prevents x86 DAZ/FTZ denormal flush of small INT32 values)

### DRAM bank count
- `emulated_program_runner.cpp`: `NUM_DRAM_BANKS` JIT define changed from `"1"` to `std::to_string(num_dram_channels)` (6 for wormhole, 8 for blackhole, 2 for quasar)

### Remote semaphore operations
- `include/jit_hw/experimental/noc_semaphore.h`: Full rewrite with remote `up()`, `set_multicast()`, `inc_multicast()` using NOC address resolution

### Thread-local state management
- `include/jit_hw/llk_defs.h`: `compute_kernel_hw_startup()` resets `__emule_pack_offset` and `__emule_l1_acc_enabled`

### DFB test fixes
- `test_dfb_emulation.cpp`, `test_dataflow_buffer.cpp`: Buffer `page_size` changed from `buffer_size` to `entry_size` to match kernel TensorAccessor addressing

## Historical Progress

| Date | Passed | Failed | Skip | Total | Key Change |
|------|--------|--------|------|-------|------------|
| 2026-03-11 | 128 | ~310 | -- | ~438 | Initial D2M regression |
| 2026-03-13 | 208 | ~251 | -- | ~459 | Multicast NOC fixes, parallel JIT |
| 2026-03-16a | 249 | ~244 | -- | ~493 | HAL-based semaphore base |
| 2026-03-16b | 614 | 84 | -- | ~698 | DRAM bank offset fix |
| 2026-03-17 | 832 | 162 | 81 | 1075 | Lock-free CB, AVX2 matmul, --forked |
| 2026-03-30 | 1589 | 147 | 142 | 1878 | LLK API stubs, tt-mlir uplift (+803 tests) |
| **2026-04-08** | **1256** | **480** | **142** | **1878** | **Nfaces, NUM_DRAM_BANKS=real, remote semaphore** |

## Next Steps

1. **P0: Fix DRAM bank interleaving for emulation.** The emulated DRAM allocator writes pages sequentially (`base + id * page_size`) but `InterleavedAddrGen` reads with bank interleaving (`bank[id % N] + (id / N) * aligned_size`). Options:
   - (a) Make the emulated DRAM allocation bank-interleaved to match `InterleavedAddrGen`
   - (b) Patch `InterleavedAddrGen` at JIT time to use flat addressing in emulation mode
   - (c) Revert `NUM_DRAM_BANKS` to 1 for D2M (but keep real count for standalone) — quick fix but diverges from hardware behavior

2. **P1: Fix tilize/untilize standalone PCC** — face-ordering issue in the nfaces layout conversion.

3. **P1: Fix DMA tiled crashes** — DRAM NOC address resolution for tiled DMA patterns.
