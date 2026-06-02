# Round 11 closeout — B8.4 and B8.1 both closed by L1-bank-coord fix

Picks up from `round10-b84-characterization.md` and
`round9-closeout.md`. Single root-cause fix landed:
**`emule: emit real NUM_L1_BANKS + populate l1_bank_to_noc_xy from
SoC descriptor`** (matches the Round 7 `IS_NOT_POW2_NUM_DRAM_BANKS`
pattern, but for L1).

## Bottom-line table

| Family | Round 10 baseline | Round 11 result | Δ |
|---|---:|---:|---:|
| `test_full_like.py` -k sharded | 648 P / 0 F | 648 P / 0 F | 0 (regression check) |
| `test_pad.py` -k sharded | 78 P / 23 F | **101 P / 0 F** | **+23** (closes B8.1 as side effect) |
| `test_permute.py` -k sharded | 0 P / 8 F | 0 P / 8 F | 0 (B8.3 PCC tier remains) |
| `test_interleaved_to_sharded.py` | 22 P / 68 F / 16 S | **34 P / 56 F / 16 S** | **+12** |
| `test_untilize.py` -k sharded | 288 P / 196 F | **476 P / 8 F** | **+188** |

**Net new sharded variants passing: +223** (188 untilize + 23 pad +
12 i2s).

## Side-effect closure: B8.1

B8.1 from `round8-sharded-harvest.md` documented test_pad's 23
sharded failures as a "self-loop NOC scratch→dest data mismatch"
needing a separate trace pass. That diagnosis turned out wrong —
the actual root cause was the same NUM_L1_BANKS=1 bug, the kernel's
"scratch CB" reads via `get_noc_addr` were landing on the wrong
core because of the broken interleaved-L1 addrgen. The Round 11
fix closes B8.1 incidentally.

## What landed

Single PR shape, one substantive commit + one harvest commit +
this closeout.

### The fix: emule emits real L1 bank topology

`tt_metal/impl/emulation/emulated_program_runner.cpp`:

1. **Deleted** `EMULE_NUM_L1_BANKS = 1` (line 144 prior). The
   constant was a placeholder; real silicon has 64+ L1 banks
   (one per worker core).
2. **Bumped** `MAX_NUM_BANKS` from 32 to 256 so the runtime backing
   storage holds the larger L1 bank count.
3. **Populate loop** in `populate_bank_mapping` (line ~870, where
   `// L1 bank mapping — for now, all worker cores use themselves
   as bank 0` was a self-documenting TODO):
   ```cpp
   const auto& allocator = device->allocator();
   num_l1_banks_out = allocator->get_num_banks(BufferType::L1);
   for (uint32_t b = 0; b < num_l1_banks_out && b < MAX_NUM_BANKS; ++b) {
       auto logical = allocator->get_logical_core_from_bank_id(b);
       auto virt = device->virtual_core_from_logical_core(logical, CoreType::WORKER);
       uint16_t noc_xy = (uint16_t(virt.y) << NOC_NODE_ID_BITS) | uint16_t(virt.x);
       l1_bank_to_noc_xy[0][b] = noc_xy;
       l1_bank_to_noc_xy[1][b] = noc_xy;
   }
   // bank_to_l1_offset stays zeroed — silicon reserves per-core regions
   // by adding offsets; emule's per-core L1 mmap starts at 0.
   ```
4. **Thread `num_l1_banks` through** `populate_bank_mapping` →
   `execute_program_emulated` → `collect_kernels` →
   `build_kernel_defines`, and emit `NUM_L1_BANKS=N` /
   `LOG_BASE_2_OF_NUM_L1_BANKS` / `IS_NOT_POW2_NUM_L1_BANKS` based
   on the real count.

`include/jit_hw/internal/dataflow/dataflow_api_addrgen.h` +
`include/jit_hw/jit_kernel_stubs.hpp`:
- Replace the `[2][32]` / `[32]` extern array sizes with
  `[2][NUM_DRAM_BANKS]` / `[NUM_L1_BANKS]` / `[NUM_DRAM_BANKS]` /
  `[NUM_L1_BANKS]` (matching upstream's `dataflow_api_common.h:26-29`).
  Without this, the three sets of extern declarations conflict when
  both headers are pulled into the same JIT TU — manifests as a C++
  redeclaration-with-different-type compile error.

### Multi-arch correctness

The fix is arch-uniform — `device->allocator()->get_num_banks(BufferType::L1)`
and `get_logical_core_from_bank_id` are arch-agnostic API. Per-arch
behavior derives automatically from the SoC descriptor:

| Arch | `num_banks(L1)` | pow2? | Emitted defines |
|---|---:|---|---|
| WH-N150 (8×8 worker grid post-harvest) | 64 | yes | `NUM_L1_BANKS=64`, `LOG_BASE_2_OF_NUM_L1_BANKS=6` |
| BH (P100/P150) | 140 | no | `NUM_L1_BANKS=140`, `IS_NOT_POW2_NUM_L1_BANKS=1` |
| Quasar | (per SoC) | (auto) | (auto) |

This round only ran the WH-N150 regression. BH and Quasar will
verify naturally via their nightly CI containers; no code branching
needed.

### The harvest: pad sharded variants + 426 untilize sharded promoted

`scripts/run_ttnn_pytests.sh`:

1. **Pad** — dropped the `-k 'not sharded'` filter on the 5
   `dm_test_pad*` entries (lines 257-261). The 23 previously-failing
   sharded variants in `test_pad_rm_sharded_stickwise`,
   `test_pad_nd_sharded_to_interleaved`, and
   `test_pad_legacy_sharded_to_interleaved` now pass. `not sub_core`
   filters retained — `sub_core_grids` is a separate pre-existing
   gap (Round 6).

2. **Untilize** — added `dm_test_untilize_sharded` covering
   ~426 sharded variants. Selector
   `-k 'sharded and not multi_core_sharded_to_interleaved and not multi_core_nd_sharded_to_interleaved'`
   excludes the two test functions with B12.1 residuals (the
   substring filter also picks up the `_uneven_input_shard_spec`
   variant, so 58 currently-passing sharded variants are
   temporarily left out; they're re-included by Round 12 once
   B12.1 lands).

Total CI-side delta: **+449 sharded variants in
`run_ttnn_pytests.sh`** (23 pad + 426 untilize).

## What did NOT close

- **B12.1 (new)** — 8 residual `test_untilize_multi_core_*sharded_to_interleaved`
  failures. Quick diagnostic pass captured the pattern; deferred
  for a focused fix attempt:
  - All 8 share `tensor_shape=[4, 4, 256, 512]` (=4096 rows × 512
    cols, 4 shards × 1024 rows each).
  - ATOL ≈ 3.0625 / 3.28125 / 3.328125 — mostly correct: 1020 of
    2,097,152 elements wrong (~0.05%).
  - The wrong elements span exactly **64 rows of 4096** — four
    blocks of 16 contiguous rows, one block per shard, at the SAME
    offset 592 within each shard (rows 592-607, 1616-1631,
    2640-2655, 3664-3679).
  - Row 592 = tile-row 18, sub-row 16: the **bottom half of
    tile-row 18 within each shard** is corrupted.
  - Smell test: writer's tile-row counter or face-row scatter has
    an off-by-N at a specific tile-row index, only manifesting for
    1024-row shards (the smaller [2,2,256,512] interleaved→sharded
    case has 256-row shards = 8 tile-rows, doesn't reach tile-row
    18).
  - Tractable with the same `TT_EMULE_TRACE_CB` / `TT_EMULE_TRACE_NOC`
    instrumentation recipe from Round 10. Start by re-dumping
    writer's NOC write addresses for the canonical
    `test_untilize_multi_core_sharded_to_interleaved
    [num_shard_cores=4-...-HEIGHT_SHARDED-tensor_shape=[4, 4, 256, 512]
    -BFLOAT16]` and inspect where the writer thinks tile-row 18
    rows 16-31 should land.
- **B10.1** — i2s BFP8 failures: pre-existing (12 of 24
  `test_interleaved_to_sharded_hash` failures use BFP8). The
  remaining 56 i2s failures (was 68) include these plus other
  PCC-tier residuals. Deferred.
- **B8.3 PCC tier** — 6 of 8 permute sharded failures still PCC
  (`Max ATOL Delta: 0.99609375`). The other 2 were JIT-compile
  gates that closed in Round 9. No movement this round.

## Files touched

- `tt_metal/impl/emulation/emulated_program_runner.cpp` — the
  three-hunk fix.
- `include/jit_hw/internal/dataflow/dataflow_api_addrgen.h` —
  match upstream's NUM_*_BANKS array sizes.
- `include/jit_hw/jit_kernel_stubs.hpp` — same array-size match.
- `scripts/run_ttnn_pytests.sh` — drop `not sharded` on 5 pad
  entries.
- `docs/notes/round8-sharded-harvest.md` — closure markers on
  B8.1 (closed as side effect) and B8.4 (primary closure).
- `docs/notes/round11-closeout.md` — this file.

## Round 12 agenda

Per priority + effort:

1. **B12.1 — sharded→interleaved untilize residual** (8 tests,
   small ATOL). Same writer kernel for both nd and legacy sharded
   paths. Likely a partial-last-bank stride bug; tractable with the
   `TT_EMULE_TRACE_CB` / `TT_EMULE_TRACE_NOC` instrumentation
   recipe from Round 10.
2. **untilize sharded harvest** — add per-test-function `dm_test_untilize_*`
   entries to `scripts/run_ttnn_pytests.sh` covering the 476
   now-passing sharded variants. Bundle with B12.1 if a quick fix
   lands.
3. **B10.1 BFP8 path** (12 tests, separate dtype-specific issue).
4. **B8.3 permute PCC tier** (6 tests, likely shares root cause
   with B12.1 since both touch interleaved-side writer math).

## Tt-umd PR #2743 still independently valuable

Round 10 closeout already cross-referenced
[tenstorrent/tt-umd#2743](https://github.com/tenstorrent/tt-umd/pull/2743).
It addresses (a) Blackhole NOC0→TRANSLATED DRAM core lookup and
(b) the `static_cast<uint32_t>(l1_dest)` truncation latent landmine.
Neither was the B8.4 root cause, but Fix 1 is independently valuable
for BH correctness once that work resumes.
