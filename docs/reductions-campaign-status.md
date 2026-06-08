<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# `arminale/reductions` — campaign status & resume guide

Self-contained handoff for the pool/reduce bring-up campaign on this branch.
Covers what's done, what's verified, and the root-caused map of what remains.
Detailed iteration-by-iteration log lives at `docs/notes/reductions-campaign.md`
(gitignored, local-only — this file is the durable, committed summary).

## Goal

Bring up the LLKs in issue **#80** and make the pytests in **#62** (pool) and
**#66** (reduce-tail) pass under emule, faithful to silicon. Single-device only
(WH N150 / BH P100). NoC-api work is merged into this branch and is **final** —
solvable gaps are fixed here, not deferred to another PR.

## Branch state

- Base: rebased on `main` + `arminale/noc-api` merged in.
- Pushed to `origin/arminale/reductions` (HEAD `dbbed0f` as of 2026-06-07).
- Build: clang-20, header-only JIT shims under `include/jit_hw/`; slow dispatch.

## DONE — landed & verified

| Commit | Change | Verified |
|---|---|---|
| (earlier) | #80 pool LLK shims (`tilizeA_B_reduce_init`, `unpack_tilizeA_B_block/uninit`, `max_reduce_with_indices`, `transpose_wh_dest`), reduce.h refactor | test_max 228/228 |
| (earlier) | Bucket 1: experimental-NoC surface aligned to global `api/dataflow` (`NocOptions`, templated `async_read_barrier`), per-RISC `thread_local` NoC read/write state, `cb_is_*_format` → pure dtype checks | pool compiles e2e |
| `3b17d84` | Surface `VALID`/`INVALID` to kernels (dataflow_api include) + `CircularBuffer::read_tile_value` | topk compiles |
| `623209f` | **BUG#2**: `pack_untilize_dest` writes `rows_per_tile = (num_faces>2?2:1)·face_r_dim` rows, not always 32 (was overflowing 31 stale rows into adjacent output sticks) | **avgpool 240/0** |
| `5a3b3a3` | Wire `pool_test_avgpool2d` into both runners | — |
| `c424e88` | `ckernel_defs.h` tile/face dimension constants (`TILE_HEIGHT` etc.) — fixes `padded_slice` writer compile error | 0 compile errors |
| `dbbed0f` | STRUCTURE.md sync | — |

### Wired & green (both runners, re-confirmed 2026-06-07, 609 passed / 0 failed)
- `pool_test_avgpool2d` → `test_avgpool2d.py::test_avg_pool2d_post_commit` — **240/0**
- `pool_test_global_avg_pool2d` (full) — 10/0
- `pool_test_upsample_nearest` `-k 'not test_bilinear_multi_core'` — 298/0
- `pool_test_rotate_nearest` `-k 'not bilinear'` — 61/0
- #66 reduce-tail (earlier): `test_row_major_reduce`, `test_reduction_program_cache`,
  `test_intimg`, `test_reduction_h_interleaved::test_2D_tensor`, `test_topk::*_dtype_raise`

### Current pool baseline (committed HEAD, no regression)
maxpool **82/22**, adaptive **65/11**, 0 compile errors.

## REMAINING — root-caused, deferred (all deep emule-internals, not shims)

Tractable shim-level wins are banked. Each item below is its own multi-iteration
investigation. Highest leverage first.

### 1. argmax + mpwi — multicore cross-program state cleanup  (task #32)
- `test_argmax.py`: each case passes **in isolation** (single-core AND multicore),
  but a multi-case sweep **aborts (core dumped) at the ~5th case** (first heavy
  multicore reduction); `--forked` also dies after 4. → NOT a per-case bug.
- Root: emule does not reset multicore semaphore / NoC / L1 **program state across
  executions**. The multicore argmax reader
  (`reader_argmax_interleaved_multicore.cpp`) uses cross-core `start_sem`/`done_sem`
  multicast (`set_multicast<MCAST_INCL_SRC>`, `done_sem.up`/`wait(num_cores)`) +
  remote `async_write` between worker cores and a reduce core. State accumulates →
  abort, likely in program/semaphore teardown (`src/`).
- Same family as the **mpwi flaky hang** (`test_mpwi_small_kernel_sizes`). Fixing
  emule's cross-program multicore cleanup likely unblocks **both**. ← best ROI.

### 2. Pool tail — bf8-TILE-in-DRAM  (task #35)
- ~25 of 33 pool-tail fails (maxpool dram_slice 15 + adaptive dram bf8 10).
- Discriminator: bf8-TILE **in DRAM** fails; bf8-TILE **in L1** (height_shard)
  passes; bf16-TILE in DRAM passes. Empirical: only output stick 0 correct, rest
  `-inf`/zero (maxpool windows with no valid input).
- Ruled OUT (verified working): real `TensorAccessor` addressing (NOT mocked —
  emule only resolves NOC→host ptr via `__emule_resolve_noc_addr`), `copy_tile` bf8
  decode (central `__emule_unpack_cb_tile_to`), `__llk_pack_untilize` scatter, page
  alignment (bf8 1088 B is 32-aligned).
- Suspect: emulated DRAM **block-float page layout on the upload/write side**
  (`src/` — `host_api.cpp` / `Buffer`), where bf8 (1088 B) and bf16 (2048 B) diverge.
- Next step: minimal repro = read a bf8 TILE DRAM tensor back through an
  interleaved-DRAM op (no pool), then inspect `src/` buffer-write.
- Chain for reference: `padded_slice` tiled = reader
  (`padded_slice_reader_tiled_interleaved_start_id.cpp`) → halo untilize
  (`sliding_window/halo/.../compute/pack_untilize.cpp` →
  `compute_kernel_lib::untilize` → emule `pack_untilize_block<8>`) → writer
  (`writer_unary_sharded_padded_tiled.cpp`).

### 3. Bucket 4 — reduce numerics  (task #31)
- `reduction_on_batch` 18/68: ND batch-sharded sum over N/C **fundamentally wrong**
  (maxdiff ~730, not precision; first elem ~right, rest off). Uses
  `fill_implicit_tile_padding(-42)` → reduce must exclude padding. Suspect: batch-
  shard gather/accumulation (wrong tiles summed, padding included, or multi-core
  shard addressing).
- `fast_reduce_nc` 17/46, `reduction_h_interleaved` (3D / full-grid) 27/15 — same
  genuine-numeric class.
- `topk` bitonic numeric (`topk_xl.h` + `__emule_topk`): merge direction / fused
  bf16|u16 index encoding — hardest, separate.

### 4. bf16 pool wide+large-kernel+ceil exact-match (3, part of task #35)
- `block_shard [*,2048/4096,16,16,5×5,s2,p2,ceil]` (bf16 & bf8) + 1 adaptive bf16.
  Dtype-independent (bf16 must match exactly) → windowing/halo for that config.

### Skipped by direction
- Bucket 3 **ema** (#53 stateful SFPU) — skipped per user.

## How to run / repro

Env (WH N150; for BH use `blackhole_P100.yaml` + `MESH_DEVICE=P100`, and wipe
`/tmp/tt_emule_jit_cache_$(id -u)` when switching arch):

```bash
TT_METAL_DIR=/localdev/arminale/tt-metal
BUILD_DIR=$TT_METAL_DIR/build_emule
export PYTHONPATH="$TT_METAL_DIR/ttnn:$TT_METAL_DIR/tools:$BUILD_DIR/lib:$TT_METAL_DIR"
export LD_LIBRARY_PATH="$BUILD_DIR/lib"
export TT_METAL_HOME="$TT_METAL_DIR" TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$TT_METAL_DIR/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 MESH_DEVICE=N150
```

- Full curated regression: `scripts/run_ttnn_pytests_{wormhole,blackhole}.sh`
  (run one at a time — shared JIT cache; `SHARD_INDEX`/`SHARD_COUNT` for sharding).
- Tests under `$TT_METAL_DIR/tests/ttnn/unit_tests/operations/{pool,reduce}/`.
- Debug: `PYTHONPATH=/tmp/emule_dbg:$PYTHONPATH` wraps `torch.allclose` to print
  maxdiff / bad sticks; `TT_EMULE_KEEP_JIT_SRC=1` keeps generated kernel sources.
- `--forked` isolates each test item (needed to localize the argmax crash).

## Process notes
- Use `/arch-lookup` before guessing silicon behavior (pinned BUG#2 in one pass).
- `pack_untilize_dest` row count, `cb_is_*_format` pure-dtype, and per-RISC NoC
  `thread_local` state were the three Bucket-1/BUG#2 keys.
- A compile error can MASK numeric failures (TILE_HEIGHT fix surfaced the bf8-DRAM
  root) — re-triage after every compile fix.
