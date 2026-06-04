# test_permute_sharded fix — running notes

Closes tt-emule#50 on branch `arminale/permute-fix`.

## Status — 8/8 PASS

```
PASSED  perm=[0, 2, 3, 1]  input=None
PASSED  perm=[0, 3, 2, 1]  input=None
PASSED  perm=[1, 2, 3, 0]  input=None
PASSED  perm=[1, 3, 2, 0]  input=None
PASSED  perm=[0, 2, 3, 1]  input=HEIGHT
PASSED  perm=[0, 3, 2, 1]  input=HEIGHT
PASSED  perm=[1, 2, 3, 0]  input=HEIGHT
PASSED  perm=[1, 3, 2, 0]  input=HEIGHT
```

## Three independent bugs

### 1. `pack_untilize_dest` was an empty stub

`include/jit_hw/api/compute/pack_untilize.h`. With no scatter, `cb_out`
stayed zero, the writer kernel read zeros, the output buffer ended up
all-zero — `ATOL 0.99609375` from `|randn| ≈ 1.0`.

Fixed by emulating the silicon LLK as a single row-major scatter
across (block_rt_dim × block_ct_dim) DST tiles. The cb page_size is
sync granularity only — the L1 layout is always contiguous row-major.
`narrow_row` is the only template that changes the per-row column
count (`row_num_datums` instead of `TILE_DIM`).

This unblocked all the BlockedGeneric paths (`input_sharding=None` or
`perm=[1,…]`).

### 2. `tilize_block` was reading sequential tile blocks, not horizontal strips

`include/jit_hw/api/compute/tilize.h`. The sharded transpose's reader
writes the H×W input into `cb_in` as a horizontal strip of stride
`Wt * TILE_WIDTH * elem_size` per row, but the old `tilize_block`
treated each tile `t` as living at `cb_read_ptr_at(icb, t)` (=
`t * page_size`, sequential 2048-byte blocks). That meant tiles
2..6 were read from unwritten reserved memory and came back zero —
hence transpose_wh of an 8×224 block left 5 of 7 tiles empty.

Fixed by reading the input as a horizontal strip: row stride =
`ntiles * TILE_DIM * elem_size`, tile `t`'s cells are at column
offset `t * TILE_DIM * elem_size`. For `ntiles=1` this collapses
to the old layout, so single-tile callers are unaffected.

This unblocked transpose_wh with partial output H (e.g. H=8 < 32).

### 3. `pack_untilize_dest` Mode B was tile-major, not row-major

After fixes 1 and 2 the failing case was the longest chain
(`perm=[0,3,2,1]` on `input_sharding=HEIGHT`, which expands to
`transpose_wh ∘ transpose_hc ∘ transpose_wh`). Step 1 is a square
224×224 `transpose_wh` that uses `pack_untilize_dest<7, 7>` with
`ps == 2048` (one tile per page).

The original impl had three branches; for `ps >= TILE_ELEMS * elem_size`
it wrote tile-major (each cb page = one full 32×32 tile in row-major
within the page, tiles stacked). But the consumer (downstream sharded
buffer) reads sticks of `W * elem_size` contiguous bytes — it needs
row-major across the WHOLE output region, not tile-major. With
tile-major layout, row 0's cols 32..63 landed at byte offset 2048
instead of 64, so step 1's output was scrambled and the chain
propagated garbage into the final transpose.

Fixed by unifying all modes into a single row-major scatter. The
page_size is purely a sync-granularity signal — the L1 byte layout
is row-major regardless. Now Mode A and Mode B collapse into one
expression; only `narrow_row` (which changes the row width to
`row_num_datums`) remains a branch.

## Diagnosis trail (preserved for the playbook)

1. `pack_untilize_dest` empty-stub → 6/8 fixed.
2. Per-call DST[0][0..7] probe in `pack_untilize_dest`: data
   correct for `narrow_row` path, ruling out the scatter.
3. Per-call DST[0][0..7] probe in `transpose_wh_tile` (pre-transpose,
   post-`copy_tile`): tiles 2..6 of a 7-tile-wide block came back
   all-zero. CB_in was getting only 2 tiles of valid data.
4. Read the reader: it strides the 8 input rows at
   `l1_write_offset_bytes = Wt * elem_size * TILE_WIDTH = 448`
   bytes per row. So `cb_in` is row-major 8×224, NOT tile-major.
5. `tilize_block` was reading tile-by-tile at 2048-byte offsets —
   wrong. Rewrote it to read horizontal-strip with the correct row
   stride. 7/8 fixed.
6. Remaining failure was `perm=[0,3,2,1]`. Traced: step 1
   (square 224×224 `transpose_wh`) takes the LARGE-PAGE branch
   (ps=2048). Realized this branch wrote tile-major, but the
   consumer expects row-major. Unified all three modes into one
   row-major scatter. 8/8.

## Probes (env-gated) — useful, kept around in adjacent commits

- `__emule_resolve_noc_addr` print on `TT_EMULE_TRACE_RESOLVE`
- core-map registration print on `TT_EMULE_TRACE_REG`
- host pack print on `TT_EMULE_TRACE_PACK` (in tt-metal
  `tensor_accessor_args.cpp`)
- first-4-bytes after `memcpy` in `noc_async_write`/`_read` on
  `TT_EMULE_TRACE_WBYTES`/`_RBYTES`
- factory iteration print on `TT_EMULE_TRACE_FACTORY` (tt-metal
  `permute_rm_program_factory.cpp`)

## Rabbit holes (DO NOT REDO blindly)

- Probing `TensorAccessor::get_noc_addr`,
  `TensorAccessorArgs::TensorAccessorArgs`, or `append_sharded_args`
  via fprintf — the symbols exist in `libtt_metal.so` but the probes
  never fire. Suspected Unity-build inlining / different code path.
  Spent ~1h. Skip this path; instrument the consumer side instead.
- Flipping `corerange_to_cores(..., row_wise=true)` in the
  BlockedGeneric factory: didn't change anything. Work-assignment
  order was already consistent with TensorAccessor.
- Workaround via `ttnn::permute` falling through to `prim_permute`
  on emule (via `TT_METAL_USE_EMULE` define): user rejected — emule
  must faithfully run the same ops as silicon.

## Sentinels (verified after fix)

- `test_permute_sharded` — 8/8 PASS (was 0/8)
- `dm_test_permute_not_sharded` — 293/293 PASS
- `dm_test_permute` — 16/16 PASS
- `dm_test_untilize_sharded` — 426/430 PASS (4 pre-existing skips)

## Re-enable the masked variants

Once this PR lands, drop the `-k 'not sharded'` filter from
`dm_test_permute_not_sharded` in
`scripts/run_ttnn_pytests_{wormhole,blackhole}.sh` and add a
`dm_test_permute_sharded` entry covering the 8 perms from #50.
