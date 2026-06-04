# test_permute_sharded fix — running notes

Tracks the investigation of tt-emule#50 on branch `arminale/permute-fix`.

## Status

| Variant (input_sharding, perm) | Result |
|---|---|
| None,                  [0, 2, 3, 1] | **PASS** |
| None,                  [0, 3, 2, 1] | **PASS** |
| None,                  [1, 2, 3, 0] | **PASS** |
| None,                  [1, 3, 2, 0] | **PASS** |
| ShardStrategy.HEIGHT,  [1, 2, 3, 0] | **PASS** |
| ShardStrategy.HEIGHT,  [1, 3, 2, 0] | **PASS** |
| ShardStrategy.HEIGHT,  [0, 2, 3, 1] | FAIL (ATOL 0.996) |
| ShardStrategy.HEIGHT,  [0, 3, 2, 1] | FAIL (ATOL 0.996) |

**6/8 PASS** (was 0/8 before this branch).

## Fix #1 — `pack_untilize_dest` was a no-op stub

`include/jit_hw/api/compute/pack_untilize.h` — `pack_untilize_dest` had an
empty body `{}`. The BlockedGeneric permute kernel uses
`tilize → transpose_wh_tile → pack_untilize_dest` to drive its
transpose; the last step is what scatters the DST register back to the
output CB. With it as a no-op, `cb_out` stayed zero, the writer kernel
read zeros, and the output buffer landed at every core as all-zero
data — hence the `output_uniform_zero=True` symptom and the
`ATOL 0.99609375` failure (`|randn| ≈ 1.0`).

Implementation: direct scatter of `(block_rt_dim × block_ct_dim)`
DST tiles to `cb_out` row-major. Does NOT delegate to
`__llk_pack_untilize` because that helper assumes the CB page size is a
full tile (`ps == 32*32*element_size`); BlockedGeneric uses a smaller
page size (one row = 32 bf16 = 64 bytes).

## Diagnosis trail

The trail that found this — for the playbook:

1. **Symptom**: 8/8 fail, `Max ATOL Delta: 0.99609375`, output uniformly zero.
2. **R-probe** (host registration of `__emule_core_map`, runner side):
   logical-to-phys mapping is correct, 64 worker cores registered. So
   chip model is fine.
3. **Resolve-probe** (`__emule_resolve_noc_addr`): noc_addr decode for
   the 64 workers' writes shows `dst = (src.y, src.x)` — looked like
   an x/y transpose at first.
4. **PACK-probe** (`tensor_accessor_args.cpp` host pack site): bank
   coords packed in ROW_MAJOR order, format `(virt.x << 8) | virt.y` —
   correct.
5. **Factory experiment** — flipped `corerange_to_cores(..., true)` so
   the work split matched `BufferDistributionSpec::cores()` ROW_MAJOR
   order. Still failed. So work assignment wasn't the issue.
6. **WBYTES-probe** (first 4 bytes of `dst` after the `memcpy` inside
   `noc_async_write`): all writes were `0x00000000`. So the writer
   kernel's **CB source** was zero. The earlier "transpose" symptom
   was an illusion — every kernel wrote ZEROS to a single dst, so the
   apparent (Sy, Sx) pattern was just "each kernel writes its own
   shard of zeros."
7. **RBYTES-probe** (inside `noc_async_read`): reader kernels read
   real non-zero DRAM data. So input → compute pipeline was the gap.
8. Compute pipeline = `tilize → transpose_wh_tile → pack_untilize_dest`.
   `transpose_wh_tile` does the real transpose (in
   `transpose_wh.h`). `pack_untilize_dest` was the empty stub.

## Probes (now cleaned up — kept here for reuse)

- `__emule_resolve_noc_addr` print on `TT_EMULE_TRACE_RESOLVE`
- core-map registration print on `TT_EMULE_TRACE_REG`
- host pack print on `TT_EMULE_TRACE_PACK` (in tt-metal
  `tensor_accessor_args.cpp`)
- first-4-bytes after `memcpy` in `noc_async_write`/`_read` on
  `TT_EMULE_TRACE_WBYTES`/`_RBYTES`
- factory iteration print on `TT_EMULE_TRACE_FACTORY` (tt-metal
  `permute_rm_program_factory.cpp`)

## Open — the 2 remaining failures

Both: `input_sharding=ShardStrategy.HEIGHT` + `perm=[0, ...]` (batch
preserved). The other 6 (`input_sharding=None` for any perm, or
`input_sharding=HEIGHT` + `perm=[1, ...]`) pass.

Hypothesis to test next: the reader kernel for sharded input may have
a related stub gap, OR the block-to-(input-page) mapping in the
reader differs for `perm[0] == 0` and emule's compute pipeline
misses some op needed only on that path.

## Sentinels (don't regress)

- `dm_test_permute_not_sharded` (currently `-k 'not sharded'` filtered)
- `dm_test_permute`
- Whatever else exercises `pack_untilize_dest`
