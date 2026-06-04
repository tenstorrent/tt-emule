# test_permute_sharded fix — running notes

Tracks the investigation of tt-emule#50 on branch `arminale/permute-fix`.

## Status

| Variant (input_sharding, perm) | x_dim | Result |
|---|---|---|
| None,                  [0, 2, 3, 1] | 1 | **PASS** |
| None,                  [0, 3, 2, 1] | 1 | **PASS** |
| None,                  [1, 2, 3, 0] | 0 | **PASS** |
| None,                  [1, 3, 2, 0] | 0 | **PASS** |
| ShardStrategy.HEIGHT,  [1, 2, 3, 0] | 0 | **PASS** |
| ShardStrategy.HEIGHT,  [1, 3, 2, 0] | 0 | **PASS** |
| ShardStrategy.HEIGHT,  [0, 2, 3, 1] | 1 | FAIL (ATOL 6.03, 3.11% non-zero) |
| ShardStrategy.HEIGHT,  [0, 3, 2, 1] | 1 | FAIL (ATOL 6.03, 3.11% non-zero) |

**6/8 PASS** (was 0/8 before this branch). The 2 remaining failures
share `input_sharding=HEIGHT` AND `x_dim=1` (perm.back()=1). The
DRAM-interleaved-input case works for both x_dim values, so the bug
is specific to the sharded-input × x_dim=1 cross.

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
preserved → x_dim = perm.back() = 1, the C dim). The other 6
(`input_sharding=None` for any perm, or `input_sharding=HEIGHT` +
`perm=[1, ...]` = x_dim 0) pass.

Symptom: output **3.11% non-zero** (199878/6422528 cells), max ATOL
6.03. Sampled outputs:

```
out[0, 0,  0, 0] = -1.2344   ref=-1.2344  ✓ correct
out[0, 0,  0, 1] = -0.0315   ref=-1.2578  ✗ non-zero but wrong
out[0, 0,  0, 2] = 1.0156    ref=0.7422   ✗
out[0, 0, 64, *] = 0.0       ref nonzero  ✗ entire w=64 slab empty
out[0,56,  0, 0] = 1.3359    ref=1.3359   ✓
out[0,56,  0, 1] = 0.9414    ref=0.3828   ✗
```

3.11% ≈ 1/32, consistent with "only every 32nd w-position written";
within a written page only c=0 is correct.

Working hypothesis: another compute LLK shim is a stub (likely in the
SHARDED-input variant of the transpose pipeline — for x_dim=1, the
tilize/transpose path differs from x_dim=0). The block-to-output-page
math in the BlockedGeneric writer is identical between the two
x_dim values; the divergence is in how compute processes a partial
X-block of 8 rows (rather than 16 rows for x_dim=0).

Things that aren't the bug:
- chip-model routing (REG probe confirmed all 64 cores registered correctly)
- noc_addr decode (RESOLVE probe verified)
- writer's noc_async_write path
- host tensor_accessor packing (PACK probe verified for the passing case)

Investigation rabbit holes (DO NOT REDO blindly):
- Tried to probe upstream `append_sharded_args` and TensorAccessorArgs
  constructor with fprintf — never fires despite the symbols being in
  `libtt_metal.so`. Cause unknown; possibly Cmake Unity build, ccache,
  or a different code path bypasses these functions. Spent ~1h on this
  and gave up.
- Flipping `corerange_to_cores(..., row_wise=true)` in the BlockedGeneric
  factory did NOT fix anything — work-assignment order is consistent
  with TensorAccessor.

Suggested next probe: instead of host-side, add a kernel-side fprintf
inside the compute kernel (`transpose_xw_rm_single_tile_size.cpp`) to
see whether the compute path differs for x_dim=1 vs x_dim=0. The
compute kernel uses `compute_kernel_lib::tilize<>` from
`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`; that helper has multiple
modes (InitAndUninit, etc.) — one mode may be a stub for x_dim != 0.

## Sentinels (don't regress)

- `dm_test_permute_not_sharded` (currently `-k 'not sharded'` filtered)
- `dm_test_permute`
- Whatever else exercises `pack_untilize_dest`
