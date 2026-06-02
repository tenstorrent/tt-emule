# Round 13 closeout — DRAM-sharded i2s segfaults closed; permute deferred

Picks up from `round11-closeout.md` (with R12's MEM_ZEROS_BASE fix
already landed at HEAD).  R13 targets the three gaps the R11
closeout left open plus one new symptom that surfaced during R12
verification:

| Gap | Tests | R12 baseline | R13 result |
|---|---|---:|---:|
| **G1** — DRAM-sharded i2s (new) | `test_interleaved_to_dram_*` | crashes mid-kernel | **+52 passing**, 4 BFP8 PCC residuals |
| **G2** — permute sharded PCC | `test_permute_sharded` -k sharded | 0 P / 8 F | **deferred** (root cause characterized) |
| **G3** — BFP8 i2s (B10.1) | `test_interleaved_to_sharded_hash` | unknown | confirmed already closed by R11+R12 |

## What landed

### G1 — uint32-truncation in shard_addr_gen runtime-arg pointers

Test_interleaved_to_dram_{height,width,sharded_convert_dtype} all
segfaulted shortly after `execute_program_emulated: 4 logical
cores`.  GDB post-mortem on the canonical repro pointed at
`mov (%rdx,%rax,4),%eax` with `rdx=0xc0006700` — a 4-byte truncated
pointer being dereferenced.

The kernel:
```cpp
const auto [mapping_table, rt_increment] =
    experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(8));
```

calls into:
```cpp
std::pair<...> get_shard_map(uint32_t L1_address) {
    const mapping_table_t* const map =
        reinterpret_cast<const mapping_table_t* const>(L1_address);
    ...
}
```

at `ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp:136`.

`L1_address` is `uint32_t`.  On silicon that's fine — real L1
addresses fit in 32 bits.  emule's `get_arg_addr` returns
`uintptr_t` pointing into `&__rt_args[...]`, which is heap-backed
above 4 GB, so the implicit `uintptr_t → uint32_t` narrowing drops
the upper 32 bits.  The kernel then dereferences a bogus address.

**Fix shape**: back rt-arg storage with `mmap(MAP_32BIT)` so the
addresses returned by `get_arg_addr` / `get_common_arg_addr` are
already 32-bit-safe.  Implementation:

`tt_metal/impl/emulation/emulated_program_runner.cpp`:
- Add `thread_local uint32_t* __rt_args_scratch` and
  `__common_rt_args_scratch`, lazily allocated via
  `mmap(PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_32BIT)`
  with capacity `EMULE_RT_ARGS_CAP = 341` (matches the upstream
  rt-args cap).
- At kernel launch, `memcpy` `ki.rt_args.data()` /
  `ki.common_rt_args.data()` into the scratch.  Guard with a
  capacity check that throws on overflow.

`include/jit_hw/jit_kernel_stubs.hpp`:
- Declare the two scratch externs (`uint32_t*`).
- Update `get_arg_addr` and `get_common_arg_addr` to return
  pointers into the scratch buffers instead of the std::vector
  storage.  `get_arg_val<T>` continues to read from the heap-side
  vectors (it never returns a pointer, so the narrowing path
  doesn't apply).

The original `__rt_args` / `__common_rt_args` vectors stay in
place for `get_arg_val<T>` — only the *pointer*-returning APIs
needed the below-4 GB path.

**Reach**: full `test_interleaved_to_sharded.py -k dram` runs to
completion, 52 passing, 4 BFP8 numerical residuals (separate
issue, not crashes).  No regressions on
`test_untilize.py -k sharded` (484 / 0 unchanged).

### G3.B — BFP8 branch added to `__llk_pack_tiled`

Latent hygiene fix in `include/jit_hw/llk_pack.h:11-29`: the
tilize-pack entrypoint had only `bf16`/`32-bit` branches; missing
a `cb_is_bfp8_b_format(ocb)` case meant any BFP8 tilize would
silently misencode (treat 1088-byte BFP8 layout as 2048-byte bf16
tile).

Mirrors the BFP8 path already correct in `pack_dst_to_buf`
(`include/jit_hw/api/compute/common.h:219-242`): per face-row,
collect 16 fp32 elements from DST via `nfaces_to_rowmajor`, call
`__emule_bfp8::encode_face_row`, write 1 exponent + 16 mantissa
bytes.

**Reach**: no currently-passing test exercises this (i2s_hash uses
pure data movement, not tilize).  Lands as latent-bug protection
for future BFP8 tilize paths.

### G3.A — B10.1 confirmed closed

`test_interleaved_to_sharded_hash` was reported as 12-of-24 BFP8
failures at R11.  Today it's 24 / 24 after the R11 NUM_L1_BANKS
+ R12 MEM_ZEROS_BASE stack.  i2s_hash uses `eltwise_copy` which
goes through `pack_dst_to_buf` (BFP8-aware), not
`__llk_pack_tiled`, so the G3.B latent bug was never reachable
here.  B10.1 is closed.

## What did NOT close

### G2 — permute sharded (B8.3 PCC tier)

Standalone repro on `shape=[16, 8, 224, 224]`, `perm=[0, 2, 3, 1]`,
output HEIGHT-sharded across an 8×8 grid: **output is uniformly
zero** (6.4 M cells wrong, all exact 0).  The writer kernel issues
802,816 NOC writes from all 64 source cores, but the destination
pattern is degenerate:

| Source phys | Destination phys |
|---|---|
| (18, 18) → logical (0, 0) | (18, 18) — self |
| (18, 19) → logical (0, 1) | (19, 18) — transposed |
| (18, 20) → logical (0, 2) | (20, 18) — transposed |
| (18, K) for K = 18..25     | (K, 18) — only row y=18 |

Each source writes to exactly one destination, and the
source→destination map is a coordinate transposition.  The output
is HEIGHT-sharded with `ShardOrientation` row-major (default), so
shard `K` should be at logical `(K % 8, K / 8)` on the destination
side.  The writer's `bank_id` derivation appears to walk the shard
grid with one stride ordering and the host's `packed_xy_coords`
populates with another — neither side is straightforwardly wrong
in isolation; the contract between them disagrees.

The two ends of the contract are:
- `tt_metal/impl/buffers/tensor_accessor_args.cpp:74-86` —
  host packs `bank_coords[i] = (coord.x << 8) | coord.y` in the
  order returned by `buffer_distribution_spec.cores()`.
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h:240-260` —
  kernel uses `packed_xy_coords[bank_id]` with `bank_id` derived
  from `flattened_shard_id` per `dspec.shard_grid_strides`.

Deferred because pinning down which side has the stride ordering
wrong requires a deeper dive into tt-metal's sharded
DistributionSpec semantics than fits the round budget.  The
diagnostic recipe (W-trace + Python post-pass) is captured in the
`memory-debug` skill's new "Dataflow-only debug" section so the
investigation is resumable.

### G1 BFP8 PCC residuals

4 of 56 `test_interleaved_to_dram_*` cases still fail after G1
with `Max ATOL Delta: 0.0311` (down from segfault).  All 4 are
`convert_dtype` with `FLOAT32 → BFLOAT8_B` output — a numerical
precision issue in the BFP8 encoder, not a crash.  The encoder
(`include/jit_hw/api/compute/bfp8.h::encode_face_row`) truncates
the significand without rounding; if silicon HW rounds-to-nearest
the ~0.03 ATOL is consistent.  Out of scope for R13 (was already
the R12-closed status — see `round11-closeout.md` §B10.1).

## Skill maintenance

`.claude/skills/memory-debug/SKILL.md` gained two new sections
in-place (no v1/v2 markers):

1. **Crashes** — gdb post-mortem recipe (handle SIGSEGV, dump
   registers + disasm), how to identify the JIT kernel from the
   `.so.tmp.*` path, the silicon-side `uint32_t` truncation
   pattern to watch for.
2. **Dataflow-only debug** — adapted trace pass for ops that have
   no compute kernel (permute, sharded transfers).  Python
   post-pass that decodes `dst_noc_addr` and surfaces the
   src↔dst mapping pattern (1-to-1, fan-out, transposition).

## Files touched

- `tt_metal/impl/emulation/emulated_program_runner.cpp` — rt-args
  scratch buffers (G1).
- `include/jit_hw/jit_kernel_stubs.hpp` — scratch externs,
  `get_arg_addr` / `get_common_arg_addr` use scratch (G1).
- `include/jit_hw/llk_pack.h` — BFP8 branch in `__llk_pack_tiled`
  (G3.B).
- `.claude/skills/memory-debug/SKILL.md` — Crashes + dataflow-only
  debug sections (methodology).
- `docs/notes/round13-closeout.md` — this file.

## Round 14 agenda (priority)

1. **G2 — permute sharded** (8 tests).  Pick up where R13 left off:
   the W-trace already captured the transposition.  Need to
   identify whether `buffer_distribution_spec.cores()` returns
   row-major or column-major and whether the kernel's
   `flattened_shard_id` math matches.  Likely a 1-2 line fix once
   located, in either `tensor_accessor_args.cpp` or in an emule
   sharded-spec shim.
2. **BFP8 convert_dtype PCC** (4 tests).  Encoder rounding mode.
   `encode_face_row` truncates the significand; silicon likely
   rounds-to-nearest-even.  Adding round-bit handling should
   close all 4 — small fix, low priority.
3. **sub_core_grids** (Round 6 carry-over, pre-existing).
