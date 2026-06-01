# Round 10 — B8.4 characterization (no fix landed)

Picks up from `round9-closeout.md`. Attacked B8.4 (the 196-test
`test_untilize` interleaved↔sharded PCC cluster) per the plan in
`/home/arminale/.claude/plans/vivid-foraging-nebula.md`. **Phase A
characterization completed; Phase B-D deferred** — the bug is
deeper than the plan's Phase B table anticipated.

## What Phase A established

Canonical small test:
```
test_untilize.py::test_untilize_single_core_interleaved_to_sharded[
  num_shard_cores=4-...
  output_shard_orientation=ShardOrientation.ROW_MAJOR-
  output_memory_layout=TensorMemoryLayout.HEIGHT_SHARDED-
  tensor_shape=[2, 2, 256, 512]-dtype=DataType.BFLOAT16]
```

Standalone repro at `/tmp/r10_b84_repro.py`. Dump at
`/tmp/round10/b8_4_dump.npz`.

### Findings (instrumented `__emule_resolve_noc_addr` + `noc_async_write` + `SWEmuleChip::read_from_device`)

1. **Output is UNIFORMLY ZERO**: 524288 of 524288 BFLOAT16 elements
   are exactly 0.0. `diff == |input|` for every element. The
   3.25 / 3.328125 ATOL is simply the max(|input|) since output is
   everywhere zero.

2. **`from_torch` → interleaved L1 → `to_torch` round-trips correctly**
   (verified by `/tmp/r10_b84_split.py`). The input data IS in L1
   intact. Bug is downstream.

3. **Writer fires 1024 NOC writes to the correct sharded output cores**:
   - 256 writes to physical core (18,18) — logical (0,0) per
     `EMULE_WORKER_COL_MAP`
   - 256 each to (18,19), (18,20), (18,21) — logical (0,1..3)
   - Each write: size=1024 bytes, dst_noc resolves to a valid host ptr
   - L1 offsets stride by 0x400 (1024 bytes per row) — correct for
     HEIGHT_SHARDED row-major output

4. **No `EMULE WARN: noc_async_write failed to resolve addr` lines.**
   All NOC addresses resolve successfully.

5. **Writer's src data is ALL ZEROS** (the key finding):
   ```
   [WRITE] src_l1=0x41421560 → dst_noc=0x49200012a000 size=1024
           src_ptr=0x41421560 first8=00000000 00000000
   ```
   All 1024 writes show `first8=00000000 00000000`. The writer reads
   from its own L1 (the output CB's read pointer) and finds zeros.

6. **Writer's NOC resolve and `to_torch`'s `read_from_device` hit the
   SAME `tt_emule::Core` object AND the SAME host pointer** (instrumented
   `core_obj` and `ptr` fields):
   ```
   WRITE: core=(18,18) l1_off=0x12a000 core_obj=0x75bc360 ptr=0x4212a000
   READ:  core=(18,18) l1_src=0x12a000 core_obj=0x75bc360 ptr=0x4212a000  first8=00000000
   ```
   Identical pointers, no caching divergence. Writer writes zero, reader
   reads zero.

7. **`__emule_cbs` IS shared between compute and writer threads**
   (per `emulated_program_runner.cpp:1751`: every thread assigns the
   same `cb_array`). CB state and counter synchronization works —
   `cb_wait_front` succeeded (no hang).

### Conclusion of Phase A

The bug is **upstream of the writer**, on the COMPUTE-kernel → OUTPUT-CB
data path:

- Reader fetches 512 tiles from interleaved L1 (confirmed via 512
  NOC reads to core (0,0) in the trace).
- Compute (`pack_untilize_block` → emule's `__llk_pack_untilize`) is
  supposed to write 16 tiles per block to output CB at sequential
  row offsets, producing 32 KB of BFLOAT16 row-major data per block.
- **Output CB ends up empty** — when writer drains it via
  `noc_async_write(get_read_ptr(cb_out0), ...)`, every byte is zero.

The bug must be one of:
- **(C1)** emule's `__llk_pack_untilize` (in
  `include/jit_hw/llk_pack.h`) writes to a different L1 region than
  `get_read_ptr(cb_out0)` returns. Possible if `__emule_compute::
  cb_write_ptr_at(ocb, 0)` and `cb_sync_read_ptr(__emule_cbs[ocb])`
  diverge for the OUTPUT_CB index 16 in this specific configuration.
- **(C2)** emule's `copy_tile(icb, src_tile, dst_idx)` (called from
  `pack_untilize_block` before `__llk_pack_untilize`) doesn't
  actually copy the input tile into DST. If DST is zeros, the
  subsequent pack_untilize writes zeros.
- **(C3)** The compute thread runs but its writes get overwritten
  before the writer drains the CB. E.g., the `cb_push_back` counter
  increments without persisting the data, or there's a copy-on-write
  page-mapping issue with the L1Pool.

Sharded↔sharded test_untilize passes (288 tests pass per Round 8
data) — that path uses the SAME compute kernel + LLK chain but a
DIFFERENT program factory that places the output buffer on the same
core. So **(C1) is the most likely** — the OUTPUT CB's read pointer
in the writer kernel and the OUTPUT CB's write pointer in the compute
kernel may resolve to different L1 regions when the writer is on a
different core than the buffer destination.

But this is speculation past the Phase A budget. **Defer to Round 11
with a focused starting hypothesis (C1).**

## What did NOT get done (deferred to Round 11)

- Phase B: pattern→file mapping (the pattern doesn't cleanly match
  any of the plan's Phase B table rows — output-is-zero pattern was
  not anticipated).
- Phase C: fix attempt.
- Phase D: regression sweep + commits.

Round 9 baseline preserved (no regressions introduced — all emule
diagnostic edits reverted).

## Round 11 starting point — focused next steps

**Update after a second instrumentation pass (TT_EMULE_TRACE_CB
in copy_tile + get_write_ptr + noc_async_read; TT_EMULE_TRACE_NOC
in write_to_device):** the bug is NOT (C1), (C2), or (C3) as
originally hypothesized. The real cause is a **bank-count mismatch
between host and kernel for INTERLEAVED L1 buffers** — the same
family of bug as Round 7's `IS_NOT_POW2_NUM_DRAM_BANKS`, but for L1.

### Evidence

Within ONE run (same mmap layout):
- `from_torch` (via `SWEmuleChip::write_to_device`) writes the
  input tensor across **many worker cores** — (18,18), (19,18),
  (20,18), (21,18), (22,18), (23,18), (24,18), (25,18), ... Each
  gets ~9 writes. Distribution matches the host buffer allocator's
  bank list.
- Kernel reader's `s.get_noc_addr(page_id)` for the INTERLEAVED L1
  input returns NOC addresses pointing **only to core (0,0)**.
  Reader fires 512 NOC reads, all to (0,0) at offsets
  `0x16a000`..`0x16e800`.
- `__emule_resolve_noc_addr((0,0), 0x16a000)` returns `src_ptr =
  0x7f618a1d0000` — a 64-bit address in the DRAM-style mmap range.
  That memory is empty.
- Reader's memcpy writes zeros into the input CB; compute reads
  zeros; output is zeros.

The wrapper.cpp JIT defines confirm:
```
#define NUM_DRAM_BANKS 12
#define NUM_L1_BANKS 1     ← here's the bug
```

The kernel-side `interleaved_addr_gen::get_bank_index<L1>(id)`
computes `id % NUM_L1_BANKS = 0` for every page, so all NOC reads
target a single bank → a single core → a non-existent input
location.

The host (`SWEmuleChip::write_to_device`) takes the buffer's actual
`cores()` list from the allocator (many worker cores) and writes
correctly. So the round-trip `from_torch → to_torch` works
(both sides use the allocator's bank list). But the kernel-side
addrgen uses the JIT-emitted `NUM_L1_BANKS=1`, so it disagrees.

### Why sharded↔sharded works

Sharded `TensorAccessor` doesn't use `interleaved_addr_gen` —
it uses the shard spec's `bank_coords` directly. So sharded paths
bypass the broken L1-interleaved addrgen.

### Why test_full_like / test_pad / test_permute don't hit this

Those tests' inputs are sharded or DRAM. Only **INTERLEAVED L1
input + sharded L1 output** combinations route through the broken
interleaved-L1 reader path. test_untilize's
`test_untilize_single_core_interleaved_to_sharded` is the canonical
failing shape; the other 195 untilize failures all share the same
input-side memory_config.

### Fix

Same shape as Round 7's `IS_NOT_POW2_NUM_DRAM_BANKS`:

1. Emit `NUM_L1_BANKS = N` matching the host's real L1 bank count
   in `build_kernel_defines`
   (`tt_metal/impl/emulation/emulated_program_runner.cpp:961`).
2. Emit the matching `bank_to_l1_offset` and `l1_bank_to_noc_xy`
   arrays so the kernel-side addrgen targets the same cores the
   host allocator distributed pages to.
3. Verify against the SOC descriptor's L1 bank list for WH-N150.

Expected impact: **all 196 test_untilize sharded failures clear**,
since they all share the broken interleaved-L1 reader path.

### Cross-reference: tt-umd PR #2743 (does NOT fix this)

[tenstorrent/tt-umd#2743](https://github.com/tenstorrent/tt-umd/pull/2743)
fixes two related-but-different issues:
- **Fix 1**: `is_dram_core()` misses on Blackhole because
  `soc.get_dram_cores()` returns NOC0 coords but the chip driver
  receives TRANSLATED. Wormhole-unaffected (NOC0 == TRANSLATED
  for WH DRAM).
- **Fix 2**: Drop `static_cast<uint32_t>(l1_dest)` truncation in
  `write_to_device`/`read_from_device`. Latent landmine; per the
  PR, "correctness-preserving at today's sizes."

B8.4 is on Wormhole, and the truncation isn't biting. PR #2743 is
independently valuable for Blackhole DRAM correctness, but does
NOT move B8.4 numbers on WH-N150. Round 11's fix is the
host/kernel L1-bank-count agreement described above.

### Diagnostic recipe (for future rounds)

The trace pattern used this round, recreatable in ~5 min of edits:

1. `emulated_program_runner.cpp:__emule_resolve_noc_addr` — env-var
   gated fprintf showing noc_addr → core_obj + host ptr.
2. `sw_emule_chip.cpp:write_to_device` / `read_from_device` — env-
   var gated fprintf showing host-side core + l1_dest + first bytes.
3. `dataflow_api.h:noc_async_read` / `noc_async_write` — env-var
   gated fprintf showing src/dst ptrs + first bytes (proves the
   memcpy source/destination data).
4. `llk_pack.h:__llk_pack_untilize` / `common.h:copy_tile` —
   env-var gated fprintf showing CB read ptr + tile data + DST
   contents (proves the data flow through the compute kernel).

Each layer reveals where the data is vs. where the kernel thinks
it should be. Stop at the first layer where the two diverge.

## Files / artifacts produced

- `/tmp/r10_b84_repro.py` — standalone canonical-test repro.
- `/tmp/r10_b84_split.py` — `from_torch`/`to_torch` round-trip check.
- `/tmp/round10/b8_4_dump.npz` — input/output/diff for the canonical
  test (deterministic via `torch.manual_seed(42)`).
- `/tmp/round10/full_trace.log`, `/tmp/round10/noc_trace.log`,
  `/tmp/round10/write_trace.log` — NOC-resolve, read, write traces.

(All `/tmp/round10/` artifacts are scratch — Round 11 can regenerate
from the repro script.)

## Effort spent vs plan

Plan estimate: 30-60 min Phase A.
Actual: ~3 hours instrumentation + 4 dump-runs + 5 layers of
hypothesis-testing. Phase A overran budget because:
- The diff pattern (output-uniformly-zero) wasn't in the plan's
  Phase B table.
- Three iterations of instrumentation needed (resolve → resolve+ptr
  → resolve+core_obj → write src + resolve + read).
- Each iteration required a tt_metal rebuild (libtt_metal.so).

Round 11's first step should be the (C1) test — should take ~30 min
including rebuild + run.
