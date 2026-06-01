# Round 8 — sharded harvest

Picks up from `round7-tensor-accessor-bringup.md`. Round 7 landed the
structural tensor-accessor work; this round harvests the sharded tests
that were filtered out of `scripts/run_ttnn_pytests.sh` (32 `run_pytest`
entries carrying `-k 'not sharded'`).

Branch: `arminale/tensor-accessor` (tt-emule).
Companion tt-metal branch: `arminale/tensor-accessor-fix` (commit
`593e36cfa63`).

## What landed in this round

Two surgical emule fixes in
`include/jit_hw/api/dataflow/dataflow_api.h`:

1. **`noc_async_{read,write}_page` → 3-way dispatch matching upstream.**
   Upstream's `noc_async_read_page` (line 1064 of
   `tt_metal/hw/inc/api/dataflow/dataflow_api.h`) tries
   `get_aligned_page_size()` first, then `page_size`, then
   `log_base_2_of_page_size`. Emule had only the last two — which
   compiled for legacy `InterleavedAddrGen*` (member `page_size`) but
   not for `TensorAccessor` (only exposes `get_aligned_page_size()`).
   Added `has_get_aligned_page_size_v<T>` SFINAE trait + first
   `if constexpr` branch in both `noc_async_read_page` and
   `noc_async_write_page`. ~10 lines.

2. **TRID no-op shims.** Silicon uses Transaction IDs to overlap
   concurrent async reads/writes on the same NOC with distinct tags,
   then polls per-tag completion. Emule is synchronous — every NOC op
   completes inline before returning — so the missing TRID APIs are
   trivially no-ops / `return true`:
   - `noc_async_read_set_trid(trid=0, noc=0)` → no-op
   - `noc_async_write_set_trid(trid=0, noc=0)` → no-op
   - `ncrisc_noc_read_with_transaction_id_flushed(noc, trid)` → `true`
   - `ncrisc_noc_nonposted_write_with_transaction_id_sent(noc, trid)` → `true`
   - `ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc, trid)` → `true`

   `noc_async_read_one_packet_with_state_with_trid` is **omitted on
   purpose** — its 16 call sites are all in experimental ccl /
   deepseek / prefetcher kernels not in the regression scope, and
   adding it correctly needs the separate `noc_async_read_one_packet_{
   set_state,with_state}` family which is its own gap.

## Bottom-line table (sharded variants, pre vs post)

| File | Pre Round 8 | After fix 1 (`get_aligned_page_size`) | After fix 2 (+ TRID shims) | Filter disposition for Phase C |
|---|---:|---:|---:|---|
| `test_full_like.py` -k sharded | 0P / 648F | **648P / 0F** | 648P / 0F | **DROP filter** |
| `test_pad.py` -k sharded | 78P / 23F | 78P / 23F | 78P / 23F (compile→PCC) | KEEP filter (PCC blocked) |
| `test_permute.py` -k sharded | 0P / 8F | 0P / 8F | 0P / 8F | KEEP filter |
| `test_untilize.py` -k sharded | 288P / 196F | 288P / 196F | 288P / 196F | partial — subset entry |
| `test_interleaved_to_sharded.py` | 2P / 88F | 2P / 88F | **6P / 84F** | partial — subset entry |
| `test_creation.py` -k sharded | 0 selected | 0 selected | 0 selected | DROP no-op filter |
| `test_reduction_mean.py` -k sharded | 0 selected | 0 selected | 0 selected | DROP no-op filter |
| `test_sum.py` -k sharded | 0 selected | 0 selected | 0 selected | DROP no-op filter |

**Net new sharded variants passing: +652** (648 full_like + 4 i2s).

## Round 8 bug register

Each entry is the cold-pickup brief for one residual failure cluster:
**reproducer**, **what you see**, **where it points**, **suspected
root cause**, **what was tried**, **why deferred**, **blast radius**.

### B8.1 — test_pad sharded stick-layout: post-TRID data mismatch  **[CLOSED in Round 11 as side effect of NUM_L1_BANKS fix, see `round11-closeout.md`]**

**Reproducer (canonical):**
```bash
cd /localdev/arminale/tt-metal && \
PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD/build_emule/lib:$PWD \
LD_LIBRARY_PATH=$PWD/build_emule/lib \
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD \
TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 \
TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml \
MESH_DEVICE=N150 \
/opt/ttmlir-toolchain/venv/bin/pytest \
  "tests/ttnn/unit_tests/operations/data_movement/test_pad.py" \
  -k 'sharded and test_pad_rm_sharded_stickwise and INT32' \
  --forked --tb=line -q
```
Full set: `-k 'sharded'` on `test_pad.py` selects 148 (78 pass, 47 skip,
23 fail). The 23 fail span `test_pad_rm_sharded_stickwise` (14),
`test_pad_nd_sharded_to_interleaved` (5),
`test_pad_legacy_sharded_to_interleaved` (4).

**What you see:**
```
E   AssertionError: Data mismatch
E   AssertionError: 0.0  (PCC zero)
E   AssertionError: 0.012004872007083925  (low PCC)
```
Before the TRID fix the same 23 failed with
`RuntimeError: jit_compile_kernel: compiler failed (exit 256) for
kernel: .../reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`
referencing `use of undeclared identifier 'noc_async_read_set_trid'`.
After the shims the kernel compiles and runs, but the multi-slot
non-aligned path produces wrong output.

**Where it points:**
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`
  — the `aligned == false` branch at lines 41-107.
- The kernel uses 4 slot states (IDLE / SRC_PENDING / SCRATCH_READY /
  SCRATCH_PENDING) plus N TRIDs to pipeline src→scratch and
  scratch→dest copies. The scratch→dest step uses
  `get_noc_addr(scratch_write_addrs[slot] + aligned_offset)` to read
  back a stick from the kernel's own L1 with a byte-aligned offset.

**Suspected root cause (Hypothesis tier 3 — host/UMD divergence,
emule-side):**
Self-loop NOC read where source and dest are both on the calling
core's L1. The `get_noc_addr(local_l1_addr + offset)` returns a
NOC address with this core's xy embedded; `noc_async_read(...)` resolves
it back and memcpys. If the resolve uses a path that hits the *original*
buffer (sharded input) rather than the kernel's *scratch CB* L1
allocation, or if `scratch_write_addrs[slot]` is being computed off the
wrong CB base, the read sees stale/aliased memory.

The `aligned == true` branch reads directly into dest and skips the
scratch dance — it's not exercised by these 23 variants (their padding
makes them non-aligned).

**What was tried:**
- Added the 5 TRID no-op shims — unblocked JIT compile but pass-count
  didn't change.
- Did *not* attempt to instrument the scratch→dest copy path.

**Why deferred:**
Likely needs an emule-internal NOC-loopback / CB-base trace
(`TT_EMULE_TRACE_NOC`-style fprintf around `__emule_resolve_noc_addr`
when source xy == this core's xy). 2-3 attempts not yet spent. Out of
routine bring-up scope until someone budgets the time for that trace.

**Instrumentation recipe (Round 9 starting point):**

1. Re-create the env-var-gated NOC trace that Round 7 used and then
   reverted (referenced in `round7-tensor-accessor-bringup.md`
   §"Tooling infrastructure"):
   ```cpp
   // In tt_metal/impl/emulation/emulated_program_runner.cpp,
   // alongside the existing __emule_resolve_noc_addr definition.
   static thread_local FILE* __emule_noc_trace_fp = []() {
       const char* p = std::getenv("TT_EMULE_TRACE_NOC");
       return p ? std::fopen(p, "a") : nullptr;
   }();
   // Then at the top of __emule_resolve_noc_addr:
   if (__emule_noc_trace_fp) {
       uint32_t x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
       uint32_t y = (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
       uint32_t off = static_cast<uint32_t>(noc_addr & ((1ULL << NOC_ADDR_LOCAL_BITS) - 1));
       bool self = (x == my_x[0] && y == my_y[0]);
       std::fprintf(__emule_noc_trace_fp,
           "resolve noc=(%u,%u) off=0x%x self=%d size=%u from=(%u,%u)\n",
           x, y, off, self ? 1 : 0, /*size=*/0, my_x[0], my_y[0]);
       std::fflush(__emule_noc_trace_fp);
   }
   ```

2. Run one canonical failing variant with the trace on:
   ```bash
   TT_EMULE_TRACE_NOC=/tmp/round9-trace.log \
   /opt/ttmlir-toolchain/venv/bin/pytest \
     'tests/ttnn/unit_tests/operations/data_movement/test_pad.py::test_pad_rm_sharded_stickwise[dtype=DataType.INT32-input_shape=(1, 1, 2, 4)-pad_to_shape=(1, 1, 4, 8)-input_tensor_start=(0, 0, 0, 0)-pad_value=3.0-input_sharded_memory_config_args={"core_grid": ttnn.CoreGrid(x=1, y=2), "strategy": <ShardStrategy.HEIGHT: 1>}]' \
     --forked --tb=line -q
   ```

3. **What to look for in the trace:**
   - Lines where `self=1` (source xy == calling core's xy). These are
     the in-kernel `noc_async_read(get_noc_addr(scratch + offset),
     dest, ...)` operations.
   - The `off` (offset) should match `scratch_write_addrs[slot] +
     aligned_offset` from the kernel — within L1, not into a DRAM bank
     and not into another core's L1.
   - Compare against the resolved pointer: does it correspond to the
     L1 region that was JUST written by the prior
     `noc_async_read(src_dram, scratch[slot], aligned_block_width_bytes)`?
     If not, you've found the routing miss.

4. Also instrument the existing emule debug paths (already in
   `dataflow_api.h:362-365` for multicast) so you can correlate
   `EMULE WARN: noc_async_read failed to resolve` with the trace.

**Blast radius:**
- 23 / 23 of `test_pad.py` `-k 'sharded'` failures (all
  `test_pad_rm_sharded_stickwise` / `test_pad_nd_sharded_to_interleaved` /
  `test_pad_legacy_sharded_to_interleaved`).
- 16 of the 20 stick-layout failures in `test_interleaved_to_sharded.py`
  (the other 4 now pass — these 16 likely share this same root cause).
- Total: ~39 sharded tests gated on this single emule bug.

---

### B8.2 — test_interleaved_to_sharded: `tensix_types.h` include gap  **[CLOSED in Round 9, see `round9-closeout.md`]**

**Reproducer:**
```bash
... pytest tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py::test_interleaved_to_sharded_hash \
  --forked --tb=line -q
```
Full set: 106 selected, 6 pass / 84 fail / 16 skip. The 84 fails span
`test_interleaved_to_sharded_hash` (24),
`test_interleaved_to_dram_sharded_via_to_memory_layout` (24),
`test_interleaved_to_dram_width_sharded` (12),
`test_interleaved_to_dram_height_sharded` (12),
`test_interleaved_to_sharded_nd_with_equivalent_2d` (8),
`test_interleaved_to_dram_sharded_convert_dtype` (8).

**What you see:**
```
E   RuntimeError: jit_compile_kernel: compiler failed (exit 256) for kernel:
    .../reader_unary_sharded_blocks_interleaved_start_id.cpp
```
Repro the underlying clang error with `TT_EMULE_KEEP_JIT_SRC=1` then
manually invoke `/usr/bin/clang++-20` on the kept wrapper:
```
.../patched_kernel.cpp:11:10: fatal error: 'tensix_types.h' file not found
```
Adding `-I.../wormhole_b0_defines` exposes the next layer:
```
tensix_types.h:11:10: fatal error: 'fmt/core.h' file not found
```

**Where it points:**
- Kernel: `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp`
- Line 11 includes `tensix_types.h`.
- That header is at
  `tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h`,
  which `#include <fmt/core.h>` when `TENSIX_FIRMWARE` is undefined
  (which it is in the JIT path).
- The kernel also uses metal2 idioms — `Noc noc;` /
  `noc.async_read(s, cb_in, ...)`, `CircularBuffer cb_in(...);
  cb_in.reserve_back(...)` — not yet wired through emule's
  `api/dataflow/noc.h` / `api/dataflow/circular_buffer.h` shims for
  TensorAccessor sources.

**Suspected root cause (Hypothesis tier 1 — emule shim thin):**
A pair of gaps:
1. Emule's `get_extra_include_flags()` in
   `tt_metal/impl/emulation/emulated_program_runner.cpp:912` doesn't
   forward the `wormhole_b0_defines/` include dir. Even if it did,
   `tensix_types.h` would still pull in `fmt/core.h` which isn't on the
   JIT path.
2. A `jit_hw/tensix_types.h` shim is needed — exposes only the enums /
   structs kernels actually consume (likely `xmov_direction_t`,
   `tdma_mover_id_t`, packer/datum-format enums) without the host-side
   `fmt` dependency.

The kernel's metal2 NOC + CB usage with TensorAccessor source needs to
be confirmed working — adding the tensix_types shim is necessary but
might not be sufficient.

**What was tried:**
- Diagnosed via `TT_EMULE_KEEP_JIT_SRC=1` + manual `clang++-20`
  invocation on the kept wrapper. Two layers of missing headers
  identified before stopping.

**Why deferred:**
Adding the shim requires deciding which symbols emule actually needs to
expose vs leave undefined. ~30 min of design + implementation, plus a
follow-up re-run to expose whatever's next. Out of scope for the
current harvest commit; targeted for Round 8 follow-up.

**Symbol audit (Round 9 starting point):**

Grep'd the failing kernel for direct references to anything declared in
`tensix_types.h`:

```bash
grep -oE 'xmov_direction|MOVER0|MOVER1|TDMA_MOVER|relu_mode|math_fidelity_t|stochastic_round|packer_config|fifo_ctl|mover_config|tile_descriptor|TileHeader|SectionHeader|DataFormat|io_queue_pointers|BUFFER_TYPE' \
  ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp
```
→ **no matches.** The kernel includes `tensix_types.h` defensively
but doesn't reference any of its symbols directly. Nor do the emule
shims it pulls before that point (`api/dataflow/{dataflow_api,
circular_buffer,noc}.h`, `api/tensor/noc_traits.h`) — they're already
self-contained.

The include is satisfying the include directive itself, not pulling in
symbols.

**Recommended shim design:**

Two options, ordered by safety-vs-effort tradeoff:

1. **Forwarding shim** (recommended): create
   `include/jit_hw/tensix_types.h` that pre-defines `TENSIX_FIRMWARE`
   and then includes the upstream header (which guards the `fmt/core.h`
   include behind `#ifndef TENSIX_FIRMWARE`):
   ```cpp
   #pragma once
   // Suppress upstream's host-side <fmt/core.h> include, which is not
   // on the JIT compile path. The actual struct/enum definitions in
   // tensix_types.h are pure C++ and compile fine without fmt.
   #ifndef TENSIX_FIRMWARE
   #define TENSIX_FIRMWARE
   #endif
   #include "internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h"
   ```
   Also requires `get_extra_include_flags()` in
   `tt_metal/impl/emulation/emulated_program_runner.cpp:912` to add
   the `wormhole_b0_defines/` directory to -I, or use the existing
   `tt_metal/hw/inc` -I and adjust the include path string accordingly.

   Pro: kernels that DO use DataFormat / packer_config_t etc.
   automatically get them. Future-proof for other sharded kernels.
   Con: pulls in unused symbols (mild compile-time cost).

2. **Empty stub**: `#pragma once` + nothing else. This kernel works
   with it (zero symbols needed). Other kernels that reference real
   tensix_types symbols would still fail, but those would be diagnosed
   one at a time.

Pick (1). The kernel also uses metal2 idioms (`Noc noc;
noc.async_read(s, cb_in, ...)`, `CircularBuffer cb_in;
cb_in.reserve_back(...)`), so after the shim lands, a re-run will
surface whether the metal2 NOC/CB integration with TensorAccessor is
already covered by Round 7 shims or if there's a second gap behind
this one.

**Blast radius:**
- 68 / 84 i2s failures gated on this single root cause.
- May overlap with other sharded-output families that pull this
  kernel — needs an audit. Specifically `test_concat` sharded
  variants (the concat kernels `reader_height_sharded_*` include
  `tensix_types.h` too — see kernel file list grep'd during B8.5).

---

### B8.3 — test_permute sharded: transpose kernels miss
`noc_async_read_one_packet_with_state` family  **[JIT half CLOSED in Round 9; PCC tier (6 of 8) deferred to Round 10, see `round9-closeout.md`]**

**Reproducer:**
```bash
... pytest tests/ttnn/unit_tests/operations/data_movement/test_permute.py \
  -k 'sharded' --forked --tb=line -q
```
Selects 8 / deselects 1581, all 8 fail. Subcategories: 2 JIT compile
on the transpose RM kernel + 6 PCC mismatches.

**What you see:**
- 2 JIT failures:
  ```
  RuntimeError: jit_compile_kernel: compiler failed (exit 256) for kernel:
    .../transpose/device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp
  ```
  with clang error `use of undeclared identifier
  'noc_async_read_one_packet_set_state'` /
  `'noc_async_read_one_packet_with_state'`.
- 6 PCC failures: data-correctness gaps on permute paths that *do*
  compile.

**Where it points (two distinct JIT sub-causes + a PCC tier):**

The 8 failing test cases all share `dtype=BFLOAT16-layout=ROW_MAJOR-shape=[16,8,224,224]`
and split by `(input_sharding, output_sharding) × perm`:

| `input_sharding` | `output_sharding` | `perm` permutations | count |
|---|---|---|---|
| None | HEIGHT | [0,2,3,1] / [0,3,2,1] / [1,2,3,0] / [1,3,2,0] | 4 |
| HEIGHT | HEIGHT | same 4 perms | 4 |

Two distinct kernels fail to JIT-compile (one occurrence each):

1. **Reader kernel `reader_unary_transpose_hc_sharded_rm.cpp`** uses
   ```cpp
   noc_async_read_one_packet_set_state(noc_read_addr, stick_size_bytes);
   ...
   noc_async_read_one_packet_with_state(noc_read_addr, l1_write_addr);
   ```
   Upstream definitions at
   `tt_metal/hw/inc/api/dataflow/dataflow_api.h:594` (`set_state`) and
   `:627` (`with_state`). Emule has `noc_async_read_one_packet` (full
   args) but no `set_state`/`with_state` pair.

2. **Compute kernel `compute/transpose_wh_rm.cpp`** uses LLK shims
   `transpose_wh_init_short`, `transpose_wh_tile`, plus
   `pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>` and
   `pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>`.
   It includes `api/compute/transpose_wh.h` and
   `api/compute/pack_untilize.h`. Check via the
   `/compute-llk-bringup` skill which of those shims is missing or
   has mismatched template signatures.

The remaining ~6 cases that do JIT-compile likely fail PCC — exact
breakdown wasn't captured (the `--tb=line` summary clumps errors at
end). Re-run with `--tb=long` on the 8 failing IDs above to map
each to its sub-cause.

**Suspected root cause:**

| Sub-cause | Tier | Effort | Tests unblocked (estimate) |
|---|---|---|---|
| Missing `noc_async_read_one_packet_{set,with}_state` | 1 | ~5 min | 1-2 of 8 |
| `transpose_wh.h` / `pack_untilize.h` LLK shim gap | 1 | ~30 min, may chain | 1-2 of 8 |
| PCC: permute interleaved↔sharded layout conversion | 3 | hours | 4-6 of 8 |

**What was tried:**
Diagnosed only. No fix attempted.

**Why deferred:**
Smaller blast radius than B8.1 / B8.2 — 8 tests total. The two JIT
sub-causes are trivial to fix; the PCC tier likely shares root cause
with B8.4 (untilize ATOL≈3.25) since both touch interleaved↔sharded
layout conversions.

**Blast radius:**
- 8 / 8 permute sharded failures (2 distinct JIT root causes + 6 PCC).
- The 6 PCC likely overlap with B8.4 — fixing B8.4 may close them too.

---

### B8.4 — test_untilize sharded: interleaved↔sharded ATOL ≈ 3.25  **[CLOSED in Round 11, see `round11-closeout.md` — root cause was NUM_L1_BANKS=1 vs host's bank distribution; 188 of 196 now pass; 8 residuals on sharded→interleaved writer become new B12.1]**

**Reproducer:**
```bash
... pytest tests/ttnn/unit_tests/operations/data_movement/test_untilize.py \
  -k 'sharded' --forked --tb=line -q
```
Selects 488 / deselects 311. Result: 288 pass / 196 fail / 4 skip /
2 xfail. **All sharded↔sharded variants pass**; the 196 failures
cluster exclusively on interleaved↔sharded paths.

**What you see:**
```
E   AssertionError: Max ATOL Delta: 3.25, Max RTOL Delta: nan
E   AssertionError: Max ATOL Delta: 3.328125, Max RTOL Delta: nan
```
Systematic numeric divergence, **not** zeros. The magnitude (≈3.25)
suggests a fixed BFLOAT16 quantization or tile/face addressing offset,
not garbage data.

**Where it points:**
- Failing test functions (by count):
  - `test_untilize_multi_core_interleaved_to_sharded` — 54
  - `test_untilize_multi_core_nd_sharded_to_interleaved` — 28
  - `test_untilize_multi_core_sharded_to_interleaved` — 24
  - `test_untilize_multi_core_interleaved_to_nd_sharded` — 16
  - `test_untilize_single_core_sharded_to_interleaved` — 12
  - `test_untilize_single_core_interleaved_to_sharded` — 12
  - `test_untilize_multi_core_interleaved_to_uneven_sharded` — 12
  - `test_untilize_single_core_interleaved_to_sharded_writer_kernel_tensor_addrgen_test` — 8
  - `test_untilize_single_core_interleaved_to_nd_sharded` — 8
  - `test_untilize_multi_core_sharded_to_interleaved_uneven_input_shard_spec` — 6
- All 10 are interleaved↔sharded (or vice versa). None are
  sharded↔sharded.

**Suspected root cause (Hypothesis tier 3 or 4):**
The untilize compute step (tile→row-major) works correctly for
sharded↔sharded data paths but produces a systematic offset on
interleaved-side reads/writes. Candidates:
- `nfaces` ↔ row-major tile-layout conversion table mis-indexed on the
  interleaved-side write path.
- A page-offset arithmetic gap in
  `InterleavedPow2AddrGenFast::get_noc_addr` (the variant added in
  Round 7 commit `7509e39`) that produces wrong offsets for non-tile-
  aligned access patterns.
- A `reset_l1_bump` / MEM_ZEROS interaction that taints a different
  region — different from #59 since that was already fixed for
  uneven shards.

**What was tried:**
Diagnosed via failure clustering only. No instrumentation attempted —
196 tests is too large to chase one at a time without a hypothesis.

**Why deferred:**
Needs a focused trace on ONE failing test (pick the smallest like
`test_untilize_single_core_interleaved_to_sharded` with the smallest
tensor) to characterize: is the ATOL constant or proportional? Is it
per-tile or per-face? Where in the output does the error appear? Once
characterized, the fix is likely a single arithmetic correction
somewhere in the addressing or packing path.

**Canonical smallest reproducer (Round 9 starting point):**

The smallest tensor in any failing `test_untilize_single_core_interleaved_to_sharded`
variant is `[2, 2, 256, 512]` (BFLOAT16, 4 cores). Pick the
HEIGHT_SHARDED + ROW_MAJOR-orientation node ID:
```
tests/ttnn/unit_tests/operations/data_movement/test_untilize.py::test_untilize_single_core_interleaved_to_sharded[num_shard_cores=4-standard_shard_core_grid={[0-0 - 0-3]}-block_shard_core_grid={[0-0 - 1-1]}-output_shard_orientation=ShardOrientation.ROW_MAJOR-output_memory_layout=TensorMemoryLayout.HEIGHT_SHARDED-tensor_shape=[2, 2, 256, 512]-dtype=DataType.BFLOAT16]
```

**Characterization trace (write this in test_untilize.py temp instrumentation):**

The test compares torch reference against ttnn output. Inject prints
around the assertion:
```python
# In the failing assertion site of test_untilize.py:
diff = (tt_output - torch_output).abs()
import torch as _t
print(f"max ATOL: {diff.max().item()}, indices: {_t.nonzero(diff > 1e-3)[:20]}")
print(f"first 8 tt rows: {tt_output.flatten()[:32]}")
print(f"first 8 torch rows: {torch_output.flatten()[:32]}")
```

Questions to answer:
1. Is the ATOL constant (every wrong element = 3.25) or varying?
2. Are the wrong elements clustered at specific positions (e.g.
   every 16th value, every 32nd, last column of each face)?
3. Are wrong values shifted versions of correct values, or unrelated?

Each of those points to a different root cause:
- Constant ATOL on specific positions → face-layout / nfaces conversion
  bug.
- Shifted values → off-by-one in addressing.
- Unrelated values → wrong buffer read entirely.

**Three candidate root causes ranked by likelihood:**

1. **(most likely)** `nfaces`↔row-major tile-layout conversion table
   mis-indexed on the interleaved-side write path. Sharded↔sharded
   doesn't go through this table; interleaved↔sharded does. Find the
   table in `include/jit_hw/` and check whether the index expression
   matches upstream's silicon-side equivalent.
2. Page-offset arithmetic gap in
   `InterleavedPow2AddrGenFast::get_noc_addr` (Round 7 commit
   `7509e39`) for non-tile-aligned access patterns. Test sharded↔interleaved
   with `tensor_shape=[2, 2, 256, 512]` against a known-good
   interleaved-only test — if the latter passes, the bug is in the
   sharded-side handoff, not the pow2 addrgen.
3. A `reset_l1_bump` / MEM_ZEROS interaction taintng a different
   region — different from #59. Less likely since #59 was uneven
   shards only and these are even shards.

**Blast radius:**
- 196 / 196 untilize sharded failures.
- Likely overlaps with the 6 permute PCC failures from B8.3 — both
  involve interleaved↔sharded layout conversions.
- May affect tilize counterparts that haven't been sharded-tested yet.

---

### B8.5 — `noc_async_read_one_packet_set_state` / `_with_state`
missing (cross-reference)  **[CLOSED in Round 9, see `round9-closeout.md`]**

Cross-reference of the JIT-compile-only sub-cause of B8.3. Trivial fix
(2 stubs, ~10 lines):

```cpp
inline thread_local uint32_t __emule_one_packet_state_size = 0;
template <bool use_vc = false>
inline void noc_async_read_one_packet_set_state(
    uint64_t /*src_noc_addr*/, uint32_t size,
    uint32_t /*vc*/ = 0, uint8_t /*noc*/ = 0) {
    __emule_one_packet_state_size = size;
}
template <bool inc_num_issued = true, bool use_vc = false>
inline void noc_async_read_one_packet_with_state(
    uint64_t src_noc_addr, uint32_t dst_local_l1_addr,
    uint32_t /*vc*/ = 0, uint8_t noc = 0) {
    noc_async_read(src_noc_addr, dst_local_l1_addr,
                   __emule_one_packet_state_size, noc);
}
```

Bundled into B8.3 because the PCC component is the real blocker.
Adding these shims alone unblocks 2 of 8 permute failures but leaves
the other 6.

---

## Diagnostic prior (carried from the Round 7 plan)

When a sharded test fails under emule, default hypothesis order:

1. **Emule shim missing / thin** — most fixes are at this tier.
2. **Emule define / config mismatch** with what the kernel reads.
3. **Host-side write/read divergence** between
   `ttnn.from_torch`/`to_torch` and the kernel's `TensorAccessor` view.
4. **`reset_l1_bump` / address-layout interaction.**
5. **Compute-kernel gap unrelated to TensorAccessor.**
6. **Upstream TensorAccessor bug** (rare — rule out the above first).

B8.1 is tier 3. B8.2 is tier 1. B8.3 is tier 1 + tier 3. B8.4 is
tier 3 or tier 4.

## Files touched this round

- `include/jit_hw/api/dataflow/dataflow_api.h` — `noc_async_*_page`
  3-way dispatch + TRID no-op shims (the two commits).
- `scripts/run_ttnn_pytests.sh` — harvest commit (per Phase C).
- `docs/notes/round7-tensor-accessor-bringup.md` — closure / branch
  state refresh.
- `docs/notes/round8-sharded-harvest.md` (this file).

## What's next (Round 9)

The Round 8 register is the agenda. Highest expected-value targets:

1. **B8.2** (`tensix_types.h` shim) — 68 tests, tier 1, ~30 min.
2. **B8.3 JIT half** (`one_packet_{set,with}_state` shims) — 2 tests,
   tier 1, ~5 min.
3. **B8.1** (self-loop NOC scratch→dest data mismatch) — 39 tests,
   tier 3, needs an instrumented NOC trace.
4. **B8.4** (untilize ATOL ≈ 3.25) — 196 tests, needs a focused single-
   test trace before any fix attempt.
