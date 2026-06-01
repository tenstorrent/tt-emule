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

### B8.1 — test_pad sharded stick-layout: post-TRID data mismatch

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

**Blast radius:**
- 23 / 23 of `test_pad.py` `-k 'sharded'` failures.
- 16 of the 20 stick-layout failures in `test_interleaved_to_sharded.py`
  (the other 4 now pass — these 16 likely share this same root cause).
- Total: ~39 sharded tests gated on this single emule bug.

---

### B8.2 — test_interleaved_to_sharded: `tensix_types.h` include gap

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

**Blast radius:**
- 68 / 84 i2s failures gated on this single root cause.
- May overlap with other sharded-output families that pull this
  kernel — needs an audit.

---

### B8.3 — test_permute sharded: transpose kernels miss
`noc_async_read_one_packet_with_state` family

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

**Where it points:**
- Kernel uses:
  ```cpp
  noc_async_read_one_packet_set_state(noc_read_addr, stick_size_bytes);
  ...
  noc_async_read_one_packet_with_state(noc_read_addr, l1_write_addr);
  ```
- Upstream definitions at
  `tt_metal/hw/inc/api/dataflow/dataflow_api.h:594` (`set_state`) and
  `:627` (`with_state`).
- Emule has `noc_async_read_one_packet` (full args) but no `set_state` +
  `with_state` pair.

**Suspected root cause:**
- 2 JIT: tier 1, missing shim pair. Easy fix — store the
  packet_size in a thread-local on `set_state`, use it on `with_state`
  to memcpy.
- 6 PCC: separate gap. Trace would be needed to characterize whether
  it's an addressing bug, a CB-binding bug, or a layout-conversion
  issue in the permute kernels.

**What was tried:**
Diagnosed only. No fix attempted.

**Why deferred:**
Smaller blast radius than B8.1 / B8.2 — 8 tests total. Bundled into
Round 8 for future cleanup. Adding the `set_state`/`with_state` pair is
trivial; should be done alongside any future TRID-state work since the
APIs are related.

**Blast radius:**
- 8 / 8 permute sharded failures (2 JIT + 6 PCC). The 6 PCC may share
  root cause with B8.4 (untilize ATOL≈3.25) since both touch
  interleaved↔sharded layout conversions.

---

### B8.4 — test_untilize sharded: interleaved↔sharded ATOL ≈ 3.25

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

**Blast radius:**
- 196 / 196 untilize sharded failures.
- Likely overlaps with the 6 permute PCC failures from B8.3 — both
  involve interleaved↔sharded layout conversions.
- May affect tilize counterparts that haven't been sharded-tested yet.

---

### B8.5 — `noc_async_read_one_packet_set_state` / `_with_state`
missing (cross-reference)

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
