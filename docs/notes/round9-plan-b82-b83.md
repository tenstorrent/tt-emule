# Round 9 plan — B8.2 (`tensix_types.h` shim) + B8.3 (permute JIT)

Picks up from `round8-sharded-harvest.md`. Targets the two highest
expected-value-per-effort items in the Round 8 register:

| Item | Tier | Effort | Tests unblocked (estimate) |
|---|---|---|---|
| B8.2 — `tensix_types.h` forwarding shim | 1 | ~30 min | up to 68 |
| B8.3 JIT half — `one_packet_{set,with}_state` shims | 1 | ~5 min | 1-2 |
| B8.3 JIT half — `transpose_wh.h` / `pack_untilize.h` LLK gap | 1 | ~30 min, may chain | 1-2 |

Both are tier 1 — missing emule shims. **No tier 3/4 work is in this
round; B8.1 / B8.4 / B8.3-PCC stay deferred.**

Branch: continue on `arminale/tensor-accessor`.

## Goal

After this round, the regression numbers should look like:

| File | Pre Round 9 | Target post Round 9 |
|---|---:|---:|
| `test_interleaved_to_sharded.py` | 6P / 84F / 16S | ~74P / ~16F / 16S — drop the 68 tensix_types-gated failures |
| `test_permute.py -k sharded` | 0P / 8F | ~2P / 6F — close the 2 JIT failures, PCC tier remains |

Net: +70 sharded variants passing on top of the +652 from Round 8.

## Approach

### Step 1: B8.3 set_state/with_state shim (5 min, lowest risk)

Open `include/jit_hw/api/dataflow/dataflow_api.h`. Add after the
existing `noc_async_read_one_packet` alias around line 332:

```cpp
// Stateful one-packet read: silicon programs the NOC with size + base
// in `set_state`, then reuses that state for `with_state` calls. Emule
// is synchronous, so we just memoize the size and use it in the read.
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

(The `inc_num_issued` and `use_vc` template params match upstream so
templated callers parse correctly. The `vc`/`noc` runtime args are
captured but unused since emule has no NOC queues.)

**Verify:**
```bash
cd /localdev/arminale/tt-metal && \
PYTHONPATH=... LD_LIBRARY_PATH=... [...standard env...] \
/opt/ttmlir-toolchain/venv/bin/pytest \
  tests/ttnn/unit_tests/operations/data_movement/test_permute.py \
  -k 'sharded and perm-0-2-3-1' \
  --forked --tb=line -q
```
Expected: the `reader_unary_transpose_hc_sharded_rm.cpp` JIT
compile errors disappear. Some tests may now reach
`transpose_wh_rm.cpp` JIT (which is the next blocker — Step 3).

**STRUCTURE.md update:** add `noc_async_read_one_packet_{set_state,
with_state}` to the dataflow_api.h entry on line 181.

### Step 2: B8.2 `tensix_types.h` forwarding shim (30 min)

#### 2a: Add the shim file

Create `include/jit_hw/tensix_types.h`:
```cpp
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Emule shim for `tensix_types.h`. Suppresses the upstream header's
// host-side `#include <fmt/core.h>` (gated by `#ifndef TENSIX_FIRMWARE`)
// so it compiles in the JIT path, where fmt isn't on -I.
//
// The enums / structs themselves are pure C++ and used by some sharded
// kernels (DataFormat, packer_config_t, ...). Forwarding to the
// upstream definitions keeps emule and silicon in sync.
//
// Real LLK reference:
//   tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h
#ifndef TENSIX_FIRMWARE
#define TENSIX_FIRMWARE
#endif
#include "internal/tt-1xx/wormhole/wormhole_b0_defines/tensix_types.h"
```

#### 2b: Add wormhole_b0_defines to the JIT include path

`tt_metal/impl/emulation/emulated_program_runner.cpp:912`,
`get_extra_include_flags()`:

```cpp
static std::string get_extra_include_flags() {
#ifdef TT_EMULE_PROJECT_SOURCE_DIR
    const std::string project_src = TT_EMULE_PROJECT_SOURCE_DIR;
    std::string extra_inc;
    extra_inc += "-I\"" + project_src + "/ttnn/cpp\"";
    extra_inc += " -I\"" + project_src + "\"";
    extra_inc += " -I\"" + project_src + "/tt_metal/hw/inc\"";
    extra_inc += " -I\"" + project_src + "/tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines\"";  // <-- new
    extra_inc += " -I\"" + project_src + "/tt_metal/hostdevcommon/api\"";
    return extra_inc;
#else
    return {};
#endif
}
```

(Alternative: keep the include path narrow and rely on the relative
path in the shim above. The `-I` approach is needed if *any* other
kernel uses the bare `#include "tensix_types.h"` form without going
through the jit_hw shim. Both kernels in the failing set use the bare
form, so the shim path resolution comes from jit_hw being on -I.)

Rebuild emule: `cmake --build build_emule --target tt_metal -j8`.

#### 2c: Verify and chase the next-layer error if any

```bash
rm -rf /tmp/tt_emule_jit_cache_$(id -u)
... pytest tests/ttnn/unit_tests/operations/data_movement/test_interleaved_to_sharded.py::test_interleaved_to_sharded_hash \
   --forked --tb=line -q | tail -10
```

Expected best case: the 68 failures on
`reader_unary_sharded_blocks_interleaved_start_id.cpp` flip to PASS.

Possible next-layer errors and their fixes:
- **`Noc noc;` doesn't compile** — emule's `api/dataflow/noc.h`
  wrapper might not cover `Noc` construction. Open the file, check
  whether it forwards to upstream or has a stub.
- **`CircularBuffer cb_in(cb_id_in0);` doesn't compile** — likewise
  for `api/dataflow/circular_buffer.h`.
- **`noc.async_read(s, cb_in, tile_bytes, {.page_id=...}, {.offset_bytes=...})`
  template instantiation failure** — TensorAccessor as the `s`
  argument might need a `noc_traits` specialization (added in Round 7
  for ShardView; might need another one).

For each next-layer error, repeat the diagnose recipe:
1. `TT_EMULE_KEEP_JIT_SRC=1` to keep the wrapper.
2. Re-run clang++-20 manually with the JIT command line to see the
   real compiler stderr.
3. Add the missing shim in `include/jit_hw/...`.

If the chain extends beyond 2 hops, stop and document as a separate
Round 10 item — the goal of this round is the tensix_types.h gap
specifically, not a deep metal2-NOC integration push.

**STRUCTURE.md update:** add the new file to the appropriate section
(probably alongside other `include/jit_hw/` top-level shims).

### Step 3: B8.3 compute LLK gap — `transpose_wh.h` / `pack_untilize.h`

This is the second JIT root cause in test_permute_sharded.

Invoke the `/compute-llk-bringup` skill on
`include/jit_hw/api/compute/transpose_wh.h` and
`include/jit_hw/api/compute/pack_untilize.h`. The skill has the
shim-pattern catalog + sfpu_split_includes wiring procedures.

Specifically:
1. Check whether emule's `api/compute/transpose_wh.h` exposes
   `transpose_wh_init_short` and the matching `transpose_wh_tile`
   templates with the same signatures as upstream.
2. Check whether `api/compute/pack_untilize.h` exposes the templated
   `pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>` and
   `pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>`
   variants.
3. If any are missing, follow the skill's "compute shim from scratch"
   recipe.

Stop after Step 3 if `transpose_wh_rm.cpp` compiles cleanly. The 4-6
remaining PCC failures in test_permute_sharded are out of scope here
(B8.4 cluster).

### Step 4: Run regression sweep + commit

Run the affected entries from `scripts/run_ttnn_pytests.sh` one at a
time (per the JIT-cache rule):
- `dm_test_permute_sharded` (if it exists; if not, just run -k sharded
  on test_permute.py directly)
- `dm_test_full_like` + `dm_test_sharded_to_interleaved_oob` to
  confirm no regression.

Then commits:
1. `jit_hw: noc_async_read_one_packet_set_state + with_state shims`
2. `jit_hw: tensix_types.h forwarding shim + wormhole_b0_defines on -I`
3. (if Step 3 lands) `jit_hw: transpose_wh / pack_untilize compute shims`
4. `docs: round 9 — close B8.2 + B8.3 JIT halves` (update
   `round8-sharded-harvest.md` with closure markers + add a tiny
   `round9-closeout.md` if anything novel surfaced)

Push with `--force-with-lease` after explicit user instruction.

## Expected pitfalls

- **`TENSIX_FIRMWARE` define collides with kernel-side code.** The
  upstream header guards `fmt/core.h` behind `#ifndef TENSIX_FIRMWARE`,
  but other code paths might check for that define and do different
  things. Audit upstream for `#ifdef TENSIX_FIRMWARE` before defining
  it globally — may need to restrict the define to just `tensix_types.h`
  via push/pop macro tricks or a wrapper that undefs it after.

- **Step 2c next-layer cascade.** The kernel uses metal2 idioms
  (`Noc noc;`, `CircularBuffer cb_in(cb_id);`) that emule may not have
  wired through for TensorAccessor sources. If the cascade extends
  beyond 2 hops, stop and re-plan rather than chasing it open-ended.

- **`thread_local` in JIT-loaded `.so`.** The `__emule_one_packet_state_size`
  thread_local in Step 1 should work since emule already uses thread_locals
  elsewhere (see `__emule_matmul_state` in `llk_reduce_primitives.h:41`).
  But verify the per-core isolation isn't broken — if two cores share a
  thread, the state will alias.

## Verification gates

Each fix must independently move the needle without regressing
prior passes:

| After step | dm_test_full_like | test_permute_sharded | test_interleaved_to_sharded |
|---|---|---|---|
| Round 8 baseline | 648 P | 0 / 8 | 6P / 84F / 16S |
| Step 1 (set/with_state) | 648 P (no change) | 0-2 / 8 | 6P / 84F / 16S |
| Step 2 (tensix shim) | 648 P (no change) | 0-2 / 8 | ~74P / ~16F / 16S |
| Step 3 (LLK shims) | 648 P (no change) | ~2 / 8 | ~74P / ~16F / 16S |

If `dm_test_full_like` regresses at any step, stop and re-examine — the
get_aligned_page_size dispatch (Round 8 commit `2fbe1ed`) is load-bearing
for that family and shouldn't interact with these shims, but a defensive
check is cheap.
