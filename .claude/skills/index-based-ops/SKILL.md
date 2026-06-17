---
name: index-based-ops
description: Bring up or debug index-based compute ops (TopK, Sort, Argmax) in tt-emule — ops that return values + indices, where ties make the index choice non-unique. Use when implementing one of these ops or triaging its value/index test failures.
---

# Index-based ops (TopK, Sort, Argmax)

These ops return **values + indices**. emule does the math in f32, silicon in
the SFPU; on value ties the index assignment can differ yet still be correct.
Match the **test's contract**, not silicon's exact index choice.

## What the test asserts (TopK; Sort/Argmax are analogous)
`test_topk.py` checks:
1. `assert_numeric_metrics` on **values** (PCC 0.9999, atol/rtol 1e-6).
2. `assert_equal` on **values** vs `torch.topk(sorted=True)` — exact. bf16
   round-trips losslessly (CB bf16 → f32 DST → bf16), so the correct top-K
   *values* match torch exactly regardless of which tied index won. The only
   requirement: select the correct top-K **set**, sorted.
3. **Gather cosine** on indices: `cosine(torch_vals, gather(input, ttnn_idx)) > 0.99`.
   Indices are never compared to torch directly — only via gather, which is
   robust to ties ⇒ indices need only be *valid* (point at the right values).

So "correct but non-identical to silicon" is a non-issue: return the correct
top-K values (sorted) + valid indices and it passes. A throwaway probe (call
the op, then `torch.equal` on values + `CosineSimilarity` on
`gather(input, indices)`) localizes failures fast.

## Sort axis — the #1 gotcha
The ttnn topk kernels run `transpose_wh_tile` on every value/index tile *before*
the LLK calls, moving the sort dim W onto the **row** index of the 32×32 DST
tile (`DST[r*32+c] = input[H=c][W=r]`). Top-K along W at fixed H=c compares
datums across **rows** at a fixed **column** → each column is an independent
sort sequence. A per-row implementation returns garbage.

## TopK kernel paths and the three LLK shims
| Path | Kernel(s) | When | Uses |
|---|---|---|---|
| Single-core insertion sort | `topk.cpp` | dim-size < `multi_core_min_width` (8192) | `topk_local_sort` (`i_end_phase=5`) |
| Multi-core bitonic + cross-core agg | `topk_local.cpp` + `topk_final.cpp` | dim-size ≥ 8192 | `topk_local_sort`, `topk_merge`, `topk_rebuild` |

Shims in `compute_kernel_api.h` (`namespace __emule_topk`): `merge_split_col` =
full-sort 64 (V0∪V1), top-32→V0 / bottom-32→V1; `sort_col` = sort one tile's 32
per column.
- `topk_local_sort(idir, i_end_phase)`: `i_end_phase>=5` → `merge_split_col`;
  `<5` → independent per-tile `sort_col`.
- `topk_merge<top_min>` **and** `topk_rebuild`: both "move the larger 32 into V0,
  lower 32 into V1" (`topk_common_funcs.hpp:100,160`) → both = `merge_split_col`.
  The full 64-sort is order-independent, so the kernel's reduction tree converges
  to the global top-K for all K (32/64/3000) without reproducing the SFPU's exact
  bitonic intermediate state.

## Sort (`ttnn.sort`)
Reuses the TopK shims verbatim (`topk_local_sort`/`topk_merge`, not
`topk_rebuild`) — `merge_split_col` is also correct for a full sort, so no new
math is needed. It selects one of three program factories by width-in-tiles
`Wt`: single-core (`Wt<=64`) and cross-core exchange are supported; the
single-row multi-core DRAM path (largest `Wt`) is not (see below).

The cross-core reader collapses an L1 pointer to a 32-bit address inside an
`#include`d header, which the JIT x86 patcher now reaches (companion tt-metal
change). For the semaphore handshake itself see *Cross-core handshakes* below.

Unsupported — multi-core DRAM path: its coordinator broadcasts a toggled VALID/0
release via `noc_semaphore_set_multicast`, which races under emule's synchronous
(zero-latency) multicast — the consumer can lose a release. Pacing the multicast
fixes sort but stalls sticky-signal multicasts (e.g. matmul DRAM-sharded), and
the two are indistinguishable at the multicast layer; the cases are excluded from
the regression entries.

## Argmax (`ttnn.argmax`)
**Dataflow-only** — no compute/SFPU; the max-scan runs in the NCRISC reader.
Indices are compared **exactly** to `torch.argmax` (first/lowest max wins), not
gather-validated, so the kernel's strict-`>` / `idx<max_idx` tie-break must run
faithfully (it does — emule just executes it). Three readers: single-core
ROW_MAJOR, single-core TILE (walks faces, excludes the `-42` implicit padding),
multi-core ROW_MAJOR. No new shim needed — `api/numeric/*` and
`api/debug/waypoint.h` resolve to real tt-metal headers (waypoint is a no-op
without `WATCHER_ENABLED`). The multi-core path depends on the cross-core
start-ordering below.

## Cross-core handshakes under emule
A reducer-style op (workers signal a collator core, which waits) leans on two
silicon timings emule collapses. When one hangs, suspect these before declaring a
path unsupportable:

1. **Zero-latency atomics.** `noc_semaphore_inc` lands instantly, so a polling
   waiter can skip past a value the NOC's latency would have let it observe. Use
   `noc_semaphore_wait`'s `>= val` for monotonic counters (exact `==0` only for a
   VALID→0 toggle). The class `Semaphore<>::wait` is exact-match — fine for a
   counter that settles at its terminal value, but not robust to overshoot.
2. **Instant reads + sequential thread spawn.** A worker can run to its increment
   before a peer has executed its prologue. If that peer is an *ungated* reducer
   resetting its counter (no start-sem gate that iteration), the reset clobbers the
   early increment and the exact-match wait hangs — worse with more cores. Fixed
   globally (not per-op) by the startup barrier + first-read latency in
   `docs/noc-emulation.md` §1.1; both were needed (barrier alone still raced).

Symptom to recognize: `Semaphore::wait(N) stuck at M<N`, nondeterministic, scaling
with core count. *Genuinely* unsupportable is different — see the sort multi-core
DRAM multicast race above (a toggled release the synchronous multicast loses).

## Infra gaps these ops surfaced (general, not topk-specific)
1. **`__emule_pack_offset` resets on `cb_push_back`, not `cb_reserve_back`**
   (`cb_api.h`, matching silicon `pack.h`). The multi-core final kernel fills a
   staging CB with `reserve_back(1)` per tile + one trailing `push_back(Wt)`,
   relying on auto-advancing `pack_tile(0,cb)` to hit distinct slots.
2. **uint16 unpack clamp** (`common.h::__emule_unpack_cb_tile_to`): clamp `n` to
   the tile bound (an oversized index page on the dim=1 path indexed
   `rowmajor_to_nfaces[]`/DST out of bounds).
3. **Program-global CB addressing** (`emulated_program_runner.cpp`): cross-core
   `get_write_ptr(cb)` for a remote-core CB — see `references/emule-mapping.md` §4.
4. **`dataflow_api.h` includes `hostdevcommon/common_values.hpp`** (INVALID/VALID).
5. **Self-recycled CB reserve** (`cb_api.h`): non-blocking `cb_reserve_back` for a
   CB the same compute thread also consumes — the in-place bitonic recycle
   deadlocks otherwise (emule runs compute single-threaded).

## Debug method that worked
Instrument the dataflow/compute boundary with `fprintf` in the JIT kernels/shims
(they run as host threads): dump per-tile values at the final-core input copy and
after the merge reduction to separate "data present but not surfaced" (merge/pack
bug) from "data missing" (inter-core transfer bug).
