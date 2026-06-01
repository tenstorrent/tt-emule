# Round 9 closeout — B8.2 + B8.3 (JIT half) landed

Round 9 plan from `round9-plan-b82-b83.md` executed in 3 steps + verification.
Three commits on `arminale/tensor-accessor` on top of Round 8.

## Bottom-line table

| File | Round 8 baseline | Round 9 result | Δ vs Round 8 |
|---|---:|---:|---:|
| `test_full_like.py` -k sharded | 648 P / 0 F | 648 P / 0 F | 0 (regression check) |
| `test_pad.py` -k sharded | 78 P / 23 F | 78 P / 23 F | 0 (B8.1 PCC remains, no regression) |
| `test_permute.py` -k sharded | 0 P / 8 F (all JIT) | 0 P / 8 F (all PCC) | 0 — **but all JIT gates closed** |
| `test_interleaved_to_sharded.py` | 6 P / 84 F / 16 S | **22 P / 68 F / 16 S** | **+16 P** |

**Net new sharded variants passing: +16** (all in test_interleaved_to_sharded).

But the bigger structural win is: **all JIT-compile gates in the test_permute
and test_interleaved_to_sharded sharded paths are now closed.** Every
remaining failure is a PCC / numeric divergence — same tier as B8.4 — which
is now the dominant blocker across the harvest.

## What landed

### Step 1: `noc_async_read_one_packet_{set,with}_state` + write-side counterpart

`include/jit_hw/api/dataflow/dataflow_api.h`. Mirror upstream's stateful
single-packet API as no-op + thread-local state. Two pairs:

- `noc_async_read_one_packet_set_state(src_noc, size, vc, noc)` →
  store size in thread_local.
- `noc_async_read_one_packet_with_state(src_noc, dst_l1, vc, noc)` →
  `noc_async_read(src_noc, dst_l1, stored_size, noc)`.
- `noc_async_write_one_packet_set_state(dst_noc, size, noc, vc)` →
  store dst_noc + size in thread_local.
- `noc_async_write_one_packet_with_state(src_l1, dst_l1_unused, noc)` →
  `noc_async_write(src_l1, stored_dst_noc, stored_size, noc)`.

Closed: B8.3 reader-kernel JIT gate (`reader_unary_transpose_hc_sharded_rm.cpp`).
Discovered + closed in Step 3: B8.3 writer-kernel JIT gate
(`writer_unary_transpose_wh_sharded_rm.cpp`) needs the write-side
counterpart that was bundled into this commit.

### Step 2: `tensix_types.h` empty shim

`include/jit_hw/tensix_types.h` (new file). Empty header with rationale.

The failing kernel uses ZERO direct symbols from `tensix_types.h` —
it's a defensive include. Tried forwarding to upstream's
`wormhole_b0_defines/tensix_types.h` first, but that path collides with
emule's existing `enum class DataFormat` in `api/compute/common_globals.h`
(pulled in via `dataflow_api_addrgen.h`). Empty shim resolves the
include directive and side-steps the conflict.

Closed: B8.2 (`reader_unary_sharded_blocks_interleaved_start_id.cpp`
JIT gate). Caveat: a future kernel that actually consumes a
tensix_types symbol (xmov_direction_t, packer_config_t, ...) will need
that symbol thin-added here rather than re-attempting the forward.

### Step 3: TILE_WIDTH constants in `ckernel` scope

`include/jit_hw/api/compute/common.h`. Upstream provides
`TILE_WIDTH = 32`, `FACE_WIDTH = 16`, etc. via
`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h:89` —
inside the `ckernel` namespace, which compute kernels pull to global
scope via `using namespace ckernel;`. Emule had them in
`tt::constants` only, so bare-name references like
`last_output_row_num_datums < TILE_WIDTH` in `transpose_wh_rm.cpp:119`
failed to resolve.

Added all 7 (TILE_WIDTH, TILE_HEIGHT, TILE_HW, FACE_WIDTH,
FACE_HEIGHT, FACE_HW, TILE_C_DIM) inside the existing `ckernel`
namespace block. `using namespace ckernel;` at the file bottom brings
them to global scope.

Closed: B8.3 compute-kernel JIT gate (`transpose_wh_rm.cpp`).

## What did NOT close

- **B8.1** — test_pad sharded stick-layout (23 fails). Still PCC at
  the same code points; needs the `TT_EMULE_TRACE_NOC` instrumentation
  pass from the Round 8 register.
- **B8.3 PCC tier** — all 8 test_permute sharded variants still fail
  PCC after JIT gates closed. Failure mode shifted from "won't
  compile" to `AssertionError: Max ATOL Delta: 0.99609375`. Likely
  shares root cause with B8.4.
- **B8.4** — 196 test_untilize sharded ATOL≈3.25. Unchanged.
- **i2s residual 68 PCC** — surfaced in Step 2 sweep, all
  `AssertionError: 0.0`. About 12 are BFLOAT8_B paths (separate dtype
  bug); the rest likely overlap with B8.1 or B8.4. New Round 10 item.

## What surfaced as a new Round 10 item

- **B10.1** — i2s BFLOAT8_B path: 12 of the 24
  `test_interleaved_to_sharded_hash` failures are BFP8 input or output.
  Repro: `pytest test_interleaved_to_sharded.py::test_interleaved_to_sharded_hash -k 'BFLOAT8_B'`.
  PCC = 0.0 (complete data loss, not numeric error). Suspect: emule
  BFP8 packing/unpacking shim incomplete for the sharded write path.

## Files touched this round

- `include/jit_hw/api/dataflow/dataflow_api.h` — Step 1 (read+write
  one_packet state).
- `include/jit_hw/tensix_types.h` — Step 2 (new empty shim).
- `include/jit_hw/api/compute/common.h` — Step 3 (TILE_WIDTH+ in ckernel).
- `STRUCTURE.md` — symbol surface updates.
- `docs/notes/round8-sharded-harvest.md` — closure markers on B8.2 +
  B8.3 JIT halves.
- `docs/notes/round9-closeout.md` (this file).

## Round 10 agenda

Per priority + effort:

1. **B10.1 / i2s BFP8** — ~30 min to ~2 hr depending on whether the
   BFP8 sharded path is a clean shim or a deeper integration gap.
   12 tests gated.
2. **B8.1 self-loop NOC trace** — instrumentation recipe in the Round
   8 register; ~2-3 hr characterization + fix. 39 tests gated
   (23 pad + 16 i2s stick_layout).
3. **B8.4 untilize ATOL≈3.25** — focused characterization on
   `[2, 2, 256, 512]` test then fix. 196 tests gated.
4. **B8.3 PCC tier** — 6 permute PCC failures. Likely closed
   alongside B8.4. If not, separate trace pass.
