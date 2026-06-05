# Pack, unpack, tilize, and untilize in tt-emule

This is the standalone reference for the four foundational compute-LLK
operations as emule mocks them. The goal is silicon-faithful behavior
without modelling SrcA/SrcB registers or the math/pack MOP machinery.

Companion doc: [cb-dataformat.md](cb-dataformat.md) for how per-CB
`tt::DataFormat` flows through emule.

## Pipeline at a glance

Silicon's compute pipeline is **UNPACK → MATH → PACK**, mediated by
SrcA/SrcB and the DST register file. Each stage is a separate RISC-V
thread driving its own engine. emule collapses MATH (the SrcA op SrcB →
DST step) and the SrcA/B-side of UNPACK into direct fp32 operations on
the DST register file:

| Stage           | Silicon                                   | emule                                                 |
| --------------- | ----------------------------------------- | ----------------------------------------------------- |
| Unpack          | CB → SrcA / SrcB (format conversion HW)   | CB → DST (format-aware decode into fp32 row-major)    |
| Math            | SrcA op SrcB → DST                        | DST op CB-scratch → DST (no SrcA/B intermediate)      |
| Pack            | DST → CB (format conversion + STACC_RELU) | DST → CB (format-aware encode + thread-global ReLU)   |

Because emule's DST stores fp32 row-major (1024 floats per tile), the
unpack/pack stages also do nfaces ↔ row-major conversion. The LUTs are
`__emule_nfaces::rowmajor_to_nfaces[]` / `nfaces_to_rowmajor[]` in
[`include/jit_hw/api/compute/nfaces.h`](../include/jit_hw/api/compute/nfaces.h).

## DST register file

```cpp
// include/jit_hw/api/compute/common.h
static constexpr uint32_t __EMULE_DST_TILES      = 16;  // bf16 SyncFull
static constexpr uint32_t __EMULE_DST_TILES_FP32 = 8;   // f32 SyncFull
static constexpr uint32_t __EMULE_TILE_ELEMS     = 1024;
static thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS];
```

Silicon's Dst is a 64 KB register file with two banks of 32 KB. The
addressable capacity (per DeepWiki tt-isa-documentation §4.7 and
`tt_llk/common/tensor_shape.h:29`) is **identical on WH-B0 and Blackhole**:

| Mode                 | bf16 / fp16 | fp32 accum |
| -------------------- | ----------- | ---------- |
| `SyncFull` (single bank, no MATH↔PACK pipelining)  | 16 tiles | 8 tiles |
| `SyncHalf` (default; double-buffered)              | 8 tiles  | 4 tiles |

`SyncHalf` is the production default — math writes one half while pack
reads the other. emule allows the `SyncFull` ceiling as the upper bound
so neither mode triggers a spurious bounds error. There is no
"FULL_DEST = 32 tiles" hardware mode on either arch.

DST slot bounds are enforced by `__emule_dst_check(slot, caller)` in
common.h. Writes go through `__emule_dst_mark_dirty(slot)`.

## CB metadata, page size, and data format

The unpack/pack paths dispatch on three things looked up from
`__emule_cbs[cb_id]`:

- `cb_page_size(cb_id)` — bytes per CB page (= one tile's encoded size
  for tile-shaped CBs).
- `cb_data_format(cb_id)` — the real `tt::DataFormat` enum value,
  threaded from the host via the `EMULE_CB_DATA_FORMATS` JIT define
  ([cb-dataformat.md](cb-dataformat.md)). `255` = Invalid (CB
  slot unconfigured); paths fall back to the page-size heuristic.
- `cb_is_*_format(cb_id)` predicates: `_32bit_format` and
  `_bfp8_b_format` use page-size heuristics (unambiguous buckets);
  `_uint16_format` uses the real DataFormat (only way to distinguish
  UInt16 from bf16 — both are 2048 B / tile).

## Unpack

Single entry point: `__emule_unpack_cb_tile_to(icb, itile, float* out)`
in [common.h](../include/jit_hw/api/compute/common.h). Reads one tile
from `CB[icb]` at index `itile`, applies the nfaces→row-major
permutation, decodes to fp32, writes into `out` (a DST slot or
`__emule_src_scratch`).

Format dispatch (in order):

1. **Bfp8_b** (`page_size == 1088`) — 64 face-row 8-bit exponents + 1024
   mantissa bytes. Decoded via `__emule_bfp8::to_f32` per element.
2. **32-bit** (`page_size > 2048`) — Float32 / Int32 / UInt32. Per-element
   `memcpy` preserves INT32 bit patterns that would otherwise be flushed
   to zero by x86 DAZ/FTZ on float assignment.
3. **uint16** (`cb_data_format(icb) == UInt16`) — widens each uint16 to
   int32 (zero-extended) and stores the bit pattern in the fp32 DST
   slot. SFPU comparison ops read this back via `__emule_dst_load_i32`.
4. **bf16** (default 2048-byte 16-bit) — `__emule_bf16::to_f32` per
   element.

`copy_tile(icb, itile, idst)` is the public entry; it calls
`__emule_dst_mark_dirty(idst)` then dispatches to the helper above.

### Tilize as a special unpack pattern

[`tilize.h`](../include/jit_hw/api/compute/tilize.h) calls
`tilize_block(icb, ntiles, ocb)`. The input CB stores `ntiles` tiles in a
**horizontal strip**: 32 rows × `ntiles * TILE_WIDTH` cols, row stride =
`ntiles * TILE_WIDTH * elem_size` bytes — NOT `ntiles` sequential
2048-byte tile blocks. (Silicon's reader kernels write strips because
the input is typically row-major data being tile-formatted on the fly.)

For each tile `t`, the body reads a 32×32 region from the strip at
column offset `t * TILE_WIDTH * elem_size` and writes it in nfaces order
to `ocb` via `cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++)`. Format
combinations (icb_format × ocb_format) covered: bf16→bf16, fp32→fp32,
bf16→fp32, fp32→bf16, 32-bit→32-bit.

`tilize_init` / `tilize_uninit` flip the thread-local
`__llk_unpack_is_tilize` flag (consumed by some downstream LLK paths
via `llk_pack.h::llk_pack`).

`fast_tilize_*` is a delegation alias on WH (and on BH where the HW
fast-tilize path is conceptually equivalent to the regular path for
emule, since the difference on silicon is just packer scheduling).

## Pack

The pack side has more variants than unpack because silicon distinguishes
*tilize-pack* (DST → nfaces CB) from *untilize-pack* (DST → row-major
CB) and offers a `pack_untilize_dest` API that does the row-major
scatter directly from the DST register file.

### `pack_dst_to_buf(buf, dst_slot, ocb)` — the canonical encoder

Lives in [common.h](../include/jit_hw/api/compute/common.h). Reads
`__emule_dst[dst_slot]` (fp32 row-major), applies the thread-global
ReLU clamp (see below), and writes to `buf` in nfaces order. Format
dispatch mirrors the unpack side but encodes:

1. **Bfp8_b** — encode 64 face-rows via
   `__emule_bfp8::encode_face_row(row16, exp, mant_row)`. L1 acc not
   supported for Bfp8_b output (would require decoding + re-encoding;
   deferred until a caller exposes it).
2. **32-bit** — fast path `memcpy` (no ReLU) preserves INT32 bit
   patterns; ReLU path reinterprets as float, clamps, writes.
3. **uint16** — write the low 16 bits of the DST int32 bit pattern.
   ReLU skipped on integer output (not a coherent silicon config).
4. **bf16** — `__emule_bf16::from_f32(__emule_apply_pack_relu(dst[i]))`,
   per element. L1 acc adds to the existing CB contents (after
   bf16→fp32 → add → fp32→bf16).

### `pack_tile(idst, ocb)`

Public entry. Computes the write pointer via
`cb_write_ptr_at(ocb, __emule_pack_offset[ocb]++)` and calls
`pack_dst_to_buf`. `pack_tile_block(idst, ocb, ntiles)` packs `ntiles`
DST slots sequentially.

### `__llk_pack_tiled(tile_idx, ocb)` and `__llk_pack_untilize(tile_idx, ocb)`

Lower-level pack paths in
[`include/jit_hw/llk_pack.h`](../include/jit_hw/llk_pack.h). The
`llk_pack<...>` templates dispatch via `__llk_pack_is_untilize`
(set by `untilize_init`, cleared by `tilize_init`/`untilize_uninit`).

- `__llk_pack_tiled` is `pack_dst_to_buf` factored to write at
  `cb_write_ptr_at(ocb, __llk_pack_offset)` (a separate offset counter
  used by tilize-pack callers). Same format dispatch + ReLU.
- `__llk_pack_untilize` is the row-major scatter: DST tile → cb_out at
  position `(tile_row, tile_col) = (__llk_pack_offset / block_c,
  __llk_pack_offset % block_c)` within a block of width
  `__llk_pack_block_c`. Same uint16/32-bit/bf16 format dispatch as
  `pack_dst_to_buf`. Bfp8_b output is silicon-asserted unsupported here
  (`untilize_helpers.inl:155`).

### `pack_untilize_dest<block_ct_dim, full_ct_dim, ..., narrow_row, row_num_datums>(ocb, block_rt_dim, block_c_index, ...)`

High-level entry in
[`pack_untilize.h`](../include/jit_hw/api/compute/pack_untilize.h).
Scatters `(block_rt_dim × block_ct_dim)` DST tiles to `ocb` in
row-major order. Output region is `(block_rt_dim * TILE_DIM)` rows ×
`(full_ct_dim * row_cols)` cols where `row_cols = TILE_DIM` normally or
`row_num_datums` when `narrow_row` is set. Writes the
`block_ct_dim`-wide column-block starting at column index
`block_c_index * block_ct_dim`. Format-aware: bf16 / fp32 / uint16
(matching the unpack format). cb page_size is sync granularity; the L1
byte layout is always row-major.

Used by transpose_wh's sharded compute kernel (with `narrow_row=true,
row_num_datums=8` for the H % 32 != 0 case).

### Untilize as a special pack pattern

[`untilize.h`](../include/jit_hw/api/compute/untilize.h):

- `untilize_init(icb, ocb)` sets `__llk_pack_is_untilize = true` so
  subsequent `pack_tile` calls route through `__llk_pack_untilize`.
- `untilize_uninit(ocb)` clears the flag.
- `untilize_block(icb, ntiles, ocb)` is the load-one-pack-one loop:
  for each of `ntiles` tiles, `copy_tile(icb, t, 0)` →
  `__llk_pack_untilize(0, ocb)`. Uses only DST slot 0 regardless of
  `ntiles` (matches silicon's SyncHalf chunked behavior). The templated
  overload `untilize_block<block_tile_count>(icb, ntiles, ocb)` is a
  forwarder; silicon's template arg is the DST sub-batch size, which
  emule doesn't model.

`fast_untilize_block<full_ct_dim>(icb, ocb)` forwards to
`untilize_block(icb, full_ct_dim, ocb)`. Silicon's BH fast-untilize is
HW-accelerated and chunks via `fast_untilize_next_unit_dim` so DEST is
never expected to hold the full block at once. Forwarding to
`untilize_block` (1 DST slot per iteration) captures that property and
handles arbitrary `full_ct_dim` — including wide tensors (W = 131072
→ `full_ct_dim = 4096`) that the pack-untilize-block path can't.

### Pack-fused ReLU clamp

Silicon's STACC_RELU is a single packer CFG reg that clamps PACK output
per-element. emule mirrors with thread-global state in
[common.h](../include/jit_hw/api/compute/common.h):

```cpp
static thread_local ReluType __emule_pack_relu_mode = ReluType::NO_RELU;
static thread_local float    __emule_pack_relu_threshold = 0.0f;
```

Writers: `llk_pack_relu_config(ReluType)`,
`pack_set_relu_threshold(float)`, `pack_relu_config(uint32_t config)`
(decodes silicon's packed format: low 4 bits = mode,
upper 16 = threshold-as-uint16-cast-to-float).

Modes (from `enum ReluType` in
[common.h](../include/jit_hw/api/compute/common.h)):

- `NO_RELU` — passthrough
- `ZERO_RELU` — `max(0, x)`
- `MIN_THRESHOLD_RELU` — `max(threshold, x)`
- `MAX_THRESHOLD_RELU` — `min(threshold, x)`

Applied at every pack site that respects STACC_RELU on silicon:
`pack_dst_to_buf`, `__llk_pack_tiled`, `__llk_pack_untilize`. Skipped
on uint16 and Bfp8_b outputs (not coherent configs).

### L1 accumulation

`llk_pack_reconfig_l1_acc(enable)` sets the thread-global
`__emule_l1_acc_enabled` flag. When true, `pack_dst_to_buf`'s bf16 and
32-bit branches add DST to the existing CB contents instead of
overwriting. Mirrors silicon's PACK L1 acc CFG — also a single
thread-global packer register.

## Pack state globals

All thread-local in
[common.h](../include/jit_hw/api/compute/common.h) or
[llk_state.h](../include/jit_hw/internal/llk_state.h):

| Name                          | Owner                  | What it controls                                          |
| ----------------------------- | ---------------------- | --------------------------------------------------------- |
| `__emule_pack_offset[32]`     | per-CB                 | `pack_tile` / `pack_tile_block` next-tile index in CB     |
| `__llk_pack_offset`           | thread-global          | LLK pack-tile-offset counter (separate from above)        |
| `__llk_pack_block_c`          | thread-global          | Block column width for `__llk_pack_untilize` scatter      |
| `__llk_pack_is_untilize`      | thread-global          | Routes `pack_tile` to `__llk_pack_untilize`               |
| `__llk_unpack_is_tilize`      | thread-global          | Marker for tilize-mode unpack (used by some downstream)   |
| `__emule_l1_acc_enabled`      | thread-global          | Accumulate vs overwrite at pack time                      |
| `__emule_pack_relu_mode`      | thread-global          | STACC_RELU mode for all pack sites                        |
| `__emule_pack_relu_threshold` | thread-global          | STACC_RELU threshold value                                |

`cb_reserve_back(cb, n)` resets `__emule_pack_offset[cb] = 0` so each
new block of pushed pages starts fresh.

## What's not modelled (and why)

By design, the following silicon details are intentionally absent in
emule:

- **SrcA / SrcB register files** — emule fuses UNPACK and MATH into
  direct DST writes. Any silicon LLK that programs SrcA/B descriptor
  regs, MOPs that drive the SrcA/B unpackers, or SrcA/B-side flags
  (UInt16 zero-flag, ALU_ACC_CTRL Zero_Flag_disabled_src, ...) is a
  correctly-empty stub.
- **MOPs and address modes** — silicon programs PACR address modes per
  call (`narrow_row`, `dense`, `diagonal`, stride). emule replicates
  the resulting L1 layout via direct addressing in the pack helpers.
- **Per-CB face dimensions** — `unpack_tile_r_dim`, `unpack_num_faces_r_dim`,
  etc. in [cb_api.h](../include/jit_hw/api/cb_api.h) are hard-coded to
  the standard 32×32 / 2×2-face layout. Partial-face tiles (face_r_dim
  < 16, num_faces ∈ {1, 2}) are not currently supported — flagged in
  the [LLK shim audit](#audit) as STRUCTURAL.
- **Pre-emption / pipelining** — silicon's MATH↔PACK barriers in
  SyncHalf and the kickoff/done semaphore ops are no-ops here. emule
  runs entirely synchronously within a thread.

## Common bug classes

Three patterns recur across the audit history; check them whenever
adding or modifying a shim:

1. **"page_size = whole tile" assumption.** Anywhere code computes
   `n = cb_page_size(icb) / element_size` to size a per-tile loop, it
   assumes one CB page equals one tile. That breaks when the input CB
   stores a horizontal strip of N tiles (e.g. sharded-reader output).
   The fix pattern is in `tilize_block`: walk `ntiles` tiles via a per-
   tile column offset and let the row stride be
   `ntiles * TILE_WIDTH * elem_size`.

2. **`elem_size = page_size / 1024` heuristic.** `__llk_pack_untilize`
   historically used this to infer the output element size; it can't
   distinguish bf16 from uint16 (both 2048-byte tiles), produces
   `elem_size = 1` for Bfp8_b (1088-byte tiles), and produces
   `elem_size = 0` for narrow-row tiles (page_size < 1024 → all rows
   alias one byte). Replaced with explicit `cb_is_uint16_format` /
   `cb_is_32bit_format` / bf16-fallback dispatch matching
   `pack_dst_to_buf`.

3. **Cross-stage state leakage.** `__llk_pack_is_untilize` and
   `__emule_l1_acc_enabled` are thread-global, so a kernel that does
   `untilize → tilize` without an explicit `untilize_uninit` between
   them leaves the packer in untilize mode. The corollary: do NOT
   "symmetrically" clear unrelated state in init/uninit — silicon's
   `tilize_uninit` only reverts unpacker state and emule must match
   (an earlier attempt to also clear `__llk_pack_is_untilize` here
   regressed `bf_test_tilize_untilize_2D`).

## File map

Critical shim files for pack/unpack/tilize/untilize:

- `include/jit_hw/api/compute/common.h` —
  `pack_dst_to_buf`, `__emule_unpack_cb_tile_to`, `copy_tile`,
  `pack_tile`, `pack_tile_block`, ReLU + L1-acc state, format predicates.
- `include/jit_hw/llk_pack.h` —
  `__llk_pack_tiled`, `__llk_pack_untilize`, `llk_pack<...>` template.
- `include/jit_hw/api/compute/tilize.h` —
  `tilize_init/_short/_uninit`, `tilize_block`, `fast_tilize_*`,
  `_with_dt` variants.
- `include/jit_hw/api/compute/untilize.h` —
  `untilize_init/_short/_uninit`, `untilize_block`,
  `untilize_block<block_tile_count>`.
- `include/jit_hw/api/compute/pack_untilize.h` —
  `pack_untilize_dest`, `pack_untilize_block`,
  `experimental::pack_untilize_block`, init/uninit no-ops.
- `include/jit_hw/api/compute/experimental/fast_untilize.h` —
  BH fast-untilize delegation.
- `include/jit_hw/api/compute/nfaces.h` —
  `rowmajor_to_nfaces` / `nfaces_to_rowmajor` permutation LUTs.
- `include/jit_hw/api/compute/bfp8.h` —
  Bfp8_b encode/decode.
- `include/jit_hw/api/cb_api.h` —
  CB metadata, `unpack_src_format[]` / `pack_dst_format[]` arrays,
  hard-coded face-dim constants.

## Audit

The full list of known shim divergences from silicon — what's a real
gap vs what's deliberately no-op — is tracked in the audit notes
(local, not committed). The categories used:

- **ACTIVE** — has a current ttnn caller; will fail or silently corrupt
- **LATENT** — confirmed divergence in code, no current caller
- **FIXED** — closed by a landed commit
- **DEFERRED** — real bug, no caller + no unit test, file as issue
- **STRUCTURAL** — needs new plumbing (e.g. per-CB face arrays via a new
  JIT define + tt-metal pin bump)
- **BY-DESIGN** — correctly no-op given emule's no-SrcA/B design
