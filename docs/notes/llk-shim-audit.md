# LLK shim audit — tt-emule vs silicon

Initial audit: 2026-06-04 (four parallel agents covered pack /
unpack / tilize / untilize against the silicon LLK).

**Validation pass:** 2026-06-04 against `arminale/permute-fix` rebased
onto `arminale/comparison-ops` @ `6da2f25` (the latter introduced
per-CB DataFormat tracking via `cb_data_format()` and the
`unpack_src_format[32]` / `pack_dst_format[32]` JIT-fed arrays — see
[docs/cb-dataformat.md](../cb-dataformat.md)).

This doc is now a **living checklist**. Each item is tagged:

- **ACTIVE** — has a real caller in current tt-metal/ttnn (named).
  Likely to silently corrupt or fail JIT the first time the caller
  runs on emule.
- **LATENT** — confirmed divergence in code, no current caller.
- **FIXED** — closed by `comparison-ops` or earlier work; kept for
  reference.
- **DEFERRED** — real bug, deferred because no caller + no unit
  test. Track via issue.
- **DONE** — closed by a fix on this branch (will be marked as
  commits land).

Headline: **the bug class we just fixed in `tilize_block` (page-size
= whole tile, ignoring horizontal-strip layout) is repeated in every
other unpack/binary op**. The `comparison-ops` work resolved one
specific format-dispatch bug (bf16/UInt16 ambiguity) and added the
plumbing (`cb_data_format()`) that lets several other findings be
one-line fixes.

## Design baseline

Emule does **not** model SrcA/SrcB registers. Ops read CB → DST
directly. Any silicon LLK that primarily exists to program SrcA/B
descriptor regs, MOPs that drive the SrcA/B unpackers, or SrcA/B-side
flags (UInt16 zero-flag, ALU_ACC_CTRL Zero_Flag_disabled_src, etc.) is
correctly a no-op in emule — see "By-design items" at the bottom.

After `comparison-ops`, emule tracks per-CB DataFormat at JIT
compile time via `EMULE_CB_DATA_FORMATS` → `unpack_src_format[32]` /
`pack_dst_format[32]` in `include/jit_hw/api/cb_api.h`. The new
`cb_data_format(cb_id)` helper in `common.h` returns the exact format
enum. **However, only the `cb_is_uint16_format` path uses it today.**
Most format-dispatch consumers (`cb_is_32bit_format`,
`cb_is_bfp8_b_format`) still use the page_size heuristic, by design —
those buckets are unambiguous from page_size alone (per
`cb-dataformat.md`).

Per-CB face / partial-face information is **not** propagated:
`unpack_tile_r_dim`, `unpack_num_faces_r_dim`, etc. in cb_api.h are
hard-coded to standard 32×32 / 2×2-face values for all 32 CB slots.
Partial-face audit findings stand.

## Tier 1 — active silent-corruption hazards

### #1 — page-size = whole tile assumption in unpack helpers
**LATENT.** All confirmed in current code:
- `add_tiles` at `common.h:325-341` (loop bound `cb_page_size(icb0)/sizeof(float)`)
- `sub_tiles` at `common.h:344-365`
- `mul_tiles` at `common.h:368-388`
- `__emule_unpack_cb_tile_to` at `common.h:429-460`
- `matmul_tiles` at `matmul.h:34-65`

Same bug class we just fixed in `tilize_block`: assumes one cb page =
one tile, can't handle horizontal-strip layouts. No current ttnn test
triggers (no sharded op produces horizontal-strip input to these).
Defer to **STRUCTURAL** section.

### #2 — `__llk_pack_untilize` elem_size = page_size / 1024 — **DONE (format dispatch)**
Was ACTIVE for uint16 outputs (PR #84 regression on
`bf_test_to_layout`, `bf_test_tilize_untilize_2D`, ATOL=99): the
elem_size heuristic couldn't distinguish uint16 from bf16 (both
2 bytes/elem), and the function then routed uint16 DST through
`bf16::from_f32` → denormal-to-zero corruption.

Fixed: `__llk_pack_untilize` now mirrors `pack_dst_to_buf`'s
format-aware dispatch via `cb_is_uint16_format`,
`cb_is_32bit_format`, fallback bf16.  uint16 path extracts the low
16 bits of the DST int32 bit pattern.

**Still LATENT (deferred):** Bfp8_b output for untilize-pack (ps=1088
would have hit elem_size=1).  Silicon's `untilize_helpers.inl`
asserts `!is_block_float_format(pack_dst_format)` so no real caller
produces Bfp8_b output via this path.  Narrow-row (ps=512 → elem_size=0)
also unreachable via current callers since `pack_untilize_dest`'s
narrow-row branch handles that case directly.

### #3 — mixed-format binary ops
**LATENT** (no existing ttnn test exercises mixed formats). At
`common.h:325, 349, 373` — `add/sub/mul_tiles` branch on
`cb_is_32bit_format(icb0)` only; `icb1` walked with same predicate. If
`icb0` bf16 and `icb1` fp32, the fp32 buffer is cast to `uint16_t*`
and the upper half of the tile is silently dropped. With
`cb_data_format(icb1)` now available, fix is mechanical → **Batch 2**.

### #4 — `copy_tile` overwrites DST
**LATENT.** At `common.h:440-444`. Always overwrites DST; ignores
silicon's `acc_to_dest=true` (additive-load) semantic. Matmul
partial-reload path uses `copy_block_matmul_partials` (`common.h:447-455`)
which loops `copy_tile` — same risk. No current ttnn test exercises
acc_to_dest reload. Defer to **DEFERRED**.

### #5 — `pack_untilize_dest` ignores face_r_dim / num_faces
**LATENT.** At `pack_untilize.h:91-93` — runtime args are
commented-out (`uint32_t /*face_r_dim*/ = 16, uint32_t /*num_faces*/ = 4`).
Per-CB face arrays at `cb_api.h:60-74` are hard-coded to 32/2. No
current caller passes non-standard face dims. Defer to **STRUCTURAL**.

### #6 — `pack_tile_block` slot indexing
**LATENT.** At `common.h:416-422` (audit's original 394-400 was
stale). Writes slots 0..ntiles-1, ignoring `__emule_pack_offset[ocb]`.
No current overlap pattern in ttnn (`pack_tile` followed by
`pack_tile_block` on same CB). Defer to **DEFERRED**.

### #7 — missing `untilize_uninit` shim — **DONE**
Was ACTIVE: `untilize_uninit(uint32_t)` shim was missing entirely;
ssm_prefix_scan called it and got an "undeclared identifier" JIT
compile error.

Fixed in Batch 1: `untilize.h` now defines `untilize_uninit(uint32_t = 0)`
that clears `__llk_pack_is_untilize`. Verified ssm_prefix_scan now
JIT-compiles (PCC remains a separate gap — see Tier 1 #2 STRUCTURAL).

The original audit also flagged "tilize_uninit doesn't symmetrically
clear `__llk_pack_is_untilize`" as a concern. **Investigation
disproved that:** clearing the flag in tilize_uninit regresses
`bf_test_tilize_untilize_2D` and `bf_test_to_layout` (ATOL=99). Silicon's
tilize_uninit reverts the unpacker config but does NOT touch packer
state, so the asymmetry is intentional. Callers that need to clear
`__llk_pack_is_untilize` use the explicit `untilize_uninit` shim.

### #8 — templated `untilize_block<block_tile_count>` — **DONE**
Was ACTIVE: `untilize.h:42-51` treated the template param as both DST
batch size AND total tile count + row stride; runtime `ntiles` was
dropped. Caller: conv3d compute kernel.

Fixed in Batch 1: the templated overload now forwards to the runtime
overload with `ntiles = runtime arg`. The template param is preserved
at the signature level for source-compatibility but no longer drives
the loop.

### #9 — `transpose_wh_tile` partial-face residue
**LATENT.** At `transpose_wh.h:18-28`. Flat 32×32 DST permute includes
the 768 unused floats (residue from prior op) for partial-face inputs.
No current partial-face caller. Defer to **STRUCTURAL**.

### #10 — `matmul.h` transpose arg ignored
**LATENT.** At `matmul.h:22-89`. `mm_init` / `matmul_block` accept
`transpose` param/template but ignore it. No current `transpose=1`
caller in ttnn. Defer to **DEFERRED**.

## Tier 2 — PACK-side risky stubs

### `pack_relu_config` / `pack_set_relu_threshold` no-op
**ACTIVE.** At `common.h:516-517`. Both empty inline functions.
30+ silicon callers fuse ReLU into pack via `llk_pack_relu_config`.
Canonical: `bmm_large_block_zm_fused_bias_activation.cpp`.

Test trigger: `test_matmul_with_fused_activations` with `[relu]`
activation. → **Batch 3**.

### `llk_pack_reconfig_l1_acc` thread-global
**ACTIVE.** At `common.h:579-581`. Single `__emule_l1_acc_enabled`
bool (at `common.h:111`); not per-CB. Cross-CB pollution risk if one
matmul output CB enables L1 acc while another doesn't. → **Batch 3**.

## Tier 3 — public-API symbol gaps

| Symbol | Emule shim | Active caller? | Batch |
|---|---|---|---|
| `untilize_uninit(uint32_t)` | **MISSING** | ssm_prefix_scan + 3 others | 1 |
| `tilizeA_B_reduce_init`, `unpack_tilizeA_B_block`, `unpack_tilizeA_B_uninit` | **MISSING** | compute_kernel_sentinel, unpack_tilizeA_B | 1 |
| `tilize_init_short_with_dt`, `tilize_uninit_with_dt`, `fast_tilize_init_with_dt[_skip_remap]` | **MISSING** | silicon defs exist; ttnn callers pending verify | 1 |
| `pack_init`, `pack_dest_init`, public `pack_reconfig_l1_acc`, public `pack_relu_config` | **MISSING** | D2M-only today | 1 (cheap to add) |
| `pack_rows*`, `pack_block_contiguous*` | MISSING | D2M-only | DEFERRED |
| `pack_untilize_init_skip_remap` | **PRESENT** (`pack_untilize.h:36`) | — | n/a |
| `fast_tilize_init_skip_remap` | **PRESENT** (`tilize.h:86`) | — | n/a |
| `_llk_unpack_bcastA_B_` | MISSING | SDPA (SrcA/B-related) | by-design |

## Tier 4 — data-format coverage gaps

| Format | Status | Active caller | Batch |
|---|---|---|---|
| bf16 (Float16_b) | OK | many | — |
| fp32 (Float32) | OK | many | — |
| int32 / uint32 | OK | many | — |
| **UInt16** | **FIXED by comparison-ops** | comparison ops | — |
| Bfp8_b (full tile) | OK | many | — |
| **Bfp4_b** | **MISSING decoder** — falls through to Bfp8_b → garbage | moe_compute (`moe_compute_program_factory.cpp:866`), moe_gpt (`moe_gpt_program_factory.cpp:86`) | 2 |
| Bfp8_b (partial-face) | broken (full-tile hard-coded) | none today | STRUCTURAL |
| IEEE Float16 (non-`_b`) | wrong — read as bf16 | none today | DEFERRED |
| Int8 / UInt8 | none | none today | DEFERRED |
| Tf32 | by-design (SrcA truncation only) | — | — |

## Tier 5 — reduce hazards

- **`reduce` MAX on tiny tiles** at `reduce.h:63-178`: silicon pads
  SrcA with `NEGINFSRC` past `face_r_dim*16`. Emule reads from CB
  directly; result depends on whether the CB contains garbage past
  the valid region. **LATENT** — no current `reduce<PoolType::MAX>`
  caller with tiny tiles (compute_pool_2d uses separate logic).
  Defer to **DEFERRED**.

## Recommendation — execution checklist

### Batch 1 — missing shims (LOW risk)  ✓ DONE
- [x] Add `untilize_uninit(uint32_t = 0)` clearing `__llk_pack_is_untilize`
  (closes Tier 1 #7)
- [~] `tilize_uninit` symmetric clear of `__llk_pack_is_untilize` —
  REJECTED. Investigation found this regresses
  `bf_test_tilize_untilize_2D` and `bf_test_to_layout` (ATOL=99); silicon's
  tilize_uninit only reverts unpacker state.
- [x] Fix templated `untilize_block<>` to forward to runtime overload
  with `ntiles = block_tile_count` (closes Tier 1 #8)
- [x] Add no-op forwarders: `tilize_init_short_with_dt`,
  `tilize_uninit_with_dt`, `fast_tilize_init_with_dt[_skip_remap]`
- [x] Add public `pack_init`, `pack_dest_init`,
  `pack_reconfig_l1_acc(uint32_t)`, `pack_relu_config(ReluType)`
- [ ] `tilizeA_B_reduce_init` + `unpack_tilizeA_B_block` +
  `unpack_tilizeA_B_uninit` — DEFERRED (callers are tt_metal C++ test
  kernels, not in ttnn pytest CI; add when needed)
- [ ] CI additions for ssm_prefix_scan, rotary_embedding_llama,
  group_attn_matmul, conv3d — DEFERRED. Each needs additional unrelated
  shims beyond Batch 1's scope. See
  [batch1-baseline.md](batch1-baseline.md).

### Batch 2 — format dispatch (LOW–MED risk)  ✓ DONE (PR #84 layout fixes)
- [x] Make `__llk_pack_untilize` format-aware via `cb_is_uint16_format`
  / `cb_is_32bit_format` / bf16 fallback. Closes Tier 1 #2 for current
  callers (uint16); resolves PR #84 layout regressions on WH
- [x] Bump `__EMULE_DST_TILES` to 32 (BH FULL_DEST capacity). WH paths
  never index > 16 so unchanged there
- [x] Make `fast_untilize_block` bypass `pack_untilize_block` (the
  per-DST-slot loop). Forward to `untilize_block` (load-one-pack-one,
  1 DST slot). Resolves PR #84 layout regressions on BH for wide
  tensors (W=131072 case)
- [ ] `add/sub/mul_tiles` consult `cb_data_format(icb1)` per-operand
  (closes Tier 1 #3) — deferred, no current ttnn caller
- [ ] Add Bfp4_b decoder (new `bfp4.h`) + branch in
  `__emule_unpack_cb_tile_to` / `pack_dst_to_buf` (closes Tier 4
  Bfp4_b) — deferred, moe_compute / moe_gpt would exercise this
- [ ] Tighten `cb_is_bfp8_b_format` to consult `cb_data_format()`
  when set — deferred
- [ ] Author `tests/jit_hw/test_mixed_format_binary.cpp` — deferred

### Batch 3 — PACK state corrections (LOW–MED risk)
- [ ] Per-CB `__emule_l1_acc_enabled[32]` array in `common_globals.h`
- [ ] Per-CB `__emule_pack_relu_mode[32]` + `__emule_pack_relu_threshold[32]`
- [ ] `pack_dst_to_buf` applies the per-CB ReLU clamp + per-CB L1 acc
- [ ] Baseline + add CI: matmul_test_fused_activations (`[relu]`
  parametrize)

## DEFERRED — file as issue, no caller today

- Tier 1 #4 — `copy_tile` doesn't honor `DST_ACCUM_MODE` /
  `acc_to_dest`
- Tier 1 #6 — `pack_tile_block` ignores `__emule_pack_offset[ocb]`
- Tier 1 #10 — matmul `transpose=1` ignored
- Tier 4 IEEE Float16 (non-`_b`), Int8, UInt8 — no caller
- Tier 5 reduce MAX on tiny tiles — needs unit test first
- Tier 3 `_llk_unpack_bcastA_B_`, `pack_rows*`, `pack_block_contiguous*`

## STRUCTURAL — separate effort

These need either a new helper or new JIT-define plumbing; not
addressable in a one-line patch:

- **Tier 1 #1** strip-layout helper. Touch `__emule_unpack_cb_tile_to`,
  `matmul_tiles`, `pack_dst_to_buf`. Sketch: helper
  `unpack_cb_tile_to_strip(icb, itile, ntiles_in_strip, dst_ptr)`
  using `unpack_tile_size[icb]` instead of `cb_page_size`. New unit
  test required since no current ttnn caller triggers.
- **Tier 1 #2** `__llk_pack_untilize` numeric. Replace
  `elem_size = ps / 1024` with `cb_data_format()` dispatch + Bfp4/Bfp8
  encoder paths.
- **Tier 1 #5** + **#9** + partial-face Bfp8_b — per-CB face plumbing.
  New `EMULE_CB_NUM_FACES` / `EMULE_CB_FACE_DIMS` JIT defines,
  populated by host-side JIT-define generator in tt-metal (requires
  tt-metal pin bump). Per-CB arrays in `cb_api.h` replacing the
  hard-coded constants.

## By-design items (NOT gaps, no SrcA/B in emule)

For reference — these would be gaps if emule modelled SrcA/SrcB but
are correctly no-ops given the design choice:

- `reconfig_data_format` family (`common.h:467-481`) — SrcA/B
  descriptor reprogramming
- `unary_op_init_common`, `binary_op_init_common`,
  `transpose_wh_init`, `mm_init`, `reduce_init` — program MOPs that
  drive SrcA/B unpackers
- `llk_unpack_A` template (`llk_unpack_a.h:36-40`) — SrcA/B-side
  broadcast modes and `acc_to_dest`
- `transpose_wh_tile` IntraFace vs InterFace distinction
- Int8 / UInt8 `ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW` zero-flag
- Tf32 SrcA truncation (emule keeps full fp32 in DST)
- `reduce_init` `enforce_fp32_accumulation` —
  `ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW`
- `copy_tile_to_dst_init_short_with_dt` — SrcA reconfig
- `_llk_unpack_bcastA_B_` (SDPA sub-bcast-row)
