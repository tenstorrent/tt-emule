# Plan: validate & start addressing the LLK shim audit (Batches 1-3)

## Context

`docs/notes/llk-shim-audit.md` lists ~25 divergences from silicon LLK
semantics across pack/unpack/tilize/untilize shims. Validated against
current code on `arminale/permute-fix` rebased onto
`arminale/comparison-ops` (which added per-CB DataFormat tracking via
`cb_data_format()`).

**Highest-impact gaps confirmed by validation:**

- 4 ttnn ops (`ssm_prefix_scan`, matmul fused-bias-activation,
  `rotary_embedding_llama`, `transformer_group_attn_matmul`, `conv3d`)
  exercise shims that don't exist or are broken. None are in tt-emule's
  current pytest regression today — first time we run them on emule,
  they will JIT-fail or silently produce wrong output.
- `moe_compute` / `moe_gpt` use `DataFormat::Bfp4_b` — emule has no
  Bfp4_b decoder, falls into the Bfp8_b branch → garbage weights.
- 30+ silicon kernels fuse ReLU into pack via `llk_pack_relu_config` —
  emule no-op silently drops the clamp.
- `llk_pack_reconfig_l1_acc` flag is thread-global; cross-CB pollution
  risk on simultaneous-acc-and-non-acc matmuls.
- Mixed-format binary (`add/sub/mul_tiles` with bf16+fp32 operands):
  no existing ttnn test, but `cb_is_32bit_format(icb0)` is mechanically
  wrong. Needs a tt-emule unit test.

**Intended outcome:** Tier 2 + Tier 3-critical + Tier 4-active gaps
closed in mechanical batches without touching SrcA/B modelling. The
audit doc becomes a living checklist — each fix updates the doc to
mark items DONE, leaving the structural items (page-size-strip helper,
pack-untilize numeric fixes, per-CB face plumbing) cleanly identified
for a follow-up plan.

## Tests to baseline pre-fix and add to CI

For each test below: (1) run on emule **before** the batch fix to
capture the pre-fix state, (2) run **after** to confirm the fix works,
(3) add the entry to BOTH `scripts/run_ttnn_pytests_wormhole.sh` and
`run_ttnn_pytests_blackhole.sh` so future regressions catch it.

Pre-fix predictions are from code inspection; the actual baseline run
in each batch's first step is authoritative.

| Op / dtype | Test file | Test function(s) | Pre-fix state | Targets batch |
|---|---|---|---|---|
| ssm_prefix_scan | `tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_prefix_scan.py` | `test_ssm_prefix_scan`, `test_chunked_ssm_prefix_scan`, `test_ssm_prefix_scan_with_program_cache` | JIT-fail (missing `untilize_uninit`) | 1 |
| rotary embedding | `tests/ttnn/nightly/unit_tests/operations/experimental/test_rotary_embedding_llama.py` | (whole file) | JIT-fail (`untilize_uninit`) | 1 |
| group attention matmul | `tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py` | `test_group_attn_matmul`, `test_group_attn_matmul_fp32`, `test_group_attn_matmul_with_program_cache_exhaustive` | silent-wrong (tilize/untilize state asymmetry) | 1 |
| conv3d | `tests/ttnn/nightly/unit_tests/operations/conv/test_conv3d.py` | `test_conv3d_sweep_shapes`, `test_conv3d_sweep_blocks` | silent-wrong (templated `untilize_block<>` mis-stride) | 1 |
| matmul fused activations | `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_activations.py` | `test_matmul_with_fused_activations` | silent-wrong for `[relu]` activation (pack-fused ReLU no-op) | 3 |
| MOE compute (Bfp4_b weights) | `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_compute_single_card.py` | `test_moe_compute_single_card_deepseek[no_bias]`, `test_moe_compute_single_card_gpt_oss` | silent-wrong (Bfp4_b decoder missing) | 2 |
| MOE GPT e2e | `tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py` | `test_moe_gpt_e2e[4x8]` if mesh-compatible, else `test_dispatch_compute` | silent-wrong (Bfp4_b) | 2 |
| mixed-format binary | **no existing ttnn test** | new `tests/` test added in tt-emule (see Batch 2) | n/a (we author the test) | 2 |

Caveat: some of these are in `tests/ttnn/nightly/…` which may have
extra fixture requirements (model files, mesh sizes). The first
baseline run of each test is also a feasibility check — if it can't be
run on emule's single-device setup, document that and adjust the CI
entry (e.g. select specific parametrize values).

## Phase 1 — audit doc validation pass (1 commit, doc-only)

Update `docs/notes/llk-shim-audit.md`:

- Stamp validation date and the rebase base (`arminale/comparison-ops`).
- For each finding, add **ACTIVE** / **LATENT** / **FIXED** status,
  with named caller(s) for ACTIVE items.
- Correct stale file:line citations (e.g. `pack_tile_block` is at
  `common.h:416` not 394-400).
- Re-state the recommendation section as a CHECKLIST grouped by batch —
  each entry can be marked `[x]` when the corresponding commit lands.

## Phase 2 — Batch 1: missing shims (2 commits)

**Commit 2a — baseline tests pre-fix:**
- Add new run_pytest entries to both WH and BH scripts for:
  `ssm_test_prefix_scan`, `rot_test_llama`, `matmul_test_group_attn`,
  `conv_test_conv3d`.
- Run each one standalone, capture the failure mode (expected JIT-fail
  for ssm/rotary, silent-wrong for group_attn/conv3d). Save the output
  snippet in the commit message or in `docs/notes/batch1-baseline.md`
  so we have a "before" record.
- This commit's regression status will FAIL — that's expected. (Commit
  on its own branch so it doesn't break CI, then rebase into the same
  PR.)

**Commit 2b — Batch 1 fix + audit checkpoint:**

Files:
- `include/jit_hw/api/compute/untilize.h` — add
  `inline void untilize_uninit(uint32_t = 0) { __llk_pack_is_untilize = false; }`.
  Update `tilize_uninit` to also clear `__llk_pack_is_untilize` for
  symmetry (Tier 1 #7).
- `include/jit_hw/api/compute/untilize.h` — fix templated
  `untilize_block<>` overload (Tier 1 #8). Drop the broken template
  variant body and forward to the runtime overload with
  `ntiles = block_tile_count` (the silicon template arg is DST batch
  size, not total tile count).
- `include/jit_hw/api/compute/tilize.h` — add no-op forwarders for
  `tilize_init_short_with_dt`, `tilize_uninit_with_dt`,
  `fast_tilize_init_with_dt`, `fast_tilize_init_with_dt_skip_remap`.
- `include/jit_hw/api/compute/common.h` — add public `pack_init`,
  `pack_dest_init`, `pack_reconfig_l1_acc(uint32_t)` (forwards
  `llk_pack_reconfig_l1_acc`), public `pack_relu_config(ReluType)`,
  `pack_untilize_init_skip_remap`.
- `include/jit_hw/api/compute/pack_untilize.h` (or new
  `tilizeA_B.h`) — `tilizeA_B_reduce_init`, `unpack_tilizeA_B_block`,
  `unpack_tilizeA_B_uninit` as no-ops / forwarders. Re-check if the
  callers (compute_kernel_sentinel, unpack_tilizeA_B) appear in any of
  the nightly tests — add a corresponding CI entry if yes.
- `STRUCTURE.md` updated for new symbols/files.

Verification: all 4 baseline-failing tests now pass; full WH + BH
regression remain green. Audit doc updated to mark Tier 1 #7, #8 and
Tier 3 untilize/tilize/pack symbols as DONE.

## Phase 3 — Batch 2: format-dispatch one-liners (2 commits)

**Commit 3a — baseline tests pre-fix:**
- Add `ttnn_moe_compute_deepseek` and `ttnn_moe_gpt_e2e` entries to
  both WH and BH scripts. Pick the simplest parametrize value that
  runs on single-device emule. Run, capture pre-fix output (expect
  PCC failures from Bfp4_b garbage). Save baseline in
  `docs/notes/batch2-baseline.md`.
- Author a new tt-emule-specific unit test for mixed-format binary at
  `tests/jit_hw/test_mixed_format_binary.cpp` (or wherever existing
  tt-emule unit tests live — check `tests/` dir layout first).
  Two CBs of different DataFormat (Float32, Float16_b), call
  `add_tiles`, assert against scalar reference. Add to `run_regression.sh`.

**Commit 3b — Batch 2 fix + audit checkpoint:**

Files (all in `include/jit_hw/api/compute/`):
- `common.h` — rewrite `add_tiles` / `sub_tiles` / `mul_tiles` (around
  lines 320-390) to route both operands through
  `__emule_unpack_cb_tile_to` into two scratch buffers, do arithmetic
  on fp32. Add a second `__emule_src_scratch_b[]` thread-local
  alongside `__emule_src_scratch` (line 118).
- `common.h` — add `Bfp4_b` branch ahead of `Bfp8_b` in
  `__emule_unpack_cb_tile_to` and `pack_dst_to_buf`.
- `common.h` — tighten `cb_is_bfp8_b_format` (line 214) to consult
  `cb_data_format()` when format is set, with page_size fallback for
  `Invalid`.
- New `include/jit_hw/api/compute/bfp4.h` modeled on `bfp8.h`.
  Cross-check encode/decode with `tt_metal/impl/data_format/bfp.cpp`.

Verification: baseline-failing Bfp4_b tests now pass; the new
mixed-format unit test passes; full WH + BH regression green. Audit
doc updated to mark Tier 1 #3 and Tier 4 Bfp4_b items as DONE.

## Phase 4 — Batch 3: PACK state corrections (2 commits)

**Commit 4a — baseline tests pre-fix:**
- Add `matmul_test_fused_activations` entry to both WH and BH scripts.
  Focus on the `[relu]` activation parametrize since that's what
  exercises the pack-ReLU clamp. Capture pre-fix PCC error.
- For per-CB L1 acc, identify a matmul test that uses simultaneous
  acc-and-non-acc output CBs. Likely `bmm_large_block_zm_*` variants
  in the same test file. If no clear single-test trigger, the
  fused_activations test with `[packer_l1_acc=True]` parametrize
  partially covers it.

**Commit 4b — Batch 3 fix + audit checkpoint:**

Files:
- `include/jit_hw/api/compute/common_globals.h` — replace single
  `__emule_l1_acc_enabled` bool (`common.h:111`) with per-CB array
  `__emule_l1_acc_enabled[32]`. Add `__emule_pack_relu_mode[32]` and
  `__emule_pack_relu_threshold[32]` for the ReLU clamp state.
- `include/jit_hw/api/compute/common.h`:
  - `llk_pack_relu_config` / `pack_set_relu_threshold` (lines
    516-517): write the per-CB state.
  - `pack_dst_to_buf` (around lines 260, 291): apply
    `__emule_pack_relu_mode[ocb]` clamp before format conversion; read
    `__emule_l1_acc_enabled[ocb]` instead of the global.
  - `llk_pack_reconfig_l1_acc` (around line 580): write
    `__emule_l1_acc_enabled[ocb] = (enable != 0)`.

Verification: the `[relu]` parametrize now passes; full WH + BH
regression remain green. Audit doc updated to mark Tier 2 (both items)
as DONE.

## Phase 5 — close and ticket remaining items (1 commit, doc-only)

Final audit-doc cleanup:

- Section "DEFERRED — file as issue" for Tier 1 #4 (copy_tile /
  `DST_ACCUM_MODE`), #6 (pack_tile_block slot overlap, no current
  overlap pattern), #10 (matmul transpose=1, no current caller),
  Tier 4 IEEE Float16 / Int8 / UInt8 (no caller), Tier 5 (reduce MAX
  tiny-tile, needs unit test).
- Section "STRUCTURAL — separate effort" for Batches 4-6:
  strip-layout helper (Tier 1 #1), pack-untilize numeric fixes
  (Tier 1 #2), per-CB face plumbing (Tier 1 #5, #9). Note Batch 6
  requires a tt-metal pin bump (host JIT-define generator).
- Update the recommendation summary to reflect the new state.

## Files likely touched (cumulative)

- `docs/notes/llk-shim-audit.md` (every phase)
- `docs/notes/batch{1,2}-baseline.md` (Phase 2-3 pre-fix records)
- `include/jit_hw/api/compute/{tilize,untilize,pack_untilize}.h`
- `include/jit_hw/api/compute/common.h`
- `include/jit_hw/api/compute/common_globals.h`
- `include/jit_hw/api/compute/bfp4.h` (new)
- `include/jit_hw/api/compute/tilizeA_B.h` (new, optional)
- `scripts/run_ttnn_pytests_wormhole.sh`
- `scripts/run_ttnn_pytests_blackhole.sh`
- `tests/jit_hw/test_mixed_format_binary.cpp` (new) or equivalent
- `STRUCTURE.md`

## Out of scope

- SrcA/SrcB modelling — by design, emule does not model SrcA/B.
- Batches 4-6 (page-size-strip helper, pack-untilize numeric fixes,
  per-CB face plumbing) — defer to follow-up plan once Batches 1-3
  land.
- tt-metal pin bump — none of Batches 1-3 needs it.
- Pushing — explicit user instruction required per workspace rules.

## Verification (end-to-end)

For each batch:
1. Run new baseline tests pre-fix, capture failure mode in
   docs/notes/batchN-baseline.md.
2. Land the fix.
3. Run the same tests post-fix → expect PASS.
4. `pytest test_permute_sharded` (build smoke).
5. `bash scripts/run_ttnn_pytests_wormhole.sh` full regression.
6. `bash scripts/run_ttnn_pytests_blackhole.sh` full regression.
7. Audit doc updated with `[x]` on the matching checklist items
   before committing.
