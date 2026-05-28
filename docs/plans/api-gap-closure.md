# Plan: Close the tt-emule API gap blocking the tt_transformers Llama vertical

> **Status**: completed 2026-05-28. Landed on `arminale/model-sim` as commit `dccda04` (jit_hw: close API gap for tt_transformers Llama vertical (bcast + reconfig)). Archived here from `~/.claude/plans/` for repo-side history. Superseded by `docs/plans/rms-norm-pcc-debug.md` (or whatever the next plan name is in `~/.claude/plans/`).

## Context

`test_embedding.py` passes under emule (both variants). The next op in the Llama vertical, `test_rms_norm.py`, JIT-fails. Surface inventory from first-iteration smoke + an Explore-agent sweep across `test_rms_norm`, `test_rope`, `test_attention`, `test_mlp`, `test_lm_head` reveals a **shared API gap** — not just a per-op issue. The same missing symbols block multiple ops in the vertical.

emule's existing compute API was designed against an earlier tt-metal API that used templated bcast/reduce primitives parameterized by `<EltwiseBinaryType, BroadcastType>`. Current tt-metal kernels use non-templated, op-specific wrappers (`mul_tiles_bcast_cols`, `add_tiles_bcast_rows`, etc.) and a new `compute_kernel_lib::reduce<...>` template from `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` that wraps reduction LLKs. emule has neither.

Fixing this as a one-off-per-op chase is wasteful. Categorizing the gap and closing it in batches gives leverage: each shim added unlocks multiple downstream ops.

**Goal:** close enough of the API gap to make `test_rms_norm.py` pass both PREFILL and DECODE parametrize variants. Leave `attention`, `mlp`, `lm_head` reachable for follow-up sessions (most of their gaps will already be covered by the same shims).

## Approach

### Three API gap categories, in order of leverage × cost

| # | Category | Symbols | Files | Real math needed | LoC | Unblocks |
|---|---|---|---|---|---|---|
| 1 | Non-templated bcast wrappers | `mul_tiles_bcast_{cols,rows}`, `add_tiles_bcast_rows`, `sub_tiles_bcast_cols`, plus 4 `*_init_short` companions | `include/jit_hw/api/compute/bcast.h` (extend) | **Yes — bf16/fp32 broadcast math** | ~150 | rms_norm + softmax + future row/col scaling |
| 2 | 4-arg `reconfig_data_format` overload | `reconfig_data_format(srca_old, srca_new, srcb_old, srcb_new)` | `include/jit_hw/api/compute/reconfig_data_format.h` or `common.h` (extend) | No — emule already no-ops the existing 1- and 2-arg forms | ~5 | rms_norm + any kernel doing dual-input format changes |
| 3 | `compute_kernel_lib::reduce<...>` + `numeric::row_wise_mean<...>` host shims | The `reduce_helpers_compute.hpp` template family, plus `numeric.h` row-wise helper | New: `include/jit_hw/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` + `include/jit_hw/ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h` | **Yes — host-side reduction** | ~250 | rms_norm (both variants) + attention softmax + any custom reduce |

### Sequencing

1. **Category 2 first** (5 LoC, trivial) — closes one error class outright.
2. **Category 1** — write the eight bcast wrappers with real bf16/fp32 math. Mirrors emule's existing `add_tiles` / `mul_tiles` patterns in `common.h:232-301` (read CB tile, walk row-major with `__emule_nfaces::rowmajor_to_nfaces`, write DST). The bcast variants change *which* index in `cb_b` to read (row 0 broadcast across rows, col 0 broadcast across cols). Pattern is uniform; one helper function + 8 thin wrappers.
3. **Category 3** — write the reduce-helpers + numeric.h shims. Host-side `compute_kernel_lib::reduce<PoolType, ReduceDim, ...>(...)` template + `numeric::row_wise_mean` that delegates to it. Follow the worked example of `tilize_helpers.hpp`'s approach (sidestep the upstream `.inl`, write our own template that uses CB byte-pool primitives directly).

After each category, re-run the smoke. Don't proceed to the next category until the current one compiles cleanly (a green compile validates the shim's shape; subsequent categories close the *next* layer of failures).

### Risks to track

- **`p_dim_stride_target` enum** referenced in `reduce_helpers_compute.inl:32,78`. Likely a Tensix-only enum used in the upstream `.inl` for reconfig dispatch. Our Category 3 shim sidesteps the upstream `.inl` entirely (writes its own template), so this should not surface — but verify after Category 3 lands.
- **PCC sensitivity to bcast math precision**: bf16 reduction inside layernorm is fundamentally noisy. RMS norm's PCC threshold is 0.9999 (per the test). If the host bcast math has a precision delta from real Tensix, the test could fail at PCC even with correct shimming. Mitigation: implement reductions in float32 accumulator throughout, downconvert to bf16 only on final pack — that matches the FLOAT32_REDUCTION template arg used in `numeric::row_wise_mean<...>`.
- **`layernorm_sharded.cpp` (DECODE variant) may surface additional gaps** beyond Categories 1-3. Plan for an Iteration 4 if so.
- **`TT_CCL` on (1,1)** — still untested. After Categories 1-3 land, if rms_norm STILL fails (with a TT_CCL-related error), that's a separate plan (likely requires shimming the all-gather / all-reduce primitives or asserting that TT_CCL degenerates cleanly when num_devices=1).

## Critical files

**Read-only references** (to mirror patterns from):
- `/localdev/arminale/tt-emule/include/jit_hw/api/compute/common.h:232-301` — `add_tiles` / `sub_tiles` / `mul_tiles` reference implementations (bf16/fp32 math against CB byte pool).
- `/localdev/arminale/tt-emule/include/jit_hw/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` — the kernel-lib shim pattern (template + host data work, bypasses upstream `.inl`).
- `/localdev/arminale/tt-emule/include/jit_hw/api/compute/bcast.h` — current (templated, no-op) bcast surface; extend, don't replace.
- `/localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/bcast.h:195-260` — real tt-metal signatures we have to match (especially `mul_tiles_bcast_cols`, `add_tiles_bcast_rows`, the `bcast_row_idx` parameter on row variants).
- `/localdev/arminale/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` + `.inl` — the upstream interface our shim must mirror (`compute_kernel_lib::reduce<PoolType, ReduceDim, ReduceInputPolicy, ReduceDataFormatReconfigMode>`).
- `/localdev/arminale/tt-metal/ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h` — `numeric::row_wise_mean` upstream definition.
- `/localdev/arminale/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/{layernorm,layernorm_sharded}.cpp` — the kernels driving the test.

**Files to create/extend:**
- `include/jit_hw/api/compute/bcast.h` — extend with 8 non-templated wrappers + real math.
- `include/jit_hw/api/compute/common.h` (or new `reconfig_data_format.h`) — add the 4-arg `reconfig_data_format` no-op overload.
- `include/jit_hw/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — new shim, full host implementation of `compute_kernel_lib::reduce<...>` template.
- `include/jit_hw/ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h` — new shim, host `numeric::row_wise_mean<...>` that delegates to the reduce shim.

**Notes file:** `/localdev/arminale/tt-emule/docs/notes/rms_norm-bring-up.md` — running notes; append per iteration.

## Verification

- **After Category 2**: `test_rms_norm.py` recompiles past the `reconfig_data_format` error and surfaces the bcast errors as the next blocker.
- **After Category 1**: `test_rms_norm.py` (PREFILL) recompiles past the bcast errors and surfaces the reduce_helpers / numeric errors as the next blocker.
- **After Category 3**: `test_rms_norm.py` PREFILL passes JIT compile cleanly. Either the test passes PCC, or surfaces a different failure mode (likely either DECODE-only gaps or TT_CCL behavior).
- **End state target**: both variants pass:
  ```
  PASSED test_rms_norm_inference[wormhole_b0-…-Mode.PREFILL-…]
  PASSED test_rms_norm_inference[wormhole_b0-…-Mode.DECODE-…]
  ```
- **Bonus check**: smoke `test_attention.py` after rms_norm passes (one pytest invocation) to see whether the same shims also moved the attention bring-up forward. Just observational — formal attention bring-up is a follow-up session.
- Update the op-bring-up skill in place after rms_norm passes (per existing user direction): add a "Worked example: RMS norm" subsection covering the bcast/reduce shim pattern.

## Out of scope

- `test_attention.py`, `test_mlp.py`, `test_lm_head.py`, `test_decoder.py`, `test_model.py` — verified that the same Category 1-3 shims unblock them, but each is its own bring-up session with its own notes file.
- `TT_CCL` on (1,1) — addressed only if it surfaces as a blocker after JIT compile is clean.
- The mask ops (Category 4 from the inventory) — only needed for attention/softmax, deferred.
- CI integration of the rms_norm test once it passes — done in a separate productionalization plan, not here.
- Modifying tt-metal or tt-mlir source — we're a downstream shim consumer.

## Why this is one plan, not five iteration plans

Each shim category opens a clear API surface that several downstream ops will reuse. Fixing them in the order above means:
- Category 2 is a one-line trivial cleanup that should have been there anyway.
- Category 1 is the canonical "extend emule's existing compute primitives to match upstream" pattern.
- Category 3 is the canonical "shim a kernel-lib helper so the whole upstream `.inl` family doesn't matter" pattern from the embedding bring-up.

If we did this per-op, we'd add half a category per op and never see the leverage. Planning as one cohesive API-gap fix produces a smaller, faster set of shims that more ops will benefit from.

## What actually happened (post-execution)

- Category 2 (4-arg `reconfig_data_format`) closed cleanly — 5 LoC to `common.h`.
- Category 1 (non-templated bcast wrappers) closed cleanly — ~100 LoC to `bcast.h`, real bf16/fp32 broadcast math driven by `__emule_bcast::apply_bcast<Op>` template + 4 thin wrappers + 4 no-op `_init_short` companions.
- Category 3 (reduce_helpers + numeric.h shims) **turned out not needed for JIT compile**: emule's `-I` paths fall through to the real tt-metal headers, which compile fine on host because they ultimately call emule's existing `reduce_init`/`reduce_tile` primitives.
- Result: `test_rms_norm.py` PREFILL compiles cleanly, runs end-to-end, but fails PCC (-0.02 with NaN). The API-surface gap is closed; a separate plan covers the PCC debug.
