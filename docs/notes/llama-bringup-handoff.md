# Llama-3.2-1B bring-up on tt-emule — session handoff

## TL;DR

Three ops parked, all in `docs/notes/`. Phase A (DECODE-only Llama end-to-end) needs to clear **lm_head PCC=0.0** and the **attention SDPA sfpi-inline blocker** to reach the finish line. Phase B (PREFILL) needs **minimal_matmul PCC=0.002** unparked. Skill rule confirmed: no sfpi emulation layer; no changes to op implementations.

## What "Llama bring-up" means here

Goal: run `models/tt_transformers/tests/test_model.py` for Llama-3.2-1B on tt-emule N150 (mock cluster, slow dispatch). The test exercises the full model forward — embedding → 16× (rms_norm → attention → residual → rms_norm → MLP → residual) → final_norm → lm_head — and asserts PCC vs a torch reference.

Two execution modes:
- **DECODE**: single-token prefill or per-step generation. Uses `ttnn.linear` for all matmuls; does NOT dispatch `minimal_matmul`.
- **PREFILL**: long-context (seq_len > 128). Dispatches `ttnn.experimental.minimal_matmul` for MLP w2 and attention QKV.

## Session entry state (what was already working)

| Op | Status entering session |
|---|---|
| `test_embedding.py` | PASS |
| `test_rms_norm.py` (PREFILL + DECODE) | PASS @ PCC 0.9999963 |
| `test_mlp.py` DECODE seq_len=32 | PASS @ PCC 0.9995492 |
| Full wormhole regression baseline | 33 PASS / 9 FAIL (the 9 are unrelated `ttnn_sum_*` / `ttnn_minmax_*`) |

Infra in place before session: Bfp8_b encode/decode (bit-exact), per-bank DRAM-sharded routing (`__emule_dram_ptr_at`), mixed-format binary ops, `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE` guard, `matmul_block` k-stride semantics, NOC L1_SLOT_MASK bounds check.

## Session work, in order

### 1. minimal_matmul bring-up attempt (multi-session effort, parked)

PR #36 on `arminale/model-sim` (emule) green; pinned to tt-metal `arminale/emule-dram` at `7e748628f89`. Attempted to clear MLP PREFILL (seq_len=512) since that's the smallest minimal_matmul shape.

**Three gates progressed through:**

| Gate | Outcome |
|---|---|
| JIT-compile gap | CLOSED via 5 incremental shims (templated `*_tiles_bcast<BT>`, fabric_edm include path, `common_values.hpp`, `noc_semaphore_set_remote` shim, helper-definition reordering) |
| Runtime SEGV in `__memcpy_evex_unaligned_erms` | CLOSED via `L1_SLOT_MASK` bounds-check in `__emule_resolve_noc_addr` (tt-metal commit `dd3943d3db4`) + `__emule_safe_memcpy` defensive wrapper in `dataflow_api.h` |
| Numerical correctness (PCC 0.002) | **PARKED** |

**Numerical bug investigation that didn't conclude:**

- `[MM-IN]` trace showed sane bf16 values in in0/in1 CBs.
- `[MB-OUT]` trace showed `matmul_block` firing with `rt=1, ct=4, kt=8` but every trace entry had `in0_t=0, in1_t=0, idst=0`.
- `[L1-ACC]` trace showed each thread sees one `enable=0` and one `enable=1` call — consistent with `K_num_blocks=1` and `N_blocks_per_core=1`, meaning PACKER_L1_ACC isn't doing K-accumulation; K-accum happens in DST within `matmul_blocks`.
- Commit-history review (`c86e62a0e5a`, `54eb77135d9`) found nothing matching our failure mode.

**Strategic reframe (mid-session, after user clarification):**
User clarified: emule is single-device only; CCLs are cross-device. `minimal_matmul` IS used in single-device tests because the MLP code dispatches `ttnn.experimental.minimal_matmul` unconditionally when `seq_len > 128 and mode != Mode.DECODE`. On a 1-device target, the kernel must degenerate to a regular sharded matmul (intra-device NoC traffic across the core grid, no inter-device fabric). Whatever it does intra-device — that's emule's surface to support.

User also clarified: we cannot change the op implementation.

Llama-vertical usage of minimal_matmul confirmed to be only 2 sites: `mlp.py:276` (w2 down-proj, PREFILL) and `attention.py:908` (QKV fused, PREFILL). DECODE never reaches it. Decision: **defer to Phase B**, proceed with DECODE-only Llama.

Full record: `docs/notes/minimal-matmul-bring-up.md`.

### 2. Plan restructure — Phase A (DECODE) + Phase B (PREFILL)

`/home/arminale/.claude/plans/jiggly-snuggling-dolphin.md` rewritten:
- Phase A: ops #1, #3, #4, #5, #6, #8, #10a + cross-cutting X2 (SDPA shims). End: greedy DECODE generation works end-to-end.
- Phase B: ops #2, #7, #9, #10b. Gated on `minimal_matmul`.

### 3. lm_head (op #3) — PARKED at PCC=0.0

Test: `test_lm_head.py::test_lm_head_inference[...mesh_device0-1-32-False]` (single param combo).

**Compile gap closed** (3 shims in `tensor_accessor.h`):
- `make_tensor_accessor_tuple<...>(args, addr_start)` — templated tuple builder.
- `AbstractTensorAccessorWrapper` + `make_abstract_tensor_accessor_wrappers` — type-erased wrappers used by concat reader.
- Real-API `TensorAccessor(TensorAccessorArgs<CTA_OFFSET>, addr)` constructor matching the single-template-arg form.

**Numerical correctness: PARKED at PCC=0.0.**

Diagnostic Python-level dump showed all 3 split outputs of the lm_head linear are **100% zero** (`nz=0/1368064` for each). A focused probe (`tests/test_lmh_probe.py`, since removed) reproduced the failure: `ttnn.linear` and `ttnn.matmul` with bfloat8_b activation × bfloat8_b weight, INTERLEAVED L1/DRAM inputs, no special program config, return all-zero output even at N=2048. Input act and weight read back correctly (non-zero, sane magnitudes). Reference torch matmul produces non-zero output.

**Distinguishing observation:** MLP DECODE passes through what's nominally the same `ttnn.linear` path (also dispatches `dram_matmul_config`), but MLP DECODE provides a block-sharded activation (output of rms_norm), while lm_head and my probe provided interleaved L1.

**Hypothesis:** the auto-picked matmul kernel for interleaved-L1 inputs at lm_head's shape (likely `bmm_large_block_zm_fused_bias_activation`) has an emule-side bug not exercised by MLP DECODE because MLP DECODE always provides sharded inputs.

**Not yet attempted:** capturing the kernel source from `/tmp/tt_emule_jit_*/patched_kernel.cpp` for the failing case; comparing against the kernel JIT'd by MLP DECODE; testing whether MLP DECODE itself fails when re-run with an interleaved input.

Full record: `docs/notes/lm_head-bring-up.md`.

### 4. attention DECODE (op #5) — PARKED at sfpi inline blocker

Test: `test_attention.py::test_attention_inference[...hf_rope-256-1-page_params0-default_attention-mesh_device0-False]` (DECODE batch=1, max_seq_len=256).

**18+ compile-gap shims closed** in this session — full table in `docs/notes/attention-bring-up.md`. Highlights:
- `compute_kernel_hw_startup`, `pack_untilize_block<>`, `noc_async_{read,write}_one_packet`, `tt_reg_ptr`, `RISCV_DEBUG_REG_WALL_CLOCK_*`, `tensix_types.h` (empty shim), `TRISC_MATH` define, `llk_math_eltwise_unary_sfpu_mask*`, `llk_math_eltwise_binary_init`, `llk_unpack_AB_init` / `llk_unpack_A_init`, non-templated 3-arg `pack_tile`, `compute_kernel_lib::tilize<>` defaults, `power_iterative_tile`, `binary_max_tile` 4-arg overload, emule softplus override (host log1p/exp), emule `reduce_custom.h` override (host per-row max), `has_get_aligned_page_size_v` trait, `add_bcast_*_init_short` non-templated overloads, `sfpi.h` empty stub.

tt-metal JIT include paths added (uncommitted on `arminale/emule-dram` — see below):
- `ttnn` (so `cpp/ttnn/...` form resolves)
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc`

**Final blocker: sfpi inline use in `compute_common.hpp`.** SDPA's compute kernel uses sfpi RISC-V SFPU vector intrinsics inline:

```cpp
sfpi::vFloat in = sfpi::dst_reg[0];
sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
out = sfpi::convert<sfpi::vFloat16b>(out, RoundMode::NearestEven);
sfpi::dst_reg[0] = out;
```

User-imposed constraint: **no sfpi emulation layer**.

**Viable strategy identified (not yet executed):** sfpi use is confined to `#ifdef TRISC_MATH`-guarded blocks in `compute_common.hpp`, and the entry points the rest of the kernel calls into are just **4 thin wrapper functions** (clean function-call boundaries):

| Wrapper | What it does | Host-side equivalent |
|---|---|---|
| `recip_tile_first_column(idst)` | `dst[r] = 1.0f / dst[r]` on column 0 of tile | trivial loop |
| `exp_tile_first_column(idst)` | `dst[r] = exp(scale * dst[r])` on column 0 | trivial loop |
| `fused_max_sub_exp_add_tile(idst, scale)` | fused softmax numerator step | ~10 lines |
| `softplus_tile_first_column(idst, beta, ...)` | per-element softplus on column 0 | reuse my existing softplus shim body |

Strategy: undefine `TRISC_MATH`, provide emule shims for these 4 wrappers in a pre-included header, and add a small emule override of `mask.h` (which also uses `MATH(...)` macros) so it doesn't need TRISC_MATH either. Total estimated: ~80 lines, well within scope. **Not done this session.**

Full record: `docs/notes/attention-bring-up.md`.

## Outstanding problems (parked)

### P1 — minimal_matmul PCC=0.002 (Phase B blocker)
Doc: `docs/notes/minimal-matmul-bring-up.md`. Compile and SEGV closed. Numerical bug uninvestigated since the strategic pivot. Untested hypotheses: (a) tile-content integrity check at compute kernel (hash in0/in1 vs host); (b) `num_devices` / `is_sink_core` runtime-arg values reaching dataflow kernels; (c) CB write_idx/read_idx race.

### P2 — lm_head PCC=0.0 (Phase A blocker)
Doc: `docs/notes/lm_head-bring-up.md`. Compile gap closed. Reproduced in isolation: any `ttnn.linear`/`ttnn.matmul` with bf8 × bf8 interleaved-L1 inputs returns all-zero output. Untested: which JIT kernel is auto-picked; whether MLP DECODE also fails if re-fed with interleaved inputs (would confirm it's a kernel-not-config issue).

### P3 — attention DECODE sfpi-inline (Phase A blocker)
Doc: `docs/notes/attention-bring-up.md`. 18+ compile gaps closed; structurally blocked on sfpi-inline. **Viable strategy ready** (4 wrapper-function emule shims, ~80 lines). Just hasn't been executed.

## Gaps not yet attempted

Items from the original plan that haven't been touched this session at all:

- **Op #4 — RoPE device path**: no standalone test in `test_rope.py` (only CPU tests). Would be exercised through attention; currently shadowed by P3.
- **Op #6 — attention DECODE batch=32**: same kernel as b=1, exercises KV cache across many steps. Gated by P3.
- **Op #7 — attention PREFILL** (seq_len 256, 4k, 16k, 32k): gated by P1 (minimal_matmul) for QKV fused.
- **Op #8 — decoder DECODE** (`test_decoder.py`): gated by P3.
- **Op #9 — decoder PREFILL**: gated by P1 + P3.
- **Op #10a — full model DECODE** (`test_model.py`, n_layers=1 → 16): gated by P2 + P3.
- **Op #10b — full model PREFILL** + `test_chunked_generation.py`: gated by P1 + P3.

So with P1, P2, P3 closed, Phase A (DECODE end-to-end) clears. Phase B (PREFILL) needs P1.

## Repo + branch state at session end

### tt-emule (`arminale/model-sim`)
- HEAD: `a94246e jit_hw: more SDPA compile-gap shims (attention DECODE in-progress)`
- Previous: `66b81d2 jit_hw: API-gap shims for attention DECODE + lm_head compile`
- Both pushed status: **NOT pushed in this session** — user instruction is never push without explicit request.
- Working tree: clean other than untracked `.rtk/` and `scripts/__pycache__/`.

### tt-metal (`arminale/emule-dram`)
- HEAD: `dd3943d3db4` (unchanged from session start; pin in `tt-metal-pin.txt` points to `585f8196abc` which is one commit older).
- **Uncommitted changes**: `tt_metal/impl/emulation/emulated_program_runner.cpp` has:
  - `__emule_resolve_noc_addr` L1_SLOT_MASK refinement (cosmetic — currently `<= L1_SLOT_SIZE` vs prior `>= core_l1_size`).
  - **New JIT include paths added** in `get_extra_include_flags()` for `ttnn`, `wormhole_b0/metal/llk_api/llk_sfpu`, `tt_llk_wormhole_b0/common/inc`.
- Action when resuming: commit those changes on `arminale/emule-dram`, push, bump `tt-metal-pin.txt` in the next emule commit.

### Pin
`tt-metal-pin.txt` = `585f8196abc24fdfee5eadbcf8b52ed64975f066` — needs bump to whatever SHA we land the include-path changes at.

## How to resume

Recommended next session order:

1. **P3 attention DECODE — 4-wrapper override** (clean, scoped, well-understood). One commit on `arminale/model-sim` adding emule overrides for `recip_tile_first_column`, `exp_tile_first_column`, `fused_max_sub_exp_add_tile`, `softplus_tile_first_column` + `mask.h` override + undef TRISC_MATH. Retry the attention smoke. Expect: more API gaps to surface, but the sfpi blocker is gone.
2. **Commit the tt-metal include-path additions** on `arminale/emule-dram`, push, bump pin.
3. **P2 lm_head — re-probe**. Capture the JIT'd kernel for failing case. Run MLP DECODE with deliberately interleaved input to test the kernel-vs-config hypothesis. If that reproduces, the bug is in the interleaved-input reader path.
4. After P2 + P3: walk #4, #6, #8, #10a in sequence.
5. **P1 minimal_matmul** when Phase A is done.

## Verification commands

Standard env block (copy verbatim, replace pytest target):

```bash
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_* /tmp/tt_emule_jit_cache_$(id -u)
export ROOT=/localdev/arminale
export PYTHONPATH=$ROOT/tt-metal/ttnn:$ROOT/tt-metal/tools:$ROOT/tt-metal/build_emule/lib:$ROOT/tt-metal:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$ROOT/tt-metal/build_emule/lib:$ROOT/tt-metal/build_emule/tt_metal/third_party/umd/lib:${LD_LIBRARY_PATH:-}
export TT_METAL_HOME=$ROOT/tt-metal
export TT_METAL_RUNTIME_ROOT=$ROOT/tt-metal
export TT_METAL_MOCK_CLUSTER_DESC_PATH=$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export MESH_DEVICE=N150
export HF_MODEL=meta-llama/Llama-3.2-1B
export TT_CACHE_PATH=/localdev/arminale/.tt_cache
cd $ROOT/tt-metal
/opt/ttmlir-toolchain/venv/bin/pytest <test target> -v --tb=line -s
```

Regression sanity (must pass after any change):
```bash
pytest "models/tt_transformers/tests/test_rms_norm.py"  # PCC ≥ 0.9999963 both modes
pytest "models/tt_transformers/tests/test_mlp.py::test_mlp_inference[wormhole_b0-device_params0-1-32-mesh_device0-False]"  # PCC ≥ 0.9995
bash scripts/run_regression_wormhole.sh  # baseline 33/9
```

## Rebuild commands

If tt-metal sources change:
```bash
cd /localdev/arminale/tt-metal && cmake --build build_emule --target tt-umd tt_metal -j
```

If tt-emule headers change: no rebuild needed (JIT picks up at next compile).

## Confirmed user constraints (this session)

- Never push to remote without explicit user request.
- Never modify op implementations.
- Never build an sfpi emulation layer.
- Always run regression tests after code changes.
- Compile JIT using clang-20; keep g++ for `jit_kernel.cpp`.
- Always log full regression output.
- Single-device N150 only; CCLs (cross-device) are exclusively for fabric/ethernet — intra-device NoC is in scope.

## Files in scope for next session

Read-first:
- `docs/notes/minimal-matmul-bring-up.md`
- `docs/notes/lm_head-bring-up.md`
- `docs/notes/attention-bring-up.md`
- `/home/arminale/.claude/plans/jiggly-snuggling-dolphin.md`

Likely-to-edit:
- `include/jit_hw/api/compute/` headers (for P3 wrapper shims).
- `tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp` (already dirty — needs commit).
- `tt-metal-pin.txt` (bump after tt-metal commit).
