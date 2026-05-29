---
name: op-bring-up
description: Use when bringing up a ttnn op under tt-emule (the Tenstorrent software emulator) so a tt-metal or tt-mlir pytest can exercise that op on a single emulated Wormhole/Blackhole chip. Walks through environment setup, failure-driven LLK gap discovery, shim placement, and iteration to a passing PCC check.
---

This skill captures the methodology for bringing up a ttnn op under tt-emule. Updated in place after each new bring-up — read the worked examples to see what's been covered and what hasn't.

## When to use this skill

You are about to add (or have just hit) a missing kernel / LLK / kernel-lib gap in tt-emule because a ttnn pytest (typically under `models/tt_transformers/`, `models/demos/`, or `tests/ttnn/integration_tests/`) JIT-fails on a kernel under `tt-metal/ttnn/cpp/ttnn/operations/<op>/device/kernels/`.

You are **NOT** using this skill if:
- The pytest fails *before* any device op runs (Python import error, `ModelArgs` assert, etc.) — that's environment work, not op bring-up.
- The pytest runs but PCC fails — that's a numerical correctness issue, treat separately.
- The op is multi-device (mesh ≥ (1, 2)). emule models one chip.

## Inputs you need before starting

- ttnn op name (e.g. `ttnn.embedding`, `ttnn.multiply`).
- The test that drives the op (path to the pytest file + parametrize variant).
- Target arch (default: Wormhole N150, `MESH_DEVICE=N150`).
- HF model (if `ModelArgs` is involved): pick the smallest in the supported family (e.g. `meta-llama/Llama-3.2-1B` for Llama tests).

## Environment prerequisites (one-time)

The `/opt/ttmlir-toolchain/venv` ships only `torch` (CPU). `models/conftest.py` requires more. One-shot install:

```bash
/opt/ttmlir-toolchain/venv/bin/pip install --quiet \
  'transformers==4.53.0' 'huggingface-hub>=0.30.0' 'torchvision==0.26.0' \
  'accelerate==1.7.0' 'Pillow==12.2.0'
```

Pin versions from `tt-metal/tt_metal/python_env/requirements-dev.txt`. The user typically has write access to the venv.

Confirm `build_emule` is up to date and has `TT_METAL_USE_EMULE=ON`:
```bash
nm -DC /localdev/arminale/tt-metal/build_emule/tt_metal/libtt_metal.so | \
  grep emule::execute_program_emulated   # must show a T line
```

## Standard env block for every smoke run

```bash
export ROOT=/localdev/arminale
export PYTHONPATH=$ROOT/tt-metal/ttnn:$ROOT/tt-metal/tools:$ROOT/tt-metal/build_emule/lib:$ROOT/tt-metal:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$ROOT/tt-metal/build_emule/lib:${LD_LIBRARY_PATH:-}
export TT_METAL_HOME=$ROOT/tt-metal
export TT_METAL_RUNTIME_ROOT=$ROOT/tt-metal
export TT_METAL_MOCK_CLUSTER_DESC_PATH=$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1   # project rule: emule always slow-dispatch
export MESH_DEVICE=N150
export HF_MODEL=<your-model>            # if the test loads weights
export TT_CACHE_PATH=$ROOT/.tt_cache    # avoids re-download across iterations
```

## Step 1 — First smoke (expect failure)

Run the test once *with* `--forked` to keep the JIT cache isolated, but **without** `--capture=no` first (to confirm the gross failure mode):

```bash
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_* /tmp/tt_emule_jit_cache_$(id -u)
timeout 600 /opt/ttmlir-toolchain/venv/bin/pytest \
  <path-to-test> -v --tb=short --forked > /tmp/op-smoke-1.log 2>&1
grep -E '^FAILED|RuntimeError' /tmp/op-smoke-1.log | head -5
```

The most likely failure is `RuntimeError: jit_compile_kernel: compiler failed (exit 256) for kernel: <path-to-kernel>.cpp`. **Note the kernel path** — that's your immediate target.

## Step 2 — Read the failing kernel source first

**Always check the kernel source before reaching for a manual compile.** The kernel's `#include` list tells you which sub-system is involved, and a quick comparison against `ls /localdev/arminale/tt-emule/include/jit_hw/api/compute/` often surfaces the gap immediately.

Two failure patterns dominate:

**(A) The kernel uses a kernel-lib helper or LLK that emule has not implemented at all.** Embedding's `compute_kernel_lib::tilize<...>` was this case — substantial new code (host-side data rearrangement) needed.

**(B) The kernel is a "fat dispatcher" (one source file handles many ops via JIT-time op-selection macros) that transitively includes op headers emule hasn't shimmed yet.** Multiply's `eltwise_binary_sfpu_scalar.cpp` was this case — pulls in ~15 binary-SFPU op headers (`atan2.h`, `isclose.h`, `xlogy.h`, …). The unselected op headers are dead code at this JIT instantiation, but C++ still parses them. emule's missing shims (`atan2.h`, `isclose.h`) fell through to the real tt-metal headers which use undefined `APPROX` / unstubbed `llk_math_*`.

**(B) is much cheaper to fix than (A)** — no-op shims with empty function bodies in the `ckernel` namespace are enough. Always try to recognize (B) first by counting `#include`s for op-shaped headers (`atan2.h`, `isclose.h`, etc.) and cross-checking against emule's coverage.

## Step 3 — Surface the actual compiler stderr (when source-reading isn't enough)

pytest's `--forked` + `std::system("…2>&1")` in `emulated_program_runner.cpp` swallows the compiler diagnostics. Two ways to surface them:

**Option A (preferred)**: drop `--forked` and add `-s`:
```bash
/opt/ttmlir-toolchain/venv/bin/pytest <path-to-test> -v -k '<one-variant>' --tb=short -s 2>&1 | \
  grep -E 'error:|warning:|kernel:' | head -30
```
The forked variant captures std::system output; without it, clang-20 stderr reaches your terminal.

**Option B (when Option A doesn't reveal enough)**: replay the compile manually outside pytest:
```bash
# 1. Run the test once so a /tmp/tt_emule_jit_XXXXXX/ tmpdir survives the failure.
# 2. Identify the dir that has your failing kernel:
grep -l '<kernel-distinctive-symbol>' /tmp/tt_emule_jit_*/patched_kernel.cpp

# 3. Reconstruct the compile from emulated_program_runner.cpp:730:
clang++-20 -std=c++20 -fPIC -shared -O2 -Wno-c++11-narrowing \
  -I/localdev/arminale/tt-emule/include/jit_hw \
  -I/localdev/arminale/tt-emule/include \
  -I<kernel-dir> \
  -I/localdev/arminale/tt-metal/ttnn/cpp \
  -I/localdev/arminale/tt-metal \
  -I/localdev/arminale/tt-metal/tt_metal/hw/inc \
  -I/localdev/arminale/tt-metal/tt_metal/hostdevcommon/api \
  -DTT_EMULE_USE_L1_POOL \
  -DKERNEL_COMPILE_TIME_ARGS=<guess-or-extract-from-test> \
  -o /tmp/test.so \
  <found-jit-dir>/wrapper.cpp 2>&1 | head -50
```
**Gotcha**: `TT_EMULE_PROJECT_SOURCE_DIR` (used in `emulated_program_runner.cpp::get_extra_include_flags`) resolves to **tt-metal's** source root, *not* tt-emule's. Easy misread; if your manual compile finds tt-emule's empty `ttnn/cpp/`, your `-I` paths are wrong.

## Step 4 — Classify the failure

| Error | Cause | Strategy |
|---|---|---|
| `'X' file not found` (path mismatch) | A `#include "X"` doesn't resolve under any `-I` path. Often emule placed the shim at a different sub-path than tt-metal uses. | One-line **redirect shim** at the tt-metal-expected path; `#include` emule's actual location. |
| `'X' file not found` for a tt-metal-side helper (e.g. `ttnn/cpp/ttnn/kernel_lib/Y.hpp`) | The helper exists in tt-metal but emule has nothing on that path → falls through to the real header which won't compile on host. | Shim the helper under `include/jit_hw/<tt-metal-path>/Y.hpp` so include-search picks it up first. |
| `'X' file not found` for an op-header pulled in by a fat dispatcher | The kernel #includes many op headers; emule hasn't shimmed this one; it falls through to real tt-metal which uses APPROX / unstubbed LLKs. **The op is dead code in this JIT instantiation.** | **No-op shim** in the `ckernel` namespace — empty bodies for the public API. Parses cleanly, never executes. ~10-20 LoC per dead op. |
| `unknown type name 'ALWI'` / `UNPACK` / `PACK` / `MATH` / `FORCE_INLINE` | Include path resolved into a real tt-metal header that assumed `common.h` was included first to define the macros. | Add `#include "jit_hw/api/compute/common.h"` (or `compute_kernel_hw_startup.h`) to the upstream shim so the macros propagate. |
| `use of undeclared identifier 'APPROX'` | Real tt-metal op header parsed without the project-wide compile-time APPROX value. | If the op is in a dead-dispatcher branch, prefer no-op shim (option above) over defining APPROX globally. The no-op shim never references APPROX. |
| `use of undeclared identifier 'llk_X'` | Real LLK function the host can't implement directly (e.g. `tilize_block`, `llk_wait_tiles`). | Either shim the user-facing API one level up (preferred — collapses N LLK calls into one host function) or no-op a low-impact stub. |
| `functions that differ only in their return type` | emule and a real tt-metal header both declare the same free function with different return types (typical: `get_local_cb_interface`). | Shim the real header's path so it never reaches the parser. emule's version is the source of truth on host. |
| Templated name lookup error in `if constexpr (use_fast)` branch on Wormhole | `if constexpr`'s discarded branch still requires names to be declared when args are template-parameter-dependent. | Don't reuse the upstream `.inl` — write your own template in the shim that never references the inapplicable variant. |
| `no matching function for call to 'foo(uint32_t, TensorAccessor&)'` | Kernel uses a free-function overload that takes an accessor, but emule only has the method form on the accessor. | Add a free function template constrained by an existing emule type-trait (e.g. `has_get_noc_addr_v`). |
| `noc_async_read<N>(…)` template undeclared | Kernel uses the templated `<max_page_size>` form; emule has only the non-templated. | Add a `template <uint32_t N, …>` overload that forwards to the existing non-templated body. |
| `silu_tile` / `gelu_tile` / `tanh_tile` undeclared in a binary_ng dispatcher kernel | "Common" SFPU activations live in `compute_kernel_api.h`; host routes them through the **default** arm of `unary_op_utils::get_macro_definition`, which sets `SFPU_OP_COMPUTE_KERNEL_API_INCLUDE = 1`. emule's `sfpu_split_includes.h` shim must honor that guard (`#include "api/compute/compute_kernel_api.h"`). | One-time fix in emule's `sfpu_split_includes.h`; future "common" activations get pulled in for free. |

### Runtime PCC-failure patterns (JIT compile clean, but output is wrong)

Once the kernel compiles, the next failure surface is numerical. The bring-up cost of these is often higher than compile-time bugs because the symptom (PCC delta) is far from the cause.

| Symptom | Likely root cause | Strategy |
|---|---|---|
| PCC ≈ 0 with `Max ATOL Delta: nan` / `Max RTOL Delta: nan` | A primitive somewhere is producing NaN/inf that propagates downstream. Could be `rsqrt(0)`, division by an uninitialized DST element, or — for ops that chain fp32-intermediate CBs with bf16 inputs — a wrapper that picks a single format for both CBs. | Add `fprintf(stderr, …)` checkpoints at each compute primitive (reduce_tile, mul_unary_tile, rsqrt_tile, pack_tile, apply_bcast, etc.) flagging `is_nan`/`is_inf`. Single run; first checkpoint that fires is the source. **Strip the fprintf before committing.** |
| PCC < threshold but no NaN | Numerical precision drift. Common when bf16 accumulators in long reductions diverge from the test's PCC threshold (often 0.9999). | Check whether the kernel's reduce path is fp32-accumulated (look for `FLOAT32_REDUCTION` template arg upstream). emule's `reduce_tile` already accumulates in float; if drift is still high, look for bf16 round-trips in intermediate CBs. |
| Wrapper switches fp32/bf16 path on **one** input's format only | Mixed-format kernel chain (e.g. `apply_bcast(bf16_input, fp32_intermediate, …)` — layernorm's normalization step does this with the rsqrt output cb_ex2pe as fp32). The single-format branch reinterprets the other CB's bytes incorrectly. | Provide a per-CB element reader that dispatches on each CB's own `cb_is_32bit_format(cb_id)`. Call it twice per (icb0, icb1) read. DST is always fp32 in emule. The shared helper `__emule_read_cb_elem_at(cb, itile, ni)` in `common.h` is now the single source of truth — used by `add/sub/mul/matmul_tiles`, `reduce_tile`, and `apply_bcast`. |
| Instrumented `reduce_tile` never fires for a `compute_kernel_lib::reduce<SUM\|AVG, REDUCE_ROW>` call | `reduce_uses_matmul<>` (in `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp`) returns **true** for SUM/AVG along REDUCE_ROW. Those reduces lower to `matmul_tiles(input × col0-scaler-tile)`, NOT `reduce_tile`. | When chasing a missing reduce, instrument BOTH `reduce_tile` and `matmul_tiles`. Layernorm-family reduces (E[x], E[x²]) hit the matmul path. |
| PCC ≈ 0 with finite ATOL on a kernel that touches `ttnn.bfloat8_b` weights or activations | Bfp8_b (block-float-8) tiles are 1088 bytes = 64 face-row exponent bytes + 1024 mantissa bytes; emule needs explicit per-element decode + per-face-row encode. Without it, bf16/fp32 readers misinterpret the bytes. | Detect via `cb_is_bfp8_b_format(cb_id)` (`0 < page_size < 2048`). Decode: `__emule_bfp8::to_f32(buf, ni)`. Encode: `__emule_bfp8::encode_face_row(in16, exp_out, mant_out)`. Wire through the single shared `__emule_read_cb_elem_at` helper; `pack_dst_to_buf` needs its own Bfp8_b branch (cannot reuse the helper because pack iterates in nfaces order to share exponents). |
| **PCC = 0 with finite ATOL** on a ttnn pytest where the input is `ttnn.from_torch(...)` + `ttnn.to_device(..., MemoryConfig(INTERLEAVED, L1/DRAM))` on a non-trivial tensor (multi-tile, e.g. `[2,2,256,512]`) | **Host→device data loss.** The host write went somewhere other than where the kernel's `TensorAccessor(...).get_noc_addr(page_id)` + `noc_async_read` resolves to — so the reader copies zeros into the CB and downstream `copy_tile` loads zero DST. Confirmed on test_untilize during data_movement bring-up. | Two-checkpoint diagnosis: (1) fprintf inside `noc_async_read` after the memcpy, dumping `dst[0..3]` — if zero, the source DRAM region is zero. (2) Confirm `copy_tile(icb, 0, 0)` produces zero DST. **Action: skip the test (mark in running notes), don't try to fix from the kernel side.** This is an emule UMD/banking issue — outside the scope of single-op bring-up. The smaller test_clone parametrizations that pass exercise the same code path with single-tile shapes, so the bank-resolution gap likely manifests only for larger interleaved layouts. |

## Step 5 — Place the shim, mirroring tt-metal's path

Decide where the new shim lives by the **path the kernel uses to include it**:
- `#include "api/compute/X.h"` → `include/jit_hw/api/compute/X.h`
- `#include "ttnn/cpp/ttnn/kernel_lib/Y.hpp"` → `include/jit_hw/ttnn/cpp/ttnn/kernel_lib/Y.hpp`
- `#include "internal/Z.h"` → `include/jit_hw/internal/Z.h`

In the shim, write a header comment pointing at:
- The **real header path** in `/localdev/arminale/tt-metal/tt_metal/hw/inc/api/compute/` (or `tt_metal/tt-llk/tt_llk_wormhole_b0/` for true LLKs).
- What semantic the host replication captures vs. what was elided.
- For no-op shims: explicitly note "dead code in this JIT instantiation; if this op ever becomes the selected branch, downstream PCC will catch it — implement real semantics then."

**Pattern of preference**, top to bottom:
1. **One-line redirect** to an emule header that already exists.
2. **No-op shim** in the `ckernel` namespace — only for dead-code op-headers pulled in by fat-dispatcher kernels. The empty body parses; never executes.
3. **Reuse a real tt-metal header** by adding the right `#include "X"` to pull in macros it depends on (e.g. `ALWI` via `common.h`). Workspace rule "use real headers where possible" applies.
4. **Host-side reimplementation** of a kernel-lib helper that wraps LLKs the host can't model. Skip the LLKs and do the data work directly on emule's CB byte pool (see `__emule_compute::cb_read_ptr_at` / `cb_write_ptr_at` / `cb_page_size`).
5. **No-op stub** for a non-dead function whose semantics genuinely don't matter on host (e.g. `reconfig_data_format_*` — host doesn't model register data formats). Annotate the comment so future maintainers don't think it's a bug.

## Step 6 — Iterate

```bash
rm -rf /tmp/tt_emule_jit_* /tmp/tt_emule_src_* /tmp/tt_emule_jit_cache_$(id -u)
/opt/ttmlir-toolchain/venv/bin/pytest <path-to-test> -v -k '<variant>' --tb=short -s 2>&1 | \
  grep -E '^FAILED|^PASSED|error:|RuntimeError'
```

**No incremental rebuild needed for emule header edits.** Headers are pulled in by JIT compile every time; clearing the JIT cache is sufficient. Only editing `tt_metal/impl/emulation/emulated_program_runner.cpp` or other compiled tt-metal files requires `cmake --build build_emule`.

Stop when the test PASSES or when the failure mode is something other than `LLK_MISSING` (e.g. PCC delta, hang) — those need separate handling.

## Step 7 — Save bring-up notes

For each new op, write running notes to `docs/notes/<op>-bring-up.md` in the tt-emule repo (gitignored). Format:

```markdown
# Running notes: bring up <op> under tt-emule

## Test driving the bring-up
[path + parametrize + env]

## Variant the test exercises
[which program factory, which kernel files]

## Smoke runs
### smoke-N — YYYY-MM-DD HH:MM — exit code N — category X
[error excerpt]
[diagnosis]

### Iteration round N
[gaps found, fixes applied]
```

Write to the notes file *before* each rebuild, not after — captures intent even if the next run gets interrupted.

## Estimate of effort

Cost of a new op is **proportional to what's left in the long tail**, not the op's own algorithmic complexity:

| Op family | Cost | Why |
|---|---|---|
| First op in a brand-new family (e.g. tilize) | High — multiple iterations, 100-200 LoC of new host code, multiple shim headers. | Need to bring up the kernel-lib helpers + supporting dataflow free functions. |
| Subsequent op in an existing family (e.g. multiply after PR #22's eltwise binary infra) | Low — 1 iteration, 2-3 no-op shims, ~30 LoC total. | The infrastructure was already in place; only long-tail dead-code headers need patching. |
| Op whose kernel composes a chain (e.g. ScaledEmbedding = embedding + multiply) | Each op in the chain is its own bring-up. | Compose costs; pick the simplest member of the chain to start. |

## Worked example #1: `ttnn.embedding`

Test: `models/tt_transformers/tests/test_embedding.py[False]` (unscaled variant).
Variant: `embeddings_fused_program_factory.cpp` non-chunked.
Kernels: `embeddings_tilize.cpp` (reader), `tilize.cpp` (compute, from data_movement/tilize), `writer_unary_interleaved_start_id.cpp` (writer).

Three rounds:

**Round 1 — header path & macro shims:**
1. `include/jit_hw/api/compute/cb_api.h` (new, redirect to `api/cb_api.h`) — kernel-lib helper expected the `compute/` sub-path.
2. `include/jit_hw/internal/circular_buffer_interface.h` (new, redirect to `internal/llk_state.h`) — avoids `get_local_cb_interface` return-type conflict.
3. `include/jit_hw/api/compute/tilize.h` (edit) — pull in `common.h` + `compute_kernel_hw_startup.h` so `ALWI` and `compute_kernel_hw_startup` propagate.

**Round 2 — kernel-lib full shim + dataflow free templates:**
4. `include/jit_hw/ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (new) — full host shim. Replicates `compute_kernel_lib::tilize_config` enums; provides `compute_kernel_lib::tilize<...>(num_blocks)` as a host template doing row-major → tile-layout (face-major) rearrangement via `tilize_block_host(icb, n, ocb)` on the CB byte pool. Never references `fast_tilize_*`.
5. `include/jit_hw/api/dataflow/dataflow_api.h` (edit) — free-function template `get_noc_addr<AddrGen>(id, accessor, …)` and templated `noc_async_read<N>(…)` overload.

**Round 3 — result:** `test_embedding[False]` PASSES with PCC > 0.99. Total: 3 new files, 2 edits. ~150 LoC of new code + comments.

## Worked example #2: `ttnn.multiply` (scalar) via `eltwise_binary_sfpu_scalar.cpp`

Test: `models/tt_transformers/tests/test_embedding.py[True]` (scaled variant; chains `ttnn.embedding` → `ttnn.multiply`).
Kernel: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_binary_sfpu_scalar.cpp`.

This is a **fat dispatcher**: one source handles ~15 binary SFPU ops (multiply, add, sub, atan2, isclose, xlogy, …) with the selected op chosen via `BINARY_SFPU_OP` macro at JIT time. The kernel transitively `#include`s op headers for ALL ops. Headers emule already has (post-PR #22): `eltwise_binary_sfpu.h`, `binary_bitwise_sfpu.h`, `mul_int_sfpu.h`, `xlogy.h`, `quantization.h`, etc. Missing: `atan2.h`, `isclose.h`.

**One round, two shims:**
1. `include/jit_hw/api/compute/atan2.h` (new) — `ckernel`-namespace stubs for `atan2_binary_tile` + `_init`. No-op bodies. Real header at `tt_metal/hw/inc/api/compute/atan2.h`.
2. `include/jit_hw/api/compute/isclose.h` (new) — same: `isclose_binary_tile<bool EQUAL_NAN>` template + `_init`. No-op.

**Result:** `test_embedding[True]` PASSES, both variants now green. Total: 2 new files, ~30 LoC.

The drastic difference in cost vs. embedding (1 round vs 3, 30 LoC vs 150) is because the *infrastructure* was already in place. Bringing up a binary SFPU op for the first time would have been the heavy work; multiply was cheap because PR #22 already paid for `eltwise_binary_sfpu.h` et al.

## Worked example #3: `tt_transformers` RMSNorm (PREFILL) via `layernorm.cpp`

Test: `models/tt_transformers/tests/test_rms_norm.py[…-Mode.PREFILL-…]`. Kernel: `ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp` (RMSNORM section).

JIT compile was already clean after the earlier API-gap closure (the bcast wrappers + `reconfig_data_format` 4-arg overload from the previous bring-up). **PCC failed at runtime** with -0.02 and NaN in the output — the first PCC-failure case in the bring-up sequence.

**Investigation pattern (mirrors the new "Runtime PCC-failure patterns" table above):**

1. Added five `fprintf(stderr, …)` checkpoints in emule's compute primitives (`reduce_tile`, `mul_unary_tile`, `rsqrt_tile`, `pack_tile`, `apply_bcast`) flagging `is_nan`/`is_inf`. Plus a per-call log in `__emule_bf16::to_f32` when it returns NaN.
2. Single test run identified the first NaN at `apply_bcast` reading `cb_ex2pe` (the rsqrt output). `to_f32(0xFFD0) → NaN` — bf16 bit patterns in the NaN exponent range.
3. Confirmed `cb_ex2pe` was actually fp32 format at runtime (`page_size=4096`), but `apply_bcast` was reading it as bf16 because `icb0` was bf16 and the branch selector only checked `icb0`'s format.

**Fix** (8 LoC core + a 2-LoC helper):

```cpp
inline float __emule_read_cb_elem(uint32_t cb_id, uint8_t* page_base, uint32_t elem_idx) {
    if (__emule_compute::cb_is_32bit_format(cb_id)) {
        return reinterpret_cast<const float*>(page_base)[elem_idx];
    }
    return __emule_bf16::to_f32(reinterpret_cast<const uint16_t*>(page_base)[elem_idx]);
}
// apply_bcast now calls __emule_read_cb_elem(icb0, …) and __emule_read_cb_elem(icb1, …)
// per output position. DST is fp32, so the op composes in fp32 regardless of CB format.
```

Plus a permanent diagnostic guardrail in `pack_dst_to_buf`: abort loudly if `cb_page_size(ocb) == 0`. Doesn't fire on the rms_norm case (the real bug was format mismatch, not size=0) but is a cheap safety net for the next time a kernel writes to a misconfigured CB.

**Stripped instrumentation** before committing. The committed change is the fix + guardrail only.

**Result:** PREFILL PCC=0.9999963 (threshold 0.9999). Mode.DECODE still fails on `layernorm_sharded.cpp` JIT compile — separate bring-up.

**Total cost: 1 fix round, ~25 LoC across two files. Most of the time spent was in the NaN-bisection phase**, not the fix itself — reinforces the value of having all five fprintf checkpoints landed in one pass.

## Worked example #4: `ttnn.untilize` JIT compile via `untilize.cpp` (kernel-lib unblock)

Test: `tests/ttnn/unit_tests/operations/data_movement/test_untilize.py`. Kernel: `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp`, calls `compute_kernel_lib::untilize<...>` from `untilize_helpers.{hpp,inl}`.

First JIT-compile attempt surfaced four clang errors (one round, `-s` to drop `--forked` and read clang stderr):

| # | Error | Fix |
|---|---|---|
| 1+2 | `circular_buffer_interface.h: functions that differ only in their return type` and `dataflow_api.h: redefinition of 'CBInterface'` | Redirect shim at `include/jit_hw/internal/circular_buffer_interface.h` that includes emule's `llk_state.h`, preempting the upstream header. |
| 3 | `use of undeclared identifier 'compute_kernel_hw_startup'` | Pull `#include "jit_hw/api/compute/compute_kernel_hw_startup.h"` into emule's `untilize.h` / `pack_untilize.h`. The function was already defined in emule — the shim header just wasn't including it. |
| 4 | `untilize_helpers.inl:196: use of undeclared identifier 'pack_untilize_block'` | Add upstream-signature `pack_untilize_block<block_ct_dim, full_ct_dim>(icb, block_rt_dim, ocb, block_c_index=0)` mirroring `tt_metal/hw/inc/api/compute/pack_untilize.h:150`. Don't reuse `experimental::pack_untilize_block` — it has different args. |

**`pack_untilize_block` adapter math** (this is non-obvious): when adapting the upstream signature for emule's `__llk_pack_untilize`-based scatter, set `__llk_pack_block_c = full_ct_dim` (the FULL row width, not the per-pass `block_ct_dim`). Seed `__llk_pack_offset = r * full_ct_dim + block_c_index * block_ct_dim` per row. The existing `experimental::pack_untilize_block` resets offset to 0 because it's called once per full output reservation — but the upstream-shape variant can be called multiple times per row when sub-blocks are needed (`use_block_based_pack` path in `untilize_helpers.inl`).

**Total: 1 round, 3 new files / edits, ~50 LoC.** After Phase 1 lands, the same set of `untilize.cpp`-using tests (test_repeat ROW_MAJOR, test_concat, test_embedding, test_fill_pad families) JIT-compile — but PCC fails because the *input* CB sees zeros. That's a host→device data-loss issue (see the PCC-failure table above), not a compile-time gap.

## Common gotchas

- **One ttnn op composes multiple kernels** (reader + compute + writer). Each may fail independently. The first JIT error you see is *one* kernel in the pipeline; the next iteration may surface the next kernel.
- **One pytest test may compose multiple ttnn ops** (e.g. `ScaledEmbedding` = `embedding` + `multiply`). Each op is its own bring-up.
- **Fat dispatcher kernels pull in many op headers**. Don't be alarmed by a long #include list — most of them are dead code at JIT time, and no-op shims work.
- **`if constexpr` branch with template-parameter-dependent args**: discarded branches still need their unqualified names declared. The cleanest workaround is to bypass the upstream `.inl` and write your own template that only instantiates the host-applicable path.
- **`pytest --forked` hides compiler stderr** — drop it (use `-s` instead) when surfacing JIT errors. Re-enable `--forked` for the final PCC check (clean per-test process state).
- **Don't bury the LLK source pointer**. Every shim header should carry a comment with the real LLK source path so future-you (or another agent) can compare semantics.
- **Mixed-format CBs in compute-LLK wrappers**: don't pick a single fp32-vs-bf16 path based on one input. Some kernels chain ops where the *intermediate* CB is fp32 even though inputs are bf16 (e.g. layernorm's rsqrt output `cb_ex2pe`). Read each CB by its own declared format. The rms_norm bring-up burned half a debug round chasing a "NaN in bf16 from inf round-trip" theory before realizing the bf16 path was reading fp32 bytes.
- **Strip instrumentation before committing**. fprintf NaN-bisection scaffolding is essential for finding the bug but noise in the committed diff. Diagnostic guardrails (loud abort with a precise root-cause hint) are the exception — those stay.
- **Don't extrapolate a failure category from one instrumented test to others without reading their actual failure lines.** During the data_movement sweep, `test_untilize` was confirmed host→DRAM via `noc_async_read` instrumentation — and we bucketed test_repeat, test_stack, test_dropout, test_tosa_gather, etc. into the same "host→DRAM SKIP" pile without checking. Re-reading their `--tb=line` logs later showed they were *JIT compile errors on different kernels* (`repeat_last_dim_rm.cpp`, `clone/compute_kernel.cpp`, `concat/.../reader_concat_stick_layout_interleaved_start_id.cpp`) — i.e. straightforward Phase-2 shim work, not deferred-emulator territory. **Lesson**: always `tee` full pytest output to per-test logs (CLAUDE.md says this), and when classifying a test as 🔴 SKIP, cite the specific error line from its log. If the log only has summary counts, the call is still ⏳ TBD.

## Skill maintenance

After each new op bring-up, update this skill in place:
1. Add a "Worked example: `<op>`" subsection if the playbook needed adapting.
2. If a new gotcha shows up, append to "Common gotchas".
3. If a new failure category emerges, extend the classification table in Step 4.
4. If the bring-up was substantially different (e.g. SFPU op, reduction, all-gather, sharded matmul, collective communication), revise the affected sections.

Keep the skill as a single cohesive document. Don't add version markers ("v1", "v2", etc.) — the worked-examples list shows what's been covered, which is the more useful signal.
