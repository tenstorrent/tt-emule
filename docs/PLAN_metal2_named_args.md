# Plan: Metal 2.0 Named-Arg Support in the Emulator

## Context

The bbradel branch `bbradel-41067_generic_reduce_host20` migrates the W-axis reduction (and later all reductions) to the Metal 2.0 host API. The new compute and dataflow kernel sources reference two auto-generated headers:

```cpp
#include "experimental/dataflow_buffer.h"
const uint32_t Ht = get_arg(args::Ht);              // RTA via accessor
constexpr uint32_t Wt = get_arg(args::Wt);          // CTA via accessor
experimental::DataflowBuffer input_buf(dfb::input); // DFB by name
```

`args::Ht`, `args::Wt`, `dfb::input`, etc. come from per-kernel auto-generated headers that tt-metal's `jit_build/genfiles.cpp` emits at compile time:

- `kernel_args_generated.h` — emits `args::*` accessors (`RtaArg<T>`, `CrtaArg<T>`, `CtaVal<T>`) for named RTA/CRTA/CTA bindings, plus `get_vararg`/`get_common_vararg` helpers.
- `kernel_bindings_generated.h` — emits `dfb::*` `DFBAccessor` constants and `sem::*` semaphore IDs.

Both are written by `jit_build_genfiles_kernel_include` / `jit_build_genfiles_triscs_src` only when `JitBuildSettings::is_metal2_kernel()` returns true. Legacy kernels skip both.

The emulator's JIT pipeline (`emulated_program_runner.cpp`) **does not** go through tt-metal's genfiles path — it writes its own wrapper that pulls in the kernel source plus `jit_kernel_stubs.hpp`, then calls g++ directly. So today, a Metal 2.0 kernel with `#include "kernel_args_generated.h"` fails to compile in the emulator with "file not found" — and even if it did, the `experimental::RtaArg<T>` types referenced by the generated header are not defined anywhere in `jit_hw/`.

**Verified on `arminale/emule-metal-base` @ `8711ac3d0b`:** the W-reduction test (`SumTensorLastDimTests/...SumTensorCorrectly/1`) compiles and (per the regression that's running now) is expected to pass, because the kernels on this base still use the legacy positional API. The gap only matters once we move to the bbradel tip.

## Required Changes

### 1. Add `include/jit_hw/experimental/kernel_args.h` (new file)

Port verbatim from upstream `tt_metal/hw/inc/experimental/kernel_args.h`. The header is self-contained (~70 lines) — defines:

- `template <typename T> struct experimental::RtaArg<T> { uint32_t byte_offset; };`
- `template <typename T> struct experimental::CrtaArg<T> { uint32_t byte_offset; };`
- `template <typename T> struct experimental::CtaVal<T> { T value; };`
- Three `get_arg()` overloads:
  - `RtaArg<T>` → `*((tt_l1_ptr T*)get_arg_addr(byte_offset / 4))`
  - `CrtaArg<T>` → `*((tt_l1_ptr T*)get_common_arg_addr(byte_offset / 4))`
  - `CtaVal<T>` → returns the constexpr value

Dependencies (`get_arg_addr`, `get_common_arg_addr`, `tt_l1_ptr`, `FORCE_INLINE`, `static_assert`) all already exist in tt-emule's `jit_kernel_stubs.hpp` / `risc_attribs.h`. **No emulator-specific adaptation needed** — same semantics as upstream because the underlying `get_arg_val<T>(idx)` already returns a host-pointer-translated value.

### 2. Hook the genfiles emit into the emulator JIT pipeline (modify `emulated_program_runner.cpp`)

The existing flow per Metal 2.0 kernel:

1. After `jit_temp_dir` is created (existing code) and before the wrapper is written, check `kernel->jit_build_settings()->is_metal2_kernel()`.
2. If Metal 2.0, call the **already-public** functions from `tt_metal/jit_build/genfiles.cpp`:
   ```cpp
   tt::tt_metal::write_kernel_bindings_generated_header(jit_temp_dir, *settings);
   tt::tt_metal::write_kernel_args_generated_header(jit_temp_dir, *settings);
   ```
3. Add `-I<jit_temp_dir>` to the g++ command line so the wrapper's `#include "kernel_args_generated.h"` resolves.

Both functions are file-scope `void` functions in `genfiles.cpp` — the second is currently anonymous-namespace (`namespace { ... }`), so we either:
- (a) Move it to file scope + declare in `tt_metal/jit_build/genfiles.hpp` (one-line tt-metal patch), or
- (b) Re-implement the ~70 lines locally in the emulator.

(a) is preferred — keeps the emit logic single-sourced. Tiny upstream patch (and emule's intent is to use real tt-metal headers).

### 3. Wrapper include order

The emulator's wrapper currently looks like:
```cpp
#include "jit_kernel_stubs.hpp"
#include "<kernel.cpp>"
```

For Metal 2.0 kernels it must become:
```cpp
#include "jit_kernel_stubs.hpp"
#include "kernel_bindings_generated.h"   // dfb::, sem::
#include "kernel_args_generated.h"       // args::, get_vararg helpers
#include "<kernel.cpp>"
```

The bindings/args headers live in `<jit_temp_dir>/`, so the `-I<jit_temp_dir>` from step 2 makes them resolvable.

### 4. Make `experimental::DataflowBuffer(DFBAccessor)` ctor work

The reduce_metal2 kernels construct DFB objects via the accessor constant: `experimental::DataflowBuffer input_buf(dfb::input);`. tt-emule already has `experimental::DataflowBuffer` in `include/jit_hw/experimental/dataflow_buffer.h`, but it currently takes a raw `uint16_t id`, not an `experimental::DFBAccessor`.

Either:
- (a) Add an overload in the existing emule `DataflowBuffer` ctor: `DataflowBuffer(experimental::DFBAccessor a) : id_(a.id) {}`, or
- (b) Verify the upstream `DFBAccessor` already provides an implicit conversion to `uint16_t` (likely; check upstream `experimental/dataflow_buffer.h` on bbradel).

Cost: one constructor overload, ~3 lines.

## Out-of-Scope

- **Host-side Metal 2.0 API** (`ProgramSpec`, `KernelSpec`, `MakeProgramFromSpec`, `SetProgramRunParameters`): tt-emule links the real `Metalium::Metal` library, so these already work — they end up populating standard `tt_metal::Program` and `Kernel` objects that the emulated runner consumes via the same `LaunchProgram` hook used today. No emulator changes needed for the host path.
- **Vararg helpers** (`get_vararg`, `get_common_vararg`): emitted as plain `FORCE_INLINE uint32_t get_vararg(uint32_t idx) { return get_arg_val<uint32_t>(N + idx); }` by the upstream genfile emitter. Needs no emulator-specific code — once `kernel_args_generated.h` is in the include path, vararg calls resolve through the existing `get_arg_val<T>` stub.

## Verification

1. **Tier 5b passes on `arminale/emule-metal-base`** (current pin). This is the regression that's running right now — confirms the legacy reduce path still works after the base bump.
2. **Bump `arminale/emule-metal-base` to bbradel-tip locally** (or check out the bbradel branch in tt-metal-main), rebuild tt-metal + tt-emule, re-run Tier 5b. Expect:
   - Without the changes in this plan: g++ JIT compile error on `kernel_args_generated.h`.
   - With the changes: test passes; PCC matches the merge-base run.
3. **Add a minimal regression-time check** in `run_regression.sh`: a test that uses a Metal 2.0 kernel with at least one named RTA, one named CTA, and one DFB binding. The W-reduce test covers all three; no extra wiring needed.

## Critical Files

- **New:** `include/jit_hw/experimental/kernel_args.h` (tt-emule)
- **Modified:** `tt_metal/impl/emulation/emulated_program_runner.cpp` (tt-metal-main) — Metal 2.0 detection + genfiles emit + `-I` flag
- **Modified:** `tt_metal/jit_build/genfiles.hpp` (tt-metal-main) — declare `write_kernel_args_generated_header` / `write_kernel_bindings_generated_header` (or move out of anonymous namespace)
- **Possibly modified:** `include/jit_hw/experimental/dataflow_buffer.h` (tt-emule) — `DFBAccessor` ctor overload

## Estimated Effort

- Header port: 5 minutes (copy + license header)
- Emulator runner hook: 1–2 hours (find the right insertion point, add Metal 2.0 detection, wire the include path, retest)
- Constructor overload: 5 minutes
- Verification rebuild + targeted regression: ~30 minutes

**Total: ~2–3 hours of focused work**, single-session.
