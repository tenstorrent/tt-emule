# noc_mode divergence: emule forces DM_DYNAMIC_NOC, silicon defaults to DM_DEDICATED_NOC

**Status:** known bug, fix landing alongside the Metal 2.0 data-movement uplift.
**Severity:** serious — emule runs every kernel in a different NOC mode than silicon would.

## The bug

`include/jit_hw/jit_kernel_stubs.hpp` hard-codes, for **every** JIT kernel:

```cpp
inline constexpr int noc_mode = DM_DYNAMIC_NOC;   // = 1
```

But silicon's host picks `noc_mode` **per kernel type**, defaulting to `DM_DEDICATED_NOC (0)`.
From `tt_metal/impl/kernels/kernel.cpp::process_defines`:

| Kernel type | `NOC_MODE` the host emits |
|---|---|
| Compute | `DM_DEDICATED_NOC` (0) |
| DRAM | `DM_DEDICATED_NOC` (0) |
| DataMovement / Ethernet | `config_.noc_mode` (default 0; 1 only if the kernel was explicitly created dynamic) |

`enum NOC_MODE { DM_DEDICATED_NOC = 0, DM_DYNAMIC_NOC = 1 }` (`tt_metal/api/tt-metalium/kernel_types.hpp`).

So emule's `noc_mode` is wrong (1 vs 0) for essentially every kernel it runs.

### Internal inconsistency (smell that exposed it)
The JIT wrapper carries **both** a `NOC_MODE` *macro* (emitted by the host — `0`/DEDICATED) and the
`noc_mode` *variable* (emule's stub — `1`/DYNAMIC). Code keyed on the macro (e.g. NOC command-buffer
selection in `dataflow_cmd_bufs.h`) sees DEDICATED; code keyed on the variable (the `static_assert`s
below) sees DYNAMIC. The two disagree inside the same translation unit.

## Why it has mostly not bitten (yet)
- emule's NOC is a host `memcpy`; the command-buffer constants `noc_mode`/`NOC_MODE` select are inert
  in emulation, so the wrong mode usually has no runtime effect.
- The only kernels that *require* DYNAMIC are out of emule's WH/BH scope:
  - `models/demos/deepseek_v3_b1/unified_kernels/broadcast.hpp:202` — `static_assert(noc_mode == DM_DYNAMIC_NOC)`
  - `models/demos/deepseek_v3_b1/unified_kernels/flash_mla.hpp:26` — same
- Conversely, **in-scope / core** code asserts the *opposite* and would mis-compile under DYNAMIC:
  `tt_metal/hw/inc/api/dataflow/noc.h:644`, `tt_metal/fabric/.../fabric_erisc_router.cpp:2921`, and
  ~50 Quasar `noc_nonblocking_api_v{1,2}.h` sites — all `static_assert(noc_mode != DM_DYNAMIC_NOC)`.

So the historical DYNAMIC hard-code was a workaround for some op (now only deepseek) at the cost of
diverging from silicon everywhere else. It violates the project rule: be faithful to silicon, no
parallel/divergent code paths.

## The fix
Make emule track the host-emitted macros, mirroring the real firmware header
(`tt_metal/hw/inc/internal/dataflow/dataflow_api_common.h`, `KERNEL_BUILD` branch:
`noc_index = NOC_INDEX; noc_mode = NOC_MODE;`). In `jit_kernel_stubs.hpp`:

```cpp
#ifndef NOC_INDEX
#define NOC_INDEX 0     // ComputeKernel::process_defines omits NOC_INDEX; emule uses NOC 0
#endif
constexpr uint8_t noc_index = NOC_INDEX;
constexpr uint8_t noc_mode  = NOC_MODE;   // DEDICATED by default; per-kernel for DM/eth
```

Safe for WH/BH scope (only out-of-scope deepseek needs DYNAMIC). Verified by the full WH+BH C++ +
ttnn regression staying green (no kernel in scope asserts `== DM_DYNAMIC_NOC`).

## Related divergence: noc_index pinned to 0 (emule NOC-1 unsupported)

`jit_kernel_stubs.hpp` also hardcoded `noc_index = 0` for every kernel, where silicon emits
`NOC_INDEX` per kernel (`config_.noc` for DataMovement/Ethernet — 0 **or** 1). Making `noc_index`
faithful (`= NOC_INDEX`) was tried and **reverted**: it corrupts data for any kernel assigned NOC 1.
Isolation proof: with `noc_index = 0` the simple tiled L1 write→CB→read test passes; with
`noc_index = NOC_INDEX` it fails with a data mismatch — `noc_index` is the only variable.

Root cause: **emule's non-zero-NOC address resolution is not correct.** NOC 1 uses a flipped
coordinate system on silicon; emule's per-NOC tables (`my_x/my_y[noc]`, `*_bank_to_noc_xy[noc]`) and
the `get_noc_addr(..., noc=1)` resolution path don't produce the right host address. The
long-standing `noc_index = 0` hardcode silently forced every kernel onto NOC 0, masking this — emule
has effectively never exercised NOC 1.

**Decision:** `noc_index` stays pinned to 0 for now (documented divergence). Making it faithful is
the real "NOC index sanitization" work and is **blocked on fixing emule's NOC-1 resolution**
(`emulated_program_runner.cpp` per-NOC coordinate/bank-table population + the resolve path) —
tracked as a follow-up. `noc_mode` (above) is fixed now because it is safe; `noc_index` is not.

## Related
This was surfaced during the Metal 2.0 data-movement uplift. The same `jit_kernel_stubs.hpp`
definitions also collide with the real `dataflow_api_common.h` for Metal 2.0 DM kernels (a separate
redefinition fix). The 8 DM/DRAM tests additionally need Metal 2.0 `SetProgramRunArgs` runtime-arg
support in `emulated_program_runner.cpp` (tracked separately). See `project_umd_untangle` memory.
