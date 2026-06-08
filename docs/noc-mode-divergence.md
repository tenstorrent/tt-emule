# noc_mode divergence: emule forces DM_DYNAMIC_NOC, silicon defaults to DM_DEDICATED_NOC

**Status:** fixed during the Metal 2.0 data-movement uplift — `noc_mode` and `noc_index` are both
faithful now (see "The fix" and "Resolved: noc_index is now faithful" below).
**Severity:** was serious — emule ran every kernel in a different NOC mode than silicon would.

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

## Resolved: noc_index is now faithful (was pinned to 0)

`jit_kernel_stubs.hpp` previously hardcoded `noc_index = 0` for every kernel, where silicon emits
`NOC_INDEX` per kernel (`config_.noc` for DataMovement/Ethernet — 0 **or** 1; BRISC→NOC0, NCRISC→NOC1).
It is now faithful (`= NOC_INDEX`, with the `#ifndef NOC_INDEX → 0` fallback for compute wrappers).

Earlier attempts at faithful `noc_index` corrupted data for kernels on NOC 1, which was mis-diagnosed
twice before the real cause was found:
- **Not** a coordinate-flip problem. `get_noc_addr(x,y,addr,noc)` is NOC-index-independent on both
  arches and `my_x[1]==my_x[0]` is faithful (coordinate virtualization) — emule must **not** add a
  `noc1 = size-1-x` flip.
- **Not** unpopulated `[1]` bank-table rows. Both `dram_bank_to_noc_xy[1]` and `l1_bank_to_noc_xy[1]`
  *were* written in `emulated_program_runner.cpp`.

**Actual root cause — a C ABI array-stride mismatch.** The runner *defines* the tables as
`uint16_t dram_bank_to_noc_xy[NUM_NOCS][MAX_NUM_BANKS]` with `MAX_NUM_BANKS = 256`, but the JIT kernel
(and the real firmware `dataflow_api_common.h`) *declare* them `[2][NUM_DRAM_BANKS]` / `[2][NUM_L1_BANKS]`,
where the JIT defines `NUM_*_BANKS` equal the **actual** bank counts (12 / 64). So the `[noc][bank]`
row stride is **256 on the write side but 12/64 on the read side**. Row 0 (`noc=0`) is at offset 0 in
both layouts → fine, which is exactly why the `noc_index=0` hardcode masked the bug. Row 1 (`noc=1`)
is written at offset 256 but read at offset 12/64 → the kernel reads **uninitialized zeros** → encodes
coord `(0,0)` → which is a DRAM core → resolves to the wrong backing (a DRAM mmap for an L1 write, or
the wrong channel for DRAM). That corrupted every `noc=1` bank access (the tiled L1 test, the DRAM-channel
test, and the ttnn reduce kernels).

**Fix:** lay the tables out flat with the **actual-count stride** (`tbl[noc*num_banks + bank]`),
matching both the kernel's `[2][NUM_*_BANKS]` view and silicon's real `std::vector<uint16_t>` layout
(`risc_firmware_initializer.cpp`, `[noc*num_banks + bank]`). Verified faithful `noc_index` green on
**WH C++ 39/0 + BH C++ 19/0**.

Two companion DRAM fixes landed with it (needed for Blackhole, where `TRANSLATED != NOC0` for DRAM),
both in tt-umd `sw_emule_chip.cpp`: (1) one physical backing per DRAM **channel** so all per-NOC
endpoint coords of a channel alias one mmap (`dram_channel_core_`); (2) classify DRAM cores in **both**
NOC0 and TRANSLATED coord forms so `is_dram_core()` succeeds regardless of which system reaches emule.

## Related
This was surfaced during the Metal 2.0 data-movement uplift. The same `jit_kernel_stubs.hpp`
definitions also collide with the real `dataflow_api_common.h` for Metal 2.0 DM kernels (a separate
redefinition fix, via an empty interception shim). The 8 DM/DRAM tests' segfault was the
experimental-NOC `UnicastEndpoint` local-source resolving to NOC core (0,0) instead of the issuing
core's own L1; fixed in `api/dataflow/endpoints.h` by branching the endpoint traits on `AddressType`
(LOCAL_L1 → `my_x/my_y[noc]`). See `project_umd_untangle` memory.
