# tt-emule ↔ tt-metal Integration

How the standalone emulator is injected into tt-metal so that real `ttnn` / D2M
workloads run on the host with no silicon. Read this before rebasing onto a new
tt-metal pin, adding a guard, or debugging the JIT runner.

For build/test mechanics see [BUILD_GUIDE.md](../BUILD_GUIDE.md); for the
top-level map see [IMPLEMENTATION_REPORT.md](../IMPLEMENTATION_REPORT.md).

---

The three design principles that govern this integration (zero fake headers,
memory isolation in `tt_emule::Core`, minimal `jit_hw/` surface) are stated in
[IMPLEMENTATION_REPORT.md § Design philosophy](../IMPLEMENTATION_REPORT.md#design-philosophy).
This doc is the mechanics.

---

> **Note on "Layer N".** The Layer 1–4 numbering below is the *build/integration*
> axis — how emule plugs into tt-metal. It is unrelated to the *kernel-API*
> layer-1 / 1.5 / 2 / 3 abstraction stack (what API a kernel targets); for that
> see [kernel-api-layers.md](kernel-api-layers.md).

## Layer 1 — UMD device injection: `SWEmuleChip`

`tt_metal/third_party/umd/device/{api/umd/device/chip/sw_emule_chip.hpp,
chip/sw_emule_chip.cpp}`. `SWEmuleChip : Chip` is memory-backed I/O; all
non-memory ops (barriers, resets, power, host channels) are no-ops.

```cpp
std::unique_ptr<tt_emule::L1Pool> worker_pool_;            // 2MB-aligned slots
std::unordered_map<tt_xy_pair, size_t> core_to_slot_;
std::unordered_map<tt_xy_pair, std::unique_ptr<tt_emule::Core>> cores_;
std::unordered_map<tt_xy_pair, uint32_t> dram_core_to_channel_;
uint32_t l1_size_; uint64_t dram_bank_size_;
```

- Builds a single `MAP_32BIT` `L1Pool` (slots = 2× the SoC Tensix count). Worker
  cores draw L1 from pool slots; DRAM cores use individual mmaps; pool exhaustion
  falls back to individual `MAP_32BIT` mmaps. (Standalone tt-emule, without the
  `TT_EMULE_USE_L1_POOL` gate, uses per-`Core` mmaps instead — see
  [l1-emulation.md](l1-emulation.md).)
- `get_core(tt_xy_pair)` lazy-creates a `Core` with the right role.
- `read_from_device` / `write_to_device` delegate uniformly to
  `get_core(xy)->l1_ptr(offset)` + `memcpy`.
- The header forward-declares `tt_emule::Core` / `L1Pool` (no UMD include
  contamination). Instantiated in `cluster.cpp` under `#ifdef TT_UMD_BUILD_EMULE`.

---

## Layer 2 — runtime activation

Both the build flag and the env vars are required:

| Knob | Effect |
|---|---|
| `-DTT_METAL_USE_EMULE=ON` (CMake) | compiles `emulated_program_runner.cpp` + `TT_METAL_USE_EMULE=1` in tt_metal/impl/llrt; propagates `TT_UMD_BUILD_EMULE=ON` (enables `SWEmuleChip`); adds `-rdynamic` to `libtt_metal.so` (to export TLS + bridge symbols to dlopen'd kernels) |
| `TT_METAL_EMULE_MODE=1` | selects `TargetDevice::Emule` + forces slow dispatch |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH=...` | SoC descriptor for topology — **required** (runtime throws if unset) |
| `TT_METAL_SLOW_DISPATCH_MODE=1` | required (no HW command queue in emulation) |

A single binary built with the flag supports both silicon and emulated runs; the
env var picks the path.

---

## Layer 3 — JIT kernel execution

`tt_metal/impl/emulation/emulated_program_runner.{hpp,cpp}` is the core
component.

**JIT compile pipeline.** Kernel `.cpp` → temp dir with `wrapper.cpp` (kernel
defines + `#include "jit_kernel_stubs.hpp"` + Metal 2.0 namespace blocks +
`#include <kernel>`). Compiled with the **build's configured compiler/standard**
(`TT_EMULE_CXX_COMPILER` / `TT_EMULE_CXX_STANDARD`, = `${CMAKE_CXX_COMPILER}` /
`${CMAKE_CXX_STANDARD}` — clang-20 / C++20 in the standard build) as
`-std=c++NN -fPIC -shared -O2`. Compile-time args via
`-DKERNEL_COMPILE_TIME_ARGS=...`; Metal 2.0 named bindings emitted inline by
`emit_metal2_namespaces()` (replacing upstream `genfiles.cpp`'s
`kernel_{args,bindings}_generated.h`). Then `dlopen` + `dlsym("__emule_kernel_entry")`.

Quasar source is regex-patched pre-compile: `mhartid`→`__processor_id`,
RISC-V `fence`→`__sync_synchronize()`, `reinterpret_cast<T*>(get_arg_val(N))` and
named-arg casts → routed through `__emule_local_l1_to_ptr`.

**JIT cache.** Keyed by `(source : compile_args : named_args : defines :
metal2_binding_suffix)`. Two tiers: an in-memory map **and a persistent on-disk
cache** at `/tmp/tt_emule_jit_cache_$UID` (override `TT_EMULE_JIT_CACHE_DIR`),
hashed + mtime-validated, so compiled `.so`s **survive across process runs**.
Misses compile in parallel.

**Execution — direct Core memory, no copies.** Collect kernels per logical core
from `ProgramImpl` → compile misses → resolve to function pointers → launch all
cores concurrently. Per core: `get_core(xy)`, init CB sync, init semaphores at
HAL-derived addresses, set the per-thread bridge TLS
(`__emule_bridge_l1`, `__emule_bridge_dram`, `__emule_cbs`, `__emule_core_map`;
Quasar adds `__emule_dfbs`, `__emule_tc_array`, `__emule_neo_id`,
`__emule_trisc_id`), launch reader+compute+writer (WH/BH) or per-DM/per-engine
threads (Quasar), join. No copy-back.

**Semaphore init.** `emule_sem_base = hal.get_dev_addr(TENSIX, KERNEL_CONFIG) +
prog_config.sem_offset`, passed as `EMULE_SEM_BASE`; each sem at
`emule_sem_base + sem_id*16`. Same values `finalize_sems()` produces for firmware.

**Memory bridge.** The dlopen'd `.so` reaches the host only through `extern "C"`
hooks (visible via `-rdynamic`): `__emule_dram_ptr`, `__emule_local_l1_ptr`,
`__emule_resolve_noc_addr`, `__emule_multicast_write` — plus the thread-local
bridge pointers. (The resolver is the live path; the single-bank `__emule_dram_ptr`
/ `__emule_local_l1_ptr` are legacy fast paths — see
[noc-emulation.md](noc-emulation.md) §8.) This avoids any ABI hazard from kernels
inlining `Device`/`Core` methods.

**DFB setup (Quasar).** L1 allocated per DFB via `Core::l1_alloc` (bridge DFBs
share backing), tile counters initialized `M=max(P,C)`, per-thread
`EmuleDFBInterface` built for STRIDED/BLOCKED. See
[DFB_EMULATION.md](DFB_EMULATION.md) §5.5.

---

## Layer 4 — dispatch interception

In `tt_metal.cpp`, `LaunchProgram()` branches under `#ifdef TT_METAL_USE_EMULE`:
when `get_target_device_type() == TargetDevice::Emule`, call
`emule::execute_program_emulated(device, program)`; otherwise the real dispatch
path. `CompileProgram` / `WriteRuntimeArgsToDevice` / `finalize_offsets()` still
run (they populate `ProgramImpl` + `ProgramConfig.sem_offset`); only HW dispatch
is bypassed. A second interception in `ConfigureDeviceWithProgram()` skips binary
writing.

---

## Guard pattern: `is_mock_or_emulated()`

Defined in `tt_cluster.hpp` (`target_type_ == Mock || Emule`), it short-circuits
HW-specific operations across subsystems — dispatch queues, RISC firmware init,
context init (fabric/FW/watcher/dprint), program/device init, fabric control
plane, buffer/event dispatch. New HW-dependent paths that need guarding announce
themselves by crashing loudly in emulated mode rather than corrupting silently.

---

## tt-metal files touched

| Kind | Files |
|---|---|
| New: program runner | `tt_metal/impl/emulation/emulated_program_runner.{hpp,cpp}` |
| New: UMD chip | `.../umd/device/.../chip/sw_emule_chip.{hpp,cpp}` |
| New: tests | `tt_emule/` C++ + ttnn test trees |
| Modified: dispatch | `tt_metal.cpp` (`#ifdef TT_METAL_USE_EMULE` branches) |
| Modified: guards | `is_mock_or_emulated()` call sites in `impl/`, `llrt/`, `umd/` |
| Modified: enums / build | `tt_cluster.hpp`, UMD types, top-level + UMD CMake |

---

## Rebase surface

The integration is concentrated in: `tt_metal/impl/emulation/` (runner),
`SWEmuleChip` (UMD), the one `tt_metal.cpp` dispatch branch, and the
`is_mock_or_emulated()` one-liner guards. Rebasing onto a new tt-metal pin
primarily means (1) checking guards still cover any new HW-dependent paths,
(2) updating `jit_hw/` stubs if kernel APIs changed, (3) verifying `ProgramImpl`
/ `CircularBufferConfig` / `ComputeConfig` interfaces are unchanged.
