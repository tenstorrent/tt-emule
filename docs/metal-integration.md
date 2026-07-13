# tt-emule ↔ tt-metal Integration

How tt-emule is injected into tt-metal so that real `ttnn` / D2M workloads run
on the host with no silicon. Read this before rebasing onto a new tt-metal pin,
adding a guard, or debugging the JIT runner.

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

- Builds a single `L1Pool` (one slot per SoC Tensix core) from a plain 64-bit
  `mmap`. Worker cores draw L1 from pool slots; DRAM cores use individual mmaps;
  pool exhaustion falls back to an individual mmap. See [l1-emulation.md](l1-emulation.md).
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

**Execution — fiber engine, direct Core memory, no copies.** `prepare_program`
resolves a program ONCE (collect kernels per logical core → compile misses →
resolve to function pointers), memoized by `ProgramId` in `g_resolved_programs`
(emule's analogue of `is_compiled()`; single-writer, written only on the
sequential dispatch path). Each kernel runs as a **cooperatively-scheduled fiber**
— one per (core, RISC) — multiplexed onto a runtime-sized worker pool
(`TT_EMULE_FIBER_WORKERS`); a blocked fiber parks (yields its worker) instead of
blocking an OS thread. Identity + bridge pointers live in the per-fiber ctx
`__emule_self` (`bridge_l1`, `bridge_dram`, `cbs`, `core_map`, `chip_id`; Quasar
adds `dfbs`, `tc_array`, `neo_id`, `trisc_id`). No copy-back. See
[fiber-engine.md](fiber-engine.md).

**Multi-chip (fabric / CCL).** A fabric send is intercepted at the worker→fabric
client API in the `jit_hw` shim and **teleported**: resolve the final destination
chip(s) from the control-plane topology, apply the terminal NOC command directly
into that chip's L1, wake any parked fiber there. No ERISC router / multi-hop. The
mesh runs all chips' fibers concurrently via a register/run split
(`begin_mesh_dispatch` defers each device's spawn; `run_mesh_dispatch` drives one
`run_until_idle`). See [fabric-ccl-emulation.md](fabric-ccl-emulation.md).

**Semaphore init.** `emule_sem_base = hal.get_dev_addr(TENSIX, KERNEL_CONFIG) +
prog_config.sem_offset`, passed as `EMULE_SEM_BASE`; each sem at
`emule_sem_base + sem_id*16`. Same values `finalize_sems()` produces for firmware.

**Memory bridge.** The dlopen'd `.so` reaches the host only through `extern "C"`
hooks (visible via `-rdynamic`): the resolvers `__emule_resolve_noc_addr`,
`__emule_multicast_write`, `__emule_dram_ptr`, `__emule_local_l1_ptr`, the fabric
hooks (`__emule_fabric_teleport`, `__emule_fabric_resolve_remote`,
`__emule_chip_relative_l1`, …), and the fiber thunks (`__emule_fiber_park/wake/…`).
All read `__emule_self`; since every hook runs inside a kernel fiber, a null
`__emule_self` is a contract violation and they fail loudly (`emule_require_self`)
rather than returning nullptr. (The single-bank `__emule_dram_ptr` /
`__emule_local_l1_ptr` are legacy fast paths — see
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
writing. `DispatchCompiledProgramToDevice()` carries the same emule branch (the
2nd..Nth device of a mesh coord range), running synchronously and skipping the
`is_finalized`/`is_compiled` asserts that never hold under emule's lazy JIT.

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
