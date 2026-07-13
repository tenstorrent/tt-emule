# tt-emule: Software-Emulated Tenstorrent Device

tt-emule is a standalone C++ software emulator for Tenstorrent device-level APIs.
It models the multi-core execution model — per-core L1 SRAM, banked DRAM,
circular-buffer synchronization, NOC communication, semaphores, and the DEST
register file — entirely on the host CPU, so kernels can be developed, run, and
debugged without silicon. It plugs into tt-metal at the UMD boundary, letting
real `ttnn` / D2M workloads execute unmodified on the host.

This document is the **entry point and doc index**. Each subsystem has its own
focused reference; start here and follow the links.

---

## Documentation index

**Memory**
- [l1-emulation.md](docs/l1-emulation.md) — per-core L1 SRAM, host-pointer
  convention, address translation, L1Pool vs bridge_l1.
- [dram-emulation.md](docs/dram-emulation.md) — banked/interleaved DRAM, bank
  topology from the SoC descriptor, interleaved addr-gen, non-pow2 banks.
- [mem-zeros-handling.md](docs/mem-zeros-handling.md) — the MEM_ZEROS L1 region
  and the zero-init contract.

**Compute**
- [dest-emulation.md](docs/dest-emulation.md) — the DEST register file, acquire/
  pack lifecycle, `DEST_AUTO_LIMIT` capacity modes.
- [cb-emulation.md](docs/cb-emulation.md) — circular-buffer FIFO sync, the
  kernel CB API, DFB↔CB bridge.
- [cb-dataformat.md](docs/cb-dataformat.md) — per-CB data-format dispatch
  (bf16 vs uint16 vs block-float).
- [tilize-untilize-pack.md](docs/tilize-untilize-pack.md) — UNPACK/PACK, nfaces↔
  row-major conversion, tilize/untilize/pack-untilize.

**Kernel API**
- [kernel-api-layers.md](docs/kernel-api-layers.md) — the layer-1 / 1.5 / 2 / 3
  kernel-API abstraction stack and which rung emule intercepts; how to choose a
  bring-up strategy by the layer a kernel bottoms out at.
- [sfpu-deep-path.md](docs/sfpu-deep-path.md) — optional deeper-than-layer-1 SFPU
  path: compile/run the real silicon `ckernel_sfpu_<op>.h` on emule's faithful
  `sfpi` backend (lane-masked registers, cregs, sign-magnitude cast); engaged per
  op via `EMULE_DEEP_SFPU_<OP>`.

**Execution model**
- [fiber-engine.md](docs/fiber-engine.md) — the cooperative M:N ucontext fiber
  scheduler (`TT_EMULE_FIBER_WORKERS`), persistent worker pool + `min(K,fibers)`
  clamp, register/run split, and §9 multichip co-scheduling.
- [state-tiers.md](docs/state-tiers.md) — per-chip / per-core / per-fiber state
  partition (the fiber ThreadCommonCtx / ComputeThreadCtx and what stays shared).

**Communication**
- [noc-emulation.md](docs/noc-emulation.md) — NOC address encoding/resolution,
  async read/write, multicast, semaphore operations, per-NOC state.
- [DFB_EMULATION.md](docs/DFB_EMULATION.md) — Quasar Dataflow Buffers (MPMC tile
  counters, STRIDED/BLOCKED).
- [fabric-ccl-emulation.md](docs/fabric-ccl-emulation.md) — inter-chip fabric
  (host-inject teleport transport) and how CCL collectives (all_gather,
  point_to_point, …) run on a 2-chip mesh.

**Architecture**
- [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md) — Quasar (Tensix Neo) threading,
  CSR emulation, full reference (incl. §8 feature-coverage table).
- [QUASAR_MATMUL_REPORT.md](docs/QUASAR_MATMUL_REPORT.md) — Quasar matmul
  bring-up (point-in-time).

**Runtime semantics**
- [riscv-intdiv-by-zero.md](docs/riscv-intdiv-by-zero.md) — RISC-V integer
  divide-by-zero behavior emule honors (quotient = all-ones, remainder =
  dividend); generic RISC-V semantics, not Quasar-specific.

**CI / testing**
- [post-commit-sweep.md](docs/post-commit-sweep.md) — the nightly post-commit
  pass-rate sweep: runs tt-metal's per-arch post-commit ttnn lane under emule
  and reports a growing pass-rate metric (not a gate). Manifest-entry
  granularity, the pytest-9.0.3 collection-error pre-filter, fleet policy
  (#189), local reproduction (`scripts/post_commit_sweep/`).

**Integration & build**
- [metal-integration.md](docs/metal-integration.md) — how emule injects into
  tt-metal (UMD chip, runtime activation, JIT runner, guards).
- [BUILD_GUIDE.md](BUILD_GUIDE.md) / [GETTING_STARTED.md](GETTING_STARTED.md) —
  build + test setup.
- [.claude/references/structure.yaml](.claude/references/structure.yaml) — authoritative file-level index of `src/` +
  `include/` and the top-level symbols in each.
- [D2M_REGRESSION_REPORT.md](docs/historical_reports/D2M_REGRESSION_REPORT.md) — D2M failure analysis.

---

## Design philosophy

The integration follows three principles — they are the reason the emulator stays
faithful and low-maintenance, and they govern every change:

1. **Zero fake headers.** Tests link the real `Metalium::Metal` library and use
   real tt-metal headers, fixtures, and buffer utilities. Emulation is injected at
   the UMD (User-Mode Driver) boundary, not via mock headers. A tt-metal header
   change surfaces immediately as a compile error in emulated tests, rather than
   silently diverging behind a fake-header layer.

2. **Memory isolation in `tt_emule::Core`.** All device memory — worker L1 and
   DRAM banks — is owned by `tt_emule::Core` objects inside `SWEmuleChip`. There
   is no intermediate copy-in / copy-out stage: the program runner points its
   bridge pointers straight at Core's mmap'd memory, and that memory persists
   across program runs, matching hardware semantics.

3. **Minimal API surface in `jit_hw/`.** Stub headers implement only what the
   target kernel needs to compile and execute correctly. Where possible, real
   tt-metal headers are included rather than duplicated — keeping the shim surface
   (and the rebase burden) as small as possible.

---

## Architecture at a glance

**Threading model (per arch).**

- **Wormhole / Blackhole** — each emulated core runs three host threads mirroring
  hardware: a NOC reader, a compute thread, and a NOC writer. They synchronize
  through circular buffers ([cb-emulation.md](docs/cb-emulation.md)).
- **Quasar (Tensix Neo)** — each Neo runs up to 12 threads (up to 8 DM + up to 4
  compute) sharing one larger L1, with Dataflow Buffers
  ([DFB_EMULATION.md](docs/DFB_EMULATION.md)) replacing SPSC CBs and CSR reads
  regex-patched to thread-locals. See [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md).

**Shared.** All device memory is owned by `tt_emule::Core` objects (worker L1 and
DRAM banks alike) — the runner uses that mmap'd memory directly, no copy-in/out,
persisting across program runs. The compute thread operates on a private DEST
register file modeled as `float[16][1024]` (16 bf16 / 8 fp32 slots) with
mode-aware bounds checking ([dest-emulation.md](docs/dest-emulation.md)). Cores
run concurrently in separate threads, so cross-core multicast and semaphore
signaling exercise real concurrency.

**How it runs inside tt-metal.** A single dispatch branch in `tt_metal.cpp`
(under `#ifdef TT_METAL_USE_EMULE`) routes to `execute_program_emulated`, which
JIT-compiles each kernel `.cpp` to a `.so` (via the build's configured C++
compiler), `dlopen`s it, and launches the per-core threads against
`SWEmuleChip`-owned memory. Full detail in
[metal-integration.md](docs/metal-integration.md).

**Codebase layout.** Host-side types live in `include/tt_emule/` (`Core`,
`L1Pool`, `CBSyncState`, tile-counter/DFB types) and are consumed header-only by
tt-metal and UMD; the JIT kernel surface is in `include/jit_hw/` (`api/compute/`,
`api/dataflow/`, `api/tensor/`, `internal/`, `experimental/`). For the
authoritative file/symbol index see [.claude/references/structure.yaml](.claude/references/structure.yaml).

---

## Test surface

Authoritative pass/fail state lives outside this report so it cannot drift:

- **C++ regression** — per-arch `scripts/run_regression_<arch>.sh` (CI matrix:
  wormhole/blackhole/quasar), cross-checked by `classify-results.py`. Wormhole and
  Blackhole are **zero-tolerance** (no allowlist — any failure fails the PR);
  Quasar's expected failures are listed in `.github/known-failures-quasar.txt`.
- **ttnn pytest** — per-arch `scripts/run_ttnn_pytests_<arch>.sh` (CI matrix:
  wormhole/blackhole).
- **D2M golden tests** — `run_d2m_regression.sh`; analysis in
  [D2M_REGRESSION_REPORT.md](docs/historical_reports/D2M_REGRESSION_REPORT.md).
- **Quasar feature coverage** — [QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md) §8.

---

## Strategic assessment

**Strengths**
- *Develop without hardware* — full kernel compile + multi-core run on any x86
  Linux host in seconds.
- *Zero fake headers* — links real `Metalium::Metal`; tt-metal header changes
  surface as compile errors, not silent drift.
- *No copy overhead* — `tt_emule::Core` is the single owner of L1 + DRAM; the
  runner points bridge pointers straight at its mmap.
- *Realistic concurrency* — cores run as real threads; catches cross-core bugs a
  single-threaded harness misses.
- *Self-maintaining fidelity where it counts* — semaphore placement comes from
  the HAL + `ProgramConfig`; bank topology from the `metal_SocDescriptor`; both
  track tt-metal automatically. A persistent on-disk JIT cache
  (`/tmp/tt_emule_jit_cache_$UID`) survives across runs.

**Limitations**
- *Correctness model, not a perf model* — NOC ops are synchronous `memcpy` (no
  latency, bandwidth, contention, or pipeline stalls); `DPRINT` is a host
  `printf`; no watcher. A test green in emule can still fail on silicon for
  timing/precision/resource reasons.
- *DEST is fp32 internally* — bf16 rounding occurs only at pack/unpack
  boundaries, not per MATH step.
- *JIT compile cost* — each unique kernel costs a compiler invocation (~1–3 s);
  the persistent cache amortizes repeats, but a fresh cache pays full cost.

**Maintainability.** The tt-metal footprint is small and concentrated (the
`emulated_program_runner`, `SWEmuleChip`, one dispatch branch, and
`is_mock_or_emulated()` guards), so rebase risk is bounded — see the rebase
surface in [metal-integration.md](docs/metal-integration.md). The main ongoing
cost is keeping `jit_hw/` stubs aligned with tt-metal kernel APIs; guards fail
loudly (not silently) when a new HW path needs covering.
