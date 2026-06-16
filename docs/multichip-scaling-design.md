# Scaling tt-emule to multi-chip and multi-host

Status: **design / architecture exploration** (no code yet). This document defines the architecture for
growing tt-emule from a single-chip functional emulator to one that runs real multi-chip CCL workloads,
and the phased path to get there.

## 1. Context & goals

emule today emulates a **single chip**. We need multi-chip:

| Target | Scale | Role |
|---|---|---|
| **Quietbox (QB)** | 8 chips | Immediate goal. Must run *proper* (real ttnn, fabric-based) CCL ops end-to-end. |
| **Quad** | 4 galaxies = 128 chips | Ultimate single-host goal. |
| **Multi-host** | up to 36 galaxies = 1152 chips | Stretch. One galaxy per host, networked. Host RAM caps ~6 TB. |

Constraints / decisions:
- **Functional correctness only** — no run-to-run determinism requirement. End-state PCC/bit-exact is the
  bar; OS-thread/scheduler nondeterminism is acceptable as long as synchronization is correct.
- **Fabric CCL only.** Legacy per-op EDM CCL (`ttnn/cpp/ttnn/operations/ccl/kernels/edm/erisc_datamover.cpp`)
  is **explicitly out of scope**.
- **Eth switch:** port craq-sim's switch *design*, implemented natively in emule.
- **Fast dispatch** is a separate workstream owned by another agent; not specified here (see §9).
- Reference simulators live as sibling checkouts: `/localdev/arminale/ttsim-private` (functional ISA sim)
  and `/localdev/arminale/craq-sim` (the heavily-modified ttsim fork that is tt-metal's production
  `SIMULATION` backend; authoritative multichip reference: `craq-sim/MULTICHIP.md`).

The hard question is whether emule's execution model scales, because emule's speed comes from two things
the reference simulators deliberately do *not* do (§4).

## 2. Current emule model (baseline)

The path that runs TTNN workloads is: `TTNN op → tt-metal IDevice → UMD SWEmuleChip (memory) + companion
emulated_program_runner.cpp (executes kernels via jit_hw)`. (The standalone `tt_emule::Device` in
`include/tt_emule/device.hpp` is a separate test-only path and is not the subject of this design.)

- **Execution — OS thread per RISC.** `launch_cores`
  (`tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp:~2063`) spawns one OS thread per logical
  core and a nested OS thread per RISC processor (BRISC / NCRISC / TRISC0–2 / ERISC), then **joins them all
  before returning**. There are no fibers — the `<ucontext.h>` include is only for the SIGFPE handler.
- **Memory/addressing — direct aliasing (the speed lever and the ceiling).** Worker L1 is `mmap`'d with
  `MAP_32BIT` (`include/tt_emule/device.hpp` `Core::mmap_region`, `include/tt_emule/l1_pool.hpp`,
  `umd/.../chip/sw_emule_chip.cpp`). A kernel's 32-bit L1 address is dereferenced **directly as a host
  pointer** — zero translation. NOC addresses are `[y:x:36-bit offset]` with **no chip field**; they
  resolve via `__emule_resolve_noc_addr` → `__emule_core_map[(x,y)]` and a synchronous `memcpy`.
- **Sync — block-on-sync, no cycle-step.** CB push/pop use `std::condition_variable`
  (`include/tt_emule/cb_sync_state.hpp`); semaphores spin → `sched_yield()` → `usleep(1)`
  (`include/jit_hw/api/dataflow/noc_semaphore.h`). A blocked kernel pins its OS thread.
- **Ethernet / multichip — none.** jit_hw eth stubs short-circuit; the device reports 0 eth cores;
  `compile_fabric`/`configure_fabric` are no-op stubs.

The two speed levers — **native JIT'd kernels** (run-to-completion, near-native speed) and **direct L1
aliasing** (no per-access translation) — are what make emule fast and are what the design must preserve.

## 3. The four scaling blockers

1. **Thread explosion (root problem; not fixed by process layout).** Thread-per-core-per-RISC ≈ 360
   threads/WH chip. One galaxy (32 chips) per host ≈ 11.5 K threads on a 384-HW-thread host (~30×
   oversubscription); 128 chips ≈ 46–77 K. Multi-process only *shards* this — the machine still schedules
   all of them. Requires a work-chunking execution engine (§6).
2. **`MAP_32BIT` virtual-address ceiling.** All worker L1 must live in the low ~2 GB. WH ≈ 144 MB/chip, BH
   ≈ 240 MB/chip of L1-pool VA. 8 chips fit (~1.2–1.9 GB, tight on BH); ~16 chips overflow; 128 chips is
   impossible (~18–31 GB).
3. **No chip_id in NOC addressing.** Cross-chip references are unrepresentable today.
4. **No ethernet/fabric model.** No ERISC execution, no inter-chip transport.

RAM is *not* a first wall (L1 is tiny; DRAM is lazily `mmap`'d / overcommitted). The walls are threads (#1)
and address space (#2).

## 4. Reference: how ttsim/craq-sim solve it — and why emule can't copy it cheaply

craq-sim runs N chips in **one process** by **translating every memory access** (RV32/Tensix ISA decode →
per-`Device` heap/`mmap` memory, no `MAP_32BIT`) — so there is no VA ceiling — and a **bounded BSP worker
pool** (`MultiDeviceTensixTileClockPool`, `craq-sim/src/libttsim.cpp:~2270`): each cycle = parallel tile
step → barrier → parallel ETH/ERISC tails → single-threaded `eth_switch_drain`. The cost is
instruction-by-instruction interpretation (slow).

emule's value is the **opposite**: native JIT + direct aliasing. Adopting craq-sim's model means losing
both speed levers and converging on craq-sim itself — at which point you would just use craq-sim. So we
reuse craq-sim's switch **design** (MAC table, source-aware `peer_map`, fabric-packet decode — see
`craq-sim/src/eth_switch.cpp`) but not its code or its cycle-stepped core.

**Chunking contrast:** craq-sim chunks work via cycle-stepped BSP units (no fibers needed, but requires the
ISA interpreter). emule must chunk via **fibers** (keeps native JIT, needs yield-based sync) — §6.

## 5. Design principles

- Preserve emule's two speed levers (native JIT, direct aliasing) wherever possible.
- Reuse craq-sim's switch *design*, implemented natively to fit emule's block-on-sync model.
- Partition along the physical topology (intra-galaxy local, inter-galaxy networked).
- Block-on-sync ⇒ no global cycle barrier: deliver bytes + wake the waiter.
- Functional correctness over timing fidelity.

**Three orthogonal axes (the backbone of this design).** A concrete configuration picks one option on each:

| Axis | Options |
|---|---|
| **1. Execution engine** | work-chunking via fibers (§6) — required in *every* configuration |
| **2. Address space** | keep `MAP_32BIT` aliasing (→ multi-process) **vs** break it (→ single-process translation) |
| **3. Transport** | the native eth switch, tiered by locality (in-proc / shared-memory / network) |

Dispatch mode is **not** an axis we own (separate workstream, §9); we only track our dependency on it.

## 6. Pillar 0 — execution engine: work-chunking via fibers (the keystone)

thread-per-core-per-RISC is the root scaling problem, and **no process layout fixes it**. A bounded
threadpool that runs kernels to completion *deadlocks*, because emule kernels **block** at sync points
(`cb_wait_front`→`cv.wait`, `Semaphore::down`→spin/yield/sleep) and pin their worker OS thread: with P
workers, if P kernels block waiting on data from kernels still queued, no worker is free to produce it.
A native JIT'd kernel can only be suspended two ways — OS-thread block (today, unbounded) or **stackful
fibers**.

**Strategy:** one stackful fiber per RISC, M:N-scheduled on a **bounded worker pool (~HW-thread count)**;
rewrite the sync primitives (CB wait/reserve, semaphore wait, NOC barrier) from "block the OS thread" to
"**yield to the scheduler**", with a wakeup registry (e.g. a producer's `cb_push_back` re-queues fibers
parked on that CB). This caps OS threads at ~HW-thread count regardless of chip/core count; fibers are
cheap (KB stacks, parked when idle), so tens of thousands of them on a few hundred workers is fine.

Fibers are a keystone because they collapse three blockers into one mechanism:
- **Thread explosion** → bounded worker pool.
- **Multi-device concurrency** (the cross-chip CCL deadlock, §9) → all chips' fibers share one runnable
  pool; a fiber blocked on a cross-chip semaphore parks and is woken by eth delivery. No per-device join
  ordering.
- **Persistent fabric routers** (§9) → a never-terminating router is just a fiber that loops forever,
  yielding at each poll; it never joins, never pins a worker; teardown = set the flag, the loop exits.

Scope: a re-architecture of the companion runner's `launch_cores` plus the jit_hw sync layer. This is the
single biggest piece of work and should be built early — a thrown-away thread-per-RISC stopgap buys little
(see §8). Note: fibers do **not** touch the `MAP_32BIT` ceiling — address space stays a separate axis.

## 7. Native eth switch + fabric

Port craq-sim's switch design natively: a process-wide switch with a **MAC table**, a **source-aware
`peer_map`** (a destination MAC that misses the table falls back to the source's wired peer — BH BCAST/MCAST
MACs are RXQ selectors, not addresses), and a **route table** built from the cluster descriptor plus the
fabric control plane. Wiring mirrors UMD's `SIMULATION` eth pre-pass (`configure_eth_link` + `register_peer`;
fabric `node_id`/`endpoint_direction`/`route`) — see `umd/.../device/cluster.cpp` and
`umd/.../device/simulation/tt_sim_communicator.cpp` as the validated reference.

**Delivery fits emule's block-on-sync machinery:** deliver bytes into the destination eth-tile (or worker)
L1, increment the target semaphore, and wake the parked fiber — **no per-cycle drain barrier** (unlike
craq-sim's BSP). Add `chip_id` to NOC/eth addressing and per-chip core maps (blocker #3). emule must
implement `compile_fabric`/`configure_fabric` (today no-op stubs) to consume the control plane's routing
tables and build the switch route table. Cross-chip traffic is *always* ethernet (never direct cross-chip
NOC), so the switch is the single interception point.

### Key fork — interception level (the emule-vs-craq-sim philosophy decision)

- **(A) ETH_TXQ register level (craq-sim fidelity).** Run the *real* persistent fabric router kernel
  (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`) on an emule ERISC fiber; model the
  eth-tile `ETH_TXQ` registers; the switch intercepts `ETH_TXQ_CMD` writes and delivers to the peer
  (cf. craq-sim's `eth_txq_regs_wr32`). Highest fidelity, but pulls in persistent-router execution +
  ETH_TXQ modeling, and requires the Pillar-0 *persistent-fiber* capability.
- **(B, recommended) fabric-client-API shim.** Shim the worker→fabric sender / EDM-client API; route
  straight to the switch using the route table. **No persistent router runs**, no ETH_TXQ modeling. Lighter,
  matches how emule shims every other LLK, and aligns with the functional-correctness-only goal. Tradeoff:
  lower fidelity (the real router code never executes), and the `tt_fabric` client API surface must be
  shimmed faithfully — validate that surface in Phase 1.

Worker / cross-device concurrency needs Pillar-0 fibers under **either** option; only **(A)** additionally
needs the persistent-router fiber. Recommend **(B)**; the doc carries both so the choice can be settled in
Phase 1.

## 8. Phased architecture

### Phase 1 — Quietbox (8 chips), single process, *proper fabric CCL*

Keep `MAP_32BIT` aliasing (8 chips fit under ~2 GB). Because QB must run the **real fabric-based ttnn CCL
ops** end-to-end, Phase 1 is **not** a thin transport bring-up. It includes:
- **Pillar-0 fibers** — required for worker/cross-device concurrency; no thread-per-RISC stopgap.
- **chip_id addressing** + a multi-chip core map.
- the **eth switch + fabric route table**, `compile_fabric`/`configure_fabric`, eth-tile bring-up.
- the chosen **interception level** (A or B).

Legacy per-op EDM CCL is explicitly out of scope. **Exit criterion:** a real ttnn fabric CCL op (e.g.
all-gather) across 8 chips passes PCC under a quietbox-class cluster descriptor; capture thread/VA/perf
numbers to settle the end-state address-space axis.

### End-state intra-host — quad / 128 chips: an address-space axis

With fibers handling thread count and concurrency (Pillar 0), the end-state choice is **purely** about the
`MAP_32BIT` VA ceiling. Two candidates, to be decided after Phase-1 measurements:

- **Multi-process, keep aliasing.** Each process keeps its private low-2 GB `MAP_32BIT` aliasing window +
  native JIT (fastest per-access). Promote the eth switch to a **shared-memory** transport; cross-process
  deliver + wake. Cost: shmem plumbing, cross-process fiber wakeup, a chip→process placement map. (Each
  process still runs its own Pillar-0 fiber pool — multi-process is for *address space*, not threads.)
- **Single-process, break aliasing (translation).** chip_id-indexed L1 translation on every access; no VA
  ceiling, one in-process fiber scheduler + one in-process switch (simplest concurrency/wakeup), could reach
  128 chips in one process (RAM permitting). Cost: sacrifices the direct-aliasing speed lever.

### Multi-host — 1 galaxy per host (up to 36 galaxies)

Same switch, a **third transport tier**. Partition along the physical topology: intra-galaxy links are local
(tier 1/2), and the comparatively *sparse* **inter-galaxy links go over the network** (TCP/RDMA) — good
locality, low cross-host link count. Seed: craq-sim's `libttsim_configure_eth_link_fd` (FD-based
cross-process eth link) is the conceptual starting point for network transport.

emule's block-on-sync model is *advantageous* here: a cross-host eth packet is an **async, ordered message**
that wakes a blocked consumer — **no distributed per-cycle barrier** (which craq-sim's BSP would require
across hosts). New work: a network switch transport; a chip→process→host placement/routing table; reliable
per-link in-order delivery (sequence numbers); distributed liveness/hang detection. Each host still runs its
Pillar-0 fiber pool (1 galaxy ≈ 11.5 K fibers on a bounded ~HW-thread pool — fine; as OS threads it would
not be). 6 TB/host comfortably holds a galaxy (DRAM lazy-faulted); galaxy-vs-quad-per-host granularity is a
measurement call, not an architectural one.

## 9. Fast-dispatch dependency + execution-model requirements

**Fast dispatch is a separate workstream owned by another agent; this document does not specify it.**
Dependency status, grounded in tt-metal source:

- **No hard dependency for the multichip transport + execution engine.** emule rides slow dispatch — its
  interception is inside `detail::LaunchProgram` (`tt_metal/impl/host_api/tt_metal.cpp:842`, gated on
  `TargetDevice::Emule`). tt-metal has a dedicated slow-dispatch mesh path: `SDMeshCommandQueue`
  (`tt_metal/distributed/sd_mesh_command_queue.cpp` → `LaunchProgram` + `WaitProgramDone`). No CCL or
  `MeshDevice` `TT_FATAL` requires fast dispatch. craq-sim's use of fast dispatch was a *choice*, not a
  tt-metal requirement.
- **Concurrency is satisfied independently** by Pillar-0 fibers — we are not blocked on the fast-dispatch
  effort to get multi-device concurrency.
- **Coordination points to confirm with the fast-dispatch agent:** (a) whether the target *production* CCL
  pytests are in practice only exercised through fast dispatch / mesh CQ — if so, "run unmodified CCL at
  scale" *softly* depends on that workstream; (b) multi-device program launch must still reach emule's
  `LaunchProgram` interception under whatever dispatch the FD agent enables; (c) fabric-init sequencing.

**Execution-model requirements (our work, independent of dispatch mode).** CCL target is **fabric only** —
modern ttnn CCL (`all_gather`, `reduce_scatter`) is fabric-based (confirmed:
`ttnn/cpp/ttnn/operations/ccl/all_gather/device/all_gather_program_factory.cpp` includes
`experimental/fabric/fabric.hpp`, delegates to `all_gather_async`, uses `tt::tt_fabric::Topology`, and
worker kernels call `command_requires_fabric()`). The fabric router
(`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`) is **persistent**
(`while(continue_running_main_run_loop(...))`, fire-and-forget launch, host-written teardown) — persistent
in *both* dispatch modes. Whether emule must *run* it is the interception fork in §7 (A runs it as a
persistent fiber; B shims the client API so no router runs).

Requirements:
1. **Multi-device concurrency** — Pillar-0 fibers (one runnable pool) or multi-process. *Verify how
   `SDMeshCommandQueue` orders multi-device launches (sequential within a coord-range would deadlock a
   synchronous launch-and-join model — the genuine execution-model gap).*
2. **Eth + fabric transport** — the switch + route table + `compile/configure_fabric`.
3. **Persistent-router support** — a Pillar-0 persistent fiber, needed *only under interception (A)*.

## 10. Cross-cutting concerns

- **3-tier transport abstraction.** One switch API, pluggable transport (in-proc queue / shared-memory /
  network), selected per link by peer locality from the placement table.
- **Placement / routing tables.** chip → (process, host) placement; the switch route table derived from the
  cluster descriptor + fabric control plane.
- **Determinism.** Functional correctness only; the only ordering requirement is per-link in-order delivery
  (preserved by TCP, or per-link sequence numbers).
- **Hang / deadlock detection.** Keep today's per-fiber/per-process timeout; extend to cross-host liveness
  (a consumer blocked on a packet dropped in transit must time out).
- **Memory & thread math, per tier.** L1 tiny; DRAM lazy. Threads bounded by Pillar-0; VA bounded by the
  address-space axis choice.

## 11. Open questions (decide in / after Phase 1)

- **Interception level A vs B** (§7) — the load-bearing fidelity-vs-effort fork.
- **End-state process model** — multi-process (aliasing) vs single-process (translation); settle with
  Phase-1 thread/VA/perf numbers.
- **Partition granularity** — galaxy vs quad per host (a measurement call).
- **Network transport** — TCP vs RDMA vs MPI for the multi-host tier.
- **`SDMeshCommandQueue` launch ordering** — confirm it gives multi-device concurrency, not sequential
  per-device launch.
- **Fast-dispatch coordination** — is there a soft dependency for production CCL at scale (§9)?
- **UMD reuse** — how much of UMD's `SIMULATION` eth/fabric wiring to reuse for the `SWEMULE` path vs build
  natively.

## 12. Milestones / verification

- **Phase 1:** a real ttnn fabric CCL op (all-gather) on an 8-chip quietbox-class descriptor passes PCC;
  record measured thread count, VA usage, and wall-clock to settle the address-space axis.
- **Per tier:** name a target CCL workload + cluster descriptor (quietbox → quad → galaxy-per-host) and its
  PCC exit criterion. No silent coverage caps — log what a milestone does *not* yet cover.
- **Reference cross-checks:** validate the switch design and descriptor-driven wiring against
  `craq-sim/MULTICHIP.md` and the UMD `tt_sim_communicator.cpp` bridge.
