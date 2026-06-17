# Implementation plan: chip-to-chip communication mocking in tt-emule (8-chip quietbox CCL)

Status: **implementation plan** (not yet implemented). A phase-by-phase, code-level plan to add chip-to-chip
communication to tt-emule so it can run **real ttnn CCL ops on an 8-chip quietbox**, with functional
correctness (PCC). Synthesizes the prior multichip docs ([scaling](scaling-architecture.md),
[fabric/CCL simulation](fabric-ccl-simulation.md), [op coverage](fabric-ccl-op-coverage.md),
[ttsim](ttsim-methodology.md), [craq-sim](craqsim-methodology.md)) with a code-level
survey of the current emule implementation.

**Assumptions / scope.** The **work-chunking execution engine (Pillar-0 fibers)** from the scaling doc is
assumed implemented separately (it provides cross-device concurrency + the wake mechanism). Otherwise we start
from emule as it is today. Target: an 8-chip quietbox in a **single process** (8 chips' worker L1 fit under
`MAP_32BIT` ≈ 1.2–1.9 GB). Multi-process / multi-host and the fiber engine itself are out of scope here.
Fabric CCL only (legacy EDM out of scope).

## 1. The three questions, answered (grounded in code)

1. **What level of mocking?** **Interception B — enhance emule's jit_hw fabric *shadow stubs*.** The JIT `-I`
   order puts `tt-emule/include/jit_hw` first, and emule shadows every fabric/eth/socket header (each forwards
   to `include/jit_hw/__emule_fabric_stubs.h`). So a CCL kernel that includes the fabric client API compiles
   against emule's stub. Strategy: make those stubs call **runtime teleport hooks** instead of no-op. No
   runtime symbol replacement; no real-router execution.
2. **How do we route interchip communication?** **Direct teleport to the final destination chip — no
   multi-hop, no router.** Resolve the destination chip from the packet header (2D `HybridMeshPacketHeader`
   carries an explicit `dst_start_node_id` = `dst_chip_id | dst_mesh_id<<16`; 1D `LowLatencyPacketHeader`
   carries `distance_in_hops` from `to_chip_unicast`, interpreted against the connection's forward/backward
   direction) via an **emule route table** built host-side from `cluster_desc->get_ethernet_connections()` +
   the control-plane chip↔FabricNodeId map. The `NocCommandFields` `noc_address` gives the destination *core*
   (x,y) + L1 offset on that chip → teleport = in-process `memcpy` into the destination `SWEmuleChip`'s core
   L1 / atomic-inc the remote semaphore + wake the parked consumer fiber. Single process ⇒ no IPC, no drain
   barrier (block-on-sync: deliver + wake).
3. **Do we execute the ethernet cores, or completely stub them?** **Completely stub them.** Fabric *firmware*
   init is already skipped in emule mode (the `is_mock_device()` guard at
   `fabric_firmware_initializer.cpp:282` — emule loads a mock cluster descriptor), so the `fabric_erisc_router`
   program is never compiled or launched, and emule's runner only handles `CoreType::WORKER`. CCL workers
   (tensix) call the shimmed fabric API → teleport; eth cores hold no executing code. Add a
   belt-and-suspenders `CoreType::ETH` skip in `setup_core_state`.

## 2. Verified current-state anchors (the baseline)

- **Runner** (`tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp`): `execute_program_emulated`
  (:2211), `launch_cores` (:2063), `setup_core_state` (:1762), `build_core_map` (:1550),
  `build_kernel_defines` (:1061), JIT `-I` assembly (:716/:803), `get_sw_emulated_chip(device_id)` (:866).
  NOC: `__emule_resolve_noc_addr` (:209, layout `[y:x:36-bit offset]` — **no chip field**), `__emule_noc_resolve`
  (:185), `__emule_multicast_write` (:240 — has the 4-byte `std::atomic` store path to reuse for atomic-inc).
  `__emule_core_map` (:141, `thread_local std::unordered_map<uint64_t, tt_emule::Core*>*`, keyed `(x<<32)|y`).
  Per-kernel thread_local context incl. `__device = nullptr` (:2128) — the chip-context insertion point.
- **Device:** the **real `tt::tt_metal::Device`**, created via `DeviceManager::activate_device`; UMD `Cluster`
  with `ChipType::SWEMULE`. Eth-core topology APIs are descriptor-backed (they work from the 8-chip mock
  descriptor automatically). `tt_emule::Device` (`tt-emule/include/tt_emule/device.hpp`) is the standalone
  test path — **not** this path.
- **SWEmuleChip** (`umd/device/chip/sw_emule_chip.cpp`): `get_core(tt_xy_pair)` (:61) → `tt_emule::Core*` →
  `l1_ptr(offset)`; one SWEmuleChip per descriptor chip (cluster.cpp loop :401-424); reach a chip via
  `MetalContext::instance().get_cluster().get_driver()->get_chip(chip_id)` → `dynamic_cast<SWEmuleChip*>`.
- **jit_hw shadows (the interception point):** `include/jit_hw/__emule_fabric_stubs.h` (the single stub all
  fabric shadows forward to) + `jit_hw/tt_metal/fabric/hw/inc/edm_fabric/{edm_fabric_worker_adapters,
  fabric_connection_manager,routing_plane_connection_manager}.hpp`, `tt_fabric_api.h`, `api_common.h`,
  `linear/api.h`, `mesh/api.h`, `packet_header_pool.h`, `noc_addr.h`; `jit_hw/api/socket_api.h` (no-op);
  `jit_hw/fabric/fabric_edm_packet_header.hpp` (opaque). All currently no-op.
- **Real fabric API to match/decode** (`tt-metal/tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`):
  `WorkerToFabricEdmSender` — `build_from_args`, `open[_start/_finish]`, `close[_start/_finish]`,
  `wait_for_empty_write_slot`, `get_num_free_write_slots`,
  `send_payload_{blocking,non_blocking,without_header_non_blocking,flush_blocking,flush_non_blocking}_from_address`,
  `send_current_slot_non_blocking`, `setup_stateful_send_cmd_bufs`,
  `send_current_slot_stateful_non_blocking[_from_address]`.
- **Packet header layout** (`tt-metal/tt_metal/fabric/fabric_edm_packet_header.hpp`): `NocCommandFields` union
  @0 (40B), `payload_size_bytes` @40 (u16), `noc_send_type` @42 (u8 enum: 0 UNICAST_WRITE / 1 INLINE_WRITE /
  2 ATOMIC_INC / 3 FUSED_ATOMIC_INC / 4 SCATTER / 5 MCAST_WRITE / 6 MCAST_ATOMIC / 7 READ), `src_ch_id` @43,
  `routing_fields` @44. 1D `LowLatencyPacketHeader` 48/64B (routing_fields 4B: low nibble = start/distance,
  high nibble = range). 2D `HybridMeshPacketHeader` 80–128B: `route_buffer` @48, then `dst_start_node_id`
  (`dst_chip_id | dst_mesh_id<<16`). Union variants: `NocUnicastCommandHeader{u64 noc_address}`;
  `NocUnicastInlineWriteCommandHeader{u64 noc_address; u32 value}`;
  `NocUnicastAtomicIncCommandHeader{u64 noc_address; u32 val; bool flush}`;
  `NocUnicastAtomicIncFusedCommandHeader{u64 noc_address; u64 semaphore_noc_address; u32 val; bool flush}`;
  `NocUnicastScatterCommandHeader{u64 noc_address[4]; u16 chunk_size[3]; u8 chunk_count; u8 chunk_encoding}`.
  `to_noc_*` setters rewrite `noc_address` to the **dest core (x,y) + L1 offset** (the shim reads the final
  address). `to_chip_unicast(distance)` / `to_chip_multicast(start,range)` set `routing_fields`.
- **Routing + semaphores:** dest chip — 2D = explicit `dst_start_node_id`; 1D = connection direction (fwd/bwd,
  from `compute_fabric_connection_rt_args`) + distance. Control plane
  (`tt-metalium/experimental/fabric/control_plane.hpp`): `get_fabric_node_id_from_physical_chip_id`,
  `get_physical_chip_id_from_fabric_node_id`, `get_forwarding_direction`, `get_connected_mesh_chip_chan_ids`,
  `get_active_fabric_eth_channels` — build the route table from these (host-side). Global semaphores are
  L1-resident (1 u32/core, same address across chips, via `create_global_semaphore`); the atomic-inc
  `noc_address` resolves to (dest core x/y + sem L1 offset) on the dst chip; the receiver `noc_semaphore_wait`s
  that L1 → emule atomic-inc + wake.
- **Open risk:** with firmware init skipped, the device-L1 routing tables (`ROUTING_TABLE_BASE`) that a
  kernel's `get_next_hop_router_direction` reads are *not* written → 2D direction lookups would read garbage.
  Mitigation: the shim resolves dest from the **explicit header dst / connection identity**, not the kernel's
  computed direction. Also confirm the host-side ControlPlane is still constructed in emule mode (Phase 0).

## 3. Architecture (single-process, 8 chips as fibers)

A process-global **EmuleFabricSwitch**: (1) a **global core map** `(chip_id,x,y) → tt_emule::Core*`; (2) a
**route table** `(src_chip, eth_channel|direction) → dst_chip` plus chip↔FabricNodeId; (3) **teleport hooks**
the shadow shim calls. Teleport = decode header → resolve dst chip + dst core L1 (in-process) → `memcpy` /
atomic-inc + wake (Pillar-0). No queues, no per-cycle drain (block-on-sync). The shadow shim makes worker
fabric sends call the hooks; the **real packet header** is used (drop the opaque shadow) so `to_noc_*` /
routing fields are correct and decodable. The hooks deliver via a `TeleportTransport` seam (the `Local` impl
here = in-process `memcpy`); the future multi-process swap is localized to that one interface — see
scaling-architecture.md §8/§10.

## 4. Comparison: this emule plan vs craq-sim vs ttsim

The defining choice is that emule intercepts at the **highest** point in the cross-chip stack — the
worker→fabric **client API** — and keeps **native** compute, whereas both reference simulators intercept
**low** (ETH_TXQ / eth link) and run the stack via an **instruction-level ISA sim**. emule's closest analogue
is **craq-sim mode B** (host-inject teleport, terminal-semantics-only, no multi-hop), pushed one level higher
(the client API, before an eth packet is even formed) with native JIT'd workers in a single aliased address
space.

| Axis | ttsim (main) | craq-sim | this emule plan |
|---|---|---|---|
| **Interception point** | ETH_TXQ register write | mode A: ETH_TXQ (real router); mode B: host-inject fabric packet | **fabric client API** (`WorkerToFabricEdmSender` shadow) — above router/ETH_TXQ |
| **Eth/ERISC cores** | RET-stub FW; host-launched active-erisc possible; no router | **real ERISC RV32** — runs the real `fabric_erisc_router` ELF | **completely stubbed** — never run; firmware init skipped |
| **Tensix/compute** | instruction-level ISA sim (slow) | instruction-level ISA sim (slow) | **native JIT'd x86, run-to-completion (fast)** |
| **Fabric layer** | none (raw eth) | full (router + packet decode + node_id/direction) | **shimmed at client API**; terminal NOC semantics only |
| **Routing / multi-hop** | compile-time `ETH_PEER_TABLE`, single hop | mode A: real multi-hop (32-chip EBREAK); mode B: teleport | **teleport to final dest, no multi-hop** |
| **Topology source** | compile-time `NUM_CHIPS`, one `.so`/topology | runtime, cluster descriptor (`register_peer`) | **runtime, cluster descriptor** (mock 8-chip) |
| **Device model** | `g_chips[]` + `ttsim_select_chip` | registry + `select_device_by_id` | N `SWEmuleChip`s + global `(chip,x,y)` map + `__emule_chip_id` |
| **Memory model** | heap/`mmap`, no aliasing (ISA translates) | heap/`mmap`, no aliasing | **`MAP_32BIT` direct L1 aliasing**; 8 chips one process |
| **Concurrency / clock** | BSP lockstep (single `g_clock`) | BSP threadpool + per-cycle drain | **block-on-sync, no cycle-step**; Pillar-0 fibers; deliver+wake |
| **Delivery trigger** | sync memcpy at ETH_TXQ write | enqueue + per-cycle `eth_switch_drain` | **sync teleport at the shimmed send + wake** (no drain) |
| **Packet header** | none (raw eth) | 44B header, **stub** noc encoding (not real silicon) | **decodes the REAL tt-metal header** (`NocCommandFields`) |
| **Cross-chip atomic** | n/a | applied at the drain barrier | applied **immediately** at teleport + wake |
| **Process / scale** | single process; compile-time count | single process; 32+; multi-host future | single process for **quietbox/8**; multi-process/host = future |
| **Runs real tt-metal CCL?** | **no** | **yes** (workers + router on sim cores) | **yes** — real CCL *workers* run natively; fabric shimmed |
| **Fidelity vs speed** | low-level eth, slow, single-hop | **highest fidelity**, slowest; the production backend | **functional correctness only** (PCC); fastest |
| **Impl effort** | n/a for fabric | very large (switch + ERISC exec + router + decode) | **moderate** — shadow stubs + teleport hooks + route table; reuse core map / atomic / `MAP_32BIT` / Pillar-0 |

**Borrows from ttsim:** single-process chip context, chip-context-switch + in-process `memcpy` delivery,
link-state faking (descriptor-backed eth topology satisfies UMD discovery without link-training firmware).
**Borrows from craq-sim:** the mode-B teleport idea (skip the router, deliver terminal NOC semantics to the
final dest), the fabric packet semantics (write / atomic-inc / fused / scatter / chip-mcast), descriptor-driven
routing, and the EthSwitch concept — but in-process and without a per-cycle drain.
**Emule-unique:** (1) highest interception (router + eth cores never run); (2) native JIT compute (vs ISA
interp); (3) `MAP_32BIT` aliasing kept (teleport = trivial in-process pointer write); (4) block-on-sync +
Pillar-0 fibers (no global barrier); (5) no multi-hop ever (sidesteps the 32-chip router EBREAK by
construction); (6) decodes the *real* packet header (vs craq-sim's stub encoding).
**Net positioning:** ttsim = raw-eth, no fabric, single-hop, ISA-slow. craq-sim = full-fidelity fabric (real
router + multi-hop), ISA-slow, the reference backend. **emule = functional-fidelity fabric** — real CCL
workers at native speed, fabric collapsed into a client-API teleport, eth cores fully stubbed.

## 5. Phases

### Phase 0 — Multi-chip bring-up + descriptor (verification gate)
- Obtain/confirm an **8-chip quietbox-class cluster descriptor** (T3000 / 6U-style yaml under
  `umd/tests/cluster_descriptor_examples/`, or a custom quietbox descriptor). Set
  `TT_METAL_MOCK_CLUSTER_DESC_PATH`.
- Verify UMD creates 8 `SWEmuleChip`s; the real `Device`/`MeshDevice` reports 8 chips + eth cores +
  `get_ethernet_connections()`; **FabricConfig can be enabled and the host-side ControlPlane is constructed**
  (so CCL ops build + emule can query routing); fabric *firmware* init is skipped (`is_mock_device`).
- Smoke: a trivial replicated single-chip program across all 8 (Pillar-0 gives concurrency).
- **Files:** `tt_metal/llrt/tt_cluster.cpp` (driver init), `umd/device/cluster.cpp` (chip creation), test env.
- **Exit:** 8 devices open, no fabric-router launch, a non-CCL multi-device program runs.

### Phase 1 — Multi-chip addressing + the switch + route table
- **Global core map.** Extend `build_core_map` (`emulated_program_runner.cpp:1550`) to a process-global map
  keyed `(uint64_t(chip_id)<<48)|(uint64_t(x)<<32)|y`, populated by iterating all 8 chips
  (`get_sw_emulated_chip(id)` each). Keep the per-launch chip's submap for local NOC. Add
  `thread_local uint32_t __emule_chip_id;`, set per-chip in `launch_cores` (replace the `__device = nullptr`
  slot at :2128).
- **Route table.** New `EmuleFabricSwitch` host struct, built at first multi-device launch: from
  `cluster_desc->get_ethernet_connections()` + `control_plane.get_fabric_node_id_from_physical_chip_id` /
  `get_connected_mesh_chip_chan_ids`, build `route[(src_chip, eth_channel)] = dst_chip` and chip↔FabricNodeId
  maps. For 1D, also `route[(src_chip, direction FWD/BWD)] = neighbor`.
- **Teleport hooks** (extern "C" in the runner; reuse `__emule_multicast_write`'s atomic-store path):
  ```cpp
  extern "C" uint8_t* __emule_fabric_resolve(uint32_t dst_chip, uint32_t noc_x, uint32_t noc_y, uint64_t l1_off);
  extern "C" void __emule_fabric_write(uint32_t dst_chip, uint64_t noc_addr, const uint8_t* src, uint32_t size);
  extern "C" void __emule_fabric_atomic_inc(uint32_t dst_chip, uint64_t noc_addr, uint32_t val, bool flush);
  // top-level: decode the real packet header, resolve dst chip + core, deliver, wake
  extern "C" void __emule_fabric_teleport(uint32_t src_chip, const void* packet_header,
                                          const void* payload, uint32_t payload_size);
  ```
  `__emule_fabric_resolve` = `__emule_resolve_noc_addr` generalized with chip_id (global-map lookup).
  Delivery into a remote SWEmuleChip core L1 is a plain in-process `memcpy`/atomic store (single process).
- **Exit:** a C++ unit test calls `__emule_fabric_teleport` with a hand-built header and verifies bytes land
  in chip N's L1 + a remote semaphore increments.

### Phase 2 — Eth-core stub (confirm) + fabric host-side availability
- Confirm fabric-router programs never reach `execute_program_emulated` (firmware init skipped). Add a guard
  in `setup_core_state` (:1762): skip kernels whose `hal.get_core_type(pct) == CoreType::ETH`.
- Ensure FabricConfig is enabled and the ControlPlane is queryable in emule mode (so CCL ops build their
  programs + global semaphores). If 2D needs the device-L1 routing tables, EITHER have the shim ignore the
  kernel's computed direction (preferred — use explicit header dst) OR add an emule host-side
  `write_routing_tables` into SWEmuleChip L1.
- **Exit:** a real ttnn CCL op *compiles + launches* (workers only) on 8 chips, with no eth-core program run.

### Phase 3 — The fabric client API shim (the heart; interception B)
- **Use the REAL packet header.** Drop/bypass the opaque shadow `jit_hw/fabric/fabric_edm_packet_header.hpp`
  so the JIT resolves the real `fabric_edm_packet_header.hpp` (header-only; `to_noc_*`/routing setters write
  real fields the teleport decodes). Verify it JIT-compiles (no missing intrinsics). Per project rule "use
  real headers."
- **Enhance `__emule_fabric_stubs.h`** so the sender + connection managers call the hooks:
  - `WorkerToFabricEdmSender`: `build_from_args` parses RT args to record the connection's dst (2D
    `dst_dev_id/mesh_id`; 1D direction) into the shim object; `open/close[_start/_finish]` = bookkeeping;
    `wait_for_empty_write_slot`/`get_num_free_write_slots` = always-free (constant, no spin); `send_payload_*`
    (blocking) and `setup_stateful_send_cmd_bufs` + `send_current_slot_stateful_*` (stateful) →
    `__emule_fabric_teleport(__emule_chip_id, header_ptr, payload_ptr, size)`.
  - `FabricConnectionManager` (fwd/bwd) and `RoutingPlaneConnectionManager` (`.get(slot).sender`, N-slot,
    FABRIC_2D) wrap the same shimmed sender; `fabric_set_unicast_route` / `open_connections` record per-slot
    dst. Mux `fabric_async_write` → same path.
- `__emule_fabric_teleport` decode: read `noc_send_type` @42, `payload_size_bytes` @40, the `NocCommandFields`
  union @0; resolve dst chip (2D explicit `dst_start_node_id` → chip via FabricNodeId map; else 1D
  connection-direction + distance via route table); resolve dst core+offset from the command `noc_address`;
  apply UNICAST_WRITE → write payload; ATOMIC_INC → `__emule_fabric_atomic_inc`; FUSED → write then inc;
  SCATTER → per-chunk; chip-multicast → replay unicast per chip in range; `flush` → immediate.
- **Exit:** a hand-rolled tensix kernel that opens a fabric connection and sends payload + atomic-inc to the
  neighbor chip works end-to-end through the JIT (the shim teleports).

### Phase 4 — Cross-chip global semaphores + wake
- The teleport's atomic-inc resolves the dst chip's core L1 (the global-semaphore address) and atomically
  increments it, then **wakes any Pillar-0 fiber parked on that L1** (the consumer's `noc_semaphore_wait`).
  Tie into the Pillar-0 wakeup registry (same mechanism CB push uses): register waiters by (chip, L1-addr);
  the atomic-inc notifies them.
- `create_global_semaphore` already places the semaphore at a known L1 addr per chip (descriptor-backed); no
  special host work — resolve+inc+wake handles it.
- **Exit:** a 2-chip producer/consumer over fabric (send + atomic-inc; consumer wakes) passes.

### Phase 5 — Bring up CCL + validate
- **Microtest harness first** (mirror craq-sim `tests/multi_device_ccl/`): topology builders (ring/linear) +
  oracle + `bit_exact`/PCC, driving the shim via small tensix kernels (or `__emule_fabric_teleport` directly).
- **Real ttnn CCL:** start **1D** (`Linear`/`Ring`) `all_gather` on the 8-chip quietbox descriptor → expect
  PCC 1.0; then `reduce_scatter`, `all_to_all`. Add **2D** (`Mesh`/`Torus`) once 1D is green (validates the
  explicit-dst path + chip-multicast replay).
- Test env: `MESH_DEVICE=<quietbox>`, `TT_METAL_MOCK_CLUSTER_DESC_PATH=<8-chip desc>`, `TT_METAL_EMULE_MODE=1`,
  `TT_METAL_SLOW_DISPATCH_MODE=1`, the standard emule env.
- **Exit:** real ttnn `all_gather` across 8 chips passes PCC; capture which send-types / 1D-vs-2D headers
  appeared (per [`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md)).

## 6. Files to create / modify

- `tt-metal/tt_metal/impl/emulation/emulated_program_runner.cpp` — global chip-keyed core map; `__emule_chip_id`
  thread_local + per-chip context in `launch_cores`; the `__emule_fabric_*` hooks + `EmuleFabricSwitch` +
  route table; `CoreType::ETH` skip in `setup_core_state`.
- `tt-emule/include/jit_hw/__emule_fabric_stubs.h` — enhance the sender / connection-manager / mux shims to
  call the hooks (replace no-ops); record per-connection dst from `build_from_args`.
- `tt-emule/include/jit_hw/fabric/fabric_edm_packet_header.hpp` — remove/bypass so the **real** header resolves.
- `tt-emule/include/jit_hw/api/socket_api.h` — (later, for tt-blaze) route `socket_push_pages`/`notify` through
  the fabric teleport (`socket_api.h` is fabric-backed). Not required for ttnn quietbox CCL.
- New C++/pytest tests (Phase 1/3/4 microtests + Phase 5 ttnn CCL); regression-script entries.

## 7. Verification (end to end)

Per-phase exit criteria above (unit tests for the hooks/shim; a 2-chip producer/consumer; then real ttnn
all_gather PCC on 8 chips). Always run the WH/BH regressions after runner changes (zero-tolerance). Cross-check
the teleport semantics against [`craqsim-methodology.md`](craqsim-methodology.md) (mode-B
terminal delivery) and the op surface against [`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md) (full
`WorkerToFabricEdmSender` surface incl. stateful + `RoutingPlaneConnectionManager` + `socket_api.h`). Scope
guards: 1D before 2D; single-process (multi-process/host = scaling-doc end-state, out of scope); fabric-only
(legacy EDM out); eth cores fully stubbed.

## 8. Open items to confirm during implementation

- Host-side ControlPlane availability in emule mode (Phase 0) — needed for op build + the route table.
- 2D `get_next_hop_router_direction` reads device-L1 routing tables that the firmware-skip leaves unwritten →
  rely on the explicit header dst (preferred) or populate the tables emule-side.
- The real packet header JIT-compiles cleanly under emule (Phase 3).
- tt-blaze surface (`RoutingPlaneConnectionManager`, stateful sends, `socket_api.h`, worker→forwarder split) —
  covered by the same hooks; validate after ttnn CCL is green.
