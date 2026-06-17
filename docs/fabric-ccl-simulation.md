# Fabric & CCL simulation: how it works, how ttsim/craq-sim simulate it, and how emule will

Status: **design / architecture exploration** (no code yet). Companion to
[`multichip-scaling-design.md`](multichip-scaling-design.md) and
[`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md). The scaling doc covers the *strategy* (the three
orthogonal axes + the phased path); the coverage doc validates the interception level below against the full
ttnn **and tt-blaze** op surfaces. **This doc is the fabric/CCL data-path deep-dive**: what the tt-metal
fabric layer actually is, a full account of how **ttsim** and **craq-sim** simulate multi-chip/fabric, a
decomposition of the real fabric router, a side-by-side comparison, how CCL ops ride fabric end-to-end, and
the recommended emule design. The interception-B recommendation here is validated across both ttnn and
tt-blaze in the coverage doc.

## 1. Context & scope

"Proper CCLs on quietbox" are in scope, and the modern ttnn CCL ops are **fabric-based** (confirmed in
`multichip-scaling-design.md` §9), so the fabric/CCL layer is Phase-1-critical. The scaling doc named a fork
— **(A)** run the real persistent fabric router vs **(B)** shim the fabric client API. **Hard constraints
(set by the team) resolve it toward B:**

- emule will **not** emulate ethernet cores;
- emule will **not** model multi-hop journeys;
- **direct teleport to the final destination is acceptable.**

So this doc reproduces only the router's *terminal NOC-command semantics* (§5), never its forwarding
mechanism. CCL target is fabric only — legacy per-op EDM CCL is out of scope. Bar: functional correctness
(PCC), not timing.

## 2. What tt-metal fabric is (the spec to reproduce)

**Control plane.** Fabric is configured on the (mesh) device by a control plane that owns the topology and
routing tables:
- `FabricContext` (`tt_metal/fabric/fabric_context.cpp`) — topology queries, packet-spec/header sizing.
- `ControlPlane` (`tt_metal/fabric/control_plane.cpp`) — mesh graph, routing direction tables, fabric-node →
  physical-chip mapping.
- `MeshGraph` (`tt_metal/fabric/mesh_graph.cpp`) — loads `.textproto` mesh-graph descriptors (N300/T3K/Galaxy/…),
  registers chip-to-chip links with a routing direction.
- `FabricNodeId{MeshId mesh_id; uint32_t chip_id}`
  (`api/tt-metalium/experimental/fabric/fabric_types.hpp`) — fabric-space identity (can differ from physical
  chip id).
- Directions `EAST=0, WEST=1, NORTH=2, SOUTH=3, Z=4` (`hostdevcommon/api/hostdevcommon/fabric_common.h`); `Z`
  is inter-mesh.
- `FabricConfig` = `FABRIC_1D / FABRIC_1D_RING / FABRIC_2D / FABRIC_2D_TORUS_{X,Y,XY}`.
- Setup path: `fabric_init.cpp` → `FabricBuilder` (discover_channels → create_routers → connect_routers →
  create_kernels) → `configure_fabric_cores`. The device hooks are `compile_fabric` / `configure_fabric`.

**Fabric router kernel** (`tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp`). A **persistent**
ERISC kernel: three channels (sender0 = local workers, sender1 = upstream EDM, receiver = inbound eth); reads
each packet header and either delivers it locally or forwards it to the next hop; flow control via stream
registers; eth transmission via ETH_TXQ; runs until the host writes a termination signal.

**Worker → fabric client API — THE interception boundary.** Worker/compute kernels do not touch the router
directly; they use a client adapter:
- `WorkerToFabricEdmSender` (`tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp`):
  `open()/open_start/open_finish`, `wait_for_empty_write_slot()`, `get_num_free_write_slots()`,
  `send_payload_*_from_address()` (blocking), and the **stateful** path `setup_stateful_send_cmd_bufs()` +
  `send_current_slot_*` (incl. `send_current_slot_stateful_non_blocking_from_address`), `close()`.
- Mux façade `fabric_async_write(conn, pkt_hdr, src, size)`
  (`tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp`).
- Wrappers over the sender: `FabricConnectionManager` (ring fwd/bwd) → `get_forward_connection()` /
  `get_backward_connection()`; and **`RoutingPlaneConnectionManager`** (N-slot, FABRIC_2D) → `get(slot).sender`
  (used by tt-blaze cross-device ops — see [`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md)).
- **`socket_api.h`** (`tt_metal/hw/inc/api/socket_api.h`) is a fabric-backed surface (it includes
  `fabric_connection_manager.hpp` and does `fabric_set_unicast_route` for D2D sockets) — i.e. sockets ride
  the *same* fabric transport, not a separate one.

**Packet headers** (`tt_metal/fabric/fabric_edm_packet_header.hpp`):
- 1D `LowLatencyPacketHeaderT` (48–64B, 2 bits/hop routing) vs 2D `HybridMeshPacketHeaderT` (80–128B, 8
  bits/hop, direction masks).
- `NocCommandFields` union (40B): unicast write / inline write / atomic-inc / fused write+atomic-inc /
  scatter / multicast. Setters: `to_noc_unicast_write`, `to_noc_unicast_atomic_inc`,
  `to_noc_unicast_scatter_write`, `to_chip_unicast(distance_in_hops)`, `to_chip_multicast(start, range)`.

**Init/launch order:** device init → control-plane setup (mesh graph, routing tables, node ids) → router
kernels launched (persistent, fire-and-forget) → CCL op enqueued → workers open fabric connections → send →
teardown (host writes termination signal).

## 3. How ttsim (main) simulates multichip/eth — full detail

ttsim-private `main` is the upstream functional ISA simulator. Its multichip model is the **simplest faithful
one**, and a useful source of reusable mechanics — but it has **no fabric layer**.

- **Topology = compile-time `ETH_PEER_TABLE`** (`src/tile.cpp:62-181`): `struct EthLink{uint8_t src_chip,
  src_tile, dst_chip, dst_tile}`, selected by `#if NUM_CHIPS==…` and `TT_ARCH_VERSION`. One **`.so` per
  topology** (`src/rules.py` CHIPS table: `wh`, `wh_x2`, `wh_x8`, `wh_x32`, `bh`, `bh_x2`, `bh_x32`). There is
  **no topology discovery** — the `tests/multichip/*.yaml` files are descriptive only, never loaded.
  `chip_coord_to_id` (`src/tile.cpp:323`) maps 2D mesh coords → linear chip id.
- **Eth-tile model** (`EthTile`, `src/sim.h:633`): SRAM (256 KB WH / 512 KB BH), 1–2 ERISC RV32 cores,
  registers `eth_txq_control/cmd/transfer_start_addr/transfer_size_bytes/dest_addr/remote_reg_data`,
  `eth_rxq_control`, MAC regs, the BH `eth_txpkt_cfg_mac_da_*` header table, and link-status / boot-results
  regions.
- **Delivery = synchronous memcpy, no queue, no latency** (`eth_txq_regs_wr32`, `src/tile.cpp:~2016`): when an
  ERISC kernel writes `ETH_TXQ_CMD`, `=2` (L1_WRITE block) reads the transfer start/size/dest regs, calls
  `eth_peer()` to find the wired peer, then `ttsim_select_chip(remote)` + `memcpy` into the peer eth-tile SRAM
  and restores chip context; `=4` (MMIO) writes a single 32-bit register via `tile_wr_bytes`. The copy happens
  immediately at the register write.
- **WH legacy remote queue** (`src/tile.cpp:332-434`): a host-mediated cross-chip read/write path. `sys_addr`
  bitfields `[0:35]=local_addr, [36:41]=noc_x, [42:47]=noc_y, [48:59]=chip_x/y` → `chip_coord_to_id` →
  select + read/write. The host triggers processing by writing the request write-pointer.
- **Link init = faked state; base firmware stubbed** (`e_tile_init` `src/tile.cpp:204-275`;
  `wh_x2_eth_link_init` `src/tile.cpp:473`): for peered tiles it injects `ETH_TRAIN_STATUS=1`, ASIC/board ids,
  heartbeat (WH), and `port_status/train_status/rx_link_up` (BH `0x7CC..`). The *base* ERISC firmware is
  stubbed to a `RET` — no link-training runs. Eth cores are **not inert**, though: a host `RUN_MSG_GO` to a
  peered eth tile (`src/tile.cpp:3017-3069`) *can* launch an active-erisc kernel that the RV32 core then
  executes. But there is **no persistent fabric router**, so this does not enable fabric routing — see the
  full account in [`ttsim-multichip-methodology.md`](ttsim-multichip-methodology.md).
- **Chip context + clock** (`src/sim.h:811`, `src/sim.cpp:75`, `src/libttsim.cpp:767`): `g_chips[NUM_CHIPS]`,
  `g_current_chip_id`, `ttsim_select_chip`; macros alias `g_t_tiles`/`g_e_tiles`/`g_dram` to the active chip.
  `libttsim_clock(n)` loops every chip one cycle each, incrementing a single global `g_clock` (BSP lockstep).
- **API = the 9 base symbols only** (no `create_device_by_id` / `switch_*`). Multichip is purely internal
  (compile-time `NUM_CHIPS`, one process, shared `g_chips[]`). The host sees N PCIe devices via a paddr stride
  (`PER_DEVICE_PADDR_STRIDE = 64 GiB`; `libttsim_pci_mem_*` divides paddr → device → select).

**CAN:** point-to-point eth L1/MMIO delivery, host-mediated remote R/W, link-state faking, per-chip isolation,
mesh/torus *layouts* (as static peer tables), lockstep clock. **CANNOT:** multi-hop routing, fabric, cross-chip
atomics, dynamic/runtime topology, real firmware/CCL. ttsim sits **below** the fabric router.

## 4. How craq-sim simulates fabric

craq-sim is the heavily-modified ttsim fork that is tt-metal's production `SIMULATION` backend (authoritative
reference: `craq-sim/MULTICHIP.md`). It adds a runtime multi-device model (one process, `create_device_by_id`
/ `select_device_by_id`, dynamic topology from the cluster descriptor) and a process-wide `EthSwitch`. Fabric
is simulated in **two modes**:

- **(A) Real persistent router ELF on a simulated ERISC.** `libttsim_load_erisc_elf` + reset vectoring
  (`src/tile.cpp:3093-3137` SOFT_RESET_0 → RV32 start) load and run the *actual* `fabric_erisc_router` ELF.
  The router's ETH_TXQ writes are intercepted by `eth_txq_regs_wr32` (`src/tile.cpp:2428`) → `e_tile_send_transaction`
  (`:4443`) → `eth_switch_enqueue`, which teleports each eth-link hop to the peer eth tile; clocked per cycle
  via `libttsim_clock_all_devices` → drain.
  This is the v3.5 production path and genuinely simulates **multi-hop** — which is why the 32-chip Galaxy halts
  on a real fabric-router `EBREAK` in `update_packet_header_for_next_hop` (a zero-route-array trap).
- **(B) Host-side fabric packet injection.** `eth_switch_enqueue_fabric_packet(src_chip, routing, command_40b,
  send_type, payload_size, payload)` (`src/eth_switch.h:239`, impl `src/eth_switch.cpp:1116`) decodes a 44-byte
  fabric header and delivers directly to the **final** destination (`maybe_apply_bh_fabric_terminal_local_delivery`,
  `src/tile.cpp:~3805`). No router runs; **no multi-hop** — pure teleport reproducing terminal semantics. Used
  by `tests/multi_device_ccl/`.

**Fabric switch state** (`src/eth_switch.cpp`): `register_fabric_route` (`fabric_routing_table` +
`fabric_synthetic_mac`), `register_fabric_node_id` (device → `FabricNodeId`), `register_fabric_endpoint_direction`
(endpoint → MY_DIRECTION, for terminal-local-delivery), `compute_route_to_chip` (a BFS that is implemented but
**unused** — delivery is direct teleport). **44-byte packet decode:** send types 0/2/3/4 = unicast write /
atomic-inc / fused write+atomic / scatter; atomics applied at the drain barrier; a **stub noc-address encoding**
packs chip_id in the high bits (explicitly *not* real silicon — `docs/v3_5_commit15_fabric_atomic_layout.md`;
the real-atomic-header decode fixed a DDR-smoke semaphore wrap).

## 5. Deeper dive: the real fabric router vs ttsim vs craq-sim

Decomposing `fabric_erisc_router.cpp` + `fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp` under the
hard constraints (no eth-core emulation, no multi-hop, teleport allowed), the router splits cleanly into what a
teleport shim must reproduce and what it can drop.

**REPLICATE — terminal NOC-command semantics** (`execute_chip_unicast_to_local_chip_impl`,
`fabric_edm_packet_transmission.hpp:137-305`): a switch on `noc_send_type`, applied at the final destination:

| send type | terminal effect |
|---|---|
| `NOC_UNICAST_WRITE` | write payload → dest L1 |
| `NOC_UNICAST_INLINE_WRITE` | write one DWORD → dest L1 |
| `NOC_UNICAST_ATOMIC_INC` | `semaphore += val` (optional flush) |
| `NOC_FUSED_UNICAST_ATOMIC_INC` | write payload, then atomic-inc a (possibly different) semaphore |
| `NOC_UNICAST_SCATTER_WRITE` | ≤4 chunks, each a write or sem-inc, all on one chip |

- **chip-multicast = a contiguous *range* of chips** (`routing_fields` START_DISTANCE + RANGE), each receiving
  a unicast write — a teleport shim **replays a unicast per chip in the range**. (The terminal path does *not*
  use NOC-multicast — the router rejects it.)
- `flush` is an ordering barrier; under synchronous teleport it can be treated as immediate.

**SKIP — forwarding mechanism** (pure "get the packet to the final chip", irrelevant to teleport):
`update_packet_header_for_next_hop` (hop/range decrement), channel arbitration (E/W/N/S), ETH_TXQ
enqueue/serialization, stream-register credits/flow-control, per-hop forwarding.

**Final destination without hops.** Routing fields carry the *absolute* distance from the origin
(`to_chip_unicast(distance_in_hops)` / `to_chip_multicast(start, range)`). The final chip(s) = `this_chip + N`
in the connection's direction, resolved by **direct index** against the control-plane mesh graph — no
intermediate-router simulation.

**Completion/credit faking.** The worker polls the EDM's free-slot count via `wait_for_empty_write_slot()` and
performs a close handshake. With no real EDM buffer, the shim must report "space available" / ack immediately so
the worker loop makes progress.

**How each simulator treats the router:**
- **ttsim** — *below* the router: single-hop raw eth `memcpy`, no router, no terminal NOC semantics → cannot run
  fabric CCL.
- **craq-sim mode A** — *runs* the real router ELF on a simulated ERISC → genuine multi-hop (each eth-link hop
  teleported); source of the 32-chip `update_packet_header_for_next_hop` EBREAK.
- **craq-sim mode B** — *skips* the router, teleports to the final destination, reproducing only terminal NOC
  semantics. This is emule's target model — and emule intercepts one level higher still (the client API), so it
  never forms an eth packet at all.

## 6. ttsim vs craq-sim — comparison, and the recommendation for emule

| Axis | ttsim (main) | craq-sim |
|---|---|---|
| Topology | compile-time `ETH_PEER_TABLE`, one `.so`/topology | runtime, from cluster descriptor (`register_peer`/`route`) |
| Device mgmt | `NUM_CHIPS` compile constant, internal | `create`/`select_device_by_id` (runtime N) |
| Eth delivery | ETH_TXQ regs → select + memcpy to peer | EthSwitch teleport (MAC/peer); ETH_TXQ in mode A |
| Fabric layer | **none** | full: packet decode, node_id, direction, atomics, terminal-delivery |
| Cross-chip atomics | none | inc / fused / scatter |
| Firmware | base FW stubbed (RET); link state faked; no router | runs **real** ERISC / fabric-router ELFs (mode A) |
| Multi-hop | n/a (single hop) | mode A: real (→ 32-chip EBREAK); mode B: none (teleport to final dest) |
| Eth cores run? | only host-launched active-erisc (no router) | mode A: yes (real ERISC ELF); mode B: no |
| Real tt-metal CCL | no | yes — production SIMULATION backend (P300 / P150 / Galaxy submesh green) |
| API | 9 base symbols | full dynamic switch ABI |

**Recommendation — a hybrid, centered on craq-sim mode B's teleport idea, intercepting at the client API:**

- **Pure ttsim is insufficient** — emule needs fabric CCL, which ttsim has none of.
- **craq-sim mode A is excluded** by the hard constraints (it emulates eth cores and does multi-hop), and it
  still carries the unresolved 32-chip EBREAK.
- **Borrow from ttsim** the lightweight, proven mechanics: chip-context-switch + `memcpy`-into-peer-L1 delivery;
  **link-init / boot-state faking** (so UMD topology discovery is satisfied without running link-training
  firmware); the single-process `g_chips[]` + select pattern.
- **Borrow from craq-sim** the fabric *design*: runtime topology from the cluster descriptor + control plane,
  the route table, the terminal NOC-command semantics, and the UMD wiring shape.
- **Intercept at the fabric client API (B)** — above both ttsim's ETH_TXQ layer and craq-sim's real router.

## 7. How CCLs ride fabric end-to-end

- `ttnn::all_gather` (`ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp`) → `all_gather_async`
  (`ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp`):
  the program factory creates **2 global semaphores (forward/backward) + 1 barrier semaphore**
  (`.../all_gather/device/all_gather_program_factory.cpp:36`) and launches reader / writer (/ mux) kernels on
  worker cores in both ring directions. (reduce_scatter is analogous, with a divided output and an accumulation
  buffer.)
- **The worker → fabric send** (`ttnn/cpp/ttnn/operations/ccl/common/kernels/ccl_send_reader_two_input.cpp`) is
  command-driven; the send is:
  `fabric_connection.get_{forward,backward}_connection().wait_for_empty_write_slot()` then
  `send_payload_without_header_non_blocking_from_address(payload, …)` +
  `send_payload_flush_blocking_from_address(pkt_hdr, sizeof(hdr))`. Atomic-inc is encoded via
  `to_noc_unicast_atomic_inc`; intra-chip work falls back to `noc_semaphore_inc` / `noc_inline_dw_write`.
- **Cross-device synchronization:** global semaphores + fabric atomic-inc + a pre-op barrier
  (`distributed::Synchronize`). **Correctness requires all participating chips' programs to run concurrently** —
  chip A blocks on a semaphore that chip B increments over fabric.
- **craq-sim's in-process CCL harness** (`tests/multi_device_ccl/`) exercises CCL semantics *without* the
  tt-metal stack: `topology.h` (Ring/Linear/Mesh2D/Torus2D builders), `ccl_primitives.h` (host orchestrators),
  `oracle.h` + verifiers (`bit_exact` / `comp_pcc`). This is a good model for emule microtests.

## 8. emule design — the fabric-client-API shim (interception B)

Shim the **`WorkerToFabricEdmSender` object** so a worker's send routes straight to the emule eth switch —
**no persistent router, no ETH_TXQ modeling**. The sender is reached via any of: `FabricConnectionManager`
(ring fwd/bwd), raw `WorkerToFabricEdmSender[]` (MoE 4-dir), the mux `fabric_async_write`,
**`RoutingPlaneConnectionManager`** (tt-blaze, N-slot), and **`socket_api.h`** (sockets-over-fabric). The shim
must cover the **full sender method surface — blocking *and* stateful** (the coverage doc found tt-blaze
`all_gather`/`all_reduce`/`sdpa_reduce` use the stateful path; a `send_payload_*`-only shim would miss them):

- **Connection lifecycle** (`open`/`close`): no-op / bookkeeping. Resolve the connection's target chip from the
  control-plane routing tables (forward/backward direction → peer `FabricNodeId` → `SWEmuleChip`).
- **Flow control** (`wait_for_empty_write_slot` / `get_num_free_write_slots`): report "always free" (or a fake
  credit) so the worker loop progresses — there is no real EDM buffer.
- **Sends — both forms:** blocking `send_payload_{without_header_non_blocking,flush_blocking}_from_address`, and
  the stateful `setup_stateful_send_cmd_bufs` + `send_current_slot_{non_blocking,stateful_non_blocking_from_address}`.
  Decode the **real** `NocCommandFields` packet header (write / inline-write / atomic-inc / fused / scatter; dest
  noc address) — note the header may have been **built by a different (worker) core and forwarded via L1 slots**
  (the tt-blaze worker→forwarder split; that worker→forwarder leg is intra-chip NOC, emule-native) — then
  **teleport** to the final destination: write payload → L1 / atomic-inc the remote semaphore / fused / scatter,
  and **wake the parked consumer fiber** (the scaling doc's Pillar 0). **chip-multicast = a contiguous chip
  range → replay a unicast per chip**; scatter = per-chunk writes/sem-incs on one chip; `flush` → immediate.
  This surface covers both ttnn and tt-blaze (see [`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md)).
- **Routing without hops:** final dest chip(s) = `this_chip + N` in the connection direction, by direct index
  against the control-plane mesh graph. No intermediate-router simulation, no multi-hop.
- emule must implement `compile_fabric` / `configure_fabric` (today no-op stubs in `device.hpp`) to consume the
  control plane and build the switch route table, and provide **cross-chip global semaphores**.

Mode (A) — running the real router — is documented only as the rejected higher-fidelity alternative craq-sim
uses; it is excluded by the hard constraints (eth-core emulation + multi-hop) and burdened by the 32-chip EBREAK.

### Minimal capability set for functional fabric CCL
1. fabric-client-API shim (the boundary above);
2. deliver bytes to peer chip L1 (the switch);
3. atomic-inc remote semaphore + wake;
4. cross-chip global semaphores;
5. concurrent multi-device execution (Pillar-0 fibers — see scaling doc);
6. real packet-header decode (`NocCommandFields`);
7. route table from the fabric control plane.

## 9. emule mapping & gaps

| Capability | emule has | emule needs |
|---|---|---|
| Per-chip memory (L1/DRAM) | ✅ `SWEmuleChip` + L1Pool | per-chip core maps keyed by chip_id (scaling doc) |
| Kernel-API shim pattern | ✅ jit_hw shims everywhere | shim the full `WorkerToFabricEdmSender` method set (blocking + stateful) via `FabricConnectionManager` / raw `[]` / mux / `RoutingPlaneConnectionManager` / `socket_api.h` |
| Inter-chip transport | ❌ (eth stubs short-circuit) | the native eth switch (scaling doc) + teleport-to-final-dest |
| Fabric setup | ❌ (`compile/configure_fabric` no-op) | consume the control plane, build the switch route table |
| Packet decode | ❌ | decode `NocCommandFields` (write/inline/atomic/fused/scatter) + chip routing fields |
| Cross-chip global semaphores | ❌ | semaphore objects reachable across chips, with wake |
| Concurrent multi-device execution | ❌ (launch-and-join, single chip) | Pillar-0 fibers (scaling doc) |

Cross-reference: `multichip-scaling-design.md` Pillar 0 (the fiber execution engine that makes cross-device
concurrency and the wake-on-delivery work) and its eth-switch section (the transport this shim delivers into).

## 10. Validation

- **Microtests first** — mirror craq-sim's in-process CCL harness: topology builders (ring/linear/2D) + host
  orchestrators + oracles + `bit_exact`/PCC verifiers, driving the emule fabric-client shim directly. This
  decouples the fabric/CCL transport from the full tt-metal stack and from dispatch.
- **Then the real op** — a real ttnn fabric all-gather across 8 chips (quietbox-class descriptor) passing PCC,
  with reduce_scatter / all-to-all to follow. Capture which send-types / header variants actually appear.
- Cross-reference craq-sim `tests/multi_device_ccl/` as the harness model and `MULTICHIP.md` + the UMD
  `tt_sim_communicator.cpp` bridge as the integration reference.

## 11. Open questions

- **Which terminal NOC-command set the target ops use** — i.e. which send-types (write / inline / atomic-inc /
  fused / scatter), whether chip-multicast is exercised, and whether 1D (`LowLatency`) and/or 2D (`HybridMesh`)
  headers must be decoded. This bounds how much of §5's REPLICATE set Phase 1 must implement.
- **Global-semaphore + barrier path** — whether `create_global_semaphore` / `distributed::Synchronize` need
  special handling under the emule shim + fiber model.
- **Completion/credit handshake fidelity** — how faithfully `wait_for_empty_write_slot` / the open/close
  handshake must be faked to keep workers live without a real EDM buffer.

*(Resolved by [`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md): the fabric client API is a single
choke point grounded across both ttnn and tt-blaze — no op bypasses the `WorkerToFabricEdmSender` object with
raw eth; the shim surface is the full sender method set incl. the stateful path. Decided OUT of scope:
multi-hop routing and ethernet-core emulation; host-side mesh CCL and blaze disaggregation.)*
