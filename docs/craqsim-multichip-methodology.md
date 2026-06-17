# craq-sim multichip methodology (code-grounded)

Reference account of how **craq-sim** (the heavily-modified ttsim fork that is tt-metal's production
`SIMULATION` backend; `/localdev/arminale/craq-sim`, branch `main`) implements multi-chip, read from source.
Companion to [`ttsim-multichip-methodology.md`](ttsim-multichip-methodology.md) (the upstream sim) and to the
emule design docs ([`multichip-scaling-design.md`](multichip-scaling-design.md),
[`fabric-ccl-simulation.md`](fabric-ccl-simulation.md),
[`fabric-ccl-op-coverage.md`](fabric-ccl-op-coverage.md)). Paths below are relative to the craq-sim repo root
unless prefixed. Authoritative narrative cross-check: `MULTICHIP.md`.

**One-line answer:** craq-sim is a *runtime, instruction-accurate, fabric-complete* multichip simulator. It
runs real ERISC/worker kernels on simulated RV32 cores, runs the **real tt-metal fabric router** (with real
multi-hop), and runs **real tt-metal CCL** end-to-end. It also exposes a faster host-side "teleport" path that
reproduces only terminal NOC-command semantics for unit tests.

## 1. Multichip substrate (runtime devices, clock, switch ABI)

Where ttsim bakes topology at compile time, craq-sim is **runtime-dynamic**:
- **Device registry.** `libttsim_create_device_by_id` / `libttsim_select_device_by_id` manage N heap
  `Device`s in one process; `device_set_active(d)` swaps `thread_local` aliases (`g_t_tiles`/`g_e_tiles`/
  `g_dram`/`g_clock_ptr`, `src/sim.h:918-949`) to the active device.
- **Clock + per-cycle drain.** `libttsim_clock_all_devices(n)` (`src/libttsim.cpp:2769-2819`): per cycle,
  step every registered device (`device_set_active` + `clock_one_cycle_locked(1)`), then `eth_switch_drain()`
  once. `libttsim_clock_devices` (`:2629`) is the explicit-list variant used by the in-process harness.
- **Full dynamic switch ABI** (vs ttsim's 9 symbols), e.g. `libttsim_switch_register`/`_peer`/`_drain`,
  `configure_eth_link_virtual`, `register_fabric_node_id`/`_endpoint_direction`/`_route`,
  `enqueue_fabric_packet`, `load_{erisc,tensix}_elf`, `eth_release_reset`, `clock_all_devices`
  (`src/libttsim.map`).

## 2. Ethernet cores — fully modeled, real RV32 execution

`EthTile` (`src/sim.h:768-837`) holds real ERISC `Rv32HartState` core(s), `erisc_iram`, `rv32_local_ram`,
the txq/rxq/mac registers, `soft_reset_0`, and `ierisc_reset_pc`. Unlike ttsim, eth kernels are **loaded and
executed**:

- **ELF load** — `libttsim_load_erisc_elf` (`src/libttsim.cpp:2869-2956`) parses the RV32 ELF and routes
  `PT_LOAD` segments to `erisc_iram` / `sram` / `rv32_local_ram[0]`, then sets `ierisc_reset_pc = e_entry`.
- **Reset release** — the `SOFT_RESET_0` MMIO handler (`src/tile.cpp:3093-3137`) clears the reset bit, sets
  `pc = ierisc_reset_pc` and `sp = 0xFFB02000`, and `ttsim_rv32_set_core_active('E', tile_id, 0, true)`. (Or
  the direct helper `libttsim_eth_release_reset`, `:3174-3199`.)
- **Real execution** — the active core is stepped each cycle. Proof it is genuine instruction execution: the
  kernel's store to `ETH_TXQ_CMD` runs *in the RV32 CPU*, and only *then* does the MMIO handler observe it
  (§3). Nothing is intercepted before the CPU executes.

`libttsim_load_tensix_elf` (`:2997-3065`) is the analogous loader for worker (BRISC) kernels (installing a
JAL trampoline at SRAM[0] when the entry ≠ 0).

## 3. The router — two paths

### Path A — the real fabric router (multi-hop)

When the loaded ERISC ELF is tt-metal's `fabric_erisc_router`, the RV32 core runs it. The router's eth sends
are MMIO writes to `ETH_TXQ_*`, caught by `eth_txq_regs_wr32` (`src/tile.cpp:2428-2553`): for `CMD==2` it
assembles an `EthMsgHeader` (dest MAC from the `eth_txpkt_cfg` table, dest addr, size) and calls
`e_tile_send_transaction` (`:4443`, which drives `e_tile_send_partial_transaction` `:4452`) →
`eth_switch_enqueue_with_source`. The switch teleports the packet to the destination eth tile's SRAM.

**Multi-hop is real.** Each chip runs its own router ELF: router A's send lands in router B's eth SRAM;
router B's RV32 core reads the packet, decrements the hop index, and writes `ETH_TXQ` again to forward to
router C, and so on. The known full-32-chip Galaxy failure — an `EBREAK` in the generated router ELF at
`update_packet_header_for_next_hop` (zero-route-array) — is direct evidence that the real router is genuinely
performing hop-by-hop routing (`MULTICHIP.md`).

### Path B — terminal helper (inactive ERISC, host-inject)

For tests (and where no router firmware is active), craq-sim decodes the fabric packet host-side and applies
its *terminal* effect directly, skipping the router and multi-hop:
- `eth_switch_enqueue_fabric_packet(src_chip, routing, command_40b, send_type, payload_size, payload)`
  (`src/eth_switch.cpp:1116`) decodes the 44-byte fabric header (a **stub noc-address encoding** that packs
  chip_id in the high bits — explicitly *not* real silicon; `docs/v3_5_commit15_fabric_atomic_layout.md`).
- `maybe_apply_bh_fabric_terminal_local_delivery` (`src/tile.cpp:3805-4439`) applies the terminal NOC command
  at the final destination: unicast-write; atomic-inc (`:4117-4177`); fused write+atomic
  (`src/eth_switch.cpp:1215-1265`); scatter (`:1277-1373`). A `hop_cmd` dispatch (`drain_locally` vs
  `push_fwd`) decides local delivery vs forward; atomics are applied at the drain barrier.

### EthSwitch internals (`src/eth_switch.cpp:86-152`)

```cpp
struct SwitchState {
    std::unordered_map<uint64_t, Endpoint> mac_table;                 // MAC -> (device, tile)
    std::unordered_map<uint64_t, std::queue<EnqueuedPacket>> fifos;   // per-endpoint
    std::unordered_map<uint64_t, Endpoint> fabric_routing_table;      // (chip,x,y) -> endpoint
    std::unordered_map<uint64_t, Endpoint> peer_map;                  // src endpoint -> peer
    std::unordered_map<uint64_t, uint32_t> fabric_endpoint_direction_map;  // -> MY_DIRECTION
    std::unordered_map<void*, FabricNodeId> fabric_node_id_map;       // device -> (mesh,chip)
    std::vector<FabricEndpointEntry> fabric_endpoints;
};
```

`endpoint_key(dev, tile)` (`:137`) keys the FIFOs/peer-map. `eth_switch_enqueue` looks up the dest MAC (with
a **source-aware peer fallback** for BH BCAST/MCAST selector MACs); `eth_switch_drain` delivers
time-eligible packets (latency-gated by `delivery_at_clock`) once per cycle. `eth_switch_compute_route_to_chip`
(a BFS over peers) exists and is invoked when a fabric packet's `hop_cmd` is NOOP, but normal delivery is
direct teleport.

## 4. tt-metal fabric APIs under craq-sim

In the production path, **real tt-metal kernels run on craq-sim's simulated cores**, so the fabric APIs work
by virtue of running the actual code:
- A worker's `WorkerToFabricEdmSender::send_payload_*` (real tt-metal) executes on a simulated Tensix core
  and writes `ETH_TXQ_*`; those writes hit `eth_txq_regs_wr32` → `e_tile_send_transaction` →
  `eth_switch_enqueue_with_source` (§3). The fabric router ELF (running on ERISC) consumes/forwards them.
- **UMD wiring.** `TTSimCommunicator` (`tt-metal/tt_metal/third_party/umd/device/simulation/tt_sim_communicator.cpp`)
  `dlopen`s one shared `libttsim.so` and `dlsym`s the switch ABI; `cluster.cpp`'s eth pre-pass (for
  `ChipType::SIMULATION`) walks the cluster descriptor's ethernet connections and calls
  `libttsim_configure_eth_link_virtual` (`src/libttsim.cpp:584`), `libttsim_switch_register_peer` (`:537`),
  and the fabric registration `libttsim_switch_register_fabric_node_id` (`:558`) /
  `_endpoint_direction`. This is how the runtime topology + fabric identity reach the switch.
- **Kernel loading.** `libttsim_load_tensix_elf` (workers) + `libttsim_load_erisc_elf` (fabric router), then
  release-reset (§2) launches them.

## 5. Running a CCL — two paths

### Real path (v3.5)

`tt-metal CCL pytest → tt-umd → libttsim`. tt-umd does one shared `dlopen`, creates a device per descriptor
chip (`create_device_by_id`), and `select_device_by_id` before each I/O. After device start, the eth pre-pass
wires peers/links (§4); CCL worker ELFs load on Tensix and the fabric router ELF loads on ERISC. Execution:
host pumps `libttsim_clock_all_devices` — each cycle steps every chip's Tensix + ERISC cores and drains the
switch. Workers send over fabric (ETH_TXQ → switch), the router forwards multi-hop, and chips coordinate via
global semaphores + fabric atomic-inc. `MULTICHIP.md` records the green matrix — all-gather / reduce-scatter /
broadcast / all-to-all on 2-chip P300, 4/8-chip P150, and Galaxy 2/4/8-device submeshes (PCC ≈ 1.0) — and the
limit: the full 32-device Galaxy reaches op execution then hits the fabric-router `EBREAK`.

### In-process harness (no tt-metal)

`tests/multi_device_ccl/` exercises CCL semantics without the tt-metal stack, driving ETH_TXQ MMIO host-side:
- `topology.h` — ring / linear / 2D-mesh / 2D-torus builders (per-chip MACs).
- `ccl_primitives.h` — host orchestrators, e.g. `all_gather_ring` = `N-1` rotations, each = `mac_setup` (per
  chip) + `N × send_data` + `clk(devs, N, 1)`.
- `mac_setup.h` — `mac_setup` programs the `ETH_TXPKT_CFG` MAC table; `send_data` writes the ETH_TXQ transfer
  descriptor + `CMD=2`, which reaches `eth_txq_regs_wr32` and enqueues to the switch.
- `oracle.h` + verifiers — `all_gather` = concat shards, `reduce_scatter`/`all_reduce` = per-slot sums;
  `bit_exact` (uint32) or `comp_pcc` (bf16/fp32).

This path is the model emule should mirror for fast functional CCL microtests (it drives the switch directly,
no kernels, no dispatch).

## 6. Capability summary

| Feature | craq-sim | Code |
|---|---|---|
| Runtime device registry | ✅ `create/select_device_by_id` | `src/libttsim.cpp` |
| ERISC real RV32 execution | ✅ (ELF load + reset + step) | `src/libttsim.cpp:2869`, `src/tile.cpp:3093` |
| Real fabric router (multi-hop) | ✅ (Path A; 32-chip EBREAK) | `src/tile.cpp:2428`, `MULTICHIP.md` |
| Host-inject terminal delivery | ✅ (Path B; teleport) | `src/eth_switch.cpp:1116`, `src/tile.cpp:3805-4439` |
| Terminal NOC commands (write/atomic/fused/scatter) | ✅ | `src/eth_switch.cpp:1215-1373`, `src/tile.cpp:4117-4177` |
| EthSwitch (MAC/peer/fabric tables, drain) | ✅ | `src/eth_switch.cpp:86-152` |
| Runtime topology from cluster descriptor (UMD) | ✅ | `tt_sim_communicator.cpp` + `cluster.cpp` |
| Real tt-metal CCL end-to-end | ✅ (P300/P150/Galaxy submesh) | `MULTICHIP.md` |
| In-process CCL harness | ✅ | `tests/multi_device_ccl/` |
| Full 32-device Galaxy CCL | ⚠️ fabric-router EBREAK | `MULTICHIP.md` |

**Conclusion.** craq-sim implements the full stack ttsim lacks: real eth-core execution, a real multi-hop
fabric router, the EthSwitch transport, runtime topology from the cluster descriptor, and real tt-metal CCL.
Its host-inject "teleport" path (Path B) is a faster shortcut that reproduces only the router's *terminal*
NOC-command semantics — and that path is the closest existing analogue to what emule should build (see
[`fabric-ccl-simulation.md`](fabric-ccl-simulation.md)), with emule intercepting even higher, at the fabric
client API, to avoid running eth cores at all.
