# ttsim multichip methodology (code-grounded)

Reference account of how **ttsim-private** (the upstream functional ISA simulator;
`/localdev/arminale/ttsim-private`, branch `main`) implements multi-chip, read from source. Companion to
[`craqsim-multichip-methodology.md`](craqsim-multichip-methodology.md) (the fork) and to the emule design
docs ([`multichip-scaling-design.md`](multichip-scaling-design.md),
[`fabric-ccl-simulation.md`](fabric-ccl-simulation.md)). Paths below are relative to the ttsim repo root.

**One-line answer:** ttsim is a *compile-time, single-hop* multichip simulator. It faithfully delivers raw
ethernet packets point-to-point and supports host-mediated remote access, but it has **no fabric router, no
multi-hop routing, and cannot run fabric-based CCL.**

## 1. Multichip substrate (chips, clock, API, topology)

- **N chips in one process, shared array.** `ChipState g_chips[NUM_CHIPS]`, an active-chip index
  `g_current_chip_id`, and a single global timebase `g_clock` (`src/sim.h:811-813`). `ttsim_select_chip(id)`
  just sets the index (`src/sim.cpp:75-78`); for `NUM_CHIPS>1`, macros redirect `g_t_tiles`/`g_e_tiles`/
  `g_dram` to `g_chips[g_current_chip_id]` (`src/sim.h:821-827`).
- **Lockstep BSP clock.** `libttsim_clock(n)` (`src/libttsim.cpp:767-786`) loops, and per iteration steps
  *every* chip one cycle (`for chip_id … { ttsim_select_chip; clock_current_chip(); }`) then increments the
  single `g_clock` once. All chips advance on one shared timebase.
- **Topology is compile-time, one `.so` per config.** `src/rules.py` `CHIPS` table builds distinct
  libraries — `wh`, `wh_x2` (n300), `wh_x8` (T3000 4×2), `wh_x32` (WH Galaxy 8×4), `bh`, `bh_x2` (P300),
  `bh_x32` (BH Galaxy) — baking `-DNUM_CHIPS` / `-DNUM_MMIO_CHIPS` / `-DTT_ARCH_VERSION`. You cannot change
  topology at runtime. The `tests/multichip/*.yaml` descriptors are documentation only — never loaded.
- **9-symbol C API** (`src/libttsim.map`): `libttsim_init/exit`, `set_pci_dma_mem_callbacks`,
  `pci_config_rd32`, `pci_mem_rd_bytes/wr_bytes`, `tile_rd_bytes/wr_bytes`, `clock`. There is **no**
  `create_device_by_id` / `select_device_by_id` / `switch_*` — multichip is entirely internal.
- **Host sees N PCIe devices** via a paddr stride `PER_DEVICE_PADDR_STRIDE = 64 GiB`
  (`src/libttsim.cpp:28-33`); `libttsim_pci_mem_*` divides paddr → device → `ttsim_select_chip`
  (`src/libttsim.cpp:494-507`).

## 2. Ethernet cores — modeled as state; executable only on explicit launch

ERISC cores are real `Rv32HartState` structs (the same core type as Tensix RISCs, `src/sim.h:146-175`),
but the **base ethernet firmware is stubbed**, not executed. In `e_tile_init` (`src/tile.cpp:204-275`):

- WH (`TT_ARCH_VERSION==0`): the firmware entry functions are replaced with a single `RET`
  (`riscv_ret_inst = 0x8067`) at a jump target (`0x440`); boot/version/link state is written to fixed L1
  offsets (`0x210`, `0x1104`, `0x1EC0…`).
- BH (`TT_ARCH_VERSION==1`): the three active-erisc function pointers (`0x7CF00/04/08`) are pointed at a
  `RET` stub; `port_status/train_status/rx_link_up` are written at `0x7CC..`.

There is **no ELF loader** for eth kernels, and IRAM is at most a memcpy mirror (no instruction fetch from it
in the normal flow).

**However, eth cores are not inert.** A host `RUN_MSG_GO` written to a *peered* eth tile's launch mailbox at
L1 `0x490` (`src/tile.cpp:3017-3069`) reads the kernel entry from the launch message, sets PC / sp
(`0xFFB02000`) / gp, and calls `ttsim_rv32_set_core_active('E', tile_id, 0, true)`. The RV32 core then steps
that "active-erisc" kernel via `rv32_step()` in the clock loop. So an ERISC core *can* execute a
host-launched kernel — but there is no **persistent fabric router** running on it, and non-peered eth tiles
stay inactive.

## 3. The router — there is none; compile-time peer table + direct delivery

ttsim has no fabric/router model. Inter-chip connectivity is a **compile-time table**:

```cpp
// src/tile.cpp:54-182
struct EthLink { uint8_t src_chip, src_tile, dst_chip, dst_tile; };
static constexpr EthLink ETH_PEER_TABLE[] = {
#if NUM_CHIPS == 2 // n300
    {0, 8, 1, 0}, {0, 9, 1, 1}, {1, 0, 0, 8}, {1, 1, 0, 9},
#elif NUM_CHIPS == 8  // T3000 4x2
    …
#elif NUM_CHIPS == 32 // WH Galaxy 8x4 torus
    …
```

`eth_peer(tile_id, &chip, &tile)` (`src/tile.cpp:186-197`) scans the table for `(g_current_chip_id, tile_id)`
and returns the wired peer (or false in single-chip builds). `chip_coord_to_id` (`src/tile.cpp:323`) maps 2D
mesh coords → linear chip id for the legacy-queue path.

Delivery is a **synchronous copy at the register write**, in `eth_txq_regs_wr32` (`src/tile.cpp:2016`):
- `ETH_TXQ_CMD == 2` (L1 block, `:2036-2055`): read transfer start/size/dest, `eth_peer()`,
  `ttsim_select_chip(remote)`, `memcpy(&g_e_tiles[remote_tile].sram[dest], &p_tile->sram[start], size)`,
  restore chip.
- `ETH_TXQ_CMD == 4` (MMIO, `:2057-2076`): write one 32-bit register into the peer eth tile via
  `tile_wr_bytes`.

There is **no NOC packet queue, no multi-hop forwarding, no credit/flow-control, no EDM**. A packet lands in
the *direct* peer's memory instantly; a destination two hops away is unreachable.

## 4. tt-metal fabric APIs — would fail

A worker using the tt-metal fabric client API (`WorkerToFabricEdmSender`) opens a connection to a fabric
router and then sends payloads through it. On ttsim that flow **stalls**:

1. The worker writes its launch config and waits for the fabric router to mark the connection live
   (a connection-live semaphore in L1).
2. **No fabric router runs** — ttsim has no persistent router kernel, and the base eth firmware is a `RET`
   stub. Nothing increments the semaphore.
3. The worker spins forever on `wait_for_empty_write_slot` / the connection handshake.

Even if a router kernel were *supplied* and launched via `RUN_MSG_GO`, it could not forward multi-hop: there
is no packet queue and no async-peer model — the router would have to push into a remote ERISC's RX queue,
which ttsim does not model. ttsim simply has no terminal NOC-command execution (no
write/atomic-inc/fused/scatter applied at a destination from a fabric packet) and no routing layer.

## 5. Running a CCL — fabric CCL cannot run; here is what can

**Cannot run:** any fabric-based collective (all_gather, reduce_scatter, all_to_all). These require a
persistent fabric router + multi-hop + terminal NOC-command semantics, none of which exist. The worker would
hang as in §4.

**Can run:**
- **Raw ETH_TXQ point-to-point** — two kernels on directly-wired chips exchanging packets via `cmd=2`/`cmd=4`.
- **WH legacy remote queue** (`wh_x2_legacy_remote_queue_update`, `src/tile.cpp:332-434`): host-mediated
  cross-chip read/write. `sys_addr` bitfields `[0:35]=local_addr, [36:41]=noc_x, [42:47]=noc_y,
  [48:59]=chip_x/y` → `chip_coord_to_id` → select + read/write; processing is triggered by the host writing
  the request write-pointer.
- **Link-state queries** — UMD topology discovery reads the faked boot-results state.

**Test inventory** (`tests/multichip/`):

| Test | Proves |
|---|---|
| `eth_link_init.cpp` (WH) / `bh_eth_link_init.cpp` (BH) | faked link-train / boot-results state per arch + topology awareness |
| `interchip_delivery.cpp` (WH n300) | host-mediated remote write+read via the legacy queue; per-chip isolation |
| `bh_interchip_delivery.cpp` (BH x2) | BH PCIe/TLB-based remote access |
| `wh_x8_interchip_delivery.cpp` | the peer table scales to a 4×2 mesh |
| `wh_x32_link_init.cpp` | Galaxy 8×4 boot-results correctness |

None of these run a fabric router or a fabric CCL — they exercise raw eth delivery, host-mediated remote
access, and link-state injection.

## 6. Capability summary

| Feature | Supported? | Code |
|---|---|---|
| Compile-time topology (`ETH_PEER_TABLE`) | ✅ | `src/tile.cpp:54-182` |
| Direct eth peer delivery (`cmd=2`/`cmd=4`) | ✅ | `src/tile.cpp:2036-2076` |
| WH legacy remote queue (host-mediated) | ✅ | `src/tile.cpp:332-434` |
| ERISC base firmware execution | ❌ (RET stub) | `src/tile.cpp:204-275` |
| Host-launched active-erisc kernel | ✅ (via `RUN_MSG_GO`) | `src/tile.cpp:3017-3069` |
| ELF loader for eth kernels | ❌ | — |
| Fabric router kernel | ❌ | — |
| Multi-hop forwarding / routing | ❌ | — |
| Fabric EDM (`WorkerToFabricEdmSender`) semantics | ❌ (would hang) | — |
| Fabric CCL (all_gather, …) | ❌ | — |
| Per-chip RISC clocking, single global clock | ✅ | `src/libttsim.cpp:767-786` |
| Per-device PCIe enumeration | ✅ | `src/libttsim.cpp:494-507` |

**Conclusion.** ttsim models the *ethernet link*, not the *fabric*. It delivers single-hop packets directly
between wired peers (and supports host-mediated remote access), with a compile-time topology and a single
lockstep clock. It has no fabric router, no multi-hop, and no terminal NOC-command semantics, so it cannot
run fabric-based CCL. Everything above the raw link — routing, the EDM client protocol, collectives — is the
fork (`craq-sim`)'s addition; see [`craqsim-multichip-methodology.md`](craqsim-multichip-methodology.md).
