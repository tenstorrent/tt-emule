<!--
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
-->
# Fabric & CCL emulation

How tt-emule emulates the Tenstorrent **fabric** (the inter-chip transport) and, as a consequence, runs
**CCL** collectives (`all_gather`, `reduce_scatter`, `point_to_point`, …) across a multi-chip mesh — n300 /
p300 (2 chips) and the 8-chip Blackhole loudbox (`blackhole_8xP150.yaml`, a 2×4 mesh). This document is
self-contained: it states both the **philosophy** of the approach and the **mechanics** of the current
implementation — what ships today.

## Philosophy: teleport, don't transport

emule's fidelity bar is **functional correctness (PCC), not timing**. For a collective, correctness depends
on two things: the **terminal effects** of each cross-chip message — the NOC write that lands the bytes, or
the atomic-increment that ticks a semaphore — and the **ordering** of the semaphore handshakes between
producers and consumers. It does *not* depend on *how* the bytes crossed the wire.

So emule throws the transport away and keeps the effect. It intercepts at the **fabric client API** (the
`WorkerToFabricEdmSender` surface that worker kernels call) — the highest, most semantic point, above the
router and the ethernet TX queues — and turns every send into a synchronous **teleport**: resolve the final
destination chip, apply the terminal NOC command directly into that chip's L1, and wake any fiber parked on
that address. Delivery is bit-exact and immediate.

Three principles follow:

- **One interception point.** *All* cross-chip traffic — every CCL and ttnn op alike — funnels
  through one shim and one teleport hook. There is exactly one place to get the cross-chip semantics right.
- **Single code path, faithful to silicon.** The terminal NOC commands are applied with the same semantics
  silicon would produce, and the multi-chip destination resolution *subsumes* the 2-chip case (2 chips = the
  degenerate distance-1 neighbor) rather than being a parallel path. emule never adds a divergent code path
  to "make a test pass."
- **Model the effect, skip the mechanism.** The router/EDM, multi-hop forwarding, packetization &
  buffer-size chunking, flow-control credits, mux back-pressure, ethernet-core execution, and link latency
  are all *transport mechanism* — they affect timing, not the result — so emule does not model them. The
  ethernet/ERISC cores never run.

The one thing emule *must* reconstruct that the hardware gets "for free": because there is no router to
forward packets hop-by-hop, the **final destination chip** has to be computed on the host before the
teleport. That host-side destination resolution (below) is the bulk of the fabric machinery.

## Inter-chip addressing — the foundation

Each `SWEmuleChip` owns its **own** L1 mapping. A kernel's 32-bit L1 address is a **0-based offset**; the
fabric shim narrows a header/payload pointer to its offset and the teleport widens it back with
`__emule_local_l1_to_ptr` onto the destination fiber's L1. It does **not** truncate the host pointer — the
offset survives worker L1 mapped above 4 GB, which a >8-chip galaxy requires (an earlier design relied on the
low-4 GB truncation being lossless; the offset model removes that ceiling). The corollary: the *same* L1
offset is a *different* host pointer on each chip, so cross-chip delivery is fundamentally a "find the right
chip's copy of this (core, offset)" problem. Two runtime hooks bridge it
(`tt_metal/impl/emulation/emulated_program_runner.cpp`):

- **`__emule_fabric_resolve_remote(dst_chip, noc_addr)`** — the teleport's address resolver. Decodes
  `noc_addr` → `(noc_x, noc_y)` + `local_addr`, looks up the **destination** chip's core map, and returns
  that core's `l1_ptr(offset)`. WORKER (tensix) coordinates are translated `src → logical → dst`
  (`get_soc_descriptor().translate_coord_to`) so per-chip **harvesting** differences map the same logical
  worker to the right physical core on the destination chip; DRAM/ETH coordinates are taken verbatim
  (identical across chips — translating them would collapse distinct banks). Worker offsets are masked to the
  per-core L1 slot.

## The teleport

`__emule_fabric_teleport(packet_header, payload, payload_size)` is the single delivery hook. It:

1. **Resolves the final destination chip(s)** — `__emule_fabric_resolve_targets(h, src)` returns one chip for
   a unicast, the line members for a multicast (see next section).
2. **Decodes the real-layout header** — `NocCommandFields`@0, `payload_size`@40, `noc_send_type`@42.
3. **Applies the terminal NOC command** to each target chip's L1 (`__emule_fabric_deliver`), then
   `__emule_fiber_wake(d)` so a peer fiber parked on that address (a CB poll or a `noc_semaphore_wait`) is
   requeued:
   - `0 NOC_UNICAST_WRITE` — `memcpy` payload → dst.
   - `1 NOC_UNICAST_INLINE_WRITE` — store a 32-bit value.
   - `2 NOC_UNICAST_ATOMIC_INC` — `fetch_add` (the CCL handshake counter).
   - `3 NOC_FUSED_UNICAST_ATOMIC_INC` — a payload write **and** a separate semaphore `fetch_add` (wakes both).
   - `4 NOC_UNICAST_SCATTER_WRITE` — per-chunk writes to several dst addresses.
   - `8 NOC_FUSED_SCATTER_WRITE_ATOMIC_INC` — a 2-chunk scatter write **and** a semaphore `fetch_add` (the
     fused reduce_scatter packet). Silicon folds the semaphore into a 3rd scatter chunk on send-type `4`;
     emule tags it with a dedicated internal send-type (multicast already isn't wire-faithful at `5/6`) so
     the delivery is a clean compose of the case-4 writes and the case-3 increment.
   - **multicast** (`5/6`, a line/range routing) — emule expresses it as the resolved target *list* and
     replays the same terminal command (write or atomic-inc) into each chip in the range. This is the CCL
     barrier broadcast.

The fabric **firmware** is never brought up: `skip_fabric_fw_for_emule()` short-circuits fabric init,
router-sync/configure, and teardown — the ERISC router would otherwise sync-timeout, exactly as for mock
devices. emule's runner only ever executes `CoreType::WORKER` programs.

## Destination resolution (the host's job)

Because there is no router, the teleport must know the **final** dst chip on the host. The kernel only knows
it *semantically* (an explicit 2D node id, or a 1D direction + hop distance), so emule captures that intent
and resolves it. Everything here is gated by `EMULE_FABRIC8`: **off** → the legacy single-neighbor path
(`__emule_fabric_neighbor`, the first ethernet neighbor — correct only for a directly-connected 2-chip
board, byte-for-byte the historical behavior); **on** → the route-table resolution below. Because the 2-chip
case is the degenerate distance-1 neighbor, gate-on subsumes gate-off (n300/p300 stay green either way; the
loudbox harness sets the gate).

Four pieces cooperate (all in `emulated_program_runner.cpp`, fed by the shim + a host hook):

- **Route table** — two ordered-chip walks, both cached, both *static topology lookups* (not per-hop router
  simulation). `__emule_fabric_walk` (compass): for each `(src_chip, RoutingDirection)`, the chips at distance
  1, 2, … reached by following `control_plane.get_chip_neighbors` in a fixed direction; used for 2D line
  multicast and as the 1D fallback. `__emule_fabric_walk_ring`: the 1D ring is a Hamiltonian cycle that
  *snakes* through the mesh (members are physically adjacent but the cycle turns at the mesh edges, so the
  fixed-direction compass walk dead-ends after one row), so the ring walk instead chains the per-chip recorded
  ring neighbors (`g_conn_route`, below), at each hop taking the neighbor that is not the one we came from —
  following the real ring all the way around. `walk[0]` is the same neighbor either way, so distance-1 relays
  are identical; the walks differ only in multi-hop reach (e.g. a barrier multicast spanning the whole ring).
- **Per-send route metadata** — the kernel's route setters (`fabric_set_unicast_route` /
  `fabric_set_mcast_route`, shimmed) record each send's intent via `__emule_fabric_set_route`, keyed by the
  packet-header's L1 address (stable and identical between the shim's write and the teleport's read). Kinds:
  `UNICAST_2D` (explicit `dst_dev`/`dst_mesh`), `MCAST_2D` (per-direction `{e,w,n,s}` hop counts), `UNICAST_1D`
  (hop distance), `MCAST_1D` (start + range).
  emule gives the 1D (`LowLatencyPacketHeader`) and 2D (`HybridMesh`/`UDMHybridMesh`) packet-header C++ types
  **distinct** definitions — the 2D variants are empty-derived from the base with the identical 48B layout —
  and selects `PACKET_HEADER_TYPE` from the build-time `EMULE_FABRIC_2D` define. So the kernel's `if constexpr`
  type dispatch (`worker_routing_utils`' `fabric_set_line_unicast_route` / `fabric_set_line_multicast_route`,
  moe_utils' portable send helpers) picks the branch that matches the fabric config, exactly as silicon does:
  under a 1D fabric the hop-distance branch — `fabric_set_unicast_route<false>(hdr, distance)` → `UNICAST_1D`;
  under 2D the explicit-destination branch — `fabric_set_unicast_route(hdr, dst_chip, dst_mesh)` → `UNICAST_2D`
  and `fabric_set_mcast_route(hdr, dst_chip, dst_mesh, e, w, n, s)` → `MCAST_2D` (the teleport resolves the
  exact physical chip / line members).
  **The route-setter shims dispatch on *arity*, not on a fixed header-pointer parameter type.** This matters
  because the 2D header is *derived* from the base 48B `PacketHeader`: a shim overload taking a fixed
  `volatile PacketHeader*` would require a derived→base conversion for a 2D `HybridMeshPacketHeader*` argument
  and therefore **lose** overload resolution to the exact-match variadic catch-all — silently dropping the
  route (kind `UNSET`), so the teleport falls back to the single physical neighbor (the wrong chip on a
  submesh line, → a quiescent barrier deadlock). Folding both the 1D `(hdr, distance)` [2-arg] and 2D
  `(hdr, dst_chip, dst_mesh)` [3-arg] / `(hdr, dst_chip, dst_mesh, e, w, n, s)` [7-arg] shapes into one
  arity-dispatched variadic per setter keeps every header type an exact match, so the route is always stamped.
- **Per-connection direction** — for 1D the dst chip is bound to the *connection* (which ethernet channel it
  opened on), information that lives in the firmware connection table emule skips. emule reconstructs it
  host-side: `append_fabric_connection_rt_args` (which runs unguarded in emule) records each connection's
  `(forwarding_direction, neighbor)` via `__emule_fabric_record_conn` (compiled only under
  `TT_METAL_USE_EMULE`) — both src-keyed (`g_conn_route`) and per connection-owner core (`g_mux_dir`, keyed
  by the mux core's *logical* coords). The tables are **reset per op** (`execute_program_emulated` marks them
  dirty; the next op's first record clears them) — without this, a later op that gives a chip a *different*
  line orientation would corrupt the ordering. The signals are recovered **entirely emule-side** (no device
  kernel is modified — emule must run the same kernel silicon runs), in precedence order:
  1. **Mux core → direction** — the *mux* path: the worker connects to a direction-specific mux core, so the
     sender carries the mux's NOC coords (`build_connection_to_fabric_endpoint`); the teleport translates
     them back to the mux's logical core and looks up the direction the mux→EDM append recorded in
     `g_mux_dir`. This resolves rings (all_gather / reduce_scatter), where the fallback below cannot.
  2. **Connection open-sequence index** — the *direct 4-directional path* (`open_direction_connections_async`
     — `all_to_all_dispatch` et al.) has no mux core. It opens its connections once, in the same order the
     host recorded them into `g_conn_route` (both walk the active directions in compass order). So emule's
     `WorkerToFabricEdmSender::build_from_args` tags each sender with the fiber's next *open-sequence index*
     (`ThreadCommonCtx::fabric_open_conn_seq`, a per-fiber counter incremented on each build, fresh per
     launch); a send stamps that index as the route's `dir_index`, and the teleport reads
     `g_conn_route[src][dir_index]` for the connection's direction + neighbor. Recovered entirely emule-side;
     the kernel is not tagged.
  3. **Fallback** — the fwd/bwd index from `FabricConnectionManager::get_{forward,backward}_connection`, else
     **range-matching**: a line multicast's range equals the walk length of the worker's direction, cached per
     `(src_chip, worker_core)` in `g_worker_dir`. Used only when the signals above are absent (so
     previously-green configs are byte-for-byte unchanged).
- **Resolve** — `UNICAST_2D` → `get_physical_chip_id_from_fabric_node_id(FabricNodeId{mesh, dev})`;
  `MCAST_2D` → the compass walk per non-zero `{E,W,N,S}` hop count; `UNICAST_1D` / `MCAST_1D` → the 1D ring
  walk in the worker's direction (compass fallback if the connection chain is incomplete), taking `distance`
  for unicast or hops `[start, start + range)` for multicast.

## The fabric client-API shim

Kernels are intercepted at the worker→fabric-client boundary, not the router. The shim is
`include/jit_hw/__emule_fabric_stubs.h`, reached through the shadowed fabric headers under
`include/jit_hw/{fabric,tt_metal/fabric}/`:

- **`PacketHeader`** — a 48 B shim whose field offsets match silicon (`NocCommandFields`@0 /
  `payload_size_bytes`@40 / `noc_send_type`@42); its `to_noc_*` setters write exactly the fields the teleport
  decodes. It duplicates the real `fabric_edm_packet_header.hpp` layout rather than including it (the real
  header drags in HW intrinsics and a host-path `TT_THROW`). Route metadata is *not* stored in the header —
  it goes through the address-keyed side-table above — so the header stays 48 B and the `PacketHeaderPool`
  slot stride is unchanged.
- **`WorkerToFabricEdmSender` / `WorkerToFabricMuxSender` + connection managers**
  (`FabricConnectionManager`, `RoutingPlaneConnectionManager`) — their `send_payload_*` and stateful
  (set-state then send-from-address) paths all route to the single teleport hook. The mux sender is just an
  `WorkerToFabricEdmSender` subclass: it teleports directly, so the mux relay is collapsed away.
- **`PacketHeaderPool`** (`tt_metal/fabric/hw/inc/packet_header_pool.h`) — hands out header storage from the
  reserved L1 region via `__emule_local_l1_to_ptr`, so a header lives in the running fiber's **real** worker
  L1 (kernels dereference `header->field`). The kernel/shim narrows `(uint32_t)header` to that header's
  **0-based L1 offset** (`ptr - bridge_l1`); the teleport re-widens it with `__emule_local_l1_to_ptr`. The
  route table is keyed by the header's **full host pointer** (`bridge_l1 + offset`) — unique per
  *(chip, core, offset)*. The offset alone is **both** chip- and core-agnostic (0-based within every core's
  L1), so an offset key — even chip-qualified — collides across cores of one chip (one core's route
  overwriting another's → wrong-chip delivery → PCC fail / a fiber waiting on an atomic-inc that lands
  elsewhere → deadlock); the full host pointer restores the per-core uniqueness the old truncated-pointer key
  had, untruncated so it survives worker L1 mapped above 4 GB (the key is host-side runner state, never a
  kernel/L1 value). Partitioned per RISC by `__emule_self->processor_id`.
  - **Per-fiber allocation state:** the allocation cursor and route table (`phdr_cursor` / `phdr_route_*`)
    live in the running fiber's `ThreadCommonCtx`, which the runner allocates fresh per program launch. That
    makes the pool reset per launch — mirroring silicon, where each program gets a fresh pool (zeroed kernel
    `.bss`). A `thread_local` here would instead leak the cursor + route-id table across ops on a persistent
    worker: the cursor overflows the per-RISC partition and the route-id table wraps, silently corrupting the
    fabric multicast routes (the failure mode behind [#221](https://github.com/tenstorrent/tt-emule/issues/221)).

## How CCLs run as a result

CCL ops are ordinary ttnn programs whose kernels call the fabric client API; the teleport makes the
cross-chip writes land, and the mesh command queue runs them through the fiber engine's register/run split:

- **all_gather** — one collective program spanning the mesh; each chip's writer teleports its shard (and, on
  a line, forwards received shards) to its peers, with a multicast atomic-inc barrier to synchronize. One
  program → one fiber `run_until_idle`.
- **point_to_point** — a **sender** program on one chip and a **receiver** on another; the receiver blocks on
  a semaphore the sender increments over fabric. Because they are distinct programs, the mesh command queue
  brackets the **whole workload** (`begin_mesh_dispatch()` … `run_mesh_dispatch()`): every program is
  *registered* (fibers spawned, deferred) before a single `run_until_idle`, so both sides co-run in one
  scheduler generation and the teleport's `__emule_fiber_wake` reaches the already-parked receiver. (See
  [`fiber-engine.md`](fiber-engine.md) §9.)
- **reduce_scatter / all_to_all** — compose the same teleport + handshake, plus an on-chip reduction.

## Faithful vs simplified

| Faithful (emulated exactly) | Simplified / not modeled |
|---|---|
| Terminal NOC semantics (write / inline / atomic-inc / fused / scatter) | The ERISC/ethernet router (never runs) |
| Bit-exact L1 delivery | Multi-hop packet forwarding (replaced by host dst resolution) |
| Semaphore-handshake ordering (producer/consumer wake) | Packetization & buffer-size chunking |
| Harvesting-correct worker-core resolution on the dst chip | Flow-control / credits, mux back-pressure |
| Multicast as per-chip replay of the terminal command | Link latency / timing |

## Current state (8-chip loudbox)

Measured on `blackhole_8xP150.yaml` (slow dispatch). The LoudBox CCL suites are gated upstream by the
scheduled **`blackhole-e2e-tests`** workflow (`bh_loudbox` sku, every 8 h) — legs `bh-ttnn-ops-fast-unit`
(`box/all_post_commit`) and `bh-ccl-nightly-integration` (`box/nightly`), both **green on real hardware**
(latest: apc 116 passed / 0 failed; nightly success). Every emule failure below is an emule gap, not a
silicon/CI failure; the emule harness (`scripts/run_ttnn_pytests_bh_loudbox.sh`) runs the same two legs.

- **Green:** `all_gather` 2D (`*_2d_fabric`, `*_turning`, PCC 1.0) and 1D `all_gather` for **26/26** `apc`
  configs — **Linear** and **Ring**, both impls, multi-worker / multi-link, including the sub-tile-width /
  row-major / `WIDTH_SHARDED` gemma shapes (`test_all_gather_failing_shapes`, `test_all_gather_broken`); and
  `reduce_scatter` **16/16** (Linear + Ring, PCC ~1.0). On `box/nightly`, `all_gather` is **54/54** and the
  2D + `minimal_*` suites are green. The **DRAM-resident** `FABRIC_2D` / fabric-MUX configs
  (`all_gather_2d_fabric[_turning]`, `reduce_scatter_2d_fabric`, `minimal_all_gather_async`,
  `minimal_reduce_scatter_async`) are green too — they exercise the arity-dispatched 2D route setters that
  stamp the barrier atomic-inc's route for the derived `HybridMeshPacketHeader`. n300/p300 2-chip CCL stays
  green (the subsumed degenerate path).
- **Direction resolution** — a 1D send's direction comes from the mux-core → direction lookup (`g_mux_dir`;
  the sender carries the mux's NOC coords, the teleport translates them back), with range-matching as a
  fallback. Routes and atomic-incs resolve to the correct chips (e.g. `src=1 →N→ 5`, `src=5 →S→ 1`;
  atomic-incs land 0→1).
- **1D ring order** — the 1D ring is a Hamiltonian cycle that turns through the 2×4 mesh, so a fixed-compass
  walk can't follow it. The ring walk chains the per-chip recorded ring neighbors (the same edges all_gather
  relays over), so the `reduce_scatter` `batch_ready_sem` barrier multicast (range `ring_size-1`) reaches all
  members in ring order. `reduce_scatter`'s stateful scatter-write honors the `UpdateMask` (patches only the
  named fields, preserving the `set_state` payload size), mirroring `api_common.h
  populate_unicast_scatter_write_fields`.
- **Broadcast-based `all_gather`** — the "semaphore-free" `all_gather` (`ttnn.all_gather`) and the
  experimental path both lower, for these shapes, to the **broadcast** program factory whose writer
  (`broadcast_tile_writer.cpp`) issues fabric **multicast** writes (`fabric_multicast_noc_{unicast,scatter}_write_*`)
  over a routing-plane connection. Two properties make it faithful under emule: (1) the multicast
  write / scatter `set_state` shims stamp the `MCAST_1D` route (`start`,`range`) exactly as the atomic-inc
  form does, so the teleport replays each data page to every chip in the line (without the stamp the header
  carried no route and fell back to the single physical neighbor — the wrong chip on a >2-chip mesh); and
  (2) the `PacketHeaderPool` cursor + route table are homed in the per-fiber `ThreadCommonCtx` (a fresh
  object per program launch), so the pool resets per launch like silicon's `.bss` rather than leaking the
  cursor/route-id across ops on a persistent worker. Together these bring the sub-tile-width / row-major /
  `WIDTH_SHARDED` gemma shapes to PCC 1.0.
- **all_to_all_dispatch** — the first consumer of the **direct 4-directional** fabric path
  (`open_direction_connections_async`). Three shims make it faithful under emule: (1) the *stateless*
  `linear::experimental::fabric_unicast/multicast_noc_unicast_atomic_inc` free functions — the portable path
  its cross-device init-semaphore barrier lowers through — configure the command and teleport like
  `linear/api.h`; (2) the distinct 1D/2D header types (above) let its `fabric_set_line_unicast_route` take the
  1D hop-distance branch under a 1D fabric, with the variadic route shim stamping `UNICAST_1D`; (3) the
  per-connection direction comes from the connection open-sequence index (above). The init barrier, per-token
  data relays, and metadata write+semaphore all land on the correct chips. `_trace` variants are
  auto-deselected under emule (trace capture needs fast dispatch).
- **Out of emule scope (need fast dispatch):** `all_reduce`, `all_broadcast`, `all_to_all`, the MoE
  dispatch/combine ops, fused matmul-CCL, and `all_gather`'s `subcore_grid` partition CCL/compute cores via a
  **sub-device manager**, which `TT_FATAL`s under slow dispatch (`sub_device_manager_tracker.cpp:98`) — the
  largest coverage gap (~35 `apc` + ~190 `nightly` configs). Every `_trace`/perf variant likewise needs fast
  dispatch (trace capture is a fast-dispatch feature; auto-deselected under emule). `deepseek_ccl_ops` is a
  separate JIT-shim gap (`is_ncrisc`). All are CI-green on hardware.

## Other limits & notes

- **DRAM-resident CCL** — faithful on both fabric paths. The 2-chip *direct* path is bit-exact after the
  DRAM bank-view fix in the fabric address recompose (`safe_get_noc_addr` no longer applies the 2 MB
  worker-slot mask to DRAM addresses). The 8-chip loudbox `FABRIC_2D` / fabric-MUX DRAM configs
  (`all_gather_2d_fabric[_turning]`, `reduce_scatter_2d_fabric`, `minimal_all_gather_async`,
  `minimal_reduce_scatter_async`) run without deadlock once the 2D route setters stamp the route for the
  derived `HybridMeshPacketHeader` (the arity-dispatch fix above) — before it, the unstamped barrier
  atomic-inc fell back to the physical neighbor (the other mesh row, outside the collective's submesh line),
  so the consumer's barrier semaphore was never incremented and every fiber parked.
- **Inert value-divergent stubs** — `get_fabric_max_packet_size()`→4096, `get_next_hop_router_direction`→EAST,
  `eth_chan_to_noc_xy` zeroed. Not load-bearing: the teleport resolves the destination from the control plane
  and the per-connection record, never from the kernel's (firmware-skipped) routing-table reads.
- **Scaling past one mapping window** — offset addressing decouples the fabric's L1 addresses from where
  worker L1 is mapped, so the single-process design can hold a mesh larger than 4 GB (a >8-chip Blackhole
  galaxy) once the worker-L1 mmap moves above the low-4 GB window (FULL-PLAN Phase 3); no shared-memory /
  multi-process transport is needed behind the `__emule_fabric_teleport` seam.

The live workarounds (WA-1 mux no-op, DM-1 chip-relative remap) are catalogued in
`.claude/skills/workarounds/`.
