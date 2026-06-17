# Fabric/eth/CCL op-coverage: which mocking level covers ttnn and tt-blaze ops

Status: **design / coverage analysis** (no code yet). This doc answers a single question across the *whole*
op surface of both **ttnn** and **tt-blaze**: at which interception level must emule mock to cover every
fabric/ethernet/CCL op, and do ttnn and blaze need the *same* level or different ones?

Companion to [`fabric-ccl-simulation.md`](fabric-ccl-simulation.md) (which recommends interception **B**, the
fabric-client-API shim) — this doc validates that recommendation against the op surface. Also a companion to
[`scaling-architecture.md`](scaling-architecture.md), [`ttsim-methodology.md`](ttsim-methodology.md),
[`craqsim-methodology.md`](craqsim-methodology.md).

**Headline:** ttnn and blaze use the **same mocking level** — the fabric client API
(`tt::tt_fabric::WorkerToFabricEdmSender`) is the single kernel-level choke point for *all* cross-chip
traffic in both — but **blaze exercises a wider surface of it**. A repo-wide sweep found **no** blaze op that
sends cross-chip by any means other than the `WorkerToFabricEdmSender` object (no raw `ETH_TXQ`/`eth_send`/
`EthernetConfig`/`erisc_datamover`), so the single-choke-point claim is grounded, not assumed.

## 1. Scope

- **In scope:** kernel-driven cross-chip ops (fabric/CCL) in ttnn (`ttnn/cpp/ttnn/operations/`, incl.
  `experimental/`) and blaze (`tt-blaze/blaze/ops/`).
- **Noted, not covered:** host-side mesh CCL (`ttnn::distributed::host_ccl`, `distribute_tensor`,
  mesh-sockets→MPI) and blaze **disaggregation** — these run via host UMD copies / MPI with no fabric kernels,
  so they need no fabric mock (host-side / multi-host story).
- **Altitude:** coverage/level analysis (matrices + conclusion + outliers), not a per-op bring-up plan.
- Path refs below are relative to `tt-metal/` or `tt-blaze/` as labelled.

## 2. The interception-level taxonomy

Cross-chip data descends through layers; an emulator could intercept at any one:

```
op host code
  └─ fabric client API   ← WorkerToFabricEdmSender / FabricConnectionManager / WorkerToFabricMuxSender /
     (THE choke point)      RoutingPlaneConnectionManager / socket_api.h        (tt_metal/fabric/hw/inc/edm_fabric/)
        └─ fabric router kernel (fabric_erisc_router.cpp)   [persistent ERISC]
             └─ ETH_TXQ registers → ethernet link
```

The recommended level (per `fabric-ccl-simulation.md`, given "no eth-core emulation, no multi-hop, teleport
OK") is the **fabric client API**: shim the worker→fabric sender, teleport to the final destination, run the
worker/forwarder kernels. This doc checks that this level covers every op.

## 3. ttnn coverage

**Single choke point = the fabric client API.** ~65+ fabric-touching op families all bottom out at
`WorkerToFabricEdmSender` (headers in `tt_metal/fabric/hw/inc/edm_fabric/`):

| Category | Examples | Cross-chip mechanism |
|---|---|---|
| Stable CCL (~9) | all_gather, reduce_scatter, all_reduce, all_broadcast, all_to_all_dispatch/combine, reduce_to_root, mesh_partition | fabric client API |
| Experimental CCL (~24) | all_gather_async (+ llama/matmul/concat/strided), reduce_scatter_minimal_async, all_reduce_async, all_to_all_async, slice_reshard_async, ring_attention_all_gather_async, rms_allgather, deepseek_moe_reduce_scatter | fabric client API |
| MoE / dispatch | moe_expert_token_remap, deepseek_prefill dispatch/combine | fabric client API |
| Distributed transformer | ring_distributed_sdpa, all_reduce_create_qkv_heads, dit_layernorm_pre/post | fabric client API |
| Point-to-point | point_to_point, send_async/recv_async | fabric client API (send/recv pairs it with a socket) |
| **Single-chip only (~12 families)** | eltwise, matmul, conv, embedding, kv_cache, loss, creation, … | **none — no fabric mock** |

**Trace:** `ttnn::all_gather` (`ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp`) → `all_gather_async`
writer (`.../experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp`) →
`FabricConnectionManager` → `WorkerToFabricEdmSender` → EDM/router → ETH_TXQ.

**Four client-API entry forms** the shim must cover in ttnn:
1. `FabricConnectionManager` (ring fwd/bwd) — most CCL ops.
2. raw `WorkerToFabricEdmSender[]` (no wrapper) — MoE all_to_all_dispatch/combine (4-direction mesh).
3. `WorkerToFabricMuxSender` (`tt_fabric_mux_interface.hpp`) — mux variant.
4. `WorkerToFabricEdmSender` + `socket_api.h` — send_recv_async (socket sequences, fabric transports).

## 4. tt-blaze coverage

tt-blaze (~162 ops: ~131 single-chip fused; ~31 cross-device/socket/intra-device) authors **no custom
eth/erisc kernels** — it reuses tt-metal's fabric + sockets. Cross-chip transport is **fabric, the same level
as ttnn**. The differences are in *which surfaces* of that transport it uses. (The sub-agent sweeps initially
over-simplified three points; each is reconciled against code below.)

### 4.1 Fabric ops via `RoutingPlaneConnectionManager`
Blaze cross-device ops (`cross_device_send`, `cross_device_signal`, `cross_device_top32_merge`,
`cross_device_topk_merge`, `all_gather`, `all_reduce`, `reduce_to_one`, `argmax`, `sdpa_reduce`, …) use
`tt::tt_fabric::RoutingPlaneConnectionManager`
(`tt-metal/tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp`) — an N-slot manager
holding one `WorkerToFabricEdmSender` per route/slot, plus `fabric_set_unicast_route` + FABRIC_2D
`dst_dev_id/mesh_id`. It is a *different wrapper* than ttnn's `FabricConnectionManager`, but the **same
underlying sender / interception level**.

### 4.2 Multi-hop is kernel-level relay (teleport-compatible)
*Reconciled (a sub-agent claimed blaze "needs real fabric multi-hop / teleport would fail" — wrong).* Reading
the kernels: `cross_device_send` (`build_sender_path`) routes by having **each device send to its adjacent
next hop**, and intermediate devices' kernels forward; `cross_device_topk_merge`'s step-2a/2b is two adjacent
sends across steps, not one router-forwarded multi-hop packet. So no fabric-router multi-hop is required —
a teleport-to-final-dest model that runs every device's kernel reproduces it.

### 4.3 Sockets are fabric-backed (blaze's primary inter-stage glue)
*Reconciled (a sub-agent claimed D2D sockets are "NOT fabric / raw NOC" — wrong).* Blaze embeds H2D/D2D/D2H
sockets in many layer ops and uses D2D sockets to forward between `pipeline_builder` stages. The D2D socket
kernels use `api/socket_api.h` (create/reserve/push/notify/wait pages + downstream encoding) + a local
`noc_async_write`. But **`tt-metal/tt_metal/hw/inc/api/socket_api.h:18-46` includes
`fabric_connection_manager.hpp`/`tt_fabric_api.h` and does `fabric_set_unicast_route(...)` from the socket's
`d2d.downstream_chip_id/mesh_id`** ("Communication over fabric only works for D2D sockets"); the host-side
`tt_metal/distributed/mesh_socket.cpp` is fabric-aware (`control_plane`, `get_fabric_node_id`). So **D2D
socket payload crosses chips over fabric** — the socket layer is a page/FIFO control abstraction *on top of*
the same fabric transport, not a separate transport. (emule already ships an `include/jit_hw/api/socket_api.h`
shim — confirm it covers the D2D fabric path.)

### 4.4 The `sdpa_reduce` worked example — stateful send + worker→forwarder split
*Reconciled (a sub-agent claimed "all sends use blocking `send_payload_*`" — wrong).*
`blaze/ops/sdpa_reduce/kernels/op.hpp` splits the work across two core roles:
- **Worker cores** hand-build the fabric packet header (`header->to_noc_fused_unicast_write_atomic_inc(...)`,
  `fabric_set_single_hop_unicast_route_from_direction`, `get_next_hop_router_direction`, `:140-154`) and
  **raw-NOC-stage** it (header + payload) into a *forwarder core's* L1 slot via `ncrisc_noc_write_with_state`
  + a semaphore bump (`send_packet`, `:157-182`). This is **intra-chip NOC** (emule-native), not a fabric send.
- **Forwarder cores** (`forwarder_impl`, `:735`/`:878`) do the cross-chip send via
  `RoutingPlaneConnectionManager.get(0).sender` using the **stateful** API:
  `setup_stateful_send_cmd_bufs` (`:749`) + `get_num_free_write_slots` (`:714`) +
  `send_current_slot_stateful_non_blocking_from_address` (`:724`) — **not** `send_payload_flush_blocking_from_address`.

Cross-chip transport is still `WorkerToFabricEdmSender` (same level), but a shim that only hooks the blocking
`send_payload_*` would **miss this op**. The shim must cover the full sender method surface, and decode a
header that was **built by a different core** and forwarded via slots.

### 4.5 Repo-wide grounding
A sweep of `blaze/ops` + `blaze/kernels`:
- **Stateful/forwarder pattern** (`setup_stateful_send_cmd_bufs` / `send_current_slot_stateful_*`): used by
  `all_gather`, `all_reduce`, **and** `sdpa_reduce` — the perf-critical CCL ops, not an edge case.
- **Blocking `send_payload_*` form:** used by `argmax`, `cross_device_send`, `cross_device_signal`,
  `cross_device_top32_merge`, `cross_device_topk_merge`, `reduce_to_one`, `sdpa`.
- **`WorkerToFabricEdmSender` methods actually used across blaze:** `wait_for_empty_write_slot`,
  `get_num_free_write_slots`, `send_payload_without_header_non_blocking_from_address`,
  `send_payload_flush_blocking_from_address`, `send_current_slot_non_blocking`,
  `send_current_slot_stateful_non_blocking_from_address`, `setup_stateful_send_cmd_bufs`, `open`, `close`.
- **No raw-eth bypass:** zero hits for `ETH_TXQ`/`eth_send`/`EthernetConfig`/`erisc_datamover` in blaze
  ops/kernels. **Every** blaze cross-chip send goes through the `WorkerToFabricEdmSender` object → the single
  choke point is grounded.

### 4.6 Out of scope: disaggregation
`tt-blaze/disaggregation/` is host-side **MPI** cross-HOST KV-cache migration (`EndpointTransport`,
`MpiSender` + local device DRAM reads) — not on-mesh fabric (relates to the multi-host story, like
`host_ccl`).

### 4.7 craq-sim coupling
Blaze's P150 CI gate runs on its embedded craq-sim submodule (pinned at a *main* commit, single-card;
multichip-on-craq-sim is a future dispatch-only lane). A few ops opt out of craq-sim (SFPU local-sort /
exponent-scaling not modeled). Loudbox (8-chip) tests are silicon-only today.

## 5. ttnn vs tt-blaze — same level, wider surface

| | ttnn | tt-blaze |
|---|---|---|
| Cross-chip choke point | `WorkerToFabricEdmSender` | `WorkerToFabricEdmSender` (same) |
| Custom eth/erisc kernels | none | none |
| Multi-hop | adjacent ring sends (kernel relay) | adjacent sends (kernel relay) — teleport-OK |
| Connection manager | `FabricConnectionManager`, raw `[]`, mux | + **`RoutingPlaneConnectionManager`** |
| Send API used | mostly blocking `send_payload_*` | blocking **and** **stateful** (`send_current_slot_stateful_*`) |
| Sockets | only `send_recv_async` | **pervasive** (D2D inter-stage), fabric-backed via `socket_api.h` |
| Packet headers | 1D / 2D | 1D `LowLatencyPacketHeader` **and** 2D `HybridMeshPacketHeader` |
| Core layout | worker sends directly | + **worker→forwarder-core split** (sdpa_reduce/all_gather/all_reduce) |
| Out of scope | host-side mesh CCL | + disaggregation (host MPI, cross-host) |

**Conclusion:** the mocking **level is the same** (the fabric client API is the single kernel choke point for
both). Blaze just exercises a **wider surface** of that level — `RoutingPlaneConnectionManager`, the stateful
send API, sockets-over-fabric, both header types, and the worker→forwarder split.

## 6. Outliers / parallel mechanisms

| Mechanism | Fabric kernels? | Verdict |
|---|---|---|
| Ethernet/mesh sockets | yes — `socket_api.h` is fabric-backed (§4.3) | covered by the fabric-client level |
| Host-side mesh CCL (`host_ccl`, `distribute_tensor`) | no — `DistributedHostBuffer` + MPI, no `CreateKernel` | host UMD copies; noted, not mocked |
| Blaze disaggregation | no — MPI cross-host + local DRAM reads | out of on-mesh scope (multi-host) |
| Legacy EDM (`ccl/kernels/edm/erisc_datamover.cpp`) | direct eth | out of scope (per scaling doc) |
| UDM (`tt_metal/fabric/hw/inc/udm/`) | no ttnn/blaze op uses it | watch item |
| Dispatch tunneling (`vc_eth_tunneler`) | FD command routing, not tensor data | out of scope (slow-dispatch) |

## 7. Implications for the emule shim

Cross-ref [`fabric-ccl-simulation.md`](fabric-ccl-simulation.md) §8. The recommended interception level
(fabric client API, interception B) is correct for **both** ttnn and blaze, but the shim's surface must be
broadened from "shim `send_payload_*`" to the **full `WorkerToFabricEdmSender` method set**:
- blocking: `send_payload_without_header_non_blocking_from_address`, `send_payload_flush_blocking_from_address`;
- stateful: `setup_stateful_send_cmd_bufs`, `send_current_slot_{non_blocking,stateful_non_blocking_from_address}`;
- flow control / lifecycle: `wait_for_empty_write_slot`, `get_num_free_write_slots`, `open`, `close`,
reached via any of `{FabricConnectionManager, raw WorkerToFabricEdmSender[], WorkerToFabricMuxSender,
RoutingPlaneConnectionManager, socket_api.h}`. The shim decodes `NocCommandFields` (1D and 2D headers,
possibly built by a different core), teleports to the final destination, and relies on emule running the
worker **and** forwarder kernels (the worker→forwarder leg is intra-chip NOC).

## 8. Open items

- **Broaden `fabric-ccl-simulation.md` §8** from "shim `send_payload_*`" to the full `WorkerToFabricEdmSender`
  surface incl. the stateful API (§7 list).
- Confirm emule's existing `include/jit_hw/api/socket_api.h` covers the D2D fabric path (§4.3).
- Which packet-header variants (1D `LowLatency` vs 2D `HybridMesh`) the target ops actually emit.
- The few craq-sim-unmodeled blaze ops (SFPU local-sort / exponent scaling).
- Disaggregation / multi-host as a future, out-of-scope track.

*Resolved this session: no blaze op bypasses the `WorkerToFabricEdmSender` object with raw eth — the single
choke point holds for both ttnn and blaze.*
