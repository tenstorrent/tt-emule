# DRAM Emulation in tt-emule

How tt-emule emulates **interleaved / banked DRAM**. Read this before debugging a
DRAM-addressing bug (wrong tile, zeros, off-by-bank), extending the addr-gen, or
adding a new arch's bank topology.

On silicon, a tensor in DRAM is striped across several banks; a kernel reaches a
page via an interleaved address generator that maps a flat page id to a (bank,
bank-local offset) pair, then to a NOC address. Emule reproduces the same tile→
bank→NOC-address math and resolves the NOC address to host memory.

Companion docs: [noc-emulation.md](noc-emulation.md) (the NOC address encoding +
resolver this builds on), [l1-emulation.md](l1-emulation.md) (the per-core
sibling).

---

## 1. Emulation model

DRAM banks are emulated as `tt_emule::Core` objects with `role_ == DRAM`. The
bank topology (how many banks, which NOC core each maps to, each bank's base
offset) is **populated at program-execution time from the real
`metal_SocDescriptor`** — so the same banking layout firmware sees is what emule
strides over. The interleaved addr-gen runs the real per-bank math; the resulting
NOC address is resolved to a host pointer and `memcpy`'d (slow dispatch — see
[noc-emulation.md](noc-emulation.md) §1).

---

## 2. Bank topology

`populate_bank_mapping` (`tt-metal/.../emulated_program_runner.cpp`) fills
runtime-injected global tables, then emits the bank-count JIT defines:

```cpp
num_dram_channels = metal_soc.get_num_dram_views();         // dynamic, per arch
for (ch in [0, num_dram_channels)) {
    auto dc0 = metal_soc.get_preferred_worker_core_for_dram_view(ch, /*noc*/0);
    auto dc1 = metal_soc.get_preferred_worker_core_for_dram_view(ch, /*noc*/1);
    dram_bank_to_noc_xy[0][ch] = (dc0.y << NOC_NODE_ID_BITS) | dc0.x;
    dram_bank_to_noc_xy[1][ch] = (dc1.y << NOC_NODE_ID_BITS) | dc1.x;
    bank_to_dram_offset[ch]    = metal_soc.get_address_offset(ch);
}
```

| Table | Meaning |
|---|---|
| `dram_bank_to_noc_xy[2][N]` | per-NOC `(y<<bits)|x` of the worker core that fronts each DRAM bank |
| `bank_to_dram_offset[N]` | per-bank firmware base offset (per SoC descriptor) |
| `l1_bank_to_noc_xy[2][N]` / `bank_to_l1_offset[N]` | L1 interleaving siblings (offset is 0 in emule) |

`NUM_DRAM_BANKS` is **not a hardcoded constant** — it is `get_num_dram_views()`
for the active arch (e.g. **12 on Wormhole N150**, 8 on Blackhole). Because that
can be **non-power-of-two**, the runner emits one of
`LOG_BASE_2_OF_NUM_DRAM_BANKS` (pow2 → bit-shift addr-gen) or
`IS_NOT_POW2_NUM_DRAM_BANKS=1` (→ fast-divide addr-gen). Emitting neither
silently collapses every page to bank 0. `DRAM_ALIGNMENT` (32) is injected from
`hal::get_dram_alignment()`.

---

## 3. Interleaved address generation

`include/jit_hw/internal/dataflow/dataflow_api_addrgen.h` carries the
header-only `InterleavedAddrGen<bool DRAM>`:

```cpp
uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0, uint8_t noc = 0) const {
    uint32_t bank_offset_index = id / NUM_DRAM_BANKS;          // stripe number
    uint32_t bank_index        = id - bank_offset_index * NUM_DRAM_BANKS;
    uint32_t aligned           = align_power_of_2(page_size, DRAM_ALIGNMENT);
    uint32_t addr = bank_offset_index * aligned + bank_base_address + offset
                  + bank_to_dram_offset[bank_index];
    uint32_t noc_xy = dram_bank_to_noc_xy[noc][bank_index];
    return (uint64_t(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;   // shift = 36
}
```

Page `id` → `(bank_index, stripe)`; the bank-local byte address packs with the
bank's NOC coords into a 64-bit NOC address (lower `NOC_ADDR_LOCAL_BITS = 36`
bits = offset, upper bits = `(y,x)` at 6 bits each). The `TensorAccessor` path
reaches the same result via `noc_traits_t<TensorAccessor>`
(`include/jit_hw/api/tensor/noc_traits.h`), which extracts the firmware DRAM
offset and routes through `__emule_resolve_noc_addr`.

---

## 4. Resolution to host memory

The NOC address produced above is decoded by `__emule_resolve_noc_addr`
(`emulated_program_runner.cpp`): it splits the upper coord bits and the lower
36-bit offset, looks the `(x,y)` up in the program runner's core map to find the
owning DRAM `Core`, and returns `core->l1_ptr(offset)` (DRAM cores use the full
36-bit offset — no `0x1FFFFF` slot mask). The legacy single-bank
`__emule_dram_ptr(offset)` fast path still exists but the bank-aware resolver is
the live path. See [noc-emulation.md](noc-emulation.md) §2 for the shared
encoding/decoding.

---

## 5. What's intentionally simplified

- No DRAM bandwidth / channel-contention / latency model — every access is a
  sync `memcpy`.
- DRAM backing is plain host memory; no refresh, no ECC, no row-buffer effects.
- Multicast does not apply to DRAM; DRAM is unicast page traffic only.

---

## 6. Where to read silicon canonical

| Topic | Canonical |
|---|---|
| Interleaved addr-gen | `tt-metal/tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h` |
| TensorAccessor | `tt-metal/tt_metal/hw/inc/api/tensor/tensor_accessor.h` |
| Bank topology | `metal_SocDescriptor` (`get_num_dram_views`, `get_preferred_worker_core_for_dram_view`, `get_address_offset`) |
| NOC address encoding | `tt-metal/tt_metal/hw/inc/.../noc_parameters.h`; [noc-emulation.md](noc-emulation.md) |
