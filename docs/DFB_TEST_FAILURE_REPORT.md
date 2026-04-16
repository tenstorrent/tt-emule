# DFB Test Failure Report

**Date:** 2026-04-16  
**Scope:** `test_dataflow_buffer_configs` (6 failing), `test_dataflow_buffer` BLOCKED multi-producer (16 failing)  
**Verdict summary:** All 6 config test failures are wrong test expectations. The 16 BLOCKED runtime failures are a code bug in `emulated_program_runner.cpp`.

---

## Part 1 — Config Test Wrong Expectations

### Background: Two BLOCKED modes, one test template

`finalize_single_dfb_config` (`dataflow_buffer.cpp:723`) distinguishes two BLOCKED sub-modes:

```cpp
// dataflow_buffer.cpp:748-756
bool dm_dm_blocked = (config.cap == dfb::AccessPattern::BLOCKED)
                     && !producer_is_tensix_only && !consumer_is_tensix_only;
bool use_remapper  = core_has_remapper
                     && (config.cap == dfb::AccessPattern::BLOCKED)
                     && !dm_dm_blocked;
```

| Mode | `dm_dm_blocked` | `use_remapper` | Producer TCs | Consumer TCs |
|------|----------------|----------------|--------------|--------------|
| DM-DM BLOCKED | `true` | `false` | `num_consumers` (broadcast) | `num_producers` |
| Tensix-involved BLOCKED | `false` | `true` | `1` (remapper) | `num_producers` |

`calculate_num_tile_counters` (`dataflow_buffer.cpp:226-268`) codifies this:
```cpp
if (config.cap == BLOCKED) {
    bool dm_dm_blocked = !producer_is_tensix_only && !consumer_is_tensix_only;
    if (is_producer) {
        if (dm_dm_blocked) return config.num_consumers;  // broadcast path
        return 1;                                        // remapper path
    }
    return config.num_producers;  // consumer always gets P TCs
}
```

All 6 failing config tests use **DM-DM BLOCKED** (both masks in bits 0–7), but their expectations were written for the **Tensix remapper path** (`expected_producer_tc_count = 1`).

---

### Bug 1 — `DMTensixTest1xDFB1Sx1SConfig` (wrong consumer risc_id)

**File:** `test_dataflow_buffer_configs.cpp:222-249`  
**Mode:** STRIDED (not BLOCKED), `producer_risc_mask=0x1`, `consumer_risc_mask=0x10`

`consumer_risc_mask = 0x10 = 0b00010000` → bit 4 set → risc_id = **4**.

Extraction loop at `dataflow_buffer.cpp:763-770`:
```cpp
for (uint8_t risc_id = 0; risc_id < 16; risc_id++) {
    if (config.consumer_risc_mask & (1 << risc_id))
        consumer_risc_ids.push_back(risc_id);
}
// → consumer_risc_ids = {4}
```

The test looks up `consumer_configs.find(consumer_risc_id)` at line 153. Test expectation:
```cpp
.producer_to_consumer_pairings = {
    {0, {{0, 0, 0}}},  // BUG: consumer_risc_id = 0, but mask gives 4
```

`consumer_configs.find(0)` returns `end()` → `ASSERT_NE` fails immediately.

**Fix:** Change `{0, {{0, 0, 0}}}` → `{0, {{4, 0, 0}}}`.

---

### Bug 2 — `DMTest1xDFB1Sx4BConfig` (wrong producer TC count for DM-DM BLOCKED 1P-4C)

**File:** `test_dataflow_buffer_configs.cpp:480-512`  
`producer_risc_mask=0x1` (DM risc 0), `consumer_risc_mask=0x1E` (DM riscs 1,2,3,4), `num_producers=1`, `num_consumers=4`, `cap=BLOCKED`.

Both masks are DM (bits 0–7) → `dm_dm_blocked = true`.

Calling `calculate_num_tile_counters(config, is_producer=true)`:
```cpp
// dataflow_buffer.cpp:233-236
if (dm_dm_blocked) {
    return config.num_consumers;  // → 4
}
```

Actual: `num_producer_tcs = 4`. Test expects:
```cpp
.expected_producer_tc_count = 1,  // BUG: comment says "BLOCKED: each producer has 1 TC"
```

That comment only holds for **Tensix-involved** BLOCKED. For DM-DM BLOCKED the producer broadcasts to one TC per consumer.

TC allocation (`dataflow_buffer.cpp:879-892`): producer 0 gets 4 TCs (tc_slot 0..3), one shared with each consumer. Pairings should be:
- TC[0] shared with consumer risc 1
- TC[1] shared with consumer risc 2
- TC[2] shared with consumer risc 3
- TC[3] shared with consumer risc 4

But the test lists all 4 consumers against `producer_tc_slot=0` (a single TC), which matches the Tensix remapper model where a single producer TC fans out via hardware. For DM-DM BLOCKED each consumer gets a distinct TC slot.

**Fix:** Set `expected_producer_tc_count = 4` and update pairings to use slots 0,1,2,3.

---

### Bug 3 — `DMTest1xDFB4Sx1BConfig` (remapper uniqueness check invalid for DM-DM BLOCKED)

**File:** `test_dataflow_buffer_configs.cpp:514-544`  
`producer_risc_mask=0xF` (DM riscs 0–3), `consumer_risc_mask=0x10` (DM risc 4), `num_producers=4`, `num_consumers=1`, `cap=BLOCKED`.

Both masks are DM → `dm_dm_blocked = true` → `use_remapper = false`.

`remapper_pair_index` is only populated at `dataflow_buffer.cpp:947`:
```cpp
if (use_remapper) {
    risc_config.config.remapper_pair_index = remapper_index_allocator_.allocate(core);
    ...
}
```

With `use_remapper = false`, all 4 producers keep the default-initialized value: `remapper_pair_index = 0`.

The test validation block (`test_dataflow_buffer_configs.cpp:122-140`) runs for ALL BLOCKED modes:
```cpp
if (config.cap == dfb::AccessPattern::BLOCKED) {
    std::set<uint8_t> seen_remapper_indices;
    for (const auto& [risc_id, rc] : producer_configs) {
        uint8_t remapper_idx = rc->config.remapper_pair_index;
        EXPECT_EQ(seen_remapper_indices.count(remapper_idx), 0)
            << "BLOCKED: Producer RISC " << risc_id << " has duplicate remapper_pair_index";
        seen_remapper_indices.insert(remapper_idx);
    }
}
```

For producers 0,1,2,3 all having `remapper_pair_index = 0`:
- Producer 0: inserts 0 → OK
- Producer 1: `seen.count(0) = 1` → **FAIL** ("duplicate remapper_pair_index 0")
- Producers 2,3: same failure

**Note:** The TC count expectation for this test (producer=1, consumer=4) is actually **correct** for 4P-1C DM-DM BLOCKED (`num_producer_tcs = num_consumers = 1`). Only the remapper check is wrong.

**Fix:** Guard the remapper uniqueness check with `use_remapper`-equivalent logic. Since tests don't have direct access to `use_remapper`, guard on whether the producer risc_mask contains only DM riscs (bits 0–7) and the consumer risc_mask also contains only DM riscs:
```cpp
bool any_tensix = (config.producer_risc_mask & 0xFF00) || (config.consumer_risc_mask & 0xFF00);
if (config.cap == dfb::AccessPattern::BLOCKED && any_tensix) {
    // remapper uniqueness check only applies when Tensix is involved
    ...
}
```

---

### Bugs 4–6 — `DMTest1xDFB4Sx4BConfig`, `DMTest1xDFB4Sx2BConfig`, `DMTest1xDFB2Sx4BConfig`

All three have the same root cause as Bug 2: `expected_producer_tc_count = 1` was written for the Tensix remapper path, but all three configs use DM-only masks.

| Test | P | C | `dm_dm_blocked` | Actual producer TCs (`num_consumers`) | Test expects |
|------|---|---|-----------------|---------------------------------------|--------------|
| `DMTest1xDFB4Sx4BConfig` (line 546) | 4 | 4 | true | **4** | 1 |
| `DMTest1xDFB4Sx2BConfig` (line 602) | 4 | 2 | true | **2** | 1 |
| `DMTest1xDFB2Sx4BConfig` (line 653) | 2 | 4 | true | **4** | 1 |

All three also hit the remapper uniqueness check failure (same as Bug 3) because all have multiple producers and `use_remapper = false`.

**Code trace for `DMTest1xDFB4Sx4BConfig`:**  
`producer_risc_mask=0xF`, `consumer_risc_mask=0xF0` — both DM → `dm_dm_blocked=true`.  
`calculate_num_tile_counters(is_producer=true)` → `return config.num_consumers = 4`.  
Test line 566: `expected_producer_tc_count = 1` → FAIL at line 82:
```cpp
EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_producer_tc_count)
// actual=4, expected=1 → FAIL
```

Then remapper check: 4 producers all with `remapper_pair_index=0` → FAIL at line 133.

---

## Part 2 — BLOCKED Runtime Test Failures (`test_dataflow_buffer`)

### Root cause: `emulated_program_runner.cpp` missing `drain_per_tc`

The standalone path (`kernel_runner.cpp`) correctly configures BLOCKED consumers with two key settings:

```cpp
// kernel_runner.cpp:142-158
if (is_blocked) {
    iface.num_tcs_to_rr = static_cast<uint8_t>(cfg.num_producers);
    iface.stride_size = cfg.entry_size;
    iface.drain_per_tc = true;                                    // ← CRITICAL
    uint32_t capacity_per_p = cfg.num_entries / cfg.num_producers;
    for (uint32_t pi = 0; pi < cfg.num_producers; ++pi) {
        auto& slot = iface.tc_slots[pi];
        slot.counter_id  = counter_base + pi * cfg.num_consumers + c;
        slot.base_addr   = base_addr + pi * capacity_per_p * cfg.entry_size;  // per-block
        slot.limit       = slot.base_addr + capacity_per_p * cfg.entry_size;  // per-block
        slot.rd_ptr      = slot.base_addr;
    }
}
```

The JIT path (`emulated_program_runner.cpp:1341-1355`) does NOT set `drain_per_tc` and uses the full-DFB base/limit for all slots:

```cpp
// emulated_program_runner.cpp:1341-1355
if (is_blocked) {
    iface.num_tcs_to_rr = static_cast<uint8_t>(alloc.num_producers);
    iface.stride_size = alloc.entry_size;
    // drain_per_tc NOT SET → stays false                        ← BUG #1
    uint32_t capacity_per_p = alloc.num_entries / alloc.num_producers;
    for (uint32_t p = 0; p < alloc.num_producers; ++p) {
        auto& slot = iface.tc_slots[p];
        slot.counter_id = counter_base + p * alloc.num_consumers + c;
        slot.base_addr  = alloc.base_addr;                       // ← BUG #2: whole-DFB base
        slot.limit      = alloc.base_addr + alloc.total;         // ← BUG #2: whole-DFB limit
        slot.rd_ptr     = alloc.base_addr + p * capacity_per_p * alloc.entry_size;
    }
}
```

### Consequence traced for 4P-1C BLOCKED, 16 entries

Setup: `num_producers=4, num_consumers=1, num_entries=16, entry_size=E`.  
`capacity_per_p = 4`. Consumer has 4 TC slots (one per producer block).

**With `drain_per_tc = false` (current buggy behavior):**

`dfb_pop_front` at `dfb_api.h:117-139`:
```cpp
auto& slot = iface.tc_slots[iface.tc_idx];
__emule_tc_array->inc_acked(...);
slot.rd_ptr += iface.stride_size;      // advances by E per pop
if (slot.rd_ptr >= slot.limit) ...wrap
iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;  // always advances
```

Consumer reads 16 entries, one per `pop_front(1)`, with `is_blocked=1` → `page_id = tile_id`:
| pop | tc_idx | rd_ptr source | entry read | written to output[tile_id] |
|-----|--------|---------------|------------|---------------------------|
| 0   | 0 | block 0 entry 0 | input[0]  | output[0] |
| 1   | 1 | block 1 entry 0 | input[4]  | output[1] ✗ (expected input[1]) |
| 2   | 2 | block 2 entry 0 | input[8]  | output[2] ✗ |
| 3   | 3 | block 3 entry 0 | input[12] | output[3] ✗ |
| 4   | 0 | block 0 entry 1 | input[1]  | output[4] ✗ |
...

Result: `output = [input[0], input[4], input[8], input[12], input[1], input[5], ...]`  
Expected: `output = [input[0], input[1], input[2], ..., input[15]]`  
`EXPECT_EQ(input, output)` at line 82 → FAIL.

**With `drain_per_tc = true` and per-block base/limit (correct):**

Drain detection at `dfb_api.h:132-136`:
```cpp
if (iface.drain_per_tc) {
    if (slot.rd_ptr == slot.base_addr)  // wraps back after capacity_per_p entries
        iface.tc_idx = (iface.tc_idx + 1) % iface.num_tcs_to_rr;
}
```

With per-block base/limit: after 4 pops on TC[0]:
- `rd_ptr = base_addr_of_block0 + 4*E = slot.limit` → wrap → `rd_ptr = slot.base_addr`
- `slot.rd_ptr == slot.base_addr` → advance to TC[1] ✓

Consumer reads: [input[0], input[1], input[2], input[3], input[4], ..., input[15]] ✓

### Why single-producer BLOCKED tests pass

For 1P-1C BLOCKED: `num_tcs_to_rr = 1`. `tc_idx` only takes value 0 regardless of round-robin or drain. There is only one TC to advance to, so the bug is invisible. Tests `DMTest1xDFB1Sx1B*` pass.

### Why the docs claim 30/30 BLOCKED tests pass

`DFB_EMULATION.md:400-407` states "All 72 DFB tests pass". This was written when the standalone path and JIT path were in sync. The JIT path in `emulated_program_runner.cpp` was subsequently written without copying the `drain_per_tc` and per-block base/limit settings from `kernel_runner.cpp`, introducing the regression for multi-producer BLOCKED tests.

### Fix required in `emulated_program_runner.cpp`

At the BLOCKED consumer setup block (lines 1341–1354), add:
```cpp
iface.drain_per_tc = true;
```
And change each slot's base/limit to be per-block:
```cpp
slot.base_addr = alloc.base_addr + p * capacity_per_p * alloc.entry_size;
slot.limit     = slot.base_addr + capacity_per_p * alloc.entry_size;
slot.rd_ptr    = slot.base_addr;
slot.wr_ptr    = slot.base_addr;
```

---

## Summary Table

| Test | Failure category | Root cause | Fix |
|------|-----------------|------------|-----|
| `DMTensixTest1xDFB1Sx1SConfig` | Wrong expectation | consumer risc_id=0 vs mask bit 4 → actual risc_id=4 | Change pairing `{0,{{0,0,0}}}` → `{0,{{4,0,0}}}` |
| `DMTest1xDFB1Sx4BConfig` | Wrong expectation | DM-DM BLOCKED 1P-4C: producer gets 4 TCs (one/consumer), test expects 1 | `expected_producer_tc_count = 4`; fix pairings to use slots 0–3 |
| `DMTest1xDFB4Sx1BConfig` | Wrong expectation | Remapper uniqueness check invalid for DM-DM BLOCKED (no remapper used) | Guard check with `any_tensix` condition |
| `DMTest1xDFB4Sx4BConfig` | Wrong expectation | DM-DM BLOCKED 4P-4C: producer gets 4 TCs, test expects 1; + remapper check | `expected_producer_tc_count = 4`; guard remapper check |
| `DMTest1xDFB4Sx2BConfig` | Wrong expectation | DM-DM BLOCKED 4P-2C: producer gets 2 TCs, test expects 1; + remapper check | `expected_producer_tc_count = 2`; guard remapper check |
| `DMTest1xDFB2Sx4BConfig` | Wrong expectation | DM-DM BLOCKED 2P-4C: producer gets 4 TCs, test expects 1; + remapper check | `expected_producer_tc_count = 4`; guard remapper check |
| 16 BLOCKED multi-producer runtime tests | **Code bug** | `emulated_program_runner.cpp` never sets `drain_per_tc=true`; uses whole-DFB base/limit for BLOCKED consumer slots | Set `drain_per_tc=true`, use per-block base_addr/limit in runner |
