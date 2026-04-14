# DFB Config Test Failure Analysis

Six tests in `test_dataflow_buffer_configs.cpp` fail when run against commit `3fa4d75355` of `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp`. This document traces each failure through the finalization code with specific file paths and line numbers.

All tests gate on `ARCH::QUASAR` and may pass on Quasar FPGA emulation or other non-standard environments that run a different version of the finalization code. The analysis below applies specifically to commit `3fa4d75355`.

---

## Background: How finalization assigns tile counters

Two functions in `dataflow_buffer.cpp` control the values these tests check:

### `calculate_num_tile_counters()` (line 226)

```cpp
// dataflow_buffer.cpp:226-239
uint8_t calculate_num_tile_counters(const DataflowBufferConfig& config, bool is_producer) {
    if (config.cap == ::dfb::AccessPattern::BLOCKED) {
        bool producer_has_dm = has_dm_risc(config.producer_risc_mask);     // line 228
        bool consumer_has_dm = has_dm_risc(config.consumer_risc_mask);     // line 229
        bool producer_is_tensix_only = !producer_has_dm && has_tensix_risc(config.producer_risc_mask);  // line 230
        bool consumer_is_tensix_only = !consumer_has_dm && has_tensix_risc(config.consumer_risc_mask);  // line 231
        bool dm_dm_blocked = !producer_is_tensix_only && !consumer_is_tensix_only;  // line 232
        if (is_producer) {
            if (dm_dm_blocked) {
                return config.num_consumers;  // line 235 — DM-DM: one TC per consumer
            }
            return 1;                         // line 237 — Tensix-involved: 1 TC (remapper fans out)
        }
        return config.num_producers;          // line 239 — consumer always gets one TC per producer
    }
    // ...
}
```

`has_dm_risc()` (line 222): `(risc_mask & 0xFF) != 0`
`has_tensix_risc()` (line 224): `(risc_mask & 0x0F00) != 0`

### `finalize_single_dfb_config()` (lines 720+)

Key assignments:
- **risc_id loop** (lines 763-769): Iterates bits 0-15 of each mask; bit position = risc_id
- **`use_remapper`** (lines 754-756): `core_has_remapper && BLOCKED && !dm_dm_blocked`
- **Producer `num_tcs_to_rr`** (line 943): Set to `num_producer_tcs` (from `calculate_num_tile_counters`)
- **Consumer `num_tcs_to_rr`** (line 1048): Set to `num_consumer_tcs` (from `calculate_num_tile_counters`)
- **`remapper_pair_index`** (line 947): Set **only inside** `if (use_remapper)` block (lines 946-972)
- **`consumer_tcs`** (line 971): Set **only inside** `if (use_remapper)` block (lines 946-972)

When `use_remapper` is false, `remapper_pair_index` and `consumer_tcs` remain at their default value of 0.

---

## Failure 1: `DMTensixTest1xDFB1Sx1SConfig` (line 221)

**Config** (lines 225-234):
```
producer_risc_mask = 0x1   (bit 0 → DM risc 0)
consumer_risc_mask = 0x10  (bit 4 → DM risc 4)
cap = STRIDED, num_producers = 1, num_consumers = 1
```

### Expected (lines 240-245)

```cpp
.expected_producer_tc_count = 1,
.expected_consumer_tc_count = 1,
.producer_to_consumer_pairings = {
    {0, {{0, 0, 0}}},  // Consumer risc_id = 0
}
```

### Actual (from finalization)

The risc_id loop at `dataflow_buffer.cpp:763-769` iterates bit positions:
- `consumer_risc_mask = 0x10` → bit 4 is set → `consumer_risc_ids = {4}`

TC counts are correct (1/1). The pairing passes consumer risc_id = **0**, but the actual consumer risc_id is **4**.

### Assertion failure

`validate_dfb_tile_counters` line 152: `consumer_configs.find(0)` returns `end()`. No consumer exists with risc_id 0. The consumer has risc_id 4.

### What the test should say

```diff
-    {0, {{0, 0, 0}}},  // Consumer risc_id = 0
+    {0, {{4, 0, 0}}},  // Consumer risc_id = 4 (bit 4 of mask 0x10)
```

---

## Failures 2-6: DM-DM BLOCKED tests

All five share the same root cause: the expectations assume every producer has 1 TC in BLOCKED mode, but DM-DM BLOCKED is a special case where the producer gets `num_consumers` TCs (no remapper to fan out). All five also validate `remapper_pair_index` uniqueness and `consumer_tcs`, which are never set because `use_remapper = false` for DM-DM BLOCKED.

### Why DM-DM BLOCKED differs

For all five tests, both producer and consumer masks have only low bits set (bits 0-7), so:

```
has_dm_risc(producer_mask) = true    (bits 0-7 check, line 222)
has_dm_risc(consumer_mask) = true    (bits 0-7 check, line 222)
producer_is_tensix_only = false      (line 230: !true && ...)
consumer_is_tensix_only = false      (line 231: !true && ...)
dm_dm_blocked = true                 (line 232: !false && !false)
```

This means:
1. **Producer TC count** = `num_consumers` (line 235), not 1
2. **`use_remapper`** = false (line 754-756: requires `!dm_dm_blocked`)
3. **`remapper_pair_index`** stays at 0 for all producers (line 947 never reached)
4. **`consumer_tcs`** stays at 0 for all producers (line 971 never reached)

The test expectations assume the Tensix-involved BLOCKED path (line 237: producer gets 1 TC, remapper fans out), which only applies when one side has Tensix-only riscs (bits 8+).

---

### Failure 2: `DMTest1xDFB1Sx4BConfig` (line 479)

**Config** (lines 483-492):
```
producer_risc_mask = 0x1   (1 DM producer, risc 0)
consumer_risc_mask = 0x1E  (4 DM consumers, riscs 1-4)
cap = BLOCKED, num_producers = 1, num_consumers = 4
```

#### Expected (lines 498-508)

```cpp
.expected_producer_tc_count = 1,  // "BLOCKED: each producer has 1 TC"
.expected_consumer_tc_count = 1,
```

#### Actual

- `dm_dm_blocked = true` (both sides DM)
- Producer TC count: `calculate_num_tile_counters(config, true)` → line 235 → `num_consumers` = **4**
- Consumer TC count: line 239 → `num_producers` = **1** (correct)

#### Assertion failure

Line 81: `producer_configs[0].num_tcs_to_rr` is 4, expected 1.

---

### Failure 3: `DMTest1xDFB4Sx1BConfig` (line 513)

**Config** (lines 517-526):
```
producer_risc_mask = 0xF   (4 DM producers, riscs 0-3)
consumer_risc_mask = 0x10  (1 DM consumer, risc 4)
cap = BLOCKED, num_producers = 4, num_consumers = 1
```

#### Expected (lines 532-540)

```cpp
.expected_producer_tc_count = 1,  // "BLOCKED: each producer has 1 TC"
.expected_consumer_tc_count = 4,
```

#### Actual

- `dm_dm_blocked = true`
- Producer TC count: line 235 → `num_consumers` = **1** (happens to match expectation)
- Consumer TC count: line 239 → `num_producers` = **4** (matches expectation)

TC counts are actually correct here. The failure is different:

#### Assertion failure

Line 132: `remapper_pair_index` uniqueness check. All 4 producers have `remapper_pair_index = 0` (default, because `use_remapper = false` at line 754-756 — the assignment at line 947 is inside `if (use_remapper)` and never executes). The test expects unique remapper indices across producers.

---

### Failure 4: `DMTest1xDFB4Sx4BConfig` (line 545)

**Config** (lines 549-558):
```
producer_risc_mask = 0xF   (4 DM producers, riscs 0-3)
consumer_risc_mask = 0xF0  (4 DM consumers, riscs 4-7)
cap = BLOCKED, num_producers = 4, num_consumers = 4
```

#### Expected (lines 564-596)

```cpp
.expected_producer_tc_count = 1,
.expected_consumer_tc_count = 4,
```

#### Actual

- Producer TC count: line 235 → `num_consumers` = **4** (expected 1)
- Consumer TC count: line 239 → `num_producers` = **4** (matches)

#### Assertion failures

1. Line 81: producer `num_tcs_to_rr` = 4, expected 1
2. Line 132: `remapper_pair_index` all 0 (same cause as Failure 3)
3. Line 207: `consumer_tcs` = 0x0 for all producers (never set; test computes non-zero expected value from pairings)

---

### Failure 5: `DMTest1xDFB4Sx2BConfig` (line 601)

**Config** (lines 605-614):
```
producer_risc_mask = 0xF   (4 DM producers, riscs 0-3)
consumer_risc_mask = 0x30  (2 DM consumers, riscs 4-5)
cap = BLOCKED, num_producers = 4, num_consumers = 2
```

#### Expected (lines 620-644)

```cpp
.expected_producer_tc_count = 1,
.expected_consumer_tc_count = 4,
```

#### Actual

- Producer TC count: line 235 → `num_consumers` = **2** (expected 1)
- Consumer TC count: line 239 → `num_producers` = **4** (matches)

#### Assertion failures

1. Line 81: producer `num_tcs_to_rr` = 2, expected 1
2. Line 132: `remapper_pair_index` all 0
3. Line 207: `consumer_tcs` = 0x0

---

### Failure 6: `DMTest1xDFB2Sx4BConfig` (line 652)

**Config** (lines 656-665):
```
producer_risc_mask = 0x3   (2 DM producers, riscs 0-1)
consumer_risc_mask = 0x3C  (4 DM consumers, riscs 2-5)
cap = BLOCKED, num_producers = 2, num_consumers = 4
```

#### Expected (lines 672-690)

```cpp
.expected_producer_tc_count = 1,
.expected_consumer_tc_count = 2,
```

#### Actual

- Producer TC count: line 235 → `num_consumers` = **4** (expected 1)
- Consumer TC count: line 239 → `num_producers` = **2** (matches)

#### Assertion failures

1. Line 81: producer `num_tcs_to_rr` = 4, expected 1
2. Line 132: `remapper_pair_index` all 0
3. Line 207: `consumer_tcs` = 0x0

---

## Summary table

| Test | Failing checks | Expected | Actual (at `3fa4d75355`) | Code path |
|------|---------------|----------|--------------------------|-----------|
| `DMTensixTest1xDFB1Sx1SConfig` | consumer risc_id in pairing | 0 | 4 | risc_id loop, line 763-769 |
| `DMTest1xDFB1Sx4BConfig` | `expected_producer_tc_count` | 1 | 4 | `calculate_num_tile_counters` line 235 |
| `DMTest1xDFB4Sx1BConfig` | `remapper_pair_index` uniqueness | unique | all 0 | line 947 inside `if(use_remapper)`, never reached |
| `DMTest1xDFB4Sx4BConfig` | producer TC + remapper + `consumer_tcs` | 1 / unique / non-zero | 4 / all 0 / 0x0 | lines 235, 947, 971 |
| `DMTest1xDFB4Sx2BConfig` | producer TC + remapper + `consumer_tcs` | 1 / unique / non-zero | 2 / all 0 / 0x0 | lines 235, 947, 971 |
| `DMTest1xDFB2Sx4BConfig` | producer TC + remapper + `consumer_tcs` | 1 / unique / non-zero | 4 / all 0 / 0x0 | lines 235, 947, 971 |

## Possible explanations

1. **The test expectations match a different version of `finalize_single_dfb_config()`.** If `calculate_num_tile_counters()` or the `use_remapper` logic differs on the branch where these tests were authored (e.g. a Quasar FPGA branch), the expectations could be correct for that branch.

2. **The test expectations assume the Tensix-involved BLOCKED path.** The comments ("BLOCKED: each producer has 1 TC") describe the path at line 237, which requires `dm_dm_blocked = false`. But all five DM-DM BLOCKED tests use DM-only masks (bits 0-7), making `dm_dm_blocked = true` at line 232.

3. **The remapper uniqueness and `consumer_tcs` checks may not be applicable to DM-DM BLOCKED.** When `use_remapper = false`, these fields are never assigned. The test validates them unconditionally.

## How to reproduce

```bash
cd tt-metal
export TT_METAL_EMULATED_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$PWD"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/quasar_1chip.yaml"

SIM_DIR="$(mktemp -d /tmp/tt_emule_sim.XXXXXX)"
ln -sf "$PWD/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$SIM_DIR/soc_descriptor.yaml"
export TT_METAL_SIMULATOR="$SIM_DIR"

# Run all 16 config tests (10 pass, 6 fail):
./build_emule_clang/test/tt_emule/test_dataflow_buffer_configs

# Run only the 6 failing:
./build_emule_clang/test/tt_emule/test_dataflow_buffer_configs \
    --gtest_filter="*DMTensixTest1xDFB1Sx1SConfig*:*1Sx4BConfig*:*4Sx1BConfig*:*4Sx4BConfig*:*4Sx2BConfig*:*2Sx4BConfig*"
```
