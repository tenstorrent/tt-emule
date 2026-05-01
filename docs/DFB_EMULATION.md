# Dataflow Buffer Emulation in tt-emule

This document describes how tt-emule models Quasar Dataflow Buffers (DFBs) — the Quasar replacement for Wormhole Circular Buffers. It is intended as a standalone reference for anyone reading or extending the DFB emulation code.

---

## 1. Background: Why DFBs Differ from CBs

On Wormhole, a Circular Buffer (CB) is a simple SPSC (single-producer, single-consumer) FIFO. One DM kernel writes tiles in, one compute kernel reads them out. The synchronization state is a single atomic `occupied` counter plus a mutex and two condition variables.

On Quasar, a single Tensix has **24 logical processors**: 8 Data Movement (DM0–DM7) plus 4 Neo compute clusters × 4 RISC slots (E0_MATH0 through E3_MATH3). All 24 processors share one 4 MB L1. A DFB must therefore support **multiple producers and multiple consumers** (MPMC) on the same buffer. A single atomic `occupied` counter is insufficient for this.

The hardware solution is a **tile counter (TC)** per consumer: each consumer gets its own `posted`/`acked` counter pair, so it tracks its own progress independently of every other consumer. A producer posts to one or more of these counters depending on the access pattern. This is the central design that the emulation replicates.

---

## 2. Data Structures

### 2.1 `TileCounter` (`include/tt_emule/tile_counter.hpp`)

The atomic synchronization primitive for one hardware tile counter.

```cpp
struct TileCounter {
    std::atomic<uint32_t> posted{0};       // incremented by producer after writing
    std::atomic<uint32_t> acked{0};        // incremented by consumer after reading
    uint32_t capacity{0};                  // set at init; constant during a program
    std::mutex mu;
    std::condition_variable space_cv;      // producer blocks here when full
    std::condition_variable data_cv;       // consumer blocks here when empty

    uint32_t occupancy() const;            // posted - acked
    uint32_t free_space() const;           // capacity - occupancy
    void reset();
};
```

Key points:
- `occupancy` and `free_space` are derived from two independent monotonically-increasing counters. They never go negative because the emulation enforces that `acked <= posted` and `occupancy <= capacity` via the blocking waits.
- The mutex guards both atomics for condition variable signaling only. The atomics use `memory_order_acquire/release` so they can be read locklessly for the fast path before taking the mutex.
- `capacity` is the number of entries this TC can hold at once. `capacity = num_entries / M` where `M = max(num_producers, num_consumers)`. For 1P-1C, M=1 so capacity equals `num_entries`.

### 2.2 `TileCounterArray` (`include/tt_emule/tile_counter.hpp`)

Owns and manages all tile counters for one Tensix.

```cpp
class TileCounterArray {
    TileCounterArray(uint32_t num_neos);
    TileCounter& get(uint8_t neo_id, uint8_t counter_id);
    void inc_posted(uint8_t neo_id, uint8_t counter_id, uint32_t n);
    void inc_acked(uint8_t neo_id, uint8_t counter_id, uint32_t n);
    void wait_free_space(uint8_t neo_id, uint8_t counter_id, uint32_t n);
    void wait_occupancy(uint8_t neo_id, uint8_t counter_id, uint32_t n);
    void reset_all();
};
```

The array holds `num_neos * 32` counters in a flat `unique_ptr<TileCounter[]>`. `num_neos` is passed at construction time and defaults to 4 for Quasar. The `get(neo_id, counter_id)` indexing is `neo_id * 32 + counter_id`.

Counter IDs are assigned per-DFB: DFB `d` uses counters at base `counter_id = d`, with one TC slot per consumer. For example, DFB 0 with 2 consumers uses counter_ids 0 and 1; DFB 1 uses counter_ids 1 and 2 (these can overlap if `num_consumers == 1` per DFB).

One `TileCounterArray` is owned by `Core` and shared across all threads. This is the hardware-accurate model: in the emulation, all threads share the same address space, so the TC array is just a shared data structure — no hardware remapper logic is needed.

### 2.3 `DFBTCSlot` (`include/tt_emule/dfb_sync_state.hpp`)

One slot in a thread's per-DFB round-robin list. Tracks the L1 memory region and the read/write cursor for one producer-consumer pairing.

```cpp
struct DFBTCSlot {
    uint32_t rd_ptr    = 0;    // current read position (uint32_t L1 address)
    uint32_t wr_ptr    = 0;    // current write position
    uint32_t base_addr = 0;    // start of the buffer region in L1
    uint32_t limit     = 0;    // base_addr + num_entries * entry_size
    uint8_t  neo_id    = 0;    // which Neo domain owns this TC
    uint8_t  counter_id = 0;   // which TC within that Neo
};
```

Pointers are `uint32_t` rather than `uint8_t*` because kernel code addresses L1 as 32-bit values. The `advance_ptr` helper in `DataflowBuffer` wraps them modulo `[base_addr, limit)`.

### 2.4 `EmuleDFBInterface` (`include/tt_emule/dfb_sync_state.hpp`)

The per-thread, per-DFB view. Each kernel thread has an array of `MAX_DFBS` of these, one for each DFB the program declares.

```cpp
struct EmuleDFBInterface {
    DFBTCSlot tc_slots[4] = {};     // up to 4 TC slots to round-robin through
    uint8_t  num_tcs_to_rr = 0;    // how many slots are active
    uint8_t  tc_idx        = 0;    // current round-robin position
    uint32_t entry_size    = 0;    // bytes per DFB entry
    uint32_t stride_size   = 0;    // entry_size * num_producers (consumer view) or
                                   // entry_size * num_consumers (producer view)
    uint32_t rd_entry_idx  = 0;    // logical count of entries read (informational)
    uint32_t wr_entry_idx  = 0;    // logical count of entries written (informational)
    uint32_t num_entries   = 0;    // total entries in the DFB
    bool     broadcast_tc  = false; // BLOCKED mode: push to all TC slots
    bool     active        = false; // false = this thread does not participate
};
```

For STRIDED mode with `M = max(P, C)`:
- Producer p: `num_tcs_to_rr = M / P` TC slots at indices `{p + k*P for k=0..M/P-1}`. Initial pointers offset by `tc_index * entry_size`. `stride_size = M * entry_size`.
- Consumer c: `num_tcs_to_rr = M / C` TC slots at indices `{c + k*C for k=0..M/C-1}`. Same pointer offset and stride.

For BLOCKED mode (`stride_in_entries = 1`): each producer owns a contiguous block of `capacity = num_entries / num_producers` entries.
- Producer p: `broadcast_tc = true`, `num_tcs_to_rr = num_consumers`. Posts to ALL consumer TCs on each `push_back`. `stride_size = entry_size`. Initial pointer offset = `p * capacity * entry_size`.
- Consumer: `num_tcs_to_rr = num_producers`, round-robins through producer blocks. `stride_size = entry_size`. Slot p starts at `base + p * capacity * entry_size`.

### 2.5 `DFBSyncState` (`include/tt_emule/dfb_sync_state.hpp`)

Per-DFB metadata stored on `Core`. Used during initialization to build `EmuleDFBInterface` objects for each thread.

```cpp
struct DFBSyncState {
    uint8_t* base         = nullptr;  // L1 backing memory (host pointer)
    uint32_t entry_size   = 0;
    uint32_t num_entries  = 0;
    uint32_t capacity     = 0;        // per-TC capacity (currently = num_entries)
    uint32_t stride_in_entries = 1;
};
```

### 2.6 `DataflowBuffer` (`include/tt_emule/dataflow_buffer.hpp`)

The kernel-facing object in the standalone path. Constructed ephemerally per API call from an `EmuleDFBInterface&` and the shared `TileCounterArray&`. The JIT path skips this class entirely and operates directly on the TLS pointers via inline free functions.

---

## 3. Tile Counter Synchronization Deep Dive

### 3.1 The Invariant

At any point in time, for a tile counter with counter_id C owned by neo_id N:

```
0 <= acked <= posted <= acked + capacity
```

- `posted - acked` = occupancy (entries written but not yet consumed)
- `capacity - (posted - acked)` = free_space (room for more entries)

A producer may only call `push_back` after `reserve_back` confirms `free_space >= n`. A consumer may only call `pop_front` after `wait_front` confirms `occupancy >= n`.

### 3.2 Four Operations on `TileCounterArray`

**`inc_posted(neo_id, counter_id, n)` — called by producer after writing**

```cpp
void TileCounterArray::inc_posted(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
    auto& tc = get(neo_id, counter_id);
    { std::lock_guard lk(tc.mu); tc.posted.fetch_add(n, memory_order_release); }
    tc.data_cv.notify_all();
}
```

The lock ensures the atomic increment and the condition variable notification are sequenced: the consumer's `data_cv.wait` predicate re-checks `occupancy()` under the lock, so it cannot miss the notification. The `notify_all` wakes every consumer waiting on this TC.

**`inc_acked(neo_id, counter_id, n)` — called by consumer after reading**

Mirror of `inc_posted`, but increments `acked` and notifies `space_cv`. This wakes producers waiting for free space.

**`wait_free_space(neo_id, counter_id, n)` — called by producer before writing**

```cpp
void TileCounterArray::wait_free_space(uint8_t neo_id, uint8_t counter_id, uint32_t n) {
    auto& tc = get(neo_id, counter_id);
    std::unique_lock lk(tc.mu);
    tc.space_cv.wait(lk, [&]{ return tc.free_space() >= n; });
}
```

Always takes the mutex. A lockless fast path was removed because `occupancy()` and `free_space()` each read two independent atomics (`posted`, `acked`) non-atomically — in MPMC scenarios, the unsigned subtraction could underflow, causing a spurious return.

**`wait_occupancy(neo_id, counter_id, n)` — called by consumer before reading**

Same structure as `wait_free_space` but blocks on `data_cv` and waits for `occupancy() >= n`.

### 3.3 The Drain Barrier: `finish()`

`finish()` is called by a producer when it has pushed all its entries and wants to wait until all consumers have acknowledged them. It iterates over all TC slots and waits for `posted == acked` on each:

```cpp
void DataflowBuffer::finish() {
    for (uint8_t i = 0; i < iface_.num_tcs_to_rr; ++i) {
        auto& slot = iface_.tc_slots[i];
        auto& tc = tc_array_.get(slot.neo_id, slot.counter_id);
        if (tc.posted == 0 && tc.acked == 0) continue;  // already drained
        std::unique_lock lk(tc.mu);
        tc.space_cv.wait(lk, [&]{ return tc.posted == tc.acked; });
    }
}
```

`space_cv` is reused here because the drain condition is exactly what `inc_acked` signals. A fully drained TC (`posted == acked`) means all written entries have been consumed.

The JIT path (`dfb_api.h:dfb_finish`) is identical in behavior but operates directly on `__emule_dfbs[dfb_id]` and `__emule_tc_array`.

### 3.4 Why Both Atomics and a Mutex?

The `posted` and `acked` counters are `std::atomic<uint32_t>` to allow lockless reads in the fast path. But `std::condition_variable` requires a `std::mutex`: the predicate must be checked under the lock to avoid the classic TOCTOU race where the notifying thread increments the counter and calls `notify` between the predicate check and the `wait` call in the blocking thread. By taking the lock before both the `fetch_add` and the `notify`, `inc_posted`/`inc_acked` ensure no notifications are lost.

---

## 4. Access Patterns

### 4.1 STRIDED

Entries are interleaved across consumers. With P producers and C consumers:
- Producer i writes entries `i, i+P, i+2P, ...` (its stride)
- Consumer j reads entries `j, j+C, j+2C, ...` (its stride)

In the emulation, the interleaving factor `M = max(P, C)` determines the TC assignment:
- Producer p round-robins over `M/P` TC slots at indices `{p + k*P for k=0..M/P-1}`. After each `push_back`, it advances `tc_idx = (tc_idx + 1) % (M/P)`.
- Consumer c round-robins over `M/C` TC slots at indices `{c + k*C for k=0..M/C-1}`. After each `pop_front`, it advances `tc_idx = (tc_idx + 1) % (M/C)`.
- For 1P-4C: M=4, producer has 4 TC slots (one per consumer), consumer has 1 TC slot.
- For 4P-1C: M=4, producer has 1 TC slot, consumer has 4 TC slots (one per producer).
- For 4P-4C: M=4, each producer has 1 TC slot, each consumer has 1 TC slot (independent pairings).

The `stride_size` field = `M * entry_size` for all participants. Initial pointer offsets: producer p slot k starts at `base + (p + k*P) * entry_size`; consumer c slot k starts at `base + (c + k*C) * entry_size`.

**Test evidence (STRIDED):** 42 tests in `test_dataflow_buffer.cpp` cover 1P-1C through 4P-4C STRIDED topologies across DM-DM, DM→Tensix, Tensix→DM, and multi-DFB pipeline topologies. Each runs with both `ImplicitSyncFalse` and `ImplicitSyncTrue`. Example filters: `DMTest1xDFB4Sx4S`, `DMTensixTest1xDFB1Sx4S`, `TensixDMTest1xDFB4Sx2S`.

**Constraint:** `num_entries_in_buffer` must be a multiple of `max(P, C)` for all entries to be processed. `capacity = floor(num_entries_in_buffer / M)`, so `capacity * M < num_entries_in_buffer` when `M` does not divide evenly — the trailing entries are never assigned to any TC slot and are never read, causing output buffer trailing zeros.

### 4.2 BLOCKED

All consumers see all data. Each producer owns a **contiguous block** of `capacity = num_entries / num_producers` entries starting at `base + p * capacity * entry_size`. The `broadcast_tc` flag on `EmuleDFBInterface` enables the producer broadcast path. `stride_size = entry_size` (`stride_in_entries = 1`).

In `reserve_back`: wait for `free_space >= n` on **all** TC slots (loop over `num_tcs_to_rr`).
In `push_back`: call `inc_posted` on **all** TC slots; advance the write pointer on slot 0 only (`tc_idx` does not advance).

TC assignment uses `P*C` counter scheme: `counter_id = counter_base + p*C + c`. Each TC slot has its own sub-range: `base_addr = alloc_base + p * capacity * entry_size`, `limit = base_addr + capacity * entry_size`. Consumer has `drain_per_tc = true`: it reads all `capacity` entries from TC0 (producer 0's block) before advancing to TC1 (producer 1's block), etc.

**Test evidence (BLOCKED):** 30 tests in `test_dataflow_buffer.cpp` cover DM-DM, DM→Tensix, and Tensix→DM topologies with both ImplicitSync variants. Example filters: `DMTest1xDFB1Sx1B`, `DMTensixTest1xDFB4Sx4B`, `TensixDMTest1xDFB4Sx2B`.

---

## 5. Program Lifecycle

The DFB lifecycle differs between the two execution paths. The standalone path (tt-emule `EnqueueProgram`) and the JIT/Metal path (`emulated_program_runner` in tt-metal) each set up `EmuleDFBInterface` objects in subtly different ways.

### 5.1 Standalone Path: Host Setup

```cpp
CreateDataflowBuffer(program, DataflowBufferConfig{
    .dfb_index = 0,
    .entry_size = 1024,
    .num_entries = 8,
    .producer_risc_mask = 0x01,  // DM0
    .num_producers = 1,
    .consumer_risc_mask = 0x02,  // DM1
    .num_consumers = 1,
    .producer_access_pattern = STRIDED,
    .consumer_access_pattern = STRIDED,
});
```

RISC masks are set manually; `BindDataflowBufferToProducerConsumerKernels` is not used in the standalone path. The config is stored in `Program::dfb_configs_`. No memory is allocated yet.

### 5.2 Standalone Path: `EnqueueProgram` DFB Initialization

`src/kernel_runner.cpp:EnqueueProgram()` handles DFB setup in two phases before launching threads:

**Phase A: TC and sync state initialization**

For each `DataflowBufferConfig`:
1. Allocate `entry_size * num_entries` bytes from `Core::l1_alloc()`.
2. Call `core.init_dfb_sync(dfb_index, base, entry_size, num_entries, capacity)` to populate `DFBSyncState`.
3. Compute `M = max(num_producers, num_consumers)` and `capacity = num_entries / M`. For each of the `M` TC slots (counter IDs spaced by `dfb_index * MAX_TC_SLOTS_PER_DFB`): set `tc.capacity = capacity` and reset `posted`/`acked` to 0.

**Phase B: `build_dfb_interfaces()`**

Constructs the per-thread `EmuleDFBInterface` arrays. For each `DataflowBufferConfig` and each kernel descriptor:
- Check whether `processor_id`'s bit is set in `producer_risc_mask` or `consumer_risc_mask`. Threads not in either mask get `active = false`.
- If producer p: `num_tcs_to_rr = M/P`; fill `M/P` `DFBTCSlot`s with TC indices `{p + k*P}`, `counter_id = counter_base + tc_idx`, initial `wr_ptr/rd_ptr = base + tc_idx * entry_size`. `counter_base = dfb_index * MAX_TC_SLOTS_PER_DFB`.
- If consumer c: `num_tcs_to_rr = M/C`; fill `M/C` `DFBTCSlot`s with TC indices `{c + k*C}`, same counter_base spacing, initial `rd_ptr/wr_ptr = base + tc_idx * entry_size`.

### 5.3 Standalone Path: Thread Launch and Barrier

A `std::barrier` with count equal to `num_kernels` ensures all threads have received their TLS pointers before any of them begin kernel execution:

```cpp
std::barrier init_barrier(num_threads);
for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
        __processor_id    = kd.processor_id;
        __dfb_ifaces      = dfb_iface_per_thread[i].data();
        __emule_dfbs      = dfb_iface_per_thread[i].data();
        __emule_tc_array  = core.tile_counters();

        init_barrier.arrive_and_wait();  // all threads sync here
        kd.fn();                          // kernel runs after barrier
    });
}
```

This replicates the hardware barrier where all RISCs synchronize after `setup_local_dfb_interfaces()` completes.

### 5.4 Standalone Path: Teardown

After all threads join, `core.reset_dfb_sync()` clears the `DFBSyncState` array. CB state is also reset. The `TileCounterArray` is not destroyed (it persists in `Core`) but its counters were zeroed at the start of Phase A. The per-thread `EmuleDFBInterface` arrays are local to `EnqueueProgram` and are destroyed on return.

### 5.5 JIT/Metal Path: `emulated_program_runner` DFB Setup

The JIT path lives in `tt_metal/impl/emulation/emulated_program_runner.cpp` and integrates with the real tt-metal host APIs. Several key differences from the standalone path:

**Host setup uses `BindDataflowBufferToProducerConsumerKernels`:**

```cpp
auto dfb_id = experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
experimental::dfb::BindDataflowBufferToProducerConsumerKernels(
    program, dfb_id, producer_kernel, consumer_kernel);
```

`Bind` populates `producer_risc_mask` and `consumer_risc_mask` from the kernel handles' processor types. The emulated runner then reads these masks back via `program.impl().get_dataflow_buffer(dfb_id)->config`.

**Per-thread TC assignment:**

The runner applies the same STRIDED TC assignment algorithm as the standalone `build_dfb_interfaces()`: `M = max(P, C)`, `stride_size = M * entry_size`, producer p owns TC slots `{p + k*P}`, consumer c owns `{c + k*C}`. Key implementation details:

1. **Multi-bank DRAM**: `NUM_DRAM_BANKS` is set to the real architecture channel count (Quasar=2, WH=6, BH=8). All bank NOC coordinates are registered in `__emule_core_map` so multi-bank interleaving works correctly.
2. **Multi-thread spawning**: `QuasarDataMovementKernel` — one thread per DM processor via `get_dm_processors()`. `QuasarComputeKernel` — one thread per compute engine (groups of 4 TRISCs into 1 thread via `get_compute_processors()`).
3. **dfb_index bounds**: Assertion that `dfb_index < TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB` (max 8 DFBs with neo_id=0).

**Shared L1 backing store for bridge DFBs:**

When a compute kernel bridges two DFBs (DM→Tensix input DFB and Tensix→DM output DFB), and both DFBs have the same `(entry_size, num_entries)`, the runner allocates L1 **once** and reuses the same base address for both:

```cpp
uint64_t dim_key = (uint64_t(entry_size) << 32) | num_entries;
auto it = dfb_l1_share.find(dim_key);
if (it != dfb_l1_share.end()) {
    base_addr = it->second;   // reuse existing allocation
} else {
    base_addr = core->l1_alloc(total);
    dfb_l1_share[dim_key] = base_addr;
}
```

This models the register file passthrough on real hardware, where the compute RISC's output is visible to the DM consumer in the same L1 region without an extra copy.

**`mhartid` CSR patching:**

Quasar device kernels use `asm volatile("csrr %0, mhartid" : "=r"(var))` to read the hardware thread ID. The runner regex-patches this in the kernel source before JIT compilation:

```cpp
static const std::regex mhartid_re(
    R"(asm\s+volatile\s*\(\s*"csrr\s+%0\s*,\s*mhartid"...)");
std::string patched = std::regex_replace(
    content, mhartid_re,
    "$1 = static_cast<decltype($1)>(__processor_id);");
```

The patched source is written to a temp file and compiled in place of the original. `__processor_id` is the TLS variable set by the runner before each thread starts, derived from `kernel->get_kernel_processor_type(0)` (DM0=0..DM7=7, TRISC0=8..TRISC23=23).

**No `std::barrier`:**

The JIT runner does not use a `std::barrier`. Threads on the same core are launched sequentially and each runs to completion before the next starts. This is a difference from the standalone path, which launches all threads simultaneously with a barrier to synchronize DFB setup.

---

## 6. Two API Paths

tt-emule supports two usage modes that share the same underlying synchronization infrastructure but differ in how kernels are compiled and how state is accessed.

### 6.1 Standalone Path (`include/kernel_api/dfb_dataflow_api.hpp`)

Used by tests under `tests/` that link tt-emule directly. Kernels are native C++ functions (type `KernelFn = std::function<void()>`). State is reached via the `__core` and `__dfb_ifaces` thread-locals:

```cpp
inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n) {
    auto* tc = __core->tile_counters();
    auto& iface = __dfb_ifaces[dfb_id];
    if (!iface.active) return;
    tt_emule::DataflowBuffer dfb(iface, *tc, static_cast<uint16_t>(dfb_id));
    dfb.reserve_back(n);
}
```

A temporary `DataflowBuffer` is constructed per call. This is safe because `DataflowBuffer` holds references, not copies.

### 6.2 JIT Path (`include/jit_hw/api/dfb_api.h`)

Used by tests in tt-metal (`test_dfb_emulation.cpp`) that JIT-compile upstream device kernels. Kernels are compiled from source `.cpp` at runtime and loaded with `dlopen`. The kernel `.so` cannot link against `tt_emule` symbols directly; it calls inline free functions that read thread-local pointers set by the runner:

```cpp
extern thread_local tt_emule::EmuleDFBInterface* __emule_dfbs;
extern thread_local tt_emule::TileCounterArray*  __emule_tc_array;

inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n) {
    auto& iface = __emule_dfbs[dfb_id];
    if (!iface.active) return;
    // ... wait_for with timeout ...
}
```

The JIT path bypasses the `DataflowBuffer` class and implements the same logic inline to avoid any linkage dependency.

The `emulated_program_runner` sets these TLS variables per kernel thread (see §5.5). After the thread completes, the runner clears them back to `nullptr` to prevent stale pointers.

All 72 DFB tests in `test_dataflow_buffer.cpp` pass:

| Access Pattern | Tests | Topology |
|----------------|-------|----------|
| STRIDED | 42 | 24 ImplicitSyncFalse + 18 ImplicitSyncTrue (DM-DM, DM-Tensix, Tensix-DM, multi-DFB) |
| BLOCKED | 30 | DM-DM, DM→Tensix, Tensix→DM, both ImplicitSync variants |

All tests use the real tt-metal host APIs: `CreateDataflowBuffer`, `BindDataflowBufferToProducerConsumerKernels`, `LaunchProgram`.

---

## 7. Timeout and Hang Detection

The JIT path wraps every blocking wait with `std::condition_variable::wait_for`:

```cpp
if (!tc.space_cv.wait_for(lk,
        std::chrono::seconds(__emule_dfb_timeout_sec()),
        [&]{ return tc.free_space() >= n; })) {
    fprintf(stderr, "EMULE HANG: dfb_reserve_back(dfb=%u, n=%u) timed out "
            "on TC(%u,%u) after %ds\n", ...);
    std::abort();
}
```

The timeout defaults to 120 seconds and can be overridden with the `TT_EMULE_DFB_TIMEOUT` environment variable. On expiry, the process aborts with a diagnostic message identifying the specific DFB and TC involved.

The standalone path (via `DataflowBuffer` and `TileCounterArray`) uses `wait` without a timeout. Hangs there require `TT_EMULE_DFB_TIMEOUT` to be surfaced via the JIT path, or manual debugging.

---

## 8. Known Limitations and Remaining Gaps

### 8.1 TRISC `finish()` variant not distinguished

Hardware `finish()` has two behaviors: DM waits for `posted == acked` (drain), while TRISC waits for `posted == 0` (all slots empty from the start). The emulation implements the DM variant only.

**TODO: No test exercises TRISC `finish()` in isolation.** See `docs/TEST_COVERAGE_TODO.md`.

### 8.2 `BindDataflowBufferToProducerConsumerKernels` not wired in standalone path

In the standalone path, RISC masks must be set manually in `DataflowBufferConfig`. The upstream `BindDataflowBufferToProducerConsumerKernels` host API is used correctly in the JIT/Metal path (§5.5) but is not available to tt-emule's `CreateDataflowBuffer` / `EnqueueProgram` directly.

### 8.3 dfb_index bounds

With `neo_id=0` (current default), the maximum number of DFBs per program is `TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB = 8`. Counter IDs are spaced by `MAX_TC_SLOTS_PER_DFB` (4) per DFB to prevent cross-DFB collision. A `std::out_of_range` exception is thrown if `dfb_index >= 8`. Multi-NEO spreading (distributing DFBs across neo_ids) would be needed for >8 DFBs.

### 8.4 L1 architecture

The 4 MB L1 is shared between all 12 cores in a Neo (8 DM processors + 4 compute engines). The emulation models this correctly with a single `Core` object per Neo that all threads access. This is documented here for architectural clarity — the emulation's shared-memory model naturally reflects the hardware's shared L1.

### 8.5 Standalone path has no timeout on blocking waits

The JIT path (`dfb_api.h`) wraps all blocking waits with `wait_for` and a configurable timeout (default 120s, `TT_EMULE_DFB_TIMEOUT`). The standalone path (`DataflowBuffer` via `TileCounterArray`) uses `wait` without any timeout, so hung standalone tests block indefinitely.

**TODO: No test exercises standalone path timeout behavior.** See `docs/TEST_COVERAGE_TODO.md`.

---

## 9. File Reference

| File | Role |
|------|------|
| `include/tt_emule/tile_counter.hpp` | `TileCounter`, `TileCounterArray` |
| `include/tt_emule/dfb_sync_state.hpp` | `DFBTCSlot`, `EmuleDFBInterface`, `DFBSyncState` |
| `include/tt_emule/dataflow_buffer.hpp` | `DataflowBuffer` (standalone kernel object) |
| `include/tt_emule/device.hpp` | `Core` — owns `TileCounterArray` and `DFBSyncState[32]` |
| `include/tt_emule/program.hpp` | `DataflowBufferConfig`, `AccessPattern`, `QuasarDM/QuasarCompute`, `processor_id` |
| `include/tt_emule/host_api.hpp` | `CreateDataflowBuffer`, Quasar `CreateKernel` |
| `src/host_api.cpp` | Implementation of above |
| `src/kernel_runner.cpp` | `build_dfb_interfaces()`, `EnqueueProgram` DFB setup, `std::barrier` |
| `include/jit_hw/emule_dfb_state.h` | TLS declarations: `__emule_dfbs`, `__emule_tc_array` |
| `include/jit_hw/api/dfb_api.h` | JIT DFB free functions with timeout detection |
| `include/jit_hw/experimental/dataflow_buffer.h` | `experimental::DataflowBuffer` class (JIT wrapper) |
| `include/kernel_api/dfb_dataflow_api.hpp` | Standalone DFB free functions |
| `tests/dfb_passthrough/` | Standalone end-to-end test: 1 DM producer + 1 DM consumer, 8×1 KB entries |
| `tests/dfb_multi_consumer/` | Standalone 1P-4C STRIDED test |
| *(tt-metal)* `tt_metal/impl/emulation/emulated_program_runner.cpp` | JIT path: DFB L1 alloc, shared-backing for bridges, per-thread interface construction, `__processor_id` TLS, `mhartid` CSR patching |
| *(tt-metal)* `tests/tt_metal/tt_metal/api/dataflow_buffer/test_dataflow_buffer.cpp` | 72 STRIDED + BLOCKED DFB tests (all P/C combinations, both ImplicitSync modes) |
