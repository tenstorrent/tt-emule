# API injection points — where emule intercepts silicon

When a test runs through emule, the journey from Python `assert
PCC` back to the kernel that produced the data crosses these
emule-controlled layers. Each layer is an *injection point* where
emule can intercept silicon behavior. Knowing which layer to edit is
half the battle when adding a mock.

---

## 1. Layer overview

```
Python test (ttnn.from_torch + compile + assert PCC)
         │
         ▼
ttnn / tt-metal Python bindings ──── TT_METAL_EMULE_MODE=1
         │                          (forces slow dispatch +
         ▼                           SWEmulatedChip)
Program build → CommandQueue.EnqueueProgram
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ INJECTION POINT 1                                          │
│ emulated_program_runner.cpp                                │
│   - source rewriter (3 regex patterns, top-level only)     │
│   - jit_compile_kernel (emit wrapper.cpp → clang++ → .so)  │
│   - cache key (incl. consumer aggregator hash if used)    │
│   - NOC bridge (__emule_resolve_noc_addr, multicast)       │
│   - thread launch (per-core BRISC/NCRISC/TRISC threads)    │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ INJECTION POINT 2                                          │
│ tt-emule/include/jit_hw/  (JIT include shim)               │
│   First-priority -I path for all JIT compiles.             │
│   Anything #include "api/..." resolves here first.         │
│   ~70 files: dataflow, compute, tensor, experimental,      │
│   fabric (shadow), LLK (stub).                             │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ INJECTION POINT 3                                          │
│ Consumer aggregator header (e.g., ops.hpp) allowlist       │
│   Scope-reduction gate: only emule-allowed ops are pulled  │
│   in under __EMULE_JIT_MODE                                │
└────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ INJECTION POINT 4                                          │
│ Per-op patches in consumer op headers                      │
│   #ifdef __EMULE_JIT_MODE branches for:                    │
│   - L1-cast translation (constexpr → __emule_local_l1_to_ptr) │
│   - Semantic rewrite of TRISC body                         │
│   - Cross-RISC sync gate-off                               │
└────────────────────────────────────────────────────────────┘
         │
         ▼
Per-core / per-RISC thread runs JIT-compiled kernel_main()
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│ INJECTION POINT 5                                          │
│ SWEmulatedChip (UMD level, sw_emule_chip.cpp)              │
│   write_to_device/read_from_device delegate to             │
│   tt_emule::Core::l1_ptr() + memcpy                        │
└────────────────────────────────────────────────────────────┘
```

---

## 2. Where to inject for which kind of change

| Want to change… | Inject at | File |
|---|---|---|
| How a silicon API is compiled (e.g. add a new define) | Point 1: `emulated_program_runner.cpp::jit_compile_kernel` | `${TT_METAL_DIR}/tt_metal/impl/emulation/emulated_program_runner.cpp` |
| How a kernel.cpp source text is rewritten before compile | Point 1: `preprocess_kernel_source_for_x86` regex | same file, line ~250 |
| What's in the JIT cache key | Point 1: `compute_cache_key` lambda | same file, line ~1265 |
| How a NOC address resolves to a host pointer | Point 1: `__emule_resolve_noc_addr` | same file, line ~190 |
| How a multicast write reaches receivers | Point 1: `__emule_multicast_write` | same file, line ~205 |
| Add a NEW silicon API surface (generic, multi-op) | Point 2: new header in `include/jit_hw/...` | various |
| Change existing emule mock semantics | Point 2: `jit_hw/api/{compute, dataflow, tensor}/*.h` | various |
| Add a consumer op to emule scope | Point 3: append to consumer's `__EMULE_JIT_MODE` allowlist header | consumer-defined |
| Fix a per-op silicon-specific pattern | Point 4: `#ifdef __EMULE_JIT_MODE` patch in consumer op header | consumer-defined |
| Reimplement an op's TRISC body for emule | Point 4: semantic rewrite under `__EMULE_JIT_MODE` | same |
| How L1 memory is allocated per core | Point 5: `Core` constructor (mmap'd L1), invoked by `SWEmuleChip` | `include/tt_emule/device.hpp` |
| How `write_to_device` lands in L1 | Point 5: `SWEmulatedChip::write_to_device` | `${TT_METAL_DIR}/tt_metal/third_party/umd/device/chip/sw_emule_chip.cpp` |
| Make a host bridge function visible to JIT kernels | Point 2 (decl) + Point 1 (def): `extern "C"` in `jit_kernel_stubs.hpp`, def in `emulated_program_runner.cpp` | both |

---

## 3. Build-time vs runtime injection

| Injection | Time | Recompile required |
|---|---|---|
| Point 1 (`emulated_program_runner.cpp`) | Runtime, compiled into `libtt_metal.so` | Yes — rebuild tt-metal (`cmake --build build_emule`) |
| Point 2 (`tt-emule/include/jit_hw/`) | JIT compile time (#include'd by every kernel) | No — header-only; new JIT compiles pick it up |
| Point 3 (consumer aggregator header) | JIT compile time | No — header-only; wipe `/tmp/tt_emule_jit_*` to force fresh compile |
| Point 4 (per-op consumer header) | JIT compile time | No — header-only |
| Point 5 (`SWEmulatedChip`, `Core`) | Runtime, in `libtt_metal.so` (UMD) | Yes — rebuild tt-metal |

**Rule of thumb**: 80% of mock work is at Points 2 and 4 (no rebuild,
fast iteration). Only touch Points 1 / 5 if the change is in
JIT-orchestration code or fundamental device emulation.

---

## 4. Bridge functions (host ↔ JIT crossing)

Anything that needs to call from JIT-compiled .so back into the host
process (e.g. to access the chip's core map) goes through a C-linkage
bridge functions declared in the JIT shim headers and defined in
`emulated_program_runner.cpp`:

    // declarations (jit_kernel_stubs.hpp):
    extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);
    extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr);

    // declarations (api/dataflow/dataflow_api.h and experimental/noc.h):
    extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr);
    extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size);

These resolve at `dlopen()` time when the kernel .so is loaded into the
host process's address space (RTLD_GLOBAL'd `libtt_metal.so` provides
the symbols).

**TLS state** is also shared this way:
```cpp
extern thread_local uint8_t* __emule_bridge_l1;     // current core's L1 base
extern thread_local __emule_cb_state* __emule_cbs;  // current core's CBSyncState[]
extern thread_local tt_emule::Core*   __core;
extern thread_local std::vector<uint32_t> __rt_args;
```

When a thread is launched for (core, processor) in
`execute_program_emulated`, the launcher sets these TLS pointers before
calling `__emule_kernel_entry()`.

---

## 5. The `__EMULE_JIT_MODE` macro

Defined in `tt-emule/include/jit_hw/jit_kernel_stubs.hpp` (which is
pre-included at the top of every JIT-emitted `wrapper.cpp`). It's the
universal "are we running under emule" flag for all code that needs an
emule branch.

Pattern used by:
- Per-op op.hpp patches (L1-cast translation)
- Semantic rewrites (gate off LLK chain + reimplement)
- Cross-RISC sync skipping
- Allowlist gate in a consumer's aggregator header
- Sentinel-related stubs (e.g. `deepseek_compute_kernel_hw_startup`
  no-op branch in `eltwise_mul_scalar.h`)

Silicon path is preserved untouched — every `__EMULE_JIT_MODE` branch
has an `#else` with the original silicon code, so on-silicon builds
are bit-exact unchanged.
