# tt-emule

A C++ software emulator of the Tenstorrent device-level kernel and host APIs. Run tt-metal kernel
and host code on a standard x86-64 Linux machine — no Tenstorrent hardware, drivers, or firmware
required.

## Overview

tt-emule provides two things:

1. **A runtime**: emulates the on-chip compute model — per-core L1 SRAM, DRAM banks, circular
   buffers (CBs) and dataflow buffers (DFBs), the DST register file, the NOC, and cross-core
   semaphores — running each kernel role as an OS thread.
2. **A host-side integration layer**: plugs into tt-metal via `-DTT_METAL_USE_EMULE=ON`. Existing
   tt-metal host code, test binaries, and TTNN ops compile and run unchanged against tt-emule.

### Supported architectures

| Architecture | Cores | Threads per core | Notes |
|---|---|---|---|
| Wormhole (N150) | Tensix worker grid + DRAM | 3 (reader, compute, writer) | Primary target |
| Blackhole (P100) | Tensix worker grid + DRAM | 3 | Tier 4 INT32 ops |
| Quasar (Q1) | Neo (Tensix Neo) | Up to 12 (DM0–DM7, E0–E3) | DFB-based sync |

### What it emulates

| Hardware concept | Emulation |
|---|---|
| Core L1 SRAM (1–4 MB) | `MAP_32BIT` `mmap` slots from a shared `L1Pool`; 32-bit kernel pointer casts work on x86-64 |
| DRAM (real bank count) | Per-bank `mmap` regions with proper NOC address routing |
| Data-movement threads | OS threads (DM0/DM1 on WH/BH; DM0–DM7 on Quasar) |
| Compute thread(s) | OS threads (1 on WH/BH; E0–E3 on Quasar) |
| Circular buffers (WH/BH) | Mutex + condition variable (`CBSyncState`) |
| Dataflow buffers (Quasar) | MPMC tile-counter synchronization (`DFBSyncState`) |
| DST register file | 8 slots (fp32) / 16 slots (bf16) with acquire/commit/wait/release state machine |
| NOC async read/write | Synchronous `memcpy`; multicast via per-core address map |
| Remote semaphores | Cross-core atomic post/wait via thread-local semaphore state |
| Compile-time args | Passed via `-DKERNEL_COMPILE_TIME_ARGS=val,...` to the JIT compiler |
| Runtime args | Thread-local `std::vector<uint32_t>` injected before each kernel runs |
| CSR reads (Quasar) | `mhartid`, `NEO_ID`, `TRISC_ID` patched to read thread-local variables at JIT time |

### What it does NOT emulate

- Real RISC-V ISA or cycle-accurate timing
- Actual async NOC (all DMA is synchronous `memcpy`)
- Ethernet/dispatch fabric, UMD, or Metal dispatch layers
- NOC stream overlay registers (`noc_overlay_parameters.h`)
- Real tile layout conversions (`tilize_block`/`untilize_block` are identity copies)

---

## Repository layout

```
tt-emule/
├── include/
│   ├── tt_emule/           # Host-side types and API
│   │   ├── tile.hpp                    # Tile: 32×32 fp32 matrix (4096 bytes)
│   │   ├── tile_counter.hpp            # Atomic tile counter for DFB sync
│   │   ├── circular_buffer.hpp         # Thread-safe bounded CB (WH/BH)
│   │   ├── cb_sync_state.hpp           # CB mutex+condvar state
│   │   ├── dataflow_buffer.hpp         # DFB handle (Quasar)
│   │   ├── dfb_sync_state.hpp          # DFB MPMC tile-counter state
│   │   ├── dst_register_file.hpp       # DST register file with mode-aware bounds
│   │   ├── l1_pool.hpp                 # Shared MAP_32BIT L1 allocator
│   │   ├── buffer.hpp                  # DRAM-backed buffer handle
│   │   ├── device.hpp                  # Core, Device, CoreRole, get_core()
│   │   ├── program.hpp                 # Program, KernelDescriptor, configs
│   │   └── host_api.hpp                # Full host API declaration
│   ├── hw/                 # Real hardware headers used without stubbing
│   │   └── inc/api/debug/dprint.h
│   └── jit_hw/             # Kernel-side stubs compiled into JIT-loaded .so files
│       ├── jit_kernel_stubs.hpp        # Thread-local forward decls, get_arg_addr, ASSERT
│       ├── api/                        # Compute + dataflow kernel APIs
│       │   ├── cb_api.h                # CB ops (wh/bh)
│       │   ├── dfb_api.h               # DFB ops (Quasar)
│       │   ├── dataflow/dataflow_api.h # NOC ops, noc_async_*, noc_inline_dw_write
│       │   ├── compute/                # Full LLK-based compute API
│       │   └── tensor/tensor_accessor.h
│       ├── experimental/               # Quasar experimental APIs (NOC, semaphore, tensor)
│       └── internal/                   # risc_attribs, firmware_common, mod_div_lib
├── src/
│   ├── host_api.cpp        # Device, buffer, program, command queue implementations
│   ├── kernel_runner.cpp   # EnqueueProgram: thread spawning, CB/DFB lifecycle
├── tests/
│   ├── tilize/             # CB round-trip: tilize → untilize
│   └── integration/        # Emulation toggle test
├── docs/                   # Architecture deep-dives (Quasar, DFB, etc.)
├── run_regression.sh       # Tiered C++ regression suite (Tier 1–6)
├── run_d2m_regression.sh   # tt-mlir D2M pytest regression driver
├── tt-metal-pin.txt        # Pinned tt-metal SHA used by CI
└── tt-mlir-pin.txt         # Pinned tt-mlir SHA used by CI
```

---

## Building

tt-emule integrates into tt-metal and is not built standalone. See **[GETTING_STARTED.md](GETTING_STARTED.md)** for the full setup — clone, submodule init, and build commands. See **[BUILD_GUIDE.md](BUILD_GUIDE.md)** for the detailed phase-by-phase walkthrough including the D2M (tt-mlir) build.

**Quick reference — C++ regression build** (clang-20 + libc++):

```bash
export ROOT=$HOME/work   # parent of tt-metal and tt-emule checkouts
cmake -S . -B build_emule -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH=$ROOT/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DENABLE_TRACY=OFF \
    -DCMAKE_INSTALL_PREFIX=$ROOT/tt-metal/build_emule
cmake --build build_emule -j$(nproc)
```

**Prerequisites**: clang-20, libc++-20-dev, libc++abi-20-dev, CMake ≥ 3.24, Ninja ≥ 1.10.

---

## Regression

### C++ regression (Tier 1–6)

```bash
cd $ROOT/tt-emule
TT_METAL_DIR=$ROOT/tt-metal ./run_regression.sh 2>&1 | tee regression.log
```

Runs six test tiers against pre-built gtest binaries in `build_emule/`:

| Tier | What it covers | Target arch |
|---|---|---|
| 1 | Host-only (bit utils, buffer ops, CoreRange/Set) | Any |
| 2/3 | Buffer allocation, L1 tile transfers, JIT kernels | Wormhole N150 |
| 3b–3j | DFB, Quasar compute, semaphores, atomics, multi-core matmul PCC | Quasar Q1 |
| 3k/3l | Data movement (loopback, NOC, DRAM unary) | Wormhole N150 |
| 4 | ttnn INT32 ops | Blackhole P100 |
| 5/6 | ttnn bf16 matmul + emulation toggle | Wormhole N150 |

Known failures are tracked in `.github/known-failures-wormhole.txt` and `.github/known-failures-blackhole.txt`; CI fails the PR if any test outside those files regresses, or if any listed entry starts passing.

### D2M regression (tt-mlir)

```bash
cd $ROOT/tt-emule
export SYSTEM_DESC_PATH=$ROOT/tt-mlir/ttrt-artifacts/system_desc.ttsys
export BUILD_DIR=$ROOT/tt-metal/build_Release
./run_d2m_regression.sh --serial 2>&1 | tee d2m_regression.log
```

Runs 33 `test/python/golden/d2m/test_*.py` files from the pinned tt-mlir checkout.
Expected: ~13 passed, ~20 failed (see `.github/known-failures-d2m-wormhole.txt`).
Runtime: ~70 minutes serial. See BUILD_GUIDE.md Phase 5–6 for full D2M setup.

---

## Concepts

### Memory model

Worker cores draw L1 from a shared `L1Pool` — a single contiguous `MAP_32BIT` mmap with 2 MB aligned
slots, one per core. This makes the address space 32-bit, so kernel `uint32_t` offsets and raw
pointer casts work identically to hardware. DRAM uses per-bank mmaps with real bank coordinates
registered in the emulated core map.

### Thread model (Wormhole / Blackhole)

Each emulated core runs three OS threads:

| Thread | Entry point | Role |
|---|---|---|
| NOC reader | `reader_kernel_main()` | Reads DRAM/L1 → CBs via NOC |
| Compute | `compute_kernel_main()` | Processes tiles CB → DST → CB |
| NOC writer | `writer_kernel_main()` | Writes CBs → DRAM/L1 via NOC |

Threads synchronize through `CBSyncState` (mutex + condition variable).

### Thread model (Quasar)

Each emulated Neo runs up to 12 OS threads:

| Thread type | Count | Entry point | Role |
|---|---|---|---|
| DM (DM0–DM7) | up to 8 | DM kernel `main()` | NOC + DFB push/pop |
| Compute (E0–E3) | up to 4 | Compute kernel `main()` | UNPACK/MATH/PACK on shared DST |

DFBs replace circular buffers: MPMC tile-counter sync allows multiple DM and compute threads to
share the same buffer without lock contention. See [docs/QUASAR_EMULATION.md](docs/QUASAR_EMULATION.md).

### DST register file

16 slots in bf16 mode / 8 slots in fp32 mode, each holding 1024 fp32 elements. Mode is set per
program. The compute thread owns DST exclusively during `acquire`/`commit`; packers read during
`wait`/`release`.

### NOC

All NOC transfers are synchronous `memcpy`. Multicast is implemented by iterating over the per-core
address map for each target. Inline DW writes (`noc_inline_dw_write`) are 4-byte memcpy. Barriers
are no-ops.

---

## JIT Kernel Headers

tt-emule provides `include/jit_hw/` — a set of headers that replace the real RISC-V hardware
headers when kernel `.cpp` files are compiled by tt-metal's JIT engine (in
`tt_metal/impl/emulation/emulated_program_runner.cpp`). These headers wire the standard
tt-metal kernel API down to tt-emule's thread-local emulation state.

| Path | Provides |
|---|---|
| `jit_kernel_stubs.hpp` | Thread-local forward decls, `get_arg_addr`, `ASSERT` |
| `api/cb_api.h` | `cb_reserve_back`, `cb_push_back`, `cb_wait_front`, `cb_pop_front` |
| `api/dfb_api.h` | DFB push/pop ops (Quasar) |
| `api/dataflow/dataflow_api.h` | `noc_async_read/write`, `noc_inline_dw_write`, multicast |
| `api/compute/` | Full LLK-based compute API (add, matmul, reduce, unary SFPU, pack, tilize, ...) |
| `api/tensor/tensor_accessor.h` | Tensor accessor stubs |
| `experimental/` | Quasar: `experimental::Noc`, semaphore, lock, tensor |
| `internal/risc_attribs.h` | `tt_l1_ptr` (no-op), `FORCE_INLINE` |
| `api/compile_time_args.h` | `get_compile_time_arg_val(N)` |
| `api/debug/dprint.h` | `DPRINT` (no-op sink) |

---

## tt-metal Integration

tt-emule integrates into tt-metal as a drop-in hardware backend. When `-DTT_METAL_USE_EMULE=ON`
is passed to cmake, tt-metal's device dispatch, program execution, and JIT kernel compilation
are redirected through tt-emule. No changes to test code or kernel files are required.

The key env vars that gate emulation at runtime:

| Variable | Value | Effect |
|---|---|---|
| `TT_METAL_EMULE_MODE` | `1` | Activates emulated device |
| `TT_METAL_SLOW_DISPATCH_MODE` | `1` | Required for emulation (no fast dispatch) |
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | path to YAML | Which architecture to emulate |

Cluster YAMLs are in `tt_metal/third_party/umd/tests/cluster_descriptor_examples/`.

---

## Known limitations

- **Real RISC-V ISA**: no cycle-accurate timing; all threads run on host CPU
- **NOC stream overlay**: `noc_overlay_parameters.h` / `stream_io_map.h` not stubbed; tests that use stream registers will fail JIT compilation
- **Inline L1 pointer dereference**: kernels that dereference raw L1 offsets as host pointers (not going through the NOC API) will segfault
- **Tilize/untilize**: identity copies; real 32×32 row-major ↔ tile-major conversion not implemented

---

## Contributing

Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
workflow, build/test expectations, and contributor-license details.

## Security

Please report security issues using GitHub's private vulnerability
reporting on this repository, or email `ospo@tenstorrent.com`. See
[SECURITY.md](SECURITY.md) for details.

## License

This project is licensed under the Apache License, Version 2.0 — see
[LICENSE](LICENSE) and [LICENSE_understanding.txt](LICENSE_understanding.txt).
Documentation under `docs/` is licensed under
[CC-BY-4.0](LICENSE-DOCS).
