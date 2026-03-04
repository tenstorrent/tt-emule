# tt-emule

A C++ software emulator of the Tenstorrent device-level kernel and host APIs. Run tt-metal kernel
and host code on a standard x86-64 Linux machine — no Tenstorrent hardware, drivers, or firmware
required.

## Overview

tt-emule provides two things:

1. **A runtime**: emulates the on-chip compute model — L1 SRAM per core, DRAM, circular buffers
   (CBs), the DST register file, and the NOC — running each kernel role as an OS thread.
2. **A host API**: mirrors the `tt::tt_metal` host API (`CreateDevice`, `CreateKernel`,
   `EnqueueProgram`, etc.) so existing tt-metal host code compiles against tt-emule with minimal
   or no changes.

### What it emulates

| Hardware concept | Emulation |
|---|---|
| Core L1 SRAM (1 MB) | `mmap`'d at hint `0x40000000` so 32-bit kernel pointer casts work on x86-64 |
| DRAM (256 MB) | `std::vector<uint8_t>` with bump allocator |
| Data-movement threads (RISCV_0 / RISCV_1) | Two `std::thread`s per core (DM0 / DM1) |
| Compute thread | One `std::thread` per core |
| Circular buffers | Thread-safe FIFO using `std::condition_variable` |
| DST register file (8 slots) | `DstRegisterFile` with acquire/commit/wait/release state machine |
| NOC async read/write | Synchronous `memcpy` |
| Compile-time args | `-DKERNEL_COMPILE_TIME_ARGS=val,...` passed to JIT compiler |
| Runtime args | Thread-local `std::vector<uint32_t>` injected before each kernel runs |

### What it does NOT emulate

- Multi-core topologies (single core at `(0,0)` only)
- Real RISC-V ISA or cycle-accurate timing
- Bfloat16 / other data types (tiles use `float`)
- Actual async NOC (all DMA is synchronous)
- Real tile layout conversions (`tilize_block`/`untilize_block` are identity copies)
- Ethernet/dispatch fabric, UMD, or Metal dispatch layers

---

## Repository layout

```
tt-emule/
├── include/
│   ├── tt_emule/          # Host-side types and API
│   │   ├── tile.hpp                  # Tile: 32×32 float matrix (4096 bytes)
│   │   ├── circular_buffer.hpp       # Thread-safe bounded FIFO of Tile slots
│   │   ├── dst_register_file.hpp     # DST register file with state machine
│   │   ├── buffer.hpp                # DRAM-backed buffer handle
│   │   ├── device.hpp                # Core, Device, MockAllocator
│   │   ├── program.hpp               # Program, KernelDescriptor, configs
│   │   ├── host_api.hpp              # Full host API declaration
│   │   ├── tt_metal_compat.hpp       # tt::tt_metal namespace bridge
│   │   └── jit_kernel.hpp            # JIT kernel compiler declaration
│   ├── kernel_api/        # In-kernel API (included by kernel .cpp files)
│   │   ├── common.hpp                # Thread-local context (__rt_args, __core, __device)
│   │   ├── dataflow_api.hpp          # CB ops, NOC ops
│   │   ├── compute_api.hpp           # DST ops, tile math (add, matmul, copy, pack)
│   │   └── kernel_includes.hpp       # Convenience: includes all kernel_api headers
│   ├── ttkernel/          # Forwarding headers for tt-metal kernel include paths
│   │   ├── dataflow_api.h            # → kernel_api/dataflow_api.hpp
│   │   └── compute_kernel_api/
│   │       ├── common.h              # → kernel_api/{common, compute_api}.hpp
│   │       ├── matmul.h              # → kernel_api/compute_api.hpp
│   │       └── tile_move_copy.h      # → kernel_api/compute_api.hpp
│   └── jit_hw/            # JIT-specific stubs (included during runtime g++ compile)
│       ├── jit_kernel_stubs.hpp      # Thread-local forward decls, get_arg_addr, ASSERT
│       ├── internal/
│       │   └── risc_attribs.h        # tt_l1_ptr (no-op), FORCE_INLINE
│       └── api/
│           ├── compile_time_args.h   # get_compile_time_arg_val(N) macro
│           ├── debug/dprint.h        # DPRINT stub
│           └── dataflow/dataflow_api.h  # JIT-flavour dataflow include
├── src/
│   ├── host_api.cpp        # Device, buffer, program, command queue implementations
│   ├── kernel_runner.cpp   # EnqueueProgram: thread spawning, CB lifecycle
│   └── jit_kernel.cpp      # JIT engine: mkdtemp → g++ → dlopen → dlsym
└── tests/
    ├── eltwise_add/        # 3-kernel pipeline: reader → add_tiles → writer
    ├── matmul/             # 3-kernel pipeline: reader → matmul_tiles → writer
    ├── tilize/             # CB round-trip: tilize → untilize (no kernels)
    └── compat/             # tt::tt_metal namespace smoke test
```

---

## Building

**Requirements**: GCC ≥ 11, CMake ≥ 3.16, POSIX threads, `libdl`.

```bash
# Debug (includes -fsanitize=thread)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build -V

# Release
cmake -B build_rel -DCMAKE_BUILD_TYPE=Release
cmake --build build_rel -j$(nproc)
```

**CMake options**:

| Option | Default | Description |
|---|---|---|
| `TT_EMULE_COMPAT_HEADERS` | `OFF` | Install `ttkernel/` forwarding headers alongside the library |

---

## Concepts

### Tile

`tt_emule::Tile` is a 32×32 matrix of `float` (4096 bytes). Kernel data flows between DRAM and
the core exclusively as tiles.

```cpp
Tile t(1.0f);              // all elements = 1.0
float v = t(row, col);     // element access
uint8_t* raw = t.bytes();  // raw byte pointer (for memcpy)
```

### Circular Buffer

`CircularBuffer` is a thread-safe bounded FIFO of `Tile` slots shared between threads. Producers
(data-movement) and consumers (compute) synchronize through condition variables.

Producer flow:
```
cb_reserve_back(cb_id, n)  // block until n slots free
<write to get_write_ptr()>
cb_push_back(cb_id, n)     // signal consumers
```

Consumer flow:
```
cb_wait_front(cb_id, n)    // block until n tiles available
<read from get_read_ptr()>
cb_pop_front(cb_id, n)     // signal producers
```

### DST Register File

The DST register file has 8 tile slots. The compute thread owns DST exclusively during
`acquire`/`commit`. A packer thread waits on `wait` and releases with `release`.

```
tile_regs_acquire()  →  tile_regs_commit()   [compute thread]
tile_regs_wait()     →  tile_regs_release()  [packer / pack_tile]
```

### Device and NOC

`Device` owns one `Core` at coordinates `(0,0)` and 256 MB of DRAM. NOC address resolution:

- `(x=0, y=0, addr)` → Core L1 at byte offset `addr`
- Any other `(x, y, addr)` → DRAM at byte offset `addr`

### Thread model

`EnqueueProgram` spawns one OS thread per `KernelDescriptor` in the program. All threads share
the same `Core` and `Device`. Thread-local variables (`__rt_args`, `__core`, `__device`) inject
per-kernel context before `KernelFn()` is called. All threads are joined before the call returns.

---

## Host API

### Native API (`tt_emule::`)

```cpp
#include "tt_emule/host_api.hpp"

// Device lifecycle
Device*      CreateDevice(uint32_t id = 0);
bool         CloseDevice(Device* device);
CommandQueue CreateCommandQueue(Device& device);

// Program construction
Program      CreateProgram();

KernelHandle CreateKernel(Program&, KernelFn, CoreCoord, DataMovementConfig);
KernelHandle CreateKernel(Program&, KernelFn, CoreCoord, ComputeConfig);

void         SetRuntimeArgs(Program&, KernelHandle, CoreCoord, std::vector<uint32_t>);
CBHandle     CreateCircularBuffer(Program&, CoreCoord, CircularBufferConfig);

// Buffers
std::shared_ptr<Buffer> CreateBuffer(Device&, size_t size_bytes, uint32_t page_size);
void EnqueueWriteBuffer(Device&, Buffer&, const void* src, bool blocking = true);
void EnqueueReadBuffer(Device&,  Buffer&, void* dst,       bool blocking = true);
// CommandQueue overloads exist for all buffer and program operations

// Execution
void EnqueueProgram(Device&, Program&, bool blocking = true);
void Finish(Device&);

// detail:: helpers (mirror tt-metal detail::)
namespace detail {
    void LaunchProgram(Device*, Program&, bool wait = true, bool force_slow = false);
    bool ReadFromDeviceL1(Device*, const CoreCoord&, uint32_t address, uint32_t size,
                          std::vector<uint32_t>& result);
    template<typename T> void WriteToBuffer(const std::shared_ptr<Buffer>&, const std::vector<T>&);
    template<typename T> void ReadFromBuffer(const std::shared_ptr<Buffer>&, std::vector<T>&);
}
```

### tt-metal Compatibility Bridge (`tt::tt_metal::`)

Include `tt_emule/tt_metal_compat.hpp` to get the full `tt::tt_metal` namespace populated with
tt-emule-backed implementations:

```cpp
#include "tt_emule/tt_metal_compat.hpp"
using namespace tt::tt_metal;

IDevice*      device = CreateDevice(0);
CommandQueue  cq     = CreateCommandQueue(*device);
Program       prog   = CreateProgram();

// InterleavedBufferConfig overload for CreateBuffer
auto buf = CreateBuffer(InterleavedBufferConfig{
    .device      = device,
    .size        = 4 * Tile::SIZE_BYTES,
    .page_size   = Tile::SIZE_BYTES,
    .buffer_type = BufferType::DRAM,
});

// SetRuntimeArgs accepts std::array or any contiguous container
std::array<uint32_t, 3> args = {offset, n_tiles, tile_size};
SetRuntimeArgs(prog, kid, core, args);

// detail:: re-exported
detail::LaunchProgram(device, prog);
```

---

## Kernel API

Kernel `.cpp` files use the kernel-side headers. Include either the convenience header or
individual headers:

```cpp
// Option A: all-in-one
#include "kernel_api/kernel_includes.hpp"

// Option B: pick what you need
#include "kernel_api/common.hpp"       // get_arg_val<T>(idx)
#include "kernel_api/dataflow_api.hpp" // CB ops, NOC ops
#include "kernel_api/compute_api.hpp"  // DST ops, tile math
```

The kernel entry function must be named after its role (to avoid linker conflicts when all three
kernels are linked into a single test binary):

| Role | Entry function |
|---|---|
| DataMovement0 (NOC reader) | `reader_kernel_main()` |
| DataMovement1 (NOC writer) | `writer_kernel_main()` |
| Compute | `compute_kernel_main()` |

> When using the JIT path (below), the entry function must be named `kernel_main()`.

### Dataflow API

```cpp
// Circular buffer
cb_reserve_back(cb_id, n);  cb_push_back(cb_id, n);    // producer
cb_wait_front(cb_id, n);    cb_pop_front(cb_id, n);     // consumer

uint8_t*       get_write_ptr(cb_id);
const uint8_t* get_read_ptr(cb_id);

// NOC
NocAddr get_noc_addr(x, y, addr);                          // resolve to raw pointer
noc_async_read(src_noc, dst_l1, size);                     // memcpy src → dst
noc_async_write(src_l1, dst_noc, size);                    // memcpy src → dst
noc_async_read_barrier();  noc_async_write_barrier();      // no-ops
```

### Compute API

```cpp
// DST state machine
tile_regs_acquire();  tile_regs_commit();    // compute thread
tile_regs_wait();     tile_regs_release();   // after pack_tile

// Tile operations
add_tiles(icb0, icb1, itile0, itile1, idst);          // DST[idst] = CB0[t0] + CB1[t1]
copy_tile(icb0, itile, idst);                          // DST[idst] = CB0[t]
pack_tile(idst, ocb);                                  // CB_out[write] = DST[idst]
matmul_tiles(icb0, icb1, itile0, itile1, idst, false); // DST[idst] += CB0[t0] × CB1[t1]
tilize_block(icb, n, ocb);                             // copy n tiles (identity)
untilize_block(icb, n, ocb);                           // copy n tiles (identity)
```

### Runtime and compile-time arguments

```cpp
// Runtime args (set by host via SetRuntimeArgs)
uint32_t val = get_arg_val<uint32_t>(idx);

// Compile-time args (set by host via DataMovementConfig::compile_args)
uint32_t val = get_compile_time_arg_val(N); // N must be a compile-time constant
```

---

## JIT Kernel Compilation

The JIT path lets you point `CreateKernel` at a `.cpp` source file path instead of a function
pointer. At runtime tt-emule:

1. Creates a temporary directory under `/tmp/tt_emule_jit_XXXXXX`.
2. Writes a `wrapper.cpp`:
   ```cpp
   #include "jit_kernel_stubs.hpp"
   #include "/absolute/path/to/kernel.cpp"
   extern "C" { void __emule_kernel_entry() { kernel_main(); } }
   ```
3. Compiles it:
   ```
   g++ -std=c++17 -fPIC -shared -O1
       -DKERNEL_COMPILE_TIME_ARGS=val0,val1,...
       -I<jit_hw dir>  -I<include dir>
       -o kernel.so wrapper.cpp
   ```
4. `dlopen`s `kernel.so` and `dlsym`s `__emule_kernel_entry`.
5. Returns a `KernelFn` that calls the resolved entry point. A `shared_ptr` deleter keeps the
   handle alive for the lifetime of the kernel function object.

The `extern "C"` trampoline avoids C++ name mangling. The test executable must be linked with
`-rdynamic` so the `dlopen`'d `.so` can resolve the `extern thread_local __rt_args/__core/__device`
symbols from the main process.

**JIT header stubs** (`include/jit_hw/`) replace the real RISC-V hardware headers:

| JIT header | Provides |
|---|---|
| `jit_kernel_stubs.hpp` | Thread-local forward decls, `get_arg_addr(idx)`, `ASSERT` |
| `internal/risc_attribs.h` | `tt_l1_ptr` (no-op), `FORCE_INLINE` |
| `api/compile_time_args.h` | `get_compile_time_arg_val(N)` via `kernel_compile_time_args_arr[]` |
| `api/debug/dprint.h` | `DPRINT` (no-op sink), `ENDL()` |
| `api/dataflow/dataflow_api.h` | Includes stubs then `kernel_api/dataflow_api.hpp` |

---

## tt-metal Integration

The `tests/tt_emule/` subdirectory inside the tt-metal repository builds existing tt-metal test
sources against tt-emule with no modification to the test or kernel files.

**Build** (from tt-metal root, requires GCC ≥ 12):

```bash
cmake -B build_emule \
    -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTT_METAL_USE_TT_EMULE=ON \
    -DTT_EMULE_PATH=/path/to/tt-emule \
    -DWITH_PYTHON_BINDINGS=OFF \
    -DENABLE_TRACY=OFF \
    -DTT_INSTALL=OFF

cmake --build build_emule --target unit_tests_emule -j$(nproc)
ctest --test-dir build_emule -R AddTwoInts -V
```

**How it works**:

- Fake `tt-metalium/` and `tt-logger/` headers intercept Metal's includes and redirect them to
  `tt_emule/tt_metal_compat.hpp`.
- A `MeshDeviceSingleCardFixture` GTest fixture is re-implemented using tt-emule in
  `tests/tt_emule/tt_emule_fixture.hpp`.
- The real `tests/tt_metal/tt_metal/common/device_fixture.hpp` has a `#ifdef TT_EMULE_MODE`
  guard; the test target is built with `-DTT_EMULE_MODE=1`.
- The test executable links with `-rdynamic` so JIT-compiled kernel `.so` files can resolve
  thread-local symbols.

---

## Example: eltwise add

```cpp
#include "tt_emule/host_api.hpp"
#include "tt_emule/tile.hpp"

extern void reader_kernel_main();
extern void writer_kernel_main();
extern void compute_kernel_main();

using namespace tt_emule;

Device* dev = CreateDevice(0);

// Allocate DRAM buffers
auto src0 = CreateBuffer(*dev, 4 * Tile::SIZE_BYTES, Tile::SIZE_BYTES);
auto src1 = CreateBuffer(*dev, 4 * Tile::SIZE_BYTES, Tile::SIZE_BYTES);
auto dst  = CreateBuffer(*dev, 4 * Tile::SIZE_BYTES, Tile::SIZE_BYTES);

std::vector<float> a(4 * Tile::NUM_ELEMENTS, 1.0f);
std::vector<float> b(4 * Tile::NUM_ELEMENTS, 2.0f);
EnqueueWriteBuffer(*dev, *src0, a.data());
EnqueueWriteBuffer(*dev, *src1, b.data());

// Build program
Program prog = CreateProgram();
CreateCircularBuffer(prog, {0,0}, {0,  2, Tile::SIZE_BYTES}); // cb_in0
CreateCircularBuffer(prog, {0,0}, {1,  2, Tile::SIZE_BYTES}); // cb_in1
CreateCircularBuffer(prog, {0,0}, {16, 2, Tile::SIZE_BYTES}); // cb_out0

// NOC coords: (1,0) → DRAM
auto rid = CreateKernel(prog, reader_kernel_main,  {0,0}, DataMovementConfig{KernelType::DataMovement0});
auto wid = CreateKernel(prog, writer_kernel_main,  {0,0}, DataMovementConfig{KernelType::DataMovement1});
auto cid = CreateKernel(prog, compute_kernel_main, {0,0}, ComputeConfig{});

SetRuntimeArgs(prog, rid, {0,0}, {(uint32_t)src0->dram_offset(), (uint32_t)src1->dram_offset(),
                                  4, (uint32_t)Tile::SIZE_BYTES, 1, 0});
SetRuntimeArgs(prog, wid, {0,0}, {(uint32_t)dst->dram_offset(), 4, (uint32_t)Tile::SIZE_BYTES, 1, 0});
SetRuntimeArgs(prog, cid, {0,0}, {4});

EnqueueProgram(*dev, prog);
Finish(*dev);

std::vector<float> result(4 * Tile::NUM_ELEMENTS);
EnqueueReadBuffer(*dev, *dst, result.data());
// result[i] == 3.0f for all i

CloseDevice(dev);
```

---

## Known limitations and future work

- **Single core**: extend `Device` to hold a grid of `Core` objects with proper NOC routing.
- **JIT temp dir cleanup**: `jit_compile_kernel` never removes `/tmp/tt_emule_jit_*` directories.
- **DRAM allocator**: the bump allocator does not support `free`; replace with a proper arena when
  long-running tests need it.
- **Data types**: all tile math uses `float`; add bfloat16 support to `Tile` and `compute_api`.
- **Tilize/untilize**: currently identity copies; implement the real 32×32 row-major ↔ tile-major
  layout conversion when needed.
- **ThreadSanitizer false positives**: the intentional data-race pattern between the JIT DM threads
  (writing via `get_write_ptr`) and the compute thread (reading via `get_read_ptr`) is correctly
  serialized by CB `push_back`/`wait_front`, but TSan may flag it without proper `happens-before`
  annotation.
