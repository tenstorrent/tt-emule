# Quasar Matmul Emulation Report

## 1. What tt-emule Does

tt-emule is a software emulator for Tenstorrent hardware. It allows Quasar (and Wormhole/Blackhole) device kernels to run on a host CPU without silicon. Kernels are JIT-compiled from their original source files using g++, loaded via `dlopen`, and executed in threads that model the hardware's processor topology.

For Quasar matmul bring-up, this means you can:
- Write and debug matmul kernels on any Linux machine
- Test DFB (Dataflow Buffer) synchronization between DM and compute threads
- Verify numerical correctness against golden reference
- Iterate rapidly without waiting for silicon access

---

## 2. How Quasar Matmul Works in Emulation

### Architecture: One Neo Core

A Quasar "Neo" core has:
- **8 Data Movement (DM) processors** — handle DRAM/L1/NoC transfers
- **4 Compute engines** — run UNPACK -> MATH -> PACK pipeline
- **4 MB shared L1** — all 12 processors share the same memory
- **Dataflow Buffers (DFBs)** — MPMC tile-based FIFOs replacing Wormhole's SPSC Circular Buffers

### Matmul Data Flow

```
 DRAM           DM0 (Reader)           Compute Engine 0           DM1 (Writer)           DRAM
  |                  |                       |                        |                    |
  |--read tiles----->|                       |                        |                    |
  |                  |--push to DFB src0---->|                        |                    |
  |                  |--push to DFB src1---->|                        |                    |
  |                  |                       |--matmul_block---------->                    |
  |                  |                       |--pack to DFB dst------>|                    |
  |                  |                       |                        |--write tiles------>|
```

### Key Source Files

| File | Role |
|------|------|
| `tests/tt_metal/tt_metal/integration/matmul/test_matmul_block.cpp` | **Demo test** — sets up DFBs, kernels, DRAM buffers, runs matmul, verifies result |
| `tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp` | **Compute kernel** — `#ifdef ARCH_QUASAR` uses DFB API (wait_front/pop_front/reserve_back/push_back) |
| `tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp` | **Reader DM kernel** — reads DRAM tiles into DFB src0/src1 |
| `tt_metal/kernels/dataflow/writer_unary.cpp` | **Writer DM kernel** — reads DFB dst, writes tiles to DRAM |
| `tt_metal/impl/emulation/emulated_program_runner.cpp` | **Program runner** — spawns threads for each DM/compute processor, manages JIT compilation |
| `tt-emule/src/dfb_sync.cpp` | **DFB synchronization** — tile counter logic (posted/acked atomics, condition variables) |

### Code Path: Host Test to Kernel Execution

1. **Test setup** (`test_matmul_block.cpp`):
   - `CreateDataflowBuffer()` — allocates DFB with producer/consumer risc masks and access pattern
   - `quasar::CreateKernel()` — registers kernel source with `QuasarDataMovementConfig` or `QuasarComputeConfig`
   - `BindDataflowBufferToProducerConsumerKernels()` — links DFBs to their producer/consumer kernels

2. **Program launch** (`emulated_program_runner.cpp`):
   - JIT-compiles each kernel source with g++ (patching CSR reads to TLS variables)
   - Spawns 1 thread per active DM processor + 1 thread per active compute engine
   - Sets thread-local state: `__emule_dfbs`, `__emule_tc_array`, `__processor_id`, `__emule_neo_id`

3. **Kernel execution** (runs in parallel threads):
   - **Reader** calls `dfb_in0.reserve_back(N)` / `noc.async_read()` / `dfb_in0.push_back(N)` to fill input DFBs
   - **Compute** calls `dfb0.wait_front(N)` (blocks until reader has pushed enough tiles), runs `matmul_block()`, then `dfb_out.push_back(N)` to signal writer
   - **Writer** calls `dfb_out.wait_front(N)` and writes tiles to DRAM

4. **Synchronization** (tile counters in `dfb_sync.cpp`):
   - Each DFB has per-consumer `posted`/`acked` atomic counters
   - `push_back` increments `posted`; `pop_front` increments `acked`
   - `wait_front` blocks on `data_cv` until `posted - acked >= requested_tiles`
   - `reserve_back` blocks on `space_cv` until `capacity - (posted - acked) >= requested_tiles`

---

## 3. What's Currently Tested

### Quasar Matmul Tests (DFB path)

| Test | Source File | DM Threads | Compute Threads | Dimensions | MathFidelity | fp32_dest_acc | DFB Pattern | Kernel |
|------|------------|-----------|----------------|------------|-------------|--------------|-------------|--------|
| TensixMatmulBlock | test_matmul_X_tile.cpp | 1 | 1 | M=K=N=2 (fp32) or 4 (bf16) | LoFi, HiFi2, HiFi4 | true, false | STRIDED 1P-1C | matmul_block.cpp |
| TensixMatmulBlockInitShort | test_matmul_X_tile.cpp | 1 | 1 | M=K=N=2 or 4 | LoFi, HiFi2, HiFi4 | true, false | STRIDED 1P-1C | matmul_block.cpp |
| QuasarMatmulBlock (new) | test_matmul_block.cpp | 1 | 1 | M=K=N=4 | HiFi4 | false | STRIDED 1P-1C | matmul_block.cpp |
| Bmm | test_bmm.cpp | 1 | 1 | Mt=4, Kt=2, Nt=3, B=2 | HiFi4 | false | STRIDED 1P-1C | bmm.cpp |

**Total Quasar matmul test cases:** ~50 (TensixMatmulBlock alone sweeps 3 fidelities x 2 fp32 modes x 2 sync modes = 12 combos, with 2 dimension sets each)

### DFB Infrastructure Tests (not matmul, but exercises the synchronization)

| Category | Test Count | What's Covered |
|----------|-----------|---------------|
| DM-DM STRIDED | 12 tests | 1P-1C, 1P-4C, 4P-1C, 4P-4C, 2P-4C, 4P-2C (explicit + implicit sync) |
| DM-Tensix STRIDED | 12 tests | Same P/C combos with DM producer, compute consumer |
| Tensix-DM STRIDED | 12 tests | Compute producer, DM consumer |
| DM-Tensix-DM pipeline | 3 tests | Multi-DFB pipeline through compute |
| BLOCKED consumer | 30 tests | DM-DM, DM-Tensix, Tensix-DM with BLOCKED access |
| Multi-core | 6 tests | 2-core DFB tests |
| Config validation | 10 tests | DFB configuration correctness |

### TTNN Matmul Sweep (Wormhole, not Quasar — CB path)

This demonstrates the emulator can run full TTNN-level matmul operations through the complete software stack (Python -> TTNN -> tt-metal -> emulated kernels), including automatic multi-core decomposition.

| M | K | N | Output Tiles | Notes |
|---|---|---|-------------|-------|
| 32 | 32 | 32 | 1 | Smallest case |
| 32 | 64 | 32 | 1 | Wide-K |
| 64 | 64 | 64 | 4 | |
| 128 | 128 | 128 | 16 | |
| 256 | 256 | 256 | 64 | |
| 512 | 512 | 512 | 256 | |
| 1024 | 1024 | 1024 | 1024 | |
| 2048 | 2048 | 2048 | 4096 | Largest case |
| 64 | 1024 | 64 | 4 | Tall K |
| 2048 | 32 | 2048 | 4096 | Small K, large M/N |
| 32 | 2048 | 32 | 1 | XL K |
| 128 | 512 | 256 | 32 | Rectangular |
| 320 | 384 | 320 | 120 | Non-power-of-2 |
| 512 | 1024 | 512 | 256 | |

- **Data type:** BFLOAT16
- **Layout:** TILE_LAYOUT
- **Multi-core:** Automatic (reported via `emulated_run_stats.num_cores`)
- **Verification:** `ttnn::allclose` with tolerance scaling by K

---

## 4. What's NOT Yet Tested (Gaps for Matmul Bring-up)

| Gap | Current State | Impact |
|-----|--------------|--------|
| **Multi-DM thread matmul** | All matmul tests use `num_threads_per_cluster=1` for DM | Real workloads pipeline across multiple DM threads |
| **Multi-compute thread matmul** | All matmul tests use `num_threads_per_cluster=1` for compute | Quasar has 4 compute engines per Neo; parallelizing across them is a key perf feature |
| **BLOCKED DFB access in matmul** | Only STRIDED tested in matmul (BLOCKED tested in DFB unit tests) | BLOCKED access is needed for some tiling strategies |
| **Multi-producer/consumer DFBs in matmul** | Only 1P-1C in matmul (MPMC tested in DFB unit tests) | Key Quasar feature for work distribution |
| **Non-square dimensions** | Only M=K=N tested (except BMM: Mt=4, Kt=2, Nt=3) | Practical matmuls are rectangular |
| **Multi-core matmul** | All Quasar matmul tests use single core {0,0} | Need to distribute work across Neo cores |
| **Multi-cluster** | Not tested | Production chips have many Neo cores |
| **matmul_large_block kernel** | No `#ifdef ARCH_QUASAR` in kernel source | Complex subblock matmul with tilize/untilize not ported to DFBs |
| **matmul_large_block_generalized** | No `#ifdef ARCH_QUASAR` in kernel source | 3D block loop matmul not ported |
| **Tilize/untilize in matmul** | Only matmul_block tested (no tilize) | matmul_large_block does inline tilize/untilize |
| **Bias addition on Quasar** | `with_bias` path skipped for Quasar in test_matmul_X_tile.cpp | Bias is common in real models |
| **init_short_with_dt on Quasar** | `TensixMatmulBlockInitShortWithDt` skipped | Data-type-switching init not ported |
| **Implicit sync DFBs in matmul** | `enable_implicit_sync=false` in all matmul DFB configs | Implicit sync simplifies kernel code |
| **TTNN matmul on Quasar** | TTNN sweep runs on Wormhole only | Need end-to-end TTNN->Quasar matmul path |

### Recommended Priority for Bring-up

1. **Non-square dimensions** — Easy to add by modifying test_matmul_block.cpp config
2. **Multi-compute threads** — Change `num_threads_per_cluster` in QuasarComputeConfig, requires kernel changes for work partitioning
3. **Multi-DM threads** — Change `num_threads_per_cluster` in QuasarDataMovementConfig
4. **matmul_large_block DFB port** — Add `#ifdef ARCH_QUASAR` paths to the kernel
5. **Multi-core** — Extend host test to dispatch across multiple cores
6. **TTNN Quasar path** — Requires TTNN layer to emit Quasar programs

---

## 5. Getting Started

### Clone and Build

```bash
# Check out repositories
cd /localdev/$USER
git clone git@github.com:tenstorrent/tt-metal.git -b arminale/quasar
git clone git@github.com:tenstorrent/tt-emule.git

# Initialize required submodules
cd tt-metal
git submodule update --init tt_metal/third_party/umd
git submodule update --init tt_metal/third_party/tracy
git submodule update --init tt_metal/third_party/tt_llk

# Build and test
cd ../tt-emule
./build_and_test.sh
```

### Run Just the Quasar Matmul Demo

```bash
cd /localdev/$USER/tt-metal

# Set up environment
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/quasar_Q1.yaml"
export ARCH_NAME=QUASAR
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$PWD"

# Fake simulator directory (required for Quasar soc descriptor)
SIM_DIR=$(mktemp -d /tmp/tt_emule_sim.XXXXXX)
ln -sf "$PWD/tt_metal/soc_descriptors/quasar_32_arch.yaml" "$SIM_DIR/soc_descriptor.yaml"
export TT_METAL_SIMULATOR="$SIM_DIR"

# Clear JIT cache and run
rm -rf /tmp/tt_emule_jit_cache_$(id -u)
./build_emule_clang/test/tt_emule/test_matmul_block

# Cleanup
rm -rf "$SIM_DIR"
```

### Modify the Test

Edit `tests/tt_metal/tt_metal/integration/matmul/test_matmul_block.cpp`:

- **Change dimensions:** Modify `M`, `K`, `N` in the `QuasarMatmulBlock` test
- **Change math fidelity:** Set `.math_fidelity = MathFidelity::LoFi` (or HiFi2)
- **Enable FP32 accumulation:** Set `.fp32_dest_acc_en = true` (halves max tile count due to DST capacity)
- **Add new test cases:** Copy the `TEST_F(MeshDispatchFixture, QuasarMatmulBlock)` block and change parameters

After editing, rebuild with:
```bash
cmake --build build_emule_clang --target test_matmul_block -j$(nproc)
```
