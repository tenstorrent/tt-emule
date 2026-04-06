# tt-emule Build Guide

A complete, step-by-step guide to building tt-emule and all its dependencies from scratch on a new machine. This guide was written and verified on 2026-03-16.

## Prerequisites

### Required Repositories

| Repo | Branch | Purpose |
|------|--------|---------|
| tt-emule | `master` | Software emulator library and regression scripts |
| tt-metal | `xchin/tt-emule` | Metal runtime with emulation support |
| tt-mlir | `arminale/tt-metal-local-build` (or `main`) | MLIR compiler, needed for D2M regression tests |

### Required Tools

| Tool | Minimum Version | Check Command |
|------|----------------|---------------|
| clang-20 | 20.x | `clang-20 --version` |
| CMake | 3.24 | `cmake --version` |
| Ninja | 1.10+ | `ninja --version` |
| Python | 3.10+ | `python3 --version` |
| Git | 2.x | `git --version` |

### Required System Packages

The ttmlir-toolchain must be installed at `/opt/ttmlir-toolchain` (provides LLVM/MLIR libraries, Python venv with torch, pybind11, nanobind, etc.). This is typically pre-installed on Tenstorrent development machines.

## Directory Layout

All three repositories should be siblings under the same parent directory:

```
/localdev/<user>/
├── tt-emule/      # This project
├── tt-metal/      # Metal runtime (xchin/tt-emule branch)
└── tt-mlir/       # MLIR compiler
```

This layout is important because tt-metal's `TT_EMULE_PATH` CMake variable defaults to `../tt-emule` relative to the tt-metal source root.

---

## Phase 1: Update Default Paths

The regression scripts (`run_regression.sh` and `run_d2m_regression.sh`) contain default paths that must match your local directory layout. Edit the defaults at the top of each file:

**`run_regression.sh` line 5:**
```bash
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/<user>/tt-metal}"
```

**`run_d2m_regression.sh` lines 5-6:**
```bash
TT_METAL_DIR="${TT_METAL_DIR:-/localdev/<user>/tt-metal}"
TT_MLIR_DIR="${TT_MLIR_DIR:-/localdev/<user>/tt-mlir}"
```

Alternatively, you can set these as environment variables instead of editing the files.

---

## Phase 2: Initialize tt-metal Submodules

tt-metal has several git submodules that must be initialized before building. The critical one is UMD, which contains cluster descriptor YAML files used by the regression tests.

```bash
cd /localdev/<user>/tt-metal
```

**Important:** The UMD submodule points to a fork (`xanderchin/tt-umd.git`) via HTTPS. If your machine does not have HTTPS authentication configured for GitHub, you need to override the submodule URLs to use SSH:

```bash
git config submodule.tt_metal/third_party/umd.url git@github.com:xanderchin/tt-umd.git
git config submodule.tt_metal/third_party/tracy.url git@github.com:tenstorrent-metal/tracy.git
git config submodule.tt_metal/third_party/tt_llk.url git@github.com:tenstorrent/tt-llk.git
git config submodule.models/demos/t3000/llama2_70b/reference/llama.url git@github.com:tenstorrent-metal/llama.git
```

Then initialize:

```bash
git submodule update --init --recursive
```

Verify the critical file exists:
```bash
ls tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
```

---

## Phase 3: Build tt-metal with Emulation Support

This is the main build step. It compiles the full tt-metal library with emulation enabled, plus all emulation test binaries and Python bindings.

```bash
cd /localdev/<user>/tt-metal

cmake -B build_emule_clang \
    -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_TT_EMULE=ON \
    -DTT_METAL_EMULATION=ON \
    -DTT_EMULE_PATH=/localdev/<user>/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DENABLE_TRACY=OFF \
    -DTT_INSTALL=OFF

cmake --build build_emule_clang -j$(nproc)
```

### CMake Options Explained

| Option | Value | Purpose |
|--------|-------|---------|
| `CMAKE_C/CXX_COMPILER` | `clang-20`/`clang++-20` | Must use clang-20 as specified by project conventions |
| `CMAKE_BUILD_TYPE` | `Release` | Release mode avoids ThreadSanitizer overhead and unused-variable warnings in debug macros |
| `TT_METAL_USE_TT_EMULE` | `ON` | Adds `tt_emule/` test subdirectory to the build |
| `TT_METAL_EMULATION` | `ON` | Compiles `emulated_program_runner.cpp`, defines `TT_METAL_EMULATION=1`, adds tt-emule include paths |
| `WITH_PYTHON_BINDINGS` | `ON` | Builds `_ttnn.so` needed by D2M Python tests |
| `TT_EMULE_PATH` | Path to tt-emule | Points to tt-emule source tree |
| `ENABLE_TRACY` | `OFF` | Profiling not needed for emulation |
| `TT_INSTALL` | `OFF` | Skip install rules |

### Build Outputs

After a successful build, you should see:
- Test binaries in `build_emule_clang/test/tt_emule/` (22 executables)
- `_ttnn.so` in `build_emule_clang/ttnn/`
- `libtt_metal.so` in `build_emule_clang/tt_metal/`

### Known Build Issue: Unused Lambda Capture

In Release mode, `log_debug` macros compile to nothing, which can cause `-Wunused-lambda-capture` errors under clang-20 with `-Werror`. If you see an error like:

```
emulated_program_runner.cpp:788: error: lambda capture 'kidx' is not required
```

Fix it by adding `(void)kidx;` at the start of the lambda body:

```cpp
[&ki, core, l1_data, dram_data, cb_array, core_map_ptr, px, py, lx, ly, kidx]() {
    (void)kidx;
    __rt_args = ki.rt_args;
    ...
```

---

## Phase 4: Run First Regression (C++ Tests)

```bash
cd /localdev/<user>/tt-emule
./run_regression.sh
```

This runs 5 tiers of tests:
1. **Tier 1 (Host-only):** bit_utils, host_buffer, tilize_untilize, blockfloat_common, CoreRange/CoreRangeSet tests
2. **Tier 2 (Buffer I/O):** SimpleL1Buffer, SimpleDramBuffer
3. **Tier 3 (JIT Kernel):** TensixL1Tile
4. **Tier 4 (TTNN Relational):** ttnn_relational
5. **Tier 5 (TTNN Matmul):** ttnn_matmul_sweep

**Expected result:** 23 passed, 1 failed, 2 skipped. The 1 failure is Tier 6 (silicon toggle proof, requires real hardware). The 2 skips are `test_emulation_toggle` (not yet in CMakeLists). All emulation tests pass.

---

## Phase 5: Build tt-mlir (for D2M Regression)

The D2M regression tests are Python tests from tt-mlir that exercise the emulated device through the MLIR compiler pipeline. Building tt-mlir requires several setup steps.

### Step 5a: Symlink tt-metal into tt-mlir

tt-mlir expects tt-metal source at `third_party/tt-metal/src/tt-metal/`:

```bash
cd /localdev/<user>/tt-mlir
mkdir -p third_party/tt-metal/src
ln -sfn /localdev/<user>/tt-metal third_party/tt-metal/src/tt-metal
```

### Step 5b: Create Build Directory Symlink

tt-mlir looks for tt-metal's build at `build_${CMAKE_BUILD_TYPE}` inside the tt-metal source tree. Since our build is named `build_emule_clang` but we configure tt-mlir with `Release`:

```bash
cd /localdev/<user>/tt-metal
ln -sfn build_emule_clang build_Release
```

### Step 5c: Create Library Symlinks

tt-mlir expects all shared libraries under `build_Release/lib/`. The emulation build places them in subdirectories. Create symlinks:

```bash
cd /localdev/<user>/tt-metal/build_emule_clang/lib
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
```

### Step 5d: Configure and Build tt-mlir

```bash
cd /localdev/<user>/tt-mlir
source env/activate

cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_LOCAL_BUILD=ON \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_USE_LINKER=lld-20

cmake --build build -j$(nproc)
```

**Important:** `TTMLIR_ENABLE_STABLEHLO=ON` is required. The D2M test builder unconditionally imports `stablehlo` from `ttmlir.dialects`. Without it, all 13 test files fail with `ImportError`.

**Important:** `-DLLVM_USE_LINKER=lld-20` is required when using clang-20. Without it, GNU ld is used and fails on the `--color-diagnostics` flag.

### Step 5e: Generate System Descriptor

The D2M tests need a system descriptor file that describes the target hardware:

```bash
cd /localdev/<user>/tt-mlir
source env/activate

export TT_METAL_MOCK_CLUSTER_DESC_PATH="/localdev/<user>/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="/localdev/<user>/tt-metal"
export LD_LIBRARY_PATH="/localdev/<user>/tt-metal/build_emule_clang/lib:${LD_LIBRARY_PATH:-}"

ttrt query --save-artifacts
```

This creates `ttrt-artifacts/system_desc.ttsys` in the tt-mlir directory.

Set the environment variable for test runs:
```bash
export SYSTEM_DESC_PATH="/localdev/<user>/tt-mlir/ttrt-artifacts/system_desc.ttsys"
```

---

## Phase 6: Run D2M Regression (Python Tests)

```bash
cd /localdev/<user>/tt-emule
export SYSTEM_DESC_PATH="/localdev/<user>/tt-mlir/ttrt-artifacts/system_desc.ttsys"
./run_d2m_regression.sh --serial
```

The `--serial` flag runs test files one at a time (recommended to avoid memory contention).

### Expected Results

Total: **1604+ individual test passes** across 13 test files. The script reports 7 file-level passes, 6 file-level failures (normal — some files have a mix of passing and failing tests).

| Test File | Expected Passed | Expected Failed | Notes |
|-----------|----------------|----------------|-------|
| test_metal_matmul | 113 | 0 (14 xfailed) | 100% pass with 1800s timeout |
| test_metal_matmul_higher_rank | 10 | 0 | 100% pass |
| test_metal_layout | 94 | 0 | 100% pass |
| test_metal_allocate | 6 | 0 | 100% pass |
| test_metal_reductions | 940 | 60 | 96 skipped; some sum variants fail PCC |
| test_metal_tms | 332 | 4 | 3 skipped; most reshapes/permutes pass |
| test_metal_dma | 41 | 8 | DRAM-backed interleaved tests crash (signal 6) |
| test_metal_virtual_grids | 39 | 0 | 100% pass |
| test_metal_tensor_collapsing | 12 | 0 | 2 skipped; 100% pass |
| test_metal_tilize | 12 | 32 | Non-aligned shapes fail PCC |
| test_metal_masking | 5 | 15 | PCC mismatch on most tests |
| test_metal_bfp8_typecast | 0 | 13 | JIT compile errors |
| test_metal_virtual_grid_rowmajor | 0 (27 skip) | 0 | Requires N300 multi-chip |

### Increasing Timeout

If you want more matmul/reduction/TMS tests to complete, increase the timeout:

```bash
TIMEOUT=600 ./run_d2m_regression.sh --serial
```

---

## Troubleshooting

### "cannot import name 'stablehlo' from 'ttmlir.dialects'"
Rebuild tt-mlir with `-DTTMLIR_ENABLE_STABLEHLO=ON`.

### UMD submodule clone fails
The UMD submodule (`xanderchin/tt-umd`) may require SSH access. Override the URL with `git config submodule.tt_metal/third_party/umd.url git@github.com:xanderchin/tt-umd.git`.

### "lambda capture 'kidx' is not required"
Add `(void)kidx;` at the start of the lambda in `tt_metal/impl/emulation/emulated_program_runner.cpp`.

### Missing libraries in build_emule_clang/lib/
Create symlinks as described in Step 5c. The emulation build places .so files in subdirectories rather than a flat `lib/` directory.

### ttrt query fails
Make sure `LD_LIBRARY_PATH` includes the emulation build lib directory, and all emulation environment variables (`TT_METAL_MOCK_CLUSTER_DESC_PATH`, `TT_METAL_EMULATED_MODE`, `TT_METAL_SLOW_DISPATCH_MODE`) are set.

### `llvm-ar-17: not found` during tt-metal build
The CMake cache has stale references to clang-17 tools. Delete `build_emule_clang/CMakeCache.txt` and reconfigure with `-DCMAKE_AR=/usr/bin/llvm-ar-20 -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20`.

### `ld: unrecognized option '--color-diagnostics'` during tt-mlir build
GNU ld doesn't support this LLD flag. Add `-DLLVM_USE_LINKER=lld-20` to the tt-mlir cmake configure command.

---

## Quick Reference: Environment Variables

| Variable | Value | When Needed |
|----------|-------|-------------|
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | `<tt-metal>/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml` | Running any emulated test |
| `TT_METAL_EMULATED_MODE` | `1` | Running any emulated test |
| `TT_METAL_SLOW_DISPATCH_MODE` | `1` | Always (emulation uses slow dispatch) |
| `TT_METAL_RUNTIME_ROOT` | `<tt-metal>` | Running emulated tests |
| `SYSTEM_DESC_PATH` | `<tt-mlir>/ttrt-artifacts/system_desc.ttsys` | D2M regression tests |
| `LD_LIBRARY_PATH` | `<tt-metal>/build_emule_clang/lib` | When running ttrt or D2M tests |
