# tt-emule Agent Build Guide

Machine-optimized build instructions. All paths use `<ROOT>` as the parent directory containing tt-emule, tt-metal, and tt-mlir as siblings.

## Preconditions

- `clang-20`, `clang++-20`, `cmake` (>=3.24), `ninja`, `python3` (>=3.10) in PATH
- `/opt/ttmlir-toolchain` installed (LLVM/MLIR toolchain with venv)
- SSH access to github.com configured
- Repos checked out:
  - `<ROOT>/tt-emule` on `master`
  - `<ROOT>/tt-metal` on `xchin/tt-emule`
  - `<ROOT>/tt-mlir` on `main` or `arminale/tt-metal-local-build`

## Step 1: Update Regression Script Paths

Replace default paths in two files. The defaults must point to the local tt-metal and tt-mlir checkouts.

**Files to edit:**
- `<ROOT>/tt-emule/run_regression.sh` line 5: set `TT_METAL_DIR` default to `<ROOT>/tt-metal`
- `<ROOT>/tt-emule/run_d2m_regression.sh` line 5: set `TT_METAL_DIR` default to `<ROOT>/tt-metal`
- `<ROOT>/tt-emule/run_d2m_regression.sh` line 6: set `TT_MLIR_DIR` default to `<ROOT>/tt-mlir`

**Success:** `grep xchin <ROOT>/tt-emule/run_regression.sh <ROOT>/tt-emule/run_d2m_regression.sh` returns nothing.

## Step 2: Initialize tt-metal Submodules

```bash
cd <ROOT>/tt-metal
git config submodule.tt_metal/third_party/umd.url git@github.com:xanderchin/tt-umd.git
git config submodule.tt_metal/third_party/tracy.url git@github.com:tenstorrent-metal/tracy.git
git config submodule.tt_metal/third_party/tt_llk.url git@github.com:tenstorrent/tt-llk.git
git config submodule.models/demos/t3000/llama2_70b/reference/llama.url git@github.com:tenstorrent-metal/llama.git
git submodule update --init --recursive
```

**Success:** `ls tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml` exits 0.

## Step 3: Build tt-metal (Emulation Mode)

```bash
cd <ROOT>/tt-metal
cmake -B build_emule_clang -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_TT_EMULE=ON -DTT_METAL_EMULATION=ON \
    -DTT_EMULE_PATH=<ROOT>/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON -DENABLE_TRACY=OFF -DTT_INSTALL=OFF
cmake --build build_emule_clang -j$(nproc)
```

**Known fix:** If `emulated_program_runner.cpp` fails with `-Wunused-lambda-capture` on `kidx`, add `(void)kidx;` as the first line inside the lambda at the reported line number.

**Success:** `ls build_emule_clang/test/tt_emule/test_bit_utils build_emule_clang/ttnn/_ttnn.so` exits 0.

## Step 4: Run C++ Regression

```bash
cd <ROOT>/tt-emule
./run_regression.sh
```

**Success:** Output ends with `Results: 109 passed, 4 failed, 2 skipped` and exit code 1. The 2 skips are `test_emulation_toggle` (not yet in CMakeLists). Known failures: DFBEmuleDMTest, DFBEmuleBridgeTest (pre-existing host-side verification issue), DmLoopbackPacketSizes (intermittent), and ttnn_add_int_silicon (requires real hardware).

## Step 5: Set Up tt-mlir

### 5a: Symlink tt-metal source

```bash
cd <ROOT>/tt-mlir
mkdir -p third_party/tt-metal/src
ln -sfn <ROOT>/tt-metal third_party/tt-metal/src/tt-metal
```

### 5b: Symlink build directory

```bash
cd <ROOT>/tt-metal
ln -sfn build_emule_clang build_Release
```

### 5c: Create library symlinks

```bash
cd <ROOT>/tt-metal/build_emule_clang/lib
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
```

**Success:** `ls <ROOT>/tt-metal/build_emule_clang/lib/libtt_metal.so` exits 0.

### 5d: Build tt-mlir

```bash
cd <ROOT>/tt-mlir
source env/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_LOCAL_BUILD=ON -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_USE_LINKER=lld-20
cmake --build build -j$(nproc)
```

**Critical:** `TTMLIR_ENABLE_STABLEHLO=ON` is mandatory. Without it, all D2M tests fail with `ImportError: cannot import name 'stablehlo'`.

**Critical:** `-DLLVM_USE_LINKER=lld-20` is required. Without it, GNU ld is used and fails on `--color-diagnostics`.

**Success:** `ls build/python_packages/ttmlir/dialects/stablehlo.py build/runtime/python/_ttmlir_runtime*.so` exits 0.

## Step 6: Generate System Descriptor

```bash
cd <ROOT>/tt-mlir
source env/activate
export TT_METAL_MOCK_CLUSTER_DESC_PATH="<ROOT>/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULATED_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="<ROOT>/tt-metal"
export LD_LIBRARY_PATH="<ROOT>/tt-metal/build_emule_clang/lib:${LD_LIBRARY_PATH:-}"
ttrt query --save-artifacts
export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"
```

**Success:** `ls <ROOT>/tt-mlir/ttrt-artifacts/system_desc.ttsys` exits 0.

## Step 7: Run D2M Regression

```bash
cd <ROOT>/tt-emule
export SYSTEM_DESC_PATH="<ROOT>/tt-mlir/ttrt-artifacts/system_desc.ttsys"
./run_d2m_regression.sh --serial
```

**Success criteria:** Total individual test passes >= 1600. The script reports file-level pass/fail (7 passed, 6 failed is normal).

Expected per-file passes: matmul=113, matmul_higher_rank=10, layout=94, allocate=6, reductions=940, tms=332, dma=41, virtual_grids=39, tilize=12, tensor_collapsing=12, masking=5, bfp8_typecast=0(JIT errors), virtual_grid_rowmajor=0(skip/N300 only).

## Error Reference

| Error | Fix |
|-------|-----|
| UMD submodule clone fails (HTTPS auth) | Override URLs to SSH with `git config submodule.<path>.url git@github.com:<org>/<repo>.git` |
| `-Wunused-lambda-capture` on `kidx` | Add `(void)kidx;` as first statement in the lambda body |
| `ImportError: cannot import name 'stablehlo'` | Rebuild tt-mlir with `-DTTMLIR_ENABLE_STABLEHLO=ON` |
| ninja: `libtt_metal.so` missing | Create symlinks in `build_emule_clang/lib/` pointing to actual .so locations |
| ttrt query fails | Ensure `LD_LIBRARY_PATH` includes build lib dir and emulation env vars are set |
| `llvm-ar-17: not found` during link | Delete `CMakeCache.txt` and reconfigure with `-DCMAKE_AR=/usr/bin/llvm-ar-20 -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20` |
| `ld: unrecognized option '--color-diagnostics'` | Add `-DLLVM_USE_LINKER=lld-20` to tt-mlir cmake configure |
