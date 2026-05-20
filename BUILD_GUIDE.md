# tt-emule Build Guide

A complete, step-by-step guide to building tt-emule and all its dependencies from scratch on a new machine.

## Prerequisites

### Required Repositories

| Repo | Revision | Purpose |
|------|--------|---------|
| tt-emule | `main` | Software emulator library and regression scripts |
| tt-metal | pinned SHA from `tt-metal-pin.txt` | Metal runtime with emulation support |
| tt-mlir | pinned SHA from `tt-mlir-pin.txt` | MLIR compiler, needed for D2M regression tests |

### Required Tools

| Tool | Minimum Version | Check Command |
|------|----------------|---------------|
| clang-20 | 20.x | `clang-20 --version` |
| libc++-20-dev | 20.x | `dpkg -l libc++-20-dev` |
| CMake | 3.24 | `cmake --version` |
| Ninja | 1.10+ | `ninja --version` |
| Python | 3.10+ | `python3 --version` |
| Git | 2.x | `git --version` |

Install libc++-20-dev if not present:
```bash
sudo apt-get install -y libc++-20-dev libc++abi-20-dev
```

### Required System Packages

The ttmlir-toolchain must be installed at `/opt/ttmlir-toolchain` (provides LLVM/MLIR libraries, Python venv with torch, pybind11, nanobind, etc.). This is typically pre-installed on Tenstorrent development machines.

## Directory Layout

All three repositories should be siblings under the same parent directory:

```
$ROOT/
├── tt-emule/      # This project
├── tt-metal/      # Metal runtime (arminale/emule-metal-base branch)
└── tt-mlir/       # MLIR compiler
```

This layout is important because tt-metal's `TT_EMULE_PATH` CMake variable defaults to `../tt-emule` relative to the tt-metal source root.

---

## Phase 0: Clone the Repositories

Pick a parent directory and export it as `ROOT` — all commands in this guide use this variable:

```bash
export ROOT=$HOME/work   # or any directory you prefer
mkdir -p "$ROOT"
```

Clone the three repos as siblings:

```bash
cd "$ROOT"
git clone git@github.com:tenstorrent/tt-emule.git
git clone git@github.com:tenstorrent/tt-metal.git
git clone git@github.com:tenstorrent/tt-mlir.git
```

Check out the pinned commits for reproducibility:

```bash
# tt-metal pinned SHA (see tt-emule/tt-metal-pin.txt)
cd "$ROOT/tt-metal"
git checkout d5a16537336229bee54fd4a6d8bd54c492abc7d1

# tt-mlir pinned SHA (see tt-emule/tt-mlir-pin.txt)
cd "$ROOT/tt-mlir"
git checkout cad4698a5156d844751c6aeb3b2660eb3df788d7
```

**Why pinned SHAs?** `tt-metal-pin.txt` and `tt-mlir-pin.txt` track the exact commits the regression CI uses. Using unpinned `main` risks hitting UMD or ABI issues not yet accounted for in the known-failure allowlist.

---

## Phase 1: Set Required Environment Variables

The regression scripts read `TT_METAL_DIR`, `TT_MLIR_DIR`, and `BUILD_DIR` from the environment. There are **no defaults to edit in the scripts** — the scripts will abort immediately (`set -euo pipefail`) if `TT_METAL_DIR` is unset.

Set these in your shell before any Phase 4 or Phase 6 command (add to `~/.bashrc` or `~/.zshrc` for persistence):

```bash
# ROOT must already be set from Phase 0 — or set it again here:
# export ROOT=$HOME/work
export TT_METAL_DIR="$ROOT/tt-metal"
export TT_MLIR_DIR="$ROOT/tt-mlir"
export BUILD_DIR="$TT_METAL_DIR/build_emule"   # IMPORTANT: not build_emule_clang
```

Explicitly setting `BUILD_DIR` is defensive but recommended: it overrides any stale shell state and ensures the right build directory is used throughout Phase 6.

---

## Phase 2: Initialize tt-metal Submodules

tt-metal has several git submodules that must be initialized before building. The critical one is UMD, which contains cluster descriptor YAML files used by the regression tests.

```bash
cd $ROOT/tt-metal
```

If your machine does not have HTTPS authentication configured for GitHub, override the submodule URLs to use SSH:

```bash
git config submodule.tt_metal/third_party/umd.url git@github.com:tenstorrent/tt-umd.git
git config submodule.tt_metal/third_party/tracy.url git@github.com:tenstorrent-metal/tracy.git
git config submodule."models/demos/t3000/llama2_70b/reference/llama".url git@github.com:tenstorrent-metal/llama.git
```

Initialize the two submodules required for the build (UMD for cluster descriptors, tracy for the CMake build system):

```bash
git submodule update --init tt_metal/third_party/umd
git submodule update --init tt_metal/third_party/tracy
```

**Note:** `--recursive` is not needed; the full recursive init also clones large/unrelated submodules. Only the two above are required for the emulation build. (Earlier commits required a third submodule `tt_metal/third_party/tt_llk` for firmware headers; that source has since been promoted into the main tree at `tt_metal/tt-llk/` and no longer needs separate initialization.)

Verify the critical file exists:
```bash
ls tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
```

---

## Phase 3: Build tt-metal with Emulation Support

This is the main build step. It compiles the full tt-metal library with emulation enabled, plus all emulation test binaries and Python bindings.

```bash
cd $ROOT/tt-metal

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
    -DENABLE_DISTRIBUTED=OFF \
    -DCMAKE_INSTALL_PREFIX=$ROOT/tt-metal/build_emule

cmake --build build_emule -j$(nproc)
```

### CMake Options Explained

| Option | Value | Purpose |
|--------|-------|---------|
| `CMAKE_TOOLCHAIN_FILE` | `cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake` | Selects clang-20 as compiler and sets `-stdlib=libc++`. **Use this for the C++ regression build.** The toolchain file also picks lld-20 automatically. Do NOT use this toolchain for the D2M tt-metal build (Phase 5a) — D2M requires libstdc++ so that tt-metal's shared libs use the same ABI as tt-mlir's runtime; use `x86_64-linux-clang-20-libstdcpp-toolchain.cmake` there instead. |
| `CMAKE_BUILD_TYPE` | `Release` (or `RelWithDebInfo`) | Either works for emule; Release if you don't need debug info |
| `TT_METAL_USE_EMULE` | `ON` | Compiles `emulated_program_runner.cpp`, defines `TT_METAL_USE_EMULE=1` in `tt_metal`, `impl`, and `llrt` libraries, and propagates `TT_UMD_BUILD_EMULE=ON` to the UMD subbuild. **This is the only correct flag.** Earlier versions of this guide listed `TT_METAL_USE_TT_EMULE` and `TT_METAL_EMULATION` — neither is a real option; both silently no-op. |
| `TT_EMULE_PATH` | Path to tt-emule | Points to your tt-emule source tree (CPM uses this instead of fetching from GitHub) |
| `WITH_PYTHON_BINDINGS` | `ON` | Builds `_ttnn.so` needed by D2M Python tests |
| `CMAKE_INSTALL_PREFIX` | `<this build dir>` | tt-mlir's `ExternalProject_Add(tt-metal)` triggers `cmake --build . --target install` on this build dir. With the default `/usr/local`, the CPM-fetched blake3 dep tries to copy headers there and the install step fails with `Permission denied`. Setting the prefix to the build dir keeps it local and harmless. |

(Note: `ENABLE_TRACY` is left at its default; either ON or OFF works as long as tt-mlir's `TT_RUNTIME_ENABLE_PERF_TRACE` is set consistently — see Phase 5.)

### Build Outputs

After a successful build, you should see:
- gtest binaries in `build_emule/test/tt_metal/unit_tests_*` (~19 binaries)
- `_ttnn.so` in `build_emule/ttnn/`
- `libtt_metal.so` in `build_emule/tt_metal/` — must contain `T tt::tt_metal::emule::execute_program_emulated`. Verify with `nm -DC build_emule/tt_metal/libtt_metal.so | grep emule::execute_program_emulated` — a `T` line proves `TT_METAL_USE_EMULE=ON` took effect.
- `libtt-umd.so` in `build_emule/tt_metal/third_party/umd/lib/` — must contain `SWEmuleChip` symbols. Verify with `nm -DC build_emule/tt_metal/third_party/umd/lib/libtt-umd.so | grep SWEmuleChip::`. If empty, the UMD subbuild was configured without `TT_UMD_BUILD_EMULE` (see the "Missing Include" issue above).

### Warning: don't use `build_emule_clang/` if it exists

A stale `build_emule_clang/` directory may exist alongside `build_emule/`. Both are clang builds; the suffix is misleading. The leftover is partial — missing `_ttnn.so`, ships a stripped-down `libtt_metal.so` — and produces spurious failures (most visibly `EMULE BUG: get_arg_val(N) out of bounds` in DFB tests). Always use `BUILD_DIR=$TT_METAL_DIR/build_emule`. The regression scripts default to that path.

### Known Build Issue: tracy Submodule Required Even with ENABLE_TRACY=OFF

`cmake/tracy.cmake` calls `add_subdirectory(${TRACY_HOME})` unconditionally. Even with `-DENABLE_TRACY=OFF`, the tracy submodule must be initialized or configure will fail with:

```
CMake Error at cmake/tracy.cmake:20 (add_subdirectory): The source directory .../tracy does not contain a CMakeLists.txt file.
```

(On commits prior to `8711ac3d0b`, a third submodule `tt_metal/third_party/tt_llk` was also required for firmware headers; on the current base it is part of the main tree at `tt_metal/tt-llk/`.)

---

### Known Build Issue: UMD `cluster.cpp` Missing `sw_emule_chip.hpp` Include

On UMD commits past `eb757f1e` ("Cleaning up includes with IWYU", PR #2536), `cluster.cpp` references `SWEmuleChip` inside `#ifdef TT_UMD_BUILD_EMULE` but is missing the corresponding `#include "umd/device/chip/sw_emule_chip.hpp"`. IWYU was run with the macro undefined and treated the include as unused. Symptom at build time:

```
cluster.cpp:191:33: error: use of undeclared identifier 'SWEmuleChip'
```

Use the `arminale/emule-include-fix` branch of `tt-umd` (or upstream `tt-umd#2645` once merged) to pick up the restored include:

```bash
cd tt_metal/third_party/umd
git fetch origin arminale/emule-include-fix
git checkout arminale/emule-include-fix
```

Then rebuild tt-metal.

---

## Phase 4: Run First Regression (C++ Tests)

```bash
cd $ROOT/tt-emule
# TT_METAL_DIR must be exported (set in Phase 1; repeated here for clarity)
export TT_METAL_DIR=$ROOT/tt-metal
./run_regression.sh 2>&1 | tee regression.log
```

Tiers covered: host-only (bit_utils, CoreRange, …), buffer I/O (Simple{L1,Dram}Buffer), JIT kernel (TensixL1Tile), DFB multi-P/C (Tiers 3b–3h), TTNN relational + matmul (Tier 4/5a), and TTNN reductions (Tier 5b).

**Expected result on `main`:** **126 passed, 17 failed, 0 skipped** (emulation-only machine, clang-20 build). The 17 failures are:
- 7 DFB Config Validation (Tier 3g): `DMTest1xDFB1Sx4SConfig`, `DMTensixTest1xDFB4Sx1SConfig`, `DMTest1xDFB4Sx1SConfig`, `DMTest1xDFB4Sx4SConfig`, `DMTest1xDFB2Sx4SConfig`, `DMTest1xDFB4Sx2SConfig`, `DMTest1xDFB1Sx1BConfig`
- 9 Tier 5b reduction (6 `ttnn_sum_*` + 3 `ttnn_minmax_*`)
- 1 Tier 6 silicon toggle (`ttnn_add_int_silicon`) — always fails on machines without real Tenstorrent hardware; the test unsets all emulation env vars and tries to open a real device

All failures are pre-existing and tracked separately. A "passed regression" means the failure set matches this allowlist.

**Load-sensitive tests:** `DmDirectWriteAddressPatterns`, `DramUnaryPacketSizes`, and `DramUnaryDRAMChannels` each pass in isolation but may fail when another heavy process (e.g. a D2M regression run) is contending for device resources concurrently. If the count reads 120–122 passed instead of 126, run those three tests in isolation to confirm they pass, then treat the suite run result as environment noise.

**Note on exit code:** `run_regression.sh` exits non-zero if any `run_test` call fails, including the 17 known failures. The exit code alone does not distinguish a real new regression from known-bad tests. Inspect the PASS/FAIL summary line and compare against the allowlist above.

Requires UMD fix [`arminale/emule-include-fix`](https://github.com/tenstorrent/tt-umd/tree/arminale/emule-include-fix) for the `SWEmuleChip` missing-include regression in UMD PR #2536.

---

## Phase 5: Build tt-mlir (for D2M Regression)

The D2M regression tests are Python tests from tt-mlir that exercise the emulated device through the MLIR compiler pipeline. Building tt-mlir requires several setup steps.

### ⚠️ Critical: Point tt-mlir at your tt-metal checkout (do this FIRST)

**Do not skip this step.** If tt-mlir is configured without `TTMLIR_TTMETAL_SOURCE_DIR`, its `third_party/CMakeLists.txt` calls `ExternalProject` to clone a **separate** tt-metal at the SHA pinned in `TT_METAL_VERSION`. That clone lives at `third_party/tt-metal/src/tt-metal/` as a real directory and is built with `ENABLE_TRACY=ON` into its own `build_Release/`. tt-mlir's runtime libraries get linked against *that* clone's headers and `libtt_metal.so`.

If you then point `LD_LIBRARY_PATH` at a different tt-metal build (the user-managed checkout under `$ROOT/tt-metal`), you get a silent **ABI mismatch**: kernels JIT and run, but every D2M test returns `actual_pcc=0.0` because struct layouts and headers diverge between the two trees.

The fix is to make tt-mlir use *your* tt-metal source, not its private clone.

### Step 5a: Set TTMLIR_TTMETAL_SOURCE_DIR (preferred)

Configure tt-mlir with `-DTTMLIR_TTMETAL_SOURCE_DIR=$ROOT/tt-metal`. tt-mlir's CMake will symlink `third_party/tt-metal/src/tt-metal` → your override and will *not* fetch a separate clone.

If a *real* directory already exists at `third_party/tt-metal/src/tt-metal/` (because a prior configure ran without the override), tt-mlir's CMake will hard-fail with:

```
TTMLIR_TTMETAL_SOURCE_DIR is set, but a real directory exists at
  <tt-mlir>/third_party/tt-metal/src/tt-metal
(likely a leftover clone from a prior non-override configure).
Remove it so cmake can replace it with a symlink to the override:
  rm -rf <tt-mlir>/third_party/tt-metal/src/tt-metal
```

You must remove (or rename) that bundled clone before re-configuring. The bundled clone is up to ~5 GB plus its own `build_Release` (~2 GB) — non-trivial disk. **Always** run `cmake -B build -DTTMLIR_TTMETAL_SOURCE_DIR=...` on the *first* tt-mlir configure so a bundled clone never appears in the first place.

### Step 5a-2: Build tt-metal for D2M (`build_Release`)

The D2M regression requires a **separate tt-metal build** using libstdc++ (not libc++). This is because tt-mlir's runtime (`libTTMLIRRuntime.so`) is compiled with libstdc++, and types like `ttnn::Tensor` cross the ABI boundary at runtime — both sides must agree on the stdlib layout.

```bash
cd $ROOT/tt-metal

cmake -S . -B build_Release -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH=$ROOT/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=ON \
    -DCMAKE_INSTALL_PREFIX=$ROOT/tt-metal/build_Release

cmake --build build_Release -j$(nproc)
```

`build_Release` uses `x86_64-linux-clang-20-libstdcpp-toolchain.cmake` (clang-20 + libstdc++, ENABLE_LIBCXX=OFF). No test binaries are needed — only the shared libs and Python bindings (`_ttnncpp.so`, `_ttnn.so`, `libtt_metal.so`, `libtt_stl.so`) are required for D2M.

> **Note:** `build_emule` and `build_Release` are two separate, independent directories. `build_emule` is used for C++ regression (Phase 4); `build_Release` is used for D2M (Phases 5–6). They must NOT be symlinked to each other.

### Step 5b: Create Library Symlinks in `build_Release`

### Step 5b: Create Library Symlinks in `build_Release`

tt-mlir expects all shared libraries under `build_Release/lib/`. The emulation build places them in subdirectories. Create symlinks:

```bash
cd $ROOT/tt-metal/build_Release/lib
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
```

### Step 5d: Configure and Build tt-mlir

```bash
cd $ROOT/tt-mlir
source env/activate

cmake -G Ninja -B build \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_C_FLAGS="-Wno-deprecated-declarations" \
    -DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
    -DTTMLIR_TTMETAL_SOURCE_DIR=$ROOT/tt-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_USE_LINKER=lld-20
```

**Why `-DCMAKE_C/CXX_COMPILER=clang-20` is required:** `env/activate` prepends `/opt/ttmlir-toolchain/bin` to `PATH`. If no compiler is specified, cmake picks up `clang` from the toolchain PATH — which is clang-17. Using clang-17 for tt-mlir while tt-metal uses clang-20 produces an ABI mismatch: clang-20 encodes C++20 `requires` constraints into mangled names (`_ZN4ttnn4fullIfQoosr3std...`), but clang-17 does not (`_ZN4ttnn4fullIfE...`). The linker cannot resolve `ttnn::full<float>` and the build fails with `undefined symbol`.

**Why `-Wno-deprecated-declarations`:** gcc-12's libstdc++ headers declare `std::get_temporary_buffer` with `_GLIBCXX17_DEPRECATED`. MLIR's `Scheduler.cpp` instantiates code that references it, and LLVM's build system sets `-Werror,-Wdeprecated-declarations`. This suppresses that error. (CI machines use `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13` to point at gcc-13 headers where the declaration was removed; the flag achieves the same effect without needing gcc-13.)

### Step 5d-pre: Block ExternalProject Before Building

**Do this before `cmake --build`.** tt-mlir wraps tt-metal in `ExternalProject_Add`. During the build step, CMake will re-run `cmake -B build_Release` on your tt-metal tree — without `-DTT_METAL_USE_EMULE=ON` — stripping the emulation code path and breaking all D2M tests with `actual_pcc=0.0`. Pre-touching the stamp files prevents this:

```bash
stamp_dir=$ROOT/tt-mlir/third_party/tt-metal/src/tt-metal-stamp
mkdir -p "$stamp_dir"
touch "$stamp_dir/tt-metal-configure" "$stamp_dir/tt-metal-build" "$stamp_dir/tt-metal-install"
```

Then build tt-mlir:

```bash
cmake --build build -j$(nproc)
```

**Important — `TTMLIR_TTMETAL_SOURCE_DIR`:** without this, tt-mlir fetches its own tt-metal clone and you get the silent PCC=0.0 disaster described above.

**Important — `TT_RUNTIME_ENABLE_PERF_TRACE=OFF`:** must be consistent with `ENABLE_TRACY` in the tt-metal build. If `ENABLE_TRACY` is OFF in your tt-metal build, `TT_RUNTIME_ENABLE_PERF_TRACE` must also be OFF — otherwise `libTTMLIRRuntime.so` will reference `tracy::GetToken` and friends, and conftest will fail at import time with:
```
ImportError: libTTMLIRRuntime.so: undefined symbol: _ZN5tracy8GetTokenEv
```
If your tt-metal build has `ENABLE_TRACY=ON`, setting `TT_RUNTIME_ENABLE_PERF_TRACE=ON` is also valid. The safest asymmetric-safe configuration is `TT_RUNTIME_ENABLE_PERF_TRACE=OFF` regardless of what tt-metal uses.

**Important:** `-DTTMLIR_ENABLE_STABLEHLO=ON` is required. The D2M test builder unconditionally imports `stablehlo` from `ttmlir.dialects`. Without it, all test files fail with `ImportError`.

**Important:** `-DLLVM_USE_LINKER=lld-20` is required. Without it, GNU ld is used and fails on the `--color-diagnostics` flag. Using lld-20 as the linker with clang-17 as the compiler is valid — the linker and compiler versions do not need to match.

### Step 5d-verify: Confirm tt-mlir is using your tt-metal

After configure, verify the canonical location really points at your checkout, not a private clone:

```bash
ls -la $ROOT/tt-mlir/third_party/tt-metal/src/tt-metal
# Expect: tt-metal -> $ROOT/tt-metal
```

If `ls -la` shows a real directory instead of a symlink, the override did not take effect (likely a stale cache or prior clone). Re-run Step 5a.

### Step 5d-verify-emule: Confirm TT_METAL_USE_EMULE Survived the Build

After `cmake --build` completes for tt-mlir, confirm the ExternalProject did not silently reconfigure your tt-metal build without `-DTT_METAL_USE_EMULE=ON`:

```bash
nm -DC $ROOT/tt-metal/build_Release/tt_metal/libtt_metal.so \
  | grep emule::execute_program_emulated
```

**Must show a `T` line.** If the output is empty, the ExternalProject flipped `TT_METAL_USE_EMULE` to OFF. Fix:

```bash
cd $ROOT/tt-metal
cmake -S . -B build_Release \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
    -DTT_METAL_USE_EMULE=ON -DTT_EMULE_PATH=$ROOT/tt-emule
cmake --build build_Release -j$(nproc)
```

Re-run the nm check before proceeding to Phase 5e.

### ⚠️ Env Var Pitfall: `TT_METAL_EMULE_MODE` is the correct name

The runtime (`tt_metal/llrt/rtoptions.cpp`) checks `TT_METAL_EMULE_MODE` — a single word "EMULE", not "EMULATED". Not setting this correctly leaves the runtime in `TargetDevice::Mock`: the device opens, the MLIR pipeline compiles, the runtime allocates output buffers, but **no kernel is ever JIT'd or executed**. The output buffer stays at its zero initial state, every PCC check sees a constant-zero output, and every D2M test fails with `actual_pcc=0.0`. There is no error message — the only signal is silence in the JIT logs and a ~0.5 s end-to-end test time (real emule runs take many seconds for kernel build alone).

Symptom-to-cause table:

| You see | Reason |
|---|---|
| `actual_pcc=0.0` on every test, ~0.5 s per test, no JIT log lines | `TT_METAL_EMULE_MODE` not set (or misspelled) |
| `TT_FATAL: TargetDevice::Emule requires building with TT_METAL_USE_EMULE=ON` | Env var is right; build was configured without `-DTT_METAL_USE_EMULE=ON`. Often happens after tt-mlir's `ExternalProject_Add(tt-metal)` silently reconfigures — see the next callout. |
| `ImportError: ... libtt_metal.so: undefined symbol: _ZTIN2tt3umd11SWEmuleChipE` | UMD subbuild didn't include `sw_emule_chip.cpp`. `TT_UMD_BUILD_EMULE` was off at compile time. See Phase 3 "Known Build Issue: UMD `cluster.cpp` Missing Include." |
| `ImportError: ... libTTMLIRRuntime.so: undefined symbol: _ZN5tracy8GetTokenEv` | tt-mlir's `TT_RUNTIME_ENABLE_PERF_TRACE` is ON but tt-metal's `ENABLE_TRACY` is OFF (or `libtracy.so` not in load path). Reconfigure tt-mlir with `-DTT_RUNTIME_ENABLE_PERF_TRACE=OFF` to match tt-metal — or build tt-metal with `ENABLE_TRACY=ON`. The two flags must be set consistently. |
| `ModuleNotFoundError: No module named 'tracy'` from `import ttnn._ttnn` | `<tt-metal>/tools/` is missing from `PYTHONPATH`. The `tracy` Python module lives in `tools/tracy/`. |

### ⚠️ tt-mlir's ExternalProject can flip your `TT_METAL_USE_EMULE` cache

tt-mlir's `third_party/CMakeLists.txt` wraps tt-metal in `ExternalProject_Add`. When tt-mlir builds, it runs `cmake -B <tt-metal>/build_Release ...` with its own `CMAKE_ARGS` list — that list **does not** include `-DTT_METAL_USE_EMULE=ON`. The reconfigure therefore resets `TT_METAL_USE_EMULE` to its option-default (`OFF`), recompiles `libtt_metal.so` and `libtt-umd.so` without the emule code path, and triggers the `TT_FATAL` above at runtime.

Two ways to defend:

1. **Re-assert the cache flag after each tt-mlir build:**
   ```bash
   cd $ROOT/tt-metal
   cmake -S . -B build_Release \
       -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
       -DTT_METAL_USE_EMULE=ON -DTT_EMULE_PATH=$ROOT/tt-emule
   cmake --build build_Release -j$(nproc)
   ```
   The flag goes back on, ninja regenerates, only the affected targets rebuild.

2. **Pre-touch the ExternalProject stamp files** so tt-mlir skips the configure step entirely (the approach PR #5 uses). Before the first `cmake --build` on the tt-mlir side:
   ```bash
   stamp_dir=$ROOT/tt-mlir/third_party/tt-metal/src/tt-metal-stamp
   mkdir -p "$stamp_dir"
   touch "$stamp_dir/tt-metal-configure" "$stamp_dir/tt-metal-build" "$stamp_dir/tt-metal-install"
   ```
   tt-mlir's ExternalProject sees the stamps and treats tt-metal as already built. **Only safe** if you have already built tt-metal yourself.

After either fix, re-verify with:
```bash
nm -DC <tt-metal>/build_emule/tt_metal/libtt_metal.so | grep emule::execute_program_emulated
```
Must show a `T` line.

### Step 5e: Generate System Descriptor

The D2M tests need a system descriptor file that describes the target hardware:

```bash
cd $ROOT/tt-mlir
source env/activate

export TT_METAL_MOCK_CLUSTER_DESC_PATH="$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$ROOT/tt-metal"

ttrt query --save-artifacts
```

**Important:** Do NOT set `LD_LIBRARY_PATH` when running `ttrt query`. Setting it to the emulation build's lib directory causes the wrong `_ttnncpp.so` to load (the emulation build has a newer API incompatible with `libTTMLIRRuntime.so`). The RPATH baked into `libTTMLIRRuntime.so` already points to the correct `_ttnncpp.so` in `build_Release/lib`.

This creates `ttrt-artifacts/system_desc.ttsys` in the tt-mlir directory.

Set the environment variable for test runs:
```bash
export SYSTEM_DESC_PATH="$ROOT/tt-mlir/ttrt-artifacts/system_desc.ttsys"
```

---

## Phase 6: Run D2M Regression (Python Tests)

```bash
cd $ROOT/tt-emule
export SYSTEM_DESC_PATH="$ROOT/tt-mlir/ttrt-artifacts/system_desc.ttsys"
export BUILD_DIR=$ROOT/tt-metal/build_emule   # override the wrong script default
./run_d2m_regression.sh --serial
```

The `--serial` flag runs test files one at a time (recommended to avoid memory contention).

### Expected Results

`run_d2m_regression.sh` auto-detects test files from `$TT_MLIR_DIR/test/python/golden/d2m/test_*.py`. Against the pinned tt-mlir SHA (`cad4698a5156d844751c6aeb3b2660eb3df788d7`) this picks up 33 test files.

tt-emule tracks a per-test allowlist of expected failures in [issue #6](https://github.com/tenstorrent/tt-emule/issues/6) (broken down by op family in sub-issues #7–#17). A "passed regression" means the failure set matches the allowlist, not that there are zero failures. **The tests on this allowlist pass on real hardware; the failures are emulator-side, not compiler bugs.**

**Notes on script PASS/FAIL accounting:**
- `test_allgather.py` and `test_fabric_apis.py`: all 32 tests deselected (N300 multi-chip hardware required). pytest exits 5 (no tests collected), which the script counts as FAIL. Expected on a single-chip N150 emulator.
- `test_fusion_with_optimizer.py` and `test_snippets.py`: all tests skip; pytest exits 0, script reports PASS.
- `test_virtual_grid_rowmajor.py`: all 27 tests skip on N150; pytest exits 0, script reports PASS.

Per-file aggregate (pinned SHA `cad4698a`, wormhole_N150 arch, clang-20 tt-metal):

| File | script result | passed | failed | xfailed | skipped | notes |
|---|---|---:|---:|---:|---:|---|
| `test_allgather.py` | FAIL | 0 | 0 | 0 | 0 | all 32 deselected (N300 multi-chip only) |
| `test_allocate.py` | PASS | 6 | 0 | 0 | 0 | |
| `test_bfp8_typecast.py` | FAIL | 0 | 13 | 0 | 0 | |
| `test_binary.py` | FAIL | 112 | 69 | 8 | 27 | |
| `test_binary_tree.py` | FAIL | 0 | 13 | 0 | 1 | |
| `test_composite_functions.py` | FAIL | 0 | 20 | 0 | 1 | |
| `test_composite_ops.py` | FAIL | 0 | 54 | 0 | 1 | |
| `test_constants.py` | PASS | 36 | 0 | 0 | 0 | |
| `test_dma.py` | PASS | 49 | 0 | 0 | 0 | |
| `test_dram_ops.py` | FAIL | 23 | 26 | 0 | 0 | |
| `test_eltwise_fusion.py` | FAIL | 8 | 56 | 0 | 0 | |
| `test_fabric_apis.py` | FAIL | 0 | 0 | 0 | 0 | all 32 deselected (N300 multi-chip only) |
| `test_fusion_with_optimizer.py` | PASS | 0 | 0 | 0 | 10 | all skipped |
| `test_generic.py` | PASS | 4 | 0 | 0 | 0 | |
| `test_layout.py` | PASS | 94 | 0 | 0 | 0 | |
| `test_masking.py` | FAIL | 5 | 15 | 0 | 0 | |
| `test_matmul.py` | PASS | 123 | 0 | 14 | 0 | |
| `test_rand.py` | FAIL | 0 | 31 | 0 | 0 | |
| `test_reduction_fusion.py` | FAIL | 0 | 2 | 0 | 0 | |
| `test_reductions.py` | FAIL | 94 | 56 | 108 | 12 | |
| `test_scalar_rt_arg.py` | PASS | 1 | 0 | 0 | 0 | |
| `test_shardy_ops.py` | PASS | 1 | 0 | 0 | 0 | |
| `test_snippets.py` | PASS | 0 | 0 | 0 | 1 | all skipped |
| `test_stablehlo_ops.py` | FAIL | 11 | 1 | 0 | 23 | |
| `test_ternary.py` | FAIL | 11 | 10 | 30 | 0 | |
| `test_tilize.py` | PASS | 44 | 0 | 0 | 0 | |
| `test_tms.py` | FAIL | 508 | 8 | 5 | 5 | |
| `test_ttir_ops.py` | FAIL | 7 | 2 | 0 | 0 | |
| `test_ttnn_mode.py` | FAIL | 0 | 1 | 0 | 0 | |
| `test_unaligned.py` | PASS | 49 | 0 | 0 | 1 | |
| `test_unary.py` | FAIL | 6 | 62 | 1 | 53 | |
| `test_virtual_grid_rowmajor.py` | PASS | 0 | 0 | 0 | 27 | all 27 skipped (N150 unsupported) |
| `test_virtual_grids.py` | FAIL | 0 | 39 | 0 | 0 | |

**Script summary: 13 passed, 20 failed, 0 hung** (over all 33 test files).

See `IMPLEMENTATION_REPORT.md` § "D2M Test Results" for the full per-file breakdown. The full D2M regression with `--serial` takes approximately 70 minutes for 33 test files.

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
The UMD submodule may require SSH access. Override the URL with `git config submodule.tt_metal/third_party/umd.url git@github.com:tenstorrent/tt-umd.git`.

### Missing libraries in build_Release/lib/
Create symlinks as described in Step 5b. The emulation build places .so files in subdirectories rather than a flat `lib/` directory.

### ttrt query fails
Ensure the three emulation env vars are set:
- `TT_METAL_MOCK_CLUSTER_DESC_PATH` pointing at `wormhole_N150.yaml`
- `TT_METAL_EMULE_MODE=1`
- `TT_METAL_SLOW_DISPATCH_MODE=1`

**Do NOT set `LD_LIBRARY_PATH` for `ttrt query`.** The RPATH baked into `libTTMLIRRuntime.so` already points to the correct `_ttnncpp.so`; adding `LD_LIBRARY_PATH` loads the emulation build's newer `_ttnncpp.so` API, which is ABI-incompatible with `libTTMLIRRuntime.so` and causes import errors.

### `llvm-ar-20: not found` during tt-metal build
Verify `llvm-ar-20` is installed: `ls /usr/bin/llvm-ar-20`. On Ubuntu, install via `sudo apt-get install llvm-20`. The `llvm-ar-20` and `llvm-ranlib-20` binaries are in the `llvm-20` package.

### `undefined symbol: ttnn::full<float>` during tt-mlir build
This is a compiler ABI mismatch: tt-mlir was configured without an explicit `-DCMAKE_CXX_COMPILER=clang-20`, so cmake picked up `clang-17` from the ttmlir-toolchain PATH (`env/activate` prepends `/opt/ttmlir-toolchain/bin`). clang-17 and clang-20 mangle C++20 `requires` constraints differently. Fix by always passing `-DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20` explicitly in the tt-mlir cmake configure command (see Step 5d). Both `build_Release` (tt-metal) and tt-mlir must use the same compiler.

### `ld: unrecognized option '--color-diagnostics'` during tt-mlir build
GNU ld doesn't support this LLD flag. Add `-DLLVM_USE_LINKER=lld-20` to the tt-mlir cmake configure command.

---

## Quick Reference: Environment Variables

| Variable | Value | When Needed |
|----------|-------|-------------|
| `TT_METAL_MOCK_CLUSTER_DESC_PATH` | `<tt-metal>/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml` | Running any emulated test |
| `TT_METAL_EMULE_MODE` | `1` | Running any emulated test — **not** `TT_METAL_EMULATED_MODE`, which is silently ignored |
| `TT_METAL_SLOW_DISPATCH_MODE` | `1` | Always (emulation uses slow dispatch) |
| `TT_METAL_RUNTIME_ROOT` | `<tt-metal>` | Running emulated tests |
| `PYTHONPATH` | must include `<tt-metal>/tools` | `ttnn` imports the `tracy` Python module from `<tt-metal>/tools/tracy/`. Omit and `import ttnn._ttnn` raises `ModuleNotFoundError: No module named 'tracy'` |
| `SYSTEM_DESC_PATH` | `<tt-mlir>/ttrt-artifacts/system_desc.ttsys` | D2M regression tests |
| `LD_LIBRARY_PATH` | `<tt-metal>/build_Release/lib` | D2M regression tests only (NOT for `ttrt query`) |
