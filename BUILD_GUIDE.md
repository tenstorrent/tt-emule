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
| libstdc++ (gcc-12+) | 12.x | `gcc --version` |
| CMake | 3.24 | `cmake --version` |
| Ninja | 1.10+ | `ninja --version` |
| Python | 3.10+ | `python3 --version` |
| Git | 2.x | `git --version` |

> **Ubuntu 22.04**: system default is gcc-11, whose libstdc++ doesn't implement enough of C++20 ranges for tt-metal (`std::ranges::for_each` over a `views::filter` in `levelized_graph.cpp` fails constraint checks). Install `gcc-13` and add `-DCMAKE_C_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" -DCMAKE_CXX_FLAGS="--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"` to the Phase 3 cmake. CI uses this flag.

### Required System Packages

The ttmlir-toolchain must be installed at `/opt/ttmlir-toolchain` (provides LLVM/MLIR libraries, Python venv with torch, pybind11, nanobind, etc.). This is typically pre-installed on Tenstorrent development machines.

## Directory Layout

All three repositories should be siblings under the same parent directory:

```
$ROOT/
├── tt-emule/
├── tt-metal/
└── tt-mlir/
```

tt-metal's `TT_EMULE_PATH` defaults to `../tt-emule` relative to the tt-metal source root.

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

Check out the pinned commits for reproducibility. Read the SHA from the pin files so this stays correct as the pins move:

```bash
cd "$ROOT/tt-metal"
git checkout $(grep -vE '^\s*(#|$)' "$ROOT/tt-emule/tt-metal-pin.txt" | head -1)

cd "$ROOT/tt-mlir"
git checkout $(grep -vE '^\s*(#|$)' "$ROOT/tt-emule/tt-mlir-pin.txt" | head -1)
```

Pinned SHAs match what CI uses; unpinned `main` risks ABI/UMD issues not yet in the known-failure allowlist.

---

## Phase 1: Set Required Environment Variables

```bash
export TT_METAL_DIR="$ROOT/tt-metal"
export TT_MLIR_DIR="$ROOT/tt-mlir"
export BUILD_DIR="$TT_METAL_DIR/build_emule"
```

---

## Phase 2: Initialize tt-metal Submodules

Two submodules are required: UMD (cluster descriptor YAMLs) and tracy (CMake build system). Do not use `--recursive` — it pulls in large unrelated submodules.

```bash
cd $ROOT/tt-metal
git submodule update --init tt_metal/third_party/umd
git submodule update --init tt_metal/third_party/tracy
```

If GitHub HTTPS auth is unavailable, override submodule URLs to SSH first:

```bash
git config submodule.tt_metal/third_party/umd.url   git@github.com:tenstorrent/tt-umd.git
git config submodule.tt_metal/third_party/tracy.url git@github.com:tenstorrent-metal/tracy.git
```

---

## Phase 3: Build tt-metal with Emulation Support

A single libstdc++ build produces everything needed for C++ gtest regression, stock ttnn pytest (`import ttnn`), and tt-mlir D2M integration:

```bash
cd $ROOT/tt-metal

cmake -S . -B build_emule -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH=$ROOT/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=ON \
    -DCMAKE_INSTALL_PREFIX=$ROOT/tt-metal/build_emule

cmake --build build_emule -j$(nproc)
```

### CMake Options Explained

| Option | Value | Purpose |
|--------|-------|---------|
| `CMAKE_TOOLCHAIN_FILE` | `cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake` | clang-20 as compiler with libstdc++ (gcc's stdlib). Matches tt-metal upstream's default (`build_metal.sh:89`) and the stdlib used by tt-mlir's runtime, so the same `libtt_metal.so` works for both C++ regression and Python tests. The toolchain file also picks lld-20 automatically. |
| `CMAKE_BUILD_TYPE` | `Release` (or `RelWithDebInfo`) | Either works for emule; Release if you don't need debug info |
| `TT_METAL_USE_EMULE` | `ON` | Compiles `emulated_program_runner.cpp`, defines `TT_METAL_USE_EMULE=1` in `tt_metal`, `impl`, and `llrt` libraries, and propagates `TT_UMD_BUILD_EMULE=ON` to the UMD subbuild. **This is the only correct flag.** Earlier versions of this guide listed `TT_METAL_USE_TT_EMULE` and `TT_METAL_EMULATION` — neither is a real option; both silently no-op. |
| `TT_EMULE_PATH` | Path to tt-emule | Points to your tt-emule source tree (CPM uses this instead of fetching from GitHub) |
| `WITH_PYTHON_BINDINGS` | `ON` | Builds `_ttnn.so` (pybind/nanobind binding); required for stock ttnn pytest and tt-mlir D2M |
| `TT_METAL_BUILD_TESTS` / `TTNN_BUILD_TESTS` | `ON` | Builds gtest binaries under `build_emule/test/{tt_metal,ttnn}/` for the C++ regression scripts |
| `ENABLE_DISTRIBUTED` | `ON` | Required for `_ttnncpp.so` (some D2M code paths reference it) |
| `CMAKE_INSTALL_PREFIX` | `<this build dir>` | tt-mlir's `ExternalProject_Add(tt-metal)` triggers `cmake --build . --target install` on this build dir. With the default `/usr/local`, the CPM-fetched blake3 dep tries to copy headers there and the install step fails with `Permission denied`. Setting the prefix to the build dir keeps it local and harmless. |

(`ENABLE_TRACY` is OFF here; tt-mlir's `TT_RUNTIME_ENABLE_PERF_TRACE` must be OFF too — see Phase 5.)

### Post-build symlinks (one-time, mandatory)

After the build completes, set up four symlinks so the standard `import ttnn` path and tt-mlir's RPATH resolution both find the libs:

```bash
# (a) for `import ttnn` from any tt-metal pytest
ln -sfn $ROOT/tt-metal/build_emule/ttnn/_ttnn.so \
       $ROOT/tt-metal/ttnn/ttnn/_ttnn.so

# (b) tt-mlir's runtime RPATH points at build_emule/lib/ — needed for D2M
cd $ROOT/tt-metal/build_emule/lib
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
```

Both subsets are cheap to set up unconditionally. (a) lets `pytest tests/ttnn/...` import the binding; (b) lets tt-mlir's `libTTMLIRRuntime.so` resolve dependencies at load time.

### Build Outputs

After a successful build, you should see:
- gtest binaries in `build_emule/test/{tt_metal,ttnn}/unit_tests_*` (for C++ regression)
- `_ttnn.so` and `_ttnncpp.so` in `build_emule/ttnn/` (for pytest and D2M)
- `libtt_metal.so` in `build_emule/tt_metal/` — must contain `T tt::tt_metal::emule::execute_program_emulated`. Verify with `nm -DC build_emule/tt_metal/libtt_metal.so | grep emule::execute_program_emulated` — a `T` line proves `TT_METAL_USE_EMULE=ON` took effect.
- `libtt-umd.so` in `build_emule/tt_metal/third_party/umd/lib/` — must contain `SWEmuleChip` symbols. Verify with `nm -DC build_emule/tt_metal/third_party/umd/lib/libtt-umd.so | grep SWEmuleChip::`. If empty, the UMD subbuild was configured without `TT_UMD_BUILD_EMULE`.

### Known Build Issue: tracy Submodule Required Even with ENABLE_TRACY=OFF

`cmake/tracy.cmake` calls `add_subdirectory(${TRACY_HOME})` unconditionally. Even with `-DENABLE_TRACY=OFF`, the tracy submodule must be initialized or configure will fail with:

```
CMake Error at cmake/tracy.cmake:20 (add_subdirectory): The source directory .../tracy does not contain a CMakeLists.txt file.
```

(On commits prior to `8711ac3d0b`, a third submodule `tt_metal/third_party/tt_llk` was also required for firmware headers; on the current base it is part of the main tree at `tt_metal/tt-llk/`.)

---

## Phase 4: Verify the Build (C++ Regression)

Three per-architecture scripts in `scripts/`, each self-contained. Run **sequentially** — they share `/tmp/tt_emule_jit_cache_$(id -u)` and concurrent runs corrupt each other's compiled kernels.

```bash
cd $ROOT/tt-emule
bash scripts/run_regression_wormhole.sh  2>&1 | tee regression-wormhole.log
bash scripts/run_regression_blackhole.sh 2>&1 | tee regression-blackhole.log
bash scripts/run_regression_quasar.sh    2>&1 | tee regression-quasar.log
```

Expected baseline:

| Job | Passed | Failed |
|-----|-------:|-------:|
| wormhole | 30 | 9 |
| blackhole | 19 | 0 |
| quasar | 105 | 9 |

A "passed regression" means failures match `.github/known-failures-{arch}.txt`; CI's `.github/scripts/classify-results.py` does the per-test cross-check. The per-arch scripts exit non-zero on any failure (including known), so the exit code alone does not distinguish a new regression — compare the PASS/FAIL summary against the allowlist.

---

## Phase 4b: Verify ttnn Pytest Smoke Test

Sanity-check that stock tt-metal Python tests work against this build (no second toolchain, no second build directory):

```bash
cd $ROOT/tt-metal
export PYTHONPATH=$PWD/ttnn:$PWD/tools:${PYTHONPATH:-}
export TT_METAL_HOME=$ROOT/tt-metal
export TT_METAL_RUNTIME_ROOT=$ROOT/tt-metal
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml

/opt/ttmlir-toolchain/venv/bin/pytest \
    tests/ttnn/unit_tests/base_functionality/test_reshape.py -v
```

Expect: the test collects and runs, the host-side reshape variants pass, and **no test fails with `TT_FATAL: TargetDevice::Emule requires building with TT_METAL_USE_EMULE=ON`**. Per-test failures with messages like `jit_compile_kernel: compiler failed` are unrelated kernel-emulation gaps and don't indicate a build problem.

---

## Phase 5: Build tt-mlir (for D2M Regression)

The D2M regression tests are Python tests from tt-mlir that exercise the emulated device through the MLIR compiler pipeline. tt-mlir links against the **same `build_emule` libs** built in Phase 3 — no second tt-metal build.

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

### Step 5d-verify-configure: Confirm tt-mlir is using your tt-metal

After configure (and before building), verify the canonical location really points at your checkout, not a private clone:

```bash
ls -la $ROOT/tt-mlir/third_party/tt-metal/src/tt-metal
# Expect: tt-metal -> $ROOT/tt-metal
```

If `ls -la` shows a real directory instead of a symlink, the override did not take effect (likely a stale cache or prior clone). Re-run Step 5a.

### Step 5d-pre: Block ExternalProject Before Building (MANDATORY)

**Do this before every `cmake --build` on tt-mlir.** tt-mlir wraps tt-metal in `ExternalProject_Add`. During the build step, CMake re-runs `cmake -B build_emule` on your tt-metal tree — without `-DTT_METAL_USE_EMULE=ON` — stripping the emulation code path. Because Phase 3 now produces a *single* `build_emule`, this reconfigure breaks **both** C++ regression and D2M (not just D2M as in the old two-build setup). Pre-touching the stamp files keeps tt-mlir away from `build_emule`:

```bash
stamp_dir=$ROOT/tt-mlir/third_party/tt-metal/src/tt-metal-stamp
mkdir -p "$stamp_dir"
touch "$stamp_dir/tt-metal-configure" "$stamp_dir/tt-metal-build" "$stamp_dir/tt-metal-install"
```

Then build tt-mlir:

```bash
cmake --build build -j$(nproc)
```

After the build, verify `TT_METAL_USE_EMULE` survived (one-liner):

```bash
nm -DC $ROOT/tt-metal/build_emule/tt_metal/libtt_metal.so | grep emule::execute_program_emulated
# Must show a T line. If empty, the stamps weren't pre-touched and tt-mlir's
# ExternalProject reconfigured build_emule without TT_METAL_USE_EMULE=ON.
# Recover by re-running the Phase 3 cmake configure + build.
```

**Important — `TTMLIR_TTMETAL_SOURCE_DIR`:** without this, tt-mlir fetches its own tt-metal clone and you get the silent PCC=0.0 disaster described above.

**Important — `TT_RUNTIME_ENABLE_PERF_TRACE=OFF`:** must be consistent with `ENABLE_TRACY` in the tt-metal build. If `ENABLE_TRACY` is OFF in your tt-metal build, `TT_RUNTIME_ENABLE_PERF_TRACE` must also be OFF — otherwise `libTTMLIRRuntime.so` will reference `tracy::GetToken` and friends, and conftest will fail at import time with:
```
ImportError: libTTMLIRRuntime.so: undefined symbol: _ZN5tracy8GetTokenEv
```
If your tt-metal build has `ENABLE_TRACY=ON`, setting `TT_RUNTIME_ENABLE_PERF_TRACE=ON` is also valid. The safest asymmetric-safe configuration is `TT_RUNTIME_ENABLE_PERF_TRACE=OFF` regardless of what tt-metal uses.

**Important:** `-DTTMLIR_ENABLE_STABLEHLO=ON` is required. The D2M test builder unconditionally imports `stablehlo` from `ttmlir.dialects`. Without it, all test files fail with `ImportError`.

**Important:** `-DLLVM_USE_LINKER=lld-20` is required. Without it, GNU ld is used and fails on the `--color-diagnostics` flag.

### Build-time symptom-to-cause table

| You see | Reason |
|---|---|
| `TT_FATAL: TargetDevice::Emule requires building with TT_METAL_USE_EMULE=ON` (at runtime, post-build) | Build was configured without `-DTT_METAL_USE_EMULE=ON`. Usually because tt-mlir's `ExternalProject_Add(tt-metal)` reconfigured `build_emule` — Step 5d-pre's stamp pre-touch prevents this. Recover by re-running the Phase 3 cmake configure + build. |
| `ImportError: ... libtt_metal.so: undefined symbol: _ZTIN2tt3umd11SWEmuleChipE` | UMD subbuild didn't include `sw_emule_chip.cpp` (i.e. `TT_UMD_BUILD_EMULE` was off). Re-verify `nm -DC build_emule/tt_metal/third_party/umd/lib/libtt-umd.so \| grep SWEmuleChip::` shows symbols. |
| `ImportError: ... libTTMLIRRuntime.so: undefined symbol: _ZN5tracy8GetTokenEv` | tt-mlir's `TT_RUNTIME_ENABLE_PERF_TRACE=ON` but tt-metal's `ENABLE_TRACY=OFF`. The two flags must be consistent. |

---

## Phase 6: Verify the Build (D2M Regression)

First generate the system descriptor (one-time, regenerate if the cluster YAML changes):

```bash
cd $ROOT/tt-mlir
source env/activate
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="$ROOT/tt-metal"
ttrt query --save-artifacts
```

Do **not** set `LD_LIBRARY_PATH` for `ttrt query` — `libTTMLIRRuntime.so`'s RPATH already points at the correct `_ttnncpp.so` in `build_emule/lib`; adding `LD_LIBRARY_PATH` loads a wrong copy and breaks the import.

Then run the regression:

```bash
cd $ROOT/tt-emule
export SYSTEM_DESC_PATH="$ROOT/tt-mlir/ttrt-artifacts/system_desc.ttsys"
./run_d2m_regression.sh --serial
```

`--serial` runs test files one at a time to avoid memory contention. Full run takes ~70 minutes for 33 test files. The expected failure set is tracked per-op-family in [issue #6](https://github.com/tenstorrent/tt-emule/issues/6) — a "passed regression" means matching that allowlist, not zero failures. See `IMPLEMENTATION_REPORT.md` § "D2M Test Results" for the per-file breakdown.

---

## Troubleshooting

### "cannot import name 'stablehlo' from 'ttmlir.dialects'"
Rebuild tt-mlir with `-DTTMLIR_ENABLE_STABLEHLO=ON`.

### UMD submodule clone fails
The UMD submodule may require SSH access. Override the URL with `git config submodule.tt_metal/third_party/umd.url git@github.com:tenstorrent/tt-umd.git`.

### Missing libraries in build_emule/lib/
Create symlinks as described in the "Post-build symlinks" subsection of Phase 3. The build places .so files in subdirectories rather than a flat `lib/` directory.

### `llvm-ar-20: not found` during tt-metal build
Verify `llvm-ar-20` is installed: `ls /usr/bin/llvm-ar-20`. On Ubuntu, install via `sudo apt-get install llvm-20`. The `llvm-ar-20` and `llvm-ranlib-20` binaries are in the `llvm-20` package.

### `undefined symbol: ttnn::full<float>` during tt-mlir build
This is a compiler ABI mismatch: tt-mlir was configured without an explicit `-DCMAKE_CXX_COMPILER=clang-20`, so cmake picked up `clang-17` from the ttmlir-toolchain PATH (`env/activate` prepends `/opt/ttmlir-toolchain/bin`). clang-17 and clang-20 mangle C++20 `requires` constraints differently. Fix by always passing `-DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20` explicitly in the tt-mlir cmake configure command (see Step 5d). Both `build_emule` (tt-metal) and tt-mlir must use the same compiler.

### `ld: unrecognized option '--color-diagnostics'` during tt-mlir build
GNU ld doesn't support this LLD flag. Add `-DLLVM_USE_LINKER=lld-20` to the tt-mlir cmake configure command.
