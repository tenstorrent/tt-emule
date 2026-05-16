# tt-emule Build Guide

A complete, step-by-step guide to building tt-emule and all its dependencies from scratch on a new machine.

## Prerequisites

### Required Repositories

| Repo | Branch | Purpose |
|------|--------|---------|
| tt-emule | `main` | Software emulator library and regression scripts |
| tt-metal | `main` | Metal runtime with emulation support |
| tt-mlir | `main` | MLIR compiler, needed for D2M regression tests |

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
├── tt-metal/      # Metal runtime (arminale/emule-metal-base branch)
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
cd /localdev/<user>/tt-metal

cmake -B build_emule \
    -G Ninja \
    -DCMAKE_C_COMPILER=clang-20 \
    -DCMAKE_CXX_COMPILER=clang++-20 \
    -DCMAKE_AR=/usr/bin/llvm-ar-20 \
    -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH=/localdev/<user>/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=/localdev/<user>/tt-metal/build_emule

cmake --build build_emule -j$(nproc)
```

### CMake Options Explained

| Option | Value | Purpose |
|--------|-------|---------|
| `CMAKE_C/CXX_COMPILER` | `clang-20`/`clang++-20` | Must use clang-20 as specified by project conventions |
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
- `libtt-umd.so.0.71.0` in `build_emule/lib/` — must contain `SWEmuleChip` symbols. Verify with `nm -DC build_emule/lib/libtt-umd.so.0.71.0 | grep SWEmuleChip::`. If empty, the UMD subbuild was configured without `TT_UMD_BUILD_EMULE` (see the "Missing Include" issue above).

### Warning: don't use `build_emule_clang/` if it exists

A stale `build_emule_clang/` directory may exist alongside `build_emule/`. Both are clang builds; the suffix is misleading. The leftover is partial — missing `_ttnn.so`, ships a stripped-down `libtt_metal.so` — and produces spurious failures (most visibly `EMULE BUG: get_arg_val(N) out of bounds` in DFB tests). Always use `BUILD_DIR=$TT_METAL_DIR/build_emule`. The regression scripts default to that path.

### Known Build Issue: tracy Submodule Required Even with ENABLE_TRACY=OFF

`cmake/tracy.cmake` calls `add_subdirectory(${TRACY_HOME})` unconditionally. Even with `-DENABLE_TRACY=OFF`, the tracy submodule must be initialized or configure will fail with:

```
CMake Error at cmake/tracy.cmake:20 (add_subdirectory): The source directory .../tracy does not contain a CMakeLists.txt file.
```

(On commits prior to `8711ac3d0b`, a third submodule `tt_metal/third_party/tt_llk` was also required for firmware headers; on the current base it is part of the main tree at `tt_metal/tt-llk/`.)

---

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
cd /localdev/<user>/tt-emule
./run_regression.sh
```

Tiers covered: host-only (bit_utils, CoreRange, …), buffer I/O (Simple{L1,Dram}Buffer), JIT kernel (TensixL1Tile), DFB multi-P/C (Tiers 3b–3h), TTNN relational + matmul (Tier 4/5a), and TTNN reductions (Tier 5b).

**Expected result on `main`:** **127 passed, 16 failed, 0 skipped**. The 16 failures are 7 DFB Config Validation (Tier 3g) + 9 Tier 5b reduction (6 `ttnn_sum_*` + 3 `ttnn_minmax_*`) — all pre-existing, tracked separately. A "passed regression" means the failure set matches this allowlist.

Requires UMD fix [`arminale/emule-include-fix`](https://github.com/tenstorrent/tt-umd/tree/arminale/emule-include-fix) for the `SWEmuleChip` missing-include regression in UMD PR #2536.

---

## Phase 5: Build tt-mlir (for D2M Regression)

The D2M regression tests are Python tests from tt-mlir that exercise the emulated device through the MLIR compiler pipeline. Building tt-mlir requires several setup steps.

### ⚠️ Critical: Point tt-mlir at your tt-metal checkout (do this FIRST)

**Do not skip this step.** If tt-mlir is configured without `TTMLIR_TTMETAL_SOURCE_DIR`, its `third_party/CMakeLists.txt` calls `ExternalProject` to clone a **separate** tt-metal at the SHA pinned in `TT_METAL_VERSION`. That clone lives at `third_party/tt-metal/src/tt-metal/` as a real directory and is built with `ENABLE_TRACY=ON` into its own `build_Release/`. tt-mlir's runtime libraries get linked against *that* clone's headers and `libtt_metal.so`.

If you then point `LD_LIBRARY_PATH` at a different tt-metal build (the user-managed checkout under `/localdev/<user>/tt-metal`), you get a silent **ABI mismatch**: kernels JIT and run, but every D2M test returns `actual_pcc=0.0` because struct layouts and headers diverge between the two trees.

The fix is to make tt-mlir use *your* tt-metal source, not its private clone.

### Step 5a: Set TTMLIR_TTMETAL_SOURCE_DIR (preferred)

Configure tt-mlir with `-DTTMLIR_TTMETAL_SOURCE_DIR=/localdev/<user>/tt-metal`. tt-mlir's CMake will symlink `third_party/tt-metal/src/tt-metal` → your override and will *not* fetch a separate clone.

If a *real* directory already exists at `third_party/tt-metal/src/tt-metal/` (because a prior configure ran without the override), tt-mlir's CMake will hard-fail with:

```
TTMLIR_TTMETAL_SOURCE_DIR is set, but a real directory exists at
  <tt-mlir>/third_party/tt-metal/src/tt-metal
(likely a leftover clone from a prior non-override configure).
Remove it so cmake can replace it with a symlink to the override:
  rm -rf <tt-mlir>/third_party/tt-metal/src/tt-metal
```

You must remove (or rename) that bundled clone before re-configuring. The bundled clone is up to ~5 GB plus its own `build_Release` (~2 GB) — non-trivial disk. **Always** run `cmake -B build -DTTMLIR_TTMETAL_SOURCE_DIR=...` on the *first* tt-mlir configure so a bundled clone never appears in the first place.

### Step 5b: Make `build_Release` point at the emule build

tt-mlir reads from `<tt-metal>/build_${CMAKE_BUILD_TYPE}/` — for a `Release` configure that is `build_Release/`. Since our actual build directory is `build_emule`, symlink it:

```bash
cd /localdev/<user>/tt-metal
# If a real build_Release exists from a non-emule configure, move it aside first.
[ -e build_Release ] && [ ! -L build_Release ] && mv build_Release build_Release.bak
ln -sfn build_emule build_Release
```

### Step 5c: Create Library Symlinks

tt-mlir expects all shared libraries under `build_Release/lib/`. The emulation build places them in subdirectories. Create symlinks:

```bash
cd /localdev/<user>/tt-metal/build_emule/lib
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
    -DTTMLIR_TTMETAL_SOURCE_DIR=/localdev/<user>/tt-metal \
    -DCMAKE_BUILD_TYPE=Release \
    -DTTMLIR_ENABLE_RUNTIME=ON \
    -DTTMLIR_ENABLE_STABLEHLO=ON \
    -DTT_RUNTIME_ENABLE_PERF_TRACE=OFF \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DLLVM_USE_LINKER=lld-20

cmake --build build -j$(nproc)
```

**Important — `TTMLIR_TTMETAL_SOURCE_DIR`:** without this, tt-mlir fetches its own tt-metal clone and you get the silent PCC=0.0 disaster described above.

**Important — `TT_RUNTIME_ENABLE_PERF_TRACE=OFF`:** must be consistent with `ENABLE_TRACY` in the tt-metal build. If `ENABLE_TRACY` is OFF in your tt-metal build, `TT_RUNTIME_ENABLE_PERF_TRACE` must also be OFF — otherwise `libTTMLIRRuntime.so` will reference `tracy::GetToken` and friends, and conftest will fail at import time with:
```
ImportError: libTTMLIRRuntime.so: undefined symbol: _ZN5tracy8GetTokenEv
```
If your tt-metal build has `ENABLE_TRACY=ON`, setting `TT_RUNTIME_ENABLE_PERF_TRACE=ON` is also valid. The safest asymmetric-safe configuration is `TT_RUNTIME_ENABLE_PERF_TRACE=OFF` regardless of what tt-metal uses.

**Important:** `-DTTMLIR_ENABLE_STABLEHLO=ON` is required. The D2M test builder unconditionally imports `stablehlo` from `ttmlir.dialects`. Without it, all 13 test files fail with `ImportError`.

**Important:** `-DLLVM_USE_LINKER=lld-20` is required when using clang-20. Without it, GNU ld is used and fails on the `--color-diagnostics` flag.

### Step 5d-verify: Confirm tt-mlir is using your tt-metal

After configure, verify the canonical location really points at your checkout, not a private clone:

```bash
ls -la /localdev/<user>/tt-mlir/third_party/tt-metal/src/tt-metal
# Expect: tt-metal -> /localdev/<user>/tt-metal
```

If `ls -la` shows a real directory instead of a symlink, the override did not take effect (likely a stale cache or prior clone). Re-run Step 5a.

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
   cd /localdev/<user>/tt-metal
   cmake -B build_emule -DTT_METAL_USE_EMULE=ON -DTT_EMULE_PATH=/localdev/<user>/tt-emule
   cmake --build build_emule -j$(nproc)
   ```
   The flag goes back on, ninja regenerates, only the affected targets rebuild.

2. **Pre-touch the ExternalProject stamp files** so tt-mlir skips the configure step entirely (the approach PR #5 uses). Before the first `cmake --build` on the tt-mlir side:
   ```bash
   stamp_dir=/localdev/<user>/tt-mlir/third_party/tt-metal/src/tt-metal-stamp
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
cd /localdev/<user>/tt-mlir
source env/activate

export TT_METAL_MOCK_CLUSTER_DESC_PATH="/localdev/<user>/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_RUNTIME_ROOT="/localdev/<user>/tt-metal"

ttrt query --save-artifacts
```

**Important:** Do NOT set `LD_LIBRARY_PATH` when running `ttrt query`. Setting it to the emulation build's lib directory causes the wrong `_ttnncpp.so` to load (the emulation build has a newer API incompatible with `libTTMLIRRuntime.so`). The RPATH baked into `libTTMLIRRuntime.so` already points to the correct `_ttnncpp.so` in `build_Release/lib`.

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

Numbers below match `run_d2m_regression.sh`'s current 11-file `TEST_FILES` list against tt-metal + tt-mlir on `main`.

tt-emule tracks a per-test allowlist of expected failures in [issue #6](https://github.com/tenstorrent/tt-emule/issues/6) (broken down by op family in sub-issues #7–#17). A "passed regression" means the failure set matches the allowlist, not that there are zero failures. **The tests on this allowlist pass on real hardware; the failures are emulator-side, not compiler bugs.**

Per-file aggregate:

| File | passed | failed | xfailed | skipped |
|---|---:|---:|---:|---:|
| `d2m/test_matmul.py` | 0 | 123 | 14 | 0 |
| `d2m/test_tilize.py` | 0 | 44 | 0 | 0 |
| `d2m/test_dma.py` | 1 | 48 | 0 | 0 |
| `d2m/test_layout.py` | 81 | 13 | 0 | 0 |
| `d2m/test_allocate.py` | 0 | 6 | 0 | 0 |
| `d2m/test_masking.py` | 0 | 20 | 0 | 0 |
| `d2m/test_reductions.py` | 0 | 150 | 108 | 12 |
| `d2m/test_bfp8_typecast.py` | 0 | 13 | 0 | 0 |
| `d2m/test_tms.py` | 26 | 490 | 5 | 5 |
| `d2m/test_virtual_grid_rowmajor.py` | 0 | 0 | 0 | 27 |
| `d2m/test_virtual_grids.py` | 0 | 39 | 0 | 0 |

`test_virtual_grid_rowmajor.py` reports a file-level PASS because all 27 of its tests skip on N150 (requires N300 multi-chip).

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

### "lambda capture 'kidx' is not required"
Add `(void)kidx;` at the start of the lambda in `tt_metal/impl/emulation/emulated_program_runner.cpp`.

### Missing libraries in build_emule/lib/
Create symlinks as described in Step 5c. The emulation build places .so files in subdirectories rather than a flat `lib/` directory.

### ttrt query fails
Make sure `LD_LIBRARY_PATH` includes the emulation build lib directory, and all emulation environment variables (`TT_METAL_MOCK_CLUSTER_DESC_PATH`, `TT_METAL_EMULE_MODE`, `TT_METAL_SLOW_DISPATCH_MODE`) are set.

### `llvm-ar-17: not found` during tt-metal build
The CMake cache has stale references to clang-17 tools. Delete `build_emule/CMakeCache.txt` and reconfigure with `-DCMAKE_AR=/usr/bin/llvm-ar-20 -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20`.

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
| `LD_LIBRARY_PATH` | `<tt-metal>/build_emule/lib` | D2M regression tests only (NOT for `ttrt query`) |
