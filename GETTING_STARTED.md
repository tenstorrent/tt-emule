# Getting Started with tt-emule

Build the emulator + tt-metal, run the C++ regression, and wire up your first test. For deeper detail see `BUILD_GUIDE.md`.

## 1. Prerequisites

| Tool | Version |
|------|---------|
| clang-20 | 20.x |
| CMake | ≥ 3.24 |
| Ninja | ≥ 1.10 |

Preferrably use an `mlir` IRD docker that includes all required dependencies.

## 2. Clone

`tt-emule` and `tt-metal` must be **siblings under the same parent directory** — `TT_EMULE_PATH` defaults to `../tt-emule` from the tt-metal source root. The parent directory itself is arbitrary.

```bash
export ROOT=$HOME/work          # any directory you like
mkdir -p "$ROOT" && cd "$ROOT"

git clone git@github.com:tenstorrent/tt-emule.git
git clone git@github.com:tenstorrent/tt-metal.git -b arminale/emule-metal-base
```

Init the two submodules tt-metal needs (UMD for cluster YAMLs, tracy for CMake):

```bash
cd "$ROOT/tt-metal"
git submodule update --init tt_metal/third_party/umd
git submodule update --init tt_metal/third_party/tracy
```

If GitHub HTTPS auth is unavailable, override the UMD submodule URL to SSH:
```bash
git config submodule.tt_metal/third_party/umd.url git@github.com:tenstorrent/tt-umd.git
```

## 3. Build

```bash
cd "$ROOT/tt-metal"
cmake -B build_emule -G Ninja \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_AR=/usr/bin/llvm-ar-20 -DCMAKE_RANLIB=/usr/bin/llvm-ranlib-20 \
  -DCMAKE_BUILD_TYPE=Release \
  -DTT_METAL_USE_TT_EMULE=ON -DTT_METAL_EMULATION=ON \
  -DTT_EMULE_PATH="$ROOT/tt-emule" \
  -DENABLE_TRACY=OFF -DTT_INSTALL=OFF
cmake --build build_emule -j$(nproc)
```

Produces `build_emule/test/tt_metal/unit_tests_*` gtest binaries that the regression script runs.

## 4. Run the Regression

```bash
cd "$ROOT/tt-emule"
TT_METAL_DIR="$ROOT/tt-metal" ./run_regression.sh 2>&1 | tee regression.log
```

Required env (set inside the script for each tier): `TT_METAL_EMULE_MODE=1`, `TT_METAL_SLOW_DISPATCH_MODE=1`, plus `TT_METAL_MOCK_CLUSTER_DESC_PATH` pointing at the right cluster YAML for the tier's device.

Cluster YAMLs live in `$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples/`:

| Device | YAML | Used by |
|--------|------|---------|
| Wormhole N150 | `wormhole_N150.yaml` | Default — Tier 2/3 buffer & JIT, Tier 3k/3l data movement, Tier 5/6 ttnn matmul + toggle |
| Quasar Q1 | `quasar_Q1.yaml` | Tier 3b–3j (DFB, Quasar compute, semaphores, atomics, multi-core matmul PCC) |
| Blackhole P100 | `blackhole_P100.yaml` | Tier 4 ttnn INT32 |

Tier 1 (host-only) unsets the cluster path entirely. See [Expected Regression Results](#expected-regression-results) below.

## 5. Add a New Test to the Regression

Tests live in tt-metal's existing gtest binaries — emulation runs them unchanged. To add one to the regression:

1. **Pick the binary** that contains your test. The existing consolidated gtest binaries are defined in tt-metal:

   | Binary | Defined in (tt-metal) | Covers |
   |--------|-----------------------|--------|
   | `unit_tests_api` | `tests/tt_metal/tt_metal/api/CMakeLists.txt` | core API, DFB, simple buffers, dst |
   | `unit_tests_integration` | `tests/tt_metal/tt_metal/integration/CMakeLists.txt` | matmul / SFPU integration |
   | `unit_tests_legacy` | `tests/tt_metal/tt_metal/CMakeLists.txt` | Quasar compute / DM |
   | `unit_tests_data_movement` | `tests/tt_metal/tt_metal/data_movement/CMakeLists.txt` | DM (loopback, NOC, DRAM unary) |
   | `unit_tests_ttnn` | `tests/ttnn/unit_tests/gtests/CMakeLists.txt` | ttnn ops |

   **Adding a source to an existing binary:** add the `.cpp` to the relevant `target_sources(unit_tests_X PRIVATE ...)` block in the CMakeLists.txt above and rebuild.

   **Adding a brand-new binary from tt-metal:** create an `add_executable(unit_tests_my_thing ...)` in the appropriate `tests/tt_metal/...` CMakeLists.txt, then expose it in `run_regression.sh` by adding a path variable next to the existing `*_BIN` declarations (around line 27) and a `run_test` line in the right tier.

2. **Verify it runs in emulation** by invoking the binary directly with the right env vars (pick the cluster YAML that matches your test's target device — `wormhole_N150.yaml`, `quasar_Q1.yaml`, or `blackhole_P100.yaml`):
   ```bash
   export CLUSTER_DESCS="$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
   export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
   export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESCS/wormhole_N150.yaml"
   "$ROOT/tt-metal/build_emule/test/tt_metal/unit_tests_api" --gtest_filter="MyNewTest.*"
   ```
   Failures usually mean a missing JIT stub (add it under `include/jit_hw/`) or an ungated hardware path (add an `is_mock_or_emulated()` guard in tt-metal).

3. **Add the entry** to `run_regression.sh` in the right tier:
   ```bash
   run_test "my_new_test" "$API_BIN" --gtest_filter="MyNewTest.*"
   ```
   Tier choice: host-only → Tier 1, buffer/JIT → Tier 2/3, Quasar-specific → Tier 3b–3j, ttnn → Tier 4/5. Match the env vars already exported in that tier block.

4. **Re-run** `./run_regression.sh` and confirm the new line is `PASS`.

## Expected Regression Results

Baseline against `arminale/emule-metal-base` @ `8711ac3d0b`: **128 passed, 11 failed, 0 skipped**. The 11 failures are 4 DFB STRIDED wraparound (Tier 3b) + 7 DFB Config Validation (Tier 3g) tests, all awaiting follow-up fixes that have not yet landed in upstream `main`.

See `IMPLEMENTATION_REPORT.md` § "Test Results" for the authoritative per-tier breakdown and the failing test names.

## Common Pitfalls

- **`unit_tests_api: binary not found`** — build is missing. Re-check `TT_METAL_USE_TT_EMULE=ON` and `TT_METAL_EMULATION=ON` were set when configuring.
- **Crash on JIT path** — likely an unguarded code path. Wrap with `is_mock_or_emulated()` in tt-metal (see `tt_metal/impl/...` for examples).
