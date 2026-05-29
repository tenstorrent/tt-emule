# Getting Started with tt-emule

Quick path: clone, build, verify, then wire up a new ttnn test. Skip to `BUILD_GUIDE.md` if you hit a snag — that doc covers all build problems.

## 1. Prerequisites

| Tool | Version |
|------|---------|
| clang-20 | 20.x |
| libstdc++ (gcc-13+) | 13.x |
| CMake | ≥ 3.24 |
| Ninja | ≥ 1.10 |

Preferably use an `mlir` IRD docker that already has these.

## 2. Clone

`tt-emule` and `tt-metal` must be siblings (tt-metal's `TT_EMULE_PATH` defaults to `../tt-emule`).

```bash
export ROOT=$HOME/work
mkdir -p "$ROOT" && cd "$ROOT"

git clone git@github.com:tenstorrent/tt-emule.git
git clone git@github.com:tenstorrent/tt-metal.git

cd "$ROOT/tt-metal"
git checkout $(grep -vE '^\s*(#|$)' "$ROOT/tt-emule/tt-metal-pin.txt" | head -1)
git submodule update --init tt_metal/third_party/umd
git submodule update --init tt_metal/third_party/tracy
```

## 3. Build

One libstdc++ build covers C++ regression, ttnn pytest, and tt-mlir D2M:

```bash
cd "$ROOT/tt-metal"
cmake -S . -B build_emule -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTT_METAL_USE_EMULE=ON \
  -DTT_EMULE_PATH="$ROOT/tt-emule" \
  -DCMAKE_INSTALL_PREFIX="$ROOT/tt-metal/build_emule" \
  -DWITH_PYTHON_BINDINGS=ON \
  -DTT_METAL_BUILD_TESTS=ON \
  -DTTNN_BUILD_TESTS=ON \
  -DENABLE_TRACY=OFF \
  -DENABLE_DISTRIBUTED=ON
cmake --build build_emule -j$(nproc)
```

Then set up symlinks (one-time):

```bash
ln -sfn "$ROOT/tt-metal/build_emule/ttnn/_ttnn.so" \
       "$ROOT/tt-metal/ttnn/ttnn/_ttnn.so"
cd "$ROOT/tt-metal/build_emule/lib"
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
```

## 4. Verify

Run the per-arch regression scripts **sequentially** (they share a JIT cache):

```bash
cd "$ROOT/tt-emule"
export TT_METAL_DIR="$ROOT/tt-metal"

bash scripts/run_regression_wormhole.sh
bash scripts/run_regression_blackhole.sh
bash scripts/run_regression_quasar.sh
```

Baseline: wormhole **30/9**, blackhole **19/0**, quasar **105/9** — failures match `.github/known-failures-{arch}.txt`.

Quick sanity check that ttnn pytest works against the same build (no second toolchain needed):

```bash
cd "$ROOT/tt-metal"
export PYTHONPATH=$PWD/ttnn:$PWD/tools:${PYTHONPATH:-}
export TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH=$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml
/opt/ttmlir-toolchain/venv/bin/pytest tests/ttnn/unit_tests/base_functionality/test_reshape.py -v | tail -5
```

No `TT_METAL_USE_EMULE=ON` FATAL means the build is wired correctly.

## 5. Add a TTNN Test

TTNN tests live in `unit_tests_ttnn` (defined at `tt-metal/tests/ttnn/unit_tests/gtests/CMakeLists.txt`). Most run in the wormhole job; INT32 ops run in blackhole.

1. **Add the source** to `target_sources(unit_tests_ttnn PRIVATE ...)` in that CMakeLists.txt and rebuild:
   ```bash
   ninja -C "$ROOT/tt-metal/build_emule" unit_tests_ttnn
   ```

2. **Smoke-test in emulation** with the cluster YAML your test targets:
   ```bash
   CLUSTER_DESCS="$ROOT/tt-metal/tt_metal/third_party/umd/tests/cluster_descriptor_examples"
   export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
   export TT_METAL_RUNTIME_ROOT="$ROOT/tt-metal"
   export TT_METAL_MOCK_CLUSTER_DESC_PATH="$CLUSTER_DESCS/wormhole_N150.yaml"   # or blackhole_P100.yaml

   "$ROOT/tt-metal/build_emule/test/ttnn/unit_tests_ttnn" --gtest_filter="MyTest.*"
   ```
   Failures usually mean a missing JIT stub (add under `include/jit_hw/`) or an ungated hardware path (add `is_mock_or_emulated()` in tt-metal).

3. **Wire into the regression** by adding a `run_test` line to the right per-arch script:
   - Most ttnn ops on wormhole → `scripts/run_regression_wormhole.sh` (under Tier 5 or 5b)
   - ttnn INT32 on blackhole → `scripts/run_regression_blackhole.sh` (under Tier 4)

   Example line:
   ```bash
   run_test "my_ttnn_test" "$TTNN_BIN" --gtest_filter="MyTest.*"
   ```
   Place it inside an existing tier block so it inherits the right env vars.

4. **Re-run** the relevant per-arch script and confirm the new line is `PASS`.

## Common Pitfalls

- **`unit_tests_ttnn: binary not found`** — build is missing or `TT_METAL_USE_EMULE` was off. Verify with `nm -DC build_emule/tt_metal/libtt_metal.so | grep emule::execute_program_emulated` — a `T` line confirms it.
- **Crash on the JIT path** — usually an ungated hardware code path. Wrap with `is_mock_or_emulated()` in tt-metal.
