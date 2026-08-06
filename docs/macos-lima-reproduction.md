# Reproducing tt-emule on an Apple-silicon Mac

This is the known-good workflow used to validate tt-emule against current
tt-metal `main` without silicon. The Mac hosts an x86_64 Ubuntu VM through Lima;
the build and tests run inside that VM.

For native x86_64 Linux and the full regression suite, use
[BUILD_GUIDE.md](../BUILD_GUIDE.md). This page records the smaller, focused
workflow that is practical for another local agent to repeat.

## Banked revisions

The compatibility fixes are stored on remote branches in both repositories:

| Repository | Branch | Fix commit |
|---|---|---|
| `tenstorrent/tt-emule` | `nkapre/tt-emule` | `86afe51360289ae08bc649211d35da548b2078a9` |
| `tenstorrent/tt-metal` | `nkapre/emule` | `fb22091fb78e7c9adfa4e7a0e289e115e5818d29` |

The tt-metal fix commit is based on main commit
`587a4f30937e8bd5eea684434ef985d32486fb55`.

Use the branch names for normal reproduction. The hashes above identify the
tested code even if more documentation is later added to a branch.

## Reuse the known-good local VM

On the development Mac, the existing Lima instance is named `ttmetal-x86`:

```bash
limactl list ttmetal-x86
limactl start ttmetal-x86       # only if it is stopped
limactl shell ttmetal-x86
```

Its relevant configuration is:

| Setting | Value |
|---|---|
| Guest | Ubuntu 24.04 x86_64 |
| VM backend | QEMU |
| CPUs | 6 |
| Memory | 12 GiB |
| Disk | 80 GiB |
| Compiler | clang 20 with gcc-13 libstdc++ |
| Build system | CMake + Ninja |

The working build tree on that VM is:

```text
/home/nkapre.guest/workspace/tt-emule-latest/
├── tt-metal/
└── build_emule/
```

The build was configured with `TT_EMULE_PATH` pointing at the host tt-emule
checkout mounted at `/Users/nkapre/workspace/tt-emule`.

## Create an equivalent VM from scratch

Install Lima and QEMU on the Mac, then create an x86_64 Ubuntu 24.04 instance:

```bash
brew install lima qemu
limactl create --name=ttmetal-x86 \
  --arch=x86_64 \
  --vm-type=qemu \
  --cpus=6 \
  --memory=12 \
  --disk=80 \
  template:ubuntu-24.04
limactl start ttmetal-x86
limactl shell ttmetal-x86
```

Inside the guest, install the prerequisites from
[BUILD_GUIDE.md](../BUILD_GUIDE.md#required-tools). The known-good toolchain
versions were clang 20.1.8, gcc 13.3, CMake 4.3.2, and Ninja 1.11.1.

Clone the two tested branches as siblings:

```bash
export EMULE_ROOT="$HOME/workspace/tt-emule-latest"
mkdir -p "$EMULE_ROOT"
cd "$EMULE_ROOT"

git clone --branch nkapre/tt-emule \
  https://github.com/tenstorrent/tt-emule.git
git clone --branch nkapre/emule \
  https://github.com/tenstorrent/tt-metal.git

git -C tt-metal submodule update --init tt_metal/third_party/umd
git -C tt-metal submodule update --init tt_metal/third_party/tracy
```

## Configure and build

Run inside the x86_64 guest:

```bash
export EMULE_ROOT="$HOME/workspace/tt-emule-latest"
export TT_METAL_DIR="$EMULE_ROOT/tt-metal"
export BUILD_DIR="$EMULE_ROOT/build_emule"

cmake -S "$TT_METAL_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TT_METAL_DIR/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTT_METAL_USE_EMULE=ON \
  -DTT_EMULE_PATH="$EMULE_ROOT/tt-emule" \
  -DCMAKE_INSTALL_PREFIX="$BUILD_DIR" \
  -DWITH_PYTHON_BINDINGS=ON \
  -DTT_METAL_BUILD_TESTS=ON \
  -DTTNN_BUILD_TESTS=ON \
  -DENABLE_TRACY=OFF \
  -DENABLE_DISTRIBUTED=OFF

cmake --build "$BUILD_DIR" \
  --target unit_tests_api unit_tests_integration \
  -j"$(nproc)"
```

Confirm that the build contains emulation support:

```bash
nm -DC "$BUILD_DIR/tt_metal/libtt_metal.so" |
  grep 'emule::execute_program_emulated'
```

Expect a `T` symbol. If it is absent, the build was not configured with
`TT_METAL_USE_EMULE=ON`.

## Run the focused verification

The checked-in smoke script creates fresh, isolated JIT cache directories and
runs the three tests sequentially:

```bash
export EMULE_ROOT="$HOME/workspace/tt-emule-latest"
export TT_METAL_DIR="$EMULE_ROOT/tt-metal"
export BUILD_DIR="$EMULE_ROOT/build_emule"

bash "$EMULE_ROOT/tt-emule/scripts/run_wormhole_blackhole_smoke.sh"
```

It verifies:

1. Wormhole JIT L1 reader/writer transfers at 2, 4, and 6 KiB.
2. Wormhole single-tile matmul across all 16 combinations of math fidelity,
   FP32 destination accumulation, and destination synchronization mode.
3. Blackhole P100 JIT L1 reader/writer transfers at 2, 4, and 6 KiB.

The script exits non-zero on the first failure and prints
`All Wormhole and Blackhole smoke tests passed.` only after all three checks
complete.

## Expected runtime and disposable state

A cold run recompiles JIT kernels and can take several minutes under x86_64
QEMU. Subsequent full regressions may reuse the normal cache at
`/tmp/tt_emule_jit_cache_$UID`, but the focused smoke script deliberately uses
temporary caches so stale kernels cannot hide a source change.

The remote branches and this workflow are durable. The Lima disk, compiled
build tree, and files under `/tmp` are local conveniences and can be recreated
from the steps above.
