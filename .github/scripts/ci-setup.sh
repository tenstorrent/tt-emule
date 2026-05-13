#!/usr/bin/env bash
# Install build toolchain for the tt-emule CI build job.
#
# Run on tt-ubuntu-2204-large-stable runners. Idempotent — safe to re-run.

set -euo pipefail

echo "== Installing build toolchain =="

# clang-20 + LLVM tooling. Tenstorrent ubuntu-2204 base images already include
# the LLVM apt repo; if not present, fall back to upstream installer.
if ! command -v clang-20 >/dev/null 2>&1; then
    if [ ! -f /etc/apt/sources.list.d/llvm-toolchain-jammy-20.list ]; then
        echo "  Adding LLVM apt repo"
        wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
            | sudo gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] \
http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" \
            | sudo tee /etc/apt/sources.list.d/llvm-toolchain-jammy-20.list >/dev/null
    fi
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        clang-20 \
        clang-tools-20 \
        llvm-20 \
        lld-20 \
        libclang-rt-20-dev
fi

# cmake from Kitware apt repo — Ubuntu 22.04 default (3.22) is too old for
# tt-metal (requires 3.24+). Mirrors tt-metal's install_dependencies.sh.
if ! command -v cmake >/dev/null 2>&1 || \
   ! cmake --version | head -1 | awk '{print $3}' | awk -F. '{exit !($1 > 3 || ($1 == 3 && $2 >= 24))}'; then
    if [ ! -f /etc/apt/sources.list.d/kitware.list ]; then
        echo "  Adding Kitware apt repo for current cmake"
        wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc \
            | sudo gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
https://apt.kitware.com/ubuntu/ jammy main" \
            | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
        sudo apt-get update -qq
        # Best-effort: install the keyring package so future apt updates work.
        sudo apt-get install -y --no-install-recommends kitware-archive-keyring || true
    fi
    sudo apt-get install -y --no-install-recommends cmake
fi

# Other build tools and system libraries tt-metal requires unconditionally
# (NUMA + hwloc are find_library'd in tt-metal/third_party/CMakeLists.txt
# regardless of ENABLE_DISTRIBUTED — they're used by the runtime for topology
# discovery and thread pinning, not just by the distributed/MPI path).
sudo apt-get install -y --no-install-recommends \
    ninja-build \
    ccache \
    pkg-config \
    python3 \
    python3-yaml \
    libnuma-dev \
    libhwloc-dev

echo ""
echo "== Toolchain versions =="
clang-20 --version | head -1
cmake --version | head -1
ninja --version
ccache --version | head -1
python3 --version

# Configure ccache. Cache lives under $HOME/.ccache so actions/cache can pick it up.
mkdir -p "$HOME/.ccache"
ccache --set-config max_size=7G
ccache --set-config compression=true
ccache --set-config compression_level=6
ccache --zero-stats >/dev/null
echo ""
echo "ccache configured: max_size=7G, compression=true"
