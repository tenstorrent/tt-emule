#!/bin/bash
# Clean rebuild of tt-metal with emule support.
set -e
export ROOT=/localdev/sgholami
cd $ROOT/tt-metal
echo "=== CONFIGURE $(date) ==="
cmake -S . -B build_emule -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=$ROOT/tt-metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DTT_METAL_USE_EMULE=ON \
    -DTT_EMULE_PATH=$ROOT/tt-emule \
    -DWITH_PYTHON_BINDINGS=ON \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DENABLE_TRACY=OFF \
    -DENABLE_DISTRIBUTED=ON \
    -DCMAKE_INSTALL_PREFIX=$ROOT/tt-metal/build_emule
echo "=== BUILD $(date) ==="
cmake --build build_emule -j$(nproc)
echo "=== POST-BUILD SYMLINKS $(date) ==="
ln -sfn $ROOT/tt-metal/build_emule/ttnn/_ttnn.so $ROOT/tt-metal/ttnn/ttnn/_ttnn.so
cd $ROOT/tt-metal/build_emule/lib
ln -sfn ../tt_metal/libtt_metal.so libtt_metal.so
ln -sfn ../tt_stl/libtt_stl.so libtt_stl.so
ln -sfn ../ttnn/_ttnncpp.so _ttnncpp.so
ln -sfn ../ttnn/_ttnn.so _ttnn.so
echo "=== VERIFY $(date) ==="
nm -DC $ROOT/tt-metal/build_emule/tt_metal/libtt_metal.so | grep -q emule::execute_program_emulated && echo "emule symbol OK" || echo "MISSING emule symbol"
echo "=== BUILD DONE $(date) ==="
