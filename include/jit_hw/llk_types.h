#pragma once
// LLK types — types-only surface mirroring upstream
// `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`. Per-domain LLK
// headers (`llk_math_eltwise_unary_datacopy.h`, `llk_unpack_a.h`, …) include
// this to see the type surface without dragging in function bodies or state.

#include <cstdint>

// Tile dimension constants (used by experimental LLK headers)
#ifndef FACE_R_DIM
#define FACE_R_DIM 16
#endif

#ifndef TILE_C_DIM
#define TILE_C_DIM 32
#endif

// Pool type for reduce operations
enum class PoolType : uint8_t {
    SUM = 0,
    AVG = 1,
    MAX = 2,
};

// Reduce dimension
enum class ReduceDim : uint8_t {
    REDUCE_ROW = 0,
    REDUCE_COL = 1,
    REDUCE_SCALAR = 2,
};

// Upstream ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp uses
// ckernel::PoolType / ckernel::ReduceDim. Real tt-metal defines them in the
// ckernel namespace; the JIT here keeps them at global scope, so re-export.
namespace ckernel {
using ::PoolType;
using ::ReduceDim;

// MathFidelity / DstSync — referenced by ttnn/cpp/ttnn/kernel_lib/
// reduce_helpers_compute.inl matmul wrappers and by dest_helpers.hpp's
// SyncFull/SyncHalf comparisons.  Mirrored from upstream
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h.
enum class MathFidelity : std::uint8_t {
    LoFi  = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4,
};
enum class DstSync : std::uint8_t {
    SyncHalf,
    SyncFull,
};

// DataCopyType — `enum class` form. D2M-emitted kernels reference it as
// `DataCopyType::A2D` etc. inside embedded `experimental::tilize_block` /
// similar bodies. Values mirror upstream tt-llk/llk_defs.h.
enum class DataCopyType : std::uint8_t {
    A2D = 0,
    B2D = 1,
};
}  // namespace ckernel

// UnpackToDestEn — bool flag used as a template parameter in D2M-emitted
// `llk_math_eltwise_unary_datacopy<…, UnpackToDestEn>(…)` calls. On the
// emulator we don't dispatch on it; default to false.
inline constexpr bool UnpackToDestEn = false;
