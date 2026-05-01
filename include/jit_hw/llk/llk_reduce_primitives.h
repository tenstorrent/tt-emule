#pragma once
// LLK stubs for the matmul-based reduce path used by ttnn::sum-on-W via
// ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl.  Most stubs are no-ops
// (MATH/UNPACK/PACK macros are identity in emule); llk_math_matmul delegates
// to ckernel::matmul_tiles for actual math.

#include <cstdint>

// Forward-declare the enums.  Including llk_defs.h here would also pull in its
// dummy get_local_cb_interface(), which would shadow dataflow_api.h's real
// CB-state-backed version in DM kernels and silently break NOC writes.
namespace ckernel {
enum class MathFidelity : std::uint8_t;
}
enum class PoolType  : std::uint8_t;
enum class ReduceDim : std::uint8_t;

#ifndef MM_THROTTLE
#  define MM_THROTTLE 0
#endif
#ifndef MATH_FIDELITY
#  define MATH_FIDELITY ckernel::MathFidelity::HiFi4
#endif

// Bridges UNPACK's tile selection across to MATH on the same host thread.
// Real HW splits these across cores; emule serializes them, and llk_math_matmul
// reads what llk_unpack_AB_matmul wrote.
namespace __emule_matmul_state {
inline thread_local uint32_t in0_cb  = 0;
inline thread_local uint32_t in1_cb  = 0;
inline thread_local uint32_t in0_idx = 0;
inline thread_local uint32_t in1_idx = 0;
}

// state_configure is called once at init by reduce_with_matmul_init; its writes
// are immediately overwritten by the per-tile llk_unpack_AB_matmul calls, so
// this can be a no-op.
inline void state_configure(uint32_t /*in1_cb*/, uint32_t /*in0_cb*/) {}

template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args>
inline void llk_math_matmul_init(Args... /*ignored*/) {}
template <typename... Args>
inline void llk_unpack_AB_matmul_init(Args... /*ignored*/) {}
template <int Mode, typename... Args>
inline void llk_unpack_reconfig_data_format_srca(Args... /*ignored*/) {}
template <int Mode, typename... Args>
inline void llk_math_reconfig_data_format_srca(Args... /*ignored*/) {}

inline void llk_unpack_AB_matmul(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_idx, uint32_t in1_idx) {
    __emule_matmul_state::in0_cb  = in0_cb;
    __emule_matmul_state::in1_cb  = in1_cb;
    __emule_matmul_state::in0_idx = in0_idx;
    __emule_matmul_state::in1_idx = in1_idx;
}

namespace ckernel {
void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile, uint32_t in1_tile, uint32_t idst);
}
template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args>
inline void llk_math_matmul(uint32_t idst, Args... /*ignored*/) {
    ckernel::matmul_tiles(
        __emule_matmul_state::in0_cb,
        __emule_matmul_state::in1_cb,
        __emule_matmul_state::in0_idx,
        __emule_matmul_state::in1_idx,
        idst);
}

template <PoolType Pool, ReduceDim Dim, typename... Args>
inline void llk_unpack_AB_reduce_init(Args... /*ignored*/) {}
template <PoolType Pool, ReduceDim Dim, int Mode, ckernel::MathFidelity Fidelity, typename... Args>
inline void llk_math_reduce_init(Args... /*ignored*/) {}
