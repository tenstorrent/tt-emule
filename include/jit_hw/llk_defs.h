#pragma once
// LLK definitions stub for emulated mode
// Core enums come from api/compute/common.h; this adds LLK-level stubs.
#include "api/compute/common.h"
#include "internal/firmware_common.h"

// Data copy type constant (used as template parameter)
#ifndef A2D
#define A2D 0
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

// Operand CB interface stub
struct CbInterface {
    uint32_t fifo_page_size = 0;
    uint32_t fifo_rd_ptr = 0;
    uint32_t fifo_wr_ptr = 0;
    uint32_t fifo_size = 0;
    uint32_t fifo_limit = 0;
    uint32_t fifo_num_pages = 0;
};

// Global operand interface stubs
inline CbInterface& get_local_cb_interface(uint32_t) {
    static CbInterface dummy;
    return dummy;
}

inline uint32_t get_operand_id(uint32_t operand) { return operand; }
inline uint32_t get_operand_face_r_dim(uint32_t) { return 16; }
inline uint32_t get_operand_num_faces(uint32_t) { return 4; }
inline bool get_operand_narrow_tile(uint32_t) { return false; }

// Format arrays (stub)
inline uint32_t unpack_src_format[32] = {};
inline uint32_t unpack_dst_format[32] = {};

// LLK function stubs (no-op in emulation)
template <typename... Args>
inline void _llk_unpack_tilize_(Args&&...) {}

inline void llk_unpack_tilize(uint32_t, uint32_t, uint32_t, uint32_t) {}
inline void llk_unpack_tilize_block(uint32_t, uint32_t, uint32_t) {}
inline void llk_unpack_untilize(uint32_t, uint32_t, uint32_t) {}

inline void llk_math_wait_for_dest_available() {}
inline void llk_packer_wait_for_math_done() {}

using ckernel::BroadcastType;

// 4-param version (used by tilize)
template <int CopyType, int AccumMode, BroadcastType Bcast, bool UnpackToDest>
inline void llk_math_eltwise_unary_datacopy(uint32_t) {}

// 3-param version (used by untilize)
template <int CopyType, int AccumMode, BroadcastType Bcast>
inline void llk_math_eltwise_unary_datacopy(uint32_t) {}

// 2-param version (used by tilize)
template <bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t, uint32_t) {}

// 3-param version (used by untilize)
template <int AccumMode, bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t, uint32_t) {}

template <int AccumMode>
inline void llk_math_dest_section_done() {}

template <int AccumMode>
inline void llk_pack_dest_section_done() {}

// Compute startup and init stubs
inline void compute_kernel_hw_startup(uint32_t, uint32_t) {}
inline void tilize_init(uint32_t, uint32_t, uint32_t) {}
inline void tilize_init_short(uint32_t, uint32_t) {}
inline void untilize_init(uint32_t, uint32_t = 0) {}
inline void untilize_init_short(uint32_t) {}
