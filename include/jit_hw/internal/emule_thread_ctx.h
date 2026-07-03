// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Per-thread (per-RISC) execution context — the single source of truth for an
// emulated RISC's thread-local state, specialized by RISC type. See
// docs/state-tiers.md for the full per-chip / per-core / per-thread map.
//
// Mechanism: ONE thread_local pointer (`__emule_self`) reaches the context. The
// context object is owned by the runner today (one per kernel-execution thread,
// allocated in the launch prologue) and by the fiber later (pillar0 — the
// scheduler repoints `__emule_self` on swap-in). No state is copied on a swap;
// only the pointer moves.
//
// Specialized by RISC type per the silicon resource partition (arch-verified):
//   - compute TRISCs (emule fuses UNPACK/MATH/PACK into one thread) own DST +
//     the SFPU; data-movement RISCs (BRISC/NCRISC) own the NOC cmd-buf state.
//   - the cross-role circular-buffer ring pointers and the set-once identity /
//     handles live in the shared base (a CB producer is often a DM RISC and the
//     consumer a compute TRISC, so the pointers are per-RISC but not exclusive
//     to either role).
//
// Fields are migrated into these structs in stages (see the plan); this header
// starts as the structural scaffold (base + two derived + the pointer + the
// typed accessors).

#include <cstdint>
#include <random>  // std::mt19937 (per-fiber rand engine, #243)
#include <unordered_map>
// CoreState (per-core) lives in its own minimal header so device.hpp/umd can embed
// it without parsing the kernel-only ThreadCommonCtx/ComputeThreadCtx below (whose
// members reference kernel-only types such as sfpi).
#include "jit_hw/internal/emule_core_state.h"  // tt_emule::CoreState
// SFPI value types + grouped SfpuState (dependency-free: only <array>/<cstdint>),
// so ComputeThreadCtx can hold the SFPU state by value without pulling in sfpi.h.
#include "jit_hw/sfpi_types.h"  // sfpi::SfpuState

namespace tt_emule {
class Core;
class Device;
struct CBSyncState;
struct EmuleDFBInterface;
class TileCounterArray;
}  // namespace tt_emule

// Mirror of upstream's LocalCBInterface — the per-RISC CB read/write register
// file shape, so kernels written against the real device API compile/run
// unchanged. fifo_{rd,wr}_ptr are 16-byte-encoded host addresses (kernel does
// `<< 4`). Defined HERE (not emule_cb_ptr.h) so it can be an inline member of
// ThreadCommonCtx without a circular include (emule_cb_ptr.h includes this header).
struct LocalCBInterface {
    uint32_t fifo_size;
    uint32_t fifo_limit;  // 16-byte-encoded exclusive end = base + size; wrap when ptr >= fifo_limit.
    uint32_t fifo_page_size;
    uint32_t fifo_num_pages;
    uint32_t fifo_rd_ptr;
    uint32_t fifo_wr_ptr;
    union {
        uint32_t tiles_acked_received_init;
        struct {
            uint16_t tiles_acked;
            uint16_t tiles_received;
        };
    };
    uint32_t fifo_wr_tile_ptr;
};

// local_cb array size: the max CB count across emule arches (BH=64, WH=32).
// Hardcoded rather than NUM_CIRCULAR_BUFFERS because this header is also pulled by
// the umd TU (via device.hpp), which lacks tt-metalium/ on its -I. A consistent
// size across all TUs also keeps the ctx layout/ABI identical; cb_id is always
// < the arch's NUM_CIRCULAR_BUFFERS ≤ this bound.
static constexpr uint32_t __EMULE_CTX_MAX_CBS = 64;

// Object-Intent (ASAN) resolved-range log capacity, per fiber. Ample — kernels resolve
// pointers into <10 distinct buffers; overflow drops the excess (see ThreadCommonCtx).
static constexpr uint32_t __EMULE_SAN_RESOLVED_CAP = 64;

// Inline direct-write (noc_inline_dw_write) set/with-state cache entry, per NOC.
struct __emule_dw_state {
    uint64_t addr = 0;
    uint32_t val = 0;
};

// unified_kernels stateful-NOC (preprogram-all-state → issue-txn) capture slot,
// per NOC. One slot per transaction category (read/write/atomic/multicast) so a
// sender that preprograms a write then an atomic-inc and issues both does not
// alias the captured operands. Used by blaze/kernels/dataflow_utils.hpp.
struct __emule_uk_state {
    uint64_t noc_addr = 0;      // write: dst noc addr; read: src noc addr; atomic: target
    uint32_t local_addr = 0;    // write: src local L1; read: dst local L1; multicast: src local
    uint32_t len = 0;
    uint32_t incr = 0;          // atomic increment value
    bool include_self = false;  // multicast: mirrors NOC_CMD_BRCST_SRC_INCLUDE
};

// Matmul operand bridge: llk_unpack_AB_matmul records the operand CBs/tiles,
// llk_math_matmul reads them back (HW splits these across cores; emule serializes).
struct __emule_matmul_bridge {
    uint32_t in0_cb  = 0;
    uint32_t in1_cb  = 0;
    uint32_t in0_idx = 0;
    uint32_t in1_idx = 0;
};

// Pack-fused ReLU clamp mode. Full definition (with enumerators) is in
// api/compute/common.h; declared opaque here (complete type, size known) so
// ComputeThreadCtx can hold it without pulling in the compute headers.
enum class ReluType : uint8_t;

// ---- Shared base: every RISC thread ---------------------------------------
// (Identity/handles + the cross-role CB ring pointers are migrated in here in a
// later stage; for now it carries the kind discriminator + the per-core link.)
struct ThreadCommonCtx {
    enum class Kind : uint8_t { Compute, Datamovement };
    const Kind kind;
    tt_emule::CoreState* core = nullptr;  // borrowed per-core state (logical coords)

    // Set-once identity / handles — written in the runner launch prologue, read by
    // both the kernel and the runner's extern-C resolvers (same thread → same ctx).
    uint32_t* rt_args = nullptr;            // was __rt_args
    uint32_t* common_rt_args = nullptr;     // was __common_rt_args
    tt_emule::Core*   core_obj = nullptr;   // was __core (the Core object handle)
    tt_emule::Device* device = nullptr;     // was __device
    uint32_t chip_id = 0;                   // owning chip / device id (was __emule_chip_id) —
                                            // per-fiber so fabric teleport resolves the right
                                            // source chip when a worker co-runs fibers from
                                            // multiple chips (mesh register/run dispatch).
    uint8_t* bridge_l1 = nullptr;           // was __emule_bridge_l1
    uint8_t* bridge_dram = nullptr;         // was __emule_bridge_dram
    tt_emule::CBSyncState* cbs = nullptr;          // was __emule_cbs
    tt_emule::EmuleDFBInterface* dfbs = nullptr;   // was __emule_dfbs
    tt_emule::TileCounterArray* tc_array = nullptr;  // was __emule_tc_array
    std::unordered_map<uint64_t, tt_emule::Core*>* core_map = nullptr;  // was __emule_core_map
    uint8_t  processor_id = 0;   // RISC-V mhartid analogue (was __processor_id)
    uint8_t  neo_id = 0;         // Quasar NEO_ID CSR  (was __emule_neo_id)
    uint8_t  trisc_id = 0;       // Quasar TRISC_ID CSR (was __emule_trisc_id)
    uint32_t num_threads = 1;    // get_num_threads()  (was __emule_num_threads)
    uint32_t my_thread_id = 0;   // get_my_thread_id() (was __emule_my_thread_id)

    // Cross-role CB state (shared base — a CB producer/consumer may be either role).
    uint32_t cb_self_consume_mask = 0;            // was __emule_cb_self_consume_mask
    uint32_t cb_self_produce_mask = 0;            // was __emule_cb_self_produce_mask
    LocalCBInterface local_cb[__EMULE_CTX_MAX_CBS]{};  // per-RISC CB ring ptrs (was __emule_local_cb)

    // Object-Intent (ASAN) per-fiber resolved-range log. Fully fiber-local: the
    // kernel-side OOB check appends each live-tensor extent it resolved a pointer into
    // here, and the runner diffs any snapshotted extent that was modified without being
    // recorded (post-launch). Living in the ctx (reached via __emule_self) means a fiber
    // swap carries it — no thread-locals, nothing for the scheduler to restore. Inactive
    // => no recording. Capacity is ample (kernels touch <10 buffers); overflow drops the
    // excess, biasing the post-launch diff toward false positives, never negatives.
    // See tt-emule #241, docs/ASAN.md, and emule_sanitizers.cpp (ObjectIntentTracker).
    bool     san_resolved_active = false;
    uint32_t san_resolved_count = 0;
    uint64_t san_resolved_log[__EMULE_SAN_RESOLVED_CAP] = {};

    // Fabric PacketHeaderPool allocation state (per-fiber). Silicon re-zeroes these
    // statics on every program launch (fresh kernel .bss); emule reuses the JIT .so,
    // so a `thread_local` would leak the cursor + route table across ops on a persistent
    // worker — the cursor overflows the per-RISC partition and the route_id table wraps,
    // corrupting the fabric multicast routes → wrong-chip / garbage relays (tt-emule #221).
    // Homing them in the per-fiber ctx (a fresh object per launch, see launch_cores) makes
    // the pool fresh per program, matching silicon. Read/written by the packet_header_pool.h
    // shadow via __emule_self. PHDR_MAX_ROUTES mirrors the shadow's MAX_ROUTES.
    static constexpr uint32_t PHDR_MAX_ROUTES = 16;
    uint32_t phdr_cursor = 0;                        // next free header index in this RISC partition
    uint8_t  phdr_route_count = 0;                   // routes registered this launch
    uint32_t phdr_route_first[PHDR_MAX_ROUTES] = {}; // route_id → first header index
    uint8_t  phdr_route_num[PHDR_MAX_ROUTES] = {};   // route_id → header count

    explicit ThreadCommonCtx(Kind k) : kind(k) {}
    virtual ~ThreadCommonCtx() = default;  // owned via base ptr (runner/fiber)
};

// ---- Compute thread (fused UNPACK/MATH/PACK) ------------------------------
// (DST, SFPU/LReg, pack/unpack config, LLK trackers, op accumulators migrate
// here in a later stage.)
struct ComputeThreadCtx : ThreadCommonCtx {
    // DST register file + adjacency (was __emule_dst / __emule_dst_fresh / __emule_src_scratch).
    // Sizes mirror common.h's __EMULE_DST_TILES(16) / __EMULE_TILE_ELEMS(1024); hardcoded
    // here so this header stays free of the compute headers (it's pulled by the runner).
    float dst[16][1024] = {};
    bool  dst_fresh[16] = {true, true, true, true, true, true, true, true,
                           true, true, true, true, true, true, true, true};
    float src_scratch[1024] = {};

    // Pack/math accumulate flags + pack sub-rect config (was __emule_l1_acc_enabled, etc.).
    bool     l1_acc_enabled = false;     // was __emule_l1_acc_enabled
    bool     dest_accum_en = false;      // was __emule_dest_accum_en
    uint32_t pack_x_end = 31;            // was __emule_pack_x_end
    uint32_t pack_face_r_dim = 16;       // was __emule_pack_face_r_dim
    uint32_t pack_num_faces = 4;         // was __emule_pack_num_faces
    uint32_t pack_num_tiles = 1;         // was __emule_pack_num_tiles
    uint32_t pack_offset[__EMULE_CTX_MAX_CBS] = {};  // was __emule_pack_offset[NUM_CIRCULAR_BUFFERS]
    uint32_t pack_width[__EMULE_CTX_MAX_CBS] = {};   // llk_pack_init blocked-width per CB (0 ⇒ width 1)

    // Op accumulators.
    float    welford_mean[32] = {};          // was __emule_welford_mean
    float    welford_m2[32] = {};            // was __emule_welford_m2
    float    welford_grp_mean[128][32] = {}; // was __emule_welford_grp_mean (__EMULE_WELFORD_MAX_GROUPS=128)
    float    welford_grp_m2[128][32] = {};   // was __emule_welford_grp_m2
    float    ema_alpha = 0.0f;               // was __emule_ema_alpha
    float    ema_beta = 0.0f;                // was __emule_ema_beta
    float    ema_prev[32] = {};              // was __emule_ema_prev
    uint32_t dropout_rng_state = 0x9E3779B9u;// was __emule_dropout_rng_state
    float    cumsum_acc[32] = {};            // was __emule_cumsum_acc
    float    cumprod_acc[1024] = {};         // was __emule_cumprod_acc
    bool     cumprod_acc_initialized = false;// was __emule_cumprod_acc_initialized
    float    quant_zero_point = 0.0f;        // was __emule_quant_zero_point
    float    requant_zero_point = 0.0f;      // was __emule_requant_zero_point
    float    dequant_zero_point = 0.0f;      // was __emule_dequant_zero_point
    // Op-state carried across an init→execute pair or a multi-tile draw (were per-worker
    // thread_locals; per-fiber here so co-scheduled fibers at >1 fiber/worker don't race — #243).
    bool     dst_holds_int[16] = {};         // was __emule_dst_holds_int (common_globals.h)
    float    exp_init_scale = 1.0f;          // was __emule_exp_init_scale (exp.h)
    uint32_t reduce_block_ct_dim = 1;        // was __emule_reduce_block_ct_dim (reduce_custom.h)
    std::mt19937 rand_engine{};              // was __emule_rand_engine (rand.h)
    bool     rand_deterministic = false;     // was __emule_rand_deterministic
    uint64_t rand_nonce = 0;                 // was __emule_rand_nonce

    // SFPU/sfpi intrinsic state — all grouped (DST window, predication stack,
    // LReg file, programmable const regs). See sfpi_types.h::SfpuState; reached
    // from sfpi.h via __emule_compute_ctx().sfpu.
    sfpi::SfpuState sfpu{};

    // LLK tilize/untilize trackers (was internal/llk_state.h thread_locals).
    uint32_t llk_unpack_src_cb = 0;          // was __emule_compute_ctx().llk_unpack_src_cb
    uint32_t llk_unpack_start_tile_idx = 0;  // was __emule_llk_unpack_start_tile_idx (thread_local global)
    uint32_t llk_unpack_block_c = 0;         // was __emule_llk_unpack_block_c
    uint32_t llk_unpack_current_tile = 0;    // was __emule_llk_unpack_current_tile
    bool     llk_unpack_is_tilize = false;   // was __emule_llk_unpack_is_tilize
    uint32_t llk_pack_offset = 0;            // was __emule_llk_pack_offset
    bool     llk_pack_is_untilize = false;   // was __emule_llk_pack_is_untilize
    uint32_t llk_pack_block_c = 0;           // was __emule_llk_pack_block_c
    // transpose=1 (mm_init/mm_block_init) => unpacker delivers IN1 (SrcB) transposed,
    // i.e. compute A*B^T. emule has no SrcB modelling, so matmul_tiles reads this and
    // transposes its decoded IN1 tile before the FMA loop. (was __llk_matmul_transpose)
    bool     llk_matmul_transpose = false;
    uint32_t pack_rows_num = 32;             // was __emule_compute_ctx().pack_rows_num (llk_pack.h)
    __emule_matmul_bridge matmul_state{};    // was __emule_compute_ctx().matmul_state
    uint32_t llk_binary_icb0 = 0;
    uint32_t llk_binary_icb1 = 0;
    uint32_t llk_binary_itile0 = 0;
    uint32_t llk_binary_itile1 = 0;

    // Pack-fused ReLU clamp (was __emule_pack_relu_mode / _threshold in common.h).
    // pack_relu_mode value-inits to 0 == ReluType::NO_RELU.
    ReluType pack_relu_mode{};
    float    pack_relu_threshold = 0.0f;

    ComputeThreadCtx() : ThreadCommonCtx(Kind::Compute) {}
};

// ---- Data-movement thread (BRISC / NCRISC) --------------------------------
// (Per-NOC cmd-buf caches, counters, addr-gen scratch migrate here in a later
// stage.)
struct DatamovementThreadCtx : ThreadCommonCtx {
    // Per-NOC[2] command-buffer set/with-state caches (emule-only). The silicon-named
    // noc_*_num_issued/acked counters stay global (read by unmodified upstream).
    uint32_t  one_packet_state_size[2] = {};       // was __emule_one_packet_state_size
    uint64_t  shard_noc_addr_base[2] = {};         // was shard_noc_addr_base
    uint32_t  shard_size[2] = {};                  // was shard_size
    uint32_t  shard_vc[2] = {};                    // was shard_vc
    uint64_t  write_one_packet_state_dst[2] = {};  // was __emule_write_one_packet_state_dst
    uint32_t  write_one_packet_state_size[2] = {}; // was __emule_write_one_packet_state_size
    __emule_dw_state dw_st[2];                      // was __emule_dw_st
    uint32_t  noc_cached_size[2] = {};             // was __emule_noc_cached_size (NUM_NOCS=2)
    uintptr_t noc_cached_write_dst[2] = {};        // was __emule_noc_cached_write_dst
    __emule_uk_state uk_rd[2];                      // was __emule_uk_rd (PR #182)
    __emule_uk_state uk_wr[2];                      // was __emule_uk_wr (PR #182)
    __emule_uk_state uk_at[2];                      // was __emule_uk_at (PR #182)
    __emule_uk_state uk_mc[2];                      // was __emule_uk_mc (PR #182)

    DatamovementThreadCtx() : ThreadCommonCtx(Kind::Datamovement) {}
};

// The single thread_local. Defined non-inline in the runner (exported via
// -rdynamic); the JIT .so resolves it at dlopen. The runner sets it to the
// derived context matching the kernel's RISC type before the startup barrier.
extern thread_local ThreadCommonCtx* __emule_self;

// Typed accessors — the only way a kernel reaches specialized state. Kernel-only
// (guarded by __EMULE_JIT_MODE, defined in the jit_kernel_stubs.hpp preamble): the
// host runner TU defines/sets `__emule_self` but never reads through it, and must
// not pull a bare `ASSERT` macro. The kind ASSERT documents the silicon contract
// (no-op in release): DST/SFPU are reachable only from a compute thread, NOC
// cmd-buf state only from a data-movement thread; a crossing is a bug to surface,
// not to support. (BH HW technically permits a TRISC to drive the NOC via MMIO —
// that is a software-convention boundary, not a hard wall; see docs/state-tiers.md.)
#ifdef __EMULE_JIT_MODE
#ifndef ASSERT
#define ASSERT(...) ((void)0)
#endif
inline ComputeThreadCtx& __emule_compute_ctx() {
    ASSERT(__emule_self != nullptr && __emule_self->kind == ThreadCommonCtx::Kind::Compute);
    return *static_cast<ComputeThreadCtx*>(__emule_self);
}
inline DatamovementThreadCtx& __emule_datamovement_ctx() {
    ASSERT(__emule_self != nullptr && __emule_self->kind == ThreadCommonCtx::Kind::Datamovement);
    return *static_cast<DatamovementThreadCtx*>(__emule_self);
}
#endif  // __EMULE_JIT_MODE
