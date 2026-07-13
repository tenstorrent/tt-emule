// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// emule shadow of the kernel-lib `unified_kernels` stateful-NOC convenience
// layer. The production header builds NOC transactions by programming the NOC
// command-buffer MMIO registers (NOC_CMD_BUF_WRITE_REG ... NOC_CTRL_SEND_REQ) in
// a "preprogram-all-state then issue" split — a silicon perf optimization that
// lives below the kernel-author API (Layer 4 in docs/LAYER_MODEL.md) and is
// therefore not emulated; under emule those register writes are no-ops, so the
// real header's writes/atomics/multicasts silently vanish.
//
// This shadow keeps the exact same signatures and the public->leaf delegation,
// but replaces the register-poking leaf bodies with emule's synchronous Layer-1
// NOC primitives: set_state/preprogram captures the operands into per-fiber
// state; with_state/issue_txn replays them through noc_async_write /
// noc_semaphore_inc / noc_async_read / __emule_multicast_write. The between-call
// capture lives in the per-fiber DatamovementThreadCtx (uk_rd/wr/at/mc[noc], see
// internal/emule_thread_ctx.h) rather than a bare thread_local, so it is safe
// under the fiber engine where several fibers share one OS thread. This mirrors
// the emule model already used for the stateful inline-write API
// (noc_inline_dw_write_set_state/with_state via DatamovementThreadCtx::dw_st[2]),
// and matches the documented handling of the mcast op's register-level path:
// route through the kernel-author NOC API instead of modelling NOC hardware.

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace unified_kernels {

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

// ---- emule synchronous state capture (one slot per transaction category) ----
// The capture slots live per-fiber in DatamovementThreadCtx (uk_rd/wr/at/mc[noc],
// __emule_uk_state defined in internal/emule_thread_ctx.h) — reached below via
// __emule_datamovement_ctx(). Writes and atomics are distinct categories, so a
// sender that preprograms a write then an atomic-inc and issues both does not
// alias the captured state.

// Set the low NOC-local bits (an L1 offset) while preserving the saved upper
// coords. The shadow does NOT slot-mask addresses at the replay site: emit-time
// canonicalization — slot-masking worker-L1 offsets (the `dst_noc_coord |
// (uint64_t)get_write_ptr(cb)` OR'd-host-pointer form) while leaving DRAM bank
// offsets (> 2 MB) intact — is done role-aware by __emule_resolve_noc_addr
// (emulated_program_runner.cpp). Masking here instead would truncate DRAM
// destinations; see issue #252.
FORCE_INLINE void __emule_uk_set_local_bits(uint64_t& noc_addr, uint32_t local) {
    constexpr uint64_t local_mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
    noc_addr = (noc_addr & ~local_mask) | (uint64_t(local) & local_mask);
}

// ---- NOC counter bookkeeping (unchanged from production; emule barriers do
// not read these back, but keeping them verbatim avoids any divergence) ----

template <bool posted>
FORCE_INLINE void unicast_write_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, count);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, count);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, count);
        }
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += count;
        } else {
            noc_nonposted_writes_num_issued[noc] += count;
            noc_nonposted_writes_acked[noc] += count;
        }
    }
}

template <bool posted>
FORCE_INLINE void unicast_atomic_inc_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (!posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_ATOMICS_ACKED>(noc, count);
        }
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (!posted) {
            noc_nonposted_atomics_acked[noc] += count;
        }
    }
}

FORCE_INLINE void noc_async_read_increment_counters(uint32_t count = 1, uint8_t noc = noc_index) {
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        inc_noc_counter_val<proc_type, NocBarrierType::READS_NUM_ISSUED>(noc, count);
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        noc_reads_num_issued[noc] += count;
    }
}

// ---- Unicast read ----

template <bool set_noc_coord, bool set_addresses, bool set_size, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_read_set_state(
    uint64_t src_noc_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index, uint8_t vc = 1) {
    if constexpr (increment_counters) {
        noc_async_read_increment_counters(1, noc);
    }
    auto& rd = __emule_datamovement_ctx().uk_rd[noc];
    if constexpr (set_addresses) {
        rd.noc_addr = src_noc_addr;
        rd.local_addr = dst_local_addr;
    } else if constexpr (set_noc_coord) {
        constexpr uint64_t local_mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
        rd.noc_addr = (rd.noc_addr & local_mask) | (src_noc_addr & ~local_mask);
    }
    if constexpr (set_size) {
        rd.len = len_bytes;
    }
}

template <bool set_addresses, bool set_size, bool wait_cmd_buf_ready, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_read_with_state(
    uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index) {
    if constexpr (increment_counters) {
        noc_async_read_increment_counters(1, noc);
    }
    auto& rd = __emule_datamovement_ctx().uk_rd[noc];
    if constexpr (set_addresses) {
        // Canonicalize the source L1 offset before OR-ing it into the saved upper
        // coords — src_local_addr can be a firmware-style offset or a truncated
        // host pointer in emule. Mirrors noc_async_read_with_state in
        // dataflow_api.h (__emule_addr_to_offset), so the replayed noc_async_read
        // resolves the correct remote address.
        __emule_uk_set_local_bits(rd.noc_addr, __emule_addr_to_offset(src_local_addr));
        rd.local_addr = dst_local_addr;
    }
    if constexpr (set_size) {
        rd.len = len_bytes;
    }
    noc_async_read(rd.noc_addr, rd.local_addr, rd.len, noc);
}

template <bool increment_counters = false, uint8_t cmd_buf = read_cmd_buf>
FORCE_INLINE void noc_async_read_preprogram_all_state(
    uint64_t src_noc_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index, uint8_t vc = 1) {
    unicast_read_set_state<true, true, true, increment_counters, cmd_buf>(
        src_noc_addr, dst_local_addr, len_bytes, noc, vc);
}

template <bool increment_counters = true, uint8_t cmd_buf = read_cmd_buf>
FORCE_INLINE void noc_async_read_issue_txn(uint8_t noc = noc_index) {
    unicast_read_with_state<false, false, false, increment_counters, cmd_buf>(0, 0, 0, noc);
}

// ---- Unicast write ----

template <bool posted, bool set_noc_coord, bool set_addresses, bool set_size, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_write_set_state(
    uint32_t src_local_addr,
    uint64_t dst_noc_addr,
    uint32_t len_bytes,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    if constexpr (increment_counters) {
        unicast_write_increment_counters<posted>(1, noc);
    }
    auto& wr = __emule_datamovement_ctx().uk_wr[noc];
    if constexpr (set_addresses) {
        wr.local_addr = src_local_addr;
        wr.noc_addr = dst_noc_addr;
    } else if constexpr (set_noc_coord) {
        constexpr uint64_t local_mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
        wr.noc_addr = (wr.noc_addr & local_mask) | (dst_noc_addr & ~local_mask);
    }
    if constexpr (set_size) {
        wr.len = len_bytes;
    }
}

template <
    bool posted,
    bool set_addresses,
    bool set_size,
    bool wait_cmd_buf_ready,
    bool increment_counters,
    uint8_t cmd_buf>
FORCE_INLINE void unicast_write_with_state(
    uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes, uint8_t noc = noc_index) {
    if constexpr (increment_counters) {
        unicast_write_increment_counters<posted>(1, noc);
    }
    auto& wr = __emule_datamovement_ctx().uk_wr[noc];
    if constexpr (set_addresses) {
        wr.local_addr = src_local_addr;
        // Canonicalize the destination L1 offset before OR-ing into the saved
        // upper coords — dst_local_addr can be a firmware offset or a truncated
        // host pointer. Mirrors the read path (unicast_read_with_state) and
        // noc_async_read_with_state in dataflow_api.h.
        __emule_uk_set_local_bits(wr.noc_addr, __emule_addr_to_offset(dst_local_addr));
    }
    if constexpr (set_size) {
        wr.len = len_bytes;
    }
    // No slot-mask here: __emule_resolve_noc_addr masks worker-L1 offsets and
    // preserves DRAM bank offsets (role-aware). Masking would truncate DRAM (#252).
    noc_async_write(wr.local_addr, wr.noc_addr, wr.len, noc);
}

template <bool posted, bool increment_counters = false, uint8_t cmd_buf = write_cmd_buf>
FORCE_INLINE void noc_async_write_preprogram_all_state(
    uint32_t src_local_addr,
    uint64_t dst_noc_addr,
    uint32_t len_bytes,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    unicast_write_set_state<posted, true, true, true, increment_counters, cmd_buf>(
        src_local_addr, dst_noc_addr, len_bytes, noc, vc);
}

template <bool posted, bool increment_counters = true, uint8_t cmd_buf = write_cmd_buf>
FORCE_INLINE void noc_async_write_issue_txn(uint8_t noc = noc_index) {
    unicast_write_with_state<posted, false, false, false, increment_counters, cmd_buf>(0, 0, 0, noc);
}

// ---- Unicast atomic increment ----

template <bool posted, bool set_addr, bool set_incr, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_atomic_inc_set_state(
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap = 31,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t atomic_ret_val = MEM_NOC_ATOMIC_RET_VAL_ADDR) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    if constexpr (increment_counters) {
        unicast_atomic_inc_increment_counters<posted>(1, noc);
    }
    auto& at = __emule_datamovement_ctx().uk_at[noc];
    if constexpr (set_addr) {
        at.noc_addr = addr;
    }
    if constexpr (set_incr) {
        at.incr = incr;
    }
}

template <bool posted, bool set_addr, bool set_incr, bool wait_cmd_buf_ready, bool increment_counters, uint8_t cmd_buf>
FORCE_INLINE void unicast_atomic_inc_with_state(
    uint64_t addr, uint32_t incr, uint32_t wrap = 31, uint8_t noc = noc_index) {
#ifdef ARCH_BLACKHOLE
    static_assert(!posted, "Blackhole does not support posted atomics");
#endif
    if constexpr (increment_counters) {
        unicast_atomic_inc_increment_counters<posted>(1, noc);
    }
    auto& at = __emule_datamovement_ctx().uk_at[noc];
    if constexpr (set_addr) {
        at.noc_addr = addr;
    }
    if constexpr (set_incr) {
        at.incr = incr;
    }
    // The atomic target is `coord | constexpr_semaphore_offset`; the offset is a
    // firmware L1 offset (< 2 MB), so noc_semaphore_inc's resolve handles it
    // directly. It resolves the remote core's L1 and does an atomic fetch_add.
    noc_semaphore_inc(at.noc_addr, at.incr, noc);
}

template <bool posted, bool increment_counters = false, uint8_t cmd_buf = write_at_cmd_buf>
FORCE_INLINE void noc_async_atomic_inc_preprogram_all_state(
    uint64_t addr,
    uint32_t incr,
    uint32_t wrap = 31,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC,
    uint32_t atomic_ret_val = MEM_NOC_ATOMIC_RET_VAL_ADDR) {
    unicast_atomic_inc_set_state<posted, true, true, increment_counters, cmd_buf>(
        addr, incr, wrap, noc, vc, atomic_ret_val);
}

template <bool posted, bool increment_counters = true, uint8_t cmd_buf = write_at_cmd_buf>
FORCE_INLINE void noc_async_atomic_inc_issue_txn(uint8_t noc = noc_index) {
    unicast_atomic_inc_with_state<posted, false, false, false, increment_counters, cmd_buf>(0, 0, 31, noc);
}

// ---- Multicast write ----

constexpr bool multicast_is_shared_write_cmd_buf = write_cmd_buf == write_reg_cmd_buf;

template <uint8_t noc>
FORCE_INLINE uint64_t get_safe_multicast_noc_addr(
    uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr) {
    if constexpr (noc == 0) {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start),
            DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end), addr);
    } else {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_end), DYNAMIC_NOC_Y(noc, noc_y_end),
            DYNAMIC_NOC_X(noc, noc_x_start), DYNAMIC_NOC_Y(noc, noc_y_start), addr);
    }
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid, bool posted, uint32_t count = 1>
FORCE_INLINE void multicast_write_increment_counters() {
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t num_dests =
        loopback ? mcast_num_cores : (is_part_of_receiver_grid ? mcast_num_cores - 1 : mcast_num_cores);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, count);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, count);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests * count);
        }
    }
    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += count;
        } else {
            noc_nonposted_writes_num_issued[noc] += count;
            noc_nonposted_writes_acked[noc] += num_dests * count;
        }
    }
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid,
          bool linked, bool posted, bool set_noc_coord, bool set_addresses, bool set_size,
          bool increment_counters_flag, uint8_t cmd_buf>
FORCE_INLINE void multicast_write_set_state(uint32_t src_local_addr, uint64_t dst_noc_addr, uint32_t num_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    if constexpr (increment_counters_flag) {
        multicast_write_increment_counters<mcast_num_cores, loopback, is_part_of_receiver_grid, posted>();
    }
    auto& mc = __emule_datamovement_ctx().uk_mc[noc];
    mc.include_self = loopback;
    if constexpr (set_addresses) {
        mc.local_addr = src_local_addr;
        mc.noc_addr = dst_noc_addr;
    } else if constexpr (set_noc_coord) {
        constexpr uint64_t local_mask = (uint64_t(1) << NOC_ADDR_LOCAL_BITS) - 1;
        mc.noc_addr = (mc.noc_addr & local_mask) | (dst_noc_addr & ~local_mask);
    }
    if constexpr (set_size) {
        mc.len = num_bytes;
    }
}

// __emule_multicast_write decodes the mcast rectangle and masks the L1 offset
// with the worker-slot mask itself, so pass the captured encoded addr directly.
template <uint8_t cmd_buf>
FORCE_INLINE void multicast_write_issue_txn(uint8_t /*noc*/ = noc_index) {
    // Index by noc_index to match multicast_write_set_state's captured slot.
    auto& mc = __emule_datamovement_ctx().uk_mc[noc_index];
    __emule_multicast_write(
        mc.noc_addr, __emule_local_l1_to_ptr(mc.local_addr), mc.len,
        mc.include_self);
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid,
          bool linked, bool posted, bool set_addresses, bool set_size,
          bool wait_cmd_buf_ready, bool increment_counters_flag, uint8_t cmd_buf>
FORCE_INLINE void multicast_write_with_state(uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t num_bytes = 0) {
    if constexpr (loopback) {
        static_assert(is_part_of_receiver_grid, "Loopback mode is only supported for receiver grid");
    }
    if constexpr (increment_counters_flag) {
        multicast_write_increment_counters<mcast_num_cores, loopback, is_part_of_receiver_grid, posted>();
    }
    auto& mc = __emule_datamovement_ctx().uk_mc[noc_index];
    mc.include_self = loopback;
    if constexpr (set_addresses) {
        mc.local_addr = src_local_addr;
        __emule_uk_set_local_bits(mc.noc_addr, dst_local_addr);
    }
    if constexpr (set_size) {
        mc.len = num_bytes;
    }
    __emule_multicast_write(
        mc.noc_addr, __emule_local_l1_to_ptr(mc.local_addr), mc.len,
        mc.include_self);
}

#endif  // defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)

}  // namespace unified_kernels
