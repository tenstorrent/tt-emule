// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Emule shim of silicon's `noc_nonblocking_api.h`. Silicon kernels (and
// shared kernel-lib headers — e.g. dataflow-utility templates that wrap
// these primitives) reference NOC counter storage, the `NocBarrierType`
// enum, `proc_type`, and `inc_noc_counter_val<>` to track NOC operations.
// Under emule, NOC ops are synchronous memcpy via
// `__emule_resolve_noc_addr`, so the counters carry no semantic load — but
// the kernel-lib headers reference them in constexpr/template contexts and
// won't compile without the surface.
//
// Storage strategy: per-thread counter arrays (`thread_local`) so each
// JIT-compiled .so gets its own definition without needing a separate
// firmware .cc. The counters are never updated — `inc/set/get_noc_counter_val<>`
// below are no-ops (get returns 0) — because emule's NOC is synchronous and
// nothing reads them back for correctness; the storage exists only so the
// kernel-lib templates that name these symbols resolve.

#include <cstdint>
#include "noc/noc_parameters.h"

// ---- Per-NOC counter storage ----
// One set per thread; emule's compute/dataflow threads each pick up their
// own. `inline` (C++17) provides a single definition per .so.
#ifndef NUM_NOCS
#define NUM_NOCS 2
#endif

inline thread_local uint32_t noc_reads_num_issued[NUM_NOCS] = {0};
inline thread_local uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] = {0};
inline thread_local uint32_t noc_nonposted_writes_acked[NUM_NOCS] = {0};
inline thread_local uint32_t noc_nonposted_atomics_acked[NUM_NOCS] = {0};
inline thread_local uint32_t noc_posted_writes_num_issued[NUM_NOCS] = {0};

// ---- NocBarrierType enum (matches upstream surface) ----
enum class NocBarrierType : uint8_t {
    READS_NUM_ISSUED,
    NONPOSTED_WRITES_NUM_ISSUED,
    NONPOSTED_WRITES_ACKED,
    NONPOSTED_ATOMICS_ACKED,
    POSTED_WRITES_NUM_ISSUED,
    COUNT
};

// ---- proc_type ----
// Silicon: chosen by COMPILE_FOR_BRISC/NCRISC/etc. branches. Under emule
// the value is only used as a template argument; the actual NOC operation
// is memcpy and doesn't read it. Provide a single value that satisfies
// the template parameter.
inline constexpr uint8_t proc_type = 0;

// ---- inc_noc_counter_val<proc_t, barrier_type>(noc, inc) ----
// No-op under emule. Returning the templates as no-ops avoids any need for
// the `get_noc_counter_address<>` plumbing silicon uses to back-target an
// L1 atomic.
template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void inc_noc_counter_val(uint32_t /*noc*/, uint32_t /*inc*/ = 1) {}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) void set_noc_counter_val(uint32_t /*noc*/, uint32_t /*val*/) {}

template <uint8_t proc_t, NocBarrierType barrier_type>
inline __attribute__((always_inline)) uint32_t get_noc_counter_val(uint32_t /*noc*/) { return 0; }

// ---- noc_cmd_buf_ready ----
// Silicon: queries the NOC command-buffer status register. Under emule
// (synchronous memcpy NOC), command buffers are always ready.
inline __attribute__((always_inline)) bool noc_cmd_buf_ready(uint32_t /*noc*/, uint32_t /*cmd_buf*/) {
    return true;
}

inline __attribute__((always_inline)) void noc_clear_outstanding_req_cnt(uint32_t /*noc*/, uint32_t /*id_mask*/) {}

// ---- Command-buffer index constants ----
// `write_at_cmd_buf` is already defined in `internal/risc_attribs.h`.
// `noc_mode` and `noc_index` are defined elsewhere in the emule chain
// (risc_common.h / dataflow_api_addrgen.h). Add only the missing ones
// that consumer kernel-lib headers reference but which haven't been
// shimmed yet.
inline constexpr uint32_t read_cmd_buf = 0;
inline constexpr uint32_t write_cmd_buf = 0;
inline constexpr uint32_t write_reg_cmd_buf = 0;

// Atomic-return-value scratch address (silicon: per-NOC L1 location for
// the result of NOC-issued atomics). Emule's NOC atomics are scalar host
// ops — no scratch needed. Provide an arbitrary address satisfying the
// uint32_t argument typing.
#ifndef MEM_NOC_ATOMIC_RET_VAL_ADDR
inline constexpr uint32_t MEM_NOC_ATOMIC_RET_VAL_ADDR = 0;
#endif

// ---- Command-buffer write/read register macros ----
// Silicon: program NOC command buffers via NOC_CMD_BUF_WRITE_REG / READ_REG
// MMIO macros. Under emule we don't actually program a NOC — these need to
// be inert at expansion time. Provide no-op variadic macros that ignore
// their arguments entirely (they expand to `((void)0)` / `(0u)` and never
// evaluate the arguments). The arguments are register IDs and address
// calculations with no side effects we need to preserve.
#ifndef NOC_CMD_BUF_WRITE_REG
#define NOC_CMD_BUF_WRITE_REG(...) ((void)0)
#endif
#ifndef NOC_CMD_BUF_READ_REG
#define NOC_CMD_BUF_READ_REG(...) (0u)
#endif
