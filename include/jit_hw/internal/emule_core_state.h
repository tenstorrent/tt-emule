// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Per-core state (owned by tt_emule::Core; the thread ctx borrows a pointer).
//
// Split out of emule_thread_ctx.h into this minimal, dependency-free header so
// that tt_emule::Core (device.hpp) can embed a CoreState member WITHOUT pulling
// in the full per-thread ThreadCommonCtx/ComputeThreadCtx definitions. The umd
// TU (sw_emule_chip.cpp -> device.hpp) only needs CoreState; it must not have to
// parse ComputeThreadCtx, whose members reference kernel-only types (sfpi, ...).
//
// Holds per-core emule-only state (the logical coordinates). The NOC coordinate
// globals my_x/my_y are deliberately NOT here — they are silicon-named symbols
// read directly by unmodified upstream firmware/kernels, so they stay as
// runner-set thread_local globals.

#include <cstdint>

namespace tt_emule {

// Inter-RISC mailbox (ckernel::mailbox_write / mailbox_read). On silicon this is a
// small blocking RISC-to-RISC FIFO used to hand a scalar (e.g. a sparse matmul's
// per-batch is_batch_valid flag) from the in0-reader (BRISC) to the compute triad,
// keeping the reader and compute in lockstep. emule fuses UNPACK/MATH/PACK into one
// compute fiber and has no per-sub-thread identity, so we can't key by the full
// (src-thread, dst-thread) pair. Every mailbox exchange in the tree is either
// DM→compute (sparse matmul) or compute→compute (deepseek/swiglu), so we key by the
// coarse (writer_group, reader_group) ∈ {DM=0, Compute=1}² — derived from the caller's
// ThreadCommonCtx::kind and the ThreadId argument (Brisc→DM, Unpack/Math/Pack→Compute).
// One ordered ring per (writer,reader) slot; the reader parks (blocking) when empty so
// a compute fiber that reaches the read before the DM writer yields instead of spinning
// (which under emule's cooperative scheduler would be a quiescent deadlock). Lives in
// per-core CoreState (both fibers on a core share it); balanced kernels drain it to empty.
struct CoreMailbox {
    // CAP > max unread depth. The sparse matmul writes 3× per batch and can run
    // ahead over all num_experts(≤32) batches before compute catches up ⇒ ≤96; 256 is ample.
    static constexpr uint32_t CAP = 256;
    uint32_t buf[CAP] = {};
    uint32_t head = 0;  // pop cursor (monotonic; index = head % CAP)
    uint32_t tail = 0;  // push cursor (monotonic; index = tail % CAP)

    bool empty() const { return head == tail; }
    void push(uint32_t v) {
        buf[tail % CAP] = v;
        ++tail;
    }
    uint32_t pop() {
        uint32_t v = buf[head % CAP];
        ++head;
        return v;
    }
};

struct CoreState {
    uint32_t logical_x = 0;  // logical core x (D2M get_absolute_logical_x)
    uint32_t logical_y = 0;  // logical core y

    // [writer_group][reader_group], group 0 = data-movement, 1 = compute.
    CoreMailbox mbox[2][2];
};
}  // namespace tt_emule
