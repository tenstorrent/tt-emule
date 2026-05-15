// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Minimal ckernel stub for JIT emulation.
// ThreadId and mailbox ops are now in jit_kernel_stubs.hpp.
// CSR emulation (NEO_ID, TRISC_ID) for Quasar compute kernels.
#include <cstdint>

// TLS variables set by the kernel runner before kernel launch.
extern thread_local uint8_t __emule_neo_id;
extern thread_local uint8_t __emule_trisc_id;

namespace ckernel {
enum class CSR : uint32_t {
    NEO_ID   = 0xBC2,
    TRISC_ID = 0xBC3,
};

template <CSR csr>
inline uint32_t csr_read();

template <>
inline uint32_t csr_read<CSR::NEO_ID>() { return __emule_neo_id; }

template <>
inline uint32_t csr_read<CSR::TRISC_ID>() { return __emule_trisc_id; }
} // namespace ckernel
