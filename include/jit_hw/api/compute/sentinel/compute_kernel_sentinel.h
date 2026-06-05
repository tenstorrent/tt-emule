// SPDX-License-Identifier: Apache-2.0
// Emule shadow for the silicon compute-kernel sentinel header.
// Provides Operand enum + state_configure no-op templates so kernels like
// custom_tilize.h compile under emule. Sentinel-state tracking and
// auto-injection isn't needed in emule's unified-thread model.
#pragma once

#include <cstdint>

namespace ckernel {

enum class Operand : uint8_t {
    SRCA = 0,
    SRCB = 1,
    PACK = 2,
};

}  // namespace ckernel

using ckernel::Operand;

// ALWI may not yet be defined if this header is pulled in before
// `api/compute/common.h`; fall back to `inline` to keep the shadow standalone.
#ifndef ALWI
#define ALWI inline
#endif

// state_configure template overloads. No-op under emule: the sentinel exists
// on silicon to inject reconfig calls when src/pack CB targets change, but in
// emule each kernel call reads CB metadata directly so we don't need to track
// state across calls. Silicon defines these at file scope after
// `namespace ckernel { ... }` closes, so we provide them in both global and
// ckernel namespaces — kernels that call them inside `namespace ckernel { ... }`
// (e.g. upstream custom_tilize.h) need the ckernel-namespace overloads.

namespace ckernel {
template <Operand /*operand*/ = Operand::SRCA>
inline void state_configure(uint32_t /*cb*/, uint32_t /*call_line*/) {}

template <Operand /*operand_a*/ = Operand::SRCA, Operand /*operand_b*/ = Operand::SRCB>
inline void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*call_line*/) {}

inline void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*cb_out*/, uint32_t /*call_line*/) {}
}  // namespace ckernel

template <Operand /*operand*/ = Operand::SRCA>
ALWI void state_configure(uint32_t /*cb*/, uint32_t /*call_line*/) {}

template <Operand /*operand_a*/ = Operand::SRCA, Operand /*operand_b*/ = Operand::SRCB>
ALWI void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*call_line*/) {}

ALWI void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*cb_out*/, uint32_t /*call_line*/) {}
