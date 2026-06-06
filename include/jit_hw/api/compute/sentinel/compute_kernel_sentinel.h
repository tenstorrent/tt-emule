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

// state_configure template overloads. No-op under emule: the sentinel exists
// on silicon to inject reconfig calls when src/pack CB targets change, but in
// emule each kernel call reads CB metadata directly so we don't need to track
// state across calls.
//
// Define the overloads ONCE in `namespace ckernel` and re-export them to global
// scope with a `using` declaration. Defining them in both namespaces would make
// unqualified calls ambiguous wherever `using namespace ckernel;` is active
// (pulled in by api/compute/common.h), and even `ckernel::state_configure(...)`
// could clash with `::state_configure` via enclosing-namespace lookup. Kernels
// that call these inside `namespace ckernel { ... }` (e.g. upstream
// custom_tilize.h) resolve to the ckernel definitions; kernels that call them at
// file scope resolve through the `using` re-export.

namespace ckernel {
template <Operand /*operand*/ = Operand::SRCA>
inline void state_configure(uint32_t /*cb*/, uint32_t /*call_line*/) {}

template <Operand /*operand_a*/ = Operand::SRCA, Operand /*operand_b*/ = Operand::SRCB>
inline void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*call_line*/) {}

inline void state_configure(uint32_t /*cb_a*/, uint32_t /*cb_b*/, uint32_t /*cb_out*/, uint32_t /*call_line*/) {}
}  // namespace ckernel

using ckernel::state_configure;
