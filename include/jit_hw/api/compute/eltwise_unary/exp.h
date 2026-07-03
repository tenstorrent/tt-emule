// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "jit_hw/api/compute/common.h"
// Emulator stub for exponential tile operations.

#include <cmath>
#include <cstdint>

enum class InputClamping : uint8_t {
    ClampToNegative = 1,
    None = 0,
};

// VectorMode lives in ckernel::; see api/compute/vector_mode.h.
#include "jit_hw/api/compute/vector_mode.h"

// Raw-Tensix SFPU surface for kernels whose (possibly never-instantiated) branches
// reference it — e.g. SDPA's `calculate_exponential_polynomial` (exp_approx_mode=false).
// The TTI_SFP* instruction *semantics* live in the sfpi backend (`sfpi.h`'s
// `__emule_sfp_*` on the single `__emule_compute_ctx().sfpu.lreg` + `__emule_compute_ctx().sfpu.mask`) and are bound
// to the macros in `ckernel_ops.h`; only the ckernel scaffolding the kernel
// references by name lives here.
#include "jit_hw/sfpi.h"
#include "jit_hw/ckernel_ops.h"

// M_LN2: the exp polynomial declares `constexpr float M_LN2` as a local; glibc's
// <cmath> #defines M_LN2, which would mangle that declaration. Undef to restore the
// silicon (bare-metal) condition.
#ifdef M_LN2
#undef M_LN2
#endif

// p_sfpu register indices (LCONST_0 = LReg9 = 0.0, LCONST_1 = LReg10 = 1.0 — the
// read-only constant regs honored by the sfpi.h SFP* backend accessors).
namespace p_sfpu {
constexpr std::uint32_t LREG0 = 0, LREG1 = 1, LREG2 = 2, LREG3 = 3,
                        LREG4 = 4, LREG5 = 5, LREG6 = 6, LREG7 = 7;
constexpr std::uint32_t LCONST_0 = 9, LCONST_1 = 10;
}  // namespace p_sfpu

namespace ckernel {
// DEST/Src address-mode programming constants + descriptor (config-only on silicon;
// DST addressing is driven explicitly by SFPLOAD/SFPSTORE + INCRWC via the sfpi cursor).
constexpr std::uint8_t ADDR_MOD_0 = 0, ADDR_MOD_1 = 1, ADDR_MOD_2 = 2, ADDR_MOD_3 = 3,
                       ADDR_MOD_4 = 4, ADDR_MOD_5 = 5, ADDR_MOD_6 = 6, ADDR_MOD_7 = 7;
struct addr_mod_t {
    struct incr_t { std::int16_t incr = 0; };
    incr_t srca{};
    incr_t srcb{};
    incr_t dest{};
    void set(std::uint32_t /*mod_index*/) const {}
};
// SFPLOAD/SFPSTORE data-format modifier. Values mirror upstream llk_defs.h.
enum class InstrModLoadStore : std::uint8_t {
    DEFAULT       = 0,
    FP16A         = 1,
    FP16B         = 2,
    FP32          = 3,
    INT32         = 4,
    INT8          = 5,
    LO16          = 6,
    HI16          = 7,
    INT32_2S_COMP = 12,
    INT8_2S_COMP  = 13,
    LO16_ONLY     = 14,
    HI16_ONLY     = 15,
};
struct p_setrwc {
    constexpr static std::uint32_t CLR_NONE = 0x0;
    constexpr static std::uint32_t CR_D     = 0x4;
    constexpr static std::uint32_t SET_D    = 0x4;
};
// math:: addr-mode base programming — emule's hook that aims the sfpi cursor at the
// DST region for the raw-TTI direct-SFPU path (recip_tile_first_column_wh_idst0_direct).
namespace math {
inline void set_addr_mod_base() {
    ::__emule_compute_ctx().sfpu.dst_base = &__emule_compute_ctx().dst[0][0];
    ::__emule_compute_ctx().sfpu.cursor = 0;
    ::__emule_compute_ctx().sfpu.mask.fill(true);
    ::__emule_compute_ctx().sfpu.sfp_cc_active = false;  // fresh CC scope (defensive; SFPENCC clears it)
    ::__emule_compute_ctx().sfpu.first_col_mode = true;
}
inline void clear_addr_mod_base() {
    ::__emule_compute_ctx().sfpu.dst_base = nullptr;
    ::__emule_compute_ctx().sfpu.cursor = 0;
    ::__emule_compute_ctx().sfpu.first_col_mode = false;
}
}  // namespace math
}  // namespace ckernel

// SFPU functor dispatcher `_llk_math_eltwise_unary_sfpu_params_` used by SDPA's
// first-column exp/recip/softplus helpers. Functional: it points the sfpi cursor at
// __emule_compute_ctx().dst[idst], resets the cursor/active-lane mask, and invokes the functor —
// once per active face, and twice under VectorMode::C (the SDPA first-column case) so
// col-0 rows 0..31 are all covered (mirrors silicon's per-face walk). See
// llk_math_eltwise_unary_sfpu_params.h for the per-face C-iteration detail.
#include "jit_hw/llk_math_eltwise_unary_sfpu_params.h"

// p_sfpu constants used by real exp_tile callers
namespace p_sfpu {
constexpr uint16_t kCONST_1_FP16B = 0x3F80;
} // namespace p_sfpu

namespace ckernel {

// Scale baked into the SFPU exp constant by exp_tile_init's template `scale`
// parameter (an fp32 bit pattern, default 1.0f). On silicon's approx path,
// exp_init folds this into the 1/ln2 constant (A_scaled = A * scale_fp32 in
// ckernel_sfpu_exp.h:exp_init), so calculate_exponential computes exp(in * scale)
// regardless of the runtime scale_en flag. SDPA fuses the 1/sqrt(d) softmax scale
// here (exp_tile_init<true, scale_fp32, None> in compute_common.hpp). Persists
// across calls until the next init, mirroring the SFPU constant register.

// Signatures mirror current upstream api/compute/eltwise_unary/exp.h.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    float s;
    uint32_t b = scale;
    std::memcpy(&s, &b, sizeof(s));
    __emule_compute_ctx().exp_init_scale = s;
}

// exp(x * scale). The effective scale follows silicon:
//   - approx mode: the scale baked in by exp_tile_init (runtime scale_en ignored,
//     matching ckernel_sfpu_exp.h's approx path which uses the pre-loaded constant);
//   - non-approx mode: the runtime `scale` (a bf16/fp16b bit pattern) when scale_en,
//     else 1.0 — matching _ckernel_sfpu_exp_accurate_.
// vector_mode + iterations gate which faces are written (see vector_mode.h).
// input_clamping is a silicon SFPU-approximation knob with no effect on std::exp.
template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                   uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    __emule_dst_check(idst, "exp_tile");
    float s = 1.0f;
    if constexpr (approx) {
        s = __emule_compute_ctx().exp_init_scale;
    } else if constexpr (scale_en) {
        uint32_t b = static_cast<uint32_t>(scale) << 16;
        std::memcpy(&s, &b, sizeof(s));
    }
    for (uint32_t i = 0; i < __EMULE_TILE_ELEMS; i++) {
        if (!__emule_vector_mode_active(i, vector_mode, iterations)) continue;
        float r = std::exp(__emule_compute_ctx().dst[idst][i] * s);
        if constexpr (approx) {
            // Fully-masked-row degeneracy (APPROX path only — silicon's LUT/polynomial
            // exp, which makes no IEEE special-value guarantee). SDPA masks out-of-window
            // scores with bf16 -inf (0xFF80); when an ENTIRE row of a K-chunk is masked,
            // the row's running max is also -inf, so the softmax max-subtraction computes
            // exp(-inf - (-inf)) = exp(NaN) = NaN. The mathematically-correct softmax
            // numerator for a masked entry is exp(-inf)=0, and that K-chunk must contribute
            // nothing (a later in-window K-chunk — at minimum the causal diagonal —
            // supplies the row's real output, and the online-softmax correction rescales
            // this chunk's 0 away). So collapse the NaN to 0, matching silicon's valid
            // (non-NaN) output for these rows. Partially-masked rows have a finite max, so
            // exp(-inf - finite)=0 already; finite masks (e.g. -1e9) never produce -inf.
            __emule_compute_ctx().dst[idst][i] = std::isnan(r) ? 0.0f : r;
        } else {
            // Accurate path: faithful IEEE semantics (exp(NaN)=NaN, exp(+inf)=+inf,
            // exp(-inf)=0) — required by ttnn.exp fp32 special-values. The NaN→0 collapse
            // above is a property of the approx polynomial exp, not the accurate exp.
            __emule_compute_ctx().dst[idst][i] = r;
        }
    }
}

// Pack-thread variants: silicon runs the exp on the PACK thread instead of MATH.
// emule executes UNPACK/MATH/PACK inline in one thread, so these are identical to
// exp_tile{,_init} — delegate.
template <bool approx = false, uint32_t scale = 0x3F800000,
          InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_packthread_tile_init() {
    exp_tile_init<approx, scale, input_clamping>();
}

template <bool approx = false, bool scale_en = false,
          InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8>
ALWI void exp_packthread_tile(uint32_t idst, VectorMode vector_mode = VectorMode::RC,
                              uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    exp_tile<approx, scale_en, input_clamping, iterations>(idst, vector_mode, scale);
}

} // namespace ckernel
