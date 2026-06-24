// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// emule shadow of the SDPA SFPU header pulled in by compute_common.hpp. The bodies
// are verbatim silicon; only the include block differs: upstream pulls the raw-TTI
// ckernel_sfpu_{exp,recip,softplus}.h + sfpu/ckernel_sfpu_converter.h (lltt.h /
// sfpi intrinsics emule doesn't compile, and a second Converter definition). emule's
// api/compute/eltwise_unary/exp.h supplies the same surface — p_sfpu / addr_mod_t /
// ADDR_MOD_* / InstrModLoadStore plus sfpi.h (Converter, _ckernel_sfpu_exp_accurate_,
// _reciprocal_compat_, sfpu_reciprocal_{init,iter}, calculate_softplus_body, the
// SFPSTOCHRND modes) and ckernel_ops.h (the TTI_SFP* macro bindings).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "api/compute/eltwise_unary/exp.h"

namespace ckernel::sfpu {

constexpr auto sdpa_bits = [](float x) constexpr { return __builtin_bit_cast(std::uint32_t, x); };
constexpr auto sdpa_lo16 = [](float x) constexpr { return static_cast<std::uint16_t>(sdpa_bits(x) & 0xFFFFu); };
constexpr auto sdpa_hi16 = [](float x) constexpr { return static_cast<std::uint16_t>(sdpa_bits(x) >> 16); };

constexpr auto sdpa_addr_mod_x_instr = ADDR_MOD_3;
constexpr auto sdpa_addr_mod_x_config = ADDR_MOD_7;

ALWI void sdpa_insert_sfpnop() { TTI_SFPNOP; }

template <bool USE_SFPARECIP_INSTR, int POLY_DEGREE>
constexpr bool sdpa_can_preload_ln2_constants() {
    return false;
}

template <bool legacy_compat = true>
inline void calculate_recip_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (legacy_compat) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
            if constexpr (!(DST_ACCUM_MODE || APPROX)) {
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    } else {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat in = sfpi::dst_reg[0];
            sfpi::vFloat out;

            if constexpr (APPROX) {
                out = ckernel::sfpu::sfpu_reciprocal_iter<0>(in);
            } else if constexpr (DST_ACCUM_MODE) {
                out = ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
            } else {
                out = ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
                out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = out;
            sfpi::dst_reg += 2;
        }
    }
}

template <
    bool SCALE_EN,
    int ITERATIONS,
    bool USE_SFPARECIP_INSTR,
    int POLY_DEGREE,
    bool IS_FP32_DEST_ACC_EN,
    uint16_t SCALE_BF16>
inline void calculate_exponential_polynomial() {
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(sdpa_addr_mod_x_config);

    constexpr float LN2_RECIP = 1.44269504088896340736f;
    constexpr float M_LN2 = -0.69314718055994530942f;

    if constexpr (!USE_SFPARECIP_INSTR) {
        static_assert(POLY_DEGREE >= 1 && POLY_DEGREE <= 4);

        constexpr float c0 = (POLY_DEGREE == 1)   ? 1.03022936050163882354355235184958220293399209290987f
                             : (POLY_DEGREE == 2) ? 0.999848792924395313327307061545061386175496934006f
                             : (POLY_DEGREE == 3) ? 0.99992449655091231753798502608929170703152709521188f
                                                  : 1.0000001510806179002040134468008959160576106495165f;
        constexpr float c1 = (POLY_DEGREE == 1)   ? 1.0201394465967894800285756834161653337107187804001f
                             : (POLY_DEGREE == 2) ? 1.01508760098521056684783640695492761469306929535975f
                             : (POLY_DEGREE == 3) ? 0.99993960415029750534472970577402987498389428593233f
                                                  : 0.99996228117047652035114096488703457970402030983204f;
        constexpr float c2 = (POLY_DEGREE == 2)   ? 0.50628367056745568861842335616023694454759126020461f
                             : (POLY_DEGREE == 3) ? 0.50502329058055065591138054839814880512001604099324f
                                                  : 0.49998365704615426417337683145647067790385638465486f;
        constexpr float c3 = (POLY_DEGREE == 3) ? 0.16817330195731531429790827442800245470170482723302f
                                                : 0.16792157982882225102649214918047336097544632172075f;
        constexpr float c4 = 4.1959439860014343843000081999668024587178974865521e-2;

        if constexpr (POLY_DEGREE >= 4) {
            TTI_SFPLOADI(p_sfpu::LREG3, 0xA, sdpa_lo16(c4));
            TTI_SFPLOADI(p_sfpu::LREG3, 0x8, sdpa_hi16(c4));
        }
        if constexpr (POLY_DEGREE >= 3) {
            TTI_SFPLOADI(p_sfpu::LREG4, 0xA, sdpa_lo16(c3));
            TTI_SFPLOADI(p_sfpu::LREG4, 0x8, sdpa_hi16(c3));
        }
        if constexpr (POLY_DEGREE >= 2) {
            TTI_SFPLOADI(p_sfpu::LREG5, 0xA, sdpa_lo16(c2));
            TTI_SFPLOADI(p_sfpu::LREG5, 0x8, sdpa_hi16(c2));
        }
        if constexpr (POLY_DEGREE >= 1) {
            TTI_SFPLOADI(p_sfpu::LREG6, 0xA, sdpa_lo16(c1));
            TTI_SFPLOADI(p_sfpu::LREG6, 0x8, sdpa_hi16(c1));
            TTI_SFPLOADI(p_sfpu::LREG7, 0xA, sdpa_lo16(c0));
            TTI_SFPLOADI(p_sfpu::LREG7, 0x8, sdpa_hi16(c0));
        }
    }

    if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
        TTI_SFPLOADI(p_sfpu::LREG3, 0xA, sdpa_lo16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG3, 0x8, sdpa_hi16(LN2_RECIP));
        TTI_SFPLOADI(p_sfpu::LREG4, 0xA, sdpa_lo16(M_LN2));
        TTI_SFPLOADI(p_sfpu::LREG4, 0x8, sdpa_hi16(M_LN2));
    }

    for (int d = 0; d < ITERATIONS; d++) {
        constexpr InstrModLoadStore input_type =
            IS_FP32_DEST_ACC_EN ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B;
        TTI_SFPLOAD(p_sfpu::LREG2, input_type, sdpa_addr_mod_x_instr, 0);

        if constexpr (SCALE_EN) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
            sdpa_insert_sfpnop();
        }

        if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        } else {
            TTI_SFPLOADI(p_sfpu::LREG1, 0xA, sdpa_lo16(LN2_RECIP));
            TTI_SFPLOADI(p_sfpu::LREG1, 0x8, sdpa_hi16(LN2_RECIP));
            TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        }
        sdpa_insert_sfpnop();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPSTOCHRND_MOD1_FP32_TO_INT8);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

        if constexpr (USE_SFPARECIP_INSTR) {
            static_assert(!USE_SFPARECIP_INSTR, "TTI_SFPARECIP instruction only supported on Blackhole");
        } else {
            if constexpr (sdpa_can_preload_ln2_constants<USE_SFPARECIP_INSTR, POLY_DEGREE>()) {
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPLOADI(p_sfpu::LREG0, 0xA, sdpa_lo16(M_LN2));
                TTI_SFPLOADI(p_sfpu::LREG0, 0x8, sdpa_hi16(M_LN2));
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
            }
            sdpa_insert_sfpnop();

            if constexpr (POLY_DEGREE == 1) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 2) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else if constexpr (POLY_DEGREE == 3) {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            } else {
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
                sdpa_insert_sfpnop();
                TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
            }
            sdpa_insert_sfpnop();
        }

        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);
        sdpa_insert_sfpnop();
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT8);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        sdpa_insert_sfpnop();

        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        if constexpr (!IS_FP32_DEST_ACC_EN) {
            TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        }
        TTI_SFPSTORE(p_sfpu::LREG2, input_type, sdpa_addr_mod_x_instr, 0);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <bool SDPA_EXP_APPROX_MODE, uint16_t scale_bf16>
inline void calculate_exponential_first_column() {
    constexpr int ITERATIONS_HALF_FACE = 4;
    if constexpr (SDPA_EXP_APPROX_MODE) {
        for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
            sfpi::vFloat val = sfpi::dst_reg[0];
            sfpi::vFloat result =
                ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                    val, scale_bf16);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg += 2;
        }
    } else {
        constexpr int polynomial_degree = DST_ACCUM_MODE ? 4 : 2;
        calculate_exponential_polynomial<
            true,
            ITERATIONS_HALF_FACE,
            false,
            polynomial_degree,
            DST_ACCUM_MODE,
            scale_bf16>();
    }
}

inline void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    constexpr uint32_t prev_max_base_idx = 0;
    constexpr uint32_t worker_max_base_idx = 32;
    constexpr uint32_t cur_max_base_idx = 64;
    constexpr uint32_t prev_sum_base_idx = 96;
    constexpr uint32_t worker_sum_base_idx = 128;

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        v_if(prev_max_vec < worker_max_vec) { sfpi::dst_reg[cur_max_base_idx] = worker_max_vec; }
        v_else { sfpi::dst_reg[cur_max_base_idx] = prev_max_vec; }
        v_endif;
        sfpi::vFloat cur_max = sfpi::dst_reg[cur_max_base_idx];

        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        sfpi::vFloat exp_prev =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_worker, scale_bf16);

        sfpi::dst_reg[prev_max_base_idx] = exp_prev;
        sfpi::dst_reg[worker_max_base_idx] = exp_worker;

        sfpi::dst_reg[worker_sum_base_idx] = exp_worker * worker_sum_vec;
        sfpi::dst_reg[prev_sum_base_idx] = exp_prev * prev_sum_vec;
        sfpi::vFloat corr_worker_sum = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat corr_prev_sum = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::dst_reg[prev_sum_base_idx] = corr_worker_sum + corr_prev_sum;
        sfpi::dst_reg += 2;
    }
}

inline void calculate_softplus_first_column(uint param0, uint param1, uint param2) {
    constexpr int ITERATIONS_HALF_FACE = 4;
    float beta = ckernel::sfpu::Converter::as_float(param0);
    float beta_reciprocal = ckernel::sfpu::Converter::as_float(param1);
    float threshold = ckernel::sfpu::Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        ckernel::sfpu::calculate_softplus_body<APPROX, DST_ACCUM_MODE>(beta, beta_reciprocal, threshold);
        sfpi::dst_reg += 2;
    }
}

// ckernel_sfpu_binop_with_unary.h. BINOP_MODE selects the op; SDPA invokes it with
// ADD (softmax-denominator + 1). emule applies the scalar per-element (silicon's
// .v[i] idiom). RSUB's upstream bf16-RNE truncation correction (logit precision) is
// omitted — it needs float32_to_bf16_rne and SDPA only uses ADD.
enum { ADD = 0, SUB = 1, MUL = 2, DIV = 3, RSUB = 4 };  // BINOP_MODE
template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
inline void calculate_binop_with_scalar(uint32_t param) {
    const float parameter = ckernel::sfpu::Converter::as_float(param);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result;
        for (uint32_t i = 0; i < 32; ++i) {
            float v = val.v[i];
            if constexpr (BINOP_MODE == ADD) {
                result.v[i] = v + parameter;
            } else if constexpr (BINOP_MODE == SUB) {
                result.v[i] = v - parameter;
            } else if constexpr (BINOP_MODE == MUL) {
                result.v[i] = v * parameter;
            } else if constexpr (BINOP_MODE == DIV) {
                result.v[i] = v * parameter;
            } else if constexpr (BINOP_MODE == RSUB) {
                result.v[i] = parameter - v;
            }
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
