#pragma once
// Emule shim for upstream llk_lib/llk_pack.h — pack helpers and llk_pack
// templates.

#include "internal/llk_state.h"
#include "api/compute/common.h"
#include "jit_hw/llk_types.h"  // ckernel::PackMode

// ---- Pack helpers ----

// Tilize pack: PACK row-major DST → nfaces CB at pack_offset.  Applies the
// thread-global pack-fused ReLU clamp (`__emule_pack_relu_mode` /
// `__emule_pack_relu_threshold`) before format conversion — silicon STACC_RELU
// is a single packer CFG reg, so all pack sites must honor it.
inline void __llk_pack_tiled(uint32_t tile_idx, uint32_t ocb) {
    __emule_compute::__emule_check_blockfloat_supported(ocb, "__llk_pack_tiled");
    uint8_t* buf = __emule_compute::cb_write_ptr_at(ocb, __llk_pack_offset);
    if (__emule_compute::cb_is_bfp4_b_format(ocb)) {
        // Bfp4_b: 64 face-row exponents + 512 mantissa bytes (two 4-bit elems
        // per byte). Symmetric with pack_dst_to_buf's Bfp4_b branch.
        uint8_t* exp_base  = buf;
        uint8_t* mant_base = buf + 64;
        for (uint32_t fr = 0; fr < 64; ++fr) {
            float row16[16];
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t ni = fr * 16 + k;
                const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[ni];
                row16[k] = __emule_apply_pack_relu(__emule_dst[tile_idx][rm]);
            }
            uint8_t packed[8];
            __emule_bfp4::encode_face_row(row16, exp_base[fr], packed);
            std::memcpy(&mant_base[fr * 8], packed, 8);
        }
    } else if (__emule_compute::cb_is_bfp8_b_format(ocb)) {
        // Bfp8_b: 64 face-rows of 16 floats each, shared 8-bit exponent per
        // face-row. Walk in nfaces order so each row16 corresponds to one
        // exponent byte; DST is row-major fp32 so use the inverse LUT.
        uint8_t* exp_base  = buf;
        uint8_t* mant_base = buf + 64;
        for (uint32_t fr = 0; fr < 64; ++fr) {
            float row16[16];
            for (uint32_t k = 0; k < 16; ++k) {
                const uint32_t ni = fr * 16 + k;
                const uint32_t rm = __emule_nfaces::nfaces_to_rowmajor[ni];
                row16[k] = __emule_apply_pack_relu(__emule_dst[tile_idx][rm]);
            }
            uint8_t mant_row[16];
            __emule_bfp8::encode_face_row(row16, exp_base[fr], mant_row);
            std::memcpy(&mant_base[fr * 16], mant_row, 16);
        }
    } else if (__emule_compute::cb_is_uint16_format(ocb)) {
        // uint16 output: write the low 16 bits of the DST int32 bit pattern.
        // Mirrors `pack_dst_to_buf` / `__llk_pack_untilize`; on silicon the
        // packer is reconfigured per OCB format so uint16 output is native.
        // ReLU skipped on integer output (not a coherent silicon config).
        uint16_t* out = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(ocb);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            uint32_t bits;
            std::memcpy(&bits, &__emule_dst[tile_idx][i], sizeof(uint32_t));
            out[ni] = static_cast<uint16_t>(bits);
        }
    } else if (__emule_compute::cb_is_32bit_format(ocb)) {
        uint32_t n = __emule_compute::cb_page_size(ocb) / sizeof(uint32_t);
        if (n > __EMULE_TILE_ELEMS) n = __EMULE_TILE_ELEMS;
        if (__emule_pack_relu_mode == ReluType::NO_RELU) {
            // Fast path: memcpy preserves INT32 bit patterns (denormals).
            uint32_t* out = reinterpret_cast<uint32_t*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                std::memcpy(&out[ni], &__emule_dst[tile_idx][i], sizeof(uint32_t));
            }
        } else {
            // ReLU clamp: reinterpret as float, clamp, write back.
            float* out = reinterpret_cast<float*>(buf);
            for (uint32_t i = 0; i < n; i++) {
                uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
                out[ni] = __emule_apply_pack_relu(__emule_dst[tile_idx][i]);
            }
        }
    } else {
        uint16_t* bf = reinterpret_cast<uint16_t*>(buf);
        uint32_t n = __emule_compute::cb_tile_elems(ocb);
        for (uint32_t i = 0; i < n; i++) {
            uint32_t ni = __emule_nfaces::rowmajor_to_nfaces[i];
            bf[ni] = __emule_bf16::from_f32(__emule_apply_pack_relu(__emule_dst[tile_idx][i]));
        }
    }
}

// Untilize pack: scatter DST tile into row-major output CB.  Format-aware
// dispatch mirrors `pack_dst_to_buf` in common.h — necessary because
// elem_size = page_size/1024 cannot distinguish bf16 from uint16 (both
// 2 bytes/elem) nor handle Bfp8_b (1088 bytes/tile → elem_size=1) or
// narrow-row tiles (page_size < 1024 → elem_size=0).
inline void __llk_pack_untilize(uint32_t tile_idx, uint32_t ocb) {
    constexpr uint32_t TILE_DIM = 32;
    const uint32_t block_c = __llk_pack_block_c;
    const uint32_t tile_row = __llk_pack_offset / block_c;
    const uint32_t tile_col = __llk_pack_offset % block_c;
    uint8_t* const base = __emule_compute::cb_write_ptr_at(ocb, 0);

    if (__emule_compute::cb_is_uint16_format(ocb)) {
        // uint16 output: write the low 16 bits of the DST int32 bit pattern.
        // Symmetric with `pack_dst_to_buf`'s uint16 branch (post-typecast or
        // uint16 layout move).  ReLU is skipped on int output.
        const uint32_t row_stride = block_c * TILE_DIM * sizeof(uint16_t);
        const uint32_t tile_row_offset = tile_row * TILE_DIM * row_stride;
        const uint32_t tile_col_offset = tile_col * TILE_DIM * sizeof(uint16_t);
        for (uint32_t r = 0; r < TILE_DIM; r++) {
            uint16_t* dst = reinterpret_cast<uint16_t*>(
                base + tile_row_offset + r * row_stride + tile_col_offset);
            for (uint32_t c = 0; c < TILE_DIM; c++) {
                uint32_t bits;
                std::memcpy(&bits, &__emule_dst[tile_idx][r * TILE_DIM + c], sizeof(uint32_t));
                dst[c] = static_cast<uint16_t>(bits);
            }
        }
        return;
    }
    if (__emule_compute::cb_is_32bit_format(ocb)) {
        // 32-bit (fp32 / int32 / uint32).  Without ReLU: preserve bit patterns
        // via memcpy.  With ReLU: clamp float values then write.
        const uint32_t row_stride = block_c * TILE_DIM * sizeof(uint32_t);
        const uint32_t tile_row_offset = tile_row * TILE_DIM * row_stride;
        const uint32_t tile_col_offset = tile_col * TILE_DIM * sizeof(uint32_t);
        if (__emule_pack_relu_mode == ReluType::NO_RELU) {
            for (uint32_t r = 0; r < TILE_DIM; r++) {
                uint32_t* dst = reinterpret_cast<uint32_t*>(
                    base + tile_row_offset + r * row_stride + tile_col_offset);
                for (uint32_t c = 0; c < TILE_DIM; c++)
                    std::memcpy(&dst[c], &__emule_dst[tile_idx][r * TILE_DIM + c], sizeof(uint32_t));
            }
        } else {
            for (uint32_t r = 0; r < TILE_DIM; r++) {
                float* dst = reinterpret_cast<float*>(
                    base + tile_row_offset + r * row_stride + tile_col_offset);
                for (uint32_t c = 0; c < TILE_DIM; c++)
                    dst[c] = __emule_apply_pack_relu(__emule_dst[tile_idx][r * TILE_DIM + c]);
            }
        }
        return;
    }
    // bfloat16: default 2-byte output.  Note: Bfp8_b output is not supported
    // for untilize-pack (silicon-side `untilize_helpers.inl:155` asserts
    // !is_block_float_format(pack_dst_format)) — no caller produces Bfp8_b
    // via this path.
    const uint32_t row_stride = block_c * TILE_DIM * sizeof(uint16_t);
    const uint32_t tile_row_offset = tile_row * TILE_DIM * row_stride;
    const uint32_t tile_col_offset = tile_col * TILE_DIM * sizeof(uint16_t);
    for (uint32_t r = 0; r < TILE_DIM; r++) {
        uint16_t* dst = reinterpret_cast<uint16_t*>(
            base + tile_row_offset + r * row_stride + tile_col_offset);
        for (uint32_t c = 0; c < TILE_DIM; c++)
            dst[c] = __emule_bf16::from_f32(__emule_apply_pack_relu(__emule_dst[tile_idx][r * TILE_DIM + c]));
    }
}

// 2-param version (used by tilize)
template <bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

// 3-template-param overload (used by untilize). Same (tile_idx, ocb) runtime
// args as the others; overloads differ only by template-parameter count.
template <int AccumMode, bool Untilize, bool IsApprox>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

// 1-template-param overload (used by custom_tilize / deepseek init paths).
template <int AccumMode>
inline void llk_pack(uint32_t tile_idx, uint32_t ocb) {
    if (__llk_pack_is_untilize) __llk_pack_untilize(tile_idx, ocb);
    else __llk_pack_tiled(tile_idx, ocb);
}

template <int AccumMode>
inline void llk_math_dest_section_done() {}

template <int AccumMode>
inline void llk_pack_dest_section_done() {
    __llk_pack_offset++;
}

// LLK pack init — silicon configures pack output format here; no-op stub.
template <bool Untilize, bool IsTilize, bool DiagonalEn>
inline void llk_pack_init(uint32_t /*ocb*/) {}
template <bool Untilize, bool IsTilize>
inline void llk_pack_init(uint32_t /*ocb*/) {}

// PackMode-based pack init (SDPA streaming configure_pack_width). The flags
// (zero_output / skip_addrmod_config / skip_packer_strides) are packer reconfig
// micro-optimizations — HW pipeline config with no numerical effect, so no-op.
template <ckernel::PackMode pack_mode = ckernel::PackMode::Default,
          bool zero_output = false,
          bool skip_addrmod_config = false,
          bool skip_packer_strides = false>
inline void llk_pack_init(uint32_t /*ocb*/, uint32_t /*pack_width*/ = 1) {}

// Pack-side dest sync stubs.
// llk_packer_wait_for_math_done lives in llk_sync_stubs.h.

// ---- llk_pack_rows_* : kernel-author API `pack_rows_to_addr` (idst,l1_addr)
// silicon writes DST rows directly to an absolute L1 address. Emule shadow:
// translate l1_addr to a host ptr via __emule_local_l1_to_ptr and bf16-pack
// DST row-major bytes to that address. `num_rows` (passed to _init) selects
// how many top rows of the DST tile are valid.
static thread_local uint32_t __emule_pack_rows_num = 32;
inline void llk_pack_rows_init(uint32_t num_rows) {
    // Silicon packs the top 0..32 rows of the DST tile; the pack loop below is
    // already bounded by __EMULE_TILE_ELEMS. Document the contract (no-op
    // ASSERT in JIT mode) rather than silently clamping a caller bug.
    ASSERT(num_rows <= 32);
    __emule_pack_rows_num = num_rows;
}
inline void llk_pack_rows_uninit() { __emule_pack_rows_num = 32; }
inline void _llk_pack_rows_(uint32_t idst, uint32_t l1_addr) {
    // bf16 default output. Layer-1 shadow: write `num_rows × 32` bf16 elements
    // from DST row-major into the target L1 region.
    __emule_dst_check(idst, "_llk_pack_rows_");
    uint8_t* dst_ptr = __emule_local_l1_to_ptr(l1_addr);
    uint16_t* bf = reinterpret_cast<uint16_t*>(dst_ptr);
    const uint32_t n = __emule_pack_rows_num * 32u;
    for (uint32_t i = 0; i < n && i < __EMULE_TILE_ELEMS; ++i) {
        bf[i] = __emule_bf16::from_f32(__emule_apply_pack_relu(__emule_dst[idst][i]));
    }
}
