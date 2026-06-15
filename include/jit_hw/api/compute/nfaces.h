// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Nfaces tile layout lookup table for UNPACK/PACK engine emulation.
//
// On real hardware, the UNPACK engine reads NFACES-format tile data from L1
// and produces row-major data in SRC registers.  The PACK engine does the
// reverse: reads row-major DST and writes NFACES to L1.
//
// A 32x32 tile in NFACES format is stored as 4 sequential 16x16 faces,
// each row-major internally:
//   face 0 (offsets   0-255): rows  0-15, cols  0-15  (top-left)
//   face 1 (offsets 256-511): rows  0-15, cols 16-31  (top-right)
//   face 2 (offsets 512-767): rows 16-31, cols  0-15  (bottom-left)
//   face 3 (offsets 768-1023): rows 16-31, cols 16-31 (bottom-right)
//
// rowmajor_to_nfaces[i] gives the nfaces offset for the element at
// row-major offset i (where i = row*32 + col).
//
// Usage (UNPACK - CB to DST):  dst[i] = buf[rowmajor_to_nfaces[i]]
// Usage (PACK   - DST to CB):  buf[rowmajor_to_nfaces[i]] = dst[i]

#include <array>
#include <cstdint>

namespace __emule_nfaces {

constexpr std::array<uint32_t, 1024> make_rowmajor_to_nfaces() {
    std::array<uint32_t, 1024> lut{};
    for (uint32_t i = 0; i < 1024; i++) {
        uint32_t r = i / 32;
        uint32_t c = i % 32;
        uint32_t face = (r >= 16 ? 2u : 0u) + (c >= 16 ? 1u : 0u);
        lut[i] = face * 256 + (r % 16) * 16 + (c % 16);
    }
    return lut;
}

inline constexpr auto rowmajor_to_nfaces = make_rowmajor_to_nfaces();

// Inverse permutation: nfaces_to_rowmajor[ni] gives the row-major offset for
// the element stored at nfaces offset ni. Needed by pack paths that iterate
// in nfaces order (e.g. Bfp8_b face-row encode, which must share an exponent
// across each contiguous 16-element nfaces face-row).
constexpr std::array<uint32_t, 1024> make_nfaces_to_rowmajor() {
    std::array<uint32_t, 1024> lut{};
    for (uint32_t i = 0; i < 1024; i++) {
        lut[rowmajor_to_nfaces[i]] = i;
    }
    return lut;
}

inline constexpr auto nfaces_to_rowmajor = make_nfaces_to_rowmajor();

// General tile-shape nfaces offset: maps an active element (r, c) of a
// (tile_h × tile_w) tile to its offset in the COMPACT nfaces buffer. Each face is
// `face_r_dim × 16` row-major, faces stored back-to-back, row-major over faces.
// A narrow tile (tile_w=16) has one column-face; a partial-height tile (tile_h<16)
// has shorter faces (face_r_dim = tile_h). No padding to 16×16.
// The return value is the CB-side offset; the caller's DST/SRC buffer stays a
// 32-strided grid (index r*32 + c). Precondition: r<tile_h, c<tile_w, tile_w∈{16,32}.
inline constexpr uint32_t tile_rc_to_nfaces(uint32_t r, uint32_t c,
                                            uint32_t tile_h, uint32_t tile_w) {
    const uint32_t face_r_dim  = tile_h < 16u ? tile_h : 16u;
    const uint32_t num_faces_c = tile_w / 16u;                 // 1 or 2
    const uint32_t face_idx    = (r / face_r_dim) * num_faces_c + (c / 16u);
    return face_idx * (face_r_dim * 16u) + (r % face_r_dim) * 16u + (c % 16u);
}

// Thin-tile offset, width always 32. Thin wrapper over tile_rc_to_nfaces (single
// source of truth); rows>=32 keeps the cached-LUT fast path. elem_idx is a
// stride-32 row-major index (r*32 + c).
inline constexpr uint32_t tile_rm_to_nfaces(uint32_t elem_idx, uint32_t rows) {
    if (rows >= 32) {
        // Standard 32×32 4-face layout (cached LUT path).
        return rowmajor_to_nfaces[elem_idx];
    }
    return tile_rc_to_nfaces(elem_idx / 32u, elem_idx % 32u, rows, 32u);
}

// Inverse of tile_rc_to_nfaces: compact nfaces offset → DST stride-32 index
// (r*32 + c). Used by the block-float pack path, which iterates compact face-rows
// and must scatter each element back to its 32-strided DST position.
inline constexpr uint32_t tile_nfaces_to_rm(uint32_t ni, uint32_t tile_h, uint32_t tile_w) {
    const uint32_t face_r_dim  = tile_h < 16u ? tile_h : 16u;
    const uint32_t num_faces_c = tile_w / 16u;
    const uint32_t face_size   = face_r_dim * 16u;
    const uint32_t face_idx    = ni / face_size;
    const uint32_t within      = ni % face_size;
    const uint32_t r = (face_idx / num_faces_c) * face_r_dim + within / 16u;
    const uint32_t c = (face_idx % num_faces_c) * 16u + within % 16u;
    return r * 32u + c;
}

// Block-float exponent layout. One shared exponent per 16-wide face-row, so the
// count is tile_h * (tile_w/16) (= 64 for a full 32×32 tile). The mantissa
// section follows the exponents, L1-aligned to 16 bytes.
inline constexpr uint32_t tile_num_exp(uint32_t tile_h, uint32_t tile_w) {
    return tile_h * (tile_w / 16u);
}
inline constexpr uint32_t tile_bfp_mant_offset(uint32_t tile_h, uint32_t tile_w) {
    return (tile_num_exp(tile_h, tile_w) + 15u) & ~15u;  // round up to 16
}

// Compute rows-per-tile from CB page size + element byte width.
// page_size = rows * 32 * elem_bytes  →  rows = page_size / (32 * elem_bytes)
inline constexpr uint32_t tile_rows_from_pagesize(uint32_t page_size_bytes, uint32_t elem_bytes) {
    if (elem_bytes == 0) return 32;
    uint32_t rows = page_size_bytes / (32u * elem_bytes);
    if (rows == 0) return 1u;
    if (rows > 32) return 32u;
    return rows;
}

} // namespace __emule_nfaces
