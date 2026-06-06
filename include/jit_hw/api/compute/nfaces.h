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

// Tile shape-aware nfaces offset.
// Silicon supports "thin" tiles with rows ∈ {1,2,4,8,16,32}, cols always = 32.
// Layout: 2 column-faces (left cols 0-15, right cols 16-31), each face is
// `rows × 16` row-major. For rows>16 (i.e. 32), there are also 2 row-faces,
// giving 4 sub-faces of 16×16 — the standard 1024-element layout.
//
// elem_idx ranges 0 .. (rows*32 - 1) in row-major order (r * 32 + c).
inline constexpr uint32_t tile_rm_to_nfaces(uint32_t elem_idx, uint32_t rows) {
    if (rows >= 32) {
        // Standard 32×32 4-face layout (cached LUT path).
        return rowmajor_to_nfaces[elem_idx];
    }
    uint32_t r = elem_idx / 32;
    uint32_t c = elem_idx % 32;
    // Two faces of `rows × 16`. Sub-row stride within a face = 16.
    uint32_t face_col = (c >= 16) ? 1u : 0u;
    uint32_t face_elems = rows * 16;  // elements per face
    return face_col * face_elems + r * 16u + (c % 16u);
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
