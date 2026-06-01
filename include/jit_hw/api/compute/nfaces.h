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

} // namespace __emule_nfaces
