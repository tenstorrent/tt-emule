// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// sfpi:: emule shim — 32-lane SIMD model executed scalar per lane.
//
// Silicon's sfpi is a vector intrinsic library; sfpi::vFloat is a
// 32-element SIMD register held in an SFPU LReg, and `v_if(cond) { ... }
// v_endif;` is per-lane predicated execution against an implicit lane
// mask stack. emule has no vector hardware, so we model each vector as
// a 32-element std::array<T, 32> and step through lanes with a scalar
// loop, gated by a thread-local lane-mask stack.
//
// dst_reg is the most subtle part: silicon's `sfpi::dst_reg[k]` is a
// per-lane reference into a 32-lane slice of the DST tile; `dst_reg++`
// advances the implicit "current slice" by one 32-lane group. emule
// emulates this with a thread-local cursor (`__emule_sfpi_cursor`)
// indexing into `__emule_dst[__emule_sfpi_idst][cursor..cursor+32]`.
//
// Iteration pattern (matches upstream sfpu kernels like calculate_clamped_silu_gate):
//   for (d = 0; d < ITERATIONS; d++) {
//       sfpi::vFloat x = sfpi::dst_reg[0];   // load 32 lanes from cursor
//       ... per-lane math ...
//       sfpi::dst_reg[0] = result;            // store 32 lanes
//       sfpi::dst_reg++;                       // cursor += 32
//   }
//
// Limitations:
//   - Sigmoid/exp/reciprocal use libm (`expf`, `1/x`); silicon SFPU
//     uses LUT+polynomial. Should match within bf16 precision (~1 ULP)
//     for the relevant input ranges of upstream ops.
//   - DST format dispatch: emule's __emule_dst is always float32. Ops
//     that bit-cast DST to int (e.g. topk's index extraction) are
//     handled via reinterpret<vInt>; the underlying storage is still
//     float32, so we round-trip through the float bit pattern.

#include <cstdint>
#include <cmath>
#include <cstring>
#include <array>
#include <limits>
#include <cstdio>
#include <cstdlib>

// Real silicon ckernel_sfpu_*.h headers (pulled in by the deep-SFPU path)
// mark functions with `sfpi_inline`. On RISC-V this maps to an
// always-inline attribute; on the x86 emule shim plain `inline` suffices.
#ifndef sfpi_inline
#define sfpi_inline inline
#endif

// `__emule_dst` is defined in api/compute/common.h as `static thread_local`
// (one copy per TU). emule's JIT model is single-TU per kernel, so accessing
// it from this header inside a TU that also includes common.h works — but
// sfpi.h is sometimes included BEFORE common.h. Workaround: declare a
// thread-local cursor pointer that callers (or common.h) set to
// &__emule_dst[idst][0]. The DstReg accessors use this pointer.
static constexpr uint32_t __EMULE_SFPI_LANES = 32;
static constexpr uint32_t __EMULE_SFPI_TILE_ELEMS = 1024;

// Pointer to the active DST tile's elements (set on tile_regs_acquire/wait
// boundaries, or implicitly by `__emule_sfpi_set_idst()`). Null until set.
inline thread_local float* __emule_sfpi_dst_base = nullptr;
// 32-lane window cursor: which 32-element slice of the active DST tile
// the next sfpi::dst_reg[k] load/store will hit.
inline thread_local uint32_t __emule_sfpi_cursor = 0;

namespace sfpi {

// ---- Rounding/instruction-modifier enums (used by reinterpret-style helpers) ----

enum class RoundMode : uint8_t {
    None = 0,
    Default = 1,
    Stochastic = 2,
    NearestEven = 3,
    RoundToNearestEven = 3,  // alias
};

// SFPSHFT2 sub-vector shuffle modifiers — silicon: per-lane crossbar
// shuffle. Emule treats these as opaque constants (consumed only by
// deep TTI_SFP intrinsics, which are out-of-scope for the scalar shim).
inline constexpr uint32_t SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 0;

// Symbolic LReg index — silicon: which SFPU LReg holds the value.
// Emule: just an opaque tag; backed-by-LReg storage lives in `l_reg[]`.
enum class LRegs : uint8_t {
    LReg0 = 0, LReg1 = 1, LReg2 = 2, LReg3 = 3,
    LReg4 = 4, LReg5 = 5, LReg6 = 6, LReg7 = 7,
    // LReg8..15 back the full SFPU register file (storage is __emule_lreg[16]);
    // LReg11..14 are the programmable-constant slots (also exposed as the
    // vConstFloatPrgm*/vConstIntPrgm* views).
    LReg8 = 8, LReg9 = 9, LReg10 = 10, LReg11 = 11,
    LReg12 = 12, LReg13 = 13, LReg14 = 14, LReg15 = 15,
};

// ---- 32-lane vector types ----

class vFloat;
class vInt;
class vUInt;

class vFloat {
public:
    std::array<float, __EMULE_SFPI_LANES> v{};
    constexpr vFloat() = default;
    constexpr vFloat(float x) { for (auto& lane : v) lane = x; }
    // Construct from int treats arg as a float (literal 0/1 commonly).
    constexpr vFloat(int x) : vFloat(static_cast<float>(x)) {}
    // Real sfpi kernels use double literals (e.g. `x < 0.0`); without this
    // ctor `vFloat`-vs-double overloads are ambiguous (int vs float ctor).
    constexpr vFloat(double x) : vFloat(static_cast<float>(x)) {}

    // Lane-masked copy assignment: an sfpi local is a register, so a write
    // inside v_if(...) updates only active lanes. Defined out-of-line below
    // (__emule_sfpi_mask is declared later in this header).
    vFloat(const vFloat&) = default;
    vFloat& operator=(const vFloat& o);

    vFloat& operator+=(const vFloat& o) { for (uint32_t i = 0; i < 32; ++i) v[i] += o.v[i]; return *this; }
    vFloat& operator-=(const vFloat& o) { for (uint32_t i = 0; i < 32; ++i) v[i] -= o.v[i]; return *this; }
    vFloat& operator*=(const vFloat& o) { for (uint32_t i = 0; i < 32; ++i) v[i] *= o.v[i]; return *this; }
};

class vInt {
public:
    std::array<int32_t, __EMULE_SFPI_LANES> v{};
    constexpr vInt() = default;
    constexpr vInt(int32_t x) { for (auto& lane : v) lane = x; }
    constexpr vInt(uint32_t x) { for (auto& lane : v) lane = static_cast<int32_t>(x); }
    // Lane-masked copy assignment (see vFloat note).
    vInt(const vInt&) = default;
    vInt& operator=(const vInt& o);

    vInt& operator+=(const vInt& o) { for (uint32_t i = 0; i < 32; ++i) v[i] += o.v[i]; return *this; }
    vInt& operator-=(const vInt& o) { for (uint32_t i = 0; i < 32; ++i) v[i] -= o.v[i]; return *this; }
    vInt& operator&=(const vInt& o) { for (uint32_t i = 0; i < 32; ++i) v[i] &= o.v[i]; return *this; }
    vInt& operator|=(const vInt& o) { for (uint32_t i = 0; i < 32; ++i) v[i] |= o.v[i]; return *this; }
};

class vUInt {
public:
    std::array<uint32_t, __EMULE_SFPI_LANES> v{};
    constexpr vUInt() = default;
    constexpr vUInt(uint32_t x) { for (auto& lane : v) lane = x; }
    constexpr vUInt(int32_t x) { for (auto& lane : v) lane = static_cast<uint32_t>(x); }
    // Lane-masked copy assignment (see vFloat note).
    vUInt(const vUInt&) = default;
    vUInt& operator=(const vUInt& o);
};

// ---- Fidelity knob: centralized SFPU result finalization ----
//
// Default is OFF — coverage-first, plain IEEE FP32, byte-identical baseline.
// Define EMULE_SFPU_BITEXACT to apply Blackhole SFPU numeric semantics on every
// vFloat ALU result (per tt-isa-documentation SFPMAD): denormal outputs flushed
// to sign-preserved zero, and any NaN canonicalized to 0x7fc00000. This single
// chokepoint is where the partially-fused FMA (a*b+c kept above FP32 then a
// single round-to-nearest-even — a port of the ISA Miscellaneous/FMA/fma.c)
// drops in later, so ops never change. Compiles to identity when the macro is
// unset (zero overhead, no baseline drift).
inline float __emule_sfpu_finalize(float x) {
#if defined(EMULE_SFPU_BITEXACT)
    uint32_t b; std::memcpy(&b, &x, 4);
    const uint32_t exp = (b >> 23) & 0xFFu, man = b & 0x7FFFFFu;
    if (exp == 0xFFu && man) { b = 0x7FC00000u; std::memcpy(&x, &b, 4); }       // NaN -> canonical
    else if (exp == 0u && man) { b &= 0x80000000u; std::memcpy(&x, &b, 4); }    // denormal -> signed 0
#endif
    return x;
}

// Partially-fused multiply-add (single rounding), matching SFPMAD's contract.
// C++ operator chains (a*b + c) round twice; ops that need the fused result
// call this explicitly. Provided as the faithful primitive + the place the
// bit-exact FMA model lands. (Internal name to avoid clashing with platform
// `mad` identifiers; the real sfpi exposes SFPMAD via operators/builtins.)
inline vFloat __emule_sfpu_mad(const vFloat& a, const vFloat& b, const vFloat& c) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(std::fmaf(a.v[i], b.v[i], c.v[i]));
    return r;
}

// ---- Arithmetic and comparison ops ----

inline vFloat operator+(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(a.v[i] + b.v[i]); return r; }
inline vFloat operator-(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(a.v[i] - b.v[i]); return r; }
inline vFloat operator*(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(a.v[i] * b.v[i]); return r; }
inline vFloat operator/(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(a.v[i] / b.v[i]); return r; }
inline vFloat operator-(const vFloat& a) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(-a.v[i]); return r; }

inline vInt operator+(const vInt& a, const vInt& b) { vInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] + b.v[i]; return r; }
inline vInt operator-(const vInt& a, const vInt& b) { vInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] - b.v[i]; return r; }
inline vInt operator&(const vInt& a, const vInt& b) { vInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] & b.v[i]; return r; }
inline vInt operator|(const vInt& a, const vInt& b) { vInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] | b.v[i]; return r; }
inline vInt operator-(const vInt& a) { vInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = -a.v[i]; return r; }

// Per-lane condition (mask) — produced by comparison operators, consumed by v_if.
struct vCond {
    std::array<bool, __EMULE_SFPI_LANES> mask{};
};

inline vCond operator>(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] > b.v[i]; return c; }
inline vCond operator<(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] < b.v[i]; return c; }
inline vCond operator>=(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] >= b.v[i]; return c; }
inline vCond operator<=(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] <= b.v[i]; return c; }
inline vCond operator==(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] == b.v[i]; return c; }
inline vCond operator!=(const vFloat& a, const vFloat& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] != b.v[i]; return c; }

inline vCond operator>(const vInt& a, const vInt& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] > b.v[i]; return c; }
inline vCond operator<(const vInt& a, const vInt& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] < b.v[i]; return c; }
inline vCond operator==(const vInt& a, const vInt& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] == b.v[i]; return c; }
inline vCond operator!=(const vInt& a, const vInt& b) { vCond c; for (uint32_t i = 0; i < 32; ++i) c.mask[i] = a.v[i] != b.v[i]; return c; }

// Logical combinators — silicon emits per-lane bitwise ops on the mask
// stack; emule does the same on the vCond mask arrays.
inline vCond v_and(const vCond& a, const vCond& b) { vCond r; for (uint32_t i = 0; i < 32; ++i) r.mask[i] = a.mask[i] && b.mask[i]; return r; }
inline vCond v_or(const vCond& a, const vCond& b) { vCond r; for (uint32_t i = 0; i < 32; ++i) r.mask[i] = a.mask[i] || b.mask[i]; return r; }
inline vCond v_not(const vCond& a) { vCond r; for (uint32_t i = 0; i < 32; ++i) r.mask[i] = !a.mask[i]; return r; }

// ---- Predication stack (v_if / v_elseif / v_else / v_endif) ----

// Active lane mask — initially all-true. v_if pushes a new mask = (current AND cond).
inline thread_local std::array<bool, __EMULE_SFPI_LANES> __emule_sfpi_mask = {
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
    true, true, true, true, true, true, true, true,
};

struct __MaskFrame {
    std::array<bool, __EMULE_SFPI_LANES> outer;   // outer scope mask
    std::array<bool, __EMULE_SFPI_LANES> taken;   // OR of all branch conds matched so far
};

inline thread_local __MaskFrame __emule_sfpi_frames[16];
inline thread_local int __emule_sfpi_frame_depth = 0;

inline void __emule_sfpi_push_if(const vCond& c) {
    auto& f = __emule_sfpi_frames[__emule_sfpi_frame_depth++];
    f.outer = __emule_sfpi_mask;
    for (uint32_t i = 0; i < 32; ++i) {
        f.taken[i] = c.mask[i];
        __emule_sfpi_mask[i] = f.outer[i] && c.mask[i];
    }
}

inline void __emule_sfpi_elseif(const vCond& c) {
    auto& f = __emule_sfpi_frames[__emule_sfpi_frame_depth - 1];
    for (uint32_t i = 0; i < 32; ++i) {
        bool fresh = c.mask[i] && !f.taken[i];
        __emule_sfpi_mask[i] = f.outer[i] && fresh;
        f.taken[i] = f.taken[i] || c.mask[i];
    }
}

inline void __emule_sfpi_else() {
    auto& f = __emule_sfpi_frames[__emule_sfpi_frame_depth - 1];
    for (uint32_t i = 0; i < 32; ++i) {
        __emule_sfpi_mask[i] = f.outer[i] && !f.taken[i];
    }
}

inline void __emule_sfpi_endif() {
    auto& f = __emule_sfpi_frames[--__emule_sfpi_frame_depth];
    __emule_sfpi_mask = f.outer;
}

// ---- dst_reg: thread-local cursor into __emule_sfpi_dst_base ----
//
// __emule_sfpi_dst_base points to the active DST tile's element-0;
// __emule_sfpi_cursor is the 32-lane window offset (advanced by dst_reg++).
// If callers forget to set the base, the shim falls back to a per-thread
// scratch tile so reads/writes don't segfault.

inline thread_local float __emule_sfpi_dst_fallback[__EMULE_SFPI_TILE_ELEMS] = {};

inline float* __emule_sfpi_active_dst() {
    return __emule_sfpi_dst_base ? __emule_sfpi_dst_base : __emule_sfpi_dst_fallback;
}

// SFPU lane i (0..31) at DST address `addr` → row-major index into the 32x32 tile.
// Silicon (WH B0, confirmed via DeepWiki ISA): one SFPLOAD/SFPSTORE covers a
// 4-row x 8-column block. `addr & ~3` selects a 4-row block (face-major row),
// `addr & 2` selects the even/odd column half, bit 0 is unused. Lane layout:
//   row_face = (addr & ~3) + i/8   (4 consecutive rows)
//   col      = (i & 7)*2 + (addr&2 ? 1 : 0)   (8 even-or-odd cols of a 16-wide face)
// emule's __emule_dst is row-major 32x32, while the SFPU addresses face-major, so
// map the face (face-major-row/16) back to the row-major quadrant. sfpi's stride
// unit is SFP_DESTREG_STRIDE=2, so dst_reg[k] uses addr = cursor + 2k and
// dst_reg += n advances the cursor by 2n (see DstReg below).
inline uint32_t __emule_sfpi_lane_index(uint32_t addr, uint32_t i) {
    const uint32_t row_face = (addr & ~3u) + (i >> 3);            // global face-major row
    // Large dst_reg[k] cross DST-tile boundaries (e.g. the fused softmax reads
    // worker_max/sum from neighbouring DST tiles via dst_reg[32]/[64]/...). Each
    // 32x32 tile spans 64 face-major rows = __EMULE_SFPI_TILE_ELEMS contiguous
    // elements in __emule_dst, so split off the tile then map within it.
    const uint32_t tile     = row_face / 64u;                     // tile offset from base
    const uint32_t frt      = row_face % 64u;                     // face-major row in tile 0..63
    const uint32_t col      = ((i & 7u) << 1) + ((addr & 2u) ? 1u : 0u);  // col 0..15 in face
    const uint32_t face     = frt >> 4;                           // 0..3
    const uint32_t face_row = frt & 15u;                          // 0..15
    const uint32_t R = ((face >> 1) << 4) + face_row;             // row-major row 0..31
    const uint32_t C = ((face & 1u) << 4) + col;                  // row-major col 0..31
    return tile * __EMULE_SFPI_TILE_ELEMS + (R << 5) + C;
}

class DstRegLane {
    uint32_t addr_delta_;  // DST-address delta (2*k for dst_reg[k]); added to cursor
public:
    constexpr DstRegLane(uint32_t d) : addr_delta_(d) {}

    // Load 32 lanes from DST into a vFloat.
    operator vFloat() const {
        vFloat r;
        float* base = __emule_sfpi_active_dst();
        const uint32_t addr = __emule_sfpi_cursor + addr_delta_;
        for (uint32_t i = 0; i < 32; ++i) {
            r.v[i] = base[__emule_sfpi_lane_index(addr, i)];
        }
        return r;
    }
    // Load 32 lanes as int (bit-cast).
    operator vInt() const {
        vInt r;
        float* base = __emule_sfpi_active_dst();
        const uint32_t addr = __emule_sfpi_cursor + addr_delta_;
        for (uint32_t i = 0; i < 32; ++i) {
            float f = base[__emule_sfpi_lane_index(addr, i)];
            int32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            r.v[i] = bits;
        }
        return r;
    }

    // Store with active-lane mask. The SFPSTORE boundary is a fidelity
    // chokepoint: under EMULE_SFPU_BITEXACT each stored lane is finalized
    // (denormal flush + NaN canonicalize). DEST-format mantissa truncation
    // (bf16/fp16 on store) is the future upgrade that hangs here — emule's DST
    // is fp32 today and deep ops convert<vFloat16b> before store, so it is not
    // yet wired. Identity when the macro is unset (byte-identical baseline).
    DstRegLane& operator=(const vFloat& x) {
        float* base = __emule_sfpi_active_dst();
        const uint32_t addr = __emule_sfpi_cursor + addr_delta_;
        for (uint32_t i = 0; i < 32; ++i) {
            if (__emule_sfpi_mask[i]) {
                // #155 finalize (value fidelity) + prefill first-column lane mapping (addressing).
                base[__emule_sfpi_lane_index(addr, i)] = __emule_sfpu_finalize(x.v[i]);
            }
        }
        return *this;
    }
    DstRegLane& operator=(const vInt& x) {
        float* base = __emule_sfpi_active_dst();
        const uint32_t addr = __emule_sfpi_cursor + addr_delta_;
        for (uint32_t i = 0; i < 32; ++i) {
            if (__emule_sfpi_mask[i]) {
                int32_t bits = x.v[i];
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                base[__emule_sfpi_lane_index(addr, i)] = f;
            }
        }
        return *this;
    }
    DstRegLane& operator=(float x) { return operator=(vFloat(x)); }
};

class DstReg {
public:
    // dst_reg[k] addresses DST Addr = cursor + 2k (SFP_DESTREG_STRIDE = 2).
    DstRegLane operator[](int i) const { return DstRegLane(2u * static_cast<uint32_t>(i)); }
    // dst_reg++ / dst_reg += n advance the DST RWC by 2 / 2n.
    DstReg& operator++() { __emule_sfpi_cursor += 2; return *this; }
    DstReg operator++(int) { DstReg t = *this; __emule_sfpi_cursor += 2; return t; }
    DstReg& operator+=(int n) { __emule_sfpi_cursor += 2u * static_cast<uint32_t>(n); return *this; }
};

inline DstReg dst_reg;

// l_reg[LRegs::LRegN] — silicon: direct access to an SFPU LReg. Emule backs
// LReg0..15 with thread-local 32-lane bit storage (vUInt). The coefficient
// LUTs (tanh/sigmoid) load packed FP16/INT bit patterns into LRegs via SFPLOADI
// (see the jit_hw ckernel_ops.h shim), then lut()/lut2() read them. l_reg is
// vUInt-typed — the only emule users (the LUT ops) treat it as raw bits.
inline thread_local vUInt __emule_lreg[16] = {};
struct LRegFile {
    vUInt& operator[](LRegs idx) { return __emule_lreg[static_cast<uint8_t>(idx)]; }
    const vUInt& operator[](LRegs idx) const { return __emule_lreg[static_cast<uint8_t>(idx)]; }
};
inline LRegFile l_reg;

// SFPLOADI into an LReg (called by the jit_hw ckernel_ops.h shim). The immediate
// is uniform across lanes. insmod per ckernel_sfpu_load_config.h:
//   2  -> imm16 unsigned, zero-extended (clears upper 16)
//   10 -> write lower 16, keep upper;  8 -> write upper 16, keep lower
inline void __emule_sfploadi(unsigned dest, unsigned insmod, unsigned val) {
    if (dest >= 16) return;
    uint32_t cur = __emule_lreg[dest].v[0];
    if (insmod == 2)       cur = val & 0xFFFFu;
    else if (insmod == 10) cur = (cur & 0xFFFF0000u) | (val & 0xFFFFu);
    else if (insmod == 8)  cur = (cur & 0x0000FFFFu) | ((val & 0xFFFFu) << 16);
    else                   cur = val;
    for (uint32_t i = 0; i < 32; ++i) __emule_lreg[dest].v[i] = cur;
}

// ---- Constants ----

inline const vFloat vConst0{0.0f};
inline const vFloat vConst1{1.0f};
inline const vFloat vConstNeg1{-1.0f};
inline const vFloat vConst0p5{0.5f};
inline const vFloat vConstNeg0p5{-0.5f};

// ---- Reinterpret (bit-cast between vFloat / vInt / vUInt) ----

template <typename Dst, typename Src>
inline Dst reinterpret(const Src& s) {
    Dst d;
    static_assert(sizeof(d.v[0]) == sizeof(s.v[0]), "sfpi::reinterpret: lane size mismatch");
    for (uint32_t i = 0; i < 32; ++i) {
        std::memcpy(&d.v[i], &s.v[i], sizeof(d.v[0]));
    }
    return d;
}

// ---- Bit-format conversions ----

// Per-lane bfloat16 truncation. Used by upstream kernels to round results
// before writeback when DST is in bf16 mode.
inline vFloat float_to_fp16b(const vFloat& f, RoundMode = RoundMode::NearestEven) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &f.v[i], sizeof(bits));
        // Round to nearest even, then mask low 16 bits.
        uint32_t rounded = bits + 0x7FFF + ((bits >> 16) & 1);
        rounded &= 0xFFFF0000;
        std::memcpy(&r.v[i], &rounded, sizeof(r.v[i]));
    }
    return r;
}
// Scalar (legacy) overloads — kept for kernels that pass a raw float.
inline uint16_t float_to_fp16b(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}
inline uint16_t float_to_fp16a(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>((bits >> 16) ^ ((bits >> 31) << 15));
}

// vFloat16b + convert<vFloat16b> come from #155's deep-SFPU backend below (single
// copy); the SDPA recip path's convert<vFloat16b> resolves to it.

// ---- Sign manipulation ----

inline vFloat setsgn(const vFloat& x, int sgn) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &x.v[i], sizeof(bits));
        bits = (bits & 0x7FFFFFFF) | (static_cast<uint32_t>(sgn & 1) << 31);
        std::memcpy(&r.v[i], &bits, sizeof(r.v[i]));
    }
    return r;
}

// ---- Converter helpers used by topk kernels (eps/scale as float bits) ----

struct Converter {
    static vFloat as_float(uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return vFloat(f);
    }
};

// ===========================================================================
// Deep-SFPU backend (faithful sfpi so real silicon ckernel_sfpu_*.h runs).
// See docs/sfpu-deep-path.md. Validated by the Phase-0 sqrt spike.
// ===========================================================================

// ---- Lane-masked register assignment (the substance of TODO #102) ----
// On silicon an sfpi local IS an SFPU register; a write inside v_if(...)
// updates only active lanes. The dst_reg store path is already masked;
// these make plain vReg-to-vReg assignment masked too. (Defined here, after
// __emule_sfpi_mask is declared above.)
inline vFloat& vFloat::operator=(const vFloat& o) {
    for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) v[i] = o.v[i];
    return *this;
}
inline vInt& vInt::operator=(const vInt& o) {
    for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) v[i] = o.v[i];
    return *this;
}
inline vUInt& vUInt::operator=(const vUInt& o) {
    for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) v[i] = o.v[i];
    return *this;
}

// ---- vUInt logical operators ----
inline vUInt operator+(const vUInt& a, const vUInt& b) { vUInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] + b.v[i]; return r; }
inline vUInt operator-(const vUInt& a, const vUInt& b) { vUInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] - b.v[i]; return r; }
inline vUInt operator>>(const vUInt& a, int s) { vUInt r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] >> s; return r; }
inline vUInt& operator>>=(vUInt& a, int s) { for (uint32_t i = 0; i < 32; ++i) a.v[i] >>= s; return a; }

// ---- vCond && (real sfpi overloads && on the lane mask) ----
inline vCond operator&&(const vCond& a, const vCond& b) { return v_and(a, b); }

// ---- Exponent/mantissa field ops (SFPEXEXP / SFPSETEXP / SFPDIVP2) ----
enum class ExponentMode { Debias, NoDebias };

inline vInt exexp(const vFloat& vf, ExponentMode mode = ExponentMode::Debias) {
    vInt r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b; std::memcpy(&b, &vf.v[i], 4);
        int e = static_cast<int>((b >> 23) & 0xFF);
        r.v[i] = (mode == ExponentMode::Debias) ? (e - 127) : e;
    }
    return r;
}
// setexp writes the raw biased exponent field (matches __builtin_rvtt_sfpsetexp).
inline vFloat setexp(const vFloat& vf, int exp) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b; std::memcpy(&b, &vf.v[i], 4);
        b = (b & 0x807FFFFFu) | ((static_cast<uint32_t>(exp & 0xFF)) << 23);
        std::memcpy(&r.v[i], &b, 4);
    }
    return r;
}
inline vFloat setexp(const vFloat& vf, const vInt& exp) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b; std::memcpy(&b, &vf.v[i], 4);
        b = (b & 0x807FFFFFu) | ((static_cast<uint32_t>(exp.v[i] & 0xFF)) << 23);
        std::memcpy(&r.v[i], &b, 4);
    }
    return r;
}
// addexp(in, e) = in * 2^e  (SFPDIVP2 ADD on the exponent field).
inline vFloat addexp(const vFloat& in, int exp) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = __emule_sfpu_finalize(std::ldexp(in.v[i], exp));
    return r;
}

// Bitwise NOT on vInt (SFPNOT).
inline vInt operator~(const vInt& a) {
    vInt r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = ~a.v[i];
    return r;
}
// setsgn on a vInt's bit pattern: set bit 31 to `sgn` (produces sign-magnitude).
inline vInt setsgn(const vInt& v, int sgn) {
    vInt r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b = static_cast<uint32_t>(v.v[i]);
        b = (b & 0x7FFFFFFFu) | (static_cast<uint32_t>(sgn & 1) << 31);
        r.v[i] = static_cast<int32_t>(b);
    }
    return r;
}
// int32_to_float (SFPCAST IntFloat): the hardware interprets the input as a
// 32-bit SIGN-MAGNITUDE integer, not two's complement. RoundMode is accepted
// but emule uses host round-to-nearest. (Callers that need a negative result
// pass a sign-magnitude value, e.g. via setsgn — see ckernel_sfpu_log.h.)
inline vFloat int32_to_float(const vInt& in, RoundMode = RoundMode::NearestEven) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b = static_cast<uint32_t>(in.v[i]);
        float mag = static_cast<float>(b & 0x7FFFFFFFu);
        r.v[i] = (b & 0x80000000u) ? -mag : mag;
    }
    return r;
}
// abs(vFloat): clear the sign bit (SFPABS float mode).
inline vFloat abs(const vFloat& v) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        uint32_t b; std::memcpy(&b, &v.v[i], 4);
        b &= 0x7FFFFFFFu;
        std::memcpy(&r.v[i], &b, 4);
    }
    return r;
}
// vec_min_max(a,b): a <- min(a,b), b <- max(a,b)  (SFPSWAP min/max; used to
// clamp, e.g. ckernel_sfpu_exp.h). vec_max_min is the swapped variant.
// Honors the active lane mask (a silicon SFPSWAP inside v_if is predicated).
inline void vec_min_max(vFloat& a, vFloat& b) {
    for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) {
        float lo = a.v[i] < b.v[i] ? a.v[i] : b.v[i];
        float hi = a.v[i] < b.v[i] ? b.v[i] : a.v[i];
        a.v[i] = lo; b.v[i] = hi;
    }
}
inline void vec_max_min(vFloat& a, vFloat& b) {
    for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) {
        float hi = a.v[i] > b.v[i] ? a.v[i] : b.v[i];
        float lo = a.v[i] > b.v[i] ? b.v[i] : a.v[i];
        a.v[i] = hi; b.v[i] = lo;
    }
}

// ---- LUT piecewise-linear evaluators (SFPLUT / SFPLUTFP32) ----
// Coefficient decoders (exact per tt-isa-documentation Blackhole).
inline float __lut8_to_fp32(uint8_t x) {
    if (x == 0xFF) return 0.0f;                       // 0xFF maps to 0
    uint32_t sign = x >> 7, exp = (x >> 4) & 7u, man = x & 0xFu;
    uint32_t bits = (sign << 31) | ((127u - exp) << 23) | (man << 19);
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline float __lut16_to_fp32(uint16_t x) {
    uint32_t sign = x >> 15, exp = (x >> 10) & 0x1Fu, man = x & 0x3FFu;
    uint32_t bits = (sign << 31) | ((exp == 0x1Fu ? 0u : 112u + exp) << 23) | (man << 13);
    float f; std::memcpy(&f, &bits, 4); return f;
}
inline float __apply_sgn_retain(float res, float in) {
    uint32_t rb; std::memcpy(&rb, &res, 4);
    rb = (rb & 0x7FFFFFFFu) | (std::signbit(in) ? 0x80000000u : 0u);  // result sign = sign(in)
    float f; std::memcpy(&f, &rb, 4); return f;
}

// SFPLUT: 3-entry, 8-bit coefficients. Ranges by Abs(v): [0,1)->l0, [1,2)->l1,
// [2,inf)->l2. a=hi8, c=lo8 of the chosen LReg. VD = a*|v| + c, sign retained.
inline vFloat lut(const vFloat& v, const vUInt& l0, const vUInt& l1, const vUInt& l2) {
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        float x = v.v[i], ax = std::fabs(x);
        uint32_t coeffs = (ax < 1.0f) ? l0.v[i] : (ax < 2.0f) ? l1.v[i] : l2.v[i];
        float a = __lut8_to_fp32((coeffs >> 8) & 0xFFu);
        float c = __lut8_to_fp32(coeffs & 0xFFu);
        r.v[i] = __apply_sgn_retain(__emule_sfpu_finalize(a * ax + c), x);
    }
    return r;
}

// SFPLUTFP32 6-entry FP16 table (TABLE1: last split 3.0; TABLE2/mode!=1: 4.0).
// a01/a23/a45 pack two FP16 slopes each (lo/hi 16b); b01/b23/b45 the intercepts.
// Ranges by Abs(v): [0,.5) [.5,1) [1,1.5) [1.5,2) [2,cut) [cut,inf). Sign retained.
inline vFloat lut2(const vFloat& v,
                   const vUInt& a01, const vUInt& a23, const vUInt& a45,
                   const vUInt& b01, const vUInt& b23, const vUInt& b45, int mode = 1) {
    const float cut = (mode == 1) ? 3.0f : 4.0f;
    vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        float x = v.v[i], ax = std::fabs(x);
        uint32_t areg, breg; bool hi;
        if      (ax < 0.5f) { areg = a01.v[i]; breg = b01.v[i]; hi = false; }
        else if (ax < 1.0f) { areg = a01.v[i]; breg = b01.v[i]; hi = true;  }
        else if (ax < 1.5f) { areg = a23.v[i]; breg = b23.v[i]; hi = false; }
        else if (ax < 2.0f) { areg = a23.v[i]; breg = b23.v[i]; hi = true;  }
        else if (ax < cut)  { areg = a45.v[i]; breg = b45.v[i]; hi = false; }
        else                { areg = a45.v[i]; breg = b45.v[i]; hi = true;  }
        uint16_t ab = hi ? (areg >> 16) : (areg & 0xFFFFu);
        uint16_t cb = hi ? (breg >> 16) : (breg & 0xFFFFu);
        r.v[i] = __apply_sgn_retain(__emule_sfpu_finalize(__lut16_to_fp32(ab) * ax + __lut16_to_fp32(cb)), x);
    }
    return r;
}

// ---- Narrow bf16 types + convert (SFPSTORE narrowing boundary) ----
class sFloat16b {
    float f_;
public:
    explicit sFloat16b(float f) {
        uint32_t b; std::memcpy(&b, &f, 4);
        uint32_t rnd = b + 0x7FFFu + ((b >> 16) & 1u); rnd &= 0xFFFF0000u;  // RNE
        std::memcpy(&f_, &rnd, 4);
    }
    operator vFloat() const { return vFloat(f_); }
};
// fp16 (IEEE half: 1-5-10). Constructed from a float (rounds to 10-bit
// mantissa) or from a packed fp16 bit pattern (decoded to fp32).
class sFloat16a {
    float f_;
public:
    explicit sFloat16a(float f) {
        uint32_t b; std::memcpy(&b, &f, 4);
        // RNE round fp32 -> fp16 mantissa width, then keep as fp32 value.
        uint32_t rnd = b + 0x0FFFu + ((b >> 13) & 1u); rnd &= 0xFFFFE000u;
        std::memcpy(&f_, &rnd, 4);
    }
    explicit sFloat16a(uint32_t bits16) {
        uint32_t h = bits16 & 0xFFFFu;
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, fbits;
        if (exp == 0) {
            fbits = sign << 31;                                   // flush subnormal/zero
        } else if (exp == 0x1F) {
            fbits = (sign << 31) | 0x7F800000u | (man << 13);     // inf / nan
        } else {
            fbits = (sign << 31) | ((exp - 15u + 127u) << 23) | (man << 13);
        }
        std::memcpy(&f_, &fbits, 4);
    }
    explicit sFloat16a(int32_t v) : sFloat16a(static_cast<uint32_t>(v)) {}
    operator vFloat() const { return vFloat(f_); }
};
class vFloat16b {
    vFloat v_;
public:
    vFloat16b() = default;
    explicit vFloat16b(const vFloat& v) : v_(v) {}
    operator vFloat() const { return v_; }
};
// convert<vFloat16b>(vFloat, RoundMode): bf16 narrowing. Coverage-first —
// the RoundMode is accepted but emule applies RNE truncation (float_to_fp16b).
template <typename ToType>
inline ToType convert(const vFloat& val, RoundMode = RoundMode::NearestEven) {
    return ToType(float_to_fp16b(val));
}

// ---- Programmable constant registers (LReg12-14 / CREG PRGM1-3) ----
// vConstFloatPrgmN and vConstIntPrgmN alias the SAME three bit-storage slots
// (set in an op's _init_, read in its _calculate_). Same kernel thread runs
// both, so a thread_local backing store is consistent across the boundary.
inline thread_local uint32_t __emule_prgm_creg[3] = {0, 0, 0};
class __vCRegFloat {
    int idx_;
public:
    constexpr __vCRegFloat(int i) : idx_(i) {}
    const __vCRegFloat& operator=(float f) const { uint32_t b; std::memcpy(&b, &f, 4); __emule_prgm_creg[idx_] = b; return *this; }
    operator vFloat() const { float f; std::memcpy(&f, &__emule_prgm_creg[idx_], 4); return vFloat(f); }
};
class __vCRegInt {
    int idx_;
public:
    constexpr __vCRegInt(int i) : idx_(i) {}
    const __vCRegInt& operator=(int32_t v) const { __emule_prgm_creg[idx_] = static_cast<uint32_t>(v); return *this; }
    const __vCRegInt& operator=(uint32_t v) const { __emule_prgm_creg[idx_] = v; return *this; }
    operator vInt() const { int32_t v; std::memcpy(&v, &__emule_prgm_creg[idx_], 4); return vInt(v); }
};
inline const __vCRegFloat vConstFloatPrgm0{0}, vConstFloatPrgm1{1}, vConstFloatPrgm2{2};
inline const __vCRegInt   vConstIntPrgm0{0},   vConstIntPrgm1{1},   vConstIntPrgm2{2};

// ---- Tier-3: shuffle / swap primitives ----
// Fail loud, never silently wrong (project rule). An op that reaches an
// unmodeled cross-lane primitive aborts with a clear message rather than
// computing garbage; model it faithfully when a target op needs it.
[[noreturn]] inline void __emule_sfpu_unsupported(const char* op) {
    std::fprintf(stderr,
        "[EMULE] SFPU primitive not yet modeled in the deep sfpi backend: %s. "
        "It is a cross-lane op; add a faithful model before using an op that needs it. "
        "See docs/sfpu-deep-path.md.\n", op);
    std::abort();
}

// vec_swap: unconditional element-wise swap of two vectors (SFPSWAP). This is
// lane-local, so it is modeled faithfully (mask-aware). The conditional SFPSWAP
// modes are vec_min_max / vec_max_min (above).
inline void vec_swap(vFloat& a, vFloat& b) { for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) { float t = a.v[i]; a.v[i] = b.v[i]; b.v[i] = t; } }
inline void vec_swap(vInt& a, vInt& b)     { for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) { int32_t t = a.v[i]; a.v[i] = b.v[i]; b.v[i] = t; } }
inline void vec_swap(vUInt& a, vUInt& b)   { for (uint32_t i = 0; i < 32; ++i) if (__emule_sfpi_mask[i]) { uint32_t t = a.v[i]; a.v[i] = b.v[i]; b.v[i] = t; } }

// subvec_transp: SFPTRANSP — cross-sub-vector transpose of 4 registers. The
// exact lane permutation is not yet modeled; fail loud until an op needs it.
inline void subvec_transp(vFloat&, vFloat&, vFloat&, vFloat&) { __emule_sfpu_unsupported("subvec_transp(vFloat)"); }
inline void subvec_transp(vInt&,   vInt&,   vInt&,   vInt&)   { __emule_sfpu_unsupported("subvec_transp(vInt)"); }
inline void subvec_transp(vUInt&,  vUInt&,  vUInt&,  vUInt&)  { __emule_sfpu_unsupported("subvec_transp(vUInt)"); }

}  // namespace sfpi

// Some kernels reference `RoundMode` unqualified (e.g. SDPA's first-column
// recip writes `convert<sfpi::vFloat16b>(out, RoundMode::NearestEven)` — sfpi::
// on the type but bare on the enum). On silicon the TRISC build preamble brings
// sfpi names into scope; emule re-exports just this enum to global scope to match.
using ::sfpi::RoundMode;

// ---- v_if / v_elseif / v_else / v_endif macros ----
//
// Silicon: per-lane mask stack pushed by v_if, popped by v_endif.
// Emule: same semantics, implemented via __emule_sfpi_push_if etc.
// The macros wrap a scope block so user code reads naturally:
//     v_if(x > limit) { x = limit; } v_endif;
#ifndef v_if
#define v_if(cond)        do { ::sfpi::__emule_sfpi_push_if((cond));
#define v_elseif(cond)    ::sfpi::__emule_sfpi_elseif((cond));
#define v_else            ::sfpi::__emule_sfpi_else();
#define v_endif           ::sfpi::__emule_sfpi_endif(); } while (0)
#endif

// ---- SFPU library functions used by upstream kernels (in ckernel::sfpu) ----
//
// On silicon these live in `ckernel::sfpu` and emit SFPU LUT+polynomial
// sequences via TTI_SFP intrinsics. emule provides scalar `expf` / `1/x` per
// lane — correct to ~bf16 precision for typical input ranges. The integer
// template params (APPROX_MODE / max_iter) select the silicon Newton-Raphson
// iteration count; emule's exact 1/x is at least as precise, so they are no-ops.

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en = false>
inline ::sfpi::vFloat _sfpu_sigmoid_(const ::sfpi::vFloat& x) {
    ::sfpi::vFloat r;
    for (uint32_t i = 0; i < 32; ++i) {
        float xv = x.v[i];
        r.v[i] = 1.0f / (1.0f + std::exp(-xv));
    }
    return r;
}

inline ::sfpi::vFloat _sfpu_exp_(const ::sfpi::vFloat& x) {
    ::sfpi::vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = std::exp(x.v[i]);
    return r;
}

template <int APPROX_MODE = 2>
inline ::sfpi::vFloat _sfpu_reciprocal_(const ::sfpi::vFloat& x) {
    ::sfpi::vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = 1.0f / x.v[i];
    return r;
}

template <bool APPROX = false>
inline void _init_sfpu_reciprocal_() {}

// Reciprocal-via-rsqrt-compat path (ckernel_sfpu_rsqrt_compat.h). On silicon a
// Newton-Raphson reciprocal with `max_iter` iterations; scalar 1/x in emule.
template <int max_iter = 3>
inline ::sfpi::vFloat _reciprocal_compat_(const ::sfpi::vFloat& in) {
    ::sfpi::vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = 1.0f / in.v[i];
    return r;
}

template <bool APPROX = false>
inline void sfpu_reciprocal_init() {}

// exp(val * scale), scale = bf16-decoded from exp_base_scale_factor's low 16 bits
// (silicon: `val * sfpi::sFloat16b(exp_base_scale_factor)` then exp). emule uses
// std::exp (more accurate than silicon's exp_21f polynomial). The SDPA softmax
// exp path runs this for exp_approx_mode=true.
template <bool SCALE_EN, bool is_fp32_dest_acc_en>
inline ::sfpi::vFloat _ckernel_sfpu_exp_accurate_(::sfpi::vFloat val, std::uint32_t exp_base_scale_factor) {
    float s = 1.0f;
    if constexpr (SCALE_EN) {
        std::uint32_t b = (exp_base_scale_factor & 0xFFFFu) << 16;
        std::memcpy(&s, &b, sizeof(s));
    }
    ::sfpi::vFloat r;
    for (uint32_t i = 0; i < 32; ++i) r.v[i] = std::exp(val.v[i] * s);
    return r;
}

// Reinterpret an fp32 bit pattern as float (kernels pass beta/threshold as bits).
// Distinct from ::sfpi::Converter, whose as_float returns a vFloat.
struct Converter {
    static float as_float(std::uint32_t bits) {
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// softplus(x) = (1/beta)*ln(1 + exp(beta*x)), linear above threshold. Operates on
// the dst_reg cursor window. NOTE: silicon uses a degree-8 polynomial approximation
// (ckernel_sfpu_softplus.h); emule uses exact libm softplus. The SDPA prefill test
// does not exercise softplus/logsigmoid (compile-only here), so the polynomial is
// not ported — revisit if a softplus-using suite is brought up.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void calculate_softplus_body(const float beta, const float beta_reciprocal, const float threshold) {
    ::sfpi::vFloat val = ::sfpi::dst_reg[0];
    ::sfpi::vFloat out;
    for (uint32_t i = 0; i < 32; ++i) {
        float x = val.v[i];
        float t = beta * x;
        out.v[i] = (t < threshold) ? beta_reciprocal * std::log1p(std::exp(t)) : x;
    }
    ::sfpi::dst_reg[0] = out;
}

}  // namespace sfpu
}  // namespace ckernel
