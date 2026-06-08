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
};

// ---- Arithmetic and comparison ops ----

inline vFloat operator+(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] + b.v[i]; return r; }
inline vFloat operator-(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] - b.v[i]; return r; }
inline vFloat operator*(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] * b.v[i]; return r; }
inline vFloat operator/(const vFloat& a, const vFloat& b) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = a.v[i] / b.v[i]; return r; }
inline vFloat operator-(const vFloat& a) { vFloat r; for (uint32_t i = 0; i < 32; ++i) r.v[i] = -a.v[i]; return r; }

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

class DstRegLane {
    uint32_t lane_offset_;  // offset from __emule_sfpi_cursor; usually 0
public:
    constexpr DstRegLane(uint32_t off) : lane_offset_(off) {}

    // Load 32 lanes from DST into a vFloat.
    operator vFloat() const {
        vFloat r;
        float* base = __emule_sfpi_active_dst();
        for (uint32_t i = 0; i < 32; ++i) {
            r.v[i] = base[__emule_sfpi_cursor + lane_offset_ * 32 + i];
        }
        return r;
    }
    // Load 32 lanes as int (bit-cast).
    operator vInt() const {
        vInt r;
        float* base = __emule_sfpi_active_dst();
        for (uint32_t i = 0; i < 32; ++i) {
            float f = base[__emule_sfpi_cursor + lane_offset_ * 32 + i];
            int32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            r.v[i] = bits;
        }
        return r;
    }

    // Store with active-lane mask.
    DstRegLane& operator=(const vFloat& x) {
        float* base = __emule_sfpi_active_dst();
        for (uint32_t i = 0; i < 32; ++i) {
            if (__emule_sfpi_mask[i]) {
                base[__emule_sfpi_cursor + lane_offset_ * 32 + i] = x.v[i];
            }
        }
        return *this;
    }
    DstRegLane& operator=(const vInt& x) {
        float* base = __emule_sfpi_active_dst();
        for (uint32_t i = 0; i < 32; ++i) {
            if (__emule_sfpi_mask[i]) {
                int32_t bits = x.v[i];
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                base[__emule_sfpi_cursor + lane_offset_ * 32 + i] = f;
            }
        }
        return *this;
    }
    DstRegLane& operator=(float x) { return operator=(vFloat(x)); }
};

class DstReg {
public:
    DstRegLane operator[](int i) const { return DstRegLane(static_cast<uint32_t>(i)); }
    // dst_reg++ : silicon advances the implicit slice pointer by 32 lanes.
    DstReg& operator++() { __emule_sfpi_cursor += 32; return *this; }
    DstReg operator++(int) { DstReg t = *this; __emule_sfpi_cursor += 32; return t; }
};

inline DstReg dst_reg;

// l_reg[LRegs::LRegN] — silicon: direct access to LRegN. Emule: 8 thread-local
// vFloats backing LReg0..7. Used by topk to stash partial results between
// TTI_SFP intrinsics; emule's scalar shim won't track TTI_SFP state, but
// the read/assign paths compile cleanly so the surface parses.
struct LRegFile {
    vFloat& operator[](LRegs idx);
    const vFloat& operator[](LRegs idx) const;
};
inline thread_local vFloat __emule_l_reg_storage[8] = {};
inline vFloat& LRegFile::operator[](LRegs idx) { return __emule_l_reg_storage[static_cast<uint8_t>(idx)]; }
inline const vFloat& LRegFile::operator[](LRegs idx) const { return __emule_l_reg_storage[static_cast<uint8_t>(idx)]; }
inline LRegFile l_reg;

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

}  // namespace sfpi

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

// ---- SFPU library functions used by upstream kernels (in ckernel::) ----
//
// On silicon, `_sfpu_sigmoid_<...>`, `_sfpu_reciprocal_<...>`, `_sfpu_exp_`
// live in `ckernel::sfpu` and emit SFPU LUT+polynomial sequences via
// TTI_SFP intrinsics. emule provides scalar `expf` / `1/x` per lane —
// correct to ~bf16 precision for typical input ranges.

namespace ckernel {

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

}  // namespace ckernel
