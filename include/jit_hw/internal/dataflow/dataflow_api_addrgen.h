#pragma once
// Minimal interleaved address generation for JIT-compiled kernels.
// Matches the real firmware interface from tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h
// but runs on x86 host using bank mapping arrays populated by emulated_program_runner.

#include <cstdint>

// Bank mapping arrays — populated by emulated_program_runner.cpp, resolved at dlopen.
// Declared with C++ linkage (matching firmware declarations in dataflow_api_common.h).
extern uint16_t dram_bank_to_noc_xy[2][32];
extern int32_t bank_to_dram_offset[32];
extern uint16_t l1_bank_to_noc_xy[2][32];
extern int32_t bank_to_l1_offset[32];

// Core coordinates (set per kernel thread by program runner).
extern thread_local uint8_t my_x[2];
extern thread_local uint8_t my_y[2];

// NOC encoding constants (matching firmware for Blackhole/Wormhole).
// NOC_XY_ADDR(x, y, addr) = (y << 42) | (x << 36) | addr
#ifndef NOC_ADDR_LOCAL_BITS
#define NOC_ADDR_LOCAL_BITS 36
#endif
#ifndef NOC_ADDR_NODE_ID_BITS
#define NOC_ADDR_NODE_ID_BITS 6
#endif
// NOC_ADDR_COORD_SHIFT = NOC_ADDR_LOCAL_BITS for Blackhole (36).
// For Wormhole the register split is at 32, but the noc_xy already has a 4-bit pre-shift.
// We use 36 (Blackhole-style) with noc_xy = (y << 6) | x.
#ifndef NOC_ADDR_COORD_SHIFT
#define NOC_ADDR_COORD_SHIFT NOC_ADDR_LOCAL_BITS
#endif

// Alignment helper (matches firmware api/alignment.h)
inline constexpr uint32_t align_power_of_2(uint32_t addr, uint32_t alignment) {
    return ((addr - 1) | (alignment - 1)) + 1;
}

// NUM_DRAM_BANKS / NUM_L1_BANKS — passed as JIT defines from program runner.
#ifndef NUM_DRAM_BANKS
#define NUM_DRAM_BANKS 1
#endif
#ifndef NUM_L1_BANKS
#define NUM_L1_BANKS 1
#endif
#ifndef DRAM_ALIGNMENT
#define DRAM_ALIGNMENT 32
#endif
#ifndef L1_ALIGNMENT
#define L1_ALIGNMENT 16
#endif

namespace interleaved_addr_gen {

template <bool DRAM>
inline uint32_t get_bank_offset_index(uint32_t id) {
    // On x86 we can use simple division (no bit tricks needed).
    if constexpr (DRAM) {
        return id / NUM_DRAM_BANKS;
    } else {
        return id / NUM_L1_BANKS;
    }
}

template <bool DRAM>
inline uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index) {
    if constexpr (DRAM) {
        return id - bank_offset_index * NUM_DRAM_BANKS;
    } else {
        return id - bank_offset_index * NUM_L1_BANKS;
    }
}

template <bool DRAM>
inline uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = 0) {
    if constexpr (DRAM) {
        return dram_bank_to_noc_xy[noc][bank_index];
    } else {
        return l1_bank_to_noc_xy[noc][bank_index];
    }
}

template <bool DRAM>
inline uint32_t get_bank_offset(uint32_t bank_index) {
    if constexpr (DRAM) {
        return bank_to_dram_offset[bank_index];
    } else {
        return bank_to_l1_offset[bank_index];
    }
}

template <bool DRAM>
inline constexpr uint32_t get_allocator_alignment() {
    if constexpr (DRAM) {
        return DRAM_ALIGNMENT;
    } else {
        return L1_ALIGNMENT;
    }
}

}  // namespace interleaved_addr_gen

inline uint64_t get_noc_addr_helper(uint32_t noc_xy, uint32_t addr) {
    return (static_cast<uint64_t>(noc_xy) << NOC_ADDR_COORD_SHIFT) | addr;
}

// get_noc_addr_from_bank_id — used directly by D2M-generated dataflow kernels.
// Matches firmware: resolves bank_id → (noc_xy, addr_with_bank_offset).
template <bool DRAM>
inline uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t bank_address_offset, uint8_t noc = noc_index) {
    uint64_t noc_addr = 0;
    if constexpr (DRAM) {
        noc_addr = dram_bank_to_noc_xy[noc][bank_id];
        bank_address_offset += bank_to_dram_offset[bank_id];
    } else {
        noc_addr = l1_bank_to_noc_xy[noc][bank_id];
    }
    return (noc_addr << NOC_ADDR_COORD_SHIFT) | (bank_address_offset);
}

// InterleavedAddrGen — matches real firmware interface.
template <bool DRAM>
struct InterleavedAddrGen {
    static constexpr bool is_dram = DRAM;
    uint32_t bank_base_address;
    const uint32_t page_size;
    const uint32_t aligned_page_size =
        align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<DRAM>());

    inline uint32_t get_addr(
        const uint32_t id,
        const uint32_t bank_offset_index,
        const uint32_t bank_index,
        const uint32_t offset = 0) const {
        return (bank_offset_index * this->aligned_page_size) + this->bank_base_address + offset +
               interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
    }

    inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t addr = this->get_addr(id, bank_offset_index, bank_index, offset);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);
        return get_noc_addr_helper(noc_xy, addr);
    }
};

// InterleavedPow2AddrGen — for power-of-2 page sizes.
template <bool DRAM>
struct InterleavedPow2AddrGen {
    static constexpr bool is_dram = DRAM;
    const uint32_t bank_base_address;
    const uint32_t log_base_2_of_page_size;

    inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t page_size = 1u << log_base_2_of_page_size;
        uint32_t aligned = align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<DRAM>());
        uint32_t addr = (bank_offset_index * aligned) + bank_base_address + offset +
                        interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);
        return get_noc_addr_helper(noc_xy, addr);
    }
};

// InterleavedAddrGenFast — matches real firmware (simplified: same as InterleavedAddrGen).
template <bool DRAM, uint32_t tile_hw = 1024>
struct InterleavedAddrGenFast {
    static constexpr bool is_dram = DRAM;
    uint32_t bank_base_address;
    uint32_t page_size;
    uint32_t data_format;  // DataFormat enum value (unused in emulation)

    inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const {
        uint32_t bank_offset_index = interleaved_addr_gen::get_bank_offset_index<DRAM>(id);
        uint32_t bank_index = interleaved_addr_gen::get_bank_index<DRAM>(id, bank_offset_index);
        uint32_t aligned = align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<DRAM>());
        uint32_t addr = (bank_offset_index * aligned) + bank_base_address + offset +
                        interleaved_addr_gen::get_bank_offset<DRAM>(bank_index);
        uint32_t noc_xy = interleaved_addr_gen::get_noc_xy<DRAM>(bank_index, noc);
        return get_noc_addr_helper(noc_xy, addr);
    }
};
