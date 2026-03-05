#pragma once
// Emulation stubs for experimental::Noc, AllocatorBank, ReadSpec, WriteSpec.
// async_read/write delegate to memcpy via __device->dram_ptr().

#include <cstdint>
#include <cstring>

// C-linkage accessor: defined in kernel_runner.cpp (compiled with the correct
// Device layout, including vtable when TT_EMULE_USE_XY_PAIR is active).
// The JIT kernel must NOT inline Device::dram_ptr() itself because the JIT
// compile does not define TT_EMULE_USE_XY_PAIR, so the inlined method would
// use the wrong struct layout (missing vtable pointer offset).
extern "C" uint8_t* __emule_dram_ptr(uint64_t offset);

namespace experimental {

enum class AllocatorBankType { DRAM, L1 };

template<AllocatorBankType>
struct AllocatorBank {};

struct ReadSpec {
    uint32_t bank_id = 0;
    uint32_t addr = 0;
};

struct WriteSpec {
    uint32_t bank_id = 0;
    uint32_t addr = 0;
};

class Noc {
public:
    // async_read: DRAM → L1
    template<AllocatorBankType BT, typename L1Mem>
    void async_read(AllocatorBank<BT>&, L1Mem& dst, uint32_t size,
                    ReadSpec src_spec, WriteSpec /*dst_spec*/) {
        std::memcpy(static_cast<uint8_t*>(dst),
                    __emule_dram_ptr(src_spec.addr), size);
    }

    // async_write: L1 → DRAM
    template<typename L1Mem, AllocatorBankType BT>
    void async_write(L1Mem& src, AllocatorBank<BT>&, uint32_t size,
                     ReadSpec /*src_spec*/, WriteSpec dst_spec) {
        std::memcpy(__emule_dram_ptr(dst_spec.addr),
                    static_cast<uint8_t*>(src), size);
    }

    void async_read_barrier() {}
    void async_write_barrier() {}
};

} // namespace experimental
