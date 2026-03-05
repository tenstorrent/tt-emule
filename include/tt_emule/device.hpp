#pragma once
#include "circular_buffer.hpp"
#include "dst_register_file.hpp"
#include <array>
#include <memory>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>

namespace tt_emule {

enum class HalMemType { L1, DRAM };

struct CoreCoord {
    uint32_t x;
    uint32_t y;
    bool operator==(const CoreCoord& o) const { return x == o.x && y == o.y; }
};

class Core {
public:
    static constexpr size_t L1_SIZE = 1024 * 1024; // 1 MB
    static constexpr size_t MAX_CBS = 32;

    explicit Core(CoreCoord coord) : coord_(coord) {
        void* hint = reinterpret_cast<void*>(uintptr_t(0x40000000));
        void* p = mmap(hint, L1_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("mmap for Core L1 failed");
        l1_ = static_cast<uint8_t*>(p);
        l1_base_ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_));
        std::memset(l1_, 0, L1_SIZE);
    }

    ~Core() {
        if (l1_) munmap(l1_, L1_SIZE);
    }

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    CoreCoord coord() const { return coord_; }

    std::shared_ptr<CircularBuffer>& cb(size_t idx) {
        if (idx >= MAX_CBS) throw std::out_of_range("CB index out of range");
        return cbs_[idx];
    }

    DstRegisterFile& dst() { return dst_; }

    uint8_t* l1_ptr(uint32_t offset) { return l1_ + offset; }

    // 32-bit address of the L1 base (valid if mmap succeeded below 4 GB).
    uint32_t l1_base_addr() const { return l1_base_; }

private:
    CoreCoord coord_;
    uint8_t*  l1_      = nullptr;
    uint32_t  l1_base_ = 0;
    std::array<std::shared_ptr<CircularBuffer>, MAX_CBS> cbs_;
    DstRegisterFile dst_;
};

// Minimal allocator mimic for tt-metal compat (get_base_allocator_addr).
class MockAllocator {
    uint32_t l1_base_;
public:
    explicit MockAllocator(uint32_t base) : l1_base_(base) {}
    uint32_t get_base_allocator_addr(HalMemType type) const {
        return (type == HalMemType::L1) ? l1_base_ : 0;
    }
};

class Device {
public:
    static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024; // 256 MB

    Device() : dram_(DRAM_SIZE, 0), dram_bump_(0),
               core_({0, 0}), alloc_(core_.l1_base_addr()) {}

    // Bump allocator for DRAM
    uint64_t dram_alloc(size_t bytes) {
        if (dram_bump_ + bytes > DRAM_SIZE)
            throw std::runtime_error("DRAM OOM");
        uint64_t offset = dram_bump_;
        dram_bump_ += bytes;
        return offset;
    }

    uint8_t* dram_ptr(uint64_t offset) { return dram_.data() + offset; }

    // Map (x, y, addr) to raw pointer.
    // Core L1: x==0, y==0, addr interpreted as byte offset in L1
    // DRAM: otherwise treat addr as dram offset
    uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
        if (x == 0 && y == 0)
            return core_.l1_ptr(static_cast<uint32_t>(addr));
        return dram_ptr(addr);
    }

    Core& core() { return core_; }

    MockAllocator* allocator() { return &alloc_; }

private:
    std::vector<uint8_t> dram_;
    uint64_t             dram_bump_;
    Core                 core_;
    MockAllocator        alloc_;
};

} // namespace tt_emule
