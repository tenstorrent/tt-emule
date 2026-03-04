#pragma once
#include "circular_buffer.hpp"
#include "dst_register_file.hpp"
#include <array>
#include <memory>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace tt_emule {

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
        l1_.fill(0);
    }

    CoreCoord coord() const { return coord_; }

    std::shared_ptr<CircularBuffer>& cb(size_t idx) {
        if (idx >= MAX_CBS) throw std::out_of_range("CB index out of range");
        return cbs_[idx];
    }

    DstRegisterFile& dst() { return dst_; }

    uint8_t* l1_ptr(uint32_t offset) {
        return l1_.data() + offset;
    }

private:
    CoreCoord coord_;
    std::array<uint8_t, L1_SIZE> l1_;
    std::array<std::shared_ptr<CircularBuffer>, MAX_CBS> cbs_;
    DstRegisterFile dst_;
};

class Device {
public:
    static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024; // 256 MB

    Device() : dram_(DRAM_SIZE, 0), dram_bump_(0), core_({0, 0}) {}

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
    // Core L1: x==0, y==0, addr < L1_SIZE
    // DRAM: otherwise treat addr as dram offset
    uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
        if (x == 0 && y == 0) {
            return core_.l1_ptr(static_cast<uint32_t>(addr));
        }
        // Treat as DRAM address for prototype
        return dram_ptr(addr);
    }

    Core& core() { return core_; }

private:
    std::vector<uint8_t> dram_;
    uint64_t dram_bump_;
    Core core_;
};

} // namespace tt_emule
