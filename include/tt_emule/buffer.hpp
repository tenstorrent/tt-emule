#pragma once
#include <cstdint>
#include <cstddef>

namespace tt_emule {

class Device;

// DRAM-backed buffer; offset into Device's DRAM vector.
class Buffer {
public:
    Buffer(Device* device, size_t size_bytes, uint32_t page_size_bytes,
           uint64_t dram_offset)
        : device_(device), size_bytes_(size_bytes),
          page_size_bytes_(page_size_bytes), dram_offset_(dram_offset) {}

    size_t   size() const { return size_bytes_; }
    uint32_t page_size() const { return page_size_bytes_; }
    uint64_t dram_offset() const { return dram_offset_; }
    Device*  device() const { return device_; }

private:
    Device*  device_;
    size_t   size_bytes_;
    uint32_t page_size_bytes_;
    uint64_t dram_offset_; // byte offset in Device::dram_
};

} // namespace tt_emule
