#pragma once
#include <cstdint>
#include <cstddef>

namespace tt_emule {

class Device;
enum class BufferType;

// Buffer backed by either DRAM or L1.
class Buffer {
public:
    // For DRAM buffers: offset_ is the byte offset in Device::dram_.
    // For L1 buffers:   offset_ is the absolute host pointer to L1 memory.
    Buffer(Device* device, size_t size_bytes, uint32_t page_size_bytes,
           uint64_t offset, BufferType type)
        : device_(device), size_bytes_(size_bytes),
          page_size_bytes_(page_size_bytes), offset_(offset), type_(type) {}

    size_t     size() const { return size_bytes_; }
    uint32_t   page_size() const { return page_size_bytes_; }
    uint64_t   dram_offset() const { return offset_; }
    uint32_t   address() const { return static_cast<uint32_t>(offset_); }
    BufferType buffer_type() const { return type_; }
    Device*    device() const { return device_; }

private:
    Device*    device_;
    size_t     size_bytes_;
    uint32_t   page_size_bytes_;
    uint64_t   offset_;   // DRAM byte offset OR absolute L1 host pointer
    BufferType type_;
};

} // namespace tt_emule
