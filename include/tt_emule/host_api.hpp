#pragma once
#include "device.hpp"
#include "buffer.hpp"
#include "program.hpp"
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <cstdint>

namespace tt_emule {

using KernelHandle = uint32_t;

// Thin wrapper that binds a CommandQueue to a Device (mirrors tt-metal's separation).
struct CommandQueue {
    Device* device;
};

// ---- Device lifecycle ----
Device* CreateDevice(uint32_t id = 0);
bool    CloseDevice(Device* device);

CommandQueue CreateCommandQueue(Device& device);

// ---- Program construction ----
Program CreateProgram();

void SetRuntimeArgs(Program& program, KernelHandle kernel_id, CoreCoord core,
                    std::vector<uint32_t> args);

CBHandle CreateCircularBuffer(Program& program, CoreCoord core,
                               CircularBufferConfig config);

// ---- Quasar DFB construction ----
DFBHandle CreateDataflowBuffer(Program& program, DataflowBufferConfig config);

// ---- Buffer operations (Device overloads) ----
std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes,
                                     uint32_t page_size_bytes,
                                     BufferType type = BufferType::DRAM);

void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src,
                        bool blocking = true);
void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst,
                       bool blocking = true);

// ---- Buffer operations (CommandQueue overloads) ----
void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buf, const void* src,
                        bool blocking = true);
void EnqueueReadBuffer(CommandQueue& cq, Buffer& buf, void* dst,
                       bool blocking = true);

// ---- Program execution ----
void EnqueueProgram(Device& device, Program& program, bool blocking = true);
void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking = true);

void Finish(Device& device);
void Finish(CommandQueue& cq);

// ---- tt-metal detail:: compatibility ----
namespace detail {

// Equivalent to EnqueueProgram + Finish (wait=true by default).
void LaunchProgram(Device* device, Program& program,
                   bool wait = true, bool force_slow = false);

// Read `size` bytes from L1 at the given 32-bit address into result (as uint32_t words).
bool ReadFromDeviceL1(Device* device, const CoreCoord& core,
                      uint32_t address, uint32_t size,
                      std::vector<uint32_t>& result);

// Write uint32_t vector to L1 at given core + offset.
void WriteToDeviceL1(Device* device, const CoreCoord& core,
                     uint32_t address, const std::vector<uint32_t>& data);

// Write byte span to L1 at given core + offset.
void WriteToDeviceL1(Device* device, const CoreCoord& core,
                     uint32_t address, std::span<const uint8_t> data);

// Read byte span from L1 at given core + offset.
void ReadFromDeviceL1(Device* device, const CoreCoord& core,
                      uint32_t address, std::span<uint8_t> data);

// Write uint32_t vector to DRAM channel at offset.
void WriteToDeviceDRAMChannel(Device* device, uint32_t channel,
                              uint32_t address, const std::vector<uint32_t>& data);

// Write byte span to DRAM channel at offset.
void WriteToDeviceDRAMChannel(Device* device, uint32_t channel,
                              uint32_t address, std::span<const uint8_t> data);

// Read uint32_t vector from DRAM channel at offset.
void ReadFromDeviceDRAMChannel(Device* device, uint32_t channel,
                               uint32_t address, uint32_t byte_size,
                               std::vector<uint32_t>& result);

// Read byte span from DRAM channel at offset.
void ReadFromDeviceDRAMChannel(Device* device, uint32_t channel,
                               uint32_t address, std::span<uint8_t> data);

template<typename T>
inline void WriteToBuffer(const std::shared_ptr<Buffer>& buf,
                           const std::vector<T>& data) {
    EnqueueWriteBuffer(*buf->device(), *buf,
                       static_cast<const void*>(data.data()), true);
}

template<typename T>
inline void ReadFromBuffer(const std::shared_ptr<Buffer>& buf,
                            std::vector<T>& data) {
    data.resize(buf->size() / sizeof(T));
    EnqueueReadBuffer(*buf->device(), *buf,
                      static_cast<void*>(data.data()), true);
}

} // namespace detail

} // namespace tt_emule

// JIT kernel creation helper — lives OUTSIDE tt_emule to avoid ADL conflicts.
// When test code has `using namespace tt::tt_metal` and arg types are tt_emule
// aliases, ADL would otherwise find tt_emule::CreateKernel instead of
// tt::tt_metal::CreateKernel.
namespace tt_emule_internal {
    tt_emule::KernelHandle create_jit_kernel(
        tt_emule::Program& program, const std::string& kernel_src_path,
        tt_emule::CoreCoord core, tt_emule::DataMovementConfig config);
} // namespace tt_emule_internal
