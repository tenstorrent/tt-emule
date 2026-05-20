// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_emule/host_api.hpp"
#include "tt_emule/jit_kernel.hpp"
#include <cstring>
#include <span>

#ifdef TT_EMULE_USE_XY_PAIR
// Include the fake allocator/core_coord headers so that tt::tt_metal::Allocator,
// AllocatorImpl, and CoreRangeSet are complete types here.
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#endif

namespace tt_emule {

#ifdef TT_EMULE_USE_XY_PAIR

// Out-of-line Device constructor: initializes allocator wrappers that need
// tt::tt_metal::Allocator/AllocatorImpl to be complete.
Device::Device()
    : dram_(DRAM_SIZE, 0),
      dram_bump_(0),
      core_({0, 0}),
      alloc_(core_.l1_base_addr()),
      allocator_(std::make_unique<tt::tt_metal::Allocator>(&alloc_)),
      allocator_impl_(std::make_unique<tt::tt_metal::AllocatorImpl>(&alloc_)),
      sub_device_ids_({tt::tt_metal::SubDeviceId{0}}),
      sub_device_stall_group_()
{}

// Out-of-line destructor: ensures unique_ptr<Allocator/AllocatorImpl> can be
// destroyed with complete types.
Device::~Device() = default;

// Out-of-line worker_cores: needs CoreRangeSet to be complete.
tt::tt_metal::CoreRangeSet Device::worker_cores(
        tt::tt_metal::HalProgrammableCoreType /*core_type*/,
        tt::tt_metal::SubDeviceId /*sub_device_id*/) const {
    return tt::tt_metal::CoreRangeSet{
        tt::tt_metal::CoreRange{{0, 0}, {0, 0}}
    };
}

#endif  // TT_EMULE_USE_XY_PAIR

Device* CreateDevice(uint32_t /*id*/) {
    return new Device();
}

bool CloseDevice(Device* device) {
    delete device;
    return true;
}

Program CreateProgram() {
    return Program();
}

void SetRuntimeArgs(Program& program, uint32_t kernel_id, CoreCoord /*core*/,
                    std::vector<uint32_t> args) {
    program.set_runtime_args(kernel_id, std::move(args));
}

CBHandle CreateCircularBuffer(Program& program, CoreCoord /*core*/,
                               CircularBufferConfig config) {
    return program.add_cb(config);
}

DFBHandle CreateDataflowBuffer(Program& program, DataflowBufferConfig config) {
    return program.add_dfb(config);
}

std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes,
                                     uint32_t page_size_bytes,
                                     BufferType type) {
    if (type == BufferType::L1) {
        uint32_t addr = device.l1_alloc(size_bytes);
        return std::make_shared<Buffer>(&device, size_bytes, page_size_bytes,
                                        static_cast<uint64_t>(addr), BufferType::L1);
    }
    // DRAM (and all other types mapped to DRAM)
    uint64_t offset = device.dram_alloc(size_bytes);
    return std::make_shared<Buffer>(&device, size_bytes, page_size_bytes,
                                    offset, BufferType::DRAM);
}

void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src, bool /*blocking*/) {
    if (buf.buffer_type() == BufferType::L1) {
        uint8_t* ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(buf.address()));
        std::memcpy(ptr, src, buf.size());
    } else {
        std::memcpy(device.dram_ptr(buf.dram_offset()), src, buf.size());
    }
}

void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst, bool /*blocking*/) {
    if (buf.buffer_type() == BufferType::L1) {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(
            static_cast<uintptr_t>(buf.address()));
        std::memcpy(dst, ptr, buf.size());
    } else {
        std::memcpy(dst, device.dram_ptr(buf.dram_offset()), buf.size());
    }
}

void Finish(Device& /*device*/) {}

CommandQueue CreateCommandQueue(Device& device) {
    return CommandQueue{&device};
}

void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buf, const void* src, bool blocking) {
    EnqueueWriteBuffer(*cq.device, buf, src, blocking);
}

void EnqueueReadBuffer(CommandQueue& cq, Buffer& buf, void* dst, bool blocking) {
    EnqueueReadBuffer(*cq.device, buf, dst, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    EnqueueProgram(*cq.device, program, blocking);
}

void Finish(CommandQueue& cq) {
    Finish(*cq.device);
}

// ---- detail:: implementations ----
namespace detail {

void LaunchProgram(Device* device, Program& program, bool /*wait*/, bool /*force_slow*/) {
    EnqueueProgram(*device, program, true);
}

bool ReadFromDeviceL1(Device* device, const CoreCoord& /*core*/,
                      uint32_t address, uint32_t size,
                      std::vector<uint32_t>& result) {
    // address is the 32-bit representation of the host pointer into L1.
    // Since L1 is mmap'd at a <=4 GB hint, this round-trip is valid.
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(
        static_cast<uintptr_t>(address));
    (void)device; // not needed — address IS the host pointer
    size_t num_words = (size + 3) / 4;
    result.resize(num_words);
    std::memcpy(result.data(), ptr, num_words * 4);
    return true;
}

void WriteToDeviceL1(Device* device, const CoreCoord& /*core*/,
                     uint32_t address, const std::vector<uint32_t>& data) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    (void)device;
    std::memcpy(ptr, data.data(), data.size() * sizeof(uint32_t));
}

void WriteToDeviceL1(Device* device, const CoreCoord& /*core*/,
                     uint32_t address, std::span<const uint8_t> data) {
    uint8_t* ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(address));
    (void)device;
    std::memcpy(ptr, data.data(), data.size());
}

void ReadFromDeviceL1(Device* device, const CoreCoord& /*core*/,
                      uint32_t address, std::span<uint8_t> data) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(address));
    (void)device;
    std::memcpy(data.data(), ptr, data.size());
}

void WriteToDeviceDRAMChannel(Device* device, uint32_t /*channel*/,
                              uint32_t address, const std::vector<uint32_t>& data) {
    std::memcpy(device->dram_ptr(address), data.data(), data.size() * sizeof(uint32_t));
}

void WriteToDeviceDRAMChannel(Device* device, uint32_t /*channel*/,
                              uint32_t address, std::span<const uint8_t> data) {
    std::memcpy(device->dram_ptr(address), data.data(), data.size());
}

void ReadFromDeviceDRAMChannel(Device* device, uint32_t /*channel*/,
                               uint32_t address, uint32_t byte_size,
                               std::vector<uint32_t>& result) {
    size_t num_words = (byte_size + 3) / 4;
    result.resize(num_words);
    std::memcpy(result.data(), device->dram_ptr(address), num_words * 4);
}

void ReadFromDeviceDRAMChannel(Device* device, uint32_t /*channel*/,
                               uint32_t address, std::span<uint8_t> data) {
    std::memcpy(data.data(), device->dram_ptr(address), data.size());
}

} // namespace detail

} // namespace tt_emule

// JIT helper: lives outside tt_emule namespace to avoid ADL conflicts when
// test code has `using namespace tt::tt_metal` and all arg types are tt_emule aliases.
namespace tt_emule_internal {

tt_emule::KernelHandle create_jit_kernel(
    tt_emule::Program& program, const std::string& kernel_src_path,
    tt_emule::CoreCoord core, tt_emule::DataMovementConfig config)
{
    using namespace tt_emule;
    KernelType type = config.type;
    if (config.processor == DataMovementProcessor::RISCV_1 &&
        type == KernelType::DataMovement0) {
        type = KernelType::DataMovement1;
    }
    // Empty string → jit_kernel.cpp uses TT_EMULE_JIT_INCLUDE_DIR from CMake
    KernelFn fn = jit_compile_kernel(kernel_src_path, config.compile_args, "");
    return program.add_kernel(type, std::move(fn), core);
}

} // namespace tt_emule_internal
