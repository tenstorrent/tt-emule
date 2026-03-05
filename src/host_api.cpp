#include "tt_emule/host_api.hpp"
#include "tt_emule/jit_kernel.hpp"
#include <cstring>

namespace tt_emule {

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

uint32_t CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                      DataMovementConfig config) {
    // Derive type from processor if default type is still DataMovement0
    // but processor is explicitly RISCV_1.
    KernelType type = config.type;
    if (config.processor == DataMovementProcessor::RISCV_1 &&
        type == KernelType::DataMovement0) {
        type = KernelType::DataMovement1;
    }
    return program.add_kernel(type, std::move(fn), core);
}

uint32_t CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                      ComputeConfig /*config*/) {
    return program.add_kernel(KernelType::Compute, std::move(fn), core);
}

void SetRuntimeArgs(Program& program, uint32_t kernel_id, CoreCoord /*core*/,
                    std::vector<uint32_t> args) {
    program.set_runtime_args(kernel_id, std::move(args));
}

CBHandle CreateCircularBuffer(Program& program, CoreCoord /*core*/,
                               CircularBufferConfig config) {
    return program.add_cb(config);
}

std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes,
                                     uint32_t page_size_bytes) {
    uint64_t offset = device.dram_alloc(size_bytes);
    return std::make_shared<Buffer>(&device, size_bytes, page_size_bytes, offset);
}

void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src, bool /*blocking*/) {
    std::memcpy(device.dram_ptr(buf.dram_offset()), src, buf.size());
}

void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst, bool /*blocking*/) {
    std::memcpy(dst, device.dram_ptr(buf.dram_offset()), buf.size());
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
