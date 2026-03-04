#include "tt_emule/host_api.hpp"
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
    return program.add_kernel(config.type, std::move(fn), core);
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

} // namespace tt_emule
