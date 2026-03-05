#pragma once
// Namespace bridge: makes tt_emule symbols available as tt::tt_metal::* so that
// host code written against the tt-metal host API compiles against tt-emule
// with no source changes.
#include "tt_emule/host_api.hpp"
#include "tt_emule/tile.hpp"
#include <memory>
#include <vector>

namespace tt {
namespace tt_metal {

// ---- Type aliases ----
using IDevice              = tt_emule::Device;
using Program              = tt_emule::Program;
using Buffer               = tt_emule::Buffer;
using CBHandle             = tt_emule::CBHandle;
using KernelHandle         = tt_emule::KernelHandle;
using CommandQueue         = tt_emule::CommandQueue;
using CoreCoord            = tt_emule::CoreCoord;
using DataMovementConfig   = tt_emule::DataMovementConfig;
using ComputeConfig        = tt_emule::ComputeConfig;
using KernelType           = tt_emule::KernelType;
using CircularBufferConfig = tt_emule::CircularBufferConfig;
using KernelFn             = tt_emule::KernelFn;

// tt-metal compatibility enums re-exported
using HalMemType           = tt_emule::HalMemType;
using DataMovementProcessor = tt_emule::DataMovementProcessor;
using NOC                  = tt_emule::NOC;

// ---- BufferType + InterleavedBufferConfig ----
enum class BufferType { DRAM, L1 };

struct InterleavedBufferConfig {
    IDevice*   device;
    size_t     size;
    size_t     page_size;
    BufferType buffer_type = BufferType::DRAM;
};

inline std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& cfg) {
    return tt_emule::CreateBuffer(*cfg.device,
                                  cfg.size,
                                  static_cast<uint32_t>(cfg.page_size));
}

// ---- Device lifecycle ----
inline IDevice* CreateDevice(uint32_t id = 0) {
    return tt_emule::CreateDevice(id);
}
inline bool CloseDevice(IDevice* device) {
    return tt_emule::CloseDevice(device);
}
inline CommandQueue CreateCommandQueue(IDevice& device) {
    return tt_emule::CreateCommandQueue(device);
}

// ---- Program construction ----
inline Program CreateProgram() {
    return tt_emule::CreateProgram();
}

// Function-pointer kernel
inline KernelHandle CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                                  DataMovementConfig config) {
    return tt_emule::CreateKernel(program, std::move(fn), core, config);
}
inline KernelHandle CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                                  ComputeConfig config) {
    return tt_emule::CreateKernel(program, std::move(fn), core, config);
}

// JIT kernel (source file path).
// Deliberately calls tt_emule_internal:: (not tt_emule::) to avoid ADL finding
// tt_emule::CreateKernel when test code has `using namespace tt::tt_metal`.
inline KernelHandle CreateKernel(Program& program,
                                  const std::string& kernel_src_path,
                                  CoreCoord core, DataMovementConfig config) {
    return tt_emule_internal::create_jit_kernel(program, kernel_src_path, core, config);
}

// SetRuntimeArgs — vector overload
inline void SetRuntimeArgs(Program& program, KernelHandle kernel_id, CoreCoord core,
                            std::vector<uint32_t> args) {
    tt_emule::SetRuntimeArgs(program, kernel_id, core, std::move(args));
}

// SetRuntimeArgs — initializer_list overload (avoids ADL ambiguity with brace-init)
inline void SetRuntimeArgs(Program& program, KernelHandle kernel_id, CoreCoord core,
                            std::initializer_list<uint32_t> args) {
    tt_emule::SetRuntimeArgs(program, kernel_id, core,
        std::vector<uint32_t>(args));
}

// SetRuntimeArgs — template for std::array and other contiguous containers
template<typename Container>
inline void SetRuntimeArgs(Program& program, KernelHandle kernel_id, CoreCoord core,
                            const Container& args) {
    tt_emule::SetRuntimeArgs(program, kernel_id, core,
        std::vector<uint32_t>(std::begin(args), std::end(args)));
}

inline CBHandle CreateCircularBuffer(Program& program, CoreCoord core,
                                      CircularBufferConfig config) {
    return tt_emule::CreateCircularBuffer(program, core, config);
}

// ---- Buffer operations ----
inline std::shared_ptr<Buffer> CreateBuffer(IDevice& device, size_t size_bytes,
                                             uint32_t page_size_bytes) {
    return tt_emule::CreateBuffer(device, size_bytes, page_size_bytes);
}
inline void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buf, const void* src,
                                bool blocking = true) {
    tt_emule::EnqueueWriteBuffer(cq, buf, src, blocking);
}
inline void EnqueueReadBuffer(CommandQueue& cq, Buffer& buf, void* dst,
                               bool blocking = true) {
    tt_emule::EnqueueReadBuffer(cq, buf, dst, blocking);
}

// ---- Program execution ----
inline void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking = true) {
    tt_emule::EnqueueProgram(cq, program, blocking);
}
inline void Finish(CommandQueue& cq) {
    tt_emule::Finish(cq);
}

// ---- detail:: re-export ----
namespace detail = tt_emule::detail;

} // namespace tt_metal
} // namespace tt
