#pragma once
// Namespace bridge: makes tt_emule symbols available as tt::tt_metal::* so that
// host code written against the tt-metal host API compiles against tt-emule
// with no source changes.
#include "tt_emule/host_api.hpp"
#include "tt_emule/tile.hpp"
#include <map>
#include <memory>
#include <vector>
// Include tt_backend_api_types for tt::DataFormat (used in CircularBufferConfig compat ctor).
// Resolves to fake_headers/tt-metalium/tt_backend_api_types.hpp in tt-metal builds.
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt {
namespace tt_metal {

// ---- Type aliases ----
// In TT_EMULE_USE_XY_PAIR builds, IDevice is the abstract tt::tt_metal::IDevice
// (defined in fake_headers/tt-metalium/device.hpp, already included via
// tt_emule/device.hpp → <tt-metalium/device.hpp>).
// In standalone builds, IDevice is a direct alias for tt_emule::Device.
#ifdef TT_EMULE_USE_XY_PAIR
using IDevice              = tt::tt_metal::IDevice;
#else
using IDevice              = tt_emule::Device;
#endif
using Program              = tt_emule::Program;
using Buffer               = tt_emule::Buffer;
using CBHandle             = tt_emule::CBHandle;
using KernelHandle         = tt_emule::KernelHandle;
using CommandQueue         = tt_emule::CommandQueue;
// When TT_EMULE_USE_XY_PAIR is defined, tt_emule::CoreCoord IS tt_xy_pair,
// so this alias makes tt::tt_metal::CoreCoord the real type.
using CoreCoord            = tt_emule::CoreCoord;
using DataMovementConfig   = tt_emule::DataMovementConfig;
using ComputeConfig        = tt_emule::ComputeConfig;
using KernelType           = tt_emule::KernelType;
// CircularBufferConfig: extends tt_emule base with a compat constructor that
// accepts the real tt-metal API form: (total_size, {{idx, tt::DataFormat}}).
struct CircularBufferConfig : tt_emule::CircularBufferConfig {
    using tt_emule::CircularBufferConfig::CircularBufferConfig;

    // Real tt-metal constructor: (total_size, std::map<uint8_t, tt::DataFormat>).
    // Brace-init {{idx, DataFormat_val}} constructs the map via its initializer_list ctor.
    CircularBufferConfig(size_t total_size,
                         const std::map<uint8_t, tt::DataFormat>& spec)
        : tt_emule::CircularBufferConfig() {
        (void)total_size;
        if (!spec.empty())
            cb_index = static_cast<uint32_t>(spec.begin()->first);
    }

    // Override set_page_size to return the derived type so chained construction works:
    // CircularBufferConfig(...).set_page_size(idx, sz) → CircularBufferConfig
    CircularBufferConfig& set_page_size(uint32_t idx, uint32_t psize) {
        tt_emule::CircularBufferConfig::set_page_size(idx, psize);
        return *this;
    }
};
using KernelFn             = tt_emule::KernelFn;

// tt-metal compatibility enums re-exported
using HalMemType            = tt_emule::HalMemType;
using DataMovementProcessor = tt_emule::DataMovementProcessor;
using NOC                   = tt_emule::NOC;
using NOC_MODE              = tt_emule::NOC_MODE;
using MathFidelity          = tt_emule::MathFidelity;
using UnpackToDestMode      = tt_emule::UnpackToDestMode;

// ---- BufferType + InterleavedBufferConfig ----
using BufferType = tt_emule::BufferType;

struct InterleavedBufferConfig {
    IDevice*   device;
    size_t     size;
    size_t     page_size;
    BufferType buffer_type = BufferType::DRAM;
};

inline std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& cfg) {
    // InterleavedBufferConfig always uses DRAM in the emulator.
#ifdef TT_EMULE_USE_XY_PAIR
    return tt_emule::CreateBuffer(*static_cast<tt_emule::Device*>(cfg.device),
                                  cfg.size,
                                  static_cast<uint32_t>(cfg.page_size),
                                  BufferType::DRAM);
#else
    return tt_emule::CreateBuffer(*cfg.device,
                                  cfg.size,
                                  static_cast<uint32_t>(cfg.page_size),
                                  BufferType::DRAM);
#endif
}

// ---- Device lifecycle ----
inline IDevice* CreateDevice(uint32_t id = 0) {
    return tt_emule::CreateDevice(id);
}
inline bool CloseDevice(IDevice* device) {
#ifdef TT_EMULE_USE_XY_PAIR
    return tt_emule::CloseDevice(static_cast<tt_emule::Device*>(device));
#else
    return tt_emule::CloseDevice(device);
#endif
}
inline CommandQueue CreateCommandQueue(IDevice& device) {
#ifdef TT_EMULE_USE_XY_PAIR
    return tt_emule::CreateCommandQueue(static_cast<tt_emule::Device&>(device));
#else
    return tt_emule::CreateCommandQueue(device);
#endif
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
                                             uint32_t page_size_bytes,
                                             BufferType type = BufferType::DRAM) {
#ifdef TT_EMULE_USE_XY_PAIR
    return tt_emule::CreateBuffer(static_cast<tt_emule::Device&>(device),
                                  size_bytes, page_size_bytes, type);
#else
    return tt_emule::CreateBuffer(device, size_bytes, page_size_bytes, type);
#endif
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

// ---- BufferType for allocator queries ----
// (defined here so MockAllocator can reference it)
// Note: tt_emule::BufferType forward-declared in device.hpp; defined here.

// ---- detail:: namespace ----
// Under TT_EMULE_USE_XY_PAIR, functions that take Device* need a downcast
// from IDevice*. We expose explicit inline wrappers instead of a namespace alias.
#ifdef TT_EMULE_USE_XY_PAIR

namespace detail {

inline void LaunchProgram(IDevice* device, tt_emule::Program& program,
                          bool wait = true, bool /*force_slow*/ = false) {
    tt_emule::detail::LaunchProgram(
        static_cast<tt_emule::Device*>(device), program, wait);
}

inline bool ReadFromDeviceL1(IDevice* device, const tt_emule::CoreCoord& core,
                              uint32_t address, uint32_t size,
                              std::vector<uint32_t>& result) {
    return tt_emule::detail::ReadFromDeviceL1(
        static_cast<tt_emule::Device*>(device), core, address, size, result);
}

inline void WriteToDeviceL1(IDevice* device, const tt_emule::CoreCoord& core,
                             uint32_t address, const std::vector<uint32_t>& data) {
    tt_emule::detail::WriteToDeviceL1(
        static_cast<tt_emule::Device*>(device), core, address, data);
}

inline void WriteToDeviceL1(IDevice* device, const tt_emule::CoreCoord& core,
                             uint32_t address, std::span<const uint8_t> data) {
    tt_emule::detail::WriteToDeviceL1(
        static_cast<tt_emule::Device*>(device), core, address, data);
}

inline void ReadFromDeviceL1(IDevice* device, const tt_emule::CoreCoord& core,
                              uint32_t address, std::span<uint8_t> data) {
    tt_emule::detail::ReadFromDeviceL1(
        static_cast<tt_emule::Device*>(device), core, address, data);
}

inline void WriteToDeviceDRAMChannel(IDevice* device, uint32_t channel,
                                     uint32_t address,
                                     const std::vector<uint32_t>& data) {
    tt_emule::detail::WriteToDeviceDRAMChannel(
        static_cast<tt_emule::Device*>(device), channel, address, data);
}

inline void WriteToDeviceDRAMChannel(IDevice* device, uint32_t channel,
                                     uint32_t address,
                                     std::span<const uint8_t> data) {
    tt_emule::detail::WriteToDeviceDRAMChannel(
        static_cast<tt_emule::Device*>(device), channel, address, data);
}

inline void ReadFromDeviceDRAMChannel(IDevice* device, uint32_t channel,
                                      uint32_t address, uint32_t byte_size,
                                      std::vector<uint32_t>& result) {
    tt_emule::detail::ReadFromDeviceDRAMChannel(
        static_cast<tt_emule::Device*>(device), channel, address, byte_size, result);
}

inline void ReadFromDeviceDRAMChannel(IDevice* device, uint32_t channel,
                                      uint32_t address, std::span<uint8_t> data) {
    tt_emule::detail::ReadFromDeviceDRAMChannel(
        static_cast<tt_emule::Device*>(device), channel, address, data);
}

template<typename T>
inline void WriteToBuffer(const std::shared_ptr<tt_emule::Buffer>& buf,
                          const std::vector<T>& data) {
    tt_emule::detail::WriteToBuffer(buf, data);
}

template<typename T>
inline void ReadFromBuffer(const std::shared_ptr<tt_emule::Buffer>& buf,
                           std::vector<T>& data) {
    tt_emule::detail::ReadFromBuffer(buf, data);
}

}  // namespace detail

#else  // !TT_EMULE_USE_XY_PAIR

namespace detail = tt_emule::detail;

#endif  // TT_EMULE_USE_XY_PAIR

} // namespace tt_metal
} // namespace tt
