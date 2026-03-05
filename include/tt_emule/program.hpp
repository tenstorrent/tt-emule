#pragma once
#include "circular_buffer.hpp"
#include "device.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace tt_emule {

using KernelFn = std::function<void()>;

enum class KernelType {
    DataMovement0, // NOC reader (DM thread 0)
    DataMovement1, // NOC writer (DM thread 1)
    Compute,
};

// tt-metal compatibility enums
enum class DataMovementProcessor { RISCV_0, RISCV_1 };
enum class NOC { RISCV_0_default, RISCV_1_default };

struct DataMovementConfig {
    // type is derived from processor; defaults to DataMovement0
    KernelType             type      = KernelType::DataMovement0;
    DataMovementProcessor  processor = DataMovementProcessor::RISCV_0;
    NOC                    noc       = NOC::RISCV_0_default;
    std::vector<uint32_t>  compile_args;
    // JIT path: non-empty means this is a source-file kernel
    std::string            kernel_src_path;
};

struct ComputeConfig {
    // placeholder for future options
};

struct KernelDescriptor {
    uint32_t id;
    KernelType type;
    KernelFn fn;
    CoreCoord core;
    std::vector<uint32_t> rt_args;
};

using CBHandle = uint32_t;

struct CircularBufferConfig {
    uint32_t cb_index;    // 0-31
    size_t   num_pages;   // capacity in tiles
    uint32_t page_size;   // bytes per page (should equal Tile::SIZE_BYTES)
};

class Program {
public:
    uint32_t add_kernel(KernelType type, KernelFn fn, CoreCoord core) {
        uint32_t id = static_cast<uint32_t>(kernels_.size());
        kernels_.push_back({id, type, std::move(fn), core, {}});
        return id;
    }

    void set_runtime_args(uint32_t kernel_id, std::vector<uint32_t> args) {
        kernels_.at(kernel_id).rt_args = std::move(args);
    }

    CBHandle add_cb(CircularBufferConfig cfg) {
        CBHandle h = static_cast<CBHandle>(cb_configs_.size());
        cb_configs_.push_back(cfg);
        return h;
    }

    std::vector<KernelDescriptor>& kernels() { return kernels_; }
    std::vector<CircularBufferConfig>& cb_configs() { return cb_configs_; }

private:
    std::vector<KernelDescriptor> kernels_;
    std::vector<CircularBufferConfig> cb_configs_;
};

} // namespace tt_emule
