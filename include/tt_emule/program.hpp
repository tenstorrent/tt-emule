#pragma once
#include "circular_buffer.hpp"
#include "device.hpp"
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstdint>

namespace tt_emule {

using KernelFn = std::function<void()>;

enum class KernelType {
    // Wormhole kernel types
    DataMovement0,
    DataMovement1,
    Compute,
    // Quasar kernel types (processor_id field carries the specific slot)
    QuasarDM,       // DM0-DM7, processor_id = 0-7
    QuasarCompute,  // E0_MATH0-E3_MATH3, processor_id = 8-23
};

// tt-metal compatibility enums
enum class DataMovementProcessor {
    RISCV_0 = 0, RISCV_1 = 1,
    RISCV_2 = 2, RISCV_3 = 3, RISCV_4 = 4,
    RISCV_5 = 5, RISCV_6 = 6, RISCV_7 = 7,
};
enum class NOC : uint8_t { RISCV_0_default = 0, RISCV_1_default = 1, NOC_0 = 0, NOC_1 = 1 };
enum class NOC_MODE : uint8_t { DM_DEDICATED_NOC = 0, DM_DYNAMIC_NOC = 1 };
enum class MathFidelity : uint8_t { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4, Invalid = 0xff };
enum class UnpackToDestMode : uint8_t { UnpackToDestFp32, Default };

// DFB access pattern (Quasar)
enum class AccessPattern : uint8_t { STRIDED = 0, BLOCKED = 1 };

struct DataMovementConfig {
    KernelType             type      = KernelType::DataMovement0;
    DataMovementProcessor  processor = DataMovementProcessor::RISCV_0;
    NOC                    noc       = NOC::RISCV_0_default;
    NOC_MODE               noc_mode  = NOC_MODE::DM_DEDICATED_NOC;
    std::vector<uint32_t>  compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    std::string            kernel_src_path;  // JIT: non-empty = source file kernel
};

struct ComputeConfig {
    MathFidelity                  math_fidelity        = MathFidelity::HiFi4;
    bool                          fp32_dest_acc_en     = false;
    bool                          dst_full_sync_en     = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool                          bfp8_pack_precise    = false;
    bool                          math_approx_mode     = false;
    std::vector<uint32_t>         compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
};

// Quasar kernel config structs (emulation-side equivalents of upstream)
struct QuasarDataMovementConfig {
    uint32_t num_threads_per_cluster = 8;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
    bool is_legacy_kernel = false;
};

struct QuasarComputeConfig {
    uint32_t num_threads_per_cluster = 4;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    bool fp32_dest_acc_en = false;
    bool dst_full_sync_en = false;
    std::vector<UnpackToDestMode> unpack_to_dest_mode;
    bool bfp8_pack_precise = false;
    bool math_approx_mode = false;
    std::vector<uint32_t> compile_args;
    std::map<std::string, std::string> defines;
    std::unordered_map<std::string, uint32_t> named_compile_args;
};

struct KernelDescriptor {
    uint32_t id;
    KernelType type;
    KernelFn fn;
    CoreCoord core;
    std::vector<uint32_t> rt_args;
    uint8_t processor_id = 0;  // Quasar: TensixProcessorTypes index (0-23)
};

using CBHandle = uint32_t;
using DFBHandle = uint32_t;

// Quasar DFB configuration
struct DataflowBufferConfig {
    uint32_t dfb_index     = 0;
    uint32_t entry_size    = 0;
    uint32_t num_entries   = 0;
    uint16_t producer_risc_mask = 0x0;  // bits 0-7 = DM, 8-15 = Tensix Neo
    uint8_t  num_producers = 1;
    AccessPattern producer_access_pattern = AccessPattern::STRIDED;
    uint16_t consumer_risc_mask = 0x0;
    uint8_t  num_consumers = 1;
    AccessPattern consumer_access_pattern = AccessPattern::STRIDED;
};

struct CircularBufferConfig {
    uint32_t cb_index    = 0;  // 0-31
    size_t   num_pages   = 0;  // capacity in tiles
    uint32_t page_size   = 0;  // bytes per page (should equal Tile::SIZE_BYTES)

    // Default constructor
    CircularBufferConfig() = default;

    // Original aggregate-style construction
    CircularBufferConfig(uint32_t idx, size_t pages, uint32_t psize)
        : cb_index(idx), num_pages(pages), page_size(psize) {}

    // tt-metal compat constructor: (total_size, {{index, data_format}}).
    // DataFormat values are ignored; only the first CB index is extracted.
    template <typename K, typename V>
    CircularBufferConfig(size_t total_size, std::map<K, V> index_format_map)
        : num_pages(0), page_size(0) {
        (void)total_size;
        for (const auto& [idx, fmt] : index_format_map) {
            cb_index = static_cast<uint32_t>(idx);
            break;
        }
    }

    // Brace-initializer-list overload: (total_size, {{index, data_format}}).
    // Deduces K/V from std::initializer_list so brace-enclosed argument works.
    template <typename K, typename V>
    CircularBufferConfig(size_t total_size,
                         std::initializer_list<std::pair<const K, V>> spec)
        : num_pages(0), page_size(0) {
        (void)total_size;
        for (const auto& [idx, fmt] : spec) {
            cb_index = static_cast<uint32_t>(idx);
            break;
        }
    }

    // Builder: set_page_size(cb_index, size) → returns *this for chaining.
    CircularBufferConfig& set_page_size(uint32_t /*idx*/, uint32_t psize) {
        page_size = psize;
        return *this;
    }
};

class Program {
public:
    uint32_t add_kernel(KernelType type, KernelFn fn, CoreCoord core,
                        uint8_t processor_id = 0) {
        uint32_t id = static_cast<uint32_t>(kernels_.size());
        kernels_.push_back({id, type, std::move(fn), core, {}, processor_id});
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

    DFBHandle add_dfb(DataflowBufferConfig cfg) {
        DFBHandle h = static_cast<DFBHandle>(dfb_configs_.size());
        cfg.dfb_index = h;
        dfb_configs_.push_back(cfg);
        return h;
    }

    std::vector<KernelDescriptor>& kernels() { return kernels_; }
    std::vector<CircularBufferConfig>& cb_configs() { return cb_configs_; }
    std::vector<DataflowBufferConfig>& dfb_configs() { return dfb_configs_; }

    bool has_dfbs() const { return !dfb_configs_.empty(); }

private:
    std::vector<KernelDescriptor> kernels_;
    std::vector<CircularBufferConfig> cb_configs_;
    std::vector<DataflowBufferConfig> dfb_configs_;
};

} // namespace tt_emule
