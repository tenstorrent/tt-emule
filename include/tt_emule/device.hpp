#pragma once
#include "cb_sync_state.hpp"
#include "circular_buffer.hpp"
#include "dfb_sync_state.hpp"
#include "tile_counter.hpp"
#include "dst_register_file.hpp"
#include <array>
#include <memory>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <sys/mman.h>

// When building as part of tt-metal (TT_EMULE_USE_XY_PAIR defined by CMake),
// use the real tt_xy_pair as CoreCoord so that tt::tt_metal::CoreCoord is the
// actual real type.  Standalone builds keep an equivalent custom struct.
#ifdef TT_EMULE_USE_XY_PAIR
#include <umd/device/types/xy_pair.hpp>
// Include the fake IDevice abstract interface BEFORE defining Device so that
// Device can inherit from it.  The fake header is self-contained (no circular deps).
#include <tt-metalium/device.hpp>
#include <set>
#endif

namespace tt_emule {

enum class HalMemType : uint8_t { L1 = 0, DRAM = 1, HOST = 2, COUNT = 3 };

// Role of a Core — determines how its mmap'd region is used.
enum class CoreRole { WORKER, DRAM };

#ifdef TT_EMULE_USE_XY_PAIR
using CoreCoord = tt_xy_pair;
#else
struct CoreCoord {
    size_t x;
    size_t y;
    bool operator==(const CoreCoord& o) const { return x == o.x && y == o.y; }
    std::string str() const {
        return "(" + std::to_string(x) + "," + std::to_string(y) + ")";
    }
};
#endif

class Core {
public:
    static constexpr size_t L1_SIZE = 1024 * 1024; // 1 MB
    static constexpr size_t MAX_CBS = 32;

    // Default constructor: WORKER role, 1 MB L1 mmap'd below 4 GB.
    explicit Core(CoreCoord coord) : coord_(coord) {
        mmap_region(L1_SIZE);
    }

    // Role-aware constructor: mmap mem_size bytes.
    Core(CoreCoord coord, CoreRole role, size_t mem_size)
        : coord_(coord), role_(role), l1_size_(mem_size) {
        mmap_region(mem_size);
    }

    // Construct with external memory (no mmap, no munmap).
    // Used by the memory bridge to wrap EmulatedChip's backing store.
    Core(CoreCoord coord, uint8_t* external_l1, size_t l1_size)
        : coord_(coord), owns_l1_(false), l1_size_(l1_size) {
        l1_ = external_l1;
        l1_base_ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_));
    }

    ~Core() {
        if (owns_l1_ && l1_) munmap(l1_, l1_size_);
    }

    Core(const Core&) = delete;
    Core& operator=(const Core&) = delete;

    CoreCoord coord() const { return coord_; }
    CoreRole  role()  const { return role_; }

    std::shared_ptr<CircularBuffer>& cb(size_t idx) {
        if (idx >= MAX_CBS) throw std::out_of_range("CB index out of range");
        return cbs_[idx];
    }

    DstRegisterFile& dst() { return dst_; }

    uint8_t* l1_ptr(uint32_t offset) { return l1_ + offset; }

    // Raw pointer to start of memory region (L1 or DRAM backing).
    uint8_t* l1_data() { return l1_; }

    // Size of the memory region (regardless of role).
    size_t l1_size() const { return l1_size_; }

    // 32-bit absolute address of the L1 base (valid if mmap succeeded below 4 GB).
    uint32_t l1_base_addr() const { return l1_base_; }

    // Bump allocate `bytes` from L1; returns absolute host address.
    uint32_t l1_alloc(size_t bytes) {
        if (l1_bump_ + bytes > l1_size_)
            throw std::runtime_error("L1 OOM");
        uint32_t addr = l1_base_ + static_cast<uint32_t>(l1_bump_);
        l1_bump_ += bytes;
        return addr;
    }

    // ---- CB sync state array (for JIT kernel threads) ----

    CBSyncState* cb_sync_array() { return cb_sync_states_; }

    void init_cb_sync(uint32_t idx, uint8_t* base, uint32_t page_size, uint32_t num_pages) {
        if (idx >= MAX_CBS) return;
        auto& s = cb_sync_states_[idx];
        s.base      = base;
        s.page_size = page_size;
        s.num_pages = num_pages;
        s.page_mask = (num_pages > 0 && (num_pages & (num_pages - 1)) == 0) ? num_pages - 1 : 0;
        s.write_idx = 0;
        s.read_idx  = 0;
        s.occupied  = 0;
    }

    void reset_cb_sync() {
        for (auto& s : cb_sync_states_) {
            s.base      = nullptr;
            s.page_size = 0;
            s.num_pages = 0;
            s.page_mask = 0;
            s.write_idx = 0;
            s.read_idx  = 0;
            s.occupied  = 0;
        }
    }

    // ---- DFB / Tile Counter infrastructure (Quasar) ----

    void init_tile_counters(uint32_t num_neos) {
        tile_counters_ = std::make_unique<TileCounterArray>(num_neos);
    }

    TileCounterArray* tile_counters() { return tile_counters_.get(); }

    DFBSyncState* dfb_sync_array() { return dfb_sync_states_; }

    void init_dfb_sync(uint32_t idx, uint8_t* base, uint32_t entry_size,
                       uint32_t num_entries, uint32_t capacity) {
        if (idx >= MAX_DFBS) return;
        auto& s = dfb_sync_states_[idx];
        s.base            = base;
        s.entry_size      = entry_size;
        s.num_entries     = num_entries;
        s.capacity        = capacity;
        s.stride_in_entries = 1;
    }

    void reset_dfb_sync() {
        for (auto& s : dfb_sync_states_) {
            s.base = nullptr;
            s.entry_size = 0;
            s.num_entries = 0;
            s.capacity = 0;
            s.stride_in_entries = 1;
        }
        if (tile_counters_) tile_counters_->reset_all();
    }

private:
    void mmap_region(size_t size) {
        l1_size_ = size;
        // Worker cores need MAP_32BIT so CB pointers fit in uint32_t.
        // DRAM cores are accessed via bridge functions (full 64-bit pointers),
        // so they use regular mmap to avoid exhausting the low 2 GB space.
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (role_ == CoreRole::WORKER) flags |= MAP_32BIT;
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (p == MAP_FAILED)
            throw std::runtime_error("mmap for Core memory failed");
        l1_ = static_cast<uint8_t*>(p);
        l1_base_ = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(l1_));
        // MAP_ANONYMOUS guarantees zero-filled pages; no memset needed.
    }

    CoreCoord coord_;
    CoreRole  role_    = CoreRole::WORKER;
    bool      owns_l1_ = true;  // false when using external memory
    size_t    l1_size_ = L1_SIZE;
    uint8_t*  l1_      = nullptr;
    uint32_t  l1_base_ = 0;
    size_t    l1_bump_ = 0;  // current L1 bump allocator offset
    std::array<std::shared_ptr<CircularBuffer>, MAX_CBS> cbs_;
    DstRegisterFile dst_;
    CBSyncState cb_sync_states_[MAX_CBS] = {};
    // Quasar DFB state
    std::unique_ptr<TileCounterArray> tile_counters_;
    DFBSyncState dfb_sync_states_[MAX_DFBS] = {};
};

enum class BufferType { DRAM, L1, SYSTEM_MEMORY, L1_SMALL, TRACE };

// Minimal allocator mimic for tt-metal compat.
class MockAllocator {
    uint32_t l1_base_;
    size_t   l1_size_;
public:
    explicit MockAllocator(uint32_t base, size_t l1_size = Core::L1_SIZE)
        : l1_base_(base), l1_size_(l1_size) {}

    uint32_t get_base_allocator_addr(HalMemType type) const {
        return (type == HalMemType::L1) ? l1_base_ : 0;
    }

    // Returns DRAM channel 0 for any bank id (single-channel prototype).
    uint32_t get_dram_channel_from_bank_id(uint32_t /*bank_id*/) const { return 0; }

    // Returns the logical core coordinate for bank 0 → always (0,0).
    CoreCoord get_logical_core_from_bank_id(uint32_t /*bank_id*/) const { return {0, 0}; }

    // Returns bank IDs for a given core. Prototype: bank 0 for everything.
    std::vector<uint32_t> get_bank_ids_from_logical_core(BufferType /*type*/, CoreCoord /*core*/) const {
        return {0};
    }

    // L1 bank size = total L1 size (single bank in prototype).
    size_t get_bank_size(BufferType /*type*/) const { return l1_size_; }
};

#ifdef TT_EMULE_USE_XY_PAIR

// Device inherits from tt::tt_metal::IDevice when built as part of tt-metal,
// so that IDevice* is a proper abstract polymorphic type.
class Device : public tt::tt_metal::IDevice {
public:
    static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024; // 256 MB

    // Constructor and destructor defined out-of-line in host_api.cpp where
    // tt::tt_metal::Allocator / AllocatorImpl are complete types.
    Device();
    ~Device() override;

    // Bump allocator for L1 — returns absolute host address
    uint32_t l1_alloc(size_t bytes) { return core_.l1_alloc(bytes); }

    // Bump allocator for DRAM
    uint64_t dram_alloc(size_t bytes) {
        if (dram_bump_ + bytes > DRAM_SIZE)
            throw std::runtime_error("DRAM OOM");
        uint64_t offset = dram_bump_;
        dram_bump_ += bytes;
        return offset;
    }

    uint8_t* dram_ptr(uint64_t offset) { return dram_.data() + offset; }

    // Map (x, y, addr) to raw pointer.
    uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
        if (x == 0 && y == 0)
            return core_.l1_ptr(static_cast<uint32_t>(addr));
        return dram_ptr(addr);
    }

    Core& core() { return core_; }

    // Access the raw MockAllocator (renamed to avoid conflict with IDevice::allocator()).
    MockAllocator* mock_allocator() { return &alloc_; }

    // ---- IDevice pure virtual implementations ----

    tt::ARCH arch() const override { return tt::ARCH::WORMHOLE_B0; }
    tt::tt_metal::ChipId id() const override { return 0; }
    tt::tt_metal::ChipId build_id() const override { return 0; }
    uint8_t num_hw_cqs() const override { return 1; }
    bool is_initialized() const override { return true; }
    int num_dram_channels() const override { return 1; }
    uint32_t l1_size_per_core() const override {
        return static_cast<uint32_t>(Core::L1_SIZE);
    }
    uint32_t dram_size_per_channel() const override {
        return static_cast<uint32_t>(DRAM_SIZE);
    }
    int get_clock_rate_mhz() const override { return 1000; }

    tt::tt_metal::CoreCoord grid_size() const override { return {1, 1}; }
    tt::tt_metal::CoreCoord logical_grid_size() const override { return {1, 1}; }
    tt::tt_metal::CoreCoord dram_grid_size() const override { return {1, 1}; }

    tt::tt_metal::CoreCoord virtual_noc0_coordinate(
            uint8_t /*noc_index*/, tt::tt_metal::CoreCoord coord) const override {
        return coord;
    }
    std::vector<tt::tt_metal::CoreCoord> worker_cores_from_logical_cores(
            const std::vector<tt::tt_metal::CoreCoord>& v) const override {
        return v;
    }
    std::vector<tt::tt_metal::CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment(
            tt::tt_metal::NOC /*noc*/) override {
        return {};
    }
    tt::tt_metal::CoreCoord virtual_core_from_logical_core(
            const tt::tt_metal::CoreCoord& c, const tt::CoreType& /*type*/) const override {
        return c;
    }
    tt::tt_metal::CoreCoord worker_core_from_logical_core(
            const tt::tt_metal::CoreCoord& c) const override {
        return c;
    }
    tt::tt_metal::CoreCoord compute_with_storage_grid_size() const override { return {1, 1}; }

    // worker_cores returns CoreRangeSet — defined out-of-line in host_api.cpp
    // where CoreRangeSet is complete (via #include <tt-metalium/core_coord.hpp>).
    tt::tt_metal::CoreRangeSet worker_cores(
        tt::tt_metal::HalProgrammableCoreType core_type,
        tt::tt_metal::SubDeviceId sub_device_id) const override;

    uint32_t num_worker_cores(
            tt::tt_metal::HalProgrammableCoreType /*core_type*/,
            tt::tt_metal::SubDeviceId /*sub_device_id*/) const override {
        return 1u;
    }

    const std::unique_ptr<tt::tt_metal::Allocator>& allocator() const override {
        return allocator_;
    }
    const std::unique_ptr<tt::tt_metal::Allocator>& allocator(
            tt::tt_metal::SubDeviceId /*sub_device_id*/) const override {
        return allocator_;
    }
    const std::unique_ptr<tt::tt_metal::AllocatorImpl>& allocator_impl() const override {
        return allocator_impl_;
    }
    const std::unique_ptr<tt::tt_metal::AllocatorImpl>& allocator_impl(
            tt::tt_metal::SubDeviceId /*sub_device_id*/) const override {
        return allocator_impl_;
    }

    tt::tt_metal::CoreCoord logical_core_from_dram_channel(uint32_t /*ch*/) const override {
        return {0, 0};
    }
    uint32_t dram_channel_from_logical_core(
            const tt::tt_metal::CoreCoord& /*c*/) const override { return 0u; }
    uint32_t dram_channel_from_virtual_core(
            const tt::tt_metal::CoreCoord& /*c*/) const override { return 0u; }

    std::optional<tt::tt_metal::DeviceAddr> lowest_occupied_compute_l1_address()
            const override {
        return std::nullopt;
    }
    std::optional<tt::tt_metal::DeviceAddr> lowest_occupied_compute_l1_address(
            tt::stl::Span<const tt::tt_metal::SubDeviceId> /*ids*/) const override {
        return std::nullopt;
    }

    const std::set<tt::tt_metal::CoreCoord>& storage_only_cores() const override {
        return storage_only_cores_;
    }

    uint32_t get_noc_unicast_encoding(
            uint8_t /*noc_index*/, const tt::tt_metal::CoreCoord& /*core*/) const override {
        return 0u;
    }
    uint32_t get_noc_multicast_encoding(
            uint8_t /*noc_index*/,
            const tt::tt_metal::CoreRange& /*cores*/) const override {
        return 0u;
    }

    tt::tt_metal::SystemMemoryManager& sysmem_manager() override {
        throw std::logic_error("sysmem_manager: not implemented in emulator");
    }

    uint32_t get_trace_buffers_size() const override { return 0u; }
    void set_trace_buffers_size(uint32_t /*size*/) override {}

    bool initialize(uint8_t /*num_hw_cqs*/, size_t /*l1_small_size*/,
                    size_t /*trace_region_size*/, size_t /*worker_l1_size*/,
                    tt::stl::Span<const std::uint32_t> /*l1_bank_remap*/ = {},
                    bool /*minimal*/ = false) override { return true; }
    void init_command_queue_host() override {}
    void init_command_queue_device() override {}
    bool compile_fabric() override { return true; }
    void configure_fabric() override {}
    bool close() override { return true; }

    void enable_program_cache() override {}
    void clear_program_cache() override {}
    void disable_and_clear_program_cache() override {}
    tt::tt_metal::program_cache::detail::ProgramCache& get_program_cache() override {
        throw std::logic_error("get_program_cache: not implemented in emulator");
    }
    std::size_t num_program_cache_entries() override { return 0; }

    tt::tt_metal::HalProgrammableCoreType get_programmable_core_type(
            tt::tt_metal::CoreCoord /*virtual_core*/) const override {
        return tt::tt_metal::HalProgrammableCoreType::TENSIX;
    }
    HalMemType get_mem_type_of_core(
            tt::tt_metal::CoreCoord /*virtual_core*/) const override {
        return HalMemType::L1;
    }

    bool has_noc_mcast_txns(tt::tt_metal::SubDeviceId /*id*/) const override { return false; }
    uint8_t num_noc_unicast_txns(tt::tt_metal::SubDeviceId /*id*/) const override { return 0u; }
    uint8_t noc_data_start_index(
            tt::tt_metal::SubDeviceId /*id*/, bool /*unicast*/ = true) const override {
        return 0u;
    }

    tt::tt_metal::SubDeviceManagerId get_active_sub_device_manager_id() const override {
        return tt::tt_metal::SubDeviceManagerId{0};
    }
    tt::tt_metal::SubDeviceManagerId get_default_sub_device_manager_id() const override {
        return tt::tt_metal::SubDeviceManagerId{0};
    }
    tt::tt_metal::SubDeviceManagerId create_sub_device_manager(
            tt::stl::Span<const tt::tt_metal::SubDevice> /*subs*/,
            tt::tt_metal::DeviceAddr /*sz*/) override {
        return tt::tt_metal::SubDeviceManagerId{0};
    }
    tt::tt_metal::SubDeviceManagerId create_sub_device_manager(
            std::initializer_list<tt::tt_metal::SubDevice> /*subs*/,
            tt::tt_metal::DeviceAddr /*sz*/) override {
        return tt::tt_metal::SubDeviceManagerId{0};
    }
    void remove_sub_device_manager(
            tt::tt_metal::SubDeviceManagerId /*id*/) override {}
    void load_sub_device_manager(
            tt::tt_metal::SubDeviceManagerId /*id*/) override {}
    void clear_loaded_sub_device_manager() override {}
    tt::tt_metal::CoreCoord virtual_program_dispatch_core(uint8_t /*cq_id*/) const override {
        return {0, 0};
    }
    const std::vector<tt::tt_metal::SubDeviceId>& get_sub_device_ids() const override {
        return sub_device_ids_;
    }
    const std::vector<tt::tt_metal::SubDeviceId>& get_sub_device_stall_group() const override {
        return sub_device_stall_group_;
    }
    void set_sub_device_stall_group(
            tt::stl::Span<const tt::tt_metal::SubDeviceId> /*ids*/) override {}
    void reset_sub_device_stall_group() override {}
    uint32_t num_sub_devices() const override { return 1u; }
    uint32_t num_virtual_eth_cores(tt::tt_metal::SubDeviceId /*id*/) override { return 0u; }

    bool is_mmio_capable() const override { return true; }

    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() override {
        return nullptr;
    }

    std::vector<tt::tt_metal::CoreCoord> ethernet_cores_from_logical_cores(
            const std::vector<tt::tt_metal::CoreCoord>& /*cores*/) const override {
        return {};
    }
    tt::tt_metal::CoreCoord logical_core_from_ethernet_core(
            const tt::tt_metal::CoreCoord& /*c*/) const override { return {0, 0}; }
    tt::tt_metal::CoreCoord ethernet_core_from_logical_core(
            const tt::tt_metal::CoreCoord& /*c*/) const override { return {0, 0}; }
    std::unordered_set<tt::tt_metal::CoreCoord> get_active_ethernet_cores(
            bool /*skip_reserved*/ = false) const override { return {}; }
    std::unordered_set<tt::tt_metal::CoreCoord> get_inactive_ethernet_cores()
            const override { return {}; }
    bool is_active_ethernet_core(tt::tt_metal::CoreCoord /*c*/,
            bool /*skip_reserved*/ = false) const override { return false; }
    bool is_inactive_ethernet_core(tt::tt_metal::CoreCoord /*c*/) const override {
        return false;
    }
    std::tuple<tt::tt_metal::ChipId, tt::tt_metal::CoreCoord>
    get_connected_ethernet_core(tt::tt_metal::CoreCoord /*c*/) const override {
        return {0, {0, 0}};
    }
    std::vector<tt::tt_metal::CoreCoord> get_ethernet_sockets(
            tt::tt_metal::ChipId /*id*/) const override { return {}; }
    const std::set<tt::tt_metal::CoreCoord>& ethernet_cores() const override {
        return ethernet_cores_;
    }

private:
    std::vector<uint8_t> dram_;
    uint64_t             dram_bump_;
    Core                 core_;
    MockAllocator        alloc_;

    // IDevice-required members (initialized in out-of-line constructor).
    std::unique_ptr<tt::tt_metal::Allocator>     allocator_;
    std::unique_ptr<tt::tt_metal::AllocatorImpl> allocator_impl_;
    std::vector<tt::tt_metal::SubDeviceId>       sub_device_ids_;
    std::vector<tt::tt_metal::SubDeviceId>       sub_device_stall_group_;
    std::set<tt::tt_metal::CoreCoord>            storage_only_cores_;
    std::set<tt::tt_metal::CoreCoord>            ethernet_cores_;
};

#else  // !TT_EMULE_USE_XY_PAIR

class Device {
public:
    static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024; // 256 MB

    Device() : dram_(DRAM_SIZE, 0), dram_bump_(0),
               core_({0, 0}), alloc_(core_.l1_base_addr()) {}

    // Construct with external memory (no ownership). Used by the memory bridge.
    Device(uint8_t* external_dram, size_t dram_size,
           uint8_t* external_l1, size_t l1_size)
        : ext_dram_(external_dram), ext_dram_size_(dram_size),
          dram_bump_(0),
          core_({0, 0}, external_l1, l1_size),
          alloc_(core_.l1_base_addr(), l1_size) {}

    // Bump allocator for L1 — returns absolute host address
    uint32_t l1_alloc(size_t bytes) { return core_.l1_alloc(bytes); }

    // Bump allocator for DRAM
    uint64_t dram_alloc(size_t bytes) {
        size_t cap = ext_dram_ ? ext_dram_size_ : dram_.size();
        if (dram_bump_ + bytes > cap)
            throw std::runtime_error("DRAM OOM");
        uint64_t offset = dram_bump_;
        dram_bump_ += bytes;
        return offset;
    }

    uint8_t* dram_ptr(uint64_t offset) {
        return ext_dram_ ? (ext_dram_ + offset) : (dram_.data() + offset);
    }

    // Map (x, y, addr) to raw pointer.
    uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr) {
        if (x == 0 && y == 0)
            return core_.l1_ptr(static_cast<uint32_t>(addr));
        return dram_ptr(addr);
    }

    Core& core() { return core_; }

    MockAllocator* allocator() { return &alloc_; }

    // Alias for allocator() — tt-metal uses allocator_impl() for low-level access.
    MockAllocator* allocator_impl() { return &alloc_; }

    // Total DRAM per channel (prototype has 1 channel).
    size_t dram_size_per_channel() const { return ext_dram_ ? ext_dram_size_ : DRAM_SIZE; }

    // Total L1 per core.
    size_t l1_size_per_core() const { return Core::L1_SIZE; }

    // Physical == logical in prototype (no NOC translation).
    CoreCoord worker_core_from_logical_core(CoreCoord c) const { return c; }

private:
    std::vector<uint8_t> dram_;
    uint8_t*             ext_dram_ = nullptr;      // external DRAM (no ownership)
    size_t               ext_dram_size_ = 0;
    uint64_t             dram_bump_;
    Core                 core_;
    MockAllocator        alloc_;
};

#endif  // TT_EMULE_USE_XY_PAIR

} // namespace tt_emule
