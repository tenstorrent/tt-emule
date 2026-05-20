// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Verifies that TT_METAL_EMULE_MODE env var actually toggles between
// emulated and silicon execution at runtime from the same binary.
//
// Run the SAME binary twice to prove the toggle:
//   Phase 1 (env var unset):  SiliconActive tests activate, EmulationActive tests skip
//   Phase 2 (env var set):    EmulationActive tests activate, SiliconActive tests skip

#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <llrt/tt_cluster.hpp>

#include "device_fixture.hpp"
#include "buffer_test_utils.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#ifdef TT_METAL_EMULATION
#include "impl/emulation/emulated_run_stats.hpp"
#endif

using namespace tt::tt_metal;

// ============================================================================
// Group 1: EmulationToggle -- host-only checks, no MetalContext needed.
// Safe to run without any emulation env vars (Tier 1 / silicon).
// ============================================================================

TEST(EmulationToggle, BuildFlagCompiled) {
#ifdef TT_METAL_EMULATION
    SUCCEED() << "TT_METAL_EMULATION is defined -- emulated code paths are compiled into this binary";
#else
    FAIL() << "TT_METAL_EMULATION is NOT defined -- "
              "emulated code paths (#ifdef TT_METAL_EMULATION) do not exist in this binary. "
              "The env var TT_METAL_EMULE_MODE alone cannot enable emulation without this build flag.";
#endif
}

TEST(EmulationToggle, DefaultIsNotEmulated) {
    const char* env = std::getenv("TT_METAL_EMULE_MODE");
    if (env != nullptr) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE is set -- run without it to test the default path";
    }
    SUCCEED() << "TT_METAL_EMULE_MODE is not set; runtime would default to TargetDevice::Silicon";
}

// ============================================================================
// Group 2: SiliconActive -- MetalContext proves silicon path was selected.
// These skip when TT_METAL_EMULE_MODE IS set (emulated runs).
// ============================================================================

TEST(SiliconActive, TargetDeviceIsSilicon) {
    if (std::getenv("TT_METAL_EMULE_MODE")) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE is set -- not in silicon mode";
    }
    auto target = MetalContext::instance().get_cluster().get_target_device_type();
    EXPECT_EQ(target, tt::TargetDevice::Silicon)
        << "Without TT_METAL_EMULE_MODE, target device should be Silicon";
}

TEST(SiliconActive, IsNotMockOrEmulated) {
    if (std::getenv("TT_METAL_EMULE_MODE")) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE is set";
    }
    EXPECT_FALSE(MetalContext::instance().get_cluster().is_mock_or_emulated())
        << "Silicon device should report is_mock_or_emulated() == false";
}

// ============================================================================
// Group 3: EmulationActive -- MetalContext proves emulated path was selected.
// These skip when TT_METAL_EMULE_MODE is not set (silicon runs).
// ============================================================================

TEST(EmulationActive, TargetDeviceIsEmulated) {
    if (!std::getenv("TT_METAL_EMULE_MODE")) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE not set";
    }
    auto target = MetalContext::instance().get_cluster().get_target_device_type();
    EXPECT_EQ(target, tt::TargetDevice::Emule)
        << "With TT_METAL_EMULE_MODE set, target device should be Emule";
}

TEST(EmulationActive, SlowDispatchForced) {
    if (!std::getenv("TT_METAL_EMULE_MODE")) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE not set";
    }
    EXPECT_FALSE(MetalContext::instance().rtoptions().get_fast_dispatch())
        << "Emulated mode must disable fast dispatch (no HWCommandQueue)";
}

TEST(EmulationActive, IsMockOrEmulated) {
    if (!std::getenv("TT_METAL_EMULE_MODE")) {
        GTEST_SKIP() << "TT_METAL_EMULE_MODE not set";
    }
    EXPECT_TRUE(MetalContext::instance().get_cluster().is_mock_or_emulated())
        << "Emulated device should report is_mock_or_emulated() == true";
}

// ============================================================================
// Group 4: MeshDeviceFixture -- kernel execution + emulated runner proof.
// Requires TT_METAL_SLOW_DISPATCH_MODE=1 (fixture skips otherwise).
// ============================================================================

TEST_F(MeshDeviceFixture, EmulatedRunnerInvoked) {
#ifndef TT_METAL_EMULATION
    GTEST_SKIP() << "TT_METAL_EMULATION not compiled in";
#else
    auto target = MetalContext::instance().get_cluster().get_target_device_type();
    if (target != tt::TargetDevice::Emule) {
        GTEST_SKIP() << "Not in emulated mode (target="
                      << static_cast<int>(target) << ")";
    }

    // Run a real program: L1 write -> reader kernel -> CB -> writer kernel -> L1 read
    // Same pattern as SimpleTiledL1WriteCBRead from test_simple_l1_buffer.cpp
    auto mesh_device = this->devices_.at(0);
    CoreCoord core{0, 0};
    size_t base_address = 768 * 1024;
    size_t input_addr = base_address + 8 * 1024;
    size_t output_addr = base_address + 16 * 1024;
    int page_size = 32 * 32 * 2;  // one bfloat16 tile
    size_t byte_size = 2 * 1024;  // one tile
    int num_tiles = byte_size / page_size;

    std::vector<uint32_t> inputs =
        tt::test_utils::generate_uniform_random_vector<uint32_t>(5, 5, byte_size / sizeof(uint32_t));
    std::vector<uint32_t> outputs;

    auto& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& prog = workload.get_programs().at(device_range);

    const uint32_t cb_index = 0;
    CircularBufferConfig cb_config =
        CircularBufferConfig(byte_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, page_size);
    CreateCircularBuffer(prog, core, cb_config);

    std::map<std::string, std::string> defines = {{"INTERFACE_WITH_L1", "1"}};
    uint32_t bank_id =
        mesh_device->allocator()->get_bank_ids_from_logical_core(BufferType::L1, core)[0];

    auto reader_kernel = CreateKernel(
        prog,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::NOC_0,
            .compile_args = {cb_index},
            .defines = defines});

    auto writer_kernel = CreateKernel(
        prog,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/direct_writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::NOC_1,
            .compile_args = {cb_index},
            .defines = defines});

    SetRuntimeArgs(prog, reader_kernel, core,
                   {(uint32_t)input_addr, bank_id, (uint32_t)num_tiles});
    SetRuntimeArgs(prog, writer_kernel, core,
                   {(uint32_t)output_addr, bank_id, (uint32_t)num_tiles});

    tt::test::buffer::detail::writeL1Backdoor(mesh_device, core, input_addr, inputs);
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    tt::test::buffer::detail::readL1Backdoor(mesh_device, core, output_addr, byte_size, outputs);

    ASSERT_EQ(inputs, outputs) << "L1 data mismatch after kernel execution";

    // The critical assertion: verify the emulated program runner was invoked
    const auto& stats = emule::get_last_emulated_run_stats();
    EXPECT_GT(stats.num_cores, 0u)
        << "Emulated runner should have executed on at least one core";
    EXPECT_FALSE(stats.kernel_paths.empty())
        << "Emulated runner should have compiled and executed at least one kernel";
#endif
}
