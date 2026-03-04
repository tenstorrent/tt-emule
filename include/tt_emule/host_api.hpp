#pragma once
#include "device.hpp"
#include "buffer.hpp"
#include "program.hpp"
#include <memory>
#include <vector>
#include <cstdint>

namespace tt_emule {

// ---- Device lifecycle ----
Device* CreateDevice(uint32_t id = 0);
bool    CloseDevice(Device* device);

// ---- Program construction ----
Program CreateProgram();

uint32_t CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                      DataMovementConfig config);
uint32_t CreateKernel(Program& program, KernelFn fn, CoreCoord core,
                      ComputeConfig config);

void SetRuntimeArgs(Program& program, uint32_t kernel_id, CoreCoord core,
                    std::vector<uint32_t> args);

CBHandle CreateCircularBuffer(Program& program, CoreCoord core,
                               CircularBufferConfig config);

// ---- Buffer operations ----
std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes,
                                     uint32_t page_size_bytes);

void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src,
                        bool blocking = true);
void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst,
                       bool blocking = true);

// ---- Program execution ----
void EnqueueProgram(Device& device, Program& program, bool blocking = true);
void Finish(Device& device);

} // namespace tt_emule
