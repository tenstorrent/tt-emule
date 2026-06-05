#pragma once
// Stub of the silicon ethernet RISC headers. emule has no ethernet
// engine; this file gets pulled in transitively by kernel_profiler.hpp.
// Provide minimal sentinels so the silicon profiler header parses.

#include <cstdint>

inline volatile uint32_t* aerisc_run_flag = nullptr;
