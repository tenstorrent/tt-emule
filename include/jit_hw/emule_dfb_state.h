#pragma once
// DFB state for JIT-compiled kernels (parallel to emule_cb_state.h).
// Thread-local pointer to per-thread DFB interface array.

#include "tt_emule/dfb_sync_state.hpp"
#include "tt_emule/tile_counter.hpp"

using __emule_dfb_iface = tt_emule::EmuleDFBInterface;

extern thread_local __emule_dfb_iface* __emule_dfbs;
extern thread_local tt_emule::TileCounterArray* __emule_tc_array;
extern thread_local uint8_t __processor_id;
