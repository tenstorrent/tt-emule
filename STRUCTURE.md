# STRUCTURE.md

Index of source files under `src/` and `include/` and their top-level symbols.
This is a navigation map only — **no behavioral or design docs**. For those, see
`IMPLEMENTATION_REPORT.md`, `docs/`, and the per-file comments.

Each line is `path — symbols` (classes, structs, enums, free functions, key
macros, globals). Grep this file for a symbol name to find its file. Generated
artifacts (`generated/`, `*.log`) and non-source trees are intentionally absent.

---

## src/

- `src/host_api.cpp` — `tt_emule::` free fns: `CreateDevice`, `CloseDevice`, `CreateProgram`, `CreateCommandQueue`, `SetRuntimeArgs`, `CreateBuffer`, `EnqueueWriteBuffer`/`EnqueueReadBuffer`/`EnqueueProgram`/`Finish` (Device + CommandQueue overloads); `detail::` `LaunchProgram`, `ReadFromDeviceL1`/`WriteToDeviceL1`, `ReadFromDeviceDRAMChannel`/`WriteToDeviceDRAMChannel`
- `src/kernel_runner.cpp` — `EnqueueProgram`; `extern "C" __emule_dram_ptr`; thread_locals `__rt_args`, `__core`, `__device`, `__processor_id`, `__emule_tc_array`, `__emule_dfbs`; anon-ns `build_dfb_interfaces`

## include/tt_emule/

- `include/tt_emule/buffer.hpp` — `class Buffer`; fwd `enum class BufferType`
- `include/tt_emule/cb_sync_state.hpp` — `struct CBSyncState`; `cb_sync_reserve`/`cb_sync_push`/`cb_sync_wait`/`cb_sync_pop`, `cb_sync_write_ptr`/`cb_sync_read_ptr`/`cb_sync_write_ptr_at`/`cb_sync_read_ptr_at`
- `include/tt_emule/circular_buffer.hpp` — `class CircularBuffer` (reserve_back/push_back/wait_front/pop_front/get_write_ptr/get_read_ptr)
- `include/tt_emule/dataflow_buffer.hpp` — `class DataflowBuffer` (reserve_back/push_back/wait_front/pop_front/finish/get_write_ptr/get_read_ptr/get_id)
- `include/tt_emule/device.hpp` — `class Device : tt::tt_metal::IDevice`, `class Core`, `class MockAllocator`; `enum class HalMemType`, `CoreRole`, `BufferType`; `CoreCoord` (`struct` or `tt_xy_pair` alias, `#ifdef TT_EMULE_USE_XY_PAIR`)
- `include/tt_emule/dfb_sync_state.hpp` — `struct DFBTCSlot`, `EmuleDFBInterface`, `DFBSyncState`; `MAX_DFBS`, `MAX_TC_SLOTS_PER_DFB`
- `include/tt_emule/dst_register_file.hpp` — `class DstRegisterFile`; `enum class State {IDLE,ACQUIRED,COMMITTED,PACKING}`; acquire/commit/wait/release
- `include/tt_emule/host_api.hpp` — decls: `CloseDevice`, `SetRuntimeArgs`, `EnqueueWriteBuffer`/`EnqueueReadBuffer`/`EnqueueProgram`/`Finish`; `struct CommandQueue`; `using KernelHandle`; `namespace detail` L1/DRAM r/w, `LaunchProgram`, `WriteToBuffer`/`ReadFromBuffer`
- `include/tt_emule/l1_pool.hpp` — `class L1Pool`; `SLOT_SIZE`, `SLOT_MASK`, `to_offset`, `num_slots`
- `include/tt_emule/program.hpp` — `class Program`; `struct KernelDescriptor`, `DataMovementConfig`, `ComputeConfig`, `QuasarDataMovementConfig`, `QuasarComputeConfig`, `CircularBufferConfig`, `DataflowBufferConfig`; `enum class KernelType`, `DataMovementProcessor`, `NOC`, `NOC_MODE`, `MathFidelity`, `UnpackToDestMode`, `AccessPattern`; `using KernelFn`/`CBHandle`/`DFBHandle`
- `include/tt_emule/tile.hpp` — `class Tile`; `ROWS`/`COLS`/`NUM_ELEMENTS`/`SIZE_BYTES`; `operator+`/`+=`/`()`
- `include/tt_emule/tile_counter.hpp` — `struct TileCounter`, `class TileCounterArray`; `TILE_COUNTERS_PER_NEO`; inc_posted/inc_acked/wait_free_space/wait_occupancy/reset_all

## include/hw/

- `include/hw/inc/api/debug/dprint.h` — macros `DPRINT`, `DPRINT_DATA0`, `DPRINT_DATA1`

---

## include/jit_hw/ (root)

- `include/jit_hw/chlkc_list.h` — empty shim (pragma-once only)
- `include/jit_hw/ckernel.h` — `enum class CSR`; `namespace ckernel`; `csr_read<>` specializations
- `include/jit_hw/ckernel_defs.h` — empty shim (pragma-once only)
- `include/jit_hw/ckernel_include.h` — `enum firmware_msg_e`; `namespace ckernel`
- `include/jit_hw/dev_mem_map.h` — macros `MEM_L1_UNCACHED_BASE`; `NUM_TRISC_CORES`/`NUM_DM_CORES`/`MEM_ZEROS_*` constants
- `include/jit_hw/emule_cb_state.h` — `using __emule_cb_state = tt_emule::CBSyncState`; `extern thread_local __emule_cbs`
- `include/jit_hw/emule_dfb_state.h` — `using __emule_dfb_iface = tt_emule::EmuleDFBInterface`; `extern thread_local __emule_dfbs`, `__emule_tc_array`
- `include/jit_hw/emule_wait.h` — `__emule_cv_wait` (cv.wait by default; `cv.wait_for`+`<chrono>` under `EMULE_WAIT_TIMEOUT`)
- `include/jit_hw/jit_kernel_stubs.hpp` — `__EMULE_JIT_MODE`; fwd `tt_emule::Core`/`Device`; `extern "C"` `__emule_dram_ptr`, `__emule_local_l1_to_ptr`; `get_arg_addr`, `get_common_arg_addr`
- `include/jit_hw/llk_defs.h` — aggregator: includes `llk_types.h`, `api/compute/common.h`, `nfaces.h`, `firmware_common.h`, `llk_state.h`
- `include/jit_hw/llk_math_eltwise_binary.h` — empty shim (pragma-once only)
- `include/jit_hw/llk_math_eltwise_unary_datacopy.h` — `enum class DataCopyType`; `copy_tile` datacopy helper
- `include/jit_hw/llk_pack.h` — `__llk_pack_tiled`, `__llk_pack_untilize`
- `include/jit_hw/llk_sync_stubs.h` — `llk_wait_tiles`/`llk_pop_tiles`/`llk_push_tiles`/`llk_wait_for_free_tiles`; `namespace p_stall`, `semaphore`
- `include/jit_hw/llk_types.h` — `enum class DstSync`, `MathFidelity`, `PoolType`, `ReduceDim`, `DataCopyType`; macros `FACE_R_DIM`, `TILE_C_DIM`
- `include/jit_hw/llk_unpack_a.h` — `llk_unpack_tilize_block`; unpack tilize/untilize state setters
- `include/jit_hw/risc_common.h` — includes `internal/firmware_common.h`

## include/jit_hw/api/

- `include/jit_hw/api/bfloat16.h` — `namespace __emule_bf16`: `to_f32`, `from_f32`
- `include/jit_hw/api/cb_api.h` — `cb_wait_front`/`cb_pop_front`/`cb_reserve_back`/`cb_push_back`, `get_write_ptr`/`get_read_ptr`, `get_tile_size`/`get_tile_hw`/`get_tile_num_faces`, `get_dataformat`, `__emule_cb_timeout_sec`; constexpr arrays `unpack_tile_size`/`unpack_tile_r_dim`/`unpack_tile_c_dim`/`unpack_num_faces_r_dim`/`unpack_num_faces_c_dim`/`unpack_src_format`/`pack_dst_format`/`unpack_dst_format`/`pack_src_format`
- `include/jit_hw/api/compile_time_args.h` — `get_compile_time_arg_val`, `get_ct_arg<>`; `KERNEL_COMPILE_TIME_ARGS`
- `include/jit_hw/api/core_local_mem.h` — `noc_traits_t<CoreLocalMem<>>` specialization
- `include/jit_hw/api/dfb_api.h` — `dfb_reserve_back`/`dfb_push_back`/`dfb_wait_front`/`dfb_pop_front`/`dfb_finish`, `dfb_get_write_ptr`/`dfb_get_read_ptr`/`dfb_get_entry_size`, `__emule_dfb_check_id`/`__emule_dfb_timeout_sec`
- `include/jit_hw/api/kernel_thread_globals.h` — `get_num_threads`, `get_my_thread_id`

## include/jit_hw/api/compute/

- `include/jit_hw/api/compute/add_int_sfpu.h` — `ckernel::` `add_int_tile`, `add_int_tile_init`
- `include/jit_hw/api/compute/atan2.h` — `ckernel::` `atan2_binary_tile`, `atan2_binary_tile_init`
- `include/jit_hw/api/compute/bcast.h` — `namespace __emule_bcast` `enum class Dim` (`apply`/`src_idx`); `any_tiles_bcast`, `add/sub/mul_tiles_bcast`, `add/sub/mul_tiles_bcast_rows`/`_cols`/`_scalar`, `unary_bcast`/`unary_bcast_init`, `init_bcast`/`add_bcast_rows_init_short`/`add_bcast_cols_init_short`/`add_bcast_scalar_init_short`
- `include/jit_hw/api/compute/bfp8.h` — `namespace __emule_bfp8`: `face_row`, `col_in_row` (BFP8 codec helpers)
- `include/jit_hw/api/compute/binary_bitwise_sfpu.h` — `ckernel::` `bitwise_and/or/xor_binary_tile` (+`_init`), `binary_bitwise_tile_init`
- `include/jit_hw/api/compute/binary_comp.h` — `ckernel::` `lt/le/gt/ge_int32_tile` (+`_init`); `lt/le/gt/ge_int_tile<DataFormat>` (+`_init`, Int32/UInt32/UInt16)
- `include/jit_hw/api/compute/binary_fmod.h` — `ckernel::` `fmod_binary_tile`, `fmod_int32_tile` (+`_init`)
- `include/jit_hw/api/compute/binary_max_min.h` — `ckernel::` `binary_max/min_tile`, `_int32_`, `_uint32_` variants (+`_init`)
- `include/jit_hw/api/compute/binary_remainder.h` — `ckernel::` `remainder_binary_tile`, `remainder_int32_tile` (+`_init`)
- `include/jit_hw/api/compute/binary_shift.h` — `ckernel::` `binary_left_shift_tile`, `binary_right_shift_tile`, `binary_logical_right_shift_tile`, `binary_shift_tile_init`
- `include/jit_hw/api/compute/cb_api.h` — includes `compute/common.h` + `api/cb_api.h` (shim)
- `include/jit_hw/api/compute/common.h` — macros `PACK`/`MATH`/`UNPACK`/`ALWI`, `DST_ACCUM_MODE`; `namespace __emule_compute`, `ckernel`; `enum class EltwiseBinaryType`/`BroadcastType`/`ReluType`/`EltwiseBinaryReuseDestType`/`p_dim_stride_target`/`MathFidelity`; `TILE_WIDTH`/`TILE_HEIGHT`/`TILE_HW`/`FACE_WIDTH`/`FACE_HEIGHT`/`FACE_HW`/`TILE_C_DIM`; `pack_tile`/`pack_tile_block`, `copy_tile`/`copy_tile_init`, `add_tiles`/`sub_tiles`/`mul_tiles`, `binary_tiles_init`, `binary_dest_reuse_tiles`, `__emule_dst_active_tiles`/`__emule_dst_check`/`__emule_dst_load_i32`/`__emule_dst_store_i32`; `namespace __emule_compute` `cb_page_size`/`cb_tile_elems`/`cb_is_32bit_format`/`cb_is_bfp8_b_format`/`cb_data_format`/`cb_is_uint16_format`/`pack_dst_to_buf`/`__emule_unpack_cb_tile_to`; no-op stubs `reconfig_data_format_srca`/`reconfig_data_format_srcb`/`pack_reconfig_data_format`/`llk_pack_relu_config`/`pack_set_relu_threshold`/`llk_pack_hw_configure`/`pack_init`/`pack_dest_init`/`pack_reconfig_l1_acc`/`pack_relu_config`
- `include/jit_hw/api/compute/common_globals.h` — `enum class DataFormat`; DST dirty/fresh tracking globals
- `include/jit_hw/api/compute/compute_kernel_api.h` — `ckernel::` `square_tile`/`sigmoid_tile`/`sign_tile`/`signbit_tile`/`silu_tile`/`log_tile`/`exp2_tile`/`expm1_tile`/`power_tile` (+`_init`), `sfpu_reduce_init`, `topk_tile_init`; `namespace __emule_topk`; `struct RowView`
- `include/jit_hw/api/compute/compute_kernel_hw_startup.h` — `compute_kernel_hw_startup` (2-arg + 3-arg)
- `include/jit_hw/api/compute/copy_dest_values.h` — `copy_block` (global scope); `ckernel::` `copy_dest_values`, `copy_dest_values_init`
- `include/jit_hw/api/compute/cumprod.h` — `ckernel::` `cumprod_tile`, `cumprod_tile_init`; TLS `__emule_cumprod_acc`/`__emule_cumprod_acc_initialized`, `__emule_cumprod_reset_acc`
- `include/jit_hw/api/compute/cumsum.h` — `ckernel::` `cumsum_tile`, `cumsum_tile_init`; TLS `__emule_cumsum_acc`
- `include/jit_hw/api/compute/div_int32_floor.h` — `ckernel::` `div_int32_floor_tile`, `div_int32_trunc_tile` (+`_init`)
- `include/jit_hw/api/compute/div_int32_sfpu.h` — `ckernel::` `div_int32_tile`, `div_int32_tile_init`
- `include/jit_hw/api/compute/eltwise_binary.h` — `ckernel::` `add_tiles_init`, `sub_tiles_init`, `mul_tiles_init`
- `include/jit_hw/api/compute/eltwise_binary_sfpu.h` — `ckernel::` `add/sub/mul/div/rsub/power_binary_tile` (+`_init`); `eq/ne/lt/gt/le/ge_binary_tile` (+`_init`, float compares → 1.0/0.0)
- `include/jit_hw/api/compute/gcd.h` — `ckernel::` `gcd_tile`, `gcd_tile_init`
- `include/jit_hw/api/compute/isclose.h` — `ckernel::` `isclose_binary_tile`, `isclose_binary_tile_init`
- `include/jit_hw/api/compute/lcm.h` — `ckernel::` `lcm_tile`, `lcm_tile_init`
- `include/jit_hw/api/compute/logsigmoid.h` — `ckernel::` `logsigmoid_tile` (3-arg in0/in1/out), `logsigmoid_tile_init`
- `include/jit_hw/api/compute/mask.h` — `ckernel::` `mask_tile`, `mask_posinf_tile` (+`_init`)
- `include/jit_hw/api/compute/matmul.h` — `ckernel::` `matmul_tiles`, `matmul_block`, `mm_init`, `mm_block_init`; `EMULE_MATMUL_USE_AVX2`
- `include/jit_hw/api/compute/mul_int_sfpu.h` — `ckernel::` `mul_int_tile`, `mul_int_tile_init`
- `include/jit_hw/api/compute/nfaces.h` — `namespace __emule_nfaces`: `rowmajor_to_nfaces[]`, `nfaces_to_rowmajor[]` LUTs
- `include/jit_hw/api/compute/pack.h` — empty shim (pragma-once only)
- `include/jit_hw/api/compute/pack_untilize.h` — `ckernel::`/`experimental::` `pack_untilize_block`, `pack_untilize_init`, `pack_untilize_dest_init`
- `include/jit_hw/api/compute/quantization.h` — `ckernel::` `quant_tile`, `dequant_tile`, `requant_tile` (+`_init`)
- `include/jit_hw/api/compute/reconfig_data_format.h` — forwarding shim → `api/compute/common.h`
- `include/jit_hw/api/compute/reduce.h` — `ckernel::` `reduce_tile`, `reduce_init`; `REDUCE_OP`/`REDUCE_DIM`
- `include/jit_hw/api/compute/reg_api.h` — `tile_regs_acquire`/`tile_regs_commit`/`tile_regs_wait`/`tile_regs_release`; `ALWI` macro; `__emule_dst_active_tiles`
- `include/jit_hw/api/compute/remainder_int32.h` — `ckernel::` `remainder_int32_tile`, `remainder_int32_tile_init`
- `include/jit_hw/api/compute/reshuffle.h` — `ckernel::` `reshuffle_rows_tile`, `reshuffle_rows_tile_init`
- `include/jit_hw/api/compute/softmax.h` — re-export of `eltwise_unary/exp.h` + `eltwise_unary/recip.h` (no own symbols)
- `include/jit_hw/api/compute/sub_int_sfpu.h` — `ckernel::` `sub_int_tile`, `rsub_int_tile` (+`_init`)
- `include/jit_hw/api/compute/tile_move_copy.h` — forwarding shim → `api/compute/common.h`
- `include/jit_hw/api/compute/tilize.h` — `tilize_block`, `tilize_init`, `tilize_init_short`, `tilize_uninit`, `tilize_init_short_with_dt`, `tilize_uninit_with_dt`, `fast_tilize_block`, `fast_tilize_init`, `fast_tilize_init_skip_remap`, `fast_tilize_init_with_dt`, `fast_tilize_init_with_dt_skip_remap`, `fast_tilize_uninit`
- `include/jit_hw/api/compute/transpose_wh.h` — `ckernel::` `transpose_wh_tile`, `transpose_wh_init`, `copy_tile`
- `include/jit_hw/api/compute/untilize.h` — `untilize_block`, `untilize_init`, `untilize_init_short`, `untilize_uninit`, `copy_tile`
- `include/jit_hw/api/compute/vector_mode.h` — `ckernel::` `enum class VectorMode`
- `include/jit_hw/api/compute/welford.h` — `ckernel::` `welford_init`, `welford_reinit`, `welford_clear`; TLS `__emule_welford_mean`/`__emule_welford_m2`/`__emule_welford_count`, `__emule_welford_clear`
- `include/jit_hw/api/compute/xlogy.h` — `ckernel::` `xlogy_binary_tile`, `xlogy_binary_tile_init`

## include/jit_hw/api/compute/eltwise_unary/

- `include/jit_hw/api/compute/eltwise_unary/activations.h` — `ckernel::` `abs_tile`, `abs_tile_int32`, `hardsigmoid_tile`, `softsign_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/binop_with_scalar.h` — `ckernel::` `add/sub/mul/div/rsub_unary_tile`, `add_unary_tile_int32`, `sub_unary_tile_int32`, `binop_with_scalar_tile_init`
- `include/jit_hw/api/compute/eltwise_unary/bitwise_and.h` — `ckernel::` `bitwise_and_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/bitwise_not.h` — `ckernel::` `bitwise_not_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/bitwise_or.h` — `ckernel::` `bitwise_or_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/bitwise_xor.h` — `ckernel::` `bitwise_xor_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/cbrt.h` — `ckernel::` `cbrt_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/clamp.h` — `ckernel::` `clamp_tile`, `clamp_tile_int32` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/comp.h` — `ckernel::` `eqz/nez/ltz/lez/gtz/gez_tile`, `unary_eq/ne/lt/le/gt/ge_tile` (+`_init`); `_int32` variants of all, plus `eqz/nez_tile_uint16`/`_uint32`
- `include/jit_hw/api/compute/eltwise_unary/digamma.h` — `ckernel::` `digamma_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/dropout.h` — `ckernel::` `dropout_tile`, `dropout_kernel_init`; TLS `__emule_dropout_rng_state`, `__emule_dropout_next` (xorshift32)
- `include/jit_hw/api/compute/eltwise_unary/eltwise_unary.h` — `ckernel::` `init_sfpu`; aggregates eltwise_unary headers
- `include/jit_hw/api/compute/eltwise_unary/elu.h` — `ckernel::` `elu_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/erf_erfc.h` — `ckernel::` `erf_tile`, `erfc_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/erfinv.h` — `ckernel::` `erfinv_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/exp.h` — `ckernel::` `exp_tile` (+`_init`); `enum class InputClamping`; `namespace p_sfpu`
- `include/jit_hw/api/compute/eltwise_unary/fill.h` — `ckernel::` `fill_tile`, `fill_tile_int`, `fill_tile_bitcast` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/fmod.h` — `ckernel::` `fmod_tile`, `fmod_tile_init`
- `include/jit_hw/api/compute/eltwise_unary/gelu.h` — `ckernel::` `gelu_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/hardmish.h` — `ckernel::` `hardmish_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/hardtanh.h` — `ckernel::` `hardtanh_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/i0.h` — `ckernel::` `i0_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/i1.h` — `ckernel::` `i1_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/identity.h` — `ckernel::` `identity_tile`, `identity_tile_uint32` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/isinf_isnan.h` — `ckernel::` `isinf_tile`, `isnan_tile`, `isfinite_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/left_shift.h` — `ckernel::` `left_shift_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/lgamma.h` — `ckernel::` `lgamma_stirling_tile`, `lgamma_stirling_float_tile` (+`_init`); anon-ns `lgamma_eval_z`
- `include/jit_hw/api/compute/eltwise_unary/log1p.h` — `ckernel::` `log1p_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/logical_not.h` — `ckernel::` `logical_not_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/mish.h` — `ckernel::` `mish_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/negative.h` — `ckernel::` `negative_tile`, `negative_tile_int32` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/polygamma.h` — `ckernel::` `polygamma_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/prelu.h` — `ckernel::` `prelu_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/rand.h` — `ckernel::` `rand_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/rdiv.h` — empty stub
- `include/jit_hw/api/compute/eltwise_unary/recip.h` — `ckernel::` `recip_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/relu.h` — `ckernel::` `relu_tile`, `relu_tile_int32` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/remainder.h` — `ckernel::` `remainder_tile`, `remainder_tile_init`
- `include/jit_hw/api/compute/eltwise_unary/right_shift.h` — `ckernel::` `right_shift_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/rounding.h` — `ckernel::` `floor_tile`, `ceil_tile`, `trunc_tile`, `frac_tile`, `rounding_op_tile_init`
- `include/jit_hw/api/compute/eltwise_unary/rpow.h` — empty stub
- `include/jit_hw/api/compute/eltwise_unary/rsqrt.h` — `ckernel::` `rsqrt_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/rsub.h` — `ckernel::` `rsub_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/selu.h` — `ckernel::` `selu_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/sfpu_split_includes.h` — conditional includes under `SFPU_OP_*_INCLUDE` guards (no own symbols)
- `include/jit_hw/api/compute/eltwise_unary/softplus.h` — `ckernel::` `softplus_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/sqrt.h` — `ckernel::` `sqrt_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/threshold.h` — `ckernel::` `threshold_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/trigonometry.h` — `ckernel::` `sin/cos/tan/tanh/asin/acos/atan_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/typecast.h` — `ckernel::` `typecast_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/where.h` — `ckernel::` `where_tile` (+`_init`)
- `include/jit_hw/api/compute/eltwise_unary/xielu.h` — `ckernel::` `xielu_tile` (+`_init`)

## include/jit_hw/api/compute/experimental/

- `include/jit_hw/api/compute/experimental/fast_untilize.h` — `ckernel::` `fast_untilize_block`, `fast_untilize_init`, `fast_untilize_init_skip_remap`, `fast_untilize_uninit`
- `include/jit_hw/api/compute/experimental/fill_arange.h` — `experimental::` `fill_arange_tile`, `write_row_mask_tile`, `write_col_mask_tile`
- `include/jit_hw/api/compute/experimental/semaphore.h` — `ckernel::Semaphore`

## include/jit_hw/api/dataflow/

- `include/jit_hw/api/dataflow/circular_buffer.h` — `class CircularBuffer`, `struct CircularBufferView`; `enum class AddrSelector`; `noc_traits_t<CircularBuffer>` + `noc_traits_t<CircularBufferView>` specializations
- `include/jit_hw/api/dataflow/dataflow_api.h` — `noc_async_read`/`noc_async_write` (+ `_one_packet`, `_tile`, `_page`), `noc_async_{read,write}_one_packet_{set_state,with_state}`, `noc_async_write_multicast`/`_multicast_loopback_src`, `noc_async_read_barrier`/`noc_async_write_barrier`/`noc_async_writes_flushed`/`noc_async_atomic_barrier`/`noc_async_full_barrier`, `noc_async_{read,write}_barrier_with_trid`/`noc_async_write_flushed_with_trid`/`noc_async_{read,write}_set_trid`, `ncrisc_noc_read_with_transaction_id_flushed`/`ncrisc_noc_nonposted_write_with_transaction_id_{flushed,sent}`, `noc_semaphore_set`/`_wait`/`_wait_min`/`_inc`/`_set_multicast`, `noc_inline_dw_write`, `dram_barrier`; `get_noc_addr`/`get_dram_noc_addr`/`get_l1_noc_addr`/`get_noc_multicast_addr`, `get_semaphore`; SFINAE traits `has_get_noc_addr_v`/`has_page_size_v`/`has_log_base_2_of_page_size_v`/`has_get_aligned_page_size_v`; thread-local state `__emule_one_packet_state_size`/`__emule_write_one_packet_state_{dst,size}`; `__emule_addr_to_offset`/`__emule_fixup_noc_addr`/`__emule_resolve_noc_addr`/`__emule_local_l1_to_ptr`; `struct CBInterface`, `__emule_dw_state`; NOC VC/burst macros, `EMULE_SEM_BASE`
- `include/jit_hw/tensix_types.h` — (empty include-shim, no symbols)
- `include/jit_hw/api/dataflow/dataflow_buffer.h` — `class DataflowBuffer` (get_write_ptr/get_read_ptr methods), `struct DFBAccessor`; `noc_traits_t<DataflowBuffer>` specialization
- `include/jit_hw/api/dataflow/endpoints.h` — `struct UnicastEndpoint`, `MulticastEndpoint`, `AllocatorBank`; `enum class AllocatorBankType`; `noc_traits_t` specializations for each (`get_noc_unicast_addr`/`get_noc_multicast_addr` are endpoint methods)
- `include/jit_hw/api/dataflow/noc.h` — `class Noc` (async_read/async_write/barriers/inline_dw_write methods); `enum class NocOptions`/`Noc::AddressType`; `struct NocOptVals`/`DataflowBufferArgs`; `operator|`/`has_flag` (NocOptions); primary `noc_traits_t` template
- `include/jit_hw/api/dataflow/noc_semaphore.h` — `class Semaphore`; `enum class ProgrammableCoreType`

## include/jit_hw/api/debug/

- `include/jit_hw/api/debug/assert.h` — macros `ASSERT`, `ASSERT_ENABLED`
- `include/jit_hw/api/debug/device_print.h` — macros `DEVICE_PRINT`, `DEVICE_PRINT_UNPACK`/`_MATH`/`_PACK`/`_DATA0`/`_DATA1`
- `include/jit_hw/api/debug/dprint.h` — macros `DPRINT`, `DPRINT_UNPACK`/`_MATH`/`_PACK`/`_DATA0`/`_DATA1`, `ENDL`

## include/jit_hw/api/tensor/

- `include/jit_hw/api/tensor/noc_traits.h` — `noc_traits_t<TensorAccessor>` specialization (nested `src_args_type`/`dst_args_type`); `NOC_ADDR_LOCAL_BITS` macro

## include/jit_hw/cpp/ttnn/operations/data_movement/common/kernels/

- `include/jit_hw/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` — forwarding shim → `ttnn/operations/data_movement/common/kernels/common.hpp`

## include/jit_hw/experimental/

- `include/jit_hw/experimental/circular_buffer.h` — `experimental::CircularBuffer`, `struct CircularBufferView`; `enum class AddrSelector`; `noc_traits_t`
- `include/jit_hw/experimental/core_local_mem.h` — `experimental::CoreLocalMem<>`; `get_unsafe_ptr`; `noc_traits_t`
- `include/jit_hw/experimental/dataflow_buffer.h` — `experimental::DataflowBuffer`, `struct DFBAccessor`; `dfb_get_write_ptr`/`dfb_get_read_ptr`
- `include/jit_hw/experimental/endpoints.h` — `experimental::` `struct UnicastEndpoint`/`MulticastEndpoint`/`AllocatorBank`/`ReadSpec`/`WriteSpec`; `enum AllocatorBankType`
- `include/jit_hw/experimental/kernel_args.h` — `experimental::` `struct RtaArg`/`CrtaArg`/`CtaVal`; `get_arg_addr`, `get_common_arg_addr`
- `include/jit_hw/experimental/lock.h` — `experimental::Lock`
- `include/jit_hw/experimental/noc.h` — `experimental::Noc`, `class DataflowBuffer`; `enum class AddressType`/`TxnIdMode`/`VcSelection`; `struct DataflowBufferArgs`; `NOC_MAX_BURST_SIZE`
- `include/jit_hw/experimental/noc_semaphore.h` — `experimental::Semaphore`
- `include/jit_hw/experimental/tensor.h` — `experimental::noc_traits_t<TensorAccessor>` specialization

## include/jit_hw/hostdevcommon/

- `include/jit_hw/hostdevcommon/common_values.hpp` — `constexpr INVALID`, `VALID`

## include/jit_hw/internal/

- `include/jit_hw/internal/circular_buffer_interface.h` — forwarding shim → `jit_hw/internal/llk_state.h`
- `include/jit_hw/internal/firmware_common.h` — `WAYPOINT` macro; cache-op stubs
- `include/jit_hw/internal/llk_state.h` — `struct CbInterface`; `get_local_cb_interface`, `get_operand_narrow_tile`, `get_output_narrow_tile`
- `include/jit_hw/internal/mod_div_lib.h` — `mulsi3`; `fast_udiv_<N>` for N = 7, 12, 20, 48, 56, 63, 70, 72, 80, 94, 108, 110, 117, 120, 124, 126, 130, 140
- `include/jit_hw/internal/risc_attribs.h` — macros `tt_l1_ptr`, `tt_reg_ptr`, `FORCE_INLINE`; `enum class InlineWriteDst`; `RISCV_DEBUG_REG_WALL_CLOCK_*`

## include/jit_hw/internal/dataflow/

- `include/jit_hw/internal/dataflow/dataflow_api_addrgen.h` — `struct InterleavedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen`; `namespace interleaved_addr_gen`; `NUM_DRAM_BANKS`/`NUM_L1_BANKS`/`NOC_XY_ADDR`/`DRAM_ALIGNMENT`/`L1_ALIGNMENT` macros

## include/jit_hw/internal/tt-2xx/quasar/overlay/

- `include/jit_hw/internal/tt-2xx/quasar/overlay/overlay_addresses.h` — `L2_FLUSH_ADDR` macro; `__emule_l2_flush_sink`

## include/jit_hw/llk/

- `include/jit_hw/llk/llk_reduce_primitives.h` — `struct __emule_matmul_bridge`; `enum class PoolType`/`ReduceDim`/`MathFidelity`/`p_dim_stride_target`; `llk_math_matmul_init`, `llk_math_reduce_init`, `matmul_tiles`; `MATH_FIDELITY`

## include/jit_hw/noc/

- `include/jit_hw/noc/noc_parameters.h` — re-exports `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` (`NOC_X_SIZE`/`NOC_Y_SIZE`/`NOC_CMD_*`/`NOC_ADDR_*` macros)

## include/jit_hw/tools/profiler/

- `include/jit_hw/tools/profiler/kernel_profiler.hpp` — `namespace kernel_profiler`; macros `DeviceZoneScopedN`/`DeviceZoneScoped`/`DeviceZoneScopedMainN`/`DeviceZoneScopedMainChildN`/`DeviceTimestampedData`
- `include/jit_hw/tools/profiler/noc_debugging_metadata.hpp` — empty stub
- `include/jit_hw/tools/profiler/noc_debugging_profiler.hpp` — `RECORD_SCOPED_LOCK_EVENT` macro

## include/jit_hw/tt-metalium/

- `include/jit_hw/tt-metalium/buffer_types.hpp` — `tt::` `enum class TensorMemoryLayout`, `ShardOrientation`, `ShardDistributionStrategy`, `BufferType`
- `include/jit_hw/tt-metalium/circular_buffer_constants.h` — `NUM_CIRCULAR_BUFFERS`, `UINT32_WORDS_PER_*_BUFFER_CONFIG` constants
- `include/jit_hw/tt-metalium/constants.hpp` — `tt::constants::` `TILE_HEIGHT`/`TILE_WIDTH`/`TILE_HW`/`FACE_*`
- `include/jit_hw/tt-metalium/tt_backend_api_types.hpp` — `tt::` `enum class DataFormat`
