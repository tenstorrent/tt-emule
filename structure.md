# Structure

## Update Rules

- Regenerate or update this file whenever source files are added, removed, renamed, or public/internal signatures change.
- Keep entries sorted by path.
- Preserve declaration order inside each file.
- Do not add prose summaries.
- Do not include generated, vendored, build, cache, log, binary, or large data files unless they are source-controlled source inputs.
- Each file entry must follow the exact format below.

## Format

```md
### `path/to/file.ext`

Signatures:
- `signature one`
- `signature two`
```

If no signatures are present:

```md
### `path/to/file.ext`

Signatures: none
```

## Files

### `.github/scripts/ci-build.sh`

Signatures: none

### `.github/scripts/ci-regression.sh`

Signatures: none

### `.github/scripts/ci-setup.sh`

Signatures: none

### `.github/scripts/classify-results.py`

Signatures:
- `def parse_xml(xml_dir: Path) -> tuple[set[str], dict[str, str]]`
- `def read_allowlist(path: Path) -> list[str]`
- `def main() -> int`

### `.github/workflows/nightly-upstream.yml`

Signatures:
- `name: Nightly Upstream Regression`
- `trigger schedule`
- `trigger workflow_dispatch: {}`
- `job build`
- `job test`

### `.github/workflows/pr-regression.yml`

Signatures:
- `name: PR Regression`
- `trigger pull_request`
- `trigger push`
- `job build`
- `job test`

### `CMakeLists.txt`

Signatures:
- `project(tt_emule ...)`
- `set(PROJECT_IS_TOP_LEVEL)`
- `set(PROJECT_IS_TOP_LEVEL)`
- `set(CMAKE_CXX_STANDARD)`
- `set(CMAKE_CXX_STANDARD_REQUIRED)`
- `find_package(Threads)`
- `add_library(tt_emule_lib ...)`
- `target_include_directories(tt_emule_lib)`
- `target_compile_definitions(tt_emule_lib)`
- `target_link_libraries(tt_emule_lib)`
- `add_library(tt_emule_headers ...)`
- `add_library(tt-emule ...)`
- `target_include_directories(tt_emule_headers)`
- `include(GNUInstallDirs)`
- `install(TARGETS ...)`
- `install(DIRECTORY ...)`
- `install(EXPORT ...)`
- `enable_testing(...)`
- `add_subdirectory(tests)`

### `build_and_test.sh`

Signatures: none

### `include/hw/inc/api/debug/dprint.h`

Signatures:
- `#define DPRINT`
- `#define DPRINT_DATA0(...)`
- `#define DPRINT_DATA1(...)`

### `include/jit_hw/api/bfloat16.h`

Signatures:
- `namespace __emule_bf16`
- `inline float to_f32(uint16_t bf16)`
- `inline uint16_t from_f32(float val)`

### `include/jit_hw/api/cb_api.h`

Signatures:
- `extern thread_local uint8_t my_x[2];`
- `extern thread_local uint8_t my_y[2];`
- `extern thread_local uint32_t __emule_logical_x;`
- `extern thread_local uint32_t __emule_logical_y;`
- `inline int __emule_cb_timeout_sec()`
- `inline void cb_reserve_back(uint32_t cb_id, uint32_t n)`
- `inline void cb_push_back(uint32_t cb_id, uint32_t n)`
- `inline void cb_wait_front(uint32_t cb_id, uint32_t n)`
- `inline void cb_pop_front(uint32_t cb_id, uint32_t n)`
- `inline void cb_reserve_back(uint32_t cb_id, int32_t n)`
- `inline void cb_push_back(uint32_t cb_id, int32_t n)`
- `inline void cb_wait_front(uint32_t cb_id, int32_t n)`
- `inline void cb_pop_front(uint32_t cb_id, int32_t n)`
- `inline uint32_t get_write_ptr(uint32_t cb_id)`
- `inline uint32_t get_read_ptr(uint32_t cb_id)`
- `constexpr inline uint32_t get_tile_size(uint32_t cb_id)`
- `constexpr inline uint32_t get_tile_hw(uint32_t cb_id)`
- `constexpr inline uint32_t get_tile_num_faces(uint32_t cb_id)`
- `constexpr inline DataFormat get_dataformat(uint32_t cb_id)`

### `include/jit_hw/api/compile_time_args.h`

Signatures:
- `#define KERNEL_COMPILE_TIME_ARGS`
- `namespace (anonymous)`
- `template<int N> constexpr uint32_t get_ct_arg()`
- `#define get_compile_time_arg_val(N)`
- `namespace (anonymous)`
- `constexpr uint32_t get_named_ct_arg(std::string_view name)`
- `constexpr uint32_t get_named_compile_time_arg_val(std::string_view name)`

### `include/jit_hw/api/compute/add_int_sfpu.h`

Signatures:
- `namespace ckernel`
- `ALWI void add_int_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void add_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/bcast.h`

Signatures:
- `namespace ckernel`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void init_bcast(uint32_t icb0 = 0, uint32_t icb1 = 1)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void add_bcast_rows_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void add_bcast_cols_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void add_bcast_scalar_init_short(uint32_t icb0 = 0, uint32_t icb1 = 1)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void any_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void add_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void sub_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `template <EltwiseBinaryType op_type, BroadcastType bcast_type> inline void mul_tiles_bcast(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`

### `include/jit_hw/api/compute/binary_bitwise_sfpu.h`

Signatures:
- `namespace ckernel`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_and_binary_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_or_binary_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_xor_binary_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_and_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_or_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void bitwise_xor_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/binary_comp.h`

Signatures:
- `namespace ckernel`
- `ALWI void lt_int32_tile_init()`
- `ALWI void gt_int32_tile_init()`
- `ALWI void ge_int32_tile_init()`
- `ALWI void le_int32_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void lt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void gt_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void ge_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void le_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/binary_fmod.h`

Signatures:
- `namespace ckernel`
- `ALWI void fmod_binary_tile_init()`
- `ALWI void fmod_int32_tile_init()`
- `ALWI void fmod_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void fmod_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/binary_max_min.h`

Signatures:
- `namespace ckernel`
- `ALWI void binary_max_tile_init()`
- `ALWI void binary_min_tile_init()`
- `ALWI void binary_max_int32_tile_init()`
- `ALWI void binary_min_int32_tile_init()`
- `ALWI void binary_max_uint32_tile_init()`
- `ALWI void binary_min_uint32_tile_init()`
- `ALWI void binary_max_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `ALWI void binary_min_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void binary_max_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void binary_min_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::UInt32> ALWI void binary_max_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::UInt32> ALWI void binary_min_uint32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/binary_shift.h`

Signatures:
- `namespace ckernel`
- `ALWI void binary_shift_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void binary_left_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void binary_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void binary_logical_right_shift_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/common.h`

Signatures:
- `#define PACK(x)`
- `#define MATH(x)`
- `#define UNPACK(x)`
- `#define ALWI`
- `namespace ckernel`
- `enum class EltwiseBinaryType`
- `enum class BroadcastType`
- `enum class EltwiseBinaryReuseDestType`
- `enum class MathFidelity`
- `enum class ReluType`
- `#define DST_ACCUM_MODE`
- `static constexpr uint32_t __EMULE_DST_TILES = 16;`
- `static constexpr uint32_t __EMULE_DST_TILES_FP32 = 8;`
- `static constexpr uint32_t __EMULE_TILE_ELEMS = 1024;`
- `static constexpr uint32_t __EMULE_DST_BYTES = __EMULE_TILE_ELEMS * sizeof(float);`
- `static thread_local float __emule_dst[__EMULE_DST_TILES][__EMULE_TILE_ELEMS];`
- `static thread_local bool __emule_l1_acc_enabled = false;`
- `inline constexpr uint32_t __emule_dst_active_tiles()`
- `inline void __emule_dst_check(uint32_t slot, const char* caller)`
- `inline int32_t __emule_dst_load_i32(uint32_t slot, uint32_t idx)`
- `inline void __emule_dst_store_i32(uint32_t slot, uint32_t idx, int32_t v)`
- `inline uint32_t get_absolute_logical_x()`
- `inline uint32_t get_absolute_logical_y()`
- `namespace __emule_compute`
- `inline uint8_t* cb_read_ptr_at(uint32_t cb_id, uint32_t tile_offset)`
- `inline uint8_t* cb_write_ptr(uint32_t cb_id)`
- `inline uint8_t* cb_write_ptr_at(uint32_t cb_id, uint32_t tile_offset)`
- `inline uint32_t cb_page_size(uint32_t cb_id)`
- `inline uint32_t cb_tile_elems(uint32_t cb_id)`
- `inline bool cb_is_32bit_format(uint32_t cb_id)`
- `inline void pack_dst_to_buf(uint8_t* buf, uint32_t dst_slot, uint32_t ocb)`
- `namespace ckernel`
- `ALWI void binary_op_init_common(uint32_t, uint32_t, uint32_t)`
- `ALWI void binary_op_init_common(uint32_t, uint32_t, uint32_t, uint32_t)`
- `template<bool FullInit = true, EltwiseBinaryType BinaryType = EltwiseBinaryType::ELWADD> ALWI void binary_tiles_init(uint32_t, uint32_t, bool = false)`
- `ALWI void add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `ALWI void sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `ALWI void mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `ALWI void pack_tile(uint32_t idst, uint32_t ocb)`
- `template <bool UseOutputOffset> ALWI void pack_tile(uint32_t idst, uint32_t ocb, uint32_t output_offset = 0)`
- `ALWI void pack_tile_block(uint32_t ifrom_dst, uint32_t ocb, uint32_t ntiles)`
- `ALWI void copy_tile(uint32_t icb, uint32_t itile, uint32_t idst)`
- `ALWI void copy_block_matmul_partials( uint32_t in_cb_id, uint32_t start_in_tile_index, uint32_t start_dst_tile_index, uint32_t ntiles)`
- `ALWI void copy_tile_to_dst_init_short(uint32_t)`
- `ALWI void copy_tile_to_dst_init_short(uint32_t, uint32_t)`
- `ALWI void copy_tile_init(uint32_t = 0)`
- `ALWI void copy_tile_to_dst_init_short_with_dt(uint32_t, uint32_t, uint32_t = 0)`
- `ALWI void reconfig_data_format(uint32_t)`
- `ALWI void reconfig_data_format(uint32_t, uint32_t)`
- `template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false> ALWI void reconfig_data_format_srca(uint32_t)`
- `template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false> ALWI void reconfig_data_format_srca(uint32_t, uint32_t)`
- `template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false> ALWI void reconfig_data_format_srcb(uint32_t)`
- `template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false> ALWI void reconfig_data_format_srcb(uint32_t, uint32_t)`
- `ALWI void pack_reconfig_data_format(uint32_t)`
- `ALWI void pack_reconfig_data_format(uint32_t, uint32_t)`
- `ALWI void llk_pack_relu_config(ReluType)`
- `ALWI void pack_set_relu_threshold(float)`
- `template<EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::NONE> ALWI void binary_dest_reuse_tiles_init(uint32_t = 0, uint32_t = 0, bool = false)`
- `template<EltwiseBinaryReuseDestType ReuseType, EltwiseBinaryType BinaryType> ALWI void binary_dest_reuse_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)`
- `ALWI void state_configure(uint32_t = 0)`
- `ALWI void llk_pack_reconfig_l1_acc(uint32_t enable)`
- `ALWI void acquire_dst()`
- `ALWI void release_dst()`
- `using namespace ckernel;`

### `include/jit_hw/api/compute/common_globals.h`

Signatures:
- `enum class DataFormat`
- `inline void __emule_dst_mark_dirty(uint32_t slot)`
- `inline bool __emule_dst_take_fresh(uint32_t slot)`

### `include/jit_hw/api/compute/compute_kernel_api.h`

Signatures: none

### `include/jit_hw/api/compute/compute_kernel_hw_startup.h`

Signatures:
- `inline void compute_kernel_hw_startup(uint32_t, uint32_t)`
- `inline void compute_kernel_hw_startup(uint32_t a, uint32_t b, uint32_t)`

### `include/jit_hw/api/compute/copy_dest_values.h`

Signatures:
- `inline void copy_block(uint32_t icb, uint32_t start_tile, uint32_t ntiles, uint32_t start_dst)`

### `include/jit_hw/api/compute/div_int32_floor.h`

Signatures:
- `namespace ckernel`
- `ALWI void div_int32_floor_tile_init()`
- `ALWI void div_int32_trunc_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void div_int32_floor_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void div_int32_trunc_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/div_int32_sfpu.h`

Signatures:
- `namespace ckernel`
- `ALWI void div_int32_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void div_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/eltwise_binary.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_binary_sfpu.h`

Signatures:
- `namespace ckernel`
- `ALWI void add_binary_tile_init()`
- `ALWI void sub_binary_tile_init()`
- `ALWI void mul_binary_tile_init()`
- `ALWI void rsub_binary_tile_init()`
- `ALWI void div_binary_tile_init()`
- `ALWI void add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `ALWI void sub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `ALWI void mul_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `ALWI void rsub_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `ALWI void div_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/eltwise_unary/activations.h`

Signatures:
- `namespace ckernel`
- `ALWI void abs_tile_init()`
- `ALWI void abs_tile(uint32_t idst)`
- `ALWI void abs_tile_int32(uint32_t idst)`

### `include/jit_hw/api/compute/eltwise_unary/binop_with_scalar.h`

Signatures:
- `namespace ckernel`
- `ALWI void binop_with_scalar_tile_init()`
- `ALWI void add_unary_tile(uint32_t idst, uint32_t param1)`
- `ALWI void sub_unary_tile(uint32_t idst, uint32_t param1)`
- `ALWI void mul_unary_tile(uint32_t idst, uint32_t param1)`
- `ALWI void div_unary_tile(uint32_t idst, uint32_t param1)`
- `ALWI void rsub_unary_tile(uint32_t idst, uint32_t param1)`
- `ALWI void add_unary_tile_int32(uint32_t idst, uint32_t param1)`
- `ALWI void sub_unary_tile_int32(uint32_t idst, uint32_t param1)`

### `include/jit_hw/api/compute/eltwise_unary/bitwise_not.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/clamp.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/comp.h`

Signatures:
- `namespace ckernel`
- `ALWI void unary_eq_tile_init()`
- `ALWI void unary_eq_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_eq_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void unary_ne_tile_init()`
- `ALWI void unary_ne_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_ne_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void unary_gt_tile_init()`
- `ALWI void unary_gt_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_gt_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void unary_ge_tile_init()`
- `ALWI void unary_ge_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_ge_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void unary_lt_tile_init()`
- `ALWI void unary_lt_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_lt_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void unary_le_tile_init()`
- `ALWI void unary_le_tile(uint32_t idst, uint32_t param0)`
- `ALWI void unary_le_tile_int32(uint32_t idst, uint32_t param0)`
- `ALWI void gtz_tile_init()`
- `ALWI void gtz_tile(uint32_t idst)`
- `ALWI void gtz_tile_int32(uint32_t idst)`
- `ALWI void ltz_tile_init()`
- `ALWI void ltz_tile(uint32_t idst)`
- `ALWI void ltz_tile_int32(uint32_t idst)`
- `ALWI void gez_tile_init()`
- `ALWI void gez_tile(uint32_t idst)`
- `ALWI void gez_tile_int32(uint32_t idst)`
- `ALWI void lez_tile_init()`
- `ALWI void lez_tile(uint32_t idst)`
- `ALWI void lez_tile_int32(uint32_t idst)`
- `ALWI void eqz_tile_init()`
- `ALWI void eqz_tile(uint32_t idst)`
- `ALWI void eqz_tile_int32(uint32_t idst)`
- `ALWI void eqz_tile_uint16(uint32_t idst)`
- `ALWI void eqz_tile_uint32(uint32_t idst)`
- `ALWI void nez_tile_init()`
- `ALWI void nez_tile(uint32_t idst)`
- `ALWI void nez_tile_int32(uint32_t idst)`
- `ALWI void nez_tile_uint16(uint32_t idst)`
- `ALWI void nez_tile_uint32(uint32_t idst)`

### `include/jit_hw/api/compute/eltwise_unary/eltwise_unary.h`

Signatures:
- `namespace ckernel`
- `ALWI void unary_op_init_common(uint32_t , uint32_t )`
- `ALWI void init_sfpu(uint32_t , uint32_t )`

### `include/jit_hw/api/compute/eltwise_unary/erf_erfc.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/exp.h`

Signatures:
- `enum class InputClamping`
- `enum class VectorMode`
- `namespace p_sfpu`
- `constexpr uint16_t kCONST_1_FP16B = 0x3F80;`
- `namespace ckernel`
- `template <bool approx = false, bool fast_and_approx = true, bool scale_en = false, bool skip_positive_check = false, InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8> ALWI void exp_tile_init(uint32_t = 0, uint32_t = 0)`
- `template <bool approx = false, bool fast_and_approx = true, bool scale_en = false, bool skip_positive_check = false, InputClamping input_clamping = InputClamping::ClampToNegative, int iterations = 8> ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = p_sfpu::kCONST_1_FP16B)`

### `include/jit_hw/api/compute/eltwise_unary/fill.h`

Signatures:
- `namespace ckernel`
- `ALWI void fill_tile_init()`
- `ALWI void fill_tile(uint32_t idst, float param0)`
- `template <DataFormat DATA_FORMAT = DataFormat::Int32> ALWI void fill_tile_int(uint32_t idst, uint32_t param0)`
- `ALWI void fill_tile_bitcast(uint32_t idst, uint32_t param0)`

### `include/jit_hw/api/compute/eltwise_unary/gelu.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/log1p.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/logical_not.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/negative.h`

Signatures:
- `namespace ckernel`
- `ALWI void negative_tile_init()`
- `ALWI void negative_tile(uint32_t idst)`
- `ALWI void negative_tile_int32(uint32_t idst)`

### `include/jit_hw/api/compute/eltwise_unary/rand.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/rdiv.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/recip.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/relu.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/rounding.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/rpow.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/rsqrt.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/selu.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/sfpu_split_includes.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/sqrt.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/trigonometry.h`

Signatures: none

### `include/jit_hw/api/compute/eltwise_unary/typecast.h`

Signatures:
- `namespace ckernel`
- `template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE> ALWI void typecast_tile_init()`
- `template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE> ALWI void typecast_tile(uint32_t idst)`

### `include/jit_hw/api/compute/eltwise_unary/where.h`

Signatures:
- `namespace ckernel`
- `ALWI void where_tile_init()`
- `template <DataFormat data_format = DataFormat::Float32> ALWI void where_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst)`

### `include/jit_hw/api/compute/experimental/semaphore.h`

Signatures:
- `namespace ckernel`
- `class Semaphore`
- `explicit Semaphore(uint32_t semaphore_id)`
- `void up(uint32_t value)`
- `void down(uint32_t value)`
- `void wait(uint32_t target)`
- `void wait_min(uint32_t min_val)`
- `void set(uint32_t value)`
- `uintptr_t local_l1_addr_;`
- `std::atomic<uint32_t>* atom() const`

### `include/jit_hw/api/compute/gcd.h`

Signatures:
- `namespace ckernel`
- `ALWI void gcd_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void gcd_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/lcm.h`

Signatures:
- `namespace ckernel`
- `ALWI void lcm_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void lcm_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/matmul.h`

Signatures:
- `#define EMULE_MATMUL_USE_AVX2`
- `namespace ckernel`
- `ALWI void mm_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t out_cb = 16, uint32_t transpose = 0)`
- `ALWI void mm_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t transpose = 0)`
- `ALWI void mm_init_short_with_dt(uint32_t in0_cb, uint32_t in1_cb, uint32_t old_in1_cb = 0, uint32_t transpose = 0)`
- `ALWI void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile, uint32_t in1_tile, uint32_t idst)`
- `ALWI void mm_block_init(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t out_cb = 16, uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)`
- `ALWI void mm_block_init_short(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)`
- `ALWI void mm_block_init_short_with_dt(uint32_t in0_cb = 0, uint32_t in1_cb = 1, uint32_t old_in1_cb = 0, uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)`
- `ALWI void matmul_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile, uint32_t in1_tile, uint32_t idst, uint32_t transpose = 0, uint32_t ct_dim = 1, uint32_t rt_dim = 1, uint32_t kt_dim = 1)`
- `namespace experimental`
- `ALWI void matmul_block(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst, const uint32_t transpose, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim, uint32_t nt_dim)`

### `include/jit_hw/api/compute/mul_int_sfpu.h`

Signatures:
- `namespace ckernel`
- `ALWI void mul_int_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void mul_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/nfaces.h`

Signatures:
- `namespace __emule_nfaces`
- `constexpr std::array<uint32_t, 1024> make_rowmajor_to_nfaces()`
- `inline constexpr auto rowmajor_to_nfaces = make_rowmajor_to_nfaces();`

### `include/jit_hw/api/compute/pack.h`

Signatures: none

### `include/jit_hw/api/compute/pack_untilize.h`

Signatures:
- `namespace ckernel`
- `inline void pack_untilize_init(uint32_t cb_out = 0)`
- `inline void pack_untilize_init_short(uint32_t cb_out = 0)`
- `inline void pack_untilize_uninit(uint32_t cb_out = 0)`
- `template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles> inline void pack_untilize_init(uint32_t , uint32_t )`
- `inline void pack_untilize_dst_init_short(uint32_t cb_out, uint32_t ct_dim = 0, uint32_t face_r_dim = 0)`
- `inline void pack_untilize_dst(uint32_t cb_out, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t block_ct_dim = 0, uint32_t pack_dst_offset = 0)`
- `template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim, bool narrow_row = false, uint32_t row_num_datums = 32, bool dense = false> inline void pack_untilize_dest_init(uint32_t ocb = 0, uint32_t face_r_dim = 16, uint32_t num_faces = 4, uint32_t call_line = 0)`
- `template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim, bool diagonal = false, bool narrow_row = false, uint32_t row_num_datums = 32, uint32_t tile_dst_ct_offset = 0, bool dense = false> inline void pack_untilize_dest(uint32_t ocb = 0, uint32_t block_rt_dim = 1, uint32_t block_c_index = 0, uint32_t face_r_dim = 16, uint32_t num_faces = 4, uint32_t tile_dst_rt_offset = 0)`
- `using namespace ckernel;`
- `namespace experimental`
- `template <uint32_t cols_per_dst_pass, uint32_t total_col_tiles> inline void pack_untilize_block(uint32_t icb, uint32_t ocb, uint32_t block_row_tiles, uint32_t block_col_tiles)`

### `include/jit_hw/api/compute/quantization.h`

Signatures:
- `namespace ckernel`
- `ALWI void quant_tile_init(uint32_t)`
- `ALWI void quant_tile(uint32_t, uint32_t, uint32_t)`
- `ALWI void requant_tile_init(uint32_t)`
- `ALWI void requant_tile(uint32_t, uint32_t, uint32_t)`
- `ALWI void dequant_tile_init(uint32_t)`
- `ALWI void dequant_tile(uint32_t, uint32_t, uint32_t)`

### `include/jit_hw/api/compute/reconfig_data_format.h`

Signatures: none

### `include/jit_hw/api/compute/reduce.h`

Signatures:
- `#define REDUCE_OP`
- `#define REDUCE_DIM`
- `namespace ckernel`
- `template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false> inline void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb, uint32_t = 0)`
- `template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false> inline void reduce_init_short(uint32_t icb, uint32_t icb_scaler = 0, uint32_t ocb = 0)`
- `template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false> inline void reduce_init_delta(uint32_t icb = 0, uint32_t icb_scaler = 0, uint32_t ocb = 0)`
- `template <bool enforce_fp32_accumulation = false> inline void reduce_uninit(uint32_t icb = 0)`
- `inline void reduce_revert_delta(uint32_t ocb = 0)`
- `template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false> inline void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst)`
- `template <PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM, bool enforce_fp32_accumulation = false> inline void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4)`
- `inline void reduce_init(uint32_t icb = 0, uint32_t ocb = 0)`
- `inline void reduce_init_short(uint32_t icb = 0, uint32_t ocb = 0)`
- `inline void reduce_init_delta(uint32_t icb = 0, uint32_t ocb = 0)`
- `using namespace ckernel;`

### `include/jit_hw/api/compute/reg_api.h`

Signatures:
- `#define ALWI`
- `ALWI void tile_regs_acquire()`
- `ALWI void tile_regs_commit()`
- `ALWI void tile_regs_wait()`
- `ALWI void tile_regs_release()`

### `include/jit_hw/api/compute/remainder_int32.h`

Signatures:
- `namespace ckernel`
- `ALWI void remainder_int32_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void remainder_int32_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/sub_int_sfpu.h`

Signatures:
- `namespace ckernel`
- `ALWI void sub_int_tile_init()`
- `ALWI void rsub_int_tile_init()`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void sub_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`
- `template<DataFormat Fmt = DataFormat::Int32> ALWI void rsub_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/compute/tile_move_copy.h`

Signatures: none

### `include/jit_hw/api/compute/tilize.h`

Signatures:
- `inline void tilize_init(uint32_t, uint32_t, uint32_t)`
- `inline void tilize_init_short(uint32_t, uint32_t)`

### `include/jit_hw/api/compute/transpose_wh.h`

Signatures:
- `namespace ckernel`
- `inline void transpose_wh_init(uint32_t icb, uint32_t ocb = 0)`
- `inline void transpose_wh_init_short(uint32_t icb = 0)`
- `inline void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t dst_idx)`
- `using namespace ckernel;`

### `include/jit_hw/api/compute/untilize.h`

Signatures:
- `inline void untilize_init(uint32_t, uint32_t = 0)`
- `inline void untilize_init_short(uint32_t)`

### `include/jit_hw/api/compute/xlogy.h`

Signatures:
- `namespace ckernel`
- `ALWI void xlogy_binary_tile_init()`
- `ALWI void xlogy_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)`

### `include/jit_hw/api/dataflow/dataflow_api.h`

Signatures:
- `#define NOC_UNICAST_WRITE_VC`
- `#define NOC_MULTICAST_WRITE_VC`
- `extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr)`
- `extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size)`
- `inline bool __emule_debug_multicast()`
- `inline uint32_t __emule_addr_to_offset(uint32_t addr)`
- `inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr)`
- `inline uint32_t get_absolute_logical_x()`
- `inline uint32_t get_absolute_logical_y()`
- `inline uint64_t get_noc_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc = 0)`
- `inline uint64_t get_noc_addr(uint32_t addr, uint8_t noc = 0)`
- `inline uint64_t get_noc_multicast_addr( uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end, uint32_t addr, uint8_t noc = 0)`
- `template <typename, typename = void> inline constexpr bool has_get_noc_addr_v = false;`
- `template <typename T> inline constexpr bool has_get_noc_addr_v< T, std::void_t<decltype(std::declval<T>().get_noc_addr( std::declval<uint32_t>(), std::declval<uint32_t>(), std::declval<uint8_t>()))`
- `template <typename, typename = void> inline constexpr bool has_page_size_v = false;`
- `template <typename T> inline constexpr bool has_page_size_v<T, std::void_t<decltype(std::declval<T>().page_size)`
- `template <typename, typename = void> inline constexpr bool has_log_base_2_of_page_size_v = false;`
- `template <typename T> inline constexpr bool has_log_base_2_of_page_size_v< T, std::void_t<decltype(std::declval<T>().log_base_2_of_page_size)`
- `template<typename AddrGen> FORCE_INLINE void noc_async_read_page( uint32_t id, const AddrGen& addrgen, uint32_t dst_local_l1_addr, uint32_t offset = 0, uint8_t noc = 0)`
- `template<typename AddrGen> FORCE_INLINE void noc_async_write_page( uint32_t id, const AddrGen& addrgen, uint32_t src_local_l1_addr, uint32_t size = 0, uint32_t offset = 0, uint8_t noc = 0)`
- `template<typename AddrGen> FORCE_INLINE void noc_async_read_tile( uint32_t id, const AddrGen& addrgen, uint32_t dst_local_l1_addr, uint32_t offset = 0, uint8_t noc = 0)`
- `template<typename AddrGen> FORCE_INLINE void noc_async_write_tile( uint32_t id, const AddrGen& addrgen, uint32_t src_local_l1_addr, uint32_t size = 0, uint32_t offset = 0, uint8_t noc = 0)`
- `inline void noc_async_read(uint64_t src_noc_addr, uint32_t dst_local_l1_addr, uint32_t size, uint8_t noc = 0, uint32_t vc = 0)`
- `inline void noc_async_write(uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t noc = 0, uint32_t vc = 0)`
- `inline void noc_async_write_multicast( uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr, uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0)`
- `inline void noc_async_write_multicast_loopback_src( uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr, uint32_t size, uint32_t num_dests, bool linked = false, uint8_t noc = 0)`
- `inline void noc_async_read_barrier()`
- `inline void noc_async_write_barrier()`
- `inline void noc_async_writes_flushed()`
- `#define EMULE_SEM_BASE`
- `#define EMULE_SEM_ALIGN`
- `inline uint32_t get_semaphore(uint32_t semaphore_id)`
- `inline std::atomic<uint32_t>* __emule_sem_atomic(volatile tt_l1_ptr uint32_t* sem_addr)`
- `inline void noc_semaphore_set(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val)`
- `inline void noc_semaphore_wait(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val)`
- `inline void noc_semaphore_wait_min(volatile tt_l1_ptr uint32_t* sem_addr, uint32_t min_val)`
- `inline void noc_semaphore_inc(uint64_t noc_addr, uint32_t incr, uint8_t noc = 0)`
- `inline void noc_semaphore_set_multicast( uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr, uint32_t num_dests, bool linked = false, uint8_t noc = 0)`
- `inline void noc_semaphore_set_multicast_loopback_src( uint32_t src_local_l1_addr, uint64_t dst_mcast_noc_addr, uint32_t num_dests, bool linked = false, uint8_t noc = 0)`
- `struct CBInterface`
- `uint32_t fifo_page_size;`
- `inline CBInterface get_local_cb_interface(uint32_t cb_id)`
- `inline uint64_t get_dram_noc_addr( const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0, uint8_t noc = 0)`
- `inline uint64_t get_l1_noc_addr( const uint32_t id, const uint32_t page_size, const uint32_t bank_base_address, const uint32_t offset = 0, uint8_t noc = 0)`
- `inline void dram_barrier()`
- `struct __emule_dw_state`
- `uint64_t addr = 0;`
- `uint32_t val = 0;`
- `inline thread_local __emule_dw_state __emule_dw_st;`
- `inline void __emule_dw_write_be(uint8_t* dst, uint32_t val, uint8_t be)`
- `template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true> inline void noc_inline_dw_write( uint64_t addr, uint32_t val, uint8_t be = 0xF, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC, uint32_t customized_src_addr = 0)`
- `template <InlineWriteDst dst_type = InlineWriteDst::DEFAULT, bool posted = false, bool flush = true> inline void noc_inline_mcast_dw_write( uint64_t addr, uint32_t val, uint8_t be = 0xF, uint8_t noc = noc_index, uint8_t vc = NOC_MULTICAST_WRITE_VC, uint32_t customized_src_addr = 0, uint32_t num_dest = 1)`
- `template <bool posted = false, bool set_val = false> inline void noc_inline_dw_write_set_state( uint64_t addr, uint32_t val = 0, uint8_t be = 0xF, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index, uint8_t vc = NOC_UNICAST_WRITE_VC)`
- `template < bool update_addr_lo = false, bool update_counter = true, bool posted = false, bool update_addr_hi = false, bool update_val = false, InlineWriteDst dst_type = InlineWriteDst::DEFAULT> inline void noc_inline_dw_write_with_state( uint32_t val, uint32_t addr = 0, uint8_t cmd_buf = write_at_cmd_buf, uint8_t noc = noc_index)`
- `namespace experimental`
- `inline std::uint64_t get_noc_multicast_addr( std::uint32_t noc_x_start, std::uint32_t noc_y_start, std::uint32_t noc_x_end, std::uint32_t noc_y_end, std::uint32_t addr, uint8_t noc = noc_index)`

### `include/jit_hw/api/debug/assert.h`

Signatures:
- `#define ASSERT(condition, ...)`
- `#define ASSERT_ENABLED`

### `include/jit_hw/api/debug/device_print.h`

Signatures:
- `#define DEVICE_PRINT(format, ...)`
- `#define DEVICE_PRINT_UNPACK(format, ...)`
- `#define DEVICE_PRINT_PACK(format, ...)`

### `include/jit_hw/api/debug/dprint.h`

Signatures:
- `namespace tt_emule_jit`
- `struct DPrintSink`
- `template<typename T> const DPrintSink& operator<<(const T&) const`
- `inline DPrintSink make_dprint()`
- `#define DPRINT`
- `#define ENDL()`
- `struct HEX`
- `struct DEC`
- `struct OCT`
- `struct BIN`
- `struct FIXED`
- `struct DEFAULTFLOAT`
- `struct SETW`
- `int v;`
- `constexpr SETW(int v_)`
- `struct SETPRECISION`
- `int v;`
- `constexpr SETPRECISION(int v_)`

### `include/jit_hw/api/dfb_api.h`

Signatures:
- `inline int __emule_dfb_timeout_sec()`
- `inline void __emule_dfb_check_id(uint32_t dfb_id, const char* caller)`
- `inline void dfb_reserve_back(uint32_t dfb_id, uint16_t n)`
- `inline void dfb_push_back(uint32_t dfb_id, uint16_t n)`
- `inline void dfb_wait_front(uint32_t dfb_id, uint16_t n)`
- `inline void dfb_pop_front(uint32_t dfb_id, uint16_t n)`
- `inline void dfb_finish(uint32_t dfb_id)`
- `inline uint32_t dfb_get_write_ptr(uint32_t dfb_id)`
- `inline uint32_t dfb_get_read_ptr(uint32_t dfb_id)`
- `inline uint32_t dfb_get_entry_size(uint32_t dfb_id)`

### `include/jit_hw/api/kernel_thread_globals.h`

Signatures:
- `extern thread_local uint32_t __emule_num_threads;`
- `extern thread_local uint32_t __emule_my_thread_id;`
- `inline uint32_t get_num_threads()`
- `inline uint32_t get_my_thread_id()`

### `include/jit_hw/api/tensor/tensor_accessor.h`

Signatures:
- `struct TensorAccessor`
- `uint32_t bank_base_address;`
- `uint32_t page_size;`
- `template<std::size_t CTA, std::size_t CRTA> TensorAccessor(const TensorAccessorArgs<CTA, CRTA>&, size_t addr, uint32_t ps = TensorAccessorArgs<CTA, CRTA>::AlignedPageSize)`
- `inline uint64_t get_noc_addr(uint32_t page_id, uint32_t offset = 0, uint8_t noc = 0) const`

### `include/jit_hw/chlkc_list.h`

Signatures: none

### `include/jit_hw/ckernel.h`

Signatures:
- `extern thread_local uint8_t __emule_neo_id;`
- `extern thread_local uint8_t __emule_trisc_id;`
- `namespace ckernel`
- `enum class CSR`
- `template <CSR csr> inline uint32_t csr_read()`
- `template <> inline uint32_t csr_read<CSR::NEO_ID>()`
- `template <> inline uint32_t csr_read<CSR::TRISC_ID>()`

### `include/jit_hw/ckernel_defs.h`

Signatures: none

### `include/jit_hw/ckernel_include.h`

Signatures:
- `namespace ckernel`
- `enum firmware_msg_e`

### `include/jit_hw/dev_mem_map.h`

Signatures:
- `extern thread_local uint8_t* __emule_bridge_l1;`
- `#define MEM_L1_UNCACHED_BASE`
- `constexpr uint32_t NUM_TRISC_CORES = 4;`
- `constexpr uint32_t NUM_DM_CORES = 8;`
- `constexpr uint32_t MEM_ZEROS_SIZE = 512;`
- `constexpr uint32_t MEM_ZEROS_BASE = 0xFFE00;`

### `include/jit_hw/emule_cb_state.h`

Signatures:
- `using __emule_cb_state = tt_emule::CBSyncState;`
- `extern thread_local __emule_cb_state* __emule_cbs;`

### `include/jit_hw/emule_dfb_state.h`

Signatures:
- `using __emule_dfb_iface = tt_emule::EmuleDFBInterface;`
- `extern thread_local __emule_dfb_iface* __emule_dfbs;`
- `extern thread_local tt_emule::TileCounterArray* __emule_tc_array;`
- `extern thread_local uint8_t __processor_id;`

### `include/jit_hw/experimental/circular_buffer.h`

Signatures:
- `namespace experimental`
- `class CircularBuffer`
- `enum class AddrSelector`
- `explicit CircularBuffer(uint32_t cb_id)`
- `uint32_t get_cb_id() const`
- `void reserve_back(int32_t n)`
- `void push_back(int32_t n)`
- `void wait_front(int32_t n)`
- `void pop_front(int32_t n)`
- `bool pages_reservable_at_back(int32_t n) const`
- `bool pages_available_at_front(int32_t n) const`
- `uint32_t get_tile_size() const`
- `uint32_t get_write_ptr() const`
- `uint32_t get_read_ptr() const`
- `uint32_t cb_id_;`
- `template <> struct noc_traits_t`
- `struct src_args_type`
- `struct dst_args_type`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const CircularBuffer& cb, const Noc&, const src_args_type& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const CircularBuffer& cb, const Noc&, const dst_args_type& args)`
- `template <CircularBuffer::AddrSelector AddrSel> struct CircularBufferView`
- `const CircularBuffer& cb;`
- `explicit constexpr CircularBufferView(const CircularBuffer& c)`
- `template <CircularBuffer::AddrSelector AddrSel> constexpr auto use(const CircularBuffer& cb)`
- `template <CircularBuffer::AddrSelector AddrSel> struct noc_traits_t`
- `struct src_args_type`
- `struct dst_args_type`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const CircularBufferView<AddrSel>& view, const Noc&, const src_args_type& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const CircularBufferView<AddrSel>& view, const Noc&, const dst_args_type& args)`
- `static constexpr auto get_local_addr(const CircularBufferView<AddrSel>& view)`

### `include/jit_hw/experimental/core_local_mem.h`

Signatures:
- `extern thread_local tt_emule::Core* __core;`
- `extern thread_local uint8_t* __emule_bridge_l1;`
- `namespace experimental`
- `static constexpr uintptr_t CORE_LOCAL_MEM_RAW_OFFSET_THRESHOLD = 0x1000000;`
- `template <typename T, typename AddressType = uintptr_t> class CoreLocalMem`
- `using difference_type = std::ptrdiff_t;`
- `static uintptr_t translate(uintptr_t addr)`
- `CoreLocalMem(AddressType address)`
- `CoreLocalMem(T* ptr)`
- `CoreLocalMem(const CoreLocalMem&) = default`
- `CoreLocalMem& operator=(const CoreLocalMem&) = default;`
- `T* get_unsafe_ptr() const`
- `AddressType get_address() const`
- `T* data()`
- `T& operator[](uint32_t index) const`
- `T& operator*() const`
- `T* operator->() const`
- `CoreLocalMem& operator+=(difference_type offset)`
- `CoreLocalMem& operator-=(difference_type offset)`
- `CoreLocalMem& operator++()`
- `CoreLocalMem& operator--()`
- `CoreLocalMem operator++(int)`
- `CoreLocalMem operator--(int)`
- `CoreLocalMem operator+(difference_type offset) const`
- `CoreLocalMem operator-(difference_type offset) const`
- `difference_type operator-(const CoreLocalMem& other) const`
- `bool operator==(const CoreLocalMem& other) const`
- `bool operator!=(const CoreLocalMem& other) const`
- `bool operator<(const CoreLocalMem& other) const`
- `bool operator<=(const CoreLocalMem& other) const`
- `bool operator>(const CoreLocalMem& other) const`
- `bool operator>=(const CoreLocalMem& other) const`
- `explicit operator bool() const`
- `operator uint8_t*()`
- `uintptr_t address_;`
- `template <typename T, typename AddressType> struct noc_traits_t`
- `struct src_args_type`
- `uintptr_t offset_bytes = 0;`
- `struct dst_args_type`
- `uintptr_t offset_bytes = 0;`
- `struct dst_args_mcast_type`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const CoreLocalMem<T, AddressType>& src, const Noc&, const src_args_type& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const CoreLocalMem<T, AddressType>& dst, const Noc&, const dst_args_type& args)`

### `include/jit_hw/experimental/dataflow_buffer.h`

Signatures:
- `namespace experimental`
- `struct DFBAccessor`
- `explicit constexpr DFBAccessor(uint16_t id) noexcept`
- `uint16_t id;`
- `class DataflowBuffer`
- `DataflowBuffer(uint16_t logical_dfb_id)`
- `DataflowBuffer(DFBAccessor accessor)`
- `uint16_t get_id() const`
- `uint32_t get_entry_size() const`
- `uint32_t get_stride_size() const`
- `void reserve_back(uint16_t num_entries)`
- `void push_back(uint16_t num_entries)`
- `void wait_front(uint16_t num_entries)`
- `void pop_front(uint16_t num_entries)`
- `void finish()`
- `void write_barrier(const Noc& noc) const`
- `uint32_t get_write_ptr() const`
- `uint32_t get_read_ptr() const`
- `template <typename Src> void read_in(const Noc& noc, const Src& src, const typename noc_traits_t<Src>::src_args_type& src_args)`
- `template <typename Dst> void write_out(const Noc& noc, const Dst& dst, const typename noc_traits_t<Dst>::dst_args_type& dst_args)`
- `[[nodiscard]] auto scoped_lock()`
- `uint16_t logical_dfb_id_;`
- `template <> struct noc_traits_t`
- `struct src_args_type`
- `struct dst_args_type`
- `struct dst_args_mcast_type`
- `template <Noc::AddressType address_type> static uintptr_t src_addr(const DataflowBuffer& src, const Noc&, const src_args_type& args)`
- `template <Noc::AddressType address_type> static uintptr_t dst_addr(const DataflowBuffer& dst, const Noc&, const dst_args_type& args)`
- `template <Noc::AddressType address_type> static uintptr_t dst_addr_mcast(const DataflowBuffer& dst, const Noc&, const dst_args_mcast_type& args)`
- `template <Noc::TxnIdMode txn_id_mode, typename Src> inline std::enable_if_t<txn_id_mode == Noc::TxnIdMode::ENABLED> Noc::async_read( const Src& src, DataflowBuffer& dst, const typename noc_traits_t<Src>::src_args_type& src_args, const DataflowBufferArgs& dst_args) const`
- `template <Noc::TxnIdMode txn_id_mode, typename Dst> inline std::enable_if_t<txn_id_mode == Noc::TxnIdMode::ENABLED> Noc::async_write( DataflowBuffer& src, const Dst& dst, const DataflowBufferArgs& src_args, const typename noc_traits_t<Dst>::dst_args_type& dst_args) const`

### `include/jit_hw/experimental/endpoints.h`

Signatures:
- `extern "C" uint8_t* __emule_dram_ptr(uint64_t offset)`
- `extern "C" uint8_t* __emule_local_l1_ptr(uint32_t offset)`
- `extern uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr)`
- `namespace experimental`
- `struct UnicastEndpoint`
- `uint64_t get_noc_unicast_addr(uint32_t noc_x, uint32_t noc_y, uint32_t addr, uint8_t noc) const`
- `struct MulticastEndpoint`
- `uint64_t get_noc_multicast_addr( uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr, uint8_t noc) const`
- `enum AllocatorBankType`
- `template <AllocatorBankType> struct AllocatorBank`
- `struct ReadSpec`
- `uint32_t bank_id = 0;`
- `uint32_t addr = 0;`
- `struct WriteSpec`
- `uint32_t bank_id = 0;`
- `uint32_t addr = 0;`
- `template <> struct noc_traits_t`
- `struct src_args_type`
- `struct dst_args_type`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const UnicastEndpoint& src, const Noc& noc, const src_args_type& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const UnicastEndpoint& dst, const Noc& noc, const dst_args_type& args)`
- `template <> struct noc_traits_t`
- `struct dst_args_mcast_type`
- `template <Noc::AddressType AT> static uint64_t dst_addr_mcast(const MulticastEndpoint& dst, const Noc& noc, const dst_args_mcast_type& args)`
- `template <> struct noc_traits_t`
- `using src_args_type = ReadSpec;`
- `using dst_args_type = WriteSpec;`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const AllocatorBank<L1>&, const Noc&, const ReadSpec& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const AllocatorBank<L1>&, const Noc&, const WriteSpec& args)`
- `template <> struct noc_traits_t`
- `using src_args_type = ReadSpec;`
- `using dst_args_type = WriteSpec;`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const AllocatorBank<DRAM>&, const Noc&, const ReadSpec& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const AllocatorBank<DRAM>&, const Noc&, const WriteSpec& args)`

### `include/jit_hw/experimental/kernel_args.h`

Signatures:
- `namespace experimental`
- `template <typename T> struct RtaArg`
- `uint32_t byte_offset;`
- `template <typename T> struct CrtaArg`
- `uint32_t byte_offset;`
- `template <typename T> struct CtaVal`
- `T value;`
- `template <typename T> FORCE_INLINE T get_arg(RtaArg<T> arg)`
- `template <typename T> FORCE_INLINE T get_arg(CrtaArg<T> arg)`
- `template <typename T> FORCE_INLINE constexpr T get_arg(CtaVal<T> arg)`

### `include/jit_hw/experimental/lock.h`

Signatures:
- `namespace experimental`
- `template <typename ReleaseFunc> class Lock`
- `inline __attribute__((always_inline))`
- `Lock(const Lock&) = delete`
- `Lock(Lock&&) = delete`
- `Lock& operator=(const Lock&) = delete;`
- `Lock& operator=(Lock&&) = delete;`
- `ReleaseFunc release_func_;`

### `include/jit_hw/experimental/noc.h`

Signatures:
- `extern "C" void __emule_multicast_write(uint64_t mcast_addr, const uint8_t* src, uint32_t size)`
- `#define NOC_MAX_BURST_SIZE`
- `namespace experimental`
- `class DataflowBuffer`
- `struct DataflowBufferArgs`
- `template <typename T> struct noc_traits_t`
- `static_assert(sizeof(T) == 0, "NoC transactions are not supported for this type")`
- `class Noc`
- `enum class AddressType`
- `enum class TxnIdMode`
- `enum class VcSelection`
- `Noc()`
- `explicit Noc(uint8_t noc_id)`
- `uint8_t get_noc_id() const`
- `template <TxnIdMode txn_id_mode = TxnIdMode::DISABLED, typename Src, typename Dst> void async_read( const Src& src, const Dst& dst, uint32_t size_bytes, const typename noc_traits_t<Src>::src_args_type& src_args, const typename noc_traits_t<Dst>::dst_args_type& dst_args) const`
- `template <TxnIdMode txn_id_mode = TxnIdMode::DISABLED, typename Src, typename Dst> void async_write( const Src& src, const Dst& dst, uint32_t size_bytes, const typename noc_traits_t<Src>::src_args_type& src_args, const typename noc_traits_t<Dst>::dst_args_type& dst_args) const`
- `template <TxnIdMode txn_id_mode, typename Src> std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED> async_read( const Src& src, DataflowBuffer& dst, const typename noc_traits_t<Src>::src_args_type& src_args, const DataflowBufferArgs& dst_args = {}) const`
- `template <TxnIdMode txn_id_mode, typename Dst> std::enable_if_t<txn_id_mode == TxnIdMode::ENABLED> async_write( DataflowBuffer& src, const Dst& dst, const DataflowBufferArgs& src_args, const typename noc_traits_t<Dst>::dst_args_type& dst_args) const`
- `template <typename... Extra, typename Src, typename Dst> void async_write_multicast( const Src& src, const Dst& dst, uint32_t size_bytes, uint32_t num_dsts, const typename noc_traits_t<Src>::src_args_type& src_args, const typename noc_traits_t<Dst>::dst_args_mcast_type& dst_args, bool linked = false, uint32_t trid = 0) const`
- `void async_read_barrier() const`
- `void async_write_barrier() const`
- `void async_writes_flushed() const`
- `void async_atomic_barrier() const`
- `void async_full_barrier() const`
- `template <VcSelection vc_selection = VcSelection::DEFAULT, uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, typename Src> void set_async_read_state( const Src& , uint32_t size_bytes, const typename noc_traits_t<Src>::src_args_type& , uint8_t = 0) const`
- `template <VcSelection vc_selection = VcSelection::DEFAULT, uint32_t max_page_size = NOC_MAX_BURST_SIZE + 1, typename Src, typename Dst> void async_read_with_state( const Src& src, const Dst& dst, uint32_t size_bytes, const typename noc_traits_t<Src>::src_args_type& src_args, const typename noc_traits_t<Dst>::dst_args_type& dst_args, uint8_t = 0) const`
- `uint8_t noc_id_;`
- `mutable uint32_t cached_size_ = 0;`

### `include/jit_hw/experimental/noc_semaphore.h`

Signatures:
- `extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr)`
- `extern thread_local uint8_t* __emule_bridge_l1;`
- `namespace experimental`
- `template <int core_type = 0> class Semaphore`
- `explicit Semaphore(uint32_t semaphore_id)`
- `void up(uint32_t value)`
- `void up(const Noc&, uint32_t remote_noc_x, uint32_t remote_noc_y, uint32_t value, uint8_t = 0)`
- `void down(uint32_t value)`
- `void wait(uint32_t target)`
- `void wait_min(uint32_t min_val)`
- `void set(uint32_t value)`
- `template <typename M = void> void set_multicast(const Noc&, uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end, [[maybe_unused]] uint32_t num_cores, bool linked = false)`
- `void inc_multicast(const Noc&, uint32_t x_start, uint32_t y_start, uint32_t x_end, uint32_t y_end, [[maybe_unused]] uint32_t num_cores, uint32_t value)`
- `uintptr_t local_l1_addr_;`
- `uint32_t l1_offset_;`
- `std::atomic<uint32_t>* atom() const`

### `include/jit_hw/experimental/tensor.h`

Signatures:
- `extern "C" uint8_t* __emule_resolve_noc_addr(uint64_t noc_addr)`
- `namespace experimental`
- `template <> struct noc_traits_t`
- `struct src_args_type`
- `uint32_t offset_bytes = 0;`
- `struct dst_args_type`
- `uint32_t offset_bytes = 0;`
- `template <Noc::AddressType AT> static uintptr_t src_addr(const TensorAccessor& src, const Noc& noc, const src_args_type& args)`
- `template <Noc::AddressType AT> static uintptr_t dst_addr(const TensorAccessor& dst, const Noc& noc, const dst_args_type& args)`

### `include/jit_hw/hostdevcommon/common_values.hpp`

Signatures:
- `constexpr uint32_t INVALID = 0;`
- `constexpr uint32_t VALID = 1;`

### `include/jit_hw/internal/dataflow/dataflow_api_addrgen.h`

Signatures:
- `extern uint16_t dram_bank_to_noc_xy[2][32];`
- `extern int32_t bank_to_dram_offset[32];`
- `extern uint16_t l1_bank_to_noc_xy[2][32];`
- `extern int32_t bank_to_l1_offset[32];`
- `extern thread_local uint8_t my_x[2];`
- `extern thread_local uint8_t my_y[2];`
- `#define NOC_ADDR_LOCAL_BITS`
- `#define NOC_ADDR_NODE_ID_BITS`
- `#define NOC_ADDR_COORD_SHIFT`
- `#define DYNAMIC_NOC_X(noc, x)`
- `#define DYNAMIC_NOC_Y(noc, y)`
- `#define NOC_XY_ADDR(x, y, addr)`
- `inline constexpr uint32_t align_power_of_2(uint32_t addr, uint32_t alignment)`
- `#define NUM_DRAM_BANKS`
- `#define NUM_L1_BANKS`
- `#define DRAM_ALIGNMENT`
- `#define L1_ALIGNMENT`
- `namespace interleaved_addr_gen`
- `template <bool DRAM> inline uint32_t get_bank_offset_index(uint32_t id)`
- `template <bool DRAM> inline uint32_t get_bank_index(uint32_t id, uint32_t bank_offset_index)`
- `template <bool DRAM> inline uint32_t get_noc_xy(uint32_t bank_index, uint8_t noc = 0)`
- `template <bool DRAM> inline uint32_t get_bank_offset(uint32_t bank_index)`
- `template <bool DRAM> inline constexpr uint32_t get_allocator_alignment()`
- `inline uint64_t get_noc_addr_helper(uint32_t noc_xy, uint32_t addr)`
- `template <bool DRAM> inline uint64_t get_noc_addr_from_bank_id(uint32_t bank_id, uint32_t bank_address_offset, uint8_t noc = noc_index)`
- `template <bool DRAM> struct InterleavedAddrGen`
- `static constexpr bool is_dram = DRAM;`
- `uint32_t bank_base_address;`
- `const uint32_t page_size;`
- `const uint32_t aligned_page_size = align_power_of_2(page_size, interleaved_addr_gen::get_allocator_alignment<DRAM>());`
- `inline uint32_t get_addr( const uint32_t id, const uint32_t bank_offset_index, const uint32_t bank_index, const uint32_t offset = 0) const`
- `inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const`
- `template <bool DRAM> struct InterleavedPow2AddrGen`
- `static constexpr bool is_dram = DRAM;`
- `const uint32_t bank_base_address;`
- `const uint32_t log_base_2_of_page_size;`
- `inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const`
- `template <bool DRAM, uint32_t tile_hw = 1024> struct InterleavedAddrGenFast`
- `static constexpr bool is_dram = DRAM;`
- `uint32_t bank_base_address;`
- `uint32_t page_size;`
- `DataFormat data_format;`
- `inline uint64_t get_noc_addr(const uint32_t id, const uint32_t offset = 0, uint8_t noc = 0) const`

### `include/jit_hw/internal/firmware_common.h`

Signatures:
- `inline void invalidate_l1_cache()`
- `inline void flush_l1_cache()`
- `#define WAYPOINT(...)`

### `include/jit_hw/internal/llk_state.h`

Signatures:
- `struct CbInterface`
- `uint32_t fifo_page_size = 0;`
- `uint32_t fifo_rd_ptr = 0;`
- `uint32_t fifo_wr_ptr = 0;`
- `uint32_t fifo_size = 0;`
- `uint32_t fifo_limit = 0;`
- `uint32_t fifo_num_pages = 0;`
- `inline CbInterface& get_local_cb_interface(uint32_t)`
- `inline uint32_t get_operand_id(uint32_t operand)`
- `inline uint32_t get_output_id(uint32_t output)`
- `inline uint32_t get_operand_face_r_dim(uint32_t)`
- `inline uint32_t get_operand_num_faces(uint32_t)`
- `inline bool get_operand_narrow_tile(uint32_t)`
- `inline bool get_output_partial_face(uint32_t)`
- `inline bool get_output_narrow_tile(uint32_t)`
- `inline uint32_t get_output_face_r_dim(uint32_t)`
- `inline uint32_t get_output_num_faces(uint32_t)`
- `static thread_local uint32_t __llk_unpack_src_cb = 0;`
- `static thread_local uint32_t __llk_unpack_start_tile_idx = 0;`
- `static thread_local uint32_t __llk_unpack_block_c = 0;`
- `static thread_local uint32_t __llk_unpack_current_tile = 0;`
- `static thread_local bool __llk_unpack_is_tilize = false;`
- `static thread_local uint32_t __llk_pack_offset = 0;`
- `static thread_local bool __llk_pack_is_untilize = false;`
- `static thread_local uint32_t __llk_pack_block_c = 0;`

### `include/jit_hw/internal/mod_div_lib.h`

Signatures:
- `inline __attribute__((always_inline))`
- `template <uint32_t d> inline __attribute__((always_inline))`

### `include/jit_hw/internal/risc_attribs.h`

Signatures:
- `#define tt_l1_ptr`
- `#define FORCE_INLINE`
- `enum class InlineWriteDst`
- `constexpr uint32_t write_at_cmd_buf = 0;`

### `include/jit_hw/internal/tt-2xx/quasar/overlay/overlay_addresses.h`

Signatures:
- `namespace (anonymous)`
- `volatile uint64_t __emule_l2_flush_sink = 0;`
- `#define L2_FLUSH_ADDR`

### `include/jit_hw/jit_kernel_stubs.hpp`

Signatures:
- `#define __EMULE_JIT_MODE`
- `namespace tt_emule`
- `class Core`
- `class Device`
- `extern thread_local std::vector<uint32_t> __rt_args;`
- `extern thread_local std::vector<uint32_t> __common_rt_args;`
- `extern thread_local tt_emule::Core* __core;`
- `extern thread_local tt_emule::Device* __device;`
- `extern "C" uint8_t* __emule_dram_ptr(uint64_t offset)`
- `extern "C" uint8_t* __emule_noc_resolve(uint32_t x, uint32_t y, uint64_t addr)`
- `extern thread_local uint8_t* __emule_bridge_l1;`
- `inline uint8_t* __emule_local_l1_to_ptr(uint32_t l1_addr)`
- `extern uint16_t dram_bank_to_noc_xy[2][32];`
- `extern int32_t bank_to_dram_offset[32];`
- `extern uint16_t l1_bank_to_noc_xy[2][32];`
- `extern int32_t bank_to_l1_offset[32];`
- `extern thread_local uint8_t my_x[2];`
- `extern thread_local uint8_t my_y[2];`
- `extern thread_local uint32_t __emule_logical_x;`
- `extern thread_local uint32_t __emule_logical_y;`
- `extern thread_local uint8_t __processor_id;`
- `extern thread_local uint8_t __emule_neo_id;`
- `extern thread_local uint8_t __emule_trisc_id;`
- `extern thread_local uint32_t __emule_num_threads;`
- `extern thread_local uint32_t __emule_my_thread_id;`
- `#define tt_l1_ptr`
- `constexpr uint8_t noc_index = 0;`
- `static inline uintptr_t get_arg_addr(int arg_idx)`
- `static inline uintptr_t get_common_arg_addr(int arg_idx)`
- `template<typename T = uint32_t> inline T get_arg_val(int arg_idx)`
- `template<typename T = uint32_t> inline T get_common_arg_val(int arg_idx)`
- `#define ENABLE_FP32_DEST_ACC`
- `#define DST_SYNC_FULL`
- `#define ASSERT(...)`
- `#define EMULE_SEM_ALIGN`
- `inline uint32_t get_semaphore(uint32_t semaphore_id)`
- `namespace ckernel`
- `enum ThreadId`
- `inline uint32_t mailbox_read(uint8_t )`
- `inline void mailbox_write(uint8_t , uint32_t )`

### `include/jit_hw/llk/llk_reduce_primitives.h`

Signatures:
- `namespace ckernel`
- `enum class MathFidelity`
- `enum class PoolType`
- `enum class ReduceDim`
- `constexpr int MM_THROTTLE = 0;`
- `#define MATH_FIDELITY`
- `struct __emule_matmul_bridge`
- `uint32_t in0_cb = 0;`
- `uint32_t in1_cb = 0;`
- `uint32_t in0_idx = 0;`
- `uint32_t in1_idx = 0;`
- `inline thread_local __emule_matmul_bridge __emule_matmul_state;`
- `inline void state_configure(uint32_t , uint32_t )`
- `template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args> inline void llk_math_matmul_init(Args... )`
- `template <typename... Args> inline void llk_unpack_AB_matmul_init(Args... )`
- `template <int Mode, typename... Args> inline void llk_unpack_reconfig_data_format_srca(Args... )`
- `template <int Mode, typename... Args> inline void llk_math_reconfig_data_format_srca(Args... )`
- `inline void llk_unpack_AB_matmul(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_idx, uint32_t in1_idx)`
- `namespace ckernel`
- `void matmul_tiles(uint32_t in0_cb, uint32_t in1_cb, uint32_t in0_tile, uint32_t in1_tile, uint32_t idst)`
- `template <ckernel::MathFidelity Fidelity, int Throttle, typename... Args> inline void llk_math_matmul(uint32_t idst, Args... )`
- `template <PoolType Pool, ReduceDim Dim, typename... Args> inline void llk_unpack_AB_reduce_init(Args... )`
- `template <PoolType Pool, ReduceDim Dim, int Mode, ckernel::MathFidelity Fidelity, typename... Args> inline void llk_math_reduce_init(Args... )`

### `include/jit_hw/llk_defs.h`

Signatures: none

### `include/jit_hw/llk_math_eltwise_binary.h`

Signatures: none

### `include/jit_hw/llk_math_eltwise_unary_datacopy.h`

Signatures:
- `using ckernel::BroadcastType;`
- `inline void __llk_tilize_datacopy(uint32_t dst_idx)`
- `inline void __llk_untilize_datacopy(uint32_t dst_idx)`
- `template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast, bool UnpackToDest> inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx)`
- `template <ckernel::DataCopyType CopyType, int AccumMode, BroadcastType Bcast> inline void llk_math_eltwise_unary_datacopy(uint32_t dst_idx)`

### `include/jit_hw/llk_pack.h`

Signatures:
- `inline void __llk_pack_tiled(uint32_t tile_idx, uint32_t ocb)`
- `inline void __llk_pack_untilize(uint32_t tile_idx, uint32_t ocb)`
- `template <bool Untilize, bool IsApprox> inline void llk_pack(uint32_t tile_idx, uint32_t ocb)`
- `template <int AccumMode, bool Untilize, bool IsApprox> inline void llk_pack(uint32_t tile_idx, uint32_t ocb)`
- `template <int AccumMode> inline void llk_math_dest_section_done()`
- `template <int AccumMode> inline void llk_pack_dest_section_done()`

### `include/jit_hw/llk_sync_stubs.h`

Signatures:
- `inline void llk_wait_tiles(int operand, std::int32_t num_tiles)`
- `inline void llk_pop_tiles(std::int32_t operand, std::int32_t num_tiles, std::int32_t = 0)`
- `template <bool = false, bool = false, bool = false> inline void llk_wait_for_free_tiles(std::int32_t operand, std::int32_t num_tiles)`
- `template <bool = false, bool = false> inline void llk_push_tiles(std::int32_t operand, std::int32_t num_tiles)`
- `inline void llk_math_wait_for_dest_available()`
- `inline void llk_packer_wait_for_math_done()`

### `include/jit_hw/llk_types.h`

Signatures:
- `#define FACE_R_DIM`
- `#define TILE_C_DIM`
- `enum class PoolType`
- `enum class ReduceDim`
- `namespace ckernel`
- `using ::PoolType;`
- `using ::ReduceDim;`
- `enum class MathFidelity`
- `enum class DstSync`
- `enum class DataCopyType`
- `inline constexpr bool UnpackToDestEn = false;`

### `include/jit_hw/llk_unpack_a.h`

Signatures:
- `template <typename... Args> inline void _llk_unpack_tilize_(Args&&...)`
- `inline void llk_unpack_tilize(uint32_t, uint32_t, uint32_t, uint32_t)`
- `inline void llk_unpack_tilize_block(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx)`
- `inline void llk_unpack_untilize(uint32_t icb, uint32_t block_c, uint32_t start_tile_idx)`
- `template <ckernel::BroadcastType BType = ckernel::BroadcastType::NONE, bool acc_to_dest = false, ckernel::EltwiseBinaryReuseDestType binary_reuse_dest = ckernel::EltwiseBinaryReuseDestType::NONE, bool unpack_to_dest = false> inline void llk_unpack_A(uint32_t , uint32_t )`

### `include/jit_hw/risc_common.h`

Signatures:
- `inline void flush_l2_cache_line(uintptr_t)`

### `include/jit_hw/tools/profiler/kernel_profiler.hpp`

Signatures:
- `#define DeviceZoneScopedN(...)`
- `#define DeviceZoneScoped(...)`
- `#define DeviceZoneScopedMainN(...)`
- `#define DeviceZoneScopedMainChildN(...)`
- `#define DeviceTimestampedData(...)`
- `namespace kernel_profiler`
- `inline void mark_time(unsigned = 0)`
- `inline void mark_padding()`
- `inline void set_host_counter(unsigned)`
- `inline void init_profiler(unsigned = 0, unsigned = 0)`
- `inline void finish_profiler()`

### `include/jit_hw/tt-metalium/buffer_types.hpp`

Signatures:
- `namespace tt::tt_metal`
- `enum class TensorMemoryLayout`
- `enum class ShardOrientation`
- `enum class ShardDistributionStrategy`
- `enum class BufferType`

### `include/jit_hw/tt-metalium/circular_buffer_constants.h`

Signatures:
- `constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 32;`
- `constexpr static std::uint32_t NUM_CIRCULAR_BUFFERS = 64;`
- `constexpr static std::uint32_t UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4;`
- `constexpr static std::uint32_t UINT32_WORDS_PER_REMOTE_CIRCULAR_BUFFER_CONFIG = 2;`
- `constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_WORD_SIZE = 16;`
- `constexpr static std::uint32_t CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT = 4;`

### `include/jit_hw/tt-metalium/constants.hpp`

Signatures:
- `namespace tt::constants`
- `using std::uint32_t;`
- `constexpr uint32_t TILE_HEIGHT = 32;`
- `constexpr uint32_t TILE_WIDTH = 32;`
- `constexpr uint32_t TILE_HW = TILE_WIDTH * TILE_HEIGHT;`
- `constexpr uint32_t FACE_HEIGHT = 16;`
- `constexpr uint32_t FACE_WIDTH = 16;`
- `constexpr uint32_t FACE_HW = FACE_WIDTH * FACE_HEIGHT;`
- `constexpr uint32_t BFLOAT8_B_TILE_HW = TILE_HW + 64;`
- `constexpr uint32_t BFLOAT4_B_TILE_HW = (TILE_HW / 2) + 64;`

### `include/jit_hw/tt-metalium/tt_backend_api_types.hpp`

Signatures:
- `namespace tt`
- `enum class DataFormat`

### `include/tt_emule/buffer.hpp`

Signatures:
- `namespace tt_emule`
- `class Device`
- `enum class BufferType`
- `class Buffer`
- `Buffer(Device* device, size_t size_bytes, uint32_t page_size_bytes, uint64_t offset, BufferType type)`
- `size_t size() const`
- `uint32_t page_size() const`
- `uint64_t dram_offset() const`
- `uint32_t address() const`
- `BufferType buffer_type() const`
- `Device* device() const`
- `Device* device_;`
- `size_t size_bytes_;`
- `uint32_t page_size_bytes_;`
- `uint64_t offset_;`
- `BufferType type_;`

### `include/tt_emule/cb_sync_state.hpp`

Signatures:
- `namespace tt_emule`
- `struct CBSyncState`
- `uint8_t* base = nullptr;`
- `uint32_t page_size = 0;`
- `uint32_t num_pages = 0;`
- `uint32_t page_mask = 0;`
- `uint32_t write_idx = 0;`
- `uint32_t read_idx = 0;`
- `std::mutex mu;`
- `std::condition_variable space_cv;`
- `std::condition_variable data_cv;`
- `inline void cb_sync_reserve(CBSyncState& cb, uint32_t n)`
- `inline void cb_sync_push(CBSyncState& cb, uint32_t n)`
- `inline void cb_sync_wait(CBSyncState& cb, uint32_t n)`
- `inline void cb_sync_pop(CBSyncState& cb, uint32_t n)`
- `inline uint8_t* cb_sync_write_ptr(CBSyncState& cb)`
- `inline uint8_t* cb_sync_read_ptr(CBSyncState& cb)`
- `inline uint8_t* cb_sync_write_ptr_at(CBSyncState& cb, uint32_t offset)`
- `inline const uint8_t* cb_sync_read_ptr_at(const CBSyncState& cb, uint32_t offset)`

### `include/tt_emule/circular_buffer.hpp`

Signatures:
- `namespace tt_emule`
- `class CircularBuffer`
- `static constexpr uint32_t DEFAULT_PAGE_SIZE = 32 * 32 * sizeof(float);`
- `explicit CircularBuffer(size_t capacity, uint32_t page_size = DEFAULT_PAGE_SIZE)`
- `size_t capacity() const`
- `uint32_t page_size() const`
- `CBSyncState& sync_state()`
- `const CBSyncState& sync_state() const`
- `void reserve_back(size_t n)`
- `void push_back(size_t n)`
- `void wait_front(size_t n)`
- `void pop_front(size_t n)`
- `uint8_t* get_write_ptr()`
- `const uint8_t* get_read_ptr() const`
- `uint8_t* get_read_ptr_mut()`
- `uint8_t* get_write_ptr_at(size_t offset)`
- `const uint8_t* get_read_ptr_at(size_t offset) const`
- `CBSyncState state_;`
- `std::vector<uint8_t> storage_;`

### `include/tt_emule/dataflow_buffer.hpp`

Signatures:
- `namespace tt_emule`
- `class DataflowBuffer`
- `DataflowBuffer(EmuleDFBInterface& iface, TileCounterArray& tc_array, uint16_t id)`
- `void reserve_back(uint16_t n)`
- `void push_back(uint16_t n)`
- `void wait_front(uint16_t n)`
- `void pop_front(uint16_t n)`
- `void finish()`
- `uint32_t get_write_ptr() const`
- `uint32_t get_read_ptr() const`
- `uint32_t get_entry_size() const`
- `uint32_t get_stride_size() const`
- `uint16_t get_id() const`
- `uint32_t advance_ptr(uint32_t ptr, uint32_t base, uint32_t limit, uint16_t n) const`
- `EmuleDFBInterface& iface_;`
- `TileCounterArray& tc_array_;`
- `uint16_t id_;`

### `include/tt_emule/device.hpp`

Signatures:
- `namespace tt_emule`
- `enum class HalMemType`
- `enum class CoreRole`
- `using CoreCoord = tt_xy_pair;`
- `struct CoreCoord`
- `size_t x;`
- `size_t y;`
- `bool operator==(const CoreCoord& o) const`
- `std::string str() const`
- `class Core`
- `static constexpr size_t L1_SIZE = 1024 * 1024;`
- `static constexpr size_t MAX_CBS = 32;`
- `explicit Core(CoreCoord coord)`
- `Core(CoreCoord coord, CoreRole role, size_t mem_size)`
- `Core(CoreCoord coord, uint8_t* external_l1, size_t l1_size)`
- `~Core()`
- `Core(const Core&) = delete`
- `Core& operator=(const Core&) = delete;`
- `CoreCoord coord() const`
- `CoreRole role() const`
- `std::shared_ptr<CircularBuffer>& cb(size_t idx)`
- `DstRegisterFile& dst()`
- `uint8_t* l1_ptr(uint32_t offset)`
- `uint8_t* l1_data()`
- `size_t l1_size() const`
- `uint32_t l1_base_addr() const`
- `static constexpr size_t L1_RESERVED_TOP = 512;`
- `uint32_t l1_alloc(size_t bytes)`
- `void reset_l1_bump()`
- `CBSyncState* cb_sync_array()`
- `void init_cb_sync(uint32_t idx, uint8_t* base, uint32_t page_size, uint32_t num_pages)`
- `void reset_cb_sync()`
- `void init_tile_counters(uint32_t num_neos)`
- `TileCounterArray* tile_counters()`
- `DFBSyncState* dfb_sync_array()`
- `void init_dfb_sync(uint32_t idx, uint8_t* base, uint32_t entry_size, uint32_t num_entries, uint32_t capacity)`
- `void reset_dfb_sync()`
- `void mmap_region(size_t size)`
- `CoreCoord coord_;`
- `CoreRole role_ = CoreRole::WORKER;`
- `bool owns_l1_ = true;`
- `size_t l1_size_ = L1_SIZE;`
- `uint8_t* l1_ = nullptr;`
- `uint32_t l1_base_ = 0;`
- `size_t l1_bump_ = 0;`
- `std::array<std::shared_ptr<CircularBuffer>, MAX_CBS> cbs_;`
- `DstRegisterFile dst_;`
- `std::unique_ptr<TileCounterArray> tile_counters_;`
- `enum class BufferType`
- `class MockAllocator`
- `uint32_t l1_base_;`
- `size_t l1_size_;`
- `explicit MockAllocator(uint32_t base, size_t l1_size = Core::L1_SIZE)`
- `uint32_t get_base_allocator_addr(HalMemType type) const`
- `uint32_t get_dram_channel_from_bank_id(uint32_t ) const`
- `CoreCoord get_logical_core_from_bank_id(uint32_t ) const`
- `std::vector<uint32_t> get_bank_ids_from_logical_core(BufferType , CoreCoord ) const`
- `size_t get_bank_size(BufferType ) const`
- `class Device`
- `static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024;`
- `Device()`
- `~Device() override`
- `uint32_t l1_alloc(size_t bytes)`
- `uint64_t dram_alloc(size_t bytes)`
- `uint8_t* dram_ptr(uint64_t offset)`
- `uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr)`
- `Core& core()`
- `MockAllocator* mock_allocator()`
- `tt::ARCH arch() const override`
- `tt::tt_metal::ChipId id() const override`
- `tt::tt_metal::ChipId build_id() const override`
- `uint8_t num_hw_cqs() const override`
- `bool is_initialized() const override`
- `int num_dram_channels() const override`
- `uint32_t l1_size_per_core() const override`
- `uint32_t dram_size_per_channel() const override`
- `int get_clock_rate_mhz() const override`
- `tt::tt_metal::CoreCoord grid_size() const override`
- `tt::tt_metal::CoreCoord logical_grid_size() const override`
- `tt::tt_metal::CoreCoord dram_grid_size() const override`
- `tt::tt_metal::CoreCoord virtual_noc0_coordinate( uint8_t , tt::tt_metal::CoreCoord coord) const override`
- `std::vector<tt::tt_metal::CoreCoord> worker_cores_from_logical_cores( const std::vector<tt::tt_metal::CoreCoord>& v) const override`
- `std::vector<tt::tt_metal::CoreCoord> get_optimal_dram_bank_to_logical_worker_assignment( tt::tt_metal::NOC ) override`
- `tt::tt_metal::CoreCoord virtual_core_from_logical_core( const tt::tt_metal::CoreCoord& c, const tt::CoreType& ) const override`
- `tt::tt_metal::CoreCoord worker_core_from_logical_core( const tt::tt_metal::CoreCoord& c) const override`
- `tt::tt_metal::CoreCoord compute_with_storage_grid_size() const override`
- `tt::tt_metal::CoreRangeSet worker_cores( tt::tt_metal::HalProgrammableCoreType core_type, tt::tt_metal::SubDeviceId sub_device_id) const override`
- `uint32_t num_worker_cores( tt::tt_metal::HalProgrammableCoreType , tt::tt_metal::SubDeviceId ) const override`
- `const std::unique_ptr<tt::tt_metal::Allocator>& allocator() const override`
- `const std::unique_ptr<tt::tt_metal::Allocator>& allocator( tt::tt_metal::SubDeviceId ) const override`
- `const std::unique_ptr<tt::tt_metal::AllocatorImpl>& allocator_impl() const override`
- `const std::unique_ptr<tt::tt_metal::AllocatorImpl>& allocator_impl( tt::tt_metal::SubDeviceId ) const override`
- `tt::tt_metal::CoreCoord logical_core_from_dram_channel(uint32_t ) const override`
- `uint32_t dram_channel_from_logical_core( const tt::tt_metal::CoreCoord& ) const override`
- `uint32_t dram_channel_from_virtual_core( const tt::tt_metal::CoreCoord& ) const override`
- `std::optional<tt::tt_metal::DeviceAddr> lowest_occupied_compute_l1_address() const override`
- `std::optional<tt::tt_metal::DeviceAddr> lowest_occupied_compute_l1_address( tt::stl::Span<const tt::tt_metal::SubDeviceId> ) const override`
- `const std::set<tt::tt_metal::CoreCoord>& storage_only_cores() const override`
- `uint32_t get_noc_unicast_encoding( uint8_t , const tt::tt_metal::CoreCoord& ) const override`
- `uint32_t get_noc_multicast_encoding( uint8_t , const tt::tt_metal::CoreRange& ) const override`
- `tt::tt_metal::SystemMemoryManager& sysmem_manager() override`
- `uint32_t get_trace_buffers_size() const override`
- `void set_trace_buffers_size(uint32_t ) override`
- `bool initialize(uint8_t , size_t , size_t , size_t , tt::stl::Span<const std::uint32_t> = {}, bool = false) override`
- `void init_command_queue_host() override`
- `void init_command_queue_device() override`
- `bool compile_fabric() override`
- `void configure_fabric() override`
- `bool close() override`
- `void enable_program_cache() override`
- `void clear_program_cache() override`
- `void disable_and_clear_program_cache() override`
- `tt::tt_metal::program_cache::detail::ProgramCache& get_program_cache() override`
- `std::size_t num_program_cache_entries() override`
- `tt::tt_metal::HalProgrammableCoreType get_programmable_core_type( tt::tt_metal::CoreCoord ) const override`
- `HalMemType get_mem_type_of_core( tt::tt_metal::CoreCoord ) const override`
- `bool has_noc_mcast_txns(tt::tt_metal::SubDeviceId ) const override`
- `uint8_t num_noc_unicast_txns(tt::tt_metal::SubDeviceId ) const override`
- `uint8_t noc_data_start_index( tt::tt_metal::SubDeviceId , bool = true) const override`
- `tt::tt_metal::SubDeviceManagerId get_active_sub_device_manager_id() const override`
- `tt::tt_metal::SubDeviceManagerId get_default_sub_device_manager_id() const override`
- `tt::tt_metal::SubDeviceManagerId create_sub_device_manager( tt::stl::Span<const tt::tt_metal::SubDevice> , tt::tt_metal::DeviceAddr ) override`
- `tt::tt_metal::SubDeviceManagerId create_sub_device_manager( std::initializer_list<tt::tt_metal::SubDevice> , tt::tt_metal::DeviceAddr ) override`
- `void remove_sub_device_manager( tt::tt_metal::SubDeviceManagerId ) override`
- `void load_sub_device_manager( tt::tt_metal::SubDeviceManagerId ) override`
- `void clear_loaded_sub_device_manager() override`
- `tt::tt_metal::CoreCoord virtual_program_dispatch_core(uint8_t ) const override`
- `const std::vector<tt::tt_metal::SubDeviceId>& get_sub_device_ids() const override`
- `const std::vector<tt::tt_metal::SubDeviceId>& get_sub_device_stall_group() const override`
- `void set_sub_device_stall_group( tt::stl::Span<const tt::tt_metal::SubDeviceId> ) override`
- `void reset_sub_device_stall_group() override`
- `uint32_t num_sub_devices() const override`
- `uint32_t num_virtual_eth_cores(tt::tt_metal::SubDeviceId ) override`
- `bool is_mmio_capable() const override`
- `std::shared_ptr<tt::tt_metal::distributed::MeshDevice> get_mesh_device() override`
- `std::vector<tt::tt_metal::CoreCoord> ethernet_cores_from_logical_cores( const std::vector<tt::tt_metal::CoreCoord>& ) const override`
- `tt::tt_metal::CoreCoord logical_core_from_ethernet_core( const tt::tt_metal::CoreCoord& ) const override`
- `tt::tt_metal::CoreCoord ethernet_core_from_logical_core( const tt::tt_metal::CoreCoord& ) const override`
- `std::unordered_set<tt::tt_metal::CoreCoord> get_active_ethernet_cores( bool = false) const override`
- `std::unordered_set<tt::tt_metal::CoreCoord> get_inactive_ethernet_cores() const override`
- `bool is_active_ethernet_core(tt::tt_metal::CoreCoord , bool = false) const override`
- `bool is_inactive_ethernet_core(tt::tt_metal::CoreCoord ) const override`
- `std::tuple<tt::tt_metal::ChipId, tt::tt_metal::CoreCoord> get_connected_ethernet_core(tt::tt_metal::CoreCoord ) const override`
- `std::vector<tt::tt_metal::CoreCoord> get_ethernet_sockets( tt::tt_metal::ChipId ) const override`
- `const std::set<tt::tt_metal::CoreCoord>& ethernet_cores() const override`
- `std::vector<uint8_t> dram_;`
- `uint64_t dram_bump_;`
- `Core core_;`
- `MockAllocator alloc_;`
- `std::unique_ptr<tt::tt_metal::Allocator> allocator_;`
- `std::unique_ptr<tt::tt_metal::AllocatorImpl> allocator_impl_;`
- `std::vector<tt::tt_metal::SubDeviceId> sub_device_ids_;`
- `std::vector<tt::tt_metal::SubDeviceId> sub_device_stall_group_;`
- `std::set<tt::tt_metal::CoreCoord> storage_only_cores_;`
- `std::set<tt::tt_metal::CoreCoord> ethernet_cores_;`
- `class Device`
- `static constexpr size_t DRAM_SIZE = 256 * 1024 * 1024;`
- `Device()`
- `Device(uint8_t* external_dram, size_t dram_size, uint8_t* external_l1, size_t l1_size)`
- `uint32_t l1_alloc(size_t bytes)`
- `uint64_t dram_alloc(size_t bytes)`
- `uint8_t* dram_ptr(uint64_t offset)`
- `uint8_t* noc_resolve(uint32_t x, uint32_t y, uint64_t addr)`
- `Core& core()`
- `MockAllocator* allocator()`
- `MockAllocator* allocator_impl()`
- `size_t dram_size_per_channel() const`
- `size_t l1_size_per_core() const`
- `CoreCoord worker_core_from_logical_core(CoreCoord c) const`
- `std::vector<uint8_t> dram_;`
- `uint8_t* ext_dram_ = nullptr;`
- `size_t ext_dram_size_ = 0;`
- `uint64_t dram_bump_;`
- `Core core_;`
- `MockAllocator alloc_;`

### `include/tt_emule/dfb_sync_state.hpp`

Signatures:
- `namespace tt_emule`
- `static constexpr uint32_t MAX_DFBS = 32;`
- `static constexpr uint32_t MAX_TC_SLOTS_PER_DFB = 4;`
- `static_assert(MAX_DFBS <= 4 * (TILE_COUNTERS_PER_NEO / MAX_TC_SLOTS_PER_DFB), "MAX_DFBS exceeds total available tile counter slots across 4 NEOs")`
- `struct EmuleDFBInterface`
- `uint8_t num_tcs_to_rr = 0;`
- `uint8_t tc_idx = 0;`
- `uint32_t entry_size = 0;`
- `uint32_t stride_size = 0;`
- `uint32_t rd_entry_idx = 0;`
- `uint32_t wr_entry_idx = 0;`
- `uint32_t num_entries = 0;`
- `bool broadcast_tc = false;`
- `bool drain_per_tc = false;`
- `bool active = false;`
- `struct DFBSyncState`
- `uint8_t* base = nullptr;`
- `uint32_t entry_size = 0;`
- `uint32_t num_entries = 0;`
- `uint32_t capacity = 0;`
- `uint32_t stride_in_entries = 1;`

### `include/tt_emule/dst_register_file.hpp`

Signatures:
- `namespace tt_emule`
- `class DstRegisterFile`
- `static constexpr size_t TOTAL_SLOTS = 16;`
- `static constexpr size_t BF16_SLOTS = 16;`
- `static constexpr size_t FP32_SLOTS = 8;`
- `enum class State`
- `DstRegisterFile()`
- `void set_fp32_mode(bool fp32)`
- `bool fp32_mode() const`
- `size_t active_slots() const`
- `void acquire()`
- `void commit()`
- `void wait()`
- `void release()`
- `Tile& operator[](size_t idx)`
- `const Tile& operator[](size_t idx) const`
- `State state() const`
- `std::array<Tile, TOTAL_SLOTS> slots_;`
- `State state_;`
- `bool fp32_mode_;`
- `mutable std::mutex mu_;`
- `std::condition_variable cv_;`

### `include/tt_emule/host_api.hpp`

Signatures:
- `namespace tt_emule`
- `using KernelHandle = uint32_t;`
- `struct CommandQueue`
- `Device* device;`
- `Device* CreateDevice(uint32_t id = 0)`
- `bool CloseDevice(Device* device)`
- `CommandQueue CreateCommandQueue(Device& device)`
- `Program CreateProgram()`
- `void SetRuntimeArgs(Program& program, KernelHandle kernel_id, CoreCoord core, std::vector<uint32_t> args)`
- `CBHandle CreateCircularBuffer(Program& program, CoreCoord core, CircularBufferConfig config)`
- `DFBHandle CreateDataflowBuffer(Program& program, DataflowBufferConfig config)`
- `std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes, uint32_t page_size_bytes, BufferType type = BufferType::DRAM)`
- `void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src, bool blocking = true)`
- `void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst, bool blocking = true)`
- `void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buf, const void* src, bool blocking = true)`
- `void EnqueueReadBuffer(CommandQueue& cq, Buffer& buf, void* dst, bool blocking = true)`
- `void EnqueueProgram(Device& device, Program& program, bool blocking = true)`
- `void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking = true)`
- `void Finish(Device& device)`
- `void Finish(CommandQueue& cq)`
- `namespace detail`
- `void LaunchProgram(Device* device, Program& program, bool wait = true, bool force_slow = false)`
- `bool ReadFromDeviceL1(Device* device, const CoreCoord& core, uint32_t address, uint32_t size, std::vector<uint32_t>& result)`
- `void WriteToDeviceL1(Device* device, const CoreCoord& core, uint32_t address, const std::vector<uint32_t>& data)`
- `void WriteToDeviceL1(Device* device, const CoreCoord& core, uint32_t address, std::span<const uint8_t> data)`
- `void ReadFromDeviceL1(Device* device, const CoreCoord& core, uint32_t address, std::span<uint8_t> data)`
- `void WriteToDeviceDRAMChannel(Device* device, uint32_t channel, uint32_t address, const std::vector<uint32_t>& data)`
- `void WriteToDeviceDRAMChannel(Device* device, uint32_t channel, uint32_t address, std::span<const uint8_t> data)`
- `void ReadFromDeviceDRAMChannel(Device* device, uint32_t channel, uint32_t address, uint32_t byte_size, std::vector<uint32_t>& result)`
- `void ReadFromDeviceDRAMChannel(Device* device, uint32_t channel, uint32_t address, std::span<uint8_t> data)`
- `template<typename T> inline void WriteToBuffer(const std::shared_ptr<Buffer>& buf, const std::vector<T>& data)`
- `template<typename T> inline void ReadFromBuffer(const std::shared_ptr<Buffer>& buf, std::vector<T>& data)`
- `namespace tt_emule_internal`
- `tt_emule::KernelHandle create_jit_kernel( tt_emule::Program& program, const std::string& kernel_src_path, tt_emule::CoreCoord core, tt_emule::DataMovementConfig config)`

### `include/tt_emule/jit_kernel.hpp`

Signatures:
- `namespace tt_emule`
- `KernelFn jit_compile_kernel(const std::string& kernel_src_path, const std::vector<uint32_t>& compile_args, const std::string& jit_include_dir)`

### `include/tt_emule/l1_pool.hpp`

Signatures:
- `namespace tt_emule`
- `class L1Pool`
- `static constexpr size_t SLOT_SIZE = 2 * 1024 * 1024;`
- `static constexpr size_t SLOT_MASK = SLOT_SIZE - 1;`
- `explicit L1Pool(size_t num_slots)`
- `~L1Pool()`
- `L1Pool(const L1Pool&) = delete`
- `L1Pool& operator=(const L1Pool&) = delete;`
- `uint8_t* slot_ptr(size_t index) const`
- `size_t num_slots() const`
- `static uint32_t to_offset(uint32_t addr)`
- `size_t num_slots_ = 0;`
- `uint8_t* raw_ = nullptr;`
- `size_t raw_size_ = 0;`
- `uint8_t* base_ = nullptr;`

### `include/tt_emule/program.hpp`

Signatures:
- `namespace tt_emule`
- `using KernelFn = std::function<void()>;`
- `enum class KernelType`
- `enum class DataMovementProcessor`
- `enum class NOC`
- `enum class NOC_MODE`
- `enum class MathFidelity`
- `enum class UnpackToDestMode`
- `enum class AccessPattern`
- `struct DataMovementConfig`
- `KernelType type = KernelType::DataMovement0;`
- `DataMovementProcessor processor = DataMovementProcessor::RISCV_0;`
- `NOC noc = NOC::RISCV_0_default;`
- `NOC_MODE noc_mode = NOC_MODE::DM_DEDICATED_NOC;`
- `std::vector<uint32_t> compile_args;`
- `std::map<std::string, std::string> defines;`
- `std::unordered_map<std::string, uint32_t> named_compile_args;`
- `std::string kernel_src_path;`
- `struct ComputeConfig`
- `MathFidelity math_fidelity = MathFidelity::HiFi4;`
- `bool fp32_dest_acc_en = false;`
- `bool dst_full_sync_en = false;`
- `std::vector<UnpackToDestMode> unpack_to_dest_mode;`
- `bool bfp8_pack_precise = false;`
- `bool math_approx_mode = false;`
- `std::vector<uint32_t> compile_args;`
- `std::map<std::string, std::string> defines;`
- `std::unordered_map<std::string, uint32_t> named_compile_args;`
- `struct QuasarDataMovementConfig`
- `uint32_t num_threads_per_cluster = 8;`
- `std::vector<uint32_t> compile_args;`
- `std::map<std::string, std::string> defines;`
- `std::unordered_map<std::string, uint32_t> named_compile_args;`
- `bool is_legacy_kernel = false;`
- `struct QuasarComputeConfig`
- `uint32_t num_threads_per_cluster = 4;`
- `MathFidelity math_fidelity = MathFidelity::HiFi4;`
- `bool fp32_dest_acc_en = false;`
- `bool dst_full_sync_en = false;`
- `std::vector<UnpackToDestMode> unpack_to_dest_mode;`
- `bool bfp8_pack_precise = false;`
- `bool math_approx_mode = false;`
- `std::vector<uint32_t> compile_args;`
- `std::map<std::string, std::string> defines;`
- `std::unordered_map<std::string, uint32_t> named_compile_args;`
- `struct KernelDescriptor`
- `uint32_t id;`
- `KernelType type;`
- `KernelFn fn;`
- `CoreCoord core;`
- `std::vector<uint32_t> rt_args;`
- `uint8_t processor_id = 0;`
- `using CBHandle = uint32_t;`
- `using DFBHandle = uint32_t;`
- `struct DataflowBufferConfig`
- `uint32_t dfb_index = 0;`
- `uint32_t entry_size = 0;`
- `uint32_t num_entries = 0;`
- `uint16_t producer_risc_mask = 0x0;`
- `uint8_t num_producers = 1;`
- `AccessPattern producer_access_pattern = AccessPattern::STRIDED;`
- `uint16_t consumer_risc_mask = 0x0;`
- `uint8_t num_consumers = 1;`
- `AccessPattern consumer_access_pattern = AccessPattern::STRIDED;`
- `struct CircularBufferConfig`
- `uint32_t cb_index = 0;`
- `size_t num_pages = 0;`
- `uint32_t page_size = 0;`
- `CircularBufferConfig() = default`
- `CircularBufferConfig(uint32_t idx, size_t pages, uint32_t psize)`
- `template <typename K, typename V> CircularBufferConfig(size_t total_size, std::map<K, V> index_format_map)`
- `template <typename K, typename V> CircularBufferConfig(size_t total_size, std::initializer_list<std::pair<const K, V>> spec)`
- `CircularBufferConfig& set_page_size(uint32_t , uint32_t psize)`
- `class Program`
- `uint32_t add_kernel(KernelType type, KernelFn fn, CoreCoord core, uint8_t processor_id = 0)`
- `void set_runtime_args(uint32_t kernel_id, std::vector<uint32_t> args)`
- `CBHandle add_cb(CircularBufferConfig cfg)`
- `DFBHandle add_dfb(DataflowBufferConfig cfg)`
- `std::vector<KernelDescriptor>& kernels()`
- `std::vector<CircularBufferConfig>& cb_configs()`
- `std::vector<DataflowBufferConfig>& dfb_configs()`
- `bool has_dfbs() const`
- `std::vector<KernelDescriptor> kernels_;`
- `std::vector<CircularBufferConfig> cb_configs_;`
- `std::vector<DataflowBufferConfig> dfb_configs_;`

### `include/tt_emule/tile.hpp`

Signatures:
- `namespace tt_emule`
- `class Tile`
- `static constexpr size_t ROWS = 32;`
- `static constexpr size_t COLS = 32;`
- `static constexpr size_t NUM_ELEMENTS = ROWS * COLS;`
- `static constexpr size_t SIZE_BYTES = NUM_ELEMENTS * sizeof(float);`
- `Tile()`
- `explicit Tile(float val)`
- `float& operator()`
- `const float& operator()`
- `Tile operator+(const Tile& other) const`
- `Tile& operator+=(const Tile& other)`
- `static constexpr size_t size_bytes()`
- `uint8_t* bytes()`
- `const uint8_t* bytes() const`
- `std::array<float, NUM_ELEMENTS>& raw()`
- `const std::array<float, NUM_ELEMENTS>& raw() const`
- `std::array<float, NUM_ELEMENTS> data_;`

### `include/tt_emule/tile_counter.hpp`

Signatures:
- `namespace tt_emule`
- `static constexpr uint32_t TILE_COUNTERS_PER_NEO = 32;`
- `struct TileCounter`
- `std::mutex mu;`
- `std::condition_variable space_cv;`
- `std::condition_variable data_cv;`
- `uint32_t occupancy() const`
- `uint32_t free_space() const`
- `void reset()`
- `class TileCounterArray`
- `explicit TileCounterArray(uint32_t num_neos)`
- `TileCounter& get(uint8_t neo_id, uint8_t counter_id)`
- `uint32_t num_neos() const`
- `void inc_posted(uint8_t neo_id, uint8_t counter_id, uint32_t n)`
- `void inc_acked(uint8_t neo_id, uint8_t counter_id, uint32_t n)`
- `void wait_free_space(uint8_t neo_id, uint8_t counter_id, uint32_t n)`
- `void wait_occupancy(uint8_t neo_id, uint8_t counter_id, uint32_t n)`
- `void reset_all()`
- `uint32_t num_neos_;`

### `run_d2m_regression.sh`

Signatures: none

### `run_regression.sh`

Signatures:
- `_gtest_xml_args()`
- `run_test()`
- `run_test_verbose()`

### `run_toggle_test.sh`

Signatures:
- `run_test()`

### `src/host_api.cpp`

Signatures:
- `namespace tt_emule`
- `Device::Device()`
- `Device::~Device() = default`
- `tt::tt_metal::CoreRangeSet Device::worker_cores( tt::tt_metal::HalProgrammableCoreType , tt::tt_metal::SubDeviceId ) const`
- `Device* CreateDevice(uint32_t )`
- `bool CloseDevice(Device* device)`
- `Program CreateProgram()`
- `void SetRuntimeArgs(Program& program, uint32_t kernel_id, CoreCoord , std::vector<uint32_t> args)`
- `CBHandle CreateCircularBuffer(Program& program, CoreCoord , CircularBufferConfig config)`
- `DFBHandle CreateDataflowBuffer(Program& program, DataflowBufferConfig config)`
- `std::shared_ptr<Buffer> CreateBuffer(Device& device, size_t size_bytes, uint32_t page_size_bytes, BufferType type)`
- `void EnqueueWriteBuffer(Device& device, Buffer& buf, const void* src, bool )`
- `void EnqueueReadBuffer(Device& device, Buffer& buf, void* dst, bool )`
- `void Finish(Device& )`
- `CommandQueue CreateCommandQueue(Device& device)`
- `void EnqueueWriteBuffer(CommandQueue& cq, Buffer& buf, const void* src, bool blocking)`
- `void EnqueueReadBuffer(CommandQueue& cq, Buffer& buf, void* dst, bool blocking)`
- `void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking)`
- `void Finish(CommandQueue& cq)`
- `namespace detail`
- `void LaunchProgram(Device* device, Program& program, bool , bool )`
- `bool ReadFromDeviceL1(Device* device, const CoreCoord& , uint32_t address, uint32_t size, std::vector<uint32_t>& result)`
- `void WriteToDeviceL1(Device* device, const CoreCoord& , uint32_t address, const std::vector<uint32_t>& data)`
- `void WriteToDeviceL1(Device* device, const CoreCoord& , uint32_t address, std::span<const uint8_t> data)`
- `void ReadFromDeviceL1(Device* device, const CoreCoord& , uint32_t address, std::span<uint8_t> data)`
- `void WriteToDeviceDRAMChannel(Device* device, uint32_t , uint32_t address, const std::vector<uint32_t>& data)`
- `void WriteToDeviceDRAMChannel(Device* device, uint32_t , uint32_t address, std::span<const uint8_t> data)`
- `void ReadFromDeviceDRAMChannel(Device* device, uint32_t , uint32_t address, uint32_t byte_size, std::vector<uint32_t>& result)`
- `void ReadFromDeviceDRAMChannel(Device* device, uint32_t , uint32_t address, std::span<uint8_t> data)`
- `namespace tt_emule_internal`
- `tt_emule::KernelHandle create_jit_kernel( tt_emule::Program& program, const std::string& kernel_src_path, tt_emule::CoreCoord core, tt_emule::DataMovementConfig config)`

### `src/jit_kernel.cpp`

Signatures:
- `namespace tt_emule`
- `static std::string jit_include_dir_from_cmake()`
- `KernelFn jit_compile_kernel(const std::string& kernel_src_path, const std::vector<uint32_t>& compile_args, const std::string& jit_include_dir_override)`

### `src/kernel_runner.cpp`

Signatures:
- `thread_local std::vector<uint32_t> __rt_args;`
- `thread_local tt_emule::Core* __core = nullptr;`
- `thread_local tt_emule::Device* __device = nullptr;`
- `thread_local uint8_t __processor_id = 0;`
- `thread_local tt_emule::TileCounterArray* __emule_tc_array = nullptr;`
- `thread_local tt_emule::EmuleDFBInterface* __emule_dfbs = nullptr;`
- `extern "C" uint8_t* __emule_dram_ptr(uint64_t offset)`
- `namespace tt_emule`
- `namespace (anonymous)`
- `std::vector<std::vector<EmuleDFBInterface>> build_dfb_interfaces( Core& core, Program& program)`
- `void EnqueueProgram(Device& device, Program& program, bool )`

### `tests/CMakeLists.txt`

Signatures:
- `add_subdirectory(tilize)`

### `tests/integration/test_emulation_toggle.cpp`

Signatures:
- `using namespace tt::tt_metal;`
- `TEST(EmulationToggle, BuildFlagCompiled)`
- `TEST(EmulationToggle, DefaultIsNotEmulated)`
- `TEST(SiliconActive, TargetDeviceIsSilicon)`
- `TEST(SiliconActive, IsNotMockOrEmulated)`
- `TEST(EmulationActive, TargetDeviceIsEmulated)`
- `TEST(EmulationActive, SlowDispatchForced)`
- `TEST(EmulationActive, IsMockOrEmulated)`
- `TEST_F(MeshDeviceFixture, EmulatedRunnerInvoked)`

### `tests/tilize/CMakeLists.txt`

Signatures:
- `add_executable(test_tilize ...)`
- `target_include_directories(test_tilize)`
- `target_link_libraries(test_tilize)`
- `add_test(NAME ...)`

### `tests/tilize/test_tilize.cpp`

Signatures:
- `using namespace tt_emule;`
- `static constexpr uint32_t NUM_TILES = 4;`
- `int main()`
