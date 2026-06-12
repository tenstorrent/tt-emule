# Circular-buffer data format in emule

A circular buffer's **data format** (`tt::DataFormat`: `Float32=0`, `Float16_b=5` (bf16),
`Bfp8_b=6`, `Int32=8`, `UInt16=9`, `UInt32=24`, …, `Invalid=255`) tells the packer/unpacker
how to encode/decode each tile in L1. emule must track it per CB so it packs/unpacks the
right bytes — most critically to tell **`UInt16` apart from `bf16`**, which share a tile
page size.

## How silicon does it (compile-time)

On silicon the per-CB data format is **compile-time**, baked into each kernel binary — it is
**not** stored in the L1 CB config:

- Host: `CircularBufferImpl::data_format(buffer_index)` (from the `CircularBufferConfig`).
- `ProgramImpl::set_cb_data_fmt_and_tile` → `tt_hlk_desc::buf_dataformat_arr[cb]`.
- `jit_build/genfiles.cpp::compute_data_formats()` emits, into the per-kernel
  `chlkc_descriptors.h`, the constexpr arrays
  `unpack_src_format[32]` / `unpack_dst_format[32]` / `pack_src_format[32]` /
  `pack_dst_format[32]`.
- Kernel reads `get_dataformat(operand) = unpack_src_format[operand]`
  (`tt_metal/hw/inc/api/dataflow/dataflow_api.h`); `llk_unpack_reconfig_data_format` /
  `llk_pack_reconfig_data_format` index `unpack_src_format[id]` / `pack_dst_format[id]`.

The L1 CB interface struct (`LocalCBInterface`) carries only fifo addresses / page size /
pointers — **never the format**.

## How emule mirrors it

emule reproduces the *same compile-time mechanism* (not a runtime field):

| Silicon | emule |
|---|---|
| `genfiles.cpp::compute_data_formats()` reads `buf_dataformat_arr` | runner `build_kernel_defines` reads `cb_impl->data_format(idx)` |
| bakes `unpack_src_format[]`/`pack_dst_format[]` into `chlkc_descriptors.h` | emits the `EMULE_CB_DATA_FORMATS` JIT define (analog of `EMULE_TILE_SIZES`) |
| kernel: `constexpr unpack_src_format[32]` | `include/jit_hw/api/cb_api.h`: `constexpr uint8_t unpack_src_format[32] = { EMULE_CB_DATA_FORMATS }` (and `pack_dst_format[32]`) |
| `get_dataformat(op) = unpack_src_format[op]` | `cb_api.h::get_dataformat(cb)` returns `unpack_src_format[cb]` |

`255` (`tt::DataFormat::Invalid`) marks unconfigured slots — the host's
`std::optional<DataFormat>` empty state. Every integrated build populates a real format for
each active CB; `Invalid` should never reach a running kernel. DFB-only programs that haven't
threaded format yet (Quasar matmul tests `QuasarMatmulBlock` / `QuasarMatmulBlockInitShort`)
are tracked as known failures until the runner-side DFB format emission lands.

## Never use page_size as a format proxy

`page_size` cannot identify a CB's data format. Every dispatch in
`include/jit_hw/api/compute/common.h` (`cb_is_32bit_format`, `cb_is_bfp8_b_format`,
`cb_is_bfp4_b_format`, `cb_is_uint16_format`) reads `unpack_src_format[cb]` directly — **there
is no `page_size > 2048` (or similar) fallback**. Adding one back would silently corrupt:

- `page_size == 2048` collides on `bf16` *and* `UInt16` — a `UInt16` comparison result
  (integer `0`/`1`) ran through `bf16::from_f32` was the root cause of issue #75 before the
  enum-driven dispatch landed.
- `page_size < 2048` collides on `Bfp8_b` (~1088 B) *and* `Bfp4_b` (~576 B) — both
  block-float with totally different decoders.
- `page_size > 2048` covers `Float32`, `Int32`, `UInt32`, `Tf32`, `RawUInt32`, plus thin-tile
  variants whose page sizes drift below the bucket boundary.

The rule, in one line: **the `EMULE_CB_DATA_FORMATS` enum is the only source of truth.** If
a new format-aware path is needed, extend `cb_is_*_format` predicates against the enum; do
not introduce a page-size shortcut.

Concrete dispatch sites today:

- `pack_dst_to_buf(buf, dst, ocb)`: format read from the **output** CB's `pack_dst_format`.
- `__emule_unpack_cb_tile_to(icb, …)`: format read from the **input** CB's
  `unpack_src_format`.

## Silicon ↔ emule mapping

| | Silicon | emule |
|---|---|---|
| Format source | compile-time `unpack_src_format[]`/`pack_dst_format[]` in `chlkc_descriptors.h` | compile-time arrays in `cb_api.h` from `EMULE_CB_DATA_FORMATS` |
| Where set | host `CircularBufferConfig` → genfiles | host `cb_impl->data_format(idx)` → runner define |
| In L1 CB config? | no | no |
| Unset slot | `std::optional` empty | `255` = `Invalid` (should never appear in integrated builds) |

## Verification

`tests/ttnn/unit_tests/operations/eltwise/test_binary_comp_init.py` (comparison + logical ops
with `uint16`/`uint32` output) goes fully green; `test_relational` / `test_binary_comp_fp32` /
`test_unary` comparison and the sharded `col_major` bcast pass, on both WH (N150) and BH
(P100). No regression on the other format-dispatch consumers (matmul / reduce / tilize /
Bfp8).
