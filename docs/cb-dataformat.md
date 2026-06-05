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

`255` (`tt::DataFormat::Invalid`) marks unconfigured CB slots — mirroring the host's
`std::optional<DataFormat>` empty state. For an `Invalid` slot (e.g. DFB bridges, or the
host-only tilize unit test that does not emit the define) emule falls back to the legacy
`page_size` heuristic, so nothing regresses.

## Why a format (not just page_size) is needed

emule's pack/unpack in `include/jit_hw/api/compute/common.h` historically inferred format
from `page_size` alone:

- `page_size > 2048` → 32-bit (Float32 / Int32 / UInt32)
- `0 < page_size < 2048` → `Bfp8_b` (1088 B)
- `page_size == 2048` → 16-bit

The `page_size == 2048` bucket is **ambiguous**: `bf16` and `UInt16` are both 2048 B. With
only page_size, a `UInt16` comparison result (integer `0`/`1`) was run through
`bf16::from_f32` and corrupted (issue #75: comparison ops with `uint16` output). The real
per-CB format resolves exactly that ambiguity:

- `pack_dst_to_buf(buf, dst, ocb)`: if `cb_is_uint16_format(ocb)` → write the low 16 bits of
  the DST int32 bit pattern (no float conversion); else bf16. (Packer uses the **output** CB
  format — `pack_dst_format`.)
- `__emule_unpack_cb_tile_to(icb, …)`: if `cb_is_uint16_format(icb)` → widen each `uint16` →
  `int32` bit pattern into the fp32 DST slot; else bf16. (Unpacker uses the **input** CB
  format — `unpack_src_format`.)

The unambiguous 32-bit / `Bfp8_b` buckets stay on the reliable `page_size` dispatch.

## Silicon ↔ emule mapping

| | Silicon | emule |
|---|---|---|
| Format source | compile-time `unpack_src_format[]`/`pack_dst_format[]` in `chlkc_descriptors.h` | compile-time arrays in `cb_api.h` from `EMULE_CB_DATA_FORMATS` |
| Where set | host `CircularBufferConfig` → genfiles | host `cb_impl->data_format(idx)` → runner define |
| In L1 CB config? | no | no |
| Unset slot | `std::optional` empty | `255` = `Invalid` (page_size fallback) |

## Verification

`tests/ttnn/unit_tests/operations/eltwise/test_binary_comp_init.py` (comparison + logical ops
with `uint16`/`uint32` output) goes fully green; `test_relational` / `test_binary_comp_fp32` /
`test_unary` comparison and the sharded `col_major` bcast pass, on both WH (N150) and BH
(P100). No regression on the other format-dispatch consumers (matmul / reduce / tilize /
Bfp8) — those remain on the page_size path for the unambiguous buckets.
