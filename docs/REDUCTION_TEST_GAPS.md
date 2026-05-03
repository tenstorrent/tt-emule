# Reduction Test Gap Analysis

Snapshot of which tests in `tests/ttnn/unit_tests/gtests/test_reduction.cpp`
work under tt-emule, captured 2026-05-03 against:

- tt-emule: `armin/quasar-reduction` (commits `a855911`, `849bc16`, `ea8687f`)
- tt-metal-main: `arminale/emule-metal-20` @ `5c6ffaca751`
  (single commit on top of `arminale/emule-metal-base` @ `8711ac3d0ba`)

## Results

3 of 16 tests pass. Pass set is exactly the tile-aligned `SumTensor*` index `/1`.

| Test | Shape | Result | Notes |
|------|-------|--------|-------|
| `SumTensorLastDimFixture/0` | 3100×63 | FAIL | unaligned-W path |
| `SumTensorLastDimFixture/1` | 3200×64 | **PASS** | Tier 5b baseline |
| `SumTensorFirstDimFixture/0` | 63×3100 | FAIL | unaligned-H path |
| `SumTensorFirstDimFixture/1` | 64×3200 | **PASS** | |
| `SumTensorBothDimsFixture/0` | 30×30 | FAIL | unaligned both |
| `SumTensorBothDimsFixture/1` | 64×64 | **PASS** | |
| `MinMaxTensorLastDimFixture/{0,1,2,3}` | 3100×63 / 3200×64, offsets 4 / -128 | FAIL × 4 | |
| `MinMaxTensorFirstDimFixture/{0,1,2,3}` | 63×3100 / 64×3200, offsets 4 / -128 | FAIL × 4 | |
| `MinMaxTensorBothDimsFixture/{0,1}` | 64×64 offset 1 / 30×30 offset -1004 | FAIL × 2 | |

## Failure Causes

Two missing JIT stubs explain every failure.

### Gap 1: `llk_math_eltwise_binary.h`

Required by the `_neg`-suffixed reduce kernels that handle non-tile-aligned
shapes (writers/compute fill the trailing partial tile with the dim's
identity element before reducing):

- `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`
- `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_h_neg.cpp`

Affects all 3 unaligned `Sum*` tests (parameter `/0`).

```text
/tmp/tt_emule_jit_*/patched_kernel.cpp:15:10: fatal error: 'llk_math_eltwise_binary.h' file not found
```

### Gap 2: `tt-metalium/buffer_types.hpp`

Required transitively by `sharded_tensor_addr_gen.hpp`, which is included by
the fill_pad writer kernel that runs as a preamble for every MinMax test:

- `ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp`

Affects all 10 `MinMaxTensor*` tests (every parameter, since the fill_pad
preamble runs unconditionally to pad partial tiles to ±∞).

```text
/localdev/arminale/tt-metal-main/ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp:10:10: fatal error: 'tt-metalium/buffer_types.hpp' file not found
```

## Follow-up Plan

These are missing-JIT-stub gaps, the same shape as the original Phase A work
that bridged the sum-on-W kernel chain. Bridging them follows the same
recipe:

1. Add an emule-side stub at `include/jit_hw/llk/llk_math_eltwise_binary.h`
   covering the templates the `_neg` reduce kernels actually instantiate.
   Inspect upstream `tt_metal/hw/inc/llk/llk_math_eltwise_binary.h` first;
   most LLK init/reconfig calls in this header are no-ops in emule (UNPACK
   and MATH are serial on one host thread).
2. Add a stub at `include/jit_hw/tt-metalium/buffer_types.hpp` covering the
   types `sharded_tensor_addr_gen.hpp` references at JIT scope. Likely a
   small enum + a couple of trivial structs.
3. Re-run `unit_tests_ttnn --gtest_filter='SumTensor*:MinMaxTensor*'` and
   confirm all 16 pass (or document which still don't and why).
4. Update the regression script `run_regression.sh` Tier 5b to add the new
   passing tests.

Estimated cost: ~1 hour per gap based on the Phase A precedent.

Full log captured at `/tmp/reduction_all.log` (run timestamp 2026-05-03 21:43).
