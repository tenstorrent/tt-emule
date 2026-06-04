# Batch 1 pre-fix baseline — missing shims

Captured before applying the Batch 1 fix to `include/jit_hw/api/compute/{untilize,tilize,common}.h`.

## Test outcomes on emule pre-Batch-1

### ssm_prefix_scan — JIT-fail confirmed

```
/tmp/tt_emule_jit_*/patched_kernel.cpp:29:5: error: use of undeclared identifier 'untilize_uninit'
   29 |     untilize_uninit(cb_in);
RuntimeError: jit_compile_kernel: compiler failed (exit 256) for kernel:
  /localdev/arminale/tt-metal/ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp
```

After Batch 1 (untilize_uninit shim added), ssm advances past the JIT
phase but the test still fails on PCC (≈ 0.003) because the kernel
exercises Bfp8_b-formatted CBs that hit the
`__llk_pack_untilize` `elem_size = page_size / 1024` bug from
Tier 1 #2. That bug is STRUCTURAL (not Batch 1 scope) — track separately.

### rotary_embedding_llama — pre-test fixture failure

`tests/ttnn/nightly/.../test_rotary_embedding_llama.py` imports from
`models/demos/t3000/llama2_70b/reference/llama` and `tt_transformers`.
Those imports succeed on the dev machine but the test fixtures need
model files. Not a clean addition to the emule pytest regression as-is.
Document and skip CI addition for Batch 1.

### conv3d (test_conv3d_no_config) — pre-Batch-1 failure mode is independent

```
/localdev/arminale/tt-emule/include/jit_hw/experimental/noc.h:36:19:
  error: static assertion failed: 'NoC transactions are not supported for this type'
/localdev/arminale/tt-metal/ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp:86:18:
  error: 'async_read_barrier' following the 'template' keyword does not refer to a template
```

conv3d's compute kernel pulls in pool/experimental_device_api.hpp
which needs additional NoC/async_read shims emule doesn't have.
Out of Batch 1 scope. Track as a separate gap (file ticket).

### group_attn_matmul — pre-Batch-1 failure mode is independent

```
/tmp/tt_emule_jit_*/patched_kernel.cpp:88:22: error: use of undeclared identifier 'VALID'
/tmp/tt_emule_jit_*/patched_kernel.cpp:312:50: error: use of undeclared identifier 'INVALID'
/tmp/tt_emule_jit_*/patched_kernel.cpp:320:51: error: use of undeclared identifier 'VALID'
RuntimeError: jit_compile_kernel: compiler failed (exit 256) for kernel:
  /localdev/arminale/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp
```

`VALID` / `INVALID` are silicon `tt_metal::DataValidIndicator`-style
enum members emule doesn't expose. Out of Batch 1 scope.

## Net for Batch 1's CI additions

None of the four candidate tests pass after just the Batch 1 shim
adds — each has additional unrelated blockers (Bfp8_b decode,
NoC/async_read, VALID/INVALID enums, model-file imports). Batch 1
lands the audit-correct fix to `untilize_uninit` / `tilize_uninit` /
templated `untilize_block` / public pack-side forwarders, but does
NOT add any new CI entries.

The Batch 1 fix is still valuable on its own:
- Tier 1 #7 (untilize_uninit + state asymmetry) is the canonical
  example of an active-caller gap; landing the shim makes the
  state-symmetric across tilize/untilize regardless of whether the
  downstream test is in CI.
- Tier 1 #8 (templated `untilize_block<>` mis-stride) is fixed by
  forwarding to the runtime overload.
- The public-name `pack_*` forwarders are cheap and remove a class of
  "unresolved symbol" failures for handwritten kernels.

Follow-up tickets to file after Batch 1 lands:
- ssm_prefix_scan PCC: Bfp8_b path through `__llk_pack_untilize` —
  blocks on Tier 1 #2 STRUCTURAL fix.
- rotary_embedding_llama: needs model-file fixture infrastructure to
  be a clean CI addition.
- conv3d: needs NoC/async_read shims (pool/experimental_device_api.hpp).
- group_attn_matmul: needs VALID/INVALID enum exposed to JIT.
