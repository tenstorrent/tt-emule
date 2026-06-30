# Single-chip LLM bring-up (Qwen / DeepSeek / Kimi / MiniMax)

Status: **investigation + current capability map.** The proven ceiling is a
1-layer **prefill** forward pass of Llama-3.2-1B on a single emulated chip
(PR #198). Qwen has not yet been run.

Scope: which transformer-LLM is runnable single-chip under emule, what ops it
needs, what is and isn't supported, and the concrete remaining blockers.

tt-metal file:line references are into the tree at the pinned SHA
(`tt-metal-pin.txt`); emule paths are prefixed `tt-emule/`. SDPA-prefill and the
model bring-up land on `origin/main` (PR #177, PR #198) — a branch checked out
before those merges will not see the SDPA `jit_hw` surface (verify with
`git merge-base --is-ancestor <sha> HEAD`).

---

## 1. Which family is single-chip-relevant

emule emulates **one chip** (N150 / P100): no Ethernet/dispatch fabric across
chips, no multi-host. The decision rule is not "small vs large model" — it is
**does a correct single-chip execution exist?**

| Family | Single-chip? | Reason |
|---|---|---|
| **Qwen** (small, dense) | ✅ | dense transformer with an N150 (1×1) code path; CCL ops are gated behind `is_multichip` and drop out on one chip; small variants fit one chip |
| DeepSeek V3 | ❌ | README target is Galaxy (WH) 2×/4×; its MoE dispatch + tensor-parallel CCL is structural, not config-gated; no 1×1 path |
| Kimi K2.6 | ❌ | not a standalone model — only a config (`KimiK26Config`) feeding the multi-chip `deepseek_v3_d_p` pipeline; no test/demo of its own |
| MiniMax M2 | ❌ | only `minimax_m2_7_config.py`, same DeepSeek-DP pipeline; no test/demo |
| Qwen 32B / 72B | ❌ | dense but exceeds one chip's DRAM |

**Qwen is the only viable target.** Kimi/MiniMax have no single-chip code to run.

## 2. Target model

- **Model:** smallest dense Qwen — `Qwen/Qwen2.5-0.5B-Instruct` (~0.5B params, 24
  decoder layers).
- **Arch:** Wormhole **N150** (emule's default; the Qwen harness names `N150` as
  a mesh option; Blackhole op coverage is at parity but the Qwen harness has no
  `P100` mesh key).
- **Framework:** `models/tt_transformers` (the same path PR #198 validated on
  Llama-3.2-1B). Qwen itself is **untested** — its distinctive paths (Qwen2 QKV
  bias `wqkv_bias`, Qwen3 `q_norm`/`k_norm`, sliding window) are exactly the ones
  Llama does not exercise (see `models/tt_transformers/tt/attention.py`).

The single-chip device bring-up works: the real-model tests run with
`fabric_config=True` + `TT_CCL`, and fabric/collective device init succeeds on a
single emulated chip (PR #198). The CCL *ops* themselves are gated behind
`is_multichip` and do not fire on a 1×1 mesh.

## 3. Prefill vs. decode (the op difference)

Both modes run the same model; embedding, RMSNorm, and the MLP matmuls are the
same ops (only shapes differ). The difference is the **attention block**:

| Attention step | Prefill op | Decode op |
|---|---|---|
| split Q/K/V heads | `nlp_create_qkv_heads` | `nlp_create_qkv_heads_decode` |
| attention math | `scaled_dot_product_attention` | `scaled_dot_product_attention_decode` |
| KV cache | `fill_cache` / `paged_fill_cache` (write prompt) | `paged_update_cache` (append one token) |
| merge heads | `nlp_concat_heads` | `nlp_concat_heads_decode` |

- **Prefill** = whole prompt in one pass. These ops are supported (PR #177/#198).
- **Decode** = one token at a time over a growing cache. Every op is a separate
  `_decode` kernel — none supported — plus a decode-specific cross-core
  semaphore-sync runtime bug.

## 4. Op support map

### Supported ✅
| Op | Used in | Evidence |
|---|---|---|
| `ttnn.embedding` | embedding | `dm_test_embedding_*` |
| `ttnn.rms_norm` | norms | `fused_test_rms_norm` + `rmsnorm.h` |
| `ttnn.matmul` / `ttnn.linear` | MLP, attn proj | `matmul_test_*` + `matmul.h` |
| `silu` / `mul` / `add` | MLP, residual | `elt_test_*` + eltwise shims |
| reshape / transpose / pad / typecast / to_memory_config / sharded↔interleaved | plumbing | curated suite |
| `scaled_dot_product_attention` (prefill / chunked / joint) | attention | `sdpa_test_{prefill,chunked,joint}` (PR #177) |
| RoPE prefill, qkv-heads prefill, KV-cache **fill** | attention | PR #198 `test_attention.py` (Llama) |

SDPA prefill `jit_hw` surface: `api/compute/experimental/sdpa_sub_custom.h`;
`cpp/ttnn/operations/transformer/sdpa/device/kernels/{sdpa_streaming_qktv,chunked_prefill_utils,q_chunk_remapping,sliding_window_geometry}.hpp`;
`experimental/llk_sfpu/ckernel_sfpu_sdpa.h`.

### Not supported ❌ (+ issue)
| Op / capability | Used in | emule issue |
|---|---|---|
| `scaled_dot_product_attention_decode` / `paged_..._decode` | decode | #147 (OPEN) |
| `paged_update_cache` / `paged_fused_update_cache` | decode | #147 |
| KV-cache sliding / bounded | decode | #147 |
| MLA prefill + decode | DeepSeek-style | #147 |
| `nlp_create_qkv_heads_decode` / `nlp_concat_heads_decode` | decode | bundled ~#147 (not separately filed) |
| CCL fused all-gather-matmul / `minimal_matmul` | full-depth model | #163 (OPEN) |
| decode semaphore-sync runtime bug | decode | unfiled (PR #198 finding; closest is #200, sort) |

Closed: #180 (SDPA prefill), #65 (BH ttnn.sdpa bring-up).

### Not needed single-chip 🔁
`all_gather_async`, `reduce_scatter_minimal_async`, `mesh_partition`,
`rms_norm_pre/post_all_gather`, `combine_device_tensors` — gated behind
`is_multichip`; drop out on a 1×1 mesh.

## 5. Test ladder

Prerequisite: an emule build (`build_emule/`) — see [BUILD_GUIDE.md](../BUILD_GUIDE.md).
Run env (from the tt-metal source root):

```bash
cd $TT_METAL_DIR                      # must be on/merged with origin/main (PR #177/#198)
export BUILD_DIR=$PWD/build_emule
export PYTHONPATH="$PWD/ttnn:$PWD/tools:$BUILD_DIR/lib:$PWD"
export LD_LIBRARY_PATH="$BUILD_DIR/lib"
export TT_METAL_HOME="$PWD" TT_METAL_RUNTIME_ROOT="$PWD"
export TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$PWD/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"
export MESH_DEVICE=N150
export HF_MODEL=Qwen/Qwen2.5-0.5B-Instruct       # or unsloth/Llama-3.2-1B-Instruct to reproduce PR #198
pip install transformers                          # into the toolchain venv (gated models also need `huggingface-cli login`)
```

Legend: ✅ proven (PR #198, Llama) · ❓ untested on Qwen · 🔧 needs local trace-off
edit · 🚧 blocked.

**Tier 1 — building-block ops (run today):**
`tests/ttnn/unit_tests/operations/fused/test_rms_norm.py`,
`.../fused/test_softmax.py`, `.../eltwise/test_silu.py`, `.../eltwise/test_mul.py`,
`.../matmul/test_matmul.py -k test_matmul_with_matched_width_height`,
`.../sdpa/test_sdpa_prefill.py`.

**Tier 2 — sub-blocks (PR #198 selectors):**
| pytest + selector | Status |
|---|---|
| `models/tt_transformers/tests/test_mlp.py -k '1-32-mesh_device0'` | ✅ Llama 0.9997 · ❓ Qwen |
| `models/tt_transformers/tests/test_attention.py -k 'hf_rope-256-1-page_params0-default_attention'` | ✅ Llama · ❓ Qwen |
| `models/tt_transformers/tests/test_attention.py -k 'hf_rope-256-1-page_params0-paged_attention'` | ✅ Llama 0.9998 · ❓ Qwen |
| `models/tt_transformers/tests/test_embedding.py` | ❓ |
| `models/tt_transformers/tests/test_rms_norm.py` | ❓ |

**Tier 3 — model level:**
| pytest + selector | Goal | Status |
|---|---|---|
| `models/tt_transformers/tests/test_model_prefill.py -k '1layer-performance-max128k-128-page_params0-paged_attention'` | 1-layer prefill forward | ✅ Llama 0.9966 · 🔧 `enable_trace=False`/`warmup_prefill=False` · ❓ Qwen |
| `models/tt_transformers/tests/test_model_prefill.py -k 'all_layers-...'` | full-depth (24-layer) prefill | 🚧 CCL all-gather-matmul shim (#163) |
| `models/tt_transformers/tests/test_model.py` | full model + generation (decode) | 🚧 decode ops (#147) + semaphore-sync |

## 6. Operational notes

**The trace-off edit (required for `test_model_prefill.py`).** emule is
slow-dispatch, and `ttnn.begin_trace_capture` throws "Trace not supported for
slow dispatch" — a tt-metal dispatch contract, not an emule bug. The prefill
generator call defaults to trace on, so the test must be edited locally to
disable it. At `models/tt_transformers/tests/test_model_prefill.py:228`, pass
both args to `prefill_forward_text` (defaults `enable_trace=True`,
`warmup_prefill=True` live at `models/tt_transformers/tt/generator.py:528`):

```python
tt_output_torch = generator.prefill_forward_text(
    tt_prefill_input,
    page_table=page_table,
    kv_cache=[tt_kv_cache],
    prompt_lens=prompt_lens,
    enable_trace=False,      # emule slow-dispatch: no trace capture
    warmup_prefill=False,    # warmup path also captures a trace
)
```

Trace is a perf optimization, so disabling it does **not** change the
computation — the PCC result is genuine. The edit lives in the tt-metal tree,
is uncommitted, and is **not** part of any emule/tt-metal PR, so a clean checkout
will not reproduce the pass without it.

**Branch / main state.** The SDPA prefill `jit_hw` surface and the two model
bring-up fixes land on `origin/main` (PR #177, PR #198). A feature branch cut
before those merges will JIT-fail on the SDPA kernel headers. Confirm the merge
base before relying on §4's "supported" rows; rebase/merge `origin/main` if not
present.

## 7. Goal tiers

| Goal | Test | Remaining work |
|---|---|---|
| 1-layer Qwen prefill | `test_model_prefill.py -k 1layer` | trace-off edit; confirm Qwen QK-norm / QKV-bias paths — **nearest** |
| Full-depth Qwen prefill (Qwen *runs*) | `test_model_prefill.py -k all_layers` | CCL fused all-gather-matmul shim (#163) |
| Qwen *generates text* | `test_model.py` | SDPA-decode + KV-update + decode head-reshape (#147) + decode semaphore-sync fix |

Metrics: functional only — **PCC vs the HuggingFace reference** (pass bar ~0.99;
PR #198 Llama baselines: MLP 0.9997, attention 0.9998, 1-layer prefill 0.9966),
with **zero regressions** on `scripts/run_regression_{wormhole,blackhole}.sh`.
emule has no timing model, so there is no performance metric.

Immediate next step: run Tier 2 + the 1-layer prefill with `HF_MODEL=Qwen/...`
and see whether Qwen tracks Llama's results — that converts every ❓ to a fact.
