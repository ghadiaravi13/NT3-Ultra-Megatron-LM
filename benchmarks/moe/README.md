# DeepSeek-V4-Pro single MoE layer benchmark (Megatron-LM baseline)

Measures forward and backward time for **one** DeepSeek-V4-Pro MoE layer using the
`flex` token dispatcher with the `hybridep` backend and the MXFP8 TE grouped MLP.

This is the **baseline** half of an A/B comparison against a MegaMoE megakernel
implementation of the same layer. Read the "Timing boundary" section before
comparing any numbers.

## Quick start

```bash
# 2 GB300 nodes / 8 GPUs, recommended first run
sbatch --nodes=2 --segment=2 benchmarks/moe/run_dsv4_moe_layer.sbatch --preset ep8

# 16 GB300 nodes / 64 GPUs, exact production shape
sbatch --nodes=16 --segment=16 benchmarks/moe/run_dsv4_moe_layer.sbatch --preset prod

# with nsys + per-stage NVTX ranges
ENABLE_NSYS=1 sbatch --nodes=2 --segment=2 \
  benchmarks/moe/run_dsv4_moe_layer.sbatch --preset ep8 --breakdown

# single node smoke test
sbatch --nodes=1 benchmarks/moe/run_dsv4_moe_layer.sbatch --preset ep4 --iters 10
```

Results and per-rank JSON land in `benchmarks/moe/results/$SLURM_JOB_ID/`.

## Authoritative DSv4-Pro MoE shapes

The production recipe is
`deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config`
(`Megatron-Bridge/src/megatron/bridge/perf_recipes/deepseek/gb300/deepseek_v4.py:53`).
It builds the model provider from the HF config at
`DEEPSEEK_V4_PRO_HF_PATH = "deepseek-ai/DeepSeek-V4-Pro"`
(`Megatron-Bridge/src/megatron/bridge/recipes/deepseek/gb300/deepseek_v4.py:32,43`),
so the shapes are **not** literals in the repo. They come from the HF config,
read here from the local snapshot:

```
/lustre/fsw/coreai_dlalgo_llm/rghadia/DSv4/hf_cache/hub/
  models--deepseek-ai--DeepSeek-V4-Pro/snapshots/
  b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
```

| Megatron field | Value | HF field (`config.json` line) | Mapping site |
| --- | --- | --- | --- |
| `hidden_size` | 7168 | `hidden_size` (15) | generic table, `model_bridge.py:459` |
| `moe_ffn_hidden_size` | 3072 | `moe_intermediate_size` (22) | generic table, `model_bridge.py:482` |
| `num_moe_experts` | 384 | `n_routed_experts` (23) | generic table, `model_bridge.py:486` |
| `moe_shared_expert_intermediate_size` | 3072 | `moe_intermediate_size * n_shared_experts` (22, 24) | `deepseek_v4_bridge.py:508` |
| `moe_router_topk` | 6 | `num_experts_per_tok` (27) | `deepseek_v4_bridge.py:495` |
| `moe_router_topk_scaling_factor` | **2.5** | `routed_scaling_factor` (55) | `deepseek_v4_bridge.py:497` |
| `moe_router_score_function` | `sqrtsoftplus` | `scoring_func` (56) | `deepseek_v4_bridge.py:490` |
| `moe_router_enable_expert_bias` | True | — | `deepseek_v4_bridge.py:491` |
| `moe_router_load_balancing_type` | `noaux_tc` | `topk_method` (60) | `deepseek_v4_bridge.py:488` |
| `activation_func_clamp_value` | 10.0 | `swiglu_limit` (58) | `deepseek_v4_bridge.py:504` |
| `moe_n_hash_layers` | 3 | `num_hash_layers` (29) | `deepseek_v4_bridge.py:500` |
| decoder layers | 61 | `num_hidden_layers` (28) | — |
| `mtp_num_layers` | 1 | `num_nextn_predict_layers` (31) | `deepseek_v4_bridge.py:511` |

Two corrections to note:

* `moe_router_topk_scaling_factor` is **2.5**, not 1.5. The inline comment at
  `deepseek_v4_bridge.py:497` says `# 1.5` but the value read from the HF config
  is `routed_scaling_factor: 2.5`. The comment is stale.
* `norm_topk_prob` (`deepseek_v4_bridge.py:496`) has **no corresponding field**
  in this Megatron-LM checkout's `TransformerConfig`. Setting it would be a
  no-op, so the benchmark does not set it. The equivalent behaviour lives in
  `moe_router_pre_softmax=False` plus the post-topk renormalisation inside
  `topk_routing_with_score_function`.

## Derived per-rank problem size

`micro_batch_size=1`, `seq_length=4096`, `TP=1`, `CP=1`,
`sequence_parallel=False` means every rank feeds the **full 4096-token
micro-batch** into the MoE layer. With `moe_router_force_load_balancing=True`
the routing is uniform, so:

| Quantity | Formula | Production value |
| --- | --- | --- |
| tokens in per rank | `mbs * seq_len` | 4096 |
| tokens received per rank | `tokens_in * topk` | 24576 (independent of EP) |
| local experts | `num_moe_experts / EP` | 6 |
| tokens per expert (grouped-GEMM M) | `tokens_in * topk * EP / num_moe_experts` | 4096 |
| forward FLOPs per rank | see `analytical_costs()` | 3.81 TFLOP |
| dispatch A2A payload per rank | `tokens_in * topk * hidden * 2 B` | 336 MiB |
| expert params per rank | `local_experts * (h*2f + f*h)` | 396 M |
| expert weight bytes read per fwd | MXFP8 primary (`--fp8-param`) | 390 MiB |
| expert weight bytes read per fwd | bf16 primary (default here) | 756 MiB |

## Timing boundary

The `--breakdown` NVTX names below match the `MoELayer` stage methods
(`megatron/core/transformer/moe/moe_layer.py`) and the six-phase
`MoETokenDispatcher` protocol (`token_dispatcher.py:124-239`).

### Inside the timed region

The timed region is exactly one call to `MoELayer.forward(hidden_states)` and one
`out.backward(grad)`, wrapped in the MXFP8 quantization context:

| NVTX range | `MoELayer` method | Work |
| --- | --- | --- |
| `shared_experts_compute` | `shared_experts_compute` | Shared expert `linear_fc1` -> clamped SwiGLU -> `linear_fc2`. `moe_shared_expert_overlap=False`, so this is serial, not hidden behind the A2A. |
| `route` | `route` | Router gate GEMM `[tokens, h] x [h, num_experts]` in fp32, `sqrtsoftplus` scoring, top-6 select, `noaux_tc` expert-bias update, `topk_scaling_factor`. With `moe_router_force_load_balancing=True` the logits are replaced by uniform random values before scoring. |
| `preprocess` | `preprocess` -> `dispatch_preprocess` | Reshape to `[tokens, h]`, expand routing map/probs to the TPxEP domain, `_HybridEPManager.setup_metadata` (capacity-budget computation). |
| `dispatch` | `dispatch` -> `token_dispatch` | HybridEP fused permute + dispatch all-to-all + permute-for-experts, 32 SMs. |
| `routed_experts_compute` | `routed_experts_compute` -> `dispatch_postprocess`, experts, `combine_preprocess` | MXFP8 padding, TE grouped `linear_fc1` (GLU, interleave 32), clamped SwiGLU, grouped `linear_fc2`, paged activation stash, unpadding. |
| `combine` | `combine` -> `token_combine` | HybridEP unpermute + combine all-to-all + unpermute-for-attention. |
| `postprocess` | `postprocess` -> `combine_postprocess` | Restore `[seq, batch, h]`, add the shared expert output. |

Backward covers the autograd reverse of all of the above, including the dispatch
and combine A2A backward passes and the expert weight gradients.
`delay_wgrad_compute=False`, so wgrad runs inline inside `backward()`.

### Outside the timed region

* **Pre-MLP layernorm and the MLP residual add.** These live in
  `TransformerLayer`/`HyperConnectionTransformerLayer`, not `MoELayer`. If MegaMoE
  fuses the layernorm into its megakernel, that comparison is not
  apples-to-apples; add the layernorm cost to this baseline explicitly.
* **Everything non-MoE**: MLA attention, DSA indexer, mHC hyper-connections, MTP
  head, embedding, loss.
* **Optimizer, gradient reduce-scatter, parameter all-gather, `backward_dw()`
  outside the autograd call.** No DDP or distributed optimizer is instantiated.
* **CUDA graph capture.** Production uses `cuda_graph_impl="full_iteration"`,
  which removes per-kernel launch overhead across the whole iteration. This
  benchmark runs eager, so **launch overhead is included here and is not
  included in production**. Expect this baseline to be pessimistic by roughly the
  CPU-launch cost of the MoE layer's kernels.
* **EP 1F1B comm/compute overlap.** `overlap_moe_expert_parallel_comm=False` in
  the production baseline (`Megatron-Bridge/src/megatron/bridge/recipes/deepseek/gb300/deepseek_v4.py:118`,
  never overridden), so the dispatch/combine A2A is fully exposed in both the
  production run and here. This is the intended baseline.
* **Warmup iterations** (default 20) and the `torch.distributed.barrier()` placed
  before each measured iteration.

### How timing is done

* `torch.cuda.Event` pairs around the forward and around the backward,
  `torch.cuda.synchronize()` after each iteration.
* `--warmup 20` iterations first (HybridEP allocates its buffer lazily on the
  first dispatch; TE JIT-compiles the CuTe DSL grouped GEMM).
* Median over `--iters` (default 50) per rank, then **max across ranks** as the
  headline number, because the A2A couples the ranks and the slowest rank sets
  the layer latency. The rank-average is also printed.
* `--breakdown` runs an extra pass that calls the stage methods one at a time
  with a `cuda.synchronize()` between each. Those numbers **serialize** the
  pipeline and therefore sum to more than the end-to-end time. Use them for
  attribution, and use the nsys trace for anything load-bearing.

### dgrad / wgrad split

There is no clean config-level split in this baseline. `delay_wgrad_compute`
(which would let `backward()` do dgrad only and `backward_dw()` do wgrad)
*requires* `overlap_moe_expert_parallel_comm=True`
(`transformer_config.py:3630-3633`), which the baseline deliberately has off.

`--dgrad-only-probe` estimates the split by timing a second backward with all
layer parameters frozen (`requires_grad=False`), so autograd computes dgrad only:

```
wgrad ≈ full_backward − dgrad_only_backward
```

Treat this as an estimate. Freezing parameters can change which tensors TE saves
for backward and skips the fused wgrad-accumulation epilogue, so the difference
is not exactly the production wgrad cost. For a load-bearing split, use the nsys
trace and attribute kernels.

## Reduced-EP presets

`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64 <= NVLINK_DOMAIN_SIZE=72` in the
production recipe, and the 256-GPU job is launched with `--segment=16` (16 nodes
x 4 GPUs = 64 GPUs per segment). **The entire EP=64 group therefore lives inside
one NVL72 NVLink domain.** MNNVL/`USE_MNNVL=1` matters for the data-parallel
dimension across domains, not for the EP all-to-all. HybridEP works fine below
64 ranks: the in-tree functional tests run it at EP=16 with
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8`
(`tests/functional_tests/test_cases/mixtral/deepseekv3_proxy_flex_tp1pp4emp16etp1cp1_release/model_config.yaml:16,104`).

So dropping EP does **not** change the transport (NVLink either way). What
changes is the peer count and, unless you compensate, the grouped-GEMM shape.
Since with balanced routing

```
tokens_received_per_rank = tokens_in * topk                       (EP-independent)
tokens_per_expert        = tokens_in * topk * EP / num_moe_experts
```

holding `num_moe_experts / EP = 6` and `tokens_in = 4096` fixed reproduces the
production per-rank shape exactly:

| preset | GPUs | EP | experts | tokens/rank | local experts | tokens/expert | A2A payload | expert params/rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prod` | 64 | 64 | 384 | 4096 | 6 | 4096 | 336 MiB | 396 M |
| `ep8` | 8 | 8 | 48 | 4096 | 6 | 4096 | 336 MiB | 396 M |
| `ep4` | 4 | 4 | 24 | 4096 | 6 | 4096 | 336 MiB | 396 M |
| `ep8-all-experts` | 8 | 8 | 384 | 4096 | 48 | 512 | 336 MiB | 3.17 B |

`ep8` differs from `prod` in exactly two ways:

1. **8 all-to-all peers instead of 64.** Same total bytes per rank, but 8 larger
   messages instead of 64 smaller ones, so `ep8` will show *better* A2A
   efficiency than `prod`. This is the one number you cannot trust from a
   reduced-EP run.
2. **Router gate is `[7168 x 48]` instead of `[7168 x 384]`** — 2.9 GFLOP vs
   22.5 GFLOP out of 3.79/3.81 TFLOP total, i.e. 0.5% of forward FLOPs. The
   `noaux_tc` top-6-of-48 select is also cheaper than top-6-of-384.

Use `ep8-all-experts` if you specifically want the production router width; it
trades away the grouped-GEMM shape.

## Deviations from the production recipe

| Setting | Production | Here | Why |
| --- | --- | --- | --- |
| `num_layers` | 61 | 1 | Unit benchmark. `MoELayer` reads `num_layers` only for pipeline bookkeeping. |
| `pipeline_model_parallel_size` / VPP | 4 / 4 | 1 / None | A single layer has no pipeline. |
| `cuda_graph_impl` | `full_iteration` | `none` | Cannot capture a whole training iteration for a detached layer. **Includes kernel launch overhead that production does not pay.** |
| `moe_n_hash_layers` | 3 | 0 | Layers 1-3 of 61 hash-route; the benchmark targets the ordinary MoE layer covering the other 58. Hash routing needs `input_ids` + a `tid2eid` table and has a cheaper router. |
| `recompute_granularity` / `recompute_modules` | `selective` / `["mla_up_proj", "mhc"]` | `None` / `[]` | Neither module is inside `MoELayer`, so this is equivalent. |
| `moe_pad_experts_for_cuda_graph_inference` | True | dropped | Only read by the inference text-generation controller. |
| `fp8_param` (`fp8_param_gather`) | True | off unless `--fp8-param` | No distributed optimizer here to own MXFP8 primary weights. Without it the expert weights are bf16 and cast per forward, so this baseline pays a cast production does not. |
| `gradient_accumulation_fusion` | True | True, with `param.main_grad` allocated manually | A bare `MoELayer` has no DDP wrapper to own the buffers. Disable with `--no-grad-accum-fusion`. |
| `main_grads_dtype` | bf16 | bf16 (`--main-grad-dtype`) | Matches `cfg.optimizer.main_grads_dtype`. |
| `overlap_moe_expert_parallel_comm` | False | False | Unchanged; the baseline is deliberately non-overlapped. |
| `moe_flex_dispatcher_num_sms` | 32 | 32 | Unchanged. |
| `moe_expert_rank_capacity_factor` / `moe_paged_stash` | 1.5 / True | 1.5 / True | Unchanged. Both require `use_transformer_engine_op_fuser=True` (`transformer_config.py:2209-2215`, `2511-2515`). |

## Verification status

**Nothing in this directory has been executed.** The script was authored and
statically validated (byte-compile, argument/preset resolution, `build_config`
across all flag combinations, analytical cost math) on a login node with no GPU
and no PyTorch. The Lyris SLURM controller was down at the time of writing:

```
$ scontrol ping
Slurmctld(primary) at slurm-lyris01 is DOWN
Slurmctld(backup) at slurm-lyris02 is DOWN
```

Unverified, in rough order of how likely they are to bite:

1. Whether `deep_ep` (HybridEP branch) is importable in
   `may20-cutedsl450-nvrx060-cudnnfedsapr-te2898-bgemm.sqsh`, and its version.
   The script prints this on rank 0 as its first action after setup.
2. Whether the container's TE satisfies `TEGroupedMLP._is_fused_impl_supported()`
   (`megatron/core/transformer/moe/experts.py:322`) so
   `use_transformer_engine_op_fuser=True` does not assert. Fall back with
   `--no-op-fuser` if it does.
3. Whether `TransformerConfig.__post_init__` accepts this exact field
   combination. Each individual constraint was read from
   `transformer_config.py`, but the combination has not been instantiated.
4. Whether the `--breakdown` stage sequence runs clean outside
   `MoELayer.forward`. It mirrors `custom_forward`
   (`moe_layer.py:739-786`) call for call, but the stage methods have never been
   driven externally.
5. All performance numbers. There are none in this file; do not add any that were
   not produced by a run.

## Requirements

* `deep_ep` with the HybridEP branch (`from deep_ep import HybridEPBuffer`,
  `megatron/core/transformer/moe/fused_a2a.py:493-498`). The script prints the
  version on rank 0 and fails loudly if the import is missing.
* HybridEP asserts FP8 **dispatch** is unsupported
  (`fused_a2a.py:548`), so the A2A payload is bf16 even under MXFP8 compute. The
  quantization happens inside the grouped MLP, after dispatch.
* HybridEP requires fp32 routing probabilities; `moe_router_dtype="fp32"` is set
  unconditionally. Otherwise `_HybridEPManager.dispatch` warns and upcasts
  (`token_dispatcher.py:1194-1200`).
* `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` and sm100+ for the fused grouped GEMM, set
  by the launcher.

## Why a new `benchmarks/` directory

`tools/` in this repo holds data-preprocessing and checkpoint utilities;
`tests/unit_tests/transformer/moe/` holds correctness tests. Neither is a natural
home for a perf script.

`tests/unit_tests/transformer/moe/test_token_dispatcher.py:111`
(`MoEModelTestContainer`) was evaluated for reuse and rejected for two reasons:
it wires up `get_gpt_layer_local_submodules` (the **local**, non-TE MLP, so no
MXFP8 `TEGroupedMLP`), and `tests/unit_tests/test_utilities.py:46` derives the
global rank from `LOCAL_RANK`, which breaks the multi-node runs this benchmark
needs. The parallel-state initialization here is the same sequence
`Utils.initialize_model_parallel` performs, with correct `RANK` handling.
