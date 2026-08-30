# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit-level forward/backward benchmark for a single DeepSeek-V4-Pro MoE layer.

This is the BASELINE half of an A/B comparison against a MegaMoE megakernel
implementation of the same layer. The timed region is defined in
``benchmarks/moe/README.md``; read it before comparing numbers.

Launch with torchrun (single node) or srun (multi node). See
``benchmarks/moe/run_dsv4_moe_layer.sbatch``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

# ---------------------------------------------------------------------------
# DeepSeek-V4-Pro MoE shapes.
#
# Authoritative source: the HF config that Megatron-Bridge loads for the
# production recipe (DEEPSEEK_V4_PRO_HF_PATH = "deepseek-ai/DeepSeek-V4-Pro",
# Megatron-Bridge/src/megatron/bridge/recipes/deepseek/gb300/deepseek_v4.py:32).
# Local snapshot used to read these values:
#   /lustre/fsw/coreai_dlalgo_llm/rghadia/DSv4/hf_cache/hub/
#     models--deepseek-ai--DeepSeek-V4-Pro/snapshots/
#     b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
#
# HF -> Megatron name mapping is the generic table in
# Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py:478-489
# plus the DSv4-specific overrides in
# Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v4_bridge.py:480-508.
# ---------------------------------------------------------------------------
DSV4_PRO = dict(
    hidden_size=7168,  # config.json:15  hidden_size
    moe_ffn_hidden_size=3072,  # config.json:22  moe_intermediate_size
    num_moe_experts=384,  # config.json:23  n_routed_experts
    n_shared_experts=1,  # config.json:24  n_shared_experts
    moe_router_topk=6,  # config.json:27  num_experts_per_tok
    num_decoder_layers=61,  # config.json:28  num_hidden_layers
    moe_router_topk_scaling_factor=2.5,  # config.json:55  routed_scaling_factor
    moe_router_score_function="sqrtsoftplus",  # config.json:56  scoring_func
    activation_func_clamp_value=10.0,  # config.json:58  swiglu_limit
    moe_n_hash_layers=3,  # config.json:29  num_hash_layers
    vocab_size=129280,  # config.json:64  vocab_size
    # ffn_hidden_size is the dense-MLP width; DSv4-Pro has no dense MLP layers
    # (moe_layer_freq = [1] * 61, deepseek_v4_bridge.py:507). TransformerConfig
    # still requires a value, so we keep the DSv3-style 18432 and never use it.
    ffn_hidden_size=18432,
    num_attention_heads=128,  # config.json:26  num_attention_heads
)


# ---------------------------------------------------------------------------
# Presets.
#
# With perfectly balanced routing the per-rank MoE work is fully determined by:
#   tokens_received_per_rank = tokens_per_rank * topk           (independent of EP)
#   num_local_experts        = num_moe_experts / EP
#   tokens_per_expert        = tokens_per_rank * topk * EP / num_moe_experts
# So holding (num_moe_experts / EP) and tokens_per_rank fixed while shrinking EP
# reproduces the production grouped-GEMM shape, the per-rank A2A payload and the
# resident expert-weight footprint exactly. What necessarily changes is the
# number of A2A peers and the router gate width (num_moe_experts).
# See README.md.
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict[str, int]] = {
    # Exact production shape. Needs 64 GPUs = 16 GB300 nodes.
    "prod": {"ep": 64, "num_moe_experts": 384, "tokens_per_rank": 4096},
    # 8 GPUs = 2 GB300 nodes. RECOMMENDED first run. Preserves num_local_experts
    # (6), tokens_per_expert (4096), tokens_received_per_rank (24576), A2A
    # payload (336 MiB) and expert weight footprint. Differences vs prod: 8 A2A
    # peers instead of 64, and the router gate is [h x 48] instead of [h x 384]
    # (0.6% of forward FLOPs).
    "ep8": {"ep": 8, "num_moe_experts": 48, "tokens_per_rank": 4096},
    # 4 GPUs = 1 GB300 node. Same invariants as ep8, 4 A2A peers.
    "ep4": {"ep": 4, "num_moe_experts": 24, "tokens_per_rank": 4096},
    # 8 GPUs = 2 GB300 nodes, full 384-expert router. Keeps the production router
    # width and total expert count, but num_local_experts inflates to 48 and
    # tokens_per_expert drops 8x to 512, so the grouped GEMM is a different shape
    # and the resident expert weights are 8x larger. Use to isolate router cost.
    "ep8-all-experts": {"ep": 8, "num_moe_experts": 384, "tokens_per_rank": 4096},
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Sets --ep, --num-moe-experts and --tokens-per-rank together. "
        "Explicit flags override the preset.",
    )

    # --- Parallelism -------------------------------------------------------
    p.add_argument(
        "--ep",
        type=int,
        default=None,
        help="Expert parallel size. Defaults to WORLD_SIZE. Production recipe uses 64.",
    )
    p.add_argument("--etp", type=int, default=1, help="Expert tensor parallel size (production: 1)")

    # --- Problem size ------------------------------------------------------
    p.add_argument(
        "--tokens-per-rank",
        type=int,
        default=None,
        help="Tokens entering the MoE layer on each rank (default 4096). Production "
        "recipe: micro_batch_size=1 x seq_length=4096 with TP=1, CP=1, "
        "sequence_parallel=False, so every rank feeds the full 4096-token "
        "micro-batch into the MoE layer.",
    )
    p.add_argument("--hidden-size", type=int, default=DSV4_PRO["hidden_size"])
    p.add_argument("--moe-ffn-hidden-size", type=int, default=DSV4_PRO["moe_ffn_hidden_size"])
    p.add_argument("--num-moe-experts", type=int, default=None)
    p.add_argument("--topk", type=int, default=DSV4_PRO["moe_router_topk"])
    p.add_argument(
        "--no-shared-expert",
        action="store_true",
        help="Drop the shared expert from the layer (and from the timed region).",
    )

    # --- Precision ---------------------------------------------------------
    p.add_argument(
        "--quant",
        choices=["mxfp8", "bf16"],
        default="mxfp8",
        help="mxfp8 matches the production recipe (compute_dtype fp8_mx).",
    )
    p.add_argument(
        "--fp8-param",
        action="store_true",
        help="MXFP8 primary weights (config.fp8_param), matching the production "
        "mixed_precision.fp8_param_gather=True. Without it the expert weights are "
        "stored in bf16 and cast to MXFP8 on every forward, which ADDS a cast the "
        "production run does not pay. Off by default because there is no "
        "distributed optimizer here to own the FP8 primary weights.",
    )

    # --- Dispatcher --------------------------------------------------------
    p.add_argument("--dispatcher-backend", default="hybridep", choices=["hybridep", "deepep", "deepepv2", "ncclep"])
    p.add_argument("--dispatcher-num-sms", type=int, default=32)
    p.add_argument(
        "--expert-rank-capacity-factor",
        type=float,
        default=1.5,
        help="moe_expert_rank_capacity_factor. Production: 1.5. Set <=0 to disable "
        "(disables moe_paged_stash too and reintroduces a CPU sync in dispatch).",
    )
    p.add_argument(
        "--no-paged-stash",
        action="store_true",
        help="Disable moe_paged_stash (production has it on).",
    )
    p.add_argument(
        "--no-op-fuser",
        action="store_true",
        help="Escape hatch: disable use_transformer_engine_op_fuser (the TE CuTe DSL "
        "fused grouped MLP). Production has it ON. Also forces "
        "moe_expert_rank_capacity_factor and moe_paged_stash off, since both require "
        "the op fuser (transformer_config.py:2209-2215, 2511-2515). Use only if the "
        "container's TE cannot activate the fused kernel.",
    )
    p.add_argument(
        "--no-force-load-balancing",
        action="store_true",
        help="Disable moe_router_force_load_balancing. Production has it ON, which "
        "replaces router logits with uniform random values so every rank sees a "
        "perfectly balanced expert load. Leaving it on makes the benchmark "
        "reproducible and matches the production perf recipe.",
    )

    # --- Measurement -------------------------------------------------------
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--breakdown",
        action="store_true",
        help="Additionally run a stage-by-stage pass with NVTX ranges and per-stage "
        "CUDA events. Stage timings SERIALIZE the pipeline, so they attribute cost "
        "but do not sum to the end-to-end number.",
    )
    p.add_argument(
        "--dgrad-only-probe",
        action="store_true",
        help="Estimate the dgrad/wgrad split by additionally timing a backward with "
        "all layer parameters frozen (requires_grad=False). wgrad ~= full_bwd - dgrad_bwd. "
        "This is an estimate; see README.",
    )
    p.add_argument(
        "--no-grad-accum-fusion",
        action="store_true",
        help="Disable gradient_accumulation_fusion. Production has it ON, which needs a "
        "param.main_grad buffer; this script allocates those manually (DDP normally does).",
    )
    p.add_argument(
        "--main-grad-dtype",
        default="bf16",
        choices=["bf16", "fp32"],
        help="dtype of the manually allocated param.main_grad buffers. Production "
        "optimizer.main_grads_dtype is torch.bfloat16.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--json-out", default=None, help="Write results JSON to this path (rank 0).")
    p.add_argument(
        "--nsys-capture",
        action="store_true",
        help="Emit cudaProfilerStart/Stop around the measured iterations only.",
    )
    args = p.parse_args(argv)

    preset = PRESETS.get(args.preset, {})
    if args.ep is None:
        args.ep = preset.get("ep")
    if args.num_moe_experts is None:
        args.num_moe_experts = preset.get("num_moe_experts", DSV4_PRO["num_moe_experts"])
    if args.tokens_per_rank is None:
        args.tokens_per_rank = preset.get("tokens_per_rank", 4096)
    return args


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
def init_distributed(args: argparse.Namespace) -> Dict[str, int]:
    """Initialize torch.distributed and Megatron parallel state.

    Note: tests/unit_tests/test_utilities.py:Utils is the usual fixture for this,
    but it derives the *global* rank from LOCAL_RANK
    (test_utilities.py:46), which is wrong for multi-node runs. EP=8 spans two
    GB300 nodes, so we do the (equivalent, single-node) init by hand.
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    ep = args.ep if args.ep is not None else world_size
    if world_size % ep != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be a multiple of --ep={ep}")
    if args.num_moe_experts % ep != 0:
        raise ValueError(
            f"--num-moe-experts={args.num_moe_experts} must be divisible by --ep={ep} "
            "(BaseMoELayer asserts this, moe_layer.py:189)"
        )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=args.etp,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(args.seed)
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size, "ep": ep}


def build_config(args: argparse.Namespace, ep: int) -> TransformerConfig:
    """Build the TransformerConfig for one DSv4-Pro MoE layer.

    Every field below is either taken from the production recipe
    (Megatron-Bridge perf_recipes/deepseek/gb300/deepseek_v4.py
    :deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config) or from the DSv4 bridge
    (models/deepseek/deepseek_v4_bridge.py:480-524). Deviations are marked
    DEVIATION and explained in README.md.
    """
    op_fuser = not args.no_op_fuser
    capacity = args.expert_rank_capacity_factor
    rank_capacity_factor = capacity if (capacity and capacity > 0 and op_fuser) else None
    paged_stash = bool(rank_capacity_factor) and not args.no_paged_stash

    kwargs: Dict[str, Any] = dict(
        # --- Geometry ------------------------------------------------------
        # DEVIATION: num_layers=1. The layer under test is one of the 61
        # decoder layers; nothing in MoELayer depends on num_layers except
        # pipeline bookkeeping we do not use.
        num_layers=1,
        hidden_size=args.hidden_size,
        ffn_hidden_size=DSV4_PRO["ffn_hidden_size"],
        num_attention_heads=DSV4_PRO["num_attention_heads"],
        # --- MoE geometry --------------------------------------------------
        num_moe_experts=args.num_moe_experts,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        moe_shared_expert_intermediate_size=(
            None
            if args.no_shared_expert
            else args.moe_ffn_hidden_size * DSV4_PRO["n_shared_experts"]
        ),
        moe_layer_freq=[1],
        # --- Router --------------------------------------------------------
        moe_router_topk=args.topk,
        moe_router_load_balancing_type="noaux_tc",
        moe_router_score_function=DSV4_PRO["moe_router_score_function"],
        moe_router_enable_expert_bias=True,
        moe_router_topk_scaling_factor=DSV4_PRO["moe_router_topk_scaling_factor"],
        moe_router_pre_softmax=False,
        moe_router_dtype="fp32",  # required by HybridEP: probs must be fp32
        moe_aux_loss_coeff=0.0,
        moe_router_force_load_balancing=not args.no_force_load_balancing,
        # DEVIATION: moe_n_hash_layers=0. DSv4-Pro hash-routes decoder layers
        # 1-3 of 61 (deepseek_v4_bridge.py:500). We benchmark the ordinary
        # (non-hash) MoE layer that covers 58/61 layers; hash routing needs
        # input_ids and a tid2eid table and has a *cheaper* router.
        moe_n_hash_layers=0,
        # --- Experts -------------------------------------------------------
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=DSV4_PRO["activation_func_clamp_value"],
        use_te_activation_func=False,  # required when clamp value is set
        moe_mlp_glu_interleave_size=32,
        # --- Dispatcher ----------------------------------------------------
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend=args.dispatcher_backend,
        moe_flex_dispatcher_num_sms=args.dispatcher_num_sms,
        moe_hybridep_num_sms_preprocessing=108,
        moe_permute_fusion=True,
        moe_permute_fusion_into_hybridep=False,
        moe_shared_expert_overlap=False,
        moe_expert_rank_capacity_factor=rank_capacity_factor,
        moe_paged_stash=paged_stash,
        # --- Parallelism ---------------------------------------------------
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=args.etp,
        sequence_parallel=False,
        # EP 1F1B comm/compute overlap is OFF in the production baseline
        # (Megatron-Bridge recipes/deepseek/gb300/deepseek_v4.py:118).
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
        # --- Misc ----------------------------------------------------------
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl="transformer_engine",
        use_transformer_engine_op_fuser=op_fuser,  # required by expert_rank_capacity_factor
        gradient_accumulation_fusion=not args.no_grad_accum_fusion,
        # DEVIATION: recompute_modules is empty. The production recipe uses
        # recompute_granularity="selective" with ["mla_up_proj", "mhc"], neither
        # of which touches the MoE layer, so an empty list is equivalent here.
        recompute_granularity=None,
        recompute_modules=[],
        # DEVIATION: cuda_graph_impl left at "none". Production captures the
        # whole iteration ("full_iteration"), which cannot be reproduced for a
        # single detached layer. See README for what this costs.
        cuda_graph_impl="none",
        # DEVIATION: moe_pad_experts_for_cuda_graph_inference dropped. It is
        # only read by the inference text-generation controller.
    )

    if args.quant == "mxfp8":
        kwargs.update(fp8="e4m3", fp8_recipe="mxfp8", fp8_param=args.fp8_param)

    return TransformerConfig(**kwargs)


def build_layer(config: TransformerConfig, layer_number: int = 8):
    """Build a single MoELayer with the TE (GroupedMLP) submodule spec.

    layer_number defaults to 8 so the layer is past the DSv4-Pro hash-routing
    prefix (layers 1-3) even if moe_n_hash_layers is re-enabled.
    """
    mlp_builder = get_moe_module_spec(
        use_te=True, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    init_ctx = (
        get_fp8_context(config, layer_number - 1, is_init=True) if config.fp8 else contextlib.nullcontext()
    )
    with init_ctx:
        layer = mlp_builder(
            config=config, pg_collection=None, is_mtp_layer=False, name="benchmark_moe"
        )
    layer = layer.cuda()
    layer.set_layer_number(layer_number)
    layer.train()
    return layer


def allocate_main_grads(layer: torch.nn.Module, dtype: torch.dtype) -> None:
    """Allocate param.main_grad, which gradient_accumulation_fusion requires.

    In real training the DDP wrapper (or Megatron-FSDP) owns these buffers. A
    bare MoELayer has no wrapper, so we create them here to keep the fused
    wgrad-accumulation epilogue on the same code path as production.
    """
    for param in layer.parameters():
        if param.requires_grad:
            param.main_grad = torch.zeros(param.shape, dtype=dtype, device=param.device)


# ---------------------------------------------------------------------------
# Analytical FLOP / byte counts
# ---------------------------------------------------------------------------
def analytical_costs(args: argparse.Namespace, config: TransformerConfig, ep: int) -> Dict[str, float]:
    """Per-rank forward FLOPs and A2A payload bytes for the timed region.

    Assumes perfectly balanced routing (which moe_router_force_load_balancing
    enforces), so tokens received == tokens sent on average.
    """
    n = args.tokens_per_rank
    h = config.hidden_size
    f = config.moe_ffn_hidden_size
    k = config.moe_router_topk
    e = config.num_moe_experts

    # Per token per expert: fc1 [h x 2f] + fc2 [f x h] with a GLU fc1.
    per_token_per_expert = 2 * h * (2 * f) + 2 * f * h
    routed = n * k * per_token_per_expert
    shared = 0 if config.moe_shared_expert_intermediate_size is None else (
        n * (2 * h * (2 * config.moe_shared_expert_intermediate_size)
             + 2 * config.moe_shared_expert_intermediate_size * h)
    )
    router = 2 * n * h * e

    fwd_flops = routed + shared + router

    # A2A payload. Dispatch sends each token to k experts; combine returns k
    # partials. (k-1)/k * ... would be the strictly off-rank share for k
    # distinct ranks, but with e/ep experts per rank some replicas land locally.
    elem_bytes = 2  # HybridEP asserts fp8_dispatch is unsupported (fused_a2a.py:548)
    dispatch_bytes = n * k * h * elem_bytes
    combine_bytes = dispatch_bytes
    local_fraction = min(1.0, (e / ep) / e * k) if ep > 0 else 0.0

    # Expert weights read once per grouped GEMM. MXFP8 primary weights are ~1
    # byte/element plus a 1-byte E8M0 scale per 32 elements; bf16 primary weights
    # are read at 2 bytes and cast per forward.
    num_local_experts = e // ep
    expert_params = num_local_experts * (h * 2 * f + f * h)
    if config.fp8 and getattr(config, "fp8_param", False):
        weight_bytes = expert_params * (1 + 1 / 32)
    else:
        weight_bytes = expert_params * 2

    return {
        "tokens_received_per_rank": float(n * k),
        "tokens_per_expert": float(n * k * ep / e),
        "fwd_flops_per_rank": float(fwd_flops),
        "fwd_flops_routed_experts": float(routed),
        "fwd_flops_shared_expert": float(shared),
        "fwd_flops_router_gate": float(router),
        "a2a_dispatch_payload_bytes": float(dispatch_bytes),
        "a2a_combine_payload_bytes": float(combine_bytes),
        "a2a_local_replica_fraction_upper_bound": float(local_fraction),
        "expert_weight_bytes_per_fwd": float(weight_bytes),
        "num_local_experts": float(num_local_experts),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class EventTimer:
    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> "EventTimer":
        self.start.record()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop.record()

    def ms(self) -> float:
        return self.start.elapsed_time(self.stop)


def make_input(args: argparse.Namespace, config: TransformerConfig) -> torch.Tensor:
    """[seq, batch, hidden] as MoELayer.forward expects."""
    x = torch.randn(
        (args.tokens_per_rank, 1, config.hidden_size),
        dtype=config.params_dtype,
        device="cuda",
    )
    x.requires_grad_(True)
    return x


def run_fwd_bwd(
    layer, x: torch.Tensor, config: TransformerConfig, layer_number: int
) -> Dict[str, float]:
    """One timed forward + backward over the whole MoELayer."""
    quant_ctx = get_fp8_context(config, layer_number - 1) if config.fp8 else contextlib.nullcontext()

    fwd = EventTimer()
    torch.cuda.nvtx.range_push("moe_layer_fwd")
    with fwd:
        with quant_ctx:
            out, _ = layer(x)
    torch.cuda.nvtx.range_pop()

    grad = torch.randn_like(out)
    bwd = EventTimer()
    torch.cuda.nvtx.range_push("moe_layer_bwd")
    with bwd:
        out.backward(grad)
    torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
    return {"fwd_ms": fwd.ms(), "bwd_ms": bwd.ms()}


def run_breakdown(
    layer, x: torch.Tensor, config: TransformerConfig, layer_number: int
) -> Dict[str, float]:
    """Stage-by-stage forward using the MoELayer stage methods.

    Mirrors MoELayer.forward's custom_forward
    (megatron/core/transformer/moe/moe_layer.py:739-786) exactly, one stage at a
    time, with NVTX ranges and CUDA events. Each stage is CUDA-synchronized, so
    the stages do not overlap and the sum EXCEEDS the fused end-to-end time.
    """
    quant_ctx = get_fp8_context(config, layer_number - 1) if config.fp8 else contextlib.nullcontext()
    out: Dict[str, float] = {}

    def stage(name, fn, *a, **kw):
        torch.cuda.synchronize()
        t = EventTimer()
        torch.cuda.nvtx.range_push(name)
        with t:
            r = fn(*a, **kw)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        out[f"{name}_ms"] = t.ms()
        return r

    with quant_ctx:
        shared_out = stage("shared_experts_compute", layer.shared_experts_compute, x)
        probs, routing_map = stage("route", layer.route, x)
        hidden, probs = stage("preprocess", layer.preprocess, x, probs, routing_map)
        dispatched, probs = stage("dispatch", layer.dispatch, hidden, probs)
        expert_out, _ = stage("routed_experts_compute", layer.routed_experts_compute, dispatched, probs)
        combined = stage("combine", layer.combine, expert_out)
        final = stage("postprocess", layer.postprocess, combined, shared_out)

    grad = torch.randn_like(final)
    torch.cuda.synchronize()
    t = EventTimer()
    torch.cuda.nvtx.range_push("breakdown_bwd")
    with t:
        final.backward(grad)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    out["breakdown_bwd_ms"] = t.ms()
    return out


def zero_grads(layer, x: torch.Tensor) -> None:
    layer.zero_grad(set_to_none=True)
    x.grad = None
    for p in layer.parameters():
        if getattr(p, "main_grad", None) is not None:
            p.main_grad.zero_()


def reduce_stat(value: float, op: torch.distributed.ReduceOp) -> float:
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(t, op=op)
    return float(t.item())


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    info = init_distributed(args)
    rank, ep, world_size = info["rank"], info["ep"], info["world_size"]
    is_main = rank == 0

    config = build_config(args, ep)
    layer_number = 8
    layer = build_layer(config, layer_number)
    if config.gradient_accumulation_fusion:
        allocate_main_grads(
            layer, torch.bfloat16 if args.main_grad_dtype == "bf16" else torch.float32
        )

    x = make_input(args, config)

    if is_main:
        print(f"[bench] world_size={world_size} ep={ep} etp={args.etp}", flush=True)
        print(
            f"[bench] hidden={config.hidden_size} moe_ffn={config.moe_ffn_hidden_size} "
            f"experts={config.num_moe_experts} local_experts={config.num_moe_experts // ep} "
            f"topk={config.moe_router_topk} tokens_per_rank={args.tokens_per_rank} "
            f"quant={args.quant} dispatcher={args.dispatcher_backend}",
            flush=True,
        )
        print(
            f"[bench] op_fuser={config.use_transformer_engine_op_fuser} "
            f"fp8_param={getattr(config, 'fp8_param', False)} "
            f"rank_capacity_factor={config.moe_expert_rank_capacity_factor} "
            f"paged_stash={config.moe_paged_stash} "
            f"force_load_balancing={config.moe_router_force_load_balancing} "
            f"num_sms={config.moe_flex_dispatcher_num_sms} "
            f"grad_accum_fusion={config.gradient_accumulation_fusion} "
            f"cuda_graph={config.cuda_graph_impl}",
            flush=True,
        )
        for key in (
            "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
            "NVLINK_DOMAIN_SIZE",
            "USE_MNNVL",
            "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        ):
            print(f"[bench] env {key}={os.environ.get(key, '<unset>')}", flush=True)
        try:
            import deep_ep

            print(f"[bench] deep_ep version: {getattr(deep_ep, '__version__', 'unknown')}", flush=True)
        except ImportError as exc:  # pragma: no cover
            print(f"[bench] deep_ep NOT importable: {exc}", flush=True)

    # ----- warmup ---------------------------------------------------------
    for _ in range(args.warmup):
        zero_grads(layer, x)
        run_fwd_bwd(layer, x, config, layer_number)
    torch.distributed.barrier()

    if args.nsys_capture:
        torch.cuda.profiler.start()

    # ----- measure --------------------------------------------------------
    fwd_ms: List[float] = []
    bwd_ms: List[float] = []
    for i in range(args.iters):
        zero_grads(layer, x)
        torch.distributed.barrier()
        r = run_fwd_bwd(layer, x, config, layer_number)
        fwd_ms.append(r["fwd_ms"])
        bwd_ms.append(r["bwd_ms"])

    if args.nsys_capture:
        torch.cuda.profiler.stop()

    med_fwd = statistics.median(fwd_ms)
    med_bwd = statistics.median(bwd_ms)
    # Comm couples the ranks, so the slowest rank sets the layer latency.
    max_fwd = reduce_stat(med_fwd, torch.distributed.ReduceOp.MAX)
    max_bwd = reduce_stat(med_bwd, torch.distributed.ReduceOp.MAX)
    avg_fwd = reduce_stat(med_fwd, torch.distributed.ReduceOp.SUM) / world_size
    avg_bwd = reduce_stat(med_bwd, torch.distributed.ReduceOp.SUM) / world_size

    # ----- optional dgrad-only probe --------------------------------------
    dgrad_ms = None
    if args.dgrad_only_probe:
        saved = [(p, p.requires_grad) for p in layer.parameters()]
        for p, _ in saved:
            p.requires_grad_(False)
        for _ in range(max(3, args.warmup // 4)):
            zero_grads(layer, x)
            run_fwd_bwd(layer, x, config, layer_number)
        probe: List[float] = []
        for _ in range(args.iters):
            zero_grads(layer, x)
            torch.distributed.barrier()
            probe.append(run_fwd_bwd(layer, x, config, layer_number)["bwd_ms"])
        for p, flag in saved:
            p.requires_grad_(flag)
        dgrad_ms = reduce_stat(statistics.median(probe), torch.distributed.ReduceOp.MAX)

    # ----- optional stage breakdown ---------------------------------------
    breakdown = None
    if args.breakdown:
        zero_grads(layer, x)
        run_breakdown(layer, x, config, layer_number)  # warm the stage path
        zero_grads(layer, x)
        breakdown = run_breakdown(layer, x, config, layer_number)

    costs = analytical_costs(args, config, ep)

    if is_main:
        tflops_fwd = costs["fwd_flops_per_rank"] / (max_fwd * 1e-3) / 1e12
        tflops_bwd = 2 * costs["fwd_flops_per_rank"] / (max_bwd * 1e-3) / 1e12
        print("", flush=True)
        print("=== DSv4-Pro MoE layer, per-rank timings (median over iters, max over ranks) ===")
        print(f"  forward           : {max_fwd:8.3f} ms   (rank-average {avg_fwd:8.3f} ms)")
        print(f"  backward          : {max_bwd:8.3f} ms   (rank-average {avg_bwd:8.3f} ms)")
        print(f"  fwd+bwd           : {max_fwd + max_bwd:8.3f} ms")
        if dgrad_ms is not None:
            print(f"  backward (dgrad)  : {dgrad_ms:8.3f} ms   [params frozen probe]")
            print(f"  backward (wgrad)  : {max_bwd - dgrad_ms:8.3f} ms   [full - dgrad, estimate]")
        print("")
        print(f"  tokens recv/rank  : {costs['tokens_received_per_rank']:8.0f}")
        print(f"  tokens per expert : {costs['tokens_per_expert']:8.0f}  (grouped-GEMM M per group)")
        print(f"  fwd FLOPs/rank    : {costs['fwd_flops_per_rank'] / 1e12:8.3f} TFLOP")
        print(f"    routed experts  : {costs['fwd_flops_routed_experts'] / 1e12:8.3f} TFLOP")
        print(f"    shared expert   : {costs['fwd_flops_shared_expert'] / 1e12:8.3f} TFLOP")
        print(f"    router gate     : {costs['fwd_flops_router_gate'] / 1e12:8.3f} TFLOP")
        print(f"  fwd achieved      : {tflops_fwd:8.1f} TFLOP/s per GPU")
        print(f"  bwd achieved (2x) : {tflops_bwd:8.1f} TFLOP/s per GPU")
        print(f"  A2A dispatch      : {costs['a2a_dispatch_payload_bytes'] / 2**20:8.1f} MiB payload/rank")
        print(f"  A2A combine       : {costs['a2a_combine_payload_bytes'] / 2**20:8.1f} MiB payload/rank")
        print(f"  expert weights    : {costs['expert_weight_bytes_per_fwd'] / 2**20:8.1f} MiB read/fwd")
        if breakdown:
            print("")
            print("=== Stage breakdown (SERIALIZED; does not sum to end-to-end) ===")
            for name, val in breakdown.items():
                print(f"  {name:34s}: {val:8.3f} ms")

    if is_main and args.json_out:
        payload = {
            "args": vars(args),
            "world_size": world_size,
            "ep": ep,
            "fwd_ms_max_rank": max_fwd,
            "bwd_ms_max_rank": max_bwd,
            "fwd_ms_rank_avg": avg_fwd,
            "bwd_ms_rank_avg": avg_bwd,
            "bwd_dgrad_ms": dgrad_ms,
            "breakdown_ms": breakdown,
            "analytical": costs,
            "shapes_source": "deepseek-ai/DeepSeek-V4-Pro HF config.json",
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[bench] wrote {args.json_out}", flush=True)

    torch.distributed.barrier()
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
