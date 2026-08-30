#!/bin/bash
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Launch the single-layer DSv4-Pro MoE benchmark on GB300 (4 GPUs/node).
#
#   # 2 nodes / 8 GPUs, expert-GEMM-shape-preserving preset:
#   sbatch --nodes=2 --segment=2 benchmarks/moe/run_dsv4_moe_layer.sbatch \
#       --preset ep8
#
#   # 16 nodes / 64 GPUs, exact production shape:
#   sbatch --nodes=16 --segment=16 benchmarks/moe/run_dsv4_moe_layer.sbatch \
#       --preset prod
#
# Everything after the script name is forwarded to bench_dsv4_moe_layer.py.
#
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=gb300
#SBATCH --job-name=coreai_dlalgo_llm-llm.dsv4_moe_unit_bench
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out

set -eux

MEGATRON_LM="${MEGATRON_LM:-/lustre/fsw/coreai_dlalgo_llm/rghadia/megamoe/repos/Megatron-LM}"
CONTAINER="${CONTAINER:-/lustre/fsw/coreai_dlalgo_llm/rghadia/megamoe/images/may20-cutedsl450-nvrx060-cudnnfedsapr-te2898-bgemm.sqsh}"
RESULTS_DIR="${RESULTS_DIR:-${MEGATRON_LM}/benchmarks/moe/results/${SLURM_JOB_ID}}"
mkdir -p "${RESULTS_DIR}"

# EP defaults to the total GPU count unless the caller passes --ep / --preset.
NUM_GPUS=$(( SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE ))

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=${MASTER_PORT:-29511}

# --- Environment ------------------------------------------------------------
# Mirrors the MoE-relevant subset of the production perf recipe
# (Megatron-Bridge/src/megatron/bridge/perf_recipes/deepseek/gb300/deepseek_v4.py
#  :deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config, env_vars block).
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN is set to the EP size so the whole EP
# group lives inside one NVL72 NVLink domain, exactly as in production where
# EP=64 <= NVLINK_DOMAIN_SIZE=72.
BENCH_ENV=(
  CUDA_DEVICE_MAX_CONNECTIONS=32
  NCCL_GRAPH_REGISTER=0
  NCCL_NVLS_ENABLE=0
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  TORCH_NCCL_AVOID_RECORD_STREAMS=0
  USE_MNNVL=1
  NVLINK_DOMAIN_SIZE=72
  NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${HYBRID_EP_RANKS_PER_DOMAIN:-${NUM_GPUS}}"
  NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
  NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
  NVTE_NORM_FWD_USE_CUDNN=1
  NVTE_NORM_BWD_USE_CUDNN=1
  NVTE_FWD_LAYERNORM_SM_MARGIN=20
  NVTE_BWD_LAYERNORM_SM_MARGIN=20
  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
  PYTHONUNBUFFERED=1
)

# --- Optional nsys ----------------------------------------------------------
# ENABLE_NSYS=1 sbatch ... ; the script emits cudaProfilerApi start/stop around
# the measured iterations, so the capture range covers only steady state.
PROFILE_CMD=()
BENCH_EXTRA=()
if [[ "${ENABLE_NSYS:-0}" == "1" ]]; then
  PROFILE_CMD=(nsys profile -s none -t cuda,nvtx
    -o "${RESULTS_DIR}/moe_bench_%q{SLURM_PROCID}"
    --force-overwrite=true --capture-range=cudaProfilerApi
    --capture-range-end=stop --cuda-graph-trace=node)
  BENCH_EXTRA+=(--nsys-capture)
fi

srun \
  --container-image "${CONTAINER}" \
  --container-mounts "/lustre:/lustre" \
  --container-workdir "${MEGATRON_LM}" \
  --no-container-mount-home \
  --container-writable \
  --mpi=pmix \
  --kill-on-bad-exit=1 \
  --output "${RESULTS_DIR}/log_%j_%t.out" \
  bash -c "
    export $(printf '%s ' "${BENCH_ENV[@]}")
    export RANK=\$SLURM_PROCID
    export LOCAL_RANK=\$SLURM_LOCALID
    export WORLD_SIZE=\$SLURM_NTASKS
    export PYTHONPATH=${MEGATRON_LM}:\${PYTHONPATH:-}
    ${PROFILE_CMD[*]} python -u ${MEGATRON_LM}/benchmarks/moe/bench_dsv4_moe_layer.py \
      --json-out ${RESULTS_DIR}/result_rank\${SLURM_PROCID}.json \
      ${BENCH_EXTRA[*]} $*
  "
