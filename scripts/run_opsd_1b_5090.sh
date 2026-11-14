#!/usr/bin/env bash
# =============================================================================
#  OPSD (On-Policy Self-Distillation) -- Qwen3-1.7B on a SINGLE rented GPU
#  Target: RTX 5090 (32 GB, sm_120) on Linux.
#
#  This is the paper's configuration (scripts/run_opsd_1b.sh) moved onto one GPU:
#    * the paper used 4xH100, per-device batch 4, grad-accum 2  -> effective 32
#      here:      1 GPU,      per-device batch 4, grad-accum 8  -> effective 32
#    * vLLM colocate generation (same as the paper), 45 % of the 32 GB for vLLM
#    * LoRA r=64 / alpha=128 on all linear projections, fixed teacher (step-0 policy)
#    * forward KL (beta=0) with per-element JSD clipping (0.05), rollouts <= 1024 tok
#    * lr 5e-6, max_grad_norm 0.1, temperature 1.1 / top_p 0.95 / top_k 20
#    * 50 optimizer steps, checkpoint every 10 steps
#
#  Usage:
#    bash scripts/run_opsd_1b_5090.sh                       # 50 steps, saves every 10
#    MAX_STEPS=100 bash scripts/run_opsd_1b_5090.sh         # more steps
#    REPORT_TO=wandb bash scripts/run_opsd_1b_5090.sh       # needs `wandb login`
#
#  Launched with torchrun so that torch.distributed is initialised: vLLM's
#  "external_launcher" backend (the upstream default) expects an existing
#  process group.  With VLLM_BACKEND=mp the script runs plain python instead.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/runs}"
RUN_CONFIG="${RUN_CONFIG:-qwen31b_5090_gen1024_fixteacher}"
MAX_STEPS="${MAX_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-10}"
MAX_COMPLETION="${MAX_COMPLETION:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"          # 4 * 8 = 32 = the paper's effective batch
LORA_R="${LORA_R:-64}"
LR="${LR:-5e-6}"
REPORT_TO="${REPORT_TO:-none}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"         # FA2 has no sm_120 kernels; sdpa is equivalent
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.45}" # 32 GB card: leave room for the training model
JSD_CHUNK_SIZE="${JSD_CHUNK_SIZE:-64}" # chunked loss = identical math, much less VRAM
VLLM_BACKEND="${VLLM_BACKEND:-external_launcher}"
USE_VLLM="${USE_VLLM:-1}"                # 0 -> plain HF generate (no vLLM needed)
MASTER_PORT="${MASTER_PORT:-29511}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "=============================================================="
echo " OPSD / Qwen3-1.7B  |  model=$MODEL_PATH  steps=$MAX_STEPS"
echo " batch=$BATCH_SIZE x accum=$GRAD_ACCUM (effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo " rollout<=$MAX_COMPLETION  lr=$LR  attn=$ATTN_IMPL  vllm_mem=$VLLM_MEM_UTIL"
echo " output: $OUTPUT_DIR/$RUN_CONFIG"
echo "=============================================================="

TRAIN_ARGS=(
    opsd_train.py
    --model_name_or_path "$MODEL_PATH"
    --learning_rate "$LR"
    --max_grad_norm 0.1
    --per_device_train_batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --gradient_checkpointing
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --run_config "$RUN_CONFIG"
    --max_steps "$MAX_STEPS"
    --max_completion_length "$MAX_COMPLETION"
    --save_steps "$SAVE_STEPS"
    --logging_steps 1
    --attn_implementation "$ATTN_IMPL"
    --torch_dtype bfloat16
    --max_length 20000
    --beta 0
    --lmbda 1
    --use_peft
    --lora_r "$LORA_R"
    --lora_alpha "$((LORA_R * 2))"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --temperature 1.1
    --top_p 0.95
    --top_k 20
    --fixed_teacher
    --jsd_token_clip 0.05
    --jsd_chunk_size "$JSD_CHUNK_SIZE"
    --report_to "$REPORT_TO"
    --seed 42
)

if [ "$USE_VLLM" = "1" ]; then
    TRAIN_ARGS+=(
        --use_vllm
        --vllm_mode colocate
        --vllm_gpu_memory_utilization "$VLLM_MEM_UTIL"
        --vllm_tensor_parallel_size 1
        --vllm_distributed_executor_backend "$VLLM_BACKEND"
    )
fi

if [ "$VLLM_BACKEND" = "external_launcher" ]; then
    # torchrun initialises the process group that vLLM's external_launcher needs
    torchrun --nproc_per_node 1 --master_port "$MASTER_PORT" "${TRAIN_ARGS[@]}"
else
    python "${TRAIN_ARGS[@]}"
fi

echo ""
echo "done. checkpoints in $OUTPUT_DIR/$RUN_CONFIG/checkpoint-*"
