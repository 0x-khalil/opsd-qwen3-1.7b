#!/usr/bin/env bash
# =============================================================================
#  Evaluate the OPSD checkpoints on a SINGLE rented GPU (RTX 5090, 32 GB).
#
#    bash eval/run_eval_5090.sh                 # base + every checkpoint
#    DATASETS="aime24" bash eval/run_eval_5090.sh
#
#  Uses the upstream vLLM evaluation (thinking mode, Avg@12 like the paper).
#  The LoRA checkpoints are fed to vLLM through --enable_lora (max_lora_rank=64),
#  so no merging step is needed.
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen3-1.7B}"
EXP_DIR="${EXP_DIR:-$HOME/runs/qwen31b_5090_gen1024_fixteacher}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25}"
# which checkpoints to evaluate (all of them is expensive!)
EVAL_STEPS="${EVAL_STEPS:-50}"
# 1 = also evaluate the step-0 base model (the reference point)
EVAL_BASE="${EVAL_BASE:-1}"
VAL_N="${VAL_N:-12}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-38912}"
TP="${TP:-1}"
GPU_MEM="${GPU_MEM:-0.85}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/eval_results}"
mkdir -p "$OUT_DIR"

run_eval () {   # $1 = dataset, $2 = extra args (adapter dir or empty)
    local ds="$1"; shift
    echo "=================================================================="
    echo " evaluating $ds  $*"
    echo "=================================================================="
    python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "$ds" \
        --val_n "$VAL_N" \
        --temperature "$TEMPERATURE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --tensor_parallel_size "$TP" \
        --gpu_memory_utilization "$GPU_MEM" \
        --enable_thinking \
        --output_file "$OUT_DIR/${ds}_$(echo "$*" | tr -dc '[:alnum:]').json" \
        "$@"
}

# 1) base model (step 0 reference)
if [ "$EVAL_BASE" = "1" ]; then
    for ds in $DATASETS; do
        run_eval "$ds"
    done
fi

# 2) the requested checkpoints of the run
for step in $EVAL_STEPS; do
    ckpt="$EXP_DIR/checkpoint-$step"
    if [ ! -d "$ckpt" ]; then
        echo "skip $ckpt (missing)"
        continue
    fi
    for ds in $DATASETS; do
        run_eval "$ds" --checkpoint_dir "$ckpt"
    done
done

echo ""
echo "results written to $OUT_DIR"
