#!/usr/bin/env bash
# =============================================================================
#  ONE-SHOT pipeline for a rented GPU (RTX 5090, 32 GB, Linux):
#
#      install everything  ->  download model + data  ->  train 50 steps
#      ->  merge + pack  ->  upload everything to the Hugging Face Hub
#
#  Run it on the rental box:
#
#      bash scripts/bootstrap_rental.sh
#
#  with (at least) these two variables set:
#
#      HF_TOKEN=hf_xxxxxxxxxxxx     # huggingface.co/settings/tokens  (write access)
#      HF_REPO_ID=your-user/opsd-qwen3-1.7b
#
#  Optional:
#      MAX_STEPS=50            optimizer steps
#      RUN_EVAL=0              set to 1 to also run the vLLM evaluation (+~2 h)
#      UPLOAD_MERGED=1         upload the merged full model next to the adapters
#      SMOKE=1                 1-step dry run of the whole pipeline (~10 min)
#      KEEP_VENV=0             delete the venv at the end (frees ~20 GB)
#
#  Everything is logged to $LOG_DIR/bootstrap.log and the run is resumable:
#  if a checkpoint exists, training continues from it.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------- settings ---
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
# WORK_DIR holds the venv + the model + the run (put it on the fast disk, e.g.
# /workspace on Lium pods; /root there is an encrypted FUSE mount)
WORK_DIR="${WORK_DIR:-$HOME}"
MODEL_DIR="${MODEL_DIR:-$WORK_DIR/models/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/runs}"
RUN_CONFIG="${RUN_CONFIG:-qwen31b_5090_gen1024_fixteacher}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_COMPLETION="${MAX_COMPLETION:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
VLLM_MEM_UTIL="${VLLM_MEM_UTIL:-0.45}"
JSD_CHUNK_SIZE="${JSD_CHUNK_SIZE:-64}"

HF_TOKEN="${HF_TOKEN:-}"
HF_REPO_ID="${HF_REPO_ID:-}"
HF_REPO_TYPE="${HF_REPO_TYPE:-model}"
HF_PRIVATE="${HF_PRIVATE:-1}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/opsd/hf_token}"

RUN_EVAL="${RUN_EVAL:-0}"
UPLOAD_MERGED="${UPLOAD_MERGED:-1}"
SMOKE="${SMOKE:-0}"
KEEP_VENV="${KEEP_VENV:-0}"

VENV="${VENV:-$WORK_DIR/venv}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
RUN_DIR="$OUTPUT_DIR/$RUN_CONFIG"

if [ "$SMOKE" = "1" ]; then
    MAX_STEPS=1
    SAVE_STEPS=1
    RUN_CONFIG="${RUN_CONFIG}_smoke"
    RUN_DIR="$OUTPUT_DIR/$RUN_CONFIG"
else
    SAVE_STEPS="${SAVE_STEPS:-10}"
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bootstrap.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# the token can also live in a file (keeps it out of the shell history)
if [ -z "$HF_TOKEN" ] && [ -f "$HF_TOKEN_FILE" ]; then
    HF_TOKEN="$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")"
    log "using the token from $HF_TOKEN_FILE"
fi

# some images (Lium, ...) pin torch via /etc/pip/constraints.txt -> we build our
# own isolated venv, so the pin must not apply to it
if [ -f /etc/pip/constraints.txt ]; then
    export PIP_CONSTRAINT=/dev/null
    log "bypassing /etc/pip/constraints.txt for our private venv"
fi

log "=============================================================="
log " OPSD bootstrap | steps=$MAX_STEPS | hf_repo=$HF_REPO_ID"
log "=============================================================="

# ------------------------------------------------------------------ 0. GPU ---
log ">>> 0/6  GPU check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "FATAL: nvidia-smi not found - this image has no GPU/driver"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
if ! nvidia-smi | grep -q "5090"; then
    log "WARNING: this does not look like an RTX 5090 - the run may be too slow / OOM"
fi

# ---------------------------------------------------------------- 1. python --
log ">>> 1/6  python environment"
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        v=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
        major=${v%%.*}; minor=${v##*.}
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    log "installing python3.10+ with apt"
    (apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip) \
        || (sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip)
    PY=python3
fi
log "using $($PY -V)"

if [ ! -d "$VENV" ]; then
    log "creating venv at $VENV"
    $PY -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
log "venv: $VENV"

# ------------------------------------------------------------------ 2. deps --
log ">>> 2/6  pip packages (torch cu128 for Blackwell / sm_120)"
pip install -q --upgrade pip wheel setuptools
pip install -q "torch==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
pip install -q "transformers==4.57.1" "accelerate==1.11.0" "trl==0.26.0" \
    "datasets==3.6.0" "peft==0.17.1" "wandb==0.22.3" "math-verify==0.8.0" \
    "safetensors==0.5.3" "tiktoken==0.9.0" "einops==0.8.1" \
    "huggingface_hub[cli]" "vllm==0.11.0"
# flash-attn has no sm_120 kernels -> the trainer falls back to SDPA automatically
pip install -q flash-attn==2.8.3 --no-build-isolation 2>/dev/null \
    || log "flash-attn not built (fine: SDPA will be used)"

python - <<'PY' || exit 1
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
assert torch.cuda.is_available(), "no CUDA device visible"
assert torch.version.cuda == "12.8", "need a CUDA 12.8 build for sm_120 (RTX 5090)"
PY

# ---------------------------------------------------------------- 3. assets --
log ">>> 3/6  model + dataset"
mkdir -p "$(dirname "$MODEL_DIR")"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR" >>"$LOG" 2>&1 || \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR')" >>"$LOG" 2>&1
python - <<'PY' >>"$LOG" 2>&1
from datasets import load_dataset
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
for split, d in ds.items():
    print(split, len(d), d.column_names)
PY

# ---------------------------------------------------------------- 4. train --
log ">>> 4/6  training ($MAX_STEPS steps, effective batch $((BATCH_SIZE * GRAD_ACCUM)))"
RESUME=""
LATEST=$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST" ]; then
    log "found an existing checkpoint -> resuming from $LATEST"
    RESUME="$LATEST"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="$MODEL_DIR" \
OUTPUT_DIR="$OUTPUT_DIR" \
RUN_CONFIG="$RUN_CONFIG" \
MAX_STEPS="$MAX_STEPS" \
SAVE_STEPS="$SAVE_STEPS" \
MAX_COMPLETION="$MAX_COMPLETION" \
BATCH_SIZE="$BATCH_SIZE" \
GRAD_ACCUM="$GRAD_ACCUM" \
VLLM_MEM_UTIL="$VLLM_MEM_UTIL" \
JSD_CHUNK_SIZE="$JSD_CHUNK_SIZE" \
RESUME_FROM_CHECKPOINT="$RESUME" \
bash scripts/run_opsd_1b_5090.sh 2>&1 | tee -a "$LOG"

TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" != "0" ]; then
    log "training failed (exit $TRAIN_RC) - see $LOG"
    exit "$TRAIN_RC"
fi
log "training finished"

# ------------------------------------------------------- 5. pack + publish ---
log ">>> 5/6  merge + package"
FINAL_ADAPTER="$RUN_DIR"
if [ -d "$RUN_DIR/checkpoint-$MAX_STEPS" ]; then
    FINAL_ADAPTER="$RUN_DIR/checkpoint-$MAX_STEPS"
fi
MERGED_DIR="$OUTPUT_DIR/merged-$(basename "$RUN_CONFIG")"

if [ "$UPLOAD_MERGED" = "1" ]; then
    log "merging $FINAL_ADAPTER into the base model -> $MERGED_DIR"
    python scripts/merge_lora.py --base_model "$MODEL_DIR" \
        --adapter "$FINAL_ADAPTER" --output "$MERGED_DIR" >>"$LOG" 2>&1
fi

PKG_DIR="$OUTPUT_DIR/$(basename "$RUN_CONFIG")_package"
python scripts/package_artifacts.py --run_dir "$RUN_DIR" \
    --base_model "$MODEL_ID" --out "$PKG_DIR" >>"$LOG" 2>&1

log ">>> 6/6  upload to the Hugging Face Hub"
if [ -z "$HF_TOKEN" ] || [ -z "$HF_REPO_ID" ]; then
    log "HF_TOKEN and/or HF_REPO_ID are not set -> SKIPPING the upload."
    log "Artifacts are on the box under: $OUTPUT_DIR"
    log "Upload them later with:"
    log "  python scripts/upload_to_hf.py --repo_id <user>/<repo> --paths $PKG_DIR $MERGED_DIR"
else
    python scripts/upload_to_hf.py --repo_id "$HF_REPO_ID" --repo_type "$HF_REPO_TYPE" \
        --token "$HF_TOKEN" --private "$HF_PRIVATE" \
        --paths "$PKG_DIR" $([ "$UPLOAD_MERGED" = "1" ] && echo "$MERGED_DIR") 2>&1 | tee -a "$LOG"
    if [ ${PIPESTATUS[0]} = "0" ]; then
        log "published: https://huggingface.co/$HF_REPO_ID"
    else
        log "UPLOAD FAILED - the artifacts are still on the pod, just retry with:"
        log "  source $VENV/bin/activate"
        log "  export HF_TOKEN=\$(tr -d ' \\\\t\\\\r\\\\n' < $HF_TOKEN_FILE)"
        log "  python scripts/upload_to_hf.py --repo_id $HF_REPO_ID --repo_type $HF_REPO_TYPE \\"
        log "      --token \"\$HF_TOKEN\" --private $HF_PRIVATE --paths $PKG_DIR $([ "$UPLOAD_MERGED" = "1" ] && echo "$MERGED_DIR")"
    fi
fi

# --------------------------------------------------------------- 6.5 eval ---
if [ "$RUN_EVAL" = "1" ]; then
    log ">>> extra: evaluation (AIME24/25 + HMMT25)"
    EXP_DIR="$RUN_DIR" BASE_MODEL="$MODEL_DIR" \
    EVAL_STEPS="${EVAL_STEPS:-50}" EVAL_BASE="${EVAL_BASE:-1}" \
    DATASETS="${EVAL_DATASETS:-aime24 aime25 hmmt25}" \
        bash eval/run_eval_5090.sh 2>&1 | tee -a "$LOG"
fi

log "=============================================================="
log " ALL DONE.  artifacts:"
log "   adapters : $RUN_DIR"
log "   package  : $PKG_DIR"
log "   merged   : $MERGED_DIR"
log "   log      : $LOG"
log "=============================================================="

if [ "$KEEP_VENV" = "0" ]; then
    log "removing the venv to free disk space (set KEEP_VENV=1 to keep it)"
    rm -rf "$VENV"
fi
