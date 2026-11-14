#!/usr/bin/env bash
# =============================================================================
#  Environment + assets for the OPSD Qwen3-1.7B reproduction on a rented GPU
#  (RTX 5090 / 32 GB / Linux).  Run once per rental.
#
#    bash scripts/setup_rental_5090.sh
#
#  Creates the conda env from environment.yml, adds the pieces that the repo
#  needs on top of it (CUDA-12.8 torch for Blackwell, vLLM, flash-attn) and
#  downloads the base model + the OPSD training set.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-1.7B}"
DATASET_ID="${DATASET_ID:-siyanzhao/Openthoughts_math_30k_opsd}"

echo "### 1/4  conda environment"
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found -- install Miniconda first (https://docs.conda.io/en/latest/miniconda.html)"
    exit 1
fi
conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml --prune
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opsd

echo "### 2/4  CUDA 12.8 build of torch (Blackwell / sm_120 needs cu128)"
# the pytorch index serves the +cu128 build, PyPI still serves the nvidia-* runtime deps
pip install --upgrade "torch==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
assert torch.version.cuda == "12.8", "expected a CUDA 12.8 build for sm_120 (RTX 5090)"
PY

echo "### 3/4  vLLM + flash-attn (optional)"
pip install "vllm==0.11.0"
# flash-attn has no sm_120 kernels; the trainer falls back to SDPA automatically.
pip install flash-attn==2.8.3 --no-build-isolation || \
    echo "!! flash-attn build failed - continuing, SDPA will be used instead"

echo "### 4/4  model + dataset"
mkdir -p "$(dirname "$MODEL_PATH")"
hf download "$MODEL_ID" --local-dir "$MODEL_PATH"

python - <<'PY'
from datasets import load_dataset
ds = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
for split, d in ds.items():
    print(f"{split}: {len(d)} rows, columns={d.column_names}")
PY

cat <<'EOF'

=====================================================================
 ready.  run the training with:

   bash scripts/run_opsd_1b_5090.sh

 (set MODEL_PATH=$HOME/models/Qwen3-1.7B if you used a different path)

 useful overrides:
   MAX_STEPS=100            more optimizer steps
   MAX_COMPLETION=512       half the rollout length (~2x faster)
   BATCH_SIZE=8             larger per-device batch (needs more VRAM)
   GRAD_ACCUM=4             keep effective batch 32 with BATCH_SIZE=8
   REPORT_TO=wandb          needs `wandb login`
=====================================================================
EOF
