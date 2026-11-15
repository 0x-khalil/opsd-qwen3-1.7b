#!/usr/bin/env bash
# =============================================================================
#  Evaluate the OPSD Qwen3-1.7B checkpoints on a rented RTX 5090 (32 GB) and
#  push the results to the Hugging Face Hub.
#
#  One command on the rental box:
#
#      HF_TOKEN=hf_xxx bash scripts/eval_5090.sh
#
#  Knobs (env vars):
#      EVAL_DATASETS="aime24"        which datasets
#      EVAL_STEPS="50"               which checkpoints (50 = the final one)
#      EVAL_VAL_N=12                samples per problem (paper: Avg@12)
#      EVAL_MAX_NEW_TOKENS=16384    thinking-mode cap
#      WORK_DIR=/workspace/opsd      where the venv/models/results live
#
#  Why a 5090 and not a Kaggle T4: on sm_75 (T4) vLLM can only use the Triton
#  attention backend, which torch.compile recompiles for every batch shape ->
#  the GPU spends its time compiling instead of decoding. A 5090 (sm_120) is
#  ~8x faster and compiles in seconds.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${HF_REPO:-0x-Khalil/opsd-qwen3-1.7b}"
BASE_ID="${BASE_ID:-Qwen/Qwen3-1.7B}"
PKG="qwen31b_5090_gen1024_fixteacher_package"
WORK_DIR="${WORK_DIR:-/workspace/opsd}"
EVAL_DATASETS="${EVAL_DATASETS:-aime24}"
EVAL_STEPS="${EVAL_STEPS:-50}"
EVAL_VAL_N="${EVAL_VAL_N:-12}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-16384}"

MODEL_DIR="$WORK_DIR/models/Qwen3-1.7B"
ADAPTER_DIR="$WORK_DIR/models/pkg/$PKG"
RESULTS="$WORK_DIR/eval_results"
LOGS="$WORK_DIR/eval_logs"
VENV="$WORK_DIR/venv"

mkdir -p "$WORK_DIR" "$RESULTS" "$LOGS"
export WORK_DIR HF_REPO RESULTS
log() { echo "[$(date +%H:%M:%S)] $*"; }

# ----------------------------------------------------------------- 0. env ---
log "checking the GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    log "FATAL: no GPU visible"; exit 1; }

if [ -f /etc/pip/constraints.txt ]; then
    export PIP_CONSTRAINT=/dev/null   # our private venv must not follow the image pin
fi

# ------------------------------------------------------------- 1. packages ---
if [ ! -x "$VENV/bin/python" ]; then
    log "creating the venv (torch cu128 + vllm 0.11 for Blackwell)"
    python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
if ! python -c "import vllm" >/dev/null 2>&1; then
    pip install -q --upgrade pip wheel
    pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple
    pip install -q "vllm==0.11.0" "transformers==4.57.1" "tokenizers>=0.22,<0.23" \
        "huggingface_hub<1.0" "math-verify==0.8.0"
fi
python -c "import torch,vllm,transformers;print('torch',torch.__version__,'| vllm',vllm.__version__,'| transformers',transformers.__version__,'| cuda',torch.version.cuda)"

# ---------------------------------------------------------------- 2. code ---
if [ ! -f "$REPO_ROOT/eval/evaluate_math.py" ]; then
    log "cloning the repo"
    git clone --depth 1 https://github.com/0x-khalil/opsd-qwen3-1.7b "$WORK_DIR/repo"
    REPO_ROOT="$WORK_DIR/repo"
    cd "$REPO_ROOT"
fi

# -------------------------------------------------------------- 3. models ---
python - <<'PY'
import os
from huggingface_hub import snapshot_download
work = os.environ.get("WORK_DIR", "/workspace/opsd")
pkg = "qwen31b_5090_gen1024_fixteacher_package"
print("downloading the base model ...", flush=True)
snapshot_download("Qwen/Qwen3-1.7B", local_dir=f"{work}/models/Qwen3-1.7B",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=8)
print("downloading the adapters ...", flush=True)
snapshot_download(os.environ.get("HF_REPO", "0x-Khalil/opsd-qwen3-1.7b"),
                  local_dir=f"{work}/models/pkg", allow_patterns=[f"{pkg}/*"], max_workers=8)
PY

# ---------------------------------------------------------------- 4. eval ---
run_one() {   # $1 dataset, $2 checkpoint step ("base" for the base model)
    local ds="$1" step="$2"
    local out="$RESULTS/${ds}_${step}.json"
    if [ -s "$out" ]; then log "skip $ds/$step (done)"; return 0; fi
    local args=(--base_model "$MODEL_DIR" --dataset "$ds"
                --val_n "$EVAL_VAL_N" --temperature 1.0 --top_p 1.0 --top_k -1
                --min_p 0.0 --presence_penalty 0.0
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --tensor_parallel_size 1
                --gpu_memory_utilization 0.80 --enable_thinking --output_file "$out")
    if [ "$step" != "base" ]; then
        args+=(--checkpoint_dir "$ADAPTER_DIR/checkpoint-$step")
    fi
    log "evaluating $ds / $step"
    python eval/evaluate_math.py "${args[@]}" >"$LOGS/${ds}_${step}.log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        log "  !! failed (rc=$rc), tail:"
        tail -12 "$LOGS/${ds}_${step}.log"
    fi
    return $rc
}

for ds in $EVAL_DATASETS; do run_one "$ds" base; done
for step in $EVAL_STEPS; do
    for ds in $EVAL_DATASETS; do run_one "$ds" "$step"; done
done

# ------------------------------------------------------------ 5. summary ---
python - <<'PY'
import glob, json, os
results = os.environ.get("RESULTS", "/workspace/opsd/eval_results")
rows = []
for f in sorted(glob.glob(os.path.join(results, "*.json"))):
    if f.endswith("summary.json"):
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    rows.append((os.path.basename(f)[:-5], d.get("average_at_n_pct"),
                 d.get("pass_at_n_pct"), d.get("majority_vote_at_n_pct"), d.get("format_rate")))
if rows:
    lines = ["| eval | Avg@12 | Pass@12 | Maj@12 | format |", "|---|---|---|---|---|"]
    for name, a, p, m, fr in rows:
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, "%.2f%%" % a if a is not None else "-",
            "%.2f%%" % p if p is not None else "-",
            "%.2f%%" % m if m is not None else "-",
            "%.1f%%" % fr if fr is not None else "-"))
    open(os.path.join(results, "summary.md"), "w", encoding="utf-8").write(
        "# OPSD Qwen3-1.7B evaluation (RTX 5090)\n\n" + "\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(results, "summary.json"), "w"), indent=2)
    print("\n".join(lines))
PY

# ------------------------------------------------------------- 6. upload ---
if [ -n "${HF_TOKEN:-}" ]; then
    python - <<'PY'
import os
from huggingface_hub import HfApi
work = os.environ.get("WORK_DIR", "/workspace/opsd")
repo = os.environ.get("HF_REPO", "0x-Khalil/opsd-qwen3-1.7b")
api = HfApi(token=os.environ["HF_TOKEN"])
for src, dst in ((f"{work}/eval_results", "eval_results"), (f"{work}/eval_logs", "eval_results/logs")):
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", path_in_repo=dst,
                          commit_message="OPSD Qwen3-1.7B eval results (RTX 5090)")
print("uploaded -> https://huggingface.co/" + repo + "/tree/main/eval_results")
PY
else
    log "HF_TOKEN not set -> results stay in $RESULTS"
fi
