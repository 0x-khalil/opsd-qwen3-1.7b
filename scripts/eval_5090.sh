#!/usr/bin/env bash
# =============================================================================
#  Evaluate the OPSD Qwen3-1.7B run on a rented RTX 5090 (32 GB) and publish
#  the results to the Hugging Face Hub.
#
#  Default evaluation matrix
#  ------------------------
#    base model      -> aime24, aime25, hmmt25      (the reference)
#    step-50         -> aime24, aime25, hmmt25      (the final model, all 3 sets)
#    step-10/20/30/40 -> aime24                     (the checkpoint sweep)
#
#  = 10 evaluations, ~10-15 min each on a 5090  ->  ~2 h  (~$1-2)
#
#  One command on the rental box:
#
#      HF_TOKEN=hf_xxx bash scripts/eval_5090.sh
#
#  Knobs (environment variables):
#      EVAL_MAIN_DATASETS="aime24 aime25 hmmt25"   sets for the final model
#      EVAL_MAIN_STEPS="50"                        checkpoints for the final model
#      EVAL_ALL_STEPS="10 20 30 40"                checkpoint sweep (first dataset)
#      EVAL_BASE_DATASETS="aime24 aime25 hmmt25"   sets for the base reference
#      EVAL_VAL_N=12                               samples per problem (Avg@12)
#      EVAL_MAX_NEW_TOKENS=16384                   thinking-mode cap
#      WORK_DIR=/workspace/opsd                     venv / models / results
#      HF_REPO=0x-Khalil/opsd-qwen3-1.7b            target repo
#
#  Resumable: finished evaluations are skipped, and every result is uploaded
#  to the Hub immediately, so a lost session never loses work.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

HF_REPO="${HF_REPO:-0x-Khalil/opsd-qwen3-1.7b}"
BASE_ID="${BASE_ID:-Qwen/Qwen3-1.7B}"
PKG="qwen31b_5090_gen1024_fixteacher_package"

WORK_DIR="${WORK_DIR:-/workspace/opsd}"
EVAL_MAIN_DATASETS="${EVAL_MAIN_DATASETS:-aime24 aime25 hmmt25}"
EVAL_MAIN_STEPS="${EVAL_MAIN_STEPS:-50}"
EVAL_ALL_STEPS="${EVAL_ALL_STEPS:-10 20 30 40}"
EVAL_BASE_DATASETS="${EVAL_BASE_DATASETS:-$EVAL_MAIN_DATASETS}"
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

DRY_RUN="${DRY_RUN:-0}"

# ----------------------------------------------------------------- 0. env ---
if [ "$DRY_RUN" != "1" ]; then
    log "GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
        log "FATAL: no GPU visible"; exit 1; }

    if [ -f /etc/pip/constraints.txt ]; then
        export PIP_CONSTRAINT=/dev/null   # our private venv must not follow the image pin
    fi
fi

# ------------------------------------------------------------- 1. packages ---
if [ "$DRY_RUN" != "1" ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        log "creating the venv"
        python3 -m venv "$VENV"
    fi
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    if ! python -c "import vllm" >/dev/null 2>&1; then
        log "installing torch cu128 + vllm 0.11 + transformers 4.57.1 (a few minutes)"
        pip install -q --upgrade pip wheel
        pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128 \
            --extra-index-url https://pypi.org/simple
        pip install -q "vllm==0.11.0" "transformers==4.57.1" "tokenizers>=0.22,<0.23" \
            "huggingface_hub<1.0" "math-verify==0.8.0"
    fi
    python -c "import torch,vllm,transformers;print('torch',torch.__version__,'| vllm',vllm.__version__,'| transformers',transformers.__version__,'| cuda',torch.version.cuda)"
fi

# ---------------------------------------------------------------- 2. code ---
if [ "$DRY_RUN" != "1" ] && [ ! -f "$REPO_ROOT/eval/evaluate_math.py" ]; then
    log "cloning the repo"
    git clone --depth 1 https://github.com/0x-khalil/opsd-qwen3-1.7b "$WORK_DIR/repo"
    REPO_ROOT="$WORK_DIR/repo"
    cd "$REPO_ROOT"
fi

# -------------------------------------------------------------- 3. models ---
if [ "$DRY_RUN" != "1" ]; then
python - <<'PY'
import os
from huggingface_hub import snapshot_download

work = os.environ.get("WORK_DIR", "/workspace/opsd")
pkg = "qwen31b_5090_gen1024_fixteacher_package"
print("downloading the base model ...", flush=True)
snapshot_download("Qwen/Qwen3-1.7B", local_dir=f"{work}/models/Qwen3-1.7B",
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=8)
print("downloading the OPSD adapters ...", flush=True)
snapshot_download(os.environ.get("HF_REPO", "0x-Khalil/opsd-qwen3-1.7b"),
                  local_dir=f"{work}/models/pkg", allow_patterns=[f"{pkg}/*"], max_workers=8)
PY
fi

# --------------------------------------------------------------- 4. evals ---
upload_results() {
    [ -n "${HF_TOKEN:-}" ] || return 0
    python - <<'PY' 2>/dev/null || echo "  (upload failed, will retry later)"
import os
from huggingface_hub import HfApi
work = os.environ.get("WORK_DIR", "/workspace/opsd")
repo = os.environ.get("HF_REPO", "0x-Khalil/opsd-qwen3-1.7b")
api = HfApi(token=os.environ["HF_TOKEN"])
for src, dst in ((f"{work}/eval_results", "eval_results"), (f"{work}/eval_logs", "eval_results/logs")):
    if os.path.isdir(src):
        api.upload_folder(folder_path=src, repo_id=repo, repo_type="model", path_in_repo=dst,
                          commit_message="OPSD Qwen3-1.7B eval results (RTX 5090)")
print("  uploaded the current results to the Hub")
PY
}

run_one() {   # $1 = dataset, $2 = checkpoint step ("base" for the base model)
    local ds="$1" step="$2"
    local tag="$ds"
    [ "$step" = "base" ] || tag="${ds}_step${step}"
    local out="$RESULTS/${tag}.json"

    if [ -s "$out" ]; then
        log "skip $tag (already done)"
        return 0
    fi

    local args=(--base_model "$MODEL_DIR" --dataset "$ds"
                --val_n "$EVAL_VAL_N" --temperature 1.0 --top_p 1.0 --top_k -1
                --min_p 0.0 --presence_penalty 0.0
                --max_new_tokens "$EVAL_MAX_NEW_TOKENS" --tensor_parallel_size 1
                --gpu_memory_utilization 0.80 --enable_thinking --output_file "$out")
    if [ "$step" != "base" ]; then
        if [ "$DRY_RUN" != "1" ] && [ ! -f "$ADAPTER_DIR/checkpoint-$step/adapter_model.safetensors" ]; then
            log "skip $tag (checkpoint-$step not found)"
            return 0
        fi
        args+=(--checkpoint_dir "$ADAPTER_DIR/checkpoint-$step")
    fi

    log "=== $tag  (val_n=$EVAL_VAL_N, max_new_tokens=$EVAL_MAX_NEW_TOKENS) ==="
    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY RUN: python eval/evaluate_math.py ${args[*]}"
        return 0
    fi
    local t0=$SECONDS
    python eval/evaluate_math.py "${args[@]}" >"$LOGS/${tag}.log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))
    if [ $rc -ne 0 ]; then
        log "  !! $tag FAILED (rc=$rc), tail:"
        tail -12 "$LOGS/${tag}.log"
        return $rc
    fi
    log "  $tag done in $(( dt / 60 ))m$(( dt % 60 ))s"
    python - <<'PY'
import glob, json, os
res = os.environ.get("RESULTS", "/workspace/opsd/eval_results")
for f in sorted(glob.glob(os.path.join(res, "*.json"))):
    if f.endswith("summary.json"):
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if isinstance(d, dict) and d.get("average_at_n_pct") is not None:
        print("    %-24s Avg@%s = %.2f%%  (Pass@%s %.2f%%, Maj@%s %.2f%%)" % (
            os.path.basename(f)[:-5], d.get("val_n"), d["average_at_n_pct"],
            d.get("val_n"), d.get("pass_at_n_pct") or 0.0,
            d.get("val_n"), d.get("majority_vote_at_n_pct") or 0.0))
PY
    upload_results
}

# --- the matrix ------------------------------------------------------------
# 1) base reference on every dataset
for ds in $EVAL_BASE_DATASETS; do run_one "$ds" base; done
# 2) the final model on every dataset
for step in $EVAL_MAIN_STEPS; do
    for ds in $EVAL_MAIN_DATASETS; do run_one "$ds" "$step"; done
done
# 3) every remaining checkpoint on the first dataset only
SWEEP_DATASET="${EVAL_MAIN_DATASETS%% *}"
for step in $EVAL_ALL_STEPS; do run_one "$SWEEP_DATASET" "$step"; done

# ------------------------------------------------------------- 5. summary ---
if [ "$DRY_RUN" != "1" ]; then
log "building the summary"
python - <<'PY'
import glob, json, os
res = os.environ.get("RESULTS", "/workspace/opsd/eval_results")
rows = []
for f in sorted(glob.glob(os.path.join(res, "*.json"))):
    if f.endswith("summary.json"):
        continue
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    rows.append((os.path.basename(f)[:-5], d.get("dataset"), d.get("val_n"),
                 d.get("average_at_n_pct"), d.get("pass_at_n_pct"),
                 d.get("majority_vote_at_n_pct"), d.get("format_rate")))
if rows:
    rows.sort(key=lambda r: (str(r[1]), str(r[0])))
    lines = ["| eval | dataset | Avg@n | Pass@n | Maj@n | format |", "|---|---|---|---|---|---|---|"]
    for name, ds, n, a, p, m, fr in rows:
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            name, ds,
            "%.2f%%" % a if a is not None else "-",
            "%.2f%%" % p if p is not None else "-",
            "%.2f%%" % m if m is not None else "-",
            "%.1f%%" % fr if fr is not None else "-"))
    open(os.path.join(res, "summary.md"), "w", encoding="utf-8").write(
        "# OPSD Qwen3-1.7B evaluation (RTX 5090, 32 GB)\n\n"
        "Protocol: thinking mode, temperature 1.0, top_p 1.0, top_k off, min_p 0, "
        "graded with math-verify.\n\n" + "\n".join(lines) + "\n")
    json.dump(rows, open(os.path.join(res, "summary.json"), "w"), indent=2)
    print("\n".join(lines))
PY
fi

# ------------------------------------------------------------- 6. upload ---
if [ "$DRY_RUN" != "1" ]; then
    upload_results
    if [ -n "${HF_TOKEN:-}" ]; then
        log "done -> https://huggingface.co/$HF_REPO/tree/main/eval_results"
    else
        log "HF_TOKEN was not set -> the results stay in $RESULTS"
    fi
fi
