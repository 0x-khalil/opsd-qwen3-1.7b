#!/usr/bin/env bash
# =============================================================================
#  Run the full OPSD pipeline on a Lium pod (RTX 5090).
#
#  One-liner from your own machine (nothing to copy over):
#
#      lium exec opsd "curl -fsSL https://raw.githubusercontent.com/0x-khalil/opsd-qwen3-1.7b/main/scripts/run_on_lium.sh | bash"
#
#  It expects the HF token to already be on the pod:
#
#      lium exec opsd "mkdir -p ~/.config/opsd && printf %s hf_xxx > ~/.config/opsd/hf_token && chmod 600 ~/.config/opsd/hf_token"
#
#  Optional: export HF_REPO_ID=<user>/<repo> on the pod to change the target repo.
# =============================================================================
set -uo pipefail

REPO_URL="${REPO_URL:-https://github.com/0x-khalil/opsd-qwen3-1.7b.git}"
REPO_DIR="${REPO_DIR:-/root/opsd-qwen3-1.7b}"
export HF_REPO_ID="${HF_REPO_ID:-0x-khalil/opsd-qwen3-1.7b}"

echo "### lium runner | repo=$REPO_DIR | target=$HF_REPO_ID"

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR" || { echo "git clone failed"; exit 1; }
fi
cd "$REPO_DIR"

TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/opsd/hf_token}"
if [ ! -s "$TOKEN_FILE" ]; then
    echo "ERROR: no HF token at $TOKEN_FILE"
    echo "create it with:"
    echo "  mkdir -p ~/.config/opsd && printf %s hf_xxx > ~/.config/opsd/hf_token && chmod 600 ~/.config/opsd/hf_token"
    exit 1
fi

echo "### GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "### disk"
df -h / | tail -1

exec bash scripts/bootstrap_rental.sh
