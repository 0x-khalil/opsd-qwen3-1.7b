# One-shot environment setup for the OPSD reproduction on Windows (single RTX 3070 8 GB).
#
#   powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
#
# Notes
#  * Python 3.13 is used (3.10/3.11 also fine). vLLM, DeepSpeed, flash-attn and
#    xformers have no Windows wheels, so this setup uses HF `generate` + SDPA instead.
#  * The CUDA build of torch is pulled from the PyTorch index (the PyPI wheel for
#    torch 2.8.0 resolves to the CPU build on some mirrors).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtualenv at $Venv"
    python -m venv $Venv
}
$py = Join-Path $Venv "Scripts\python.exe"

& $py -m pip install --upgrade pip
& $py -m pip install pandas numpy
& $py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install `
    "transformers==4.57.1" `
    "accelerate==1.11.0" `
    "trl==0.26.0" `
    "datasets==3.6.0" `
    "peft==0.17.1" `
    "wandb==0.22.3" `
    "math-verify==0.8.0" `
    "safetensors==0.5.3" `
    "tiktoken==0.9.0" `
    "einops==0.8.1"

Write-Host ""
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
Write-Host "Setup done. Activate with: $Venv\Scripts\Activate.ps1"
