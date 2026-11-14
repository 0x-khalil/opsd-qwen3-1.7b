# OPSD on Qwen3-1.7B -- single RTX 3070 (8 GB) reproduction of the paper setup.
#
# Differences vs. scripts/run_opsd_1b.sh (4x H100):
#   * 1 process instead of 4, per-device batch 1, grad-accum 32 -> effective batch 32 (same as paper)
#   * no vLLM (not available on Windows) -> generation falls back to HF `generate` (slower)
#   * attn_implementation=sdpa instead of flash_attention_2 (no FA2 wheels on Windows)
#   * chunked JSD loss (--jsd_chunk_size) so the full-vocab logits are never materialised
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/run_opsd_1b_3070.ps1
#   ... -MaxSteps 100 -MaxCompletionLength 1024
param(
    [string]$ModelPath   = "C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B",
    [string]$OutputDir   = "C:\Users\Khali\projects\OPSD\runs",
    [string]$RunConfig   = "qwen31b_3070_gen1024_fixteacher",
    [int]$MaxSteps       = 50,         # stop after this many optimizer steps
    [int]$NumEpochs     = 30,
    [int]$SaveSteps     = 10,         # write a checkpoint every N optimizer steps
    [int]$MaxCompletion = 1024,
    [int]$BatchSize     = 2,          # per-device batch (paper used 4 per GPU x 4 GPUs)
    [int]$EffBatch      = 32,         # effective batch size (same as the paper)
    [int]$ChunkSize     = 64,         # JSD chunk size in tokens
    [int]$LoraR         = 64,
    [double]$Lr          = 5e-6,
    [string]$ReportTo    = "none"      # "none" | "wandb"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$py = Join-Path (Split-Path -Parent $RepoRoot) ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$GradAccum = [Math]::Max(1, [Math]::Floor($EffBatch / $BatchSize))

$env:TOKENIZERS_PARALLELISM = "false"
$env:WANDB_MODE = "offline"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
# Windows consoles default to cp1252 -> wandb/TRL banners with emoji crash the run
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# NOTE: opsd_train.py appends --run_config to --output_dir itself, so we pass the
# parent directory here; checkpoints land in <OutputDir>\<RunConfig>\checkpoint-N.
$out = $OutputDir

$argsList = @(
    "opsd_train.py",
    "--model_name_or_path", $ModelPath,
    "--learning_rate", "$Lr",
    "--max_grad_norm", "0.1",
    "--per_device_train_batch_size", "$BatchSize",
    "--gradient_accumulation_steps", "$GradAccum",
    "--gradient_checkpointing",
    "--bf16", "True",
    "--output_dir", $out,
    "--run_config", $RunConfig,
    "--max_completion_length", "$MaxCompletion",
    "--save_steps", "$SaveSteps",
    "--logging_steps", "1",
    "--attn_implementation", "sdpa",
    "--torch_dtype", "bfloat16",
    "--max_length", "20000",
    "--beta", "0",
    "--lmbda", "1",
    "--use_peft", "True",
    "--lora_r", "$LoraR",
    "--lora_alpha", "$($LoraR * 2)",
    "--lora_target_modules", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "--temperature", "1.1",
    "--top_p", "0.95",
    "--top_k", "20",
    "--fixed_teacher",
    "--jsd_token_clip", "0.05",
    "--jsd_chunk_size", "$ChunkSize",
    "--report_to", $ReportTo,
    "--seed", "42"
)

if ($MaxSteps -gt 0) {
    $argsList += @("--max_steps", "$MaxSteps")
} else {
    $argsList += @("--num_train_epochs", "$NumEpochs")
}

Write-Host "Running: $py $($argsList -join ' ')"
& $py @argsList
