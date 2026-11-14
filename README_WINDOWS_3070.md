# Reproducing OPSD (Qwen3-1.7B) on a single RTX 3070 (8 GB)

This directory contains a **Windows / single-GPU** port of
[`siyan-zhao/OPSD`](https://github.com/siyan-zhao/OPSD) (*Self-Distilled Reasoner:
On-Policy Self-Distillation for Large Language Models*, arXiv:2601.18734).

Everything in the upstream repo that does not depend on vLLM / DeepSpeed / FlashAttention
is kept **bit-for-bit identical**; the changes are listed below.

---

## 1. What the paper does (and what the GPU forces us to change)

The paper trains **one** model to be both student and teacher:

* the **student** only sees the problem,
* the **teacher** additionally sees the ground-truth reference solution,
* the two distributions are matched with a token-level **Jensen-Shannon divergence**
  along the student's *own* on-policy rollouts.

Upstream reference config for Qwen3-1.7B (`scripts/run_opsd_1b.sh`):
4×H100, per-device batch 4, grad-accum 2 (effective batch 32), LoRA r=64 (all linear
projections), `--fixed_teacher` (teacher = step-0 policy, implemented by disabling the
LoRA adapters), `--beta 0` (forward KL), `--jsd_token_clip 0.05`, `--max_completion_length 1024`,
`temperature 1.1 / top_p 0.95 / top_k 20`, `lr 5e-6`, `max_grad_norm 0.1`, ~15 min of training.

On an 8 GB card the following had to change:

| upstream | here | why |
| --- | --- | --- |
| vLLM colocate generation | HF `generate` | vLLM has no Windows wheels |
| `flash_attention_2` | `sdpa` | no FA2 wheels on Windows |
| 4 procs × batch 4 × accum 2 | 1 proc × batch 1 × accum 32 | 8 GB VRAM; **effective batch stays 32** |
| full-vocab JSD in one shot | chunked + checkpointed JSD | the `(seq_len × 151936)` logits alone are ~2.5 GB |

The chunked loss is **numerically identical** to the upstream implementation
(see `test_loss_equivalence.py`), it only changes *when* the logits are computed.

---

## 2. Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

Creates `.venv` (Python 3.13), installs torch 2.8.0+cu128 (RTX 3070 = sm_86),
transformers 4.57.1, trl 0.26.0, peft 0.17.1, datasets 3.6.0, wandb, math-verify.

Fetch the model + dataset (once):

```powershell
python ..\fetch_assets.py
# -> models\Qwen3-1.7B  and  the HF dataset siyanzhao/Openthoughts_math_30k_opsd
```

---

## 3. Training

```powershell
# the real thing: paper hyper-parameters, effective batch 32, 50 steps,
# a checkpoint every 10 steps
powershell -ExecutionPolicy Bypass -File scripts\run_opsd_1b_3070.ps1

# equivalent explicit call
powershell -ExecutionPolicy Bypass -File scripts\run_opsd_1b_3070.ps1 `
    -RunConfig qwen31b_3070_gen1024_fixteacher -MaxSteps 50 -SaveSteps 10 `
    -MaxCompletion 1024 -BatchSize 4 -EffBatch 32 -ChunkSize 32
```

Knobs: `-MaxCompletion` (rollout length, 1024 in the paper), `-BatchSize` /
`-EffBatch` (effective batch = `-EffBatch`, 32 in the paper), `-ChunkSize`
(JSD chunk size, smaller = less VRAM), `-MaxSteps`, `-SaveSteps`, `-ReportTo wandb`.

### Measured speed on this machine (RTX 3070 Laptop, 8 GB)

Per micro-step (batch 4, 1024-token rollouts) the run splits as:

| phase | time |
| --- | --- |
| student rollout (`model.generate`, 4 x ~1000 tokens) | **117-139 s** |
| student backbone forward | 0.8-1.0 s |
| teacher backbone forward | 1.0-1.4 s |
| chunked JSD loss | 1.6-2.0 s |
| peak VRAM (allocated / reserved) | 5.5 / 5.9 GB |

So **generation is ~97 % of the step time** - the same thing that vLLM solves on the
paper's 4xH100 box. HF decoding is kernel-launch bound here (GPU util ~39 %), and the
PEFT wrapper costs another ~40 % (extra LoRA matmuls per layer). Batching is the only
lever that matters:

| batch | plain bf16 | + LoRA r=64 |
| --- | --- | --- |
| 1 | 13.5 tok/s | 9.2 tok/s |
| 2 | 31.8 tok/s | 18.2 tok/s |
| 4 | 65.0 tok/s | 38.3 tok/s |
| 8 | - | 41.6 tok/s |

With `BatchSize 4` + `EffBatch 32` (grad-accum 8) a step takes ~18 min, i.e. **~15 h
for the 50 steps**. Halve it with `-MaxCompletion 512` (~7.5 h) at the cost of
truncating the longest rollouts.

(Static KV caches would help further but need Triton, which has no Windows wheels.)

### What gets written where

```
runs\<RunConfig>\
├── adapter_config.json          <- final adapter (saved by trainer.save_model)
├── adapter_model.safetensors    <- final adapter weights
├── checkpoint-10\               <- LoRA adapter at step 10  (save_steps=10)
├── checkpoint-20\
├── checkpoint-30\
├── checkpoint-40\
├── checkpoint-50\
├── trainer_state.json           <- optimizer / scheduler / RNG state
└── generations\                 <- sample student rollouts every 5 steps
```

Only the LoRA adapters are trained (69.7 M params), so every checkpoint is ~270 MB
(saved in fp32).

### Launching detached + packaging for transfer

```powershell
# detached launch (survives closing the terminal), logs to runs\opsd_50steps.log
powershell -ExecutionPolicy Bypass -File scripts\launch_50steps.ps1

# collect final adapter + all checkpoints + a metadata json
python scripts\package_artifacts.py --run_dir runs\qwen31b_3070_gen1024_fixteacher `
    --base_model models\Qwen3-1.7B --zip
```

### Merging for remote evaluation

The adapters are standard PEFT adapters; on the evaluation machine either point
vLLM at the adapter (`--enable_lora`, `max_lora_rank=64`) or merge it into a plain
model directory:

```powershell
python scripts\merge_lora.py ^
    --base_model C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B ^
    --adapter    C:\Users\Khali\projects\OPSD\runs\qwen31b_3070_gen1024_fixteacher\checkpoint-50 ^
    --output     C:\Users\Khali\projects\OPSD\runs\merged-50
```

This writes a self-contained HF model dir (config + safetensors + tokenizer),
which is what the upstream `eval/evaluate_math.py` (vLLM) expects.

---

## 4. Evaluation

`eval/evaluate_math.py` needs vLLM. Use the HF fallback instead:

```powershell
python eval\evaluate_math_hf.py `
    --base_model_path C:\Users\Khali\projects\OPSD\models\Qwen3-1.7B `
    --adapter_path C:\Users\Khali\projects\OPSD\runs\qwen31b_3070\checkpoint-100 `
    --dataset aime24 --num_samples 4 --enable_thinking
```

---

## 5. Files changed relative to upstream

| file | change |
| --- | --- |
| `opsd_trainer.py` | new `_compute_loss_chunked` (+`_chunk_loss_fn`, `_get_backbone_and_head`); upstream `compute_loss` kept as `_compute_loss_reference` and selected by `--jsd_chunk_size` (`0` = upstream behaviour) |
| `opsd_train.py` | new `--jsd_chunk_size` flag |
| `scripts/run_opsd_1b_3070.ps1` | single-GPU launcher with the paper hyper-parameters |
| `scripts/setup_windows.ps1` | environment bootstrap |
| `eval/evaluate_math_hf.py` | vLLM-free evaluation |
| `test_loss_equivalence.py` | proves chunked loss == upstream loss (value + gradient) |
