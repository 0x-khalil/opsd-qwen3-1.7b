# Reproducing OPSD (Qwen3-1.7B)

Port of [`siyan-zhao/OPSD`](https://github.com/siyan-zhao/OPSD) — *Self-Distilled Reasoner:
On-Policy Self-Distillation for Large Language Models* (arXiv:2601.18734) — to a
**single rented GPU (RTX 5090, 32 GB, Linux)**, plus a fallback path for a local
8 GB card.

Everything that does not depend on the hardware is **bit-for-bit identical to upstream**;
the diff is listed in §6.

---

## 1. What the paper does

One model plays both roles:

* the **student** only sees the problem,
* the **teacher** additionally sees the ground-truth reference solution,
* their token distributions are matched with a **Jensen-Shannon divergence** along the
  student's *own* on-policy rollouts (the teacher is frozen at step 0 via
  `--fixed_teacher`, implemented by disabling the LoRA adapters).

Upstream Qwen3-1.7B config (`scripts/run_opsd_1b.sh`): 4×H100, per-device batch 4,
grad-accum 2 (effective batch 32), LoRA r=64 on all linear projections, `--beta 0`
(forward KL), `--jsd_token_clip 0.05`, `--max_completion_length 1024`,
`temperature 1.1 / top_p 0.95 / top_k 20`, `lr 5e-6`, `max_grad_norm 0.1`.
The paper reports the peak within ~100 steps.

---

## 2. Setup on the rented machine (Linux, RTX 5090)

```bash
git clone <this repo> && cd OPSD_repo
bash scripts/setup_rental_5090.sh
```

That script

1. creates the conda env from `environment.yml` (python 3.10, torch 2.8.0,
   transformers 4.57.1, trl 0.26.0, peft 0.17.1, vllm 0.11.0, deepspeed 0.18.2),
2. installs the **CUDA 12.8** build of torch (Blackwell / sm_120 needs cu128),
3. installs vLLM 0.11.0 and flash-attn (flash-attn is optional — see §5),
4. downloads `Qwen/Qwen3-1.7B` to `~/models/Qwen3-1.7B` and the training set
   `siyanzhao/Openthoughts_math_30k_opsd` (29 434 rows of `problem` + `solution`).

---

## 3. Training (50 steps, checkpoint every 10)

```bash
bash scripts/run_opsd_1b_5090.sh
```

Paper hyper-parameters, moved onto one GPU:

| upstream (4×H100) | here (1×5090, 32 GB) |
| --- | --- |
| per-device batch 4 × grad-accum 2 | per-device batch 4 × grad-accum 8 → **effective 32 (same)** |
| vLLM colocate, 60 % of 80 GB | vLLM colocate, 45 % of 32 GB (`VLLM_MEM_UTIL`) |
| `flash_attention_2` | `sdpa` (flash-attn 2.x has no sm_120 kernels; same maths) |
| `--num_train_epochs 30` | `--max_steps 50` (`MAX_STEPS=100` for more) |
| `--save_steps 25` | `--save_steps 10` |

Everything else is unchanged: LoRA r=64/α=128 on
`q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`, `--fixed_teacher`,
`--beta 0 --lmbda 1`, `--jsd_token_clip 0.05`, `--max_completion_length 1024`,
`--temperature 1.1 --top_p 0.95 --top_k 20`, `--learning_rate 5e-6`,
`--max_grad_norm 0.1`, `--max_length 20000`, bf16, gradient checkpointing.

Useful overrides:

```bash
MAX_STEPS=100 bash scripts/run_opsd_1b_5090.sh   # more steps
MAX_COMPLETION=512 bash scripts/run_opsd_1b_5090.sh # cheaper rollouts
USE_VLLM=0 bash scripts/run_opsd_1b_5090.sh        # plain HF generate (no vLLM)
BATCH_SIZE=8 GRAD_ACCUM=4 bash scripts/run_opsd_1b_5090.sh  # bigger micro-batch
REPORT_TO=wandb bash scripts/run_opsd_1b_5090.sh   # needs `wandb login`
```

The script launches through `torchrun --nproc_per_node 1` because vLLM's
`external_launcher` backend (upstream default) expects an initialised process group;
`VLLM_BACKEND=mp` runs vLLM in its own worker process instead.

### Artifacts

```
~/runs/qwen31b_5090_gen1024_fixteacher/
├── adapter_config.json          # final adapter (trainer.save_model)
├── adapter_model.safetensors    # final adapter weights
├── checkpoint-10/ … checkpoint-50/   # LoRA adapters every 10 steps
├── trainer_state.json           # optimizer / scheduler / RNG state
└── generations/                 # sample student rollouts every 5 steps
```

Only LoRA is trained (69.7 M params → ~270 MB per checkpoint in fp32).

For evaluation on another machine:

```bash
# 1) pack everything (adapters + metadata json)
python scripts/package_artifacts.py --run_dir ~/runs/qwen31b_5090_gen1024_fixteacher \
    --base_model ~/models/Qwen3-1.7B --zip

# 2a) on the eval box: let vLLM use the adapter directly
#     (eval/evaluate_math.py already supports --lora_adapter_path, max_lora_rank=64)

# 2b) or merge into a plain HF model directory
python scripts/merge_lora.py --base_model ~/models/Qwen3-1.7B \
    --adapter ~/runs/qwen31b_5090_gen1024_fixteacher/checkpoint-50 \
    --output ~/runs/merged-50
```

Evaluation (upstream, vLLM):

```bash
cd eval && bash run_eval.sh          # AIME24 / AIME25 / HMMT25, Avg@12, thinking mode
```

---

## 4. Fallback: local 8 GB card (RTX 3070, Windows)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1   # venv + torch cu128
powershell -ExecutionPolicy Bypass -File scripts\run_opsd_1b_3070.ps1 `
    -RunConfig qwen31b_3070_gen512_fixteacher -MaxSteps 50 -SaveSteps 10 `
    -MaxCompletion 512 -BatchSize 4 -EffBatch 32 -ChunkSize 32
```

No vLLM/DeepSpeed/flash-attn on Windows → HF `generate` + SDPA, and the rollouts are
capped at 512 tokens because HF decoding is ~25× slower than vLLM
(~30 tok/s vs ~1500 tok/s). Measured per micro-step (batch 4):

| phase | 1024-token rollouts | 512-token rollouts |
| --- | --- | --- |
| student rollout | 117-139 s | 60-63 s |
| student backbone forward | 0.8-1.0 s | 0.4 s |
| teacher backbone forward | 1.0-1.4 s | 0.7 s |
| chunked JSD loss | 1.6-2.0 s | 1.0 s |
| peak VRAM (allocated / reserved) | 5.5 / 5.9 GB | 5.0 / 5.2 GB |

→ ~15 h for 50 steps at 1024, ~7 h at 512.

---

## 5. Notes / gotchas

* **flash-attn on Blackwell**: flash-attn 2.x ships kernels only up to sm_90, so a
  5090 (sm_120) has to use SDPA. `opsd_train.py` now auto-detects this and falls back
  (`--attn_implementation flash_attention_2` → `sdpa` with a printed warning).
* **The chunked loss is mandatory here.** The upstream loss materialises
  `(seq_len × 151936)` fp32 logits four times over; with vLLM resident on the same
  32 GB card that does not fit. `_compute_loss_chunked` computes the identical
  quantity in token chunks with gradient checkpointing (~1 GB instead of ~16 GB).
  `--jsd_chunk_size 0` restores the upstream implementation.
* **vLLM + PEFT**: after every optimizer step the trainer merges the LoRA adapter
  into the base weights and pushes them into the vLLM engine
  (`vllm_sync_frequency=1`), so the rollouts are always on-policy.
* `VLLM_MEM_UTIL=0.45` is what leaves room for the training model (weights + optimizer
  + activations) next to vLLM on a 32 GB card. Raise it if you shrink the batch.

---

## 6. Files changed relative to upstream

| file | change |
| --- | --- |
| `opsd_trainer.py` | new `_compute_loss_chunked` (+`_chunk_loss_fn`, `_get_backbone_and_head`); upstream `compute_loss` kept as `_compute_loss_reference`, selected by `--jsd_chunk_size` (`0` = upstream behaviour); `--vllm_distributed_executor_backend` is now configurable |
| `opsd_train.py` | new `--jsd_chunk_size` and `--vllm_distributed_executor_backend` flags; attention-implementation auto-fallback (flash-attn → sdpa) |
| `scripts/run_opsd_1b_5090.sh` | single-GPU launcher with the paper hyper-parameters (50 steps, save every 10) |
| `scripts/setup_rental_5090.sh` | conda env + cu128 torch + vLLM + model/dataset download |
| `scripts/run_opsd_1b_3070.ps1`, `scripts/launch_50steps.ps1`, `scripts/setup_windows.ps1` | local 8 GB path |
| `scripts/merge_lora.py`, `scripts/package_artifacts.py` | merge adapters / pack artifacts for remote evaluation |
| `eval/evaluate_math_hf.py` | vLLM-free evaluation (local card) |
| `test_loss_equivalence.py` | proves chunked loss == upstream loss (value + gradient) |
